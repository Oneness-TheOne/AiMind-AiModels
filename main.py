import io
import os
import json
import base64
import sys
import shutil
import markdown
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot.guideChatbot import ask_to_website_guide_chatbot
from chatbot.psychologicalAnalysisChatbot import get_answer_for_more_question_about_analysis
from analysis_metrics import (
    compute_image_metrics,
    compute_peer_summary_by_folder,
    load_peer_stats,
)
from drawing_score import compute_scores_for_analysis as compute_drawing_scores


current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)

# uvicorn CLI 기본 포트(8000)를 env 값으로 맞추기 위한 fallback
aimodels_port = os.getenv("AIMODELS_PORT", "8080")
os.environ.setdefault("PORT", aimodels_port)
os.environ.setdefault("UVICORN_PORT", aimodels_port)

app = FastAPI()


BASE_DIR = Path(current_dir)
# 그림 분석(/analyze) 전용 경로
IMAGE_TO_JSON_DIR = BASE_DIR / "image_to_json"
IMAGE_UPLOAD_DIR = IMAGE_TO_JSON_DIR / "uploads"
IMAGE_RESULT_DIR = IMAGE_TO_JSON_DIR / "result"
JSON_TO_LLM_DIR = BASE_DIR / "jsonToLlm"
JSON_TO_LLM_RESULTS_DIR = JSON_TO_LLM_DIR / "results"
LABEL_STATS_PATH = BASE_DIR / "data" / "label_stats_by_group.json"
LABEL_STATS_CACHE = None

IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
JSON_TO_LLM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(IMAGE_TO_JSON_DIR) not in sys.path:
    sys.path.append(str(IMAGE_TO_JSON_DIR))
if str(JSON_TO_LLM_DIR) not in sys.path:
    sys.path.append(str(JSON_TO_LLM_DIR))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Docker health check endpoint"""
    return {"status": "healthy"}


class ChatbotRequest(BaseModel):
    question: str
    analysis_context: dict | None = None


class AnalyzeScoreRequest(BaseModel):
    """T-Score 산출용 요청 (이미 분석된 results + 아동 정보)."""
    results: dict
    age: int
    gender: str


class ChatbotResponse(BaseModel):
    question: str
    answer: str


@app.post("/chatbot", response_model=ChatbotResponse)
def chatbot(payload: ChatbotRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")
    
    print(question)
    analysis_context = payload.analysis_context
    answer = ""

    try:
        # 심리 분석 질문 라우팅
        if analysis_context: # 분석 결과가 payload에 같이 왔을 때
            answer = get_answer_for_more_question_about_analysis(question, analysis_context)

        # 웹 사이트 이용 방법 질문 라우팅
        else:
            answer = ask_to_website_guide_chatbot(question)

            # guide answer은 markdown 형식이기 때문에 HTML로 바꿔서 반환합니다.
            answer = markdown.markdown(answer, extensions=["nl2br"])

        if not answer:
            raise HTTPException(status_code=500, detail="답변을 준비할 수 없습니다.")
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="답변을 준비하는 과정에서 오류가 발생했습니다.")
    
    return ChatbotResponse(question=question, answer=answer)


def _save_upload_file(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _encode_image_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    ext = path.suffix.lower().lstrip(".") or "jpg"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def _encode_image_base64_resized(path: Path, max_bytes: int = 1_400_000, max_long_side: int = 1200, quality: int = 85) -> str | None:
    """이미지를 읽어, 크기가 max_bytes 초과면 해상도/품질을 낮춰 JPEG로 압축한 뒤 data URL 반환."""
    if not path.exists() or not path.is_file():
        return None
    raw = path.read_bytes()
    if len(raw) <= max_bytes:
        ext = path.suffix.lower().lstrip(".") or "jpg"
        mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
        return f"data:image/{mime};base64,{base64.b64encode(raw).decode('utf-8')}"
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_long_side:
            img.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        while len(data) > max_bytes and (max_long_side > 320 or quality > 40):
            if quality > 40:
                quality -= 10
            else:
                max_long_side = int(max_long_side * 0.75)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                if max(img.size) > max_long_side:
                    img.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
                quality = 75
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"[diary-ocr] 이미지 리사이즈 실패 (원본 사용 시도): {e}")
        if len(raw) <= 2 * max_bytes:
            return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('utf-8')}"
        return None


def _normalize_gender(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"male", "m", "남", "남아"}:
        return "남"
    if v in {"female", "f", "여", "여아"}:
        return "여"
    return "미상"


def _get_label_stats() -> dict:
    global LABEL_STATS_CACHE
    if LABEL_STATS_CACHE is None:
        LABEL_STATS_CACHE = load_peer_stats(LABEL_STATS_PATH)
    return LABEL_STATS_CACHE or {}


def _get_ocr_paths() -> tuple[Path, Path]:
    """그림일기 OCR 전용 경로. 이미지는 ocr/img, JSON 저장은 ocr."""
    ocr_dir = BASE_DIR / "ocr"
    ocr_upload = ocr_dir / "uploads"
    ocr_upload.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "img").mkdir(parents=True, exist_ok=True)
    return ocr_dir, ocr_upload


@app.post("/diary-ocr")
async def diary_ocr(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일을 업로드해 주세요.")

    ocr_dir, ocr_upload_dir = _get_ocr_paths()
    ext = Path(file.filename).suffix or ".jpg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"diary_{timestamp}"
    input_path = ocr_upload_dir / f"{base_name}{ext}"
    _save_upload_file(file, input_path)

    # diary_ocr_only.run(): 이미지 저장 + 텍스트 추출 + Gemini 후처리 (bbox 제거된 원본)
    if str(ocr_dir) not in sys.path:
        sys.path.insert(0, str(ocr_dir))
    try:
        import diary_ocr_only
        response_item = diary_ocr_only.run(str(input_path.resolve()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"일기 OCR 실패: {e}")

    # null → 빈 문자열
    def _str(v):
        return v if isinstance(v, str) else (v or "")

    response_item = {k: _str(v) for k, v in response_item.items()}

    # 크롭된 그림(그림_저장경로)을 data URL로 넣어 프론트 카드 사진란에서 사용. 크면 해상도 낮춰서라도 포함.
    MAX_IMAGE_BYTES_FOR_RESPONSE = 1_400_000  # 약 1.4MB 초과 시 리사이즈 후 포함
    try:
        cropped_path = (response_item.get("그림_저장경로") or "").strip()
        if cropped_path:
            p = Path(cropped_path)
            if p.exists() and p.is_file():
                url = _encode_image_base64_resized(p, max_bytes=MAX_IMAGE_BYTES_FOR_RESPONSE)
                response_item["image_data_url"] = url
                if url and len(url) > 500:
                    print(f"[diary-ocr] 크롭 이미지 포함 (data URL 길이={len(url)})")
            else:
                response_item["image_data_url"] = None
        else:
            response_item["image_data_url"] = None
    except Exception as e:
        print(f"[diary-ocr] 크롭 이미지 data URL 변환 실패 (무시): {e}")
        response_item["image_data_url"] = None

    payload_for_json = {k: v for k, v in response_item.items() if k != "image_data_url"}
    json_path = ocr_dir / f"{base_name}_diary_result.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload_for_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[diary-ocr] JSON 저장 실패 (무시): {e}")

    print("\n" + "=" * 60)
    print("[그림일기 OCR] 분석 결과 (diary_ocr_only)")
    print("=" * 60)
    print(json.dumps(payload_for_json, ensure_ascii=False, indent=2))
    print("=" * 60 + "\n")

    try:
        return [{**response_item, "교정된_내용": response_item.get("내용", "") or ""}]
    except Exception as e:
        print(f"[diary-ocr] 응답 반환 직전 오류 (이미지 제외 후 재시도): {e}")
        response_item["image_data_url"] = None
        return [{**response_item, "교정된_내용": response_item.get("내용", "") or ""}]


@app.post("/analyze")
async def analyze(
    tree: UploadFile = File(...),
    house: UploadFile = File(...),
    man: UploadFile = File(...),
    woman: UploadFile = File(...),
    child_name: str = Form(""),
    child_age: str = Form(""),
    child_gender: str = Form(""),
):
    try:
        from image_to_json import run as image_to_json_run
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"image_to_json 로딩 실패: {e}")

    try:
        from gemini_integration import analyze_and_interpret, generate_overall_psychology, resolve_gemini_api_key
        from legacy_converter import is_new_format, convert_new_to_legacy
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"jsonToLlm 로딩 실패: {e}")

    if not resolve_gemini_api_key():
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY 또는 GEMINI_API_KEYS가 필요합니다.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gender_kr = _normalize_gender(child_gender)
    age_str = (child_age or "").strip() or "0"

    # 성별을 영문 male/female로 변환 (image_to_json에서 사용)
    gender_en = "male" if gender_kr == "남" else ("female" if gender_kr == "여" else "male")

    uploads = [
        ("tree", "나무", tree),
        ("house", "집", house),
        ("man", "남자사람", man),
        ("woman", "여자사람", woman),
    ]

    results = {}
    per_object_metrics = []
    folder_keys = []
    for object_type, label_kr, upload in uploads:
        ext = Path(upload.filename or "").suffix or ".jpg"
        stem = f"{label_kr}_{age_str}_{gender_kr}_{timestamp}"
        input_path = IMAGE_UPLOAD_DIR / f"{stem}{ext}"
        output_json_path = IMAGE_RESULT_DIR / f"{stem}.json"
        output_image_path = IMAGE_RESULT_DIR / f"{stem}_box.jpg"

        _save_upload_file(upload, input_path)

        try:
            rag_result = image_to_json_run(
                image_path=str(input_path),
                object_type=object_type,
                output_format="rag",
                output_image_path=str(output_image_path),
                gender=gender_en,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{label_kr} 분석 실패: {e}")

        output_json_path.write_text(
            json.dumps(rag_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        box_image_base64 = _encode_image_base64(output_image_path)

        original = rag_result
        if is_new_format(original):
            original = convert_new_to_legacy(original, str(output_json_path))

        interp_results = analyze_and_interpret(
            original,
            model_name="gemini-2.5-flash-lite",
            api_key=None,  # env에서 GEMINI_API_KEYS/GEMINI_API_KEY 사용 (503 시 키 순환)
            temperature=0.2,
            use_rag=True,
            rag_db_path=str(JSON_TO_LLM_DIR / "htp_knowledge_base"),
            rag_k=10,
            max_output_tokens=8192,
        )

        if not interp_results.get("success"):
            raise HTTPException(status_code=500, detail=f"{label_kr} 해석 실패: {interp_results.get('error')}")

        interpretation = interp_results.get("interpretation")
        interpretation_path = JSON_TO_LLM_RESULTS_DIR / f"interpretation_{stem}.json"
        if interpretation:
            interpretation_path.write_text(
                json.dumps(interpretation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        object_metrics = compute_image_metrics(original, str(input_path), use_color=False)
        per_object_metrics.append(object_metrics)
        folder_keys.append(
            {
                "tree": "TL_나무",
                "house": "TL_집",
                "man": "TL_남자사람",
                "woman": "TL_여자사람",
            }.get(object_type, "")
        )

        results[object_type] = {
            "label": label_kr,
            "image_json": rag_result,
            "interpretation": interpretation,
            "analysis": interp_results.get("analysis"),
            "box_image_base64": box_image_base64,
            "json_path": str(output_json_path),
            "interpretation_path": str(interpretation_path),
            "metrics": object_metrics,
        }

    comparison = {}
    try:
        age_int = int(age_str)
    except ValueError:
        age_int = 0
    if age_int and gender_kr in {"남", "여"}:
        stats = _get_label_stats()
        if stats:
            comparison = compute_peer_summary_by_folder(
                per_object_metrics,
                stats,
                age_int,
                gender_kr,
                folder_keys,
            )
        # T-Score 기반 drawing_norm_dist_stats 비교 점수 (에너지/위치안정성/표현력)
        drawing_scores = compute_drawing_scores(results, age_int, gender_kr)
        comparison["drawing_scores"] = drawing_scores
        # T-Score 기반 종합 점수·발달 단계·정서 상태 (프론트 요약 카드 연동)
        agg = drawing_scores.get("aggregated") if drawing_scores else None
        if agg:
            avg_t = (
                agg["에너지_점수"] + agg["위치_안정성_점수"] + agg["표현력_점수"]
            ) / 3
            comparison["overall_score"] = round(avg_t, 1)
            if avg_t >= 55:
                stage = "정상 발달"
            elif avg_t >= 35:
                stage = "보통 발달"
            else:
                stage = "지원이 필요한 영역 있음"
            if "development" not in comparison:
                comparison["development"] = {}
            comparison["development"]["stage"] = stage
            raw_emotional = (agg.get("종합_평가") or "").strip()
            comparison["emotional_state"] = raw_emotional or "분석 완료"

    # Aggregate recommendation items from 4 drawings by category (LLM outputs Korean keys)
    LLM_CATEGORY_KEYS = ("정서_심리_지원", "대인관계_사회성", "양육_일상_활동")
    CATEGORY_EN = ("emotional_psychological_support", "interpersonal_social", "parenting_daily_activities")
    by_category = {k: [] for k in CATEGORY_EN}
    legacy_items = []

    def _norm(s):
        if isinstance(s, str) and s.strip():
            return s.strip()
        if isinstance(s, dict) and s.get("내용"):
            return str(s["내용"]).strip()
        return None

    for key in ("tree", "house", "man", "woman"):
        interp = (results.get(key) or {}).get("interpretation")
        if not interp or not isinstance(interp, dict):
            continue
        rec = interp.get("추천_사항")
        if not isinstance(rec, dict):
            continue
        for i, llm_key in enumerate(LLM_CATEGORY_KEYS):
            if rec.get(llm_key):
                for x in rec[llm_key]:
                    v = _norm(x)
                    if v and v not in by_category[CATEGORY_EN[i]]:
                        by_category[CATEGORY_EN[i]].append(v)
        if rec.get("항목"):
            for x in rec["항목"]:
                v = _norm(x)
                if v:
                    legacy_items.append(v)

    if any(by_category[k] for k in CATEGORY_EN):
        recommendations_payload = [
            {"category": cat_en, "items": by_category[cat_en]}
            for cat_en in CATEGORY_EN
            if by_category[cat_en]
        ]
    else:
        recommendations_payload = (
            [{"category": "analysis_based", "items": legacy_items}] if legacy_items else []
        )

    # 심리 해석(results)과 함께 전체 심리 결과 4개 필드 생성 (구조 고정: 종합/인상적/구조적/표상적)
    전체_심리_결과 = {
        "종합_요약": "",
        "인상적_분석": "",
        "구조적_분석_요약": "",
        "표상적_분석_종합": "",
    }
    # 분석 결과(results)를 보내고 전체 심리 결과 4필드를 같이 요청·수신 (요약이 아닌 분석 기반 작성)
    try:
        overall = generate_overall_psychology(
            results, child_name, age_str, gender_kr,
            api_key=None,  # env에서 GEMINI_API_KEYS/GEMINI_API_KEY 사용 (503 시 키 순환)
            rag_db_path=str(JSON_TO_LLM_DIR / "htp_knowledge_base"),
        )
        if overall and isinstance(overall, dict):
            전체_심리_결과["종합_요약"] = (overall.get("종합_요약") or "").strip()
            전체_심리_결과["인상적_분석"] = (overall.get("인상적_분석") or "").strip()
            전체_심리_결과["구조적_분석_요약"] = (overall.get("구조적_분석_요약") or "").strip()
            전체_심리_결과["표상적_분석_종합"] = (overall.get("표상적_분석_종합") or "").strip()
    except Exception as e:
        print(f"[analyze] 전체 심리 결과 생성 실패(무시): {e}")

    payload = {
        "success": True,
        "child": {
            "name": child_name,
            "age": age_str,
            "gender": gender_kr,
        },
        "results": results,
        "comparison": comparison,
        "recommendations": recommendations_payload,
        "전체_심리_결과": 전체_심리_결과,
    }

    # 전체결과 JSON 저장 (그림별_해석 + 전체_심리_결과)
    try:
        그림별_해석 = {
            "나무": (results.get("tree") or {}).get("interpretation"),
            "집": (results.get("house") or {}).get("interpretation"),
            "남자사람": (results.get("man") or {}).get("interpretation"),
            "여자사람": (results.get("woman") or {}).get("interpretation"),
        }
        save_obj = {
            "child": payload["child"],
            "그림별_해석": 그림별_해석,
            "comparison": comparison,
            "전체_심리_결과": 전체_심리_결과,
        }
        def _strip_base64(obj):
            if isinstance(obj, dict):
                return {k: None if k == "box_image_base64" else _strip_base64(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_strip_base64(x) for x in obj]
            return obj
        save_obj = _strip_base64(save_obj)
        safe_name = (child_name or "분석").strip() or "분석"
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)[:50]
        full_path = JSON_TO_LLM_RESULTS_DIR / f"전체결과_{safe_name}_{timestamp}.json"
        full_path.write_text(json.dumps(save_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[analyze] 전체결과 JSON 저장 실패(무시): {e}")

    return payload


@app.post("/analyze/score")
def analyze_score(payload: AnalyzeScoreRequest):
    """분석 결과(results)와 아동 정보로 T-Score를 산출합니다."""
    gender_kr = _normalize_gender(payload.gender)
    if gender_kr not in {"남", "여"}:
        raise HTTPException(status_code=400, detail="gender는 남/여 중 하나여야 합니다.")
    age = max(7, min(13, int(payload.age) or 8))
    drawing_scores = compute_drawing_scores(payload.results, age, gender_kr)
    return drawing_scores


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("AIMODELS_PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

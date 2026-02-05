import os
import json
import base64
import sys
import shutil
import markdown
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot.chatbot import get_chatbot_answer


current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)

# uvicorn CLI 기본 포트(8000)를 env 값으로 맞추기 위한 fallback
aimodels_port = os.getenv("AIMODELS_PORT", "8080")
os.environ.setdefault("PORT", aimodels_port)
os.environ.setdefault("UVICORN_PORT", aimodels_port)

app = FastAPI()

BASE_DIR = Path(current_dir)
IMAGE_TO_JSON_DIR = BASE_DIR / "image_to_json"
IMAGE_UPLOAD_DIR = IMAGE_TO_JSON_DIR / "uploads"
IMAGE_RESULT_DIR = IMAGE_TO_JSON_DIR / "result"
JSON_TO_LLM_DIR = BASE_DIR / "jsonToLlm"
JSON_TO_LLM_RESULTS_DIR = JSON_TO_LLM_DIR / "results"

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


class ChatbotRequest(BaseModel):
    question: str


class ChatbotResponse(BaseModel):
    question: str
    answer: str


@app.post("/chatbot", response_model=ChatbotResponse)
def chatbot(payload: ChatbotRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")
    answer = get_chatbot_answer(question)

    if not answer:
        raise HTTPException(status_code=500, detail="답변을 준비할 수 없습니다.")
    
    # answer는 markdown 형식이기 때문에 HTML로 바꿔서 반환합니다.
    answer = markdown.markdown(answer, extensions=["nl2br"])
    
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


def _normalize_gender(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"male", "m", "남", "남아"}:
        return "남"
    if v in {"female", "f", "여", "여아"}:
        return "여"
    return "미상"


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
        from gemini_integration import analyze_and_interpret
        from legacy_converter import is_new_format, convert_new_to_legacy
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"jsonToLlm 로딩 실패: {e}")

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY가 필요합니다.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gender_kr = _normalize_gender(child_gender)
    age_str = (child_age or "").strip() or "0"

    uploads = [
        ("tree", "나무", tree),
        ("house", "집", house),
        ("man", "남자사람", man),
        ("woman", "여자사람", woman),
    ]

    results = {}
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
            api_key=os.getenv("GEMINI_API_KEY"),
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

        results[object_type] = {
            "label": label_kr,
            "image_json": rag_result,
            "interpretation": interpretation,
            "analysis": interp_results.get("analysis"),
            "box_image_base64": box_image_base64,
            "json_path": str(output_json_path),
            "interpretation_path": str(interpretation_path),
        }

    return {
        "success": True,
        "child": {
            "name": child_name,
            "age": age_str,
            "gender": gender_kr,
        },
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("AIMODELS_PORT", "6000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

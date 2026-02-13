#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Gemini API를 사용한 그림 분석 해석 통합 스크립트
RAG: ChromaDB에 저장된 HTP 지표를 먼저 검색해 참고 지표로 넣고, 그 위에 LLM 해석을 수행 (토큰 절약 + 근거 반영)
"""

import json
import os
import time
from tree_analyzer import process_json
from interpretation_prompts import get_interpretation_prompt

# RAG 메타데이터 키 (htp_indicator_parser 미import로 interpret 단독 실행 시 의존성 경량화)
SOURCE_FIELD = "source"

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GEMINI_AVAILABLE = False
    print("경고: google-genai 패키지가 설치되지 않았습니다.")
    print("설치 방법: pip install google-genai")

DEFAULT_MODEL = "gemini-2.5-flash-lite"

# interpret_with_gemini 전용: 503/429 등 일시 오류 시 재시도
GEMINI_RETRY_MAX = 5
GEMINI_RETRY_BACKOFF_SEC = 2


def _get_gemini_api_keys():
    """
    GEMINI_API_KEYS(쉼표 구분) 또는 GEMINI_API_KEY에서 키 목록 반환.
    """
    keys_str = os.getenv("GEMINI_API_KEYS", "").strip()
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if keys:
            return keys
    single = os.getenv("GEMINI_API_KEY", "").strip()
    return [single] if single else []


def resolve_gemini_api_key(api_key=None, index=0):
    """
    사용할 API 키 결정. api_key가 주어지면 그대로 사용, 아니면 env에서 가져옴.
    index: 503 재시도 시 다음 키 선택용 (GEMINI_API_KEYS일 때)
    """
    if api_key:
        return api_key
    keys = _get_gemini_api_keys()
    if not keys:
        return None
    return keys[index % len(keys)]


def _is_retryable_gemini_error(e):
    """503, 429, JSON 파싱 실패 등 재시도 가능 오류 여부 (interpret_with_gemini용)."""
    if isinstance(e, json.JSONDecodeError):
        return True
    msg = str(e).upper()
    return (
        "503" in msg
        or "429" in msg
        or "UNAVAILABLE" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "OVERLOADED" in msg
    )


# RAG용 ChromaDB (store_to_chroma.py와 동일 경로·임베딩)
HTP_DB_PATH = "./htp_knowledge_base"
try:
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


def setup_gemini(api_key=None):
    """
    Gemini API 클라이언트 생성 (google.genai SDK 사용).
    
    Args:
        api_key: Gemini API 키 (없으면 GEMINI_API_KEYS/GEMINI_API_KEY에서 가져옴)
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("google-genai 패키지가 필요합니다. pip install google-genai")
    
    resolved = resolve_gemini_api_key(api_key)
    if not resolved:
        raise ValueError("API 키가 필요합니다. GEMINI_API_KEYS 또는 GEMINI_API_KEY 환경변수를 설정하세요.")
    
    return genai.Client(api_key=resolved)


def _build_rag_query_from_analysis(analysis_data):
    """
    분석 결과(딕셔너리)에서 ChromaDB 유사도 검색용 쿼리 문자열 생성.
    요소_개수, 전체_구성, 타입별 키 등을 요약해 검색 품질을 높인다.
    """
    parts = []
    meta = analysis_data.get("이미지_메타정보", {})
    age = meta.get("연령", "")
    sex = meta.get("성별", "")
    if age or sex:
        parts.append(f"{age}세 {sex}아")
    counts = analysis_data.get("요소_개수", {})
    if counts:
        parts.append(" ".join(counts.keys()))
    comp = analysis_data.get("전체_구성", {})
    if isinstance(comp, dict):
        for k, v in comp.items():
            if k.endswith("_요약") and v:
                parts.append(str(v))
            elif k.endswith("_면적비율") and v is not None:
                parts.append(f"{k} {v}")
    if "나무_구성요소_관계" in analysis_data:
        parts.append("나무 수관 기둥 가지 뿌리")
        for k in analysis_data.get("나무_내_부가요소", {}):
            parts.append(k)
        for k in analysis_data.get("하늘_요소", {}):
            parts.append(k)
    if "얼굴_구성요소" in analysis_data:
        parts.append("사람 얼굴 신체")
        for k in analysis_data.get("얼굴_구성요소", {}):
            parts.append(k)
        for k in analysis_data.get("신체_구성요소", {}):
            parts.append(k)
    if "집_구성요소" in analysis_data:
        parts.append("집 지붕 문 창문")
        for k in analysis_data.get("집_구성요소", {}):
            parts.append(k)
        for k in analysis_data.get("집_주변_요소", {}):
            parts.append(k)
    return " ".join(parts).strip() or json.dumps(analysis_data, ensure_ascii=False)[:2000]


def _has_per_item_sources(interp):
    """해석 결과가 항목별로 {내용, 논문_근거} 형식인지 여부"""
    if not interp or not isinstance(interp, dict):
        return False
    v = interp.get("전체_요약")
    return isinstance(v, dict) and "내용" in v and ("논문_근거" in v or "source" in v)


def get_rag_context(analysis_data, db_path=None, api_key=None, k=10):
    """
    ChromaDB에서 분석 결과와 관련된 HTP 지표를 검색해 (참고 문자열, 참고 논문 목록) 반환.
    db_path 없거나 DB가 없으면 ("", []) 반환.

    Returns:
        tuple: (rag_context_str, references)
            - rag_context_str: LLM 프롬프트에 넣을 참고 지표 텍스트
            - references: 검색된 지표의 출처 논문 목록 (중복 제거, 순서 유지)
    """
    if not RAG_AVAILABLE:
        return "", []
    db_path = db_path or HTP_DB_PATH
    if not os.path.isdir(db_path):
        return "", []
    api_key = resolve_gemini_api_key(api_key)
    if not api_key:
        return "", []
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
        )
        query = _build_rag_query_from_analysis(analysis_data)
        docs = vector_db.similarity_search(query, k=k)
        lines = []
        seen_sources = []
        for i, doc in enumerate(docs, 1):
            lines.append(f"[지표 {i}]")
            lines.append(doc.page_content.strip())
            src = doc.metadata.get(SOURCE_FIELD, "")
            page = doc.metadata.get("page", "").strip()
            if src:
                out = f"(출처: {src}"
                if page:
                    out += f", p.{page}"
                out += ")"
                lines.append(out)
                if src not in seen_sources:
                    seen_sources.append(src)
            lines.append("")
        return "\n".join(lines).strip(), seen_sources
    except Exception:
        return "", []


def get_rag_context_by_query(query_string, db_path=None, api_key=None, k=10):
    """
    쿼리 문자열로 ChromaDB 검색 후 (참고 문자열, 참고 논문 목록) 반환.
    종합 심리 등 분석 결과 없이 RAG만으로 컨텍스트를 가져올 때 사용.

    Returns:
        tuple: (rag_context_str, references) — get_rag_context와 동일 형식
    """
    if not RAG_AVAILABLE or not (query_string and query_string.strip()):
        return "", []
    db_path = db_path or HTP_DB_PATH
    if not os.path.isdir(db_path):
        return "", []
    api_key = resolve_gemini_api_key(api_key)
    if not api_key:
        return "", []
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
        )
        docs = vector_db.similarity_search(query_string.strip(), k=k)
        lines = []
        seen_sources = []
        for i, doc in enumerate(docs, 1):
            lines.append(f"[지표 {i}]")
            lines.append(doc.page_content.strip())
            src = doc.metadata.get(SOURCE_FIELD, "")
            page = doc.metadata.get("page", "").strip()
            if src:
                out = f"(출처: {src}"
                if page:
                    out += f", p.{page}"
                out += ")"
                lines.append(out)
                if src not in seen_sources:
                    seen_sources.append(src)
            lines.append("")
        return "\n".join(lines).strip(), seen_sources
    except Exception:
        return "", []


def interpret_with_gemini(analysis_result, model_name=None, api_key=None, temperature=0.2, use_rag=True, rag_db_path=None, rag_k=5, max_output_tokens=4096):
    """
    Gemini를 사용하여 분석 결과를 해석.
    use_rag=True이면 ChromaDB에서 관련 HTP 지표를 먼저 검색해 프롬프트에 넣고( RAG ), 그 위에 해석을 요청해 토큰을 아끼고 근거를 반영한다.
    
    Args:
        analysis_result: process_json()의 결과 (JSON 문자열 또는 딕셔너리)
        model_name: 사용할 Gemini 모델명 (기본값: gemini-2.5-flash-lite, None이면 DEFAULT_MODEL)
        api_key: Gemini API 키 (없으면 환경변수에서 가져옴)
        temperature: 생성 온도 (0.0 ~ 1.0, 기본값: 0.7) 0.1 또는 0.2로 설정하는 것이 좋음
        use_rag: True이면 ChromaDB에서 참고 지표 검색 후 프롬프트에 포함 (기본값: True)
        rag_db_path: ChromaDB persist 디렉터리 (None이면 HTP_DB_PATH)
        rag_k: RAG 검색 상위 k개 (기본값: 5). 늘리면 참고 지표 많아지고, 줄이면 LLM 속도 향상
        max_output_tokens: LLM 최대 출력 토큰 (기본값: 4096). 늘리면 해석 잘림 방지, 줄이면 생성 속도 향상
    
    Returns:
        해석 결과 (딕셔너리). 성공 시 "timing" 키로 { "rag_sec", "llm_sec" } 포함
    """
    timing = {"rag_sec": 0.0, "llm_sec": 0.0}
    if model_name is None:
        model_name = DEFAULT_MODEL

    # RAG: 분석 결과로 ChromaDB 검색 후 참고 지표 문자열·참고 논문 목록 생성
    rag_context = ""
    references = []
    if use_rag:
        t0 = time.perf_counter()
        analysis_data = json.loads(analysis_result) if isinstance(analysis_result, str) else analysis_result
        rag_key = resolve_gemini_api_key(api_key, 0)
        rag_context, references = get_rag_context(analysis_data, db_path=rag_db_path, api_key=rag_key, k=rag_k)
        timing["rag_sec"] = time.perf_counter() - t0
    
    # 프롬프트 생성 (RAG 컨텍스트·참고 논문 목록 있으면 참고 지표 + 항목별 출처 요청)
    prompt = get_interpretation_prompt(
        analysis_result,
        rag_context=rag_context if rag_context else None,
        references=references if references else None,
    )
    
    # 생성 설정 (해석 JSON이 길어지므로 토큰 상한 확대, 잘림 방지)
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        max_output_tokens=max_output_tokens,
    )

    response_text = None
    last_error = None
    for attempt in range(GEMINI_RETRY_MAX):
        # 503/429 시 다음 키로 전환 (GEMINI_API_KEYS가 여러 개일 때)
        current_key = resolve_gemini_api_key(api_key, attempt)
        if not current_key:
            return {
                "success": False,
                "error": "API 키가 필요합니다. GEMINI_API_KEYS 또는 GEMINI_API_KEY를 설정하세요.",
                "references": references,
                "timing": timing,
                "raw_response": None,
            }
        client = setup_gemini(current_key)
        try:
            t0 = time.perf_counter()
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            timing["llm_sec"] = time.perf_counter() - t0
            response_text = response.text if hasattr(response, "text") and response.text else ""
            last_error = None
            break
        except Exception as e:
            last_error = e
            if not _is_retryable_gemini_error(e) or attempt == GEMINI_RETRY_MAX - 1:
                return {
                    "success": False,
                    "error": f"Gemini API 오류: {str(e)}",
                    "references": references,
                    "timing": timing,
                    "raw_response": None,
                }
            wait_sec = GEMINI_RETRY_BACKOFF_SEC * (2 ** attempt)
            time.sleep(wait_sec)

    if response_text is None:
        return {
            "success": False,
            "error": f"Gemini API 오류: {str(last_error)}",
            "references": references,
            "timing": timing,
            "raw_response": None,
        }

    try:
        # JSON 파싱 시도
        try:
            text = response_text.strip()
            # 1) 코드 블록 안의 JSON 추출
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            # 2) 마크다운만 반환된 경우: 첫 번째 '{' ~ 마지막 '}' 구간 추출 후 파싱
            if not text.startswith("{"):
                start = text.find("{")
                if start >= 0:
                    depth = 0
                    end = -1
                    for i in range(start, len(text)):
                        if text[i] == "{":
                            depth += 1
                        elif text[i] == "}":
                            depth -= 1
                            if depth == 0:
                                end = i
                                break
                    if end >= start:
                        text = text[start : end + 1]
            
            interpretation = json.loads(text)
            if references and isinstance(interpretation, dict) and not _has_per_item_sources(interpretation):
                interpretation["source"] = references
            return {
                "success": True,
                "interpretation": interpretation,
                "references": references,
                "timing": timing,
                "raw_response": response_text
            }
        except json.JSONDecodeError as e:
            # 잘린 응답(Unterminated string 등)일 때 잘린 위치에서 끊고 괄호 보완 후 재시도
            err_msg = str(e)
            repaired = None
            if getattr(e, "pos", None) is not None and ("Unterminated string" in err_msg or "Expecting" in err_msg):
                pos = min(e.pos, len(text))
                cut = text[:pos].rstrip()
                if cut.endswith(","):
                    cut = cut[:-1]
                if not cut.endswith('"'):
                    cut += '"'
                open_brackets = cut.count("[") - cut.count("]")
                open_braces = cut.count("{") - cut.count("}")
                cut += "]" * open_brackets + "}" * open_braces
                try:
                    repaired = json.loads(cut)
                except json.JSONDecodeError:
                    pass
            if repaired is not None:
                if references and isinstance(repaired, dict) and not _has_per_item_sources(repaired):
                    repaired["source"] = references
                return {
                    "success": True,
                    "interpretation": repaired,
                    "references": references,
                    "timing": timing,
                    "raw_response": response_text
                }
            return {
                "success": False,
                "error": f"JSON 파싱 오류: {str(e)}",
                "references": references,
                "timing": timing,
                "raw_response": response_text
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Gemini API 오류: {str(e)}",
            "references": references,
            "timing": timing,
            "raw_response": None
        }


def _build_interpretations_summary(results):
    """results에서 그림별 해석(interpretation)을 추출해 종합용 요약 문자열 생성."""
    label_map = {"tree": "나무", "house": "집", "man": "남자사람", "woman": "여자사람"}
    extract_keys = ("인상적_해석", "구조적_해석", "표상적_해석", "정서_영역_소견")
    parts = []
    for en_key in ("tree", "house", "man", "woman"):
        r = results.get(en_key) if isinstance(results, dict) else None
        interp = (r or {}).get("interpretation") if isinstance(r, dict) else None
        if not interp or not isinstance(interp, dict):
            continue
        label = label_map.get(en_key, en_key)
        lines = [f"[{label} 해석]"]
        for key in extract_keys:
            val = interp.get(key)
            if isinstance(val, str) and val.strip():
                lines.append(f"- {key}: {val.strip()}")
            elif isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, str) and v.strip():
                        lines.append(f"- {k}: {v.strip()}")
        if len(lines) > 1:
            parts.append("\n".join(lines))
    return "\n\n".join(parts) if parts else ""


def _call_name_with_ui(full_name: str) -> str:
    """
    풀네임을 호칭+의 형태(예: O이의)로 변환.
    한글 마지막 글자 받침 있으면 '이의', 없으면 '의'.
    """
    s = (full_name or "").strip()
    for suf in ("님", "군", "양"):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    if not s or s == "아이":
        return "아이의"
    given = s[1:] if len(s) > 1 else s
    if not given:
        return "아이의"
    last = given[-1]
    code = ord(last)
    has_batchim = (
        0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0
    )
    return given + ("이의" if has_batchim else "의")


def generate_overall_psychology(results, child_name, age_str, gender_kr, api_key=None, model_name=None, use_rag=True, rag_db_path=None, rag_k=10):
    """
    RAG(ChromaDB)로 HTP 참고 지표를 검색한 뒤, 그림별 해석(interpretation)을 바탕으로
    전체_심리_결과 4개 필드를 한 번에 생성합니다.
    """
    default_out = {
        "종합_요약": "",
        "인상적_분석": "",
        "구조적_분석_요약": "",
        "표상적_분석_종합": "",
    }
    if not GEMINI_AVAILABLE:
        print("[generate_overall_psychology] google-genai 미설치로 스킵")
        return default_out

    # 그림별 해석 요약 (종합의 핵심 입력)
    interpretations_summary = _build_interpretations_summary(results)

    # RAG 쿼리: 네 그림 분석이 있으면 합쳐서, 없으면 종합용 고정 쿼리
    query_parts = []
    for en_key in ("tree", "house", "man", "woman"):
        r = results.get(en_key) if isinstance(results, dict) else None
        analysis = (r or {}).get("analysis") if isinstance(r, dict) else None
        if analysis and isinstance(analysis, dict):
            query_parts.append(_build_rag_query_from_analysis(analysis))
    combined_query = " ".join(p for p in query_parts if p).strip() or "HTP 나무 집 남자사람 여자사람 종합 심리 해석"

    rag_context = ""
    if use_rag and RAG_AVAILABLE:
        rag_context, _ = get_rag_context_by_query(combined_query, db_path=rag_db_path, api_key=api_key, k=rag_k)

    model_name = model_name or DEFAULT_MODEL
    age = age_str or "0"
    sex = "남아" if gender_kr == "남" else "여아" if gender_kr == "여" else "아동"
    child_name = (child_name or "").strip()
    name_part = f"{child_name} " if child_name else ""
    call_name_ui = _call_name_with_ui(child_name) if child_name else "아이의"

    # 프롬프트: 그림별 해석을 반드시 포함 (종합의 근거)
    interp_block = f"\n[그림별 해석]\n{interpretations_summary}\n\n" if interpretations_summary.strip() else ""
    rag_block = f"\n[참고 지표]\n{rag_context}\n\n" if rag_context.strip() else ""

    prompt = f"""{name_part}{age}세 {sex}의 HTP 네 그림(나무, 집, 남자사람, 여자사람)에 대한 **종합** 심리 결과 4개 필드를 작성해주세요.
{interp_block}{rag_block}
**요청**:
- 위 그림별 해석을 **반드시 종합**하여 전체 심리 결과를 작성하세요.
- 아동을 지칭할 때는 반드시 "{call_name_ui}" 형태로만 쓸 것. 예: "{call_name_ui} 그림에서", "{call_name_ui} 그림 전반에서". "{child_name} 아동", "{child_name} 아동의" 같은 표현은 사용하지 마세요.
- "종합_요약", "인상적_분석", "표상적_분석_종합"은 **서로 다른 관점·다른 문장**으로 쓸 것. 같은 문장 반복 금지.
- "구조적_분석_요약"은 빈 문자열 "" 로 두세요.

**아래 4개 키만 가진 JSON 하나만** 출력하세요. 다른 설명·마크다운 없이 JSON만 출력합니다.

출력 구조 (키 이름 그대로):
- "종합_요약": 전체 심리·정서·발달 종합 한 문단 (2~3문장)
- "인상적_분석": 인상적 관점 한 문단 (2~3문장)
- "구조적_분석_요약": ""
- "표상적_분석_종합": 표상적 관점 한 문단 (2~3문장)
"""

    config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1024,
    )

    last_error = None
    for attempt in range(GEMINI_RETRY_MAX):
        current_key = resolve_gemini_api_key(api_key, attempt)
        if not current_key:
            print("[generate_overall_psychology] API 키 없음")
            return default_out
        try:
            client = setup_gemini(current_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                print("[generate_overall_psychology] Gemini 응답 텍스트 없음")
                return default_out
            if "```" in text:
                for part in text.split("```"):
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[3:].strip()
                    if part.startswith("{"):
                        text = part
                        break
            if not text.startswith("{"):
                start = text.find("{")
                if start >= 0:
                    depth, end = 0, -1
                    for i in range(start, len(text)):
                        if text[i] == "{": depth += 1
                        elif text[i] == "}": depth -= 1
                        if depth == 0: end = i; break
                    if end >= start:
                        text = text[start : end + 1]
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return {
                    "종합_요약": (parsed.get("종합_요약") or "").strip(),
                    "인상적_분석": (parsed.get("인상적_분석") or "").strip(),
                    "구조적_분석_요약": (parsed.get("구조적_분석_요약") or "").strip(),
                    "표상적_분석_종합": (parsed.get("표상적_분석_종합") or "").strip(),
                }
            print("[generate_overall_psychology] 응답이 JSON 객체가 아님")
            return default_out
        except Exception as e:
            last_error = e
            if _is_retryable_gemini_error(e) and attempt < GEMINI_RETRY_MAX - 1:
                wait_sec = GEMINI_RETRY_BACKOFF_SEC * (2 ** attempt)
                time.sleep(wait_sec)
                continue
            print(f"[generate_overall_psychology] 실패: {e}")
            return default_out
    return default_out


def analyze_and_interpret(json_data, model_name=None, api_key=None, temperature=0.2, use_rag=True, rag_db_path=None, rag_k=5, max_output_tokens=4096):
    """
    JSON 데이터를 분석하고 Gemini로 해석하는 통합 함수.
    use_rag=True이면 ChromaDB에서 관련 HTP 지표를 검색해 RAG로 프롬프트에 넣은 뒤 해석한다.
    
    Args:
        json_data: 원본 JSON 데이터 (딕셔너리)
        model_name: 사용할 Gemini 모델명 (None이면 gemini-2.0-flash)
        api_key: Gemini API 키 (없으면 환경변수에서 가져옴)
        temperature: 생성 온도 (0.0 ~ 1.0, 기본값: 0.7)
        use_rag: True이면 ChromaDB 참고 지표 검색 후 프롬프트에 포함 (기본값: True)
        rag_db_path: ChromaDB persist 디렉터리 (None이면 HTP_DB_PATH)
        rag_k: RAG 검색 상위 k개 (기본값: 5). 늘리면 참고 지표 증가
        max_output_tokens: LLM 최대 출력 토큰 (기본값: 4096). 늘리면 해석 잘림 방지
    
    Returns:
        {
            "analysis": 분석 결과 (딕셔너리),
            "interpretation": 해석 결과 (딕셔너리 또는 None),
            "success": 성공 여부 (bool),
            "error": 오류 메시지 (있는 경우),
            "timing": { "analysis_sec", "rag_sec", "llm_sec" } (구간별 소요 시간)
        }
    """
    timing = {"analysis_sec": 0.0, "rag_sec": 0.0, "llm_sec": 0.0}
    try:
        # 1. 분석 실행 (로컬, 보통 1초 미만)
        t0 = time.perf_counter()
        json_str = json.dumps(json_data, ensure_ascii=False)
        analysis_result = process_json(json_str)
        analysis_data = json.loads(analysis_result)
        timing["analysis_sec"] = time.perf_counter() - t0
        
        # 2. Gemini로 해석 (RAG 사용 시 ChromaDB 검색 후 프롬프트에 포함. LLM 호출이 대부분의 시간 차지)
        result = interpret_with_gemini(
            analysis_result,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            use_rag=use_rag,
            rag_db_path=rag_db_path,
            rag_k=rag_k,
            max_output_tokens=max_output_tokens,
        )
        t_rag_llm = result.get("timing") or {}
        timing["rag_sec"] = t_rag_llm.get("rag_sec", 0.0)
        timing["llm_sec"] = t_rag_llm.get("llm_sec", 0.0)
        
        if result["success"]:
            interp = result["interpretation"]
            refs = result.get("references", [])
            if refs and interp and isinstance(interp, dict) and not _has_per_item_sources(interp):
                interp["source"] = refs
            return {
                "analysis": analysis_data,
                "interpretation": interp,
                "source": refs,
                "success": True,
                "error": None,
                "timing": timing,
            }
        else:
            return {
                "analysis": analysis_data,
                "interpretation": None,
                "source": result.get("references", []),
                "success": False,
                "error": result.get("error", "알 수 없는 오류"),
                "raw_response": result.get("raw_response"),
                "timing": timing,
            }
            
    except Exception as e:
        return {
            "analysis": None,
            "interpretation": None,
            "source": [],
            "success": False,
            "error": f"처리 오류: {str(e)}",
            "timing": timing,
        }


def save_results(results, output_dir="results"):
    """
    분석 및 해석 결과를 파일로 저장

    Args:
        results: analyze_and_interpret()의 결과
        output_dir: 저장할 디렉토리
    """
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 분석 결과 저장
    if results["analysis"]:
        analysis_file = os.path.join(output_dir, f"analysis_{timestamp}.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(results["analysis"], f, ensure_ascii=False, indent=2)
        print(f"✓ 분석 결과 저장: {analysis_file}")
    
    # 해석 결과 저장
    if results["interpretation"]:
        interpretation_file = os.path.join(output_dir, f"interpretation_{timestamp}.json")
        with open(interpretation_file, 'w', encoding='utf-8') as f:
            json.dump(results["interpretation"], f, ensure_ascii=False, indent=2)
        print(f"✓ 해석 결과 저장: {interpretation_file}")
    
    # 통합 결과 저장
    combined_file = os.path.join(output_dir, f"combined_{timestamp}.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 통합 결과 저장: {combined_file}")


if __name__ == "__main__":
    print("해석은 main.py interpret 로 실행하세요.")
    print("  python main.py interpret 원본그림.json -o results/")

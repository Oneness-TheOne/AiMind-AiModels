#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Gemini API를 사용한 그림 분석 해석 통합 스크립트
RAG: ChromaDB에 저장된 HTP 지표를 먼저 검색해 참고 지표로 넣고, 그 위에 LLM 해석을 수행 (토큰 절약 + 근거 반영)
"""

import json
import os
import time
import warnings
from tree_analyzer import process_json
from interpretation_prompts import get_interpretation_prompt
from htp_indicator_parser import SOURCE_FIELD

# google.generativeai deprecated 경고 무시 (현재 API는 동작함)
warnings.filterwarnings("ignore", message=".*google.generativeai.*", category=FutureWarning)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("경고: google-generativeai 패키지가 설치되지 않았습니다.")
    print("설치 방법: pip install google-generativeai")

DEFAULT_MODEL = "gemini-2.5-flash-lite"

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
    Gemini API 설정
    
    Args:
        api_key: Gemini API 키 (없으면 환경변수에서 가져옴)
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai 패키지가 필요합니다. pip install google-generativeai")
    
    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key is None:
            raise ValueError("API 키가 필요합니다. api_key 파라미터로 전달하거나 GEMINI_API_KEY 환경변수를 설정하세요.")
    
    genai.configure(api_key=api_key)
    return genai


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
    api_key = api_key or os.getenv("GEMINI_API_KEY")
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


def interpret_with_gemini(analysis_result, model_name=None, api_key=None, temperature=0.2, use_rag=True, rag_db_path=None, rag_k=10, max_output_tokens=8192):
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
        rag_k: RAG 검색 상위 k개 (기본값: 10). 줄이면(예: 5) 프롬프트 짧아져 LLM 속도 향상
        max_output_tokens: LLM 최대 출력 토큰 (기본값: 8192). 줄이면(예: 4096) 생성 속도 향상
    
    Returns:
        해석 결과 (딕셔너리). 성공 시 "timing" 키로 { "rag_sec", "llm_sec" } 포함
    """
    timing = {"rag_sec": 0.0, "llm_sec": 0.0}
    if model_name is None:
        model_name = DEFAULT_MODEL
    # Gemini 설정
    genai = setup_gemini(api_key)
    
    # RAG: 분석 결과로 ChromaDB 검색 후 참고 지표 문자열·참고 논문 목록 생성
    rag_context = ""
    references = []
    if use_rag:
        t0 = time.perf_counter()
        analysis_data = json.loads(analysis_result) if isinstance(analysis_result, str) else analysis_result
        rag_context, references = get_rag_context(analysis_data, db_path=rag_db_path, api_key=api_key, k=rag_k)
        timing["rag_sec"] = time.perf_counter() - t0
    
    # 프롬프트 생성 (RAG 컨텍스트·참고 논문 목록 있으면 참고 지표 + 항목별 출처 요청)
    prompt = get_interpretation_prompt(
        analysis_result,
        rag_context=rag_context if rag_context else None,
        references=references if references else None,
    )
    
    # 모델 생성
    model = genai.GenerativeModel(model_name)
    
    # 생성 설정 (해석 JSON이 길어지므로 토큰 상한 확대, 잘림 방지)
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": max_output_tokens,
    }
    
    try:
        # Gemini에 요청 (가장 시간 많이 소요)
        t0 = time.perf_counter()
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        timing["llm_sec"] = time.perf_counter() - t0
        
        # 응답 텍스트 추출
        response_text = response.text
        
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


def analyze_and_interpret(json_data, model_name=None, api_key=None, temperature=0.2, use_rag=True, rag_db_path=None, rag_k=10, max_output_tokens=8192):
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
        rag_k: RAG 검색 상위 k개 (기본값: 10). 줄이면 속도 향상
        max_output_tokens: LLM 최대 출력 토큰 (기본값: 8192). 줄이면 속도 향상
    
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
    import os
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

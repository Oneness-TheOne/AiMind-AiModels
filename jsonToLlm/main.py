#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AiMind-AiModels 메인 진입점

두 가지 워크플로우만 지원합니다.

1) ingest  : 논문 PDF → 청킹/지표 추출 → results/ JSON → 벡터화 → ChromaDB 저장
2) interpret: 원본 그림 JSON → 중간 분석 → RAG(ChromaDB 검색) → LLM → 해석 결과
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_JSON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "image_to_json", "result"))


def cmd_ingest(args: argparse.Namespace) -> None:
    """워크플로우 1: PDF → JSON → ChromaDB"""
    thesis_dir = getattr(args, "thesis_dir", "thesis")
    result_dir = getattr(args, "result_dir", "results")  # ingest 출력: htp_final_dataset*.json 저장 위치
    db_path = getattr(args, "db_path", "htp_knowledge_base")
    json_path = getattr(args, "json", None)
    from gemini_integration import resolve_gemini_api_key
    api_key = getattr(args, "api_key", None) or resolve_gemini_api_key()

    if json_path:
        # 이미 있는 JSON만 ChromaDB에 적재
        if not os.path.isfile(json_path):
            print(f"✗ JSON 파일을 찾을 수 없습니다: {json_path}")
            sys.exit(1)
        from store_to_chroma import create_vector_db
        create_vector_db(json_path, db_path=db_path)
        return

    # PDF → JSON → ChromaDB 전체 파이프라인
    if not api_key:
        print("✗ GEMINI_API_KEY 또는 GEMINI_API_KEYS가 필요합니다. (환경변수 또는 --api-key)")
        sys.exit(1)
    from htp_indicator_parser import run_pdf_to_json
    from store_to_chroma import create_vector_db

    out_path = run_pdf_to_json(thesis_dir=thesis_dir, result_dir=result_dir, api_key=api_key)
    if not out_path:
        sys.exit(1)
    create_vector_db(out_path, db_path=db_path)
    print("✓ ingest 완료: PDF → JSON → ChromaDB")


def cmd_interpret(args: argparse.Namespace) -> None:
    """워크플로우 2: 원본 JSON → 분석 → RAG → LLM → 해석 결과"""
    try:
        from gemini_integration import analyze_and_interpret, save_results, resolve_gemini_api_key
    except ImportError as e:
        print("✗ gemini_integration을 불러올 수 없습니다. pip install -r requirements.txt")
        print(f"  상세: {e}")
        sys.exit(1)

    api_key = getattr(args, "api_key", None)  # None이면 env에서 키 순환(503 시 다음 키)
    if not resolve_gemini_api_key(api_key):
        print("✗ GEMINI_API_KEY 또는 GEMINI_API_KEYS가 필요합니다. (환경변수 또는 --api-key)")
        sys.exit(1)

    output_dir = getattr(args, "output", None) or "results"

    def _convert_if_needed(data, source_path):
        try:
            from legacy_converter import is_new_format, convert_new_to_legacy
            if is_new_format(data):
                return convert_new_to_legacy(data, source_path)
        except ImportError:
            pass
        return data

    def _save_results_with_stem(results, out_dir, stem):
        os.makedirs(out_dir, exist_ok=True)
        if results.get("analysis"):
            analysis_file = os.path.join(out_dir, f"analysis_{stem}.json")
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(results["analysis"], f, ensure_ascii=False, indent=2)
            print(f"✓ 분석 결과 저장: {analysis_file}")
        if results.get("interpretation"):
            interpretation_file = os.path.join(out_dir, f"interpretation_{stem}.json")
            with open(interpretation_file, "w", encoding="utf-8") as f:
                json.dump(results["interpretation"], f, ensure_ascii=False, indent=2)
            print(f"✓ 해석 결과 저장: {interpretation_file}")
        combined_file = os.path.join(out_dir, f"combined_{stem}.json")
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ 통합 결과 저장: {combined_file}")

    def _run_single(path):
        with open(path, "r", encoding="utf-8") as f:
            original = json.load(f)
        original = _convert_if_needed(original, path)
        t0 = time.perf_counter()
        results = analyze_and_interpret(
            original,
            model_name=getattr(args, "model", "gemini-2.5-flash-lite"),
            api_key=api_key,
            temperature=getattr(args, "temperature", 0.7),
            use_rag=not getattr(args, "no_rag", False),
            rag_db_path=getattr(args, "rag_db_path", None),
            rag_k=getattr(args, "rag_k", 5),
            max_output_tokens=getattr(args, "max_output_tokens", 4096),
        )
        elapsed = time.perf_counter() - t0
        return results, elapsed

    if os.path.isdir(args.input):
        json_paths = [
            os.path.join(args.input, name)
            for name in os.listdir(args.input)
            if name.lower().endswith(".json")
        ]
        json_paths = [p for p in json_paths if os.path.isfile(p)]
        json_paths.sort(key=lambda p: os.path.basename(p).casefold())
        if not json_paths:
            print(f"✗ JSON 파일을 찾을 수 없습니다: {args.input}")
            sys.exit(1)

        for i, path in enumerate(json_paths, 1):
            stem = os.path.splitext(os.path.basename(path))[0]
            print(f"\n[{i}/{len(json_paths)}] {path}")
            results, elapsed = _run_single(path)
            if not results.get("success"):
                print(f"✗ 해석 실패: {results.get('error')}")
                if results.get("raw_response"):
                    print("\n[원본 응답 (처음 500자)]")
                    print(results["raw_response"][:500])
                continue
            print("✓ 해석 성공")
            analysis = results.get("analysis") or {}
            interp = results.get("interpretation") or {}
            if analysis:
                meta = analysis.get("이미지_메타정보", {})
                print("\n[분석 요약] 연령:", meta.get("연령"), "성별:", meta.get("성별"), "요소 수:", len(analysis.get("요소_개수", {})))
            if interp:
                if "전체_요약" in interp:
                    print("\n[해석 요약]", interp["전체_요약"][:200] + "..." if len(interp["전체_요약"]) > 200 else interp["전체_요약"])
            _save_results_with_stem(results, output_dir, stem)
            timing = results.get("timing") or {}
            if timing:
                a, r, l_ = timing.get("analysis_sec", 0), timing.get("rag_sec", 0), timing.get("llm_sec", 0)
                print(f"   └ 분석: {a:.1f}초 | RAG(ChromaDB): {r:.1f}초 | LLM(Gemini): {l_:.1f}초")
            print(f"⏱ 소요 시간: {elapsed:.1f}초")
        print("\n✓ 배치 완료")
        return

    results, elapsed = _run_single(args.input)
    if not results.get("success"):
        print(f"✗ 해석 실패: {results.get('error')}")
        if results.get("raw_response"):
            print("\n[원본 응답 (처음 500자)]")
            print(results["raw_response"][:500])
        sys.exit(1)

    print("✓ 해석 성공")
    analysis = results.get("analysis") or {}
    interp = results.get("interpretation") or {}
    if analysis:
        meta = analysis.get("이미지_메타정보", {})
        print("\n[분석 요약] 연령:", meta.get("연령"), "성별:", meta.get("성별"), "요소 수:", len(analysis.get("요소_개수", {})))
    if interp:
        if "전체_요약" in interp:
            print("\n[해석 요약]", interp["전체_요약"][:200] + "..." if len(interp["전체_요약"]) > 200 else interp["전체_요약"])

    if output_dir:
        save_results(results, output_dir=output_dir)

    print(f"\n⏱ 총 소요 시간: {elapsed:.1f}초 (시작 → JSON 저장까지)")
    timing = results.get("timing") or {}
    if timing:
        a, r, l_ = timing.get("analysis_sec", 0), timing.get("rag_sec", 0), timing.get("llm_sec", 0)
        print(f"   └ 분석: {a:.1f}초 | RAG(ChromaDB): {r:.1f}초 | LLM(Gemini): {l_:.1f}초  ← LLM이 대부분 차지")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AiMind-AiModels: (1) PDF→ChromaDB 적재 (2) 그림 JSON→RAG·LLM 해석",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1) ingest: PDF → JSON → ChromaDB
    p_ingest = sub.add_parser("ingest", help="논문 PDF → 청킹/지표 추출 → results/ JSON → ChromaDB 저장")
    p_ingest.add_argument("--thesis-dir", default="thesis", help="PDF 폴더 (기본: thesis)")
    p_ingest.add_argument("--result-dir", default="results", help="지표 JSON 저장 폴더 (기본: results)")
    p_ingest.add_argument("--db-path", default="htp_knowledge_base", help="ChromaDB 저장 경로 (기본: htp_knowledge_base)")
    p_ingest.add_argument("--json", metavar="PATH", help="이미 있는 JSON만 ChromaDB에 적재 (PDF 단계 생략)")
    p_ingest.add_argument("--api-key", help="Gemini API 키 (미지정 시 GEMINI_API_KEYS/GEMINI_API_KEY)")
    p_ingest.set_defaults(func=cmd_ingest)

    # 2) interpret: 원본 JSON → 분석 → RAG → LLM
    p_interp = sub.add_parser("interpret", help="원본 그림 JSON → 중간 분석 → RAG → LLM 해석")
    p_interp.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_IMAGE_JSON_DIR,
        help=f"원본 그림 라벨링 JSON 경로 또는 JSON 폴더 경로 (기본: {DEFAULT_IMAGE_JSON_DIR})",
    )
    p_interp.add_argument("-o", "--output", default="results", help="결과 저장 디렉터리 (analysis/interpretation/combined JSON)")
    p_interp.add_argument("-m", "--model", default="gemini-2.5-flash-lite", help="Gemini 모델 (기본: gemini-2.5-flash-lite)")
    p_interp.add_argument("--api-key", help="Gemini API 키 (미지정 시 GEMINI_API_KEYS/GEMINI_API_KEY)")
    p_interp.add_argument("--temperature", type=float, default=0.2, help="생성 온도 (기본: 0.2)")
    p_interp.add_argument("--no-rag", action="store_true", help="ChromaDB RAG 미사용 (프롬프트 짧아져 속도 향상)")
    p_interp.add_argument("--rag-k", type=int, default=10, help="RAG 검색 상위 k개 (기본: 10). 5로 줄이면 속도 향상")
    p_interp.add_argument("--max-output-tokens", type=int, default=8192, help="LLM 최대 출력 토큰 (기본: 8192). 4096으로 줄이면 속도 향상")
    p_interp.add_argument("--rag-db-path", default=None, help="ChromaDB 경로 (기본: htp_knowledge_base)")
    p_interp.set_defaults(func=cmd_interpret)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

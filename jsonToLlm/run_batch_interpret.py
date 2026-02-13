#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새 형식 JSON 4개(나무, 남자사람, 여자사람, 집)를 분석해 interpretation JSON을 results/에 저장.

사용법:
  cd jsonToLlm
  python run_batch_interpret.py [--no-rag] [-o results/]

기본 입력 파일: 나무_7_남_00367.json, 남자사람_7_남_00978.json, 여자사람_7_남_01606.json, 집_7_남_00885.json
"""

import argparse
import json
import os
import sys
import time

# jsonToLlm 기준 실행 가정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_JSON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "image_to_json", "result"))
os.chdir(SCRIPT_DIR)

DEFAULT_INPUTS = [
    os.path.join(IMAGE_JSON_DIR, "나무_7_남_00367.json"),
    os.path.join(IMAGE_JSON_DIR, "남자사람_7_남_00978.json"),
    os.path.join(IMAGE_JSON_DIR, "여자사람_7_남_01606.json"),
    os.path.join(IMAGE_JSON_DIR, "집_7_남_00885.json"),
]


def main():
    parser = argparse.ArgumentParser(description="4개 그림 JSON → interpretation JSON 배치 실행")
    parser.add_argument("-o", "--output", default="results", help="결과 저장 디렉터리 (기본: results)")
    parser.add_argument("--no-rag", action="store_true", help="RAG(ChromaDB) 미사용")
    parser.add_argument("--inputs", nargs="+", default=None, help="입력 JSON 경로 (기본: 나무/남자/여자/집 4개)")
    parser.add_argument("--api-key", default=None, help="Gemini API 키 (미지정 시 GEMINI_API_KEYS/GEMINI_API_KEY)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    try:
        from gemini_integration import analyze_and_interpret, resolve_gemini_api_key
    except ImportError as e:
        print("✗ gemini_integration을 불러올 수 없습니다. pip install -r requirements.txt")
        print(f"  상세: {e}")
        sys.exit(1)

    api_key = args.api_key  # None이면 env에서 키 순환(503 시 다음 키)
    if not resolve_gemini_api_key(api_key):
        print("✗ GEMINI_API_KEY 또는 GEMINI_API_KEYS가 필요합니다.")
        sys.exit(1)

    try:
        from legacy_converter import is_new_format, convert_new_to_legacy
    except ImportError:
        is_new_format = lambda d: False
        convert_new_to_legacy = lambda d, p: d

    inputs = args.inputs or DEFAULT_INPUTS
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(inputs):
        if not os.path.isfile(path):
            print(f"✗ 파일 없음: {path}")
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"\n[{i+1}/{len(inputs)}] {path} → interpretation_{stem}.json")
        with open(path, "r", encoding="utf-8") as f:
            original = json.load(f)
        if is_new_format(original):
            original = convert_new_to_legacy(original, path)
        t0 = time.perf_counter()
        results = analyze_and_interpret(
            original,
            model_name="gemini-2.5-flash-lite",
            api_key=api_key,
            temperature=0.2,
            use_rag=not args.no_rag,
            rag_db_path=None,
            rag_k=5,
            max_output_tokens=4096,
        )
        elapsed = time.perf_counter() - t0
        if not results.get("success"):
            print(f"  ✗ 해석 실패: {results.get('error')}")
            continue
        interp = results.get("interpretation")
        if interp:
            out_path = os.path.join(output_dir, f"interpretation_{stem}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(interp, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 저장: {out_path} ({elapsed:.1f}초)")
        else:
            print(f"  ✗ 해석 결과 없음")

    print("\n✓ 배치 완료.")


if __name__ == "__main__":
    main()

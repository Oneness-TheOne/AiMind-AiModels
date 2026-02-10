from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from analysis_metrics import (
    compute_image_metrics,
    compute_peer_summary_by_folder,
    load_peer_stats,
    normalize_sex,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_age_sex(json_items: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[str]]:
    age = None
    sex = None
    for item in json_items:
        meta = item.get("meta", {})
        if age is None:
            age = meta.get("age")
        if sex is None:
            sex = meta.get("sex")
    if isinstance(age, str) and age.isdigit():
        age = int(age)
    if isinstance(age, float):
        age = int(age)
    if isinstance(sex, str):
        sex = normalize_sex(sex)
    return age if isinstance(age, int) and age > 0 else None, sex


def _collect_metrics(json_paths: Dict[str, Path]) -> Tuple[List[Dict[str, Optional[float]]], List[str]]:
    per_object = []
    folder_keys = []
    order = [
        ("tree", "TL_나무"),
        ("house", "TL_집"),
        ("man", "TL_남자사람"),
        ("woman", "TL_여자사람"),
    ]
    for key, folder in order:
        item = _load_json(json_paths[key])
        metrics = compute_image_metrics(item, image_path=None, use_color=False)
        per_object.append(metrics)
        folder_keys.append(folder)
    return per_object, folder_keys


def _build_output(
    age: int,
    sex: str,
    folder_keys: List[str],
    comparison: Dict[str, Any],
    per_object: Optional[List[Dict[str, Optional[float]]]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "age": age,
        "sex": sex,
        "folder_keys": folder_keys,
        "comparison": comparison,
    }
    if per_object is not None:
        payload["metrics"] = per_object
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="또래 비교용 백분위 요약 JSON 생성 도구",
    )
    parser.add_argument("--tree", required=True, help="TL_나무 JSON 경로")
    parser.add_argument("--house", required=True, help="TL_집 JSON 경로")
    parser.add_argument("--man", required=True, help="TL_남자사람 JSON 경로")
    parser.add_argument("--woman", required=True, help="TL_여자사람 JSON 경로")
    parser.add_argument("--age", type=int, default=0, help="나이(미입력 시 JSON에서 추정)")
    parser.add_argument("--sex", default="", help="성별(남/여, 미입력 시 JSON에서 추정)")
    parser.add_argument(
        "--stats",
        default=str(Path(__file__).parent / "data" / "label_stats_by_group.json"),
        help="label_stats_by_group.json 경로",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "data" / "peer_compare_latest.json"),
        help="출력 JSON 경로",
    )
    parser.add_argument(
        "--include-metrics",
        action="store_true",
        help="per-object 지표를 결과에 포함",
    )
    args = parser.parse_args()

    json_paths = {
        "tree": Path(args.tree),
        "house": Path(args.house),
        "man": Path(args.man),
        "woman": Path(args.woman),
    }
    missing = [k for k, p in json_paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"JSON 파일이 없습니다: {', '.join(missing)}")

    json_items = [_load_json(path) for path in json_paths.values()]
    inferred_age, inferred_sex = _infer_age_sex(json_items)

    age = args.age or (inferred_age or 0)
    sex = normalize_sex(args.sex) if args.sex else (inferred_sex or "미상")

    if age <= 0 or sex not in {"남", "여"}:
        raise SystemExit("나이/성별을 확인할 수 없습니다. --age/--sex를 지정해 주세요.")

    per_object, folder_keys = _collect_metrics(json_paths)

    stats = load_peer_stats(Path(args.stats))
    if not stats:
        raise SystemExit("label_stats_by_group.json을 불러올 수 없습니다.")

    comparison = compute_peer_summary_by_folder(per_object, stats, age, sex, folder_keys)
    output = _build_output(
        age,
        sex,
        folder_keys,
        comparison,
        per_object if args.include_metrics else None,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

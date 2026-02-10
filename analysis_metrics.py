from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


CORE_LABELS = {
    "나무": {"나무전체", "수관", "기둥", "가지", "뿌리"},
    "집": {"집전체", "지붕", "집벽", "문", "창문"},
    "남자사람": {
        "사람전체",
        "머리",
        "얼굴",
        "눈",
        "코",
        "입",
        "귀",
        "머리카락",
        "목",
        "상체",
        "팔",
        "손",
        "다리",
        "발",
    },
    "여자사람": {
        "사람전체",
        "머리",
        "얼굴",
        "눈",
        "코",
        "입",
        "귀",
        "머리카락",
        "목",
        "상체",
        "팔",
        "손",
        "다리",
        "발",
    },
}

HOUSE_OPENNESS_LABELS = {"문", "창문", "굴뚝", "연기"}
PERSON_DETAIL_LABELS = {
    "머리",
    "얼굴",
    "눈",
    "코",
    "입",
    "귀",
    "머리카락",
    "목",
    "상체",
    "팔",
    "손",
    "다리",
    "발",
}


def _parse_resolution(meta: Dict[str, Any]) -> Tuple[int, int]:
    raw = (meta.get("img_resolution") or "0x0").split("x")
    try:
        return int(raw[0]), int(raw[1])
    except (IndexError, ValueError):
        return 0, 0


def _bbox_area(bbox: Dict[str, Any]) -> float:
    return float(bbox.get("w", 0)) * float(bbox.get("h", 0))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _overall_bbox(bboxes: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    overall = None
    max_area = 0.0
    for bbox in bboxes:
        label = str(bbox.get("label", ""))
        area = _bbox_area(bbox)
        if label.endswith("전체"):
            return bbox
        if area > max_area:
            max_area = area
            overall = bbox
    return overall


def _centroid(bbox: Dict[str, Any]) -> Tuple[float, float]:
    return (
        float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2.0,
        float(bbox.get("y", 0)) + float(bbox.get("h", 0)) / 2.0,
    )


def _compute_color_usage(image_path: Optional[str]) -> Optional[float]:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((64, 64))
            palette = img.convert("P", palette=Image.ADAPTIVE, colors=16)
            used_colors = len(set(palette.getdata()))
            return float(used_colors)
    except Exception:
        return None


def _compute_spatial_dispersion(
    bboxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> float:
    if not bboxes or img_w <= 0 or img_h <= 0:
        return 0.0
    xs = []
    ys = []
    for bbox in bboxes:
        cx, cy = _centroid(bbox)
        xs.append(_safe_div(cx, img_w))
        ys.append(_safe_div(cy, img_h))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
    var_y = sum((y - mean_y) ** 2 for y in ys) / len(ys)
    return math.sqrt(var_x + var_y)


def _compute_center_balance(
    bboxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> float:
    if not bboxes or img_w <= 0 or img_h <= 0:
        return 0.5
    xs = []
    ys = []
    for bbox in bboxes:
        cx, cy = _centroid(bbox)
        xs.append(_safe_div(cx, img_w))
        ys.append(_safe_div(cy, img_h))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dist = math.sqrt((mean_x - 0.5) ** 2 + (mean_y - 0.5) ** 2)
    max_dist = math.sqrt(0.5**2 + 0.5**2)
    return max(0.0, 1.0 - _safe_div(dist, max_dist))


def compute_image_metrics(
    image_json: Dict[str, Any],
    image_path: Optional[str] = None,
    use_color: bool = True,
) -> Dict[str, Optional[float]]:
    meta = image_json.get("meta", {})
    img_w, img_h = _parse_resolution(meta)
    bboxes = image_json.get("annotations", {}).get("bbox", []) or []
    class_type = image_json.get("annotations", {}).get("class", "")

    overall = _overall_bbox(bboxes)
    overall_area = _bbox_area(overall) if overall else 0.0
    img_area = float(img_w * img_h)

    total_count = len(bboxes)
    detail_count = len([b for b in bboxes if not str(b.get("label", "")).endswith("전체")])
    space_usage = _safe_div(overall_area, img_area)

    part_areas = [
        _bbox_area(b) for b in bboxes if b is not overall and _bbox_area(b) > 0
    ]
    if part_areas and overall_area > 0:
        ratios = [min(1.0, _safe_div(area, overall_area)) for area in part_areas]
        mean_ratio = sum(ratios) / len(ratios)
        std_ratio = math.sqrt(sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios))
        proportion_score = 1.0 / (1.0 + std_ratio)
    else:
        proportion_score = 0.0

    core = CORE_LABELS.get(class_type, set())
    unique_labels = {str(b.get("label", "")) for b in bboxes}
    accessory_labels = [label for label in unique_labels if label and label not in core]
    creativity_score = float(len(accessory_labels))

    spatial_dispersion = _compute_spatial_dispersion(bboxes, img_w, img_h)
    center_balance = _compute_center_balance(bboxes, img_w, img_h)

    house_openness = None
    if class_type == "집":
        house_openness = float(
            sum(1 for b in bboxes if str(b.get("label", "")) in HOUSE_OPENNESS_LABELS)
        )

    person_detail = None
    if class_type in {"남자사람", "여자사람"}:
        person_detail = float(
            sum(1 for b in bboxes if str(b.get("label", "")) in PERSON_DETAIL_LABELS)
        )

    color_usage = _compute_color_usage(image_path) if use_color else None

    emotional_balance = (center_balance + proportion_score) / 2.0

    return {
        "detail": float(detail_count),
        "color_usage": color_usage,
        "space_usage": float(space_usage),
        "proportion": float(proportion_score),
        "creativity": float(creativity_score),
        "complexity": float(total_count),
        "spatial_awareness": float(spatial_dispersion),
        "center_balance": float(center_balance),
        "emotional_balance": float(emotional_balance),
        "house_openness": house_openness,
        "person_detail": person_detail,
        "person_size": float(space_usage) if class_type in {"남자사람", "여자사람"} else None,
    }


def aggregate_metrics(per_object: List[Dict[str, Optional[float]]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for metrics in per_object:
        for key, value in metrics.items():
            if value is None:
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def compute_percentile(value: float, sorted_values: List[float]) -> float:
    if not sorted_values:
        return 0.0
    import bisect

    idx = bisect.bisect_right(sorted_values, value)
    return round((idx / len(sorted_values)) * 100, 1)


def normalize_sex(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"male", "m", "남", "남아"}:
        return "남"
    if v in {"female", "f", "여", "여아"}:
        return "여"
    return "미상"


def _stage_from_age(age: int) -> str:
    if age <= 6:
        return "전도식기 (4-6세)"
    if age <= 9:
        return "도식기 (7-9세)"
    if age <= 13:
        return "사실기 (10-13세)"
    return "사실기 (10세 이상)"


def compute_peer_summary(
    per_object: List[Dict[str, Optional[float]]],
    stats: Dict[str, Any],
    age: int,
    sex: str,
) -> Dict[str, Any]:
    group_key = f"{age}_{sex}"
    group = stats.get("groups", {}).get(group_key)
    if not group:
        return {}

    combined = aggregate_metrics(per_object)

    peer_mapping = {
        "세부묘사": "detail",
        "공간활용": "space_usage",
        "비율표현": "proportion",
        "창의성": "creativity",
    }
    dev_mapping = {
        "그림 복잡도": "complexity",
        "세부 표현력": "detail",
        "공간 인식": "spatial_awareness",
        "비율 표현": "proportion",
    }
    psych_mapping = {
        "자아 존중감": "person_size",
        "정서 안정": "emotional_balance",
        "사회성": "person_detail",
        "창의성": "creativity",
        "가족 관계": "house_openness",
    }

    peer_scores = {}
    for label, key in peer_mapping.items():
        value = combined.get(key)
        if value is None:
            continue
        peer_scores[label] = compute_percentile(float(value), group.get(key, []))

    dev_scores = {}
    for label, key in dev_mapping.items():
        value = combined.get(key)
        if value is None:
            continue
        dev_scores[label] = compute_percentile(float(value), group.get(key, []))

    psych_scores = {}
    for label, key in psych_mapping.items():
        value = combined.get(key)
        if value is None:
            continue
        psych_scores[label] = compute_percentile(float(value), group.get(key, []))

    overall_values = [v for v in peer_scores.values() if isinstance(v, (int, float))]
    overall_score = round(sum(overall_values) / len(overall_values), 1) if overall_values else 0.0

    return {
        "peer": peer_scores,
        "development": {
            "stage": _stage_from_age(age),
            "scores": dev_scores,
        },
        "psychology": {
            "scores": psych_scores,
        },
        "overall_score": overall_score,
    }


def load_peer_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_peer_summary_by_folder(
    per_object: List[Dict[str, Optional[float]]],
    folder_stats: Dict[str, Any],
    age: int,
    sex: str,
    folder_keys: List[str],
) -> Dict[str, Any]:
    """
    label_stats_by_group.json 기반 백분위 계산.
    folder_keys는 per_object의 순서에 맞춰 TL_* 폴더명을 전달한다.
    """
    group_key = f"{age}_{sex}"
    if not folder_stats:
        return {}

    peer_mapping = {
        "세부묘사": "detail",
        "공간활용": "space_usage",
        "비율표현": "proportion",
        "창의성": "creativity",
    }
    dev_mapping = {
        "그림 복잡도": "complexity",
        "세부 표현력": "detail",
        "공간 인식": "spatial_awareness",
        "비율 표현": "proportion",
    }
    psych_mapping = {
        "자아 존중감": "person_size",
        "정서 안정": "emotional_balance",
        "사회성": "person_detail",
        "창의성": "creativity",
        "가족 관계": "house_openness",
    }

    peer_scores: Dict[str, List[float]] = {k: [] for k in peer_mapping}
    dev_scores: Dict[str, List[float]] = {k: [] for k in dev_mapping}
    psych_scores: Dict[str, List[float]] = {k: [] for k in psych_mapping}

    for metrics, folder_key in zip(per_object, folder_keys):
        folder_data = folder_stats.get("folders", {}).get(folder_key, {})
        group = folder_data.get("groups", {}).get(group_key, {})
        if not group:
            continue

        for label, key in peer_mapping.items():
            value = metrics.get(key)
            if value is None:
                continue
            peer_scores[label].append(compute_percentile(float(value), group.get(key, [])))

        for label, key in dev_mapping.items():
            value = metrics.get(key)
            if value is None:
                continue
            dev_scores[label].append(compute_percentile(float(value), group.get(key, [])))

        for label, key in psych_mapping.items():
            value = metrics.get(key)
            if value is None:
                continue
            psych_scores[label].append(compute_percentile(float(value), group.get(key, [])))

    def _avg(values: List[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 1)

    peer_out = {label: _avg(values) for label, values in peer_scores.items() if values}
    dev_out = {label: _avg(values) for label, values in dev_scores.items() if values}
    psych_out = {label: _avg(values) for label, values in psych_scores.items() if values}

    overall_values = [v for v in peer_out.values() if isinstance(v, (int, float))]
    overall_score = round(sum(overall_values) / len(overall_values), 1) if overall_values else 0.0

    return {
        "peer": peer_out,
        "development": {
            "stage": _stage_from_age(age),
            "scores": dev_out,
        },
        "psychology": {
            "scores": psych_out,
        },
        "overall_score": overall_score,
    }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새 형식 JSON (features: ratio, center_x, center_y) → 기존 형식 (meta + annotations.bbox) 변환.
tree_analyzer.process_json()가 기대하는 형식으로 변환해 interpret 파이프라인에 넣을 수 있게 함.
"""

import json
import math
import os


def _parse_age_sex_from_filename(filepath):
    """
    파일명에서 연령·성별 추출. 예: 나무_7_남_00367.json → (7, '남')
    """
    basename = os.path.basename(filepath or "").replace(".json", "")
    parts = basename.split("_")
    age = 7
    sex = "남"
    if len(parts) >= 3:
        try:
            age = int(parts[1])
        except (ValueError, IndexError):
            pass
        if parts[2] in ("남", "여"):
            sex = parts[2]
    return age, sex


def _feature_to_bbox(feat, img_w, img_h):
    """
    feature 한 개 { ratio, center_x, center_y } → 픽셀 bbox { x, y, w, h }.
    ratio = 면적비율(0~1), center_x/y = 정규화 중심(0~1).
    정사각형 bbox 가정: area = ratio * W*H, w = h = sqrt(area).
    """
    ratio = float(feat.get("ratio") or 0)
    cx = float(feat.get("center_x") or 0.5)
    cy = float(feat.get("center_y") or 0.5)
    if ratio <= 0 or cx < 0 or cy < 0:
        return None
    area = ratio * img_w * img_h
    if area <= 0:
        return None
    side = math.sqrt(area)
    w = h = max(1, min(int(round(side)), img_w, img_h))
    x = int(round(cx * img_w - w / 2))
    y = int(round(cy * img_h - h / 2))
    x = max(0, min(x, img_w - w))
    y = max(0, min(y, img_h - h))
    return {"x": x, "y": y, "w": w, "h": h}


def convert_new_to_legacy(data, filename_for_meta=None):
    """
    새 형식 JSON 딕셔너리 → 기존 형식(meta + annotations.bbox) 딕셔너리.

    새 형식: image_size, object_type_kr, features (label → { has, ratio, center_x, center_y }).
    기존 형식: meta (img_resolution, age, sex), annotations (class, bbox: [ { label, x, y, w, h } ]).

    filename_for_meta: 연령/성별 추출용 파일경로 (예: 나무_7_남_00367.json).
    """
    if "annotations" in data and "bbox" in data.get("annotations"):
        return data

    img_size = data.get("image_size") or {}
    img_w = int(img_size.get("width") or 1280)
    img_h = int(img_size.get("height") or 1280)
    resolution = f"{img_w}x{img_h}"

    age, sex = _parse_age_sex_from_filename(filename_for_meta)

    class_kr = data.get("object_type_kr") or "나무"
    # object_type이 man/woman이면 남자사람/여자사람으로
    ot = data.get("object_type", "")
    if ot == "man":
        class_kr = "남자사람"
    elif ot == "woman":
        class_kr = "여자사람"
    elif ot == "house":
        class_kr = "집"
    elif ot == "tree":
        class_kr = "나무"

    features = data.get("features") or {}
    bbox_list = []

    # 집인데 집전체가 없거나 has==0이면 다른 집 요소로 합성 bbox 생성
    if class_kr == "집":
        house_main = ["지붕", "집벽", "문", "창문", "굴뚝", "연기"]
        fc = features.get("집전체") or {}
        if not fc.get("has"):
            cx_sum, cy_sum, ratio_sum, n = 0, 0, 0, 0
            for key in house_main:
                f = features.get(key)
                if f and f.get("has"):
                    cx_sum += f.get("center_x", 0.5)
                    cy_sum += f.get("center_y", 0.5)
                    ratio_sum += f.get("ratio", 0)
                    n += 1
            if n:
                features = dict(features)
                features["집전체"] = {
                    "has": 1,
                    "ratio": min(0.15, ratio_sum * 1.5) or 0.1,
                    "center_x": cx_sum / n,
                    "center_y": cy_sum / n,
                    "confidence": 0.9,
                }
            else:
                features = dict(features)
                features["집전체"] = {
                    "has": 1,
                    "ratio": 0.1,
                    "center_x": 0.5,
                    "center_y": 0.5,
                    "confidence": 0.8,
                }

    for label, feat in features.items():
        if not feat.get("has"):
            continue
        bbox = _feature_to_bbox(feat, img_w, img_h)
        if bbox:
            bbox_list.append({"label": label, **bbox})

    return {
        "meta": {
            "img_resolution": resolution,
            "age": age,
            "sex": sex,
        },
        "annotations": {
            "class": class_kr,
            "bbox": bbox_list,
        },
    }


def is_new_format(data):
    """딕셔너리가 새 형식(features + image_size)이면 True."""
    return (
        isinstance(data, dict)
        and "features" in data
        and "image_size" in data
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python legacy_converter.py <new_format.json>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    legacy = convert_new_to_legacy(data, path)
    print(json.dumps(legacy, ensure_ascii=False, indent=2))

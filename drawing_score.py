"""
아동 그림 T-Score 산출 모듈.
drawing_norm_dist_stats.csv 기준표와 비교하여 에너지, 위치안정성, 표현력 점수를 계산합니다.
"""
from pathlib import Path
from typing import Any

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent
STATS_CSV_PATH = _BASE_DIR / "drawing_norm_dist_stats.csv"

_df_stats: pd.DataFrame | None = None


def _load_stats() -> pd.DataFrame:
    global _df_stats
    if _df_stats is None:
        _df_stats = pd.read_csv(STATS_CSV_PATH, header=[0, 1, 2], index_col=[0, 1, 2])
        _df_stats.index.names = ["Type", "Age", "Sex"]
    return _df_stats


TARGET_MAP = {
    "나무": "나무전체",
    "남자사람": "사람전체",
    "여자사람": "사람전체",
    "집": "집전체",
}


def calculate_drawing_score_from_json(
    data: dict[str, Any],
    draw_type: str,
    age: int,
    sex: str,
) -> dict[str, Any] | None:
    """
    image_json( features 포맷 )에서 T-Score를 계산합니다.

    Args:
        data: image_json 딕셔너리 (features 포함)
        draw_type: '나무' | '남자사람' | '여자사람' | '집'
        age: 연령 (7~13)
        sex: '남' | '여'

    Returns:
        { 에너지_점수, 위치_안정성_점수, 표현력_점수, 종합_평가 } 또는 None
    """
    try:
        features = data.get("features", {})
        if not features:
            return None

        main_key = TARGET_MAP.get(draw_type)
        if main_key not in features:
            for k in features.keys():
                if "전체" in k:
                    main_key = k
                    break

        if not main_key or main_key not in features:
            return None

        obj_data = features[main_key]

        my_size = float(obj_data.get("ratio", 0) or 0) * 100
        my_x = float(obj_data.get("center_x", 0.5) or 0.5)
        my_y = float(obj_data.get("center_y", 0.5) or 0.5)
        my_count = len(features)

        df_stats = _load_stats()
        try:
            norms = df_stats.loc[(draw_type, age, sex)]
        except (KeyError, TypeError):
            return None

        def get_t_score(val: float, col_name: str) -> float:
            cols_mean = [c for c in df_stats.columns if c[0] == col_name and c[1] == "mean"]
            cols_std = [c for c in df_stats.columns if c[0] == col_name and c[1] == "std"]
            if not cols_mean or not cols_std:
                return 50.0
            mu = float(norms[cols_mean[0]])
            sigma = float(norms[cols_std[0]])
            if sigma == 0:
                return 50.0
            z = (val - mu) / sigma
            t = 50 + (z * 10)
            return round(t, 1)

        pos_x_score = get_t_score(my_x, "Pos_X")
        pos_y_score = get_t_score(my_y, "Pos_Y")
        위치_안정성_점수 = round((pos_x_score + pos_y_score) / 2, 1)
        에너지_점수 = get_t_score(my_size, "Size_Ratio")
        표현력_점수 = get_t_score(my_count, "Item_Count")

        종합_평가 = ""
        if 에너지_점수 < 35:
            종합_평가 += "다소 위축됨; "
        elif 에너지_점수 > 65:
            종합_평가 += "에너지 넘침; "
        else:
            종합_평가 += "적절한 에너지; "

        if 표현력_점수 < 35:
            종합_평가 += "표현이 절제됨; "
        elif 표현력_점수 > 65:
            종합_평가 += "풍부한 표현; "
        else:
            종합_평가 += "적절한 표현력; "

        return {
            "에너지_점수": 에너지_점수,
            "위치_안정성_점수": 위치_안정성_점수,
            "표현력_점수": 표현력_점수,
            "종합_평가": 종합_평가.strip(),
        }
    except Exception:
        return None


def _get_peer_norms(age: int, sex: str, draw_types: list[str]) -> dict[str, float] | None:
    """해당 나이/성별 또래 평균(raw mean)을 CSV에서 조회합니다."""
    df_stats = _load_stats()
    vals = {"Size_Ratio": [], "Pos_X": [], "Pos_Y": [], "Item_Count": []}
    for dt in draw_types:
        try:
            norms = df_stats.loc[(dt, age, sex)]
            for col_name, key in [("Size_Ratio", "Size_Ratio"), ("Pos_X", "Pos_X"), ("Pos_Y", "Pos_Y"), ("Item_Count", "Item_Count")]:
                cols = [c for c in df_stats.columns if c[0] == col_name and c[1] == "mean"]
                if cols:
                    vals[key].append(float(norms[cols[0]]))
        except (KeyError, TypeError):
            continue
    if not vals["Size_Ratio"]:
        return None
    return {
        "에너지_또래평균": round(sum(vals["Size_Ratio"]) / len(vals["Size_Ratio"]), 1),
        "위치_X_또래평균": round(sum(vals["Pos_X"]) / len(vals["Pos_X"]), 3),
        "위치_Y_또래평균": round(sum(vals["Pos_Y"]) / len(vals["Pos_Y"]), 3),
        "표현력_또래평균": round(sum(vals["Item_Count"]) / len(vals["Item_Count"]), 1),
    }


def compute_scores_for_analysis(
    results: dict[str, Any],
    age: int,
    sex: str,
) -> dict[str, Any]:
    """
    analyze 응답의 results에서 각 객체별 T-Score를 계산하고 집계합니다.

    Returns:
        {
            "by_object": { "tree": {...}, "house": {...}, ... },
            "aggregated": { 에너지_점수, 위치_안정성_점수, 표현력_점수, 종합_평가 },
            "peer_average": 50,
            "peer_norms": { 에너지_또래평균, ... },
            "age": int, "sex": str
        }
    """
    type_map = {
        "tree": "나무",
        "house": "집",
        "man": "남자사람",
        "woman": "여자사람",
    }

    by_object: dict[str, dict] = {}
    scores_list: list[dict] = []
    draw_types_used: list[str] = []

    for obj_key, label_kr in type_map.items():
        r = results.get(obj_key)
        if not r:
            continue
        img_json = r.get("image_json") or {}
        s = calculate_drawing_score_from_json(img_json, label_kr, age, sex)
        if s:
            by_object[obj_key] = s
            scores_list.append(s)
            draw_types_used.append(label_kr)

    peer_norms = _get_peer_norms(age, sex, draw_types_used) if draw_types_used else None

    if not scores_list:
        return {
            "by_object": by_object,
            "aggregated": None,
            "peer_average": 50,
            "peer_norms": peer_norms,
            "age": age,
            "sex": sex,
        }

    n = len(scores_list)
    agg = {
        "에너지_점수": round(sum(x["에너지_점수"] for x in scores_list) / n, 1),
        "위치_안정성_점수": round(sum(x["위치_안정성_점수"] for x in scores_list) / n, 1),
        "표현력_점수": round(sum(x["표현력_점수"] for x in scores_list) / n, 1),
        "종합_평가": " ".join(x["종합_평가"] for x in scores_list),
    }

    return {
        "by_object": by_object,
        "aggregated": agg,
        "peer_average": 50,
        "peer_norms": peer_norms,
        "age": age,
        "sex": sex,
    }

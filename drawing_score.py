"""
아동 그림 T-Score 산출 모듈.
drawing_norm_dist_stats.csv 기준표와 비교하여 에너지, 위치안정성, 표현력 점수를 계산합니다.

T-Score 근거:
  손성희(2015) 「모바일 기반 HTP그림검사 앱 개발을 위한 표준화 연구」에서 HTP 그림 분석에
  사용된 표준화 점수(T-점수) 체계를 따릅니다. 평균 50, 표준편차 10으로 해석합니다.

구간 해석(35/65):
  T 35 미만 / 65 초과 구간은 T-점수 해석에서 통상 사용되는 ±1.5 표준편차(평균 대비) 기준을
  적용한 참고 구간이며, 심리측정 관례에 따른 것입니다.

참고:
  - "또래 평균 50점": T-Score 정의상 평균=50, 표준편차=10이므로 표시용 또래 평균은 항상 50입니다.
  - 표현력(Item_Count): 기준표에서 사람 그림의 Item_Count 표준편차가 매우 작아(약 0.5~0.8),
    우리가 쓰는 len(features)와 norm 정의가 다르면 극단 T(-29 등)가 나올 수 있어, T는 20~80으로 제한합니다.
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

        # T-Score 해석 가능 범위로 제한. 기준표 Item_Count 표준편차가 매우 작을 때(특히 사람 그림)
        # 우리 len(features)와 norm 정의가 다르면 극단값(-29 등)이 나올 수 있어 상·하한 적용.
        T_MIN, T_MAX = 20.0, 80.0

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
            t = max(T_MIN, min(T_MAX, t))
            return round(t, 1)

        pos_x_score = get_t_score(my_x, "Pos_X")
        pos_y_score = get_t_score(my_y, "Pos_Y")
        위치_안정성_점수 = round((pos_x_score + pos_y_score) / 2, 1)
        에너지_점수 = get_t_score(my_size, "Size_Ratio")
        표현력_점수 = get_t_score(my_count, "Item_Count")

        # 종합_평가 구간: T ±1.5 SD(35/65) 기준. 심리측정 관례 적용.
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


def _get_peer_tscore_from_csv(
    age: int, sex: str, draw_types: list[str]
) -> dict[str, float] | None:
    """기준표의 또래 평균(raw mean)에 해당하는 T-Score를 구합니다. 정의상 μ의 T는 50입니다."""
    if not draw_types:
        return None
    peer_norms = _get_peer_norms(age, sex, draw_types)
    if not peer_norms:
        return None
    # T(μ) = 50 + 10*0 = 50. CSV 또래 평균에 대응하는 T는 항상 50.
    return {
        "에너지_점수": 50.0,
        "위치_안정성_점수": 50.0,
        "표현력_점수": 50.0,
    }


def compute_scores_for_analysis(
    results: dict[str, Any],
    age: int,
    sex: str,
) -> dict[str, Any]:
    """
    업로드한 그림 4장(tree/house/man/woman)의 image_json만 사용하여 T-Score를 계산합니다.
    results는 /analyze 응답의 results이며, 각 객체별 image_json은 해당 업로드 이미지 처리 결과입니다.

    Returns:
        {
            "by_object": { "tree": {...}, "house": {...}, ... },
            "aggregated": { 에너지_점수, 위치_안정성_점수, 표현력_점수, 종합_평가 },
            "peer_average": 또래 평균 T (50),
            "peer_norms": 기준표에서 조회한 또래 raw 평균 { 에너지_또래평균, ... },
            "peer_tscore_from_csv": 또래 T (에너지/위치/표현력 각 50),
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
        if not r or not isinstance(r, dict):
            continue
        img_json = r.get("image_json")
        if not img_json or not isinstance(img_json, dict) or "features" not in img_json:
            continue
        s = calculate_drawing_score_from_json(img_json, label_kr, age, sex)
        if s:
            by_object[obj_key] = s
            scores_list.append(s)
            draw_types_used.append(label_kr)

    peer_norms = _get_peer_norms(age, sex, draw_types_used) if draw_types_used else None
    peer_tscore_from_csv = (
        _get_peer_tscore_from_csv(age, sex, draw_types_used) if draw_types_used else None
    )
    # 또래 평균 T는 기준표의 평균(μ)에 대응하는 T이므로 50. 기준표에서 명시적으로 조회한 값 사용.
    peer_average = 50.0
    if peer_tscore_from_csv:
        peer_average = (
            peer_tscore_from_csv["에너지_점수"]
            + peer_tscore_from_csv["위치_안정성_점수"]
            + peer_tscore_from_csv["표현력_점수"]
        ) / 3

    if not scores_list:
        return {
            "by_object": by_object,
            "aggregated": None,
            "peer_average": peer_average,
            "peer_norms": peer_norms,
            "peer_tscore_from_csv": peer_tscore_from_csv,
            "age": age,
            "sex": sex,
        }

    n = len(scores_list)
    e = round(sum(x["에너지_점수"] for x in scores_list) / n, 1)
    w = round(sum(x["위치_안정성_점수"] for x in scores_list) / n, 1)
    p = round(sum(x["표현력_점수"] for x in scores_list) / n, 1)
    # 집계 T점수 기준으로 종합_평가 한 문장만 생성 (4장 이어붙이면 반복되어 이상하게 나옴)
    종합_평가 = ""
    if e < 35:
        종합_평가 += "다소 위축됨; "
    elif e > 65:
        종합_평가 += "에너지 넘침; "
    else:
        종합_평가 += "적절한 에너지; "
    if w < 35:
        종합_평가 += "위치 불안정; "
    elif w > 65:
        종합_평가 += "위치 안정적; "
    else:
        종합_평가 += "적절한 안정성; "
    if p < 35:
        종합_평가 += "표현이 절제됨; "
    elif p > 65:
        종합_평가 += "풍부한 표현; "
    else:
        종합_평가 += "적절한 표현력; "
    agg = {
        "에너지_점수": e,
        "위치_안정성_점수": w,
        "표현력_점수": p,
        "종합_평가": 종합_평가.strip(),
    }

    return {
        "by_object": by_object,
        "aggregated": agg,
        "peer_average": peer_average,
        "peer_norms": peer_norms,
        "peer_tscore_from_csv": peer_tscore_from_csv,
        "age": age,
        "sex": sex,
    }

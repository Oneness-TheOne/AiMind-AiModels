import pandas as pd
import json
import os
import numpy as np

# 1. 통계 데이터(기준표) 로드
# CSV 파일 경로를 지정하세요.
STATS_CSV_PATH = 'drawing_norm_dist_stats.csv'
df_stats = pd.read_csv(STATS_CSV_PATH, header=[0, 1, 2], index_col=[0, 1, 2])
df_stats.index.names = ['Type', 'Age', 'Sex'] # 인덱스 이름 설정

def calculate_drawing_score(drawing_json_path):
    """개별 그림 파일의 점수를 계산합니다."""
    try:
        with open(drawing_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 파일명에서 메타데이터 파싱 (예: 나무_7_남_00367.json)
        filename = os.path.basename(drawing_json_path)
        parts = filename.split('_')
        draw_type, age, sex = parts[0], int(parts[1]), parts[2]
        
        # '남자사람', '여자사람'의 경우 CSV 인덱스와 매칭 (폴더명 기준)
        # CSV에 '남자사람', '여자사람'으로 저장되어 있다고 가정
        
        # 분석할 객체 찾기 (JSON 구조에 따라 조정)
        features = data.get('features', {})
        
        # 메인 객체 키 매핑
        target_map = {
            '나무': '나무전체',
            '남자사람': '사람전체', # 혹은 '남자사람'
            '여자사람': '사람전체', # 혹은 '여자사람'
            '집': '집전체'
        }
        
        main_key = target_map.get(draw_type)
        # 만약 features에 '사람전체'가 없고 다른 키가 있다면 찾기
        if main_key not in features:
            for k in features.keys():
                if '전체' in k:
                    main_key = k
                    break
        
        if not main_key or main_key not in features:
            return None # 메인 객체 감지 실패
            
        obj_data = features[main_key]
        
        # 내 그림 수치 추출
        # ratio는 0~1 범위로 추정되므로 %로 변환 (*100)
        my_size = obj_data.get('ratio', 0) * 100 
        my_x = obj_data.get('center_x', 0.5)
        my_y = obj_data.get('center_y', 0.5)
        my_count = len(features) # 감지된 요소 개수
        
        # 통계 기준값 가져오기
        try:
            norms = df_stats.loc[(draw_type, age, sex)]
        except KeyError:
            return None # 해당 나이/성별 통계 없음

        # Z-Score 및 T-Score 계산 함수
        def get_t_score(val, col_name):
            # 컬럼 구조: (Metric, mean/std, Unnamed)
            # 안전하게 찾기
            col_mean = [c for c in df_stats.columns if c[0] == col_name and c[1] == 'mean'][0]
            col_std = [c for c in df_stats.columns if c[0] == col_name and c[1] == 'std'][0]
            
            mu = norms[col_mean]
            sigma = norms[col_std]
            
            if sigma == 0: return 50
            
            z = (val - mu) / sigma
            t = 50 + (z * 10)
            return round(t, 1)

        scores = {
            "에너지_점수": get_t_score(my_size, 'Size_Ratio'),
            "위치_안정성_점수": get_t_score(my_x, 'Pos_X'), # X축 편향도 (중앙=50이 아님, 또래 대비 위치)
            "표현력_점수": get_t_score(my_count, 'Item_Count'),
            "종합_평가": ""
        }
        
        # 간단한 텍스트 평가 추가
        if scores['에너지_점수'] < 35: scores['종합_평가'] += "다소 위축됨; "
        elif scores['에너지_점수'] > 65: scores['종합_평가'] += "에너지 넘침; "
        else: scores['종합_평가'] += "적절한 에너지; "
        
        return scores

    except Exception as e:
        print(f"Error calculating score for {drawing_json_path}: {e}")
        return None

# 3. 실행 및 결과 확인
# 예시: 업로드된 파일들에 대해 실행
files = [
    "나무_7_남_00367.json", 
    "남자사람_12_여_20260204_155005.json",
    "여자사람_9_남_20260204_153058.json",
    "집_9_남_20260204_153011.json"
]

results = {}
for f in files:
    score = calculate_drawing_score(f)
    if score:
        results[f] = score
        print(f"[{f}] 점수 산출 완료: {score}")
    else:
        print(f"[{f}] 점수 산출 실패 (객체 미감지 또는 통계 부재)")
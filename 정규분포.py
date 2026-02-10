import os
import json
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm  # 진행상황 표시바 (없으면 pip install tqdm)

# 1. 데이터셋 경로 설정 (사용자 경로 반영)
BASE_DIRS = {
    "나무": r"C:\Honey\Projects\mid-term\AiMind-AiModels\data\266.AI 기반 아동 미술심리 진단을 위한 그림 데이터 구축\01-1.정식개방데이터\Training\02.라벨링데이터\TL_나무",
    "남자사람": r"C:\Honey\Projects\mid-term\AiMind-AiModels\data\266.AI 기반 아동 미술심리 진단을 위한 그림 데이터 구축\01-1.정식개방데이터\Training\02.라벨링데이터\TL_남자사람",
    "여자사람": r"C:\Honey\Projects\mid-term\AiMind-AiModels\data\266.AI 기반 아동 미술심리 진단을 위한 그림 데이터 구축\01-1.정식개방데이터\Training\02.라벨링데이터\TL_여자사람",
    "집": r"C:\Honey\Projects\mid-term\AiMind-AiModels\data\266.AI 기반 아동 미술심리 진단을 위한 그림 데이터 구축\01-1.정식개방데이터\Training\02.라벨링데이터\TL_집"
}

# 분석할 메인 객체 라벨 정의
MAIN_LABELS = ['나무전체', '사람전체', '집전체']

def process_json_file(file_path, draw_type):
    """개별 JSON 파일을 분석하여 수치 데이터를 추출합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        meta = data.get('meta', {})
        annos = data.get('annotations', {}).get('bbox', [])
        
        # 메타 데이터 추출 (나이, 성별)
        age = meta.get('age')
        sex = meta.get('sex')
        
        # 해상도 파싱
        if 'img_resolution' in meta:
            img_w, img_h = map(int, meta['img_resolution'].split('x'))
        else:
            return None # 해상도 없으면 스킵

        canvas_area = img_w * img_h
        
        # 메인 객체(전체) 찾기
        main_obj = next((item for item in annos if item['label'] in MAIN_LABELS), None)
        
        stats = {
            "Type": draw_type,
            "Age": age,
            "Sex": sex,
            "Item_Count": len(annos) # 세부 묘사 수
        }

        if main_obj:
            # 크기 비율 (Size Ratio %)
            obj_area = main_obj['w'] * main_obj['h']
            stats['Size_Ratio'] = (obj_area / canvas_area) * 100
            
            # 위치 (Centroid X, Y - 0~1 정규화)
            center_x = main_obj['x'] + (main_obj['w'] / 2)
            center_y = main_obj['y'] + (main_obj['h'] / 2)
            stats['Pos_X'] = center_x / img_w
            stats['Pos_Y'] = center_y / img_h
        else:
            # 메인 객체(전체) 박스가 없는 경우도 있을 수 있음 (이 경우 통계 제외하거나 NaN 처리)
            stats['Size_Ratio'] = np.nan
            stats['Pos_X'] = np.nan
            stats['Pos_Y'] = np.nan
            
        return stats

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

# 2. 전체 데이터 순회 및 수집
all_data = []

print("데이터 분석을 시작합니다...")

for draw_type, dir_path in BASE_DIRS.items():
    # 해당 폴더 내의 모든 json 파일 검색
    json_files = glob(os.path.join(dir_path, "*.json"))
    print(f"[{draw_type}] 폴더에서 {len(json_files)}개의 파일을 찾았습니다.")
    
    for file_path in tqdm(json_files, desc=f"{draw_type} 처리 중"):
        result = process_json_file(file_path, draw_type)
        if result:
            all_data.append(result)

# 3. 데이터프레임 변환 및 정규분포 통계(평균, 표준편차) 계산
df = pd.DataFrame(all_data)

# 결측치 제거 (메인 박스가 없어서 계산 안된 데이터 제외)
df_clean = df.dropna()

print("\n--- 데이터 통계 집계 중 ---")

# 그룹화: [그림유형, 나이, 성별] 별로 묶어서 평균(mean)과 표준편차(std) 계산
# 관측 대상 컬럼: Size_Ratio, Pos_X, Pos_Y, Item_Count
stat_cols = ['Size_Ratio', 'Pos_X', 'Pos_Y', 'Item_Count']
grouped_stats = df_clean.groupby(['Type', 'Age', 'Sex'])[stat_cols].agg(['mean', 'std', 'count'])

# 4. 결과 저장
output_path = "drawing_norm_dist_stats.csv"
grouped_stats.to_csv(output_path) # 한글 깨짐 방지 위해 utf-8-sig 권장 (엑셀용)
# grouped_stats.to_csv(output_path, encoding='utf-8-sig') 

print(f"\n분석 완료! 결과가 '{output_path}'에 저장되었습니다.")
print("\n[미리보기 - 7세 남자 데이터]")
try:
    # 예시로 나무, 7세, 남자 데이터만 출력
    print(grouped_stats.loc[('나무', 7, '남')])
except KeyError:
    print("7세 남자 나무 데이터가 충분하지 않습니다.")
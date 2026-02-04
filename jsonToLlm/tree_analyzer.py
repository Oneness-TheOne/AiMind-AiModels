import json

def calculate_area_ratio(w, h, img_w, img_h):
    return round((w * h) / (img_w * img_h), 4)

def get_centroid(x, y, w, h):
    return x + w / 2, y + h / 2

def check_inclusion(inner_bbox, outer_bbox):
    # 중심점이 외곽 박스 안에 있는지 확인 (느슨한 포함 관계)
    ix, iy, iw, ih = inner_bbox['x'], inner_bbox['y'], inner_bbox['w'], inner_bbox['h']
    ox, oy, ow, oh = outer_bbox['x'], outer_bbox['y'], outer_bbox['w'], outer_bbox['h']
    
    icx, icy = get_centroid(ix, iy, iw, ih)
    
    return (ox <= icx <= ox + ow) and (oy <= icy <= oy + oh)

def get_position_text(x, y, w, h, context_w, context_h, relative_to="화면"):
    # 중심점 기준 위치 텍스트 생성
    cx, cy = get_centroid(x, y, w, h)
    nx, ny = cx / context_w, cy / context_h # 정규화된 좌표 (0~1)

    # X축 판단
    if nx < 0.33: h_pos = "왼쪽"
    elif nx > 0.66: h_pos = "오른쪽"
    else: h_pos = "중앙"

    # Y축 판단
    if ny < 0.33: v_pos = "상단"
    elif ny > 0.66: v_pos = "하단"
    else: v_pos = "중단"

    if relative_to == "나무":
        return f"나무 {h_pos} {v_pos}"
    elif relative_to == "사람":
        return f"사람 {h_pos} {v_pos}"
    elif relative_to == "집":
        return f"집 {h_pos} {v_pos}"
    else:
        # 화면 전체 기준일 때의 미세 조정 텍스트
        if v_pos == "상단" and h_pos == "중앙": return "화면 상단 중앙"
        return f"화면 {h_pos} {v_pos}"

def process_tree_json(json_input):
    data = json.loads(json_input)
    
    # 1. 메타 정보 추출
    meta = data['meta']
    res_str = meta['img_resolution'].split('x')
    img_w, img_h = int(res_str[0]), int(res_str[1])
    
    # 2. 요소 개수 카운팅 및 객체 분류
    bboxes = data['annotations']['bbox']
    counts = {}
    objects_by_label = {}
    
    tree_total_bbox = None

    for obj in bboxes:
        label = obj['label']
        counts[label] = counts.get(label, 0) + 1
        
        if label not in objects_by_label:
            objects_by_label[label] = []
        objects_by_label[label].append(obj)
        
        if label == "나무전체":
            tree_total_bbox = obj

    # 나무 전체 정보 (기준점)
    if tree_total_bbox:
        tx, ty, tw, th = tree_total_bbox['x'], tree_total_bbox['y'], tree_total_bbox['w'], tree_total_bbox['h']
        tree_cx, tree_cy = get_centroid(tx, ty, tw, th)
        tree_area_ratio = calculate_area_ratio(tw, th, img_w, img_h)
        tree_nx, tree_ny = round(tree_cx/img_w, 4), round(tree_cy/img_h, 4)
        
        # 나무 위치 요약 텍스트 생성
        t_h_desc = "왼쪽" if tree_nx < 0.45 else ("오른쪽" if tree_nx > 0.55 else "중앙")
        t_v_desc = "상단" if tree_ny < 0.4 else ("하단" if tree_ny > 0.6 else "중상단")
        tree_pos_summary = f"화면 {t_h_desc} 편향, 세로 기준 {t_v_desc}"
    else:
        # 나무 전체가 없는 경우 예외 처리
        tree_total_bbox = {'x':0, 'y':0, 'w':img_w, 'h':img_h} # 임시 전체 화면
        tx, ty, tw, th = 0, 0, img_w, img_h
        tree_cx, tree_cy = get_centroid(tx, ty, tw, th)
        tree_area_ratio = 0
        tree_nx, tree_ny = 0.5, 0.5
        tree_pos_summary = "나무 전체 식별 불가"

    # 3. 데이터 구조화 함수
    result = {
        "이미지_메타정보": {
            "해상도": [img_w, img_h],
            "연령": meta['age'],
            "성별": meta['sex']
        },
        "요소_개수": counts,
        "전체_구성": {
            "나무_중심_좌표_정규화": [tree_nx, tree_ny],
            "나무_위치_요약": tree_pos_summary,
            "나무_전체_면적비율": tree_area_ratio
        },
        "나무_구성요소_관계": {},
        "나무_내_부가요소": {},
        "하늘_요소": {}
    }

    # 4. 세부 그룹 처리 로직
    
    # (1) 나무 구성요소 (수관, 기둥, 가지, 뿌리)
    core_parts = ["수관", "기둥", "가지", "뿌리"]
    for part in core_parts:
        if part in objects_by_label:
            obj = objects_by_label[part][0] # 주로 1개라고 가정
            result["나무_구성요소_관계"][part] = {
                "위치": get_position_text(obj['x'], obj['y'], obj['w'], obj['h'], img_w, img_h, relative_to="나무"),
                "면적비율": calculate_area_ratio(obj['w'], obj['h'], img_w, img_h),
                "나무_내부_포함": check_inclusion(obj, tree_total_bbox)
            }

    accessories = ["나뭇잎", "열매", "그네", "새", "다람쥐", "꽃"]
    for acc in accessories:
        if acc in objects_by_label:
            objs = objects_by_label[acc]
            total_area = sum([calculate_area_ratio(o['w'], o['h'], img_w, img_h) for o in objs])
            
            if len(objs) == 1:
                obj = objs[0]
                # [수정됨] 나무 기준 상대 위치 계산을 위해 tree_total_bbox 정보 사용
                # 단, 나무가 없는 경우(tree_total_bbox 가 임시값인 경우) 대비
                if tree_total_bbox['w'] > 0 and tree_total_bbox['h'] > 0:
                    # 객체 중심 좌표
                    obj_cx, obj_cy = get_centroid(obj['x'], obj['y'], obj['w'], obj['h'])
                    # 나무 내 상대 좌표 (0~1)
                    rel_x = (obj_cx - tree_total_bbox['x']) / tree_total_bbox['w']
                    rel_y = (obj_cy - tree_total_bbox['y']) / tree_total_bbox['h']
                    
                    # 위치 텍스트 생성 로직 (0.33, 0.66 기준)
                    if rel_x < 0.33: h_pos = "왼쪽"
                    elif rel_x > 0.66: h_pos = "오른쪽"
                    else: h_pos = "중앙"
                    
                    if rel_y < 0.33: v_pos = "상단"
                    elif rel_y > 0.66: v_pos = "하단"
                    else: v_pos = "중단"
                    
                    pos_text = f"나무 {h_pos} {v_pos}"
                else:
                    pos_text = "위치 판단 불가"

                is_inside = check_inclusion(obj, tree_total_bbox)
                
                result["나무_내_부가요소"][acc] = {
                    "위치": pos_text,
                    "면적비율": round(total_area, 4),
                    "나무_내부_포함": is_inside
                }
            else:
                # 여러 개일 경우 (기존 로직 유지하되, 텍스트가 조금 더 정확해질 수 있음)
                avg_cx = sum([get_centroid(o['x'], o['y'], o['w'], o['h'])[0] for o in objs]) / len(objs)
                # 나무 중심(tree_cx)과 비교하여 좌우 판단
                rel_pos = "오른쪽" if avg_cx > tree_cx + (tw*0.1) else ("왼쪽" if avg_cx < tree_cx - (tw*0.1) else "중앙")
                
                result["나무_내_부가요소"][acc] = {
                    "개수": len(objs),
                    "주요_위치": f"수관 내부, {rel_pos} 영역에 집중", 
                    "면적비율_합": round(total_area, 4)
                }

    # (3) 하늘 요소 (달, 별, 구름)
    sky_elements = ["달", "별", "구름"]
    for sky in sky_elements:
        if sky in objects_by_label:
            objs = objects_by_label[sky]
            total_area = sum([calculate_area_ratio(o['w'], o['h'], img_w, img_h) for o in objs])
            
            if len(objs) == 1:
                pos_text = get_position_text(objs[0]['x'], objs[0]['y'], objs[0]['w'], objs[0]['h'], img_w, img_h, relative_to="화면")
                result["하늘_요소"][sky] = {
                    "위치": pos_text,
                    "면적비율": round(total_area, 4)
                }
            else:
                # 별, 구름 등 복수 객체 위치 요약
                cx_list = [get_centroid(o['x'], o['y'], o['w'], o['h'])[0] for o in objs]
                avg_nx = (sum(cx_list) / len(objs)) / img_w
                
                dist_desc = "띠 형태로 분포" if len(objs) > 2 else "분산 분포"
                bias_desc = "우측 비중 큼" if avg_nx > 0.6 else ("좌측 비중 큼" if avg_nx < 0.4 else "골고루 분포")

                result["하늘_요소"][sky] = {
                    "개수": len(objs),
                    "위치_요약": f"화면 상단, {dist_desc}, {bias_desc}",
                    "면적비율_합": round(total_area, 4)
                }

    return json.dumps(result, ensure_ascii=False, indent=4)

def process_person_json(json_input, person_type="남자사람"):
    """남자사람 또는 여자사람 JSON 처리"""
    data = json.loads(json_input)
    
    # 1. 메타 정보 추출
    meta = data['meta']
    res_str = meta['img_resolution'].split('x')
    img_w, img_h = int(res_str[0]), int(res_str[1])
    
    # 2. 요소 개수 카운팅 및 객체 분류
    bboxes = data['annotations']['bbox']
    counts = {}
    objects_by_label = {}
    
    person_total_bbox = None

    for obj in bboxes:
        label = obj['label']
        counts[label] = counts.get(label, 0) + 1
        
        if label not in objects_by_label:
            objects_by_label[label] = []
        objects_by_label[label].append(obj)
        
        if label == "사람전체":
            person_total_bbox = obj

    # 사람 전체 정보 (기준점)
    if person_total_bbox:
        px, py, pw, ph = person_total_bbox['x'], person_total_bbox['y'], person_total_bbox['w'], person_total_bbox['h']
        person_cx, person_cy = get_centroid(px, py, pw, ph)
        person_area_ratio = calculate_area_ratio(pw, ph, img_w, img_h)
        person_nx, person_ny = round(person_cx/img_w, 4), round(person_cy/img_h, 4)
        
        # 사람 위치 요약 텍스트 생성
        p_h_desc = "왼쪽" if person_nx < 0.45 else ("오른쪽" if person_nx > 0.55 else "중앙")
        p_v_desc = "상단" if person_ny < 0.4 else ("하단" if person_ny > 0.6 else "중상단")
        person_pos_summary = f"화면 {p_h_desc} 편향, 세로 기준 {p_v_desc}"
    else:
        person_total_bbox = {'x':0, 'y':0, 'w':img_w, 'h':img_h}
        px, py, pw, ph = 0, 0, img_w, img_h
        person_cx, person_cy = get_centroid(px, py, pw, ph)
        person_area_ratio = 0
        person_nx, person_ny = 0.5, 0.5
        person_pos_summary = "사람 전체 식별 불가"

    # 3. 데이터 구조화
    result = {
        "이미지_메타정보": {
            "해상도": [img_w, img_h],
            "연령": meta['age'],
            "성별": meta['sex']
        },
        "요소_개수": counts,
        "전체_구성": {
            "사람_중심_좌표_정규화": [person_nx, person_ny],
            "사람_위치_요약": person_pos_summary,
            "사람_전체_면적비율": person_area_ratio
        },
        "얼굴_구성요소": {},
        "신체_구성요소": {},
        "의류_및_액세서리": {}
    }

    # 4. 세부 그룹 처리 로직
    
    # (1) 얼굴 구성요소
    face_parts = ["머리", "얼굴", "눈", "코", "입", "귀", "머리카락"]
    for part in face_parts:
        if part in objects_by_label:
            objs = objects_by_label[part]
            total_area = sum([calculate_area_ratio(o['w'], o['h'], img_w, img_h) for o in objs])
            
            if len(objs) == 1:
                obj = objs[0]
                if person_total_bbox['w'] > 0 and person_total_bbox['h'] > 0:
                    obj_cx, obj_cy = get_centroid(obj['x'], obj['y'], obj['w'], obj['h'])
                    rel_x = (obj_cx - person_total_bbox['x']) / person_total_bbox['w']
                    rel_y = (obj_cy - person_total_bbox['y']) / person_total_bbox['h']
                    
                    if rel_x < 0.33: h_pos = "왼쪽"
                    elif rel_x > 0.66: h_pos = "오른쪽"
                    else: h_pos = "중앙"
                    
                    if rel_y < 0.33: v_pos = "상단"
                    elif rel_y > 0.66: v_pos = "하단"
                    else: v_pos = "중단"
                    
                    pos_text = f"사람 {h_pos} {v_pos}"
                else:
                    pos_text = "위치 판단 불가"
                
                is_inside = check_inclusion(obj, person_total_bbox)
                
                result["얼굴_구성요소"][part] = {
                    "위치": pos_text,
                    "면적비율": round(total_area, 4),
                    "사람_내부_포함": is_inside
                }
            else:
                avg_cx = sum([get_centroid(o['x'], o['y'], o['w'], o['h'])[0] for o in objs]) / len(objs)
                rel_pos = "오른쪽" if avg_cx > person_cx + (pw*0.1) else ("왼쪽" if avg_cx < person_cx - (pw*0.1) else "중앙")
                
                result["얼굴_구성요소"][part] = {
                    "개수": len(objs),
                    "주요_위치": f"얼굴 영역, {rel_pos} 편향",
                    "면적비율_합": round(total_area, 4)
                }

    # (2) 신체 구성요소
    body_parts = ["목", "상체", "팔", "손", "다리", "발"]
    for part in body_parts:
        if part in objects_by_label:
            objs = objects_by_label[part]
            total_area = sum([calculate_area_ratio(o['w'], o['h'], img_w, img_h) for o in objs])
            
            if len(objs) == 1:
                obj = objs[0]
                pos_text = get_position_text(obj['x'], obj['y'], obj['w'], obj['h'], img_w, img_h, relative_to="사람")
                is_inside = check_inclusion(obj, person_total_bbox)
                
                result["신체_구성요소"][part] = {
                    "위치": pos_text,
                    "면적비율": round(total_area, 4),
                    "사람_내부_포함": is_inside
                }
            else:
                avg_cx = sum([get_centroid(o['x'], o['y'], o['w'], o['h'])[0] for o in objs]) / len(objs)
                rel_pos = "오른쪽" if avg_cx > person_cx + (pw*0.1) else ("왼쪽" if avg_cx < person_cx - (pw*0.1) else "중앙")
                
                result["신체_구성요소"][part] = {
                    "개수": len(objs),
                    "주요_위치": f"신체 {rel_pos} 편향",
                    "면적비율_합": round(total_area, 4)
                }

    # (3) 의류 및 액세서리
    accessories = ["단추", "주머니", "운동화"]
    if person_type == "남자사람":
        accessories.append("남자구두")
    else:
        accessories.append("여자구두")
    
    for acc in accessories:
        if acc in objects_by_label:
            objs = objects_by_label[acc]
            total_area = sum([calculate_area_ratio(o['w'], o['h'], img_w, img_h) for o in objs])
            
            if len(objs) == 1:
                obj = objs[0]
                pos_text = get_position_text(obj['x'], obj['y'], obj['w'], obj['h'], img_w, img_h, relative_to="사람")
                is_inside = check_inclusion(obj, person_total_bbox)
                
                result["의류_및_액세서리"][acc] = {
                    "위치": pos_text,
                    "면적비율": round(total_area, 4),
                    "사람_내부_포함": is_inside
                }
            else:
                avg_cx = sum([get_centroid(o['x'], o['y'], o['w'], o['h'])[0] for o in objs]) / len(objs)
                rel_pos = "오른쪽" if avg_cx > person_cx + (pw*0.1) else ("왼쪽" if avg_cx < person_cx - (pw*0.1) else "중앙")
                
                result["의류_및_액세서리"][acc] = {
                    "개수": len(objs),
                    "주요_위치": f"신체 {rel_pos} 편향",
                    "면적비율_합": round(total_area, 4)
                }

    return json.dumps(result, ensure_ascii=False, indent=4)

def process_house_json(json_input):
    """집 JSON 처리"""
    data = json.loads(json_input)
    
    # 1. 메타 정보 추출
    meta = data['meta']
    res_str = meta['img_resolution'].split('x')
    img_w, img_h = int(res_str[0]), int(res_str[1])
    
    # 2. 요소 개수 카운팅 및 객체 분류
    bboxes = data['annotations']['bbox']
    counts = {}
    objects_by_label = {}
    
    house_total_bbox = None

    for obj in bboxes:
        label = obj['label']
        counts[label] = counts.get(label, 0) + 1
        
        if label not in objects_by_label:
            objects_by_label[label] = []
        objects_by_label[label].append(obj)
        
        if label == "집전체":
            house_total_bbox = obj

    # 집 전체 정보 (기준점)
    if house_total_bbox:
        hx, hy, hw, hh = house_total_bbox['x'], house_total_bbox['y'], house_total_bbox['w'], house_total_bbox['h']
        house_cx, house_cy = get_centroid(hx, hy, hw, hh)
        house_area_ratio = calculate_area_ratio(hw, hh, img_w, img_h)
        house_nx, house_ny = round(house_cx/img_w, 4), round(house_cy/img_h, 4)
        
        # 집 위치 요약 텍스트 생성
        h_h_desc = "왼쪽" if house_nx < 0.45 else ("오른쪽" if house_nx > 0.55 else "중앙")
        h_v_desc = "상단" if house_ny < 0.4 else ("하단" if house_ny > 0.6 else "중상단")
        house_pos_summary = f"화면 {h_h_desc} 편향, 세로 기준 {h_v_desc}"
    else:
        house_total_bbox = {'x':0, 'y':0, 'w':img_w, 'h':img_h}
        hx, hy, hw, hh = 0, 0, img_w, img_h
        house_cx, house_cy = get_centroid(hx, hy, hw, hh)
        house_area_ratio = 0
        house_nx, house_ny = 0.5, 0.5
        house_pos_summary = "집 전체 식별 불가"

    # 3. 데이터 구조화
    result = {
        "이미지_메타정보": {
            "해상도": [img_w, img_h],
            "연령": meta['age'],
            "성별": meta['sex']
        },
        "요소_개수": counts,
        "전체_구성": {
            "집_중심_좌표_정규화": [house_nx, house_ny],
            "집_위치_요약": house_pos_summary,
            "집_전체_면적비율": house_area_ratio
        },
        "집_구성요소": {},
        "집_주변_요소": {},
        "하늘_및_자연_요소": {}
    }

    # 4. 세부 그룹 처리 로직
    
    # (1) 집 구성요소
    house_parts = ["지붕", "집벽", "문", "창문", "굴뚝", "연기"]
    for part in house_parts:
        if part in objects_by_label:
            objs = objects_by_label[part]
            total_area = sum([calculate_area_ratio(o['w'], o['h'], img_w, img_h) for o in objs])
            
            if len(objs) == 1:
                obj = objs[0]
                if house_total_bbox['w'] > 0 and house_total_bbox['h'] > 0:
                    obj_cx, obj_cy = get_centroid(obj['x'], obj['y'], obj['w'], obj['h'])
                    rel_x = (obj_cx - house_total_bbox['x']) / house_total_bbox['w']
                    rel_y = (obj_cy - house_total_bbox['y']) / house_total_bbox['h']
                    
                    if rel_x < 0.33: h_pos = "왼쪽"
                    elif rel_x > 0.66: h_pos = "오른쪽"
                    else: h_pos = "중앙"
                    
                    if rel_y < 0.33: v_pos = "상단"
                    elif rel_y > 0.66: v_pos = "하단"
                    else: v_pos = "중단"
                    
                    pos_text = f"집 {h_pos} {v_pos}"
                else:
                    pos_text = "위치 판단 불가"
                
                is_inside = check_inclusion(obj, house_total_bbox)
                
                result["집_구성요소"][part] = {
                    "위치": pos_text,
                    "면적비율": round(total_area, 4),
                    "집_내부_포함": is_inside
                }
            else:
                avg_cx = sum([get_centroid(o['x'], o['y'], o['w'], o['h'])[0] for o in objs]) / len(objs)
                rel_pos = "오른쪽" if avg_cx > house_cx + (hw*0.1) else ("왼쪽" if avg_cx < house_cx - (hw*0.1) else "중앙")
                
                result["집_구성요소"][part] = {
                    "개수": len(objs),
                    "주요_위치": f"집 {rel_pos} 편향",
                    "면적비율_합": round(total_area, 4)
                }

    # (2) 집 주변 요소
    surrounding = ["울타리", "길", "연못"]
    for elem in surrounding:
        if elem in objects_by_label:
            objs = objects_by_label[elem]
            total_area = sum([calculate_area_ratio(o['w'], o['h'], img_w, img_h) for o in objs])
            
            if len(objs) == 1:
                obj = objs[0]
                pos_text = get_position_text(obj['x'], obj['y'], obj['w'], obj['h'], img_w, img_h, relative_to="화면")
                result["집_주변_요소"][elem] = {
                    "위치": pos_text,
                    "면적비율": round(total_area, 4)
                }
            else:
                avg_cx = sum([get_centroid(o['x'], o['y'], o['w'], o['h'])[0] for o in objs]) / len(objs)
                avg_nx = avg_cx / img_w
                rel_pos = "오른쪽" if avg_nx > 0.6 else ("왼쪽" if avg_nx < 0.4 else "중앙")
                
                result["집_주변_요소"][elem] = {
                    "개수": len(objs),
                    "주요_위치": f"화면 {rel_pos} 편향",
                    "면적비율_합": round(total_area, 4)
                }

    # (3) 하늘 및 자연 요소
    nature_elements = ["산", "나무", "꽃", "잔디", "태양"]
    for elem in nature_elements:
        if elem in objects_by_label:
            objs = objects_by_label[elem]
            total_area = sum([calculate_area_ratio(o['w'], o['h'], img_w, img_h) for o in objs])
            
            if len(objs) == 1:
                pos_text = get_position_text(objs[0]['x'], objs[0]['y'], objs[0]['w'], objs[0]['h'], img_w, img_h, relative_to="화면")
                result["하늘_및_자연_요소"][elem] = {
                    "위치": pos_text,
                    "면적비율": round(total_area, 4)
                }
            else:
                cx_list = [get_centroid(o['x'], o['y'], o['w'], o['h'])[0] for o in objs]
                avg_nx = (sum(cx_list) / len(objs)) / img_w
                
                dist_desc = "띠 형태로 분포" if len(objs) > 2 else "분산 분포"
                bias_desc = "우측 비중 큼" if avg_nx > 0.6 else ("좌측 비중 큼" if avg_nx < 0.4 else "골고루 분포")
                
                result["하늘_및_자연_요소"][elem] = {
                    "개수": len(objs),
                    "위치_요약": f"화면 {bias_desc}, {dist_desc}",
                    "면적비율_합": round(total_area, 4)
                }

    return json.dumps(result, ensure_ascii=False, indent=4)

def process_json(json_input):
    """JSON 타입을 자동 감지하여 적절한 처리 함수 호출"""
    data = json.loads(json_input)
    class_type = data['annotations']['class']
    
    if class_type == "나무":
        return process_tree_json(json_input)
    elif class_type == "남자사람":
        return process_person_json(json_input, "남자사람")
    elif class_type == "여자사람":
        return process_person_json(json_input, "여자사람")
    elif class_type == "집":
        return process_house_json(json_input)
    else:
        raise ValueError(f"지원하지 않는 클래스 타입: {class_type}")

# --- 실행 예시 ---
if __name__ == "__main__":
    # 입력으로 주어진 JSON 문자열 (실제 사용 시 파일에서 읽어오거나 API 응답을 사용)
    json_str = """
    {
        "meta":{
            "img_id":"나무_7_남_A066_jtg_20220902_14071525_0830",
            "contributor":"A066",
            "date_created":"2022-09-21",
            "img_path":"../../원천데이터/나무/나무_7_남_00242.jpg",
            "label_path":"./나무_7_남_00242.json",
            "img_size":65227,
            "img_resolution":"1280x1280",
            "age":7,
            "sex":"남"
        },
        "annotations":{
            "anno_id":"m6nn3y9lv6poyxhnhao7",
            "class":"나무",
            "bbox_count":25,
            "bbox":[
                {"label":"나무전체","x":274,"y":226,"w":504,"h":675},
                {"label":"기둥","x":471,"y":498,"w":107,"h":333},
                {"label":"수관","x":279,"y":225,"w":484,"h":338},
                {"label":"가지","x":542,"y":579,"w":208,"h":115},
                {"label":"뿌리","x":399,"y":822,"w":303,"h":78},
                {"label":"나뭇잎","x":746,"y":420,"w":27,"h":40},
                {"label":"나뭇잎","x":735,"y":500,"w":41,"h":75},
                {"label":"나뭇잎","x":551,"y":528,"w":34,"h":48},
                {"label":"나뭇잎","x":594,"y":579,"w":27,"h":45},
                {"label":"나뭇잎","x":667,"y":650,"w":44,"h":53},
                {"label":"꽃","x":764,"y":766,"w":100,"h":127},
                {"label":"열매","x":279,"y":395,"w":18,"h":43},
                {"label":"열매","x":302,"y":482,"w":25,"h":46},
                {"label":"열매","x":376,"y":533,"w":20,"h":46},
                {"label":"그네","x":601,"y":650,"w":117,"h":151},
                {"label":"새","x":656,"y":551,"w":95,"h":80},
                {"label":"다람쥐","x":425,"y":630,"w":92,"h":67},
                {"label":"구름","x":741,"y":168,"w":507,"h":266},
                {"label":"구름","x":186,"y":157,"w":561,"h":164},
                {"label":"달","x":49,"y":21,"w":98,"h":219},
                {"label":"별","x":190,"y":20,"w":195,"h":162},
                {"label":"별","x":397,"y":18,"w":176,"h":123},
                {"label":"별","x":577,"y":16,"w":206,"h":141},
                {"label":"별","x":761,"y":2,"w":279,"h":161},
                {"label":"별","x":1029,"y":0,"w":213,"h":167}
            ]
        },
        "shape_description": {}
    }
    """
    
    # 결과 출력
    processed_json = process_json(json_str)
    print(processed_json)
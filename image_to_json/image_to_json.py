"""
이미지 경로를 주면, 학습된 HTP 모델(나무/집/여자/남자)로 감지·분류한 결과를 JSON으로 출력.
- --format raw: 원시 bbox/segment (개발·검증용)
- --format rag: RAG용 세분화 형식 (객체별 한글명, 비율, 위치, 존재여부, 요약 문장)

사용:
  python src/image_to_json.py --image "경로/이미지.jpg" --object tree
  python src/image_to_json.py --image "경로/이미지.jpg" --object tree --format rag --output result.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CONFIG_MAP = {
    "tree": "tree_mvp_14cls",
    "house": "house_15cls",
    "woman": "woman_18cls",
    "man": "man_18cls",
}
CONF_DEFAULT = 0.15
IOU_DEFAULT = 0.45
IMGSZ = 768

# 영문 클래스명 -> 한글 (RAG 해석용). 객체별로 동일 키 사용.
CLASS_EN_TO_KR = {
    "tree": {
        "tree": "나무전체", "trunk": "기둥", "crown": "수관", "branch": "가지", "root": "뿌리",
        "leaf": "나뭇잎", "flower": "꽃", "fruit": "열매", "swing": "그네", "bird": "새",
        "squirrel": "다람쥐", "cloud": "구름", "moon": "달", "star": "별",
    },
    "house": {
        "house": "집전체", "roof": "지붕", "wall": "집벽", "door": "문", "window": "창문",
        "chimney": "굴뚝", "smoke": "연기", "fence": "울타리", "path": "길", "pond": "연못",
        "mountain": "산", "tree": "나무", "flower": "꽃", "grass": "잔디", "sun": "태양",
    },
    "woman": {
        "person": "사람전체", "head": "머리", "face": "얼굴", "eye": "눈", "nose": "코", "mouth": "입",
        "ear": "귀", "hair": "머리카락", "neck": "목", "torso": "상체", "arm": "팔", "hand": "손",
        "leg": "다리", "foot": "발", "button": "단추", "pocket": "주머니", "sneaker": "운동화", "woman_shoe": "여자구두",
    },
    "man": {
        "person": "사람전체", "head": "머리", "face": "얼굴", "eye": "눈", "nose": "코", "mouth": "입",
        "ear": "귀", "hair": "머리카락", "neck": "목", "torso": "상체", "arm": "팔", "hand": "손",
        "leg": "다리", "foot": "발", "button": "단추", "pocket": "주머니", "sneaker": "운동화", "man_shoe": "남자구두",
    },
}
OBJECT_KR = {"tree": "나무", "house": "집", "woman": "여자사람", "man": "남자사람"}

# --------------- 여기서 경로 설정 (아래만 채우고 실행하면 결과가 JSON으로 나옴) ---------------
IMAGE_PATH = r"C:\Honey\Projects\mid-term\AiMind-AiModels\image_to_json\남자사람_7_남_00978.jpg"   # 입력 이미지 경로. 예: "runs/predict_tree_mvp/7_여_TS_나무_나무_7_여_06136.jpg"
OBJECT = "man"   # 모델: "tree" | "house" | "woman" | "man"
OUTPUT_JSON = "test_man.json"  # 결과 JSON 파일 경로. 예: "result.json" (비우면 터미널에 출력)
OUTPUT_IMAGE = "test_man_box.jpg"  # 박스 표시 이미지 저장 경로. 비우면 저장 안 함
OUTPUT_DIR = r"C:\Honey\Projects\mid-term\AiMind-AiModels\image_to_json\result"  # 배치 출력 폴더 (디렉터리 입력시)
# -------------------------------------------------------------------------------------------

def get_weights_path(object_type: str, gender: str = "male") -> Path:
    """성별에 따라 가중치 경로를 반환. gender: 'male' | 'female'"""
    weights_dir = Path(__file__).parent / f"{object_type}_weights" / gender / "best.pt"
    return weights_dir

FILENAME_PREFIX_TO_OBJECT = {
    "나무": "tree",
    "남자사람": "man",
    "여자사람": "woman",
    "집": "house",
}


def find_weights(run_name: str) -> Path:
    for base in (ROOT / "runs" / "segment", ROOT / "runs" / "segment" / "runs" / "segment"):
        w = base / run_name / "weights" / "best.pt"
        if w.exists():
            return w
    return ROOT / "yolov8n-seg.pt"


def _compute_rag_features(r, h, w, names, en_to_kr: dict) -> tuple[dict, dict, list]:
    """마스크/박스에서 클래스별 비율·중심·존재·신뢰도 계산. (features_by_class, list_detected_kr, class_order)"""
    total_pixels = h * w
    class_pixels = {}
    class_centroid = {}
    class_conf = {}

    if r.boxes is None or len(r.boxes) == 0:
        return {}, {}, []

    boxes = r.boxes
    clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls.numpy()
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()
    m_data = None
    if r.masks is not None and r.masks.data is not None:
        m_data = r.masks.data.cpu().numpy() if hasattr(r.masks.data, "cpu") else np.asarray(r.masks.data)

    for i in range(len(clss)):
        cid = int(clss[i])
        en = names.get(cid, f"class_{cid}")
        kr = en_to_kr.get(en, en)
        conf = float(confs[i])
        if en not in class_conf:
            class_conf[en] = conf
        else:
            class_conf[en] = max(class_conf[en], conf)

        area = 0.0
        cx_sum, cy_sum, cnt = 0.0, 0.0, 0
        if m_data is not None and i < m_data.shape[0]:
            mask = m_data[i]
            if mask.ndim == 3:
                mask = mask.squeeze()
            area = float(np.sum(mask > 0.5))
            ys, xs = np.where(mask > 0.5)
            if len(ys) > 0:
                cx_sum = float(np.sum(xs))
                cy_sum = float(np.sum(ys))
                cnt = len(ys)
        else:
            xyxy = boxes.xyxy[i]
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            area = (x2 - x1) * (y2 - y1)
            cx_sum = (x1 + x2) / 2 * 1
            cy_sum = (y1 + y2) / 2 * 1
            cnt = 1

        if en not in class_pixels:
            class_pixels[en] = 0.0
            class_centroid[en] = [0.0, 0.0, 0]
        class_pixels[en] += area
        class_centroid[en][0] += cx_sum
        class_centroid[en][1] += cy_sum
        class_centroid[en][2] += cnt

    features = {}
    detected_kr = []
    for en, pixels in class_pixels.items():
        kr = en_to_kr.get(en, en)
        ratio = round(pixels / total_pixels, 6) if total_pixels > 0 else 0.0
        cx, cy = -1.0, -1.0
        if class_centroid[en][2] > 0:
            cx = round(class_centroid[en][0] / class_centroid[en][2] / w, 6)
            cy = round(class_centroid[en][1] / class_centroid[en][2] / h, 6)
        features[kr] = {
            "has": 1 if pixels > 0 else 0,
            "ratio": ratio,
            "center_x": cx,
            "center_y": cy,
            "confidence": round(class_conf.get(en, 0), 4),
        }
        if pixels > 0:
            detected_kr.append(kr)

    class_order = list(en_to_kr.values())
    return features, detected_kr, class_order


def _make_summary_text(object_type: str, features: dict, detected_kr: list) -> str:
    """RAG가 읽기 쉬운 한글 요약 문장 생성."""
    obj_kr = OBJECT_KR.get(object_type, object_type)
    if not detected_kr:
        return f"{obj_kr} 이미지에서 감지된 객체가 없습니다."
    parts = []
    parts.append(f"{obj_kr} 이미지에서 " + ", ".join(detected_kr) + "가 감지되었습니다.")
    main_class = list(features.keys())[0] if features else None
    if main_class and features[main_class].get("ratio", 0) > 0:
        r = features[main_class]["ratio"]
        parts.append(f"{main_class}가 이미지의 약 {round(r * 100, 1)}%를 차지합니다.")
    for kr in ["기둥", "수관", "달", "별", "구름"]:
        if kr in features and features[kr].get("has") == 1:
            cx = features[kr].get("center_x", -1)
            if cx >= 0:
                pos = "왼쪽" if cx < 0.4 else ("오른쪽" if cx > 0.6 else "가운데")
                parts.append(f"{kr}는 {pos}에 위치합니다.")
    return " ".join(parts)


def build_rag_output(r, path: Path, object_type: str, weights: Path, w: int, h: int) -> dict:
    """추론 결과를 RAG용 세분화 JSON으로 변환."""
    names = r.names or {}
    en_to_kr = dict(CLASS_EN_TO_KR.get(object_type, {}))
    for cid, en in (names.items() if isinstance(names, dict) else []):
        if en not in en_to_kr:
            en_to_kr[en] = en
    features, detected_kr, class_order = _compute_rag_features(r, h, w, names, en_to_kr)
    for kr in class_order:
        if kr not in features:
            features[kr] = {"has": 0, "ratio": 0.0, "center_x": -1, "center_y": -1, "confidence": 0}
    summary = _make_summary_text(object_type, features, detected_kr)

    return {
        "image_path": str(path),
        "image_size": {"width": w, "height": h},
        "object_type": object_type,
        "object_type_kr": OBJECT_KR.get(object_type, object_type),
        "model_weights": str(weights),
        "classes_kr": class_order,
        "features": features,
        "detected_classes_kr": detected_kr,
        "detection_count": len(detected_kr),
        "summary": summary,
    }


def _save_annotated_image(r, output_image: Path, object_type: str = "tree") -> None:
    """커스텀 annotate: 박스 + 단색 마스크 + 한글 클래스명 (확률 제외)"""
    output_image.parent.mkdir(parents=True, exist_ok=True)

    # 원본 이미지 가져오기
    img = r.orig_img.copy()
    names = r.names or {}
    en_to_kr = CLASS_EN_TO_KR.get(object_type, {})

    # 색상 팔레트 (BGR)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0)
    ]

    if r.boxes is None or len(r.boxes) == 0:
        cv2.imwrite(str(output_image), img)
        return

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.numpy()
    clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls.numpy()

    # 마스크가 있으면 단색으로 채우기
    if r.masks is not None and r.masks.data is not None:
        mask_data = r.masks.data.cpu().numpy() if hasattr(r.masks.data, "cpu") else np.asarray(r.masks.data)

        for i in range(len(clss)):
            if i >= len(mask_data):
                continue

            cid = int(clss[i])
            color = colors[cid % len(colors)]

            # 마스크를 이미지에 단색으로 오버레이
            mask = mask_data[i]
            if mask.ndim == 3:
                mask = mask.squeeze()

            # 마스크 리사이즈 (원본 이미지 크기에 맞춤)
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

            # 마스크 영역을 단색으로 오버레이 (투명도 0.4)
            mask_bool = mask > 0.5
            overlay = img.copy()
            overlay[mask_bool] = color
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # 박스 그리기 + 한글 클래스명 표시
    for i in range(len(clss)):
        cid = int(clss[i])
        en_name = names.get(cid, f"class_{cid}")
        kr_name = en_to_kr.get(en_name, en_name)

        x1, y1, x2, y2 = xyxy[i].astype(int)
        color = colors[cid % len(colors)]

        # 박스 그리기 (두께 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 한글 라벨 배경
        label = kr_name
        font_scale = 0.6
        thickness = 2
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # 라벨 배경 사각형
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)

        # 한글 텍스트 (PIL 사용)
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 기본 폰트 사용 (한글 지원)
        try:
            # Windows: 맑은 고딕
            font = ImageFont.truetype("malgun.ttf", 16)
        except:
            try:
                # Mac/Linux: 나눔고딕
                font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 16)
            except:
                # 기본 폰트
                font = ImageFont.load_default()

        draw.text((x1 + 5, y1 - h - 5), label, font=font, fill=(255, 255, 255))

        # PIL -> OpenCV
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_image), img)


def run(
    image_path: str,
    object_type: str,
    conf: float = CONF_DEFAULT,
    iou: float = IOU_DEFAULT,
    output_format: str = "raw",
    output_image_path: str | None = None,
    gender: str = "male",
) -> dict:
    """이미지 1장 추론 후 JSON용 dict 반환. output_format: raw | rag. gender: 'male' | 'female'"""
    run_name = CONFIG_MAP[object_type]
    weights = get_weights_path(object_type, gender)
    if not weights.exists():
        # 성별 폴더가 없으면 기존 방식으로 폴백
        weights = find_weights(run_name)
    if not weights.exists():
        raise FileNotFoundError(f"모델 없음: {weights}. 가중치 파일을 {object_type}_weights/{gender}/best.pt 경로에 넣어주세요.")

    path = Path(image_path)
    if not path.is_absolute():
        path = (ROOT / image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"이미지 없음: {path}")

    model = YOLO(str(weights))
    results = model.predict(
        source=str(path),
        imgsz=IMGSZ,
        conf=conf,
        iou=iou,
        save=False,
        verbose=False,
    )
    if not results:
        if output_format == "rag":
            en_to_kr = CLASS_EN_TO_KR.get(object_type, {})
            return {
                "image_path": str(path),
                "image_size": {"width": 0, "height": 0},
                "object_type": object_type,
                "object_type_kr": OBJECT_KR.get(object_type, object_type),
                "model_weights": str(weights),
                "classes_kr": list(en_to_kr.values()),
                "features": {kr: {"has": 0, "ratio": 0.0, "center_x": -1, "center_y": -1, "confidence": 0} for kr in en_to_kr.values()},
                "detected_classes_kr": [],
                "detection_count": 0,
                "summary": f"{OBJECT_KR.get(object_type, object_type)} 이미지에서 감지된 객체가 없습니다.",
            }
        return {
            "image_path": str(path),
            "image_size": {"width": 0, "height": 0},
            "object_type": object_type,
            "model_weights": str(weights),
            "detections": [],
        }

    r = results[0]
    h, w = int(r.orig_shape[0]), int(r.orig_shape[1])

    if output_image_path:
        out_img = Path(output_image_path)
        if not out_img.is_absolute():
            out_img = (ROOT / out_img).resolve()
        _save_annotated_image(r, out_img, object_type)

    if output_format == "rag":
        return build_rag_output(r, path, object_type, weights, w, h)

    names = r.names or {}
    out = {
        "image_path": str(path),
        "image_size": {"width": w, "height": h},
        "object_type": object_type,
        "model_weights": str(weights),
        "detections": [],
    }

    if r.boxes is None or len(r.boxes) == 0:
        return out

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.numpy()
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()
    clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls.numpy()

    for i in range(len(clss)):
        cid = int(clss[i])
        name = names.get(cid, f"class_{cid}")
        x1, y1, x2, y2 = xyxy[i].tolist()
        det = {
            "class_id": cid,
            "class_name": name,
            "confidence": round(float(confs[i]), 6),
            "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "bbox_normalized": [
                round(x1 / w, 6),
                round(y1 / h, 6),
                round(x2 / w, 6),
                round(y2 / h, 6),
            ],
        }
        try:
            if r.masks is not None:
                xy_list = getattr(r.masks, "xy", None) or getattr(r.masks, "xyn", None)
                if xy_list is not None and i < len(xy_list):
                    mask = xy_list[i]
                    if mask is not None and len(mask) > 0:
                        poly = mask.tolist() if hasattr(mask, "tolist") else list(mask)
                        det["segment_polygon_xy"] = [[round(float(a), 2), round(float(b), 2)] for a, b in poly]
                        det["segment_polygon_normalized"] = [
                            [round(float(a) / w, 6), round(float(b) / h, 6)] for a, b in poly
                        ]
        except Exception:
            pass
        out["detections"].append(det)

    return out


def _detect_object_type_from_filename(filename: str) -> str | None:
    for prefix, obj in FILENAME_PREFIX_TO_OBJECT.items():
        if filename.startswith(prefix):
            return obj
    return None


def run_batch_dir(input_dir: Path, output_dir: Path, conf: float, iou: float, output_format: str, gender: str = "male") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not image_files:
        print(f"입력 폴더에 이미지가 없습니다: {input_dir}", file=sys.stderr)
        return

    for img_path in image_files:
        obj = _detect_object_type_from_filename(img_path.name)
        if obj is None:
            print(f"[SKIP] 타입 추정 실패: {img_path.name}", file=sys.stderr)
            continue
        out_json = output_dir / f"{img_path.stem}.json"
        out_img = output_dir / f"{img_path.stem}_box.jpg"
        try:
            result = run(
                str(img_path),
                obj,
                conf=conf,
                iou=iou,
                output_format=output_format,
                output_image_path=str(out_img),
                gender=gender,
            )
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            continue
        s = json.dumps(result, ensure_ascii=False, indent=2)
        out_json.write_text(s, encoding="utf-8")
        n = result.get("detection_count", len(result.get("detections", [])))
        print(f"[DONE] {img_path.name} ({obj}) -> {n} detections")


def main():
    parser = argparse.ArgumentParser(description="이미지 1장 → 학습 모델 감지·분류 → JSON (경로는 파일 상단 IMAGE_PATH 등으로도 설정 가능)")
    parser.add_argument("--image", "-i", default=None, help="입력 이미지 경로 (없으면 위 IMAGE_PATH 사용)")
    parser.add_argument("--object", "-o", choices=["tree", "house", "woman", "man"], default=None, help="모델 (없으면 위 OBJECT 사용)")
    parser.add_argument("--gender", "-g", choices=["male", "female"], default="male", help="성별: male(남아) 또는 female(여아) [기본: male]")
    parser.add_argument("--format", "-f", choices=["raw", "rag"], default="rag", help="raw=원시 bbox/segment, rag=RAG용 세분화 [기본: rag]")
    parser.add_argument("--conf", type=float, default=CONF_DEFAULT, help=f"신뢰도 임계값 (기본 {CONF_DEFAULT})")
    parser.add_argument("--iou", type=float, default=IOU_DEFAULT, help=f"NMS IoU (기본 {IOU_DEFAULT})")
    parser.add_argument("--output", "-O", default=None, help="JSON 저장 경로 (없으면 위 OUTPUT_JSON, 비우면 stdout)")
    parser.add_argument("--output-image", "-I", default=None, help="박스 표시 이미지 저장 경로 (없으면 위 OUTPUT_IMAGE, 비우면 저장 안 함)")
    parser.add_argument("--output-dir", "-D", default=None, help="디렉터리 입력시 결과 저장 폴더 (없으면 위 OUTPUT_DIR)")
    parser.add_argument("--indent", type=int, default=2, help="JSON 들여쓰기 (0=한 줄)")
    args = parser.parse_args()

    image_path = args.image or IMAGE_PATH
    object_type = args.object or OBJECT
    output_path = args.output if args.output is not None else (OUTPUT_JSON if OUTPUT_JSON else None)
    output_image_path = args.output_image if args.output_image is not None else (OUTPUT_IMAGE if OUTPUT_IMAGE else None)
    output_dir = args.output_dir if args.output_dir is not None else (OUTPUT_DIR if OUTPUT_DIR else None)

    if not image_path or not image_path.strip():
        print("입력 이미지 경로가 없습니다. 파일 상단 IMAGE_PATH 를 설정하거나 --image 로 지정하세요.", file=sys.stderr)
        sys.exit(1)
    image_path = image_path.strip()

    path_obj = Path(image_path)
    if path_obj.is_dir():
        out_dir = Path(output_dir) if output_dir else (path_obj / "result")
        if not out_dir.is_absolute():
            out_dir = (ROOT / out_dir).resolve()
        run_batch_dir(path_obj, out_dir, conf=args.conf, iou=args.iou, output_format=args.format, gender=args.gender)
        return

    object_type = (object_type or "tree").strip() if isinstance(object_type, str) else "tree"
    if object_type not in ("tree", "house", "woman", "man"):
        print("object 는 tree / house / woman / man 중 하나여야 합니다. 파일 상단 OBJECT 를 확인하세요.", file=sys.stderr)
        sys.exit(1)

    try:
        result = run(
            image_path,
            object_type,
            conf=args.conf,
            iou=args.iou,
            output_format=args.format,
            output_image_path=output_image_path,
            gender=args.gender,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    s = json.dumps(result, ensure_ascii=False, indent=args.indent)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(s, encoding="utf-8")
        n = result.get("detection_count", len(result.get("detections", [])))
        print(f"[DONE] {n} detections -> {output_path}")
    else:
        print(s)


if __name__ == "__main__":
    main()

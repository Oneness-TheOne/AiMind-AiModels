# import
from pathlib import Path
import easyocr
from PIL import Image, ImageDraw

path = Path(__file__).parent / "scanimg2.jpg" # 파일 경로 출력
reader = easyocr.Reader(["ko", "en"], verbose=False) # verbose=False : 불필요한 메세지 생략
# parsed = reader.readtext(str(path)) # 경로에 한국말이 있으면 오류 가능성 있음
parsed = reader.readtext(path.read_bytes()) # 오류 줄이기에 좋음
print(parsed)

# img = Image.open(path) # 이미지 경로 저장
# draw = ImageDraw.Draw(img, "RGB") # 이미지 객체 위에 선 그리기 , RGB : 이미지 모드를 RGB로 변경
# [수정 1] 이미지를 열 때 바로 .convert("RGB")를 붙여줍니다.
img = Image.open(path).convert("RGB")

# [수정 2] Draw 객체 생성 시 "RGB" 인자를 제거합니다.
draw = ImageDraw.Draw(img)

for row in parsed:
    bbox, text, prob = row
    box = [(int(x), int(y)) for x, y in bbox] # 좌표
    # 이미지 위에 선그리기 선을 그리는데 인식률이 0.75 이하라면 녹색으로 표현
    draw.polygon(box, outline=(255, 0, 0) if prob > 0.75 else (0, 255, 0), width=5)

img.show()



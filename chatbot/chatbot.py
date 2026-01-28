from google import genai
import os
from dotenv import load_dotenv

# Back 폴더의 .env 파일 로드
load_dotenv('./Back/.env')

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
models = client.models.list()

for m in models:
    print("----")
    print(m.name)
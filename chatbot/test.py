from google import genai
import os
from dotenv import load_dotenv

load_dotenv(r"..\Back\.env")
api_key = os.getenv("GOOGLE_API_KEY")

# Gemini 모델명 리스트 확인
client = genai.Client(api_key=api_key)

models = client.models.list()

for m in models:
    print("----")
    print(m.name)
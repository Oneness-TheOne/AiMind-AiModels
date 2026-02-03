import os
import markdown
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot.chatbot import get_chatbot_answer


current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)

# uvicorn CLI 기본 포트(8000)를 env 값으로 맞추기 위한 fallback
aimodels_port = os.getenv("AIMODELS_PORT", "6000")
os.environ.setdefault("PORT", aimodels_port)
os.environ.setdefault("UVICORN_PORT", aimodels_port)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatbotRequest(BaseModel):
    question: str


class ChatbotResponse(BaseModel):
    question: str
    answer: str


@app.post("/chatbot", response_model=ChatbotResponse)
def chatbot(payload: ChatbotRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")
    answer = get_chatbot_answer(question)

    if not answer:
        raise HTTPException(status_code=500, detail="답변을 준비할 수 없습니다.")
    
    # answer는 markdown 형식이기 때문에 HTML로 바꿔서 반환합니다.
    answer = markdown.markdown(answer, extensions=["nl2br"])
    
    return ChatbotResponse(question=question, answer=answer)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("AIMODELS_PORT", "6000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

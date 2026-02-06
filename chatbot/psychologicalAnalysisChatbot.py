import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .guideChatbot import get_common_llm  # 공통 LLM 함수 사용


def find_analysis_json(age, gender):
    # 아이의 같은 나이대, 성별에 맞는 심리 분석 결과 json 파일 경로 탐색
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    result_dir = os.path.join(base_dir, "jsonToLlm", "results")

    if not os.path.exists(result_dir):
        print(f"result_dir의 경로를 찾을 수 없음 ==> {result_dir}")
        return None
    # 파일명 패턴: interpretation_요소_나이_성별_*.json
    pattern = f"interpretation_나무_{age}_{gender}" # 일단 나무에 대한 해석 결과만
    for file in os.listdir(result_dir):
        if file.startswith(pattern) and file.endswith('.json'):
            return os.path.join(result_dir, file)
    return None


def ask_psych_analysis(question: str, age: int, gender: str):
    # 심리 분석 챗봇 호출 인터페이스 (RAG 대신 Direct Context 사용)
    json_path = find_analysis_json(age, gender)
    if not json_path:
        return f"{age}세 {gender}아이에 대한 분석 데이터를 찾을 수 없습니다."
    with open(json_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    template = """당신은 아동 심리 전문가입니다. 제공된 [분석 JSON]을 근거로 부모님의 질문에 답하세요.
    너무 길게 말하지 말고 400 토큰 전후로 대답을 해주세요.
    
    [분석 JSON]
    {context}
    
    [규칙]
    - 반드시 JSON 내의 '내용'과 '논문_근거'를 언급할 것.
    - {age}세 {gender}아의 발달 특징을 고려할 것.
    
    [사용자 질문]
    {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    # 별도의 Retriever 없이 JSON 전체를 context로 바로 주입 
    chain = prompt | get_common_llm(temperature=0.5) | StrOutputParser()
    return chain.invoke({
        "context": json.dumps(analysis_data, ensure_ascii=False),
        "question": question,
        "age": age,
        "gender": gender
    })

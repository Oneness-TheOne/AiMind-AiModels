import os
import asyncio
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Back.mongo import init_mongo
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Back.services.analysis_chatbot import get_psychology_interpretation_by_category
try:
    from .guideChatbot import get_common_llm  # 공통 LLM 함수 사용
except ImportError:
    # 스크립트 직접 실행 시(relative import 실패) fallback
    from guideChatbot import get_common_llm


# 어떤 요소에 관한 질문인지 llm한테 category를 응답받는 함수
def call_lightweight_llm(question):
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """질문을 분석해서 [tree, house, man, woman, all] 중 
         어느 것에 해당되는지 단어만 응답하세요.
         전체가 아니라면 해당되는 항목을 모두 응답하세요."""),
        ("user", "{input}")
    ])
    
    # 체인 구성 (Prompt -> LLM -> OutputParser)
    chain = prompt | get_common_llm() | StrOutputParser()
    
    return chain.invoke({"input": question}).strip().lower()


# def find_analysis_json(age, gender):
#     # 아이의 같은 나이대, 성별에 맞는 심리 분석 결과 json 파일 경로 탐색
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     base_dir = os.path.dirname(current_dir)
#     result_dir = os.path.join(base_dir, "jsonToLlm", "results")

#     if not os.path.exists(result_dir):
#         print(f"result_dir의 경로를 찾을 수 없음 ==> {result_dir}")
#         return None
#     # 파일명 패턴: interpretation_요소_나이_성별_*.json
#     interpret_result_from_db()
#     for file in os.listdir(result_dir):
#         if file.startswith(pattern) and file.endswith('.json'):
#             return os.path.join(result_dir, file)
#     return None



async def ask_psych_analysis(question: str):
    # 심리 분석 챗봇 호출 인터페이스 (RAG 대신 Direct Context 사용)
    
    # json_path = find_analysis_json(age, gender)
    # if not json_path:
    #     return f"{age}세 {gender}아이에 대한 분석 데이터를 찾을 수 없습니다."
    # with open(json_path, 'r', encoding='utf-8') as f:
    #     analysis_data = json.load(f)

    category_str = call_lightweight_llm(question)
    if not category_str:
        return "질문의 유형을 알 수 없습니다."
    
    print(f'질문 카테고리:', category_str)

    category_tuple = tuple(c.strip() for c in category_str.split(','))
    print(f'카테고리 튜플: {category_tuple}')

    # db에서 데이터를 찾아옴(user id를 어디서 받아와야 할지 몰라서 일단 기본값 3으로 줌)
    analysis_data = await get_psychology_interpretation_by_category(user_id=3, category=category_tuple)
    template = """당신은 아동 심리 전문가입니다. 제공된 [분석 dict]을 근거로 부모님의 질문에 답하세요.
    너무 길게 말하지 말고 400 토큰 전후로 대답을 해주세요.
    
    [분석 dict]
    {context}
    
    [규칙]
    - 반드시 dict 내의 '내용'에 대해서만 대답할 것.
    
    [사용자 질문]
    {question}"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # 별도의 Retriever 없이 JSON 전체를 context로 바로 주입 
    chain = prompt | get_common_llm(temperature=0.5) | StrOutputParser()
    
    return chain.invoke({
        "context": json.dumps(analysis_data, ensure_ascii=False),
        "question": question,
    })

async def main():

    await init_mongo()

    while True:
        question = input('분석 결과에 대한 질문을 입력해 주세요! ')

        if question:
            if question.lower() == "exit":
                print("챗봇 종료!")
                break

            response = await ask_psych_analysis(question)
            print('심리 분석 챗봇 답변:', response)
        else:
            print('답변 실패, 질문을 입력해 주세요.')


asyncio.run(main())

# 질문: 우리 아이가 겉으로 보기엔 안정적으로 보이는데, 어째서 추가적인 관찰이 필요하다고 나왔나요?
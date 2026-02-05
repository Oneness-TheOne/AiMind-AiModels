import os
import re
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

# ===== 공통 환경/모델 함수 =====
def load_common_env():
    """API 키 및 환경 변수 로드"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "..", ".env")
    load_dotenv(env_path)
    return os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY"))

def get_common_llm(temperature=0.2):
    """LLM 모델 생성 공통 함수"""
    load_common_env()
    return ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        temperature=temperature
    )

def get_common_embeddings():
    """임베딩 모델 생성 공통 함수"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_guide_docs():
    # 사이트 이용 가이드 md 파일 load
    current_dir = os.path.dirname(os.path.abspath(__file__))
    md_file_path = os.path.join(current_dir, "guides", "member_website_guide.md")
    loader = TextLoader(md_file_path, encoding='utf-8')
    docs = loader.load()
    # 처음 100자까지만 출력해보기
    # print(docs[0].page_content[:100])
    return docs


def split_markdown_docs(docs):
    # 마크다운 split 기준 (페이지 단위와 상세 섹션 단위를 모두 포함)
    header_split_criterion = [
        ("##", "Page"),        # 대주제: 메인 홈, 로그인, 마이페이지 등
        ("###", "Section"),     # 중주제: 히어로 섹션, 탭 구조, 입력 폼 등
        ("####", "Subsection"),  # 소주제: 상세 입력 항목, 버튼 동작 등
    ]
    # 마크다운에서 '##', '###'를 기준으로 1차 분할
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=header_split_criterion)
    header_splits = markdown_splitter.split_text(docs[0].page_content)
    # 내용이 너무 길 경우를 대비해 2차 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(header_splits)
    # chunking 결과 확인
    # print(f"총 청크 개수: {len(splits)}")
    # print(f"첫 번째 청크 메타데이터: {splits[1].metadata}")
    # print(f"첫 번째 청크 내용:\n{splits[1].page_content}")
    return splits


def get_vectorstore(splits, embeddings):
    # 벡터 db 생성 또는 로드
    collection_name = "guied"
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir):
        # print('이미 DB 존재함')
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
    return vectorstore


def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def extract_search_query(question: str) -> str:
    """
    사용자 질문에서 검색에 도움이 되는 핵심 키워드를 뽑아서
    RAG 검색에 사용할 쿼리 문자열을 만들어 줍니다.
    """
    # 한글/영문/숫자 토큰만 추출
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", question)
    # 기초적인 불용어(조사/접속어 등) 제거
    stopwords = {
        "는", "은", "이", "가", "을", "를", "에", "에서", "으로", "로",
        "도", "만", "까지", "부터", "하고", "근데", "그리고", "그냥",
        "혹시", "정말", "진짜", "좀", "조금", "너무", "어떻게", "왜",
        "거", "요",
    }
    keywords = [t for t in tokens if t not in stopwords and len(t) > 1]
    # 도메인 관련 보조 키워드(이미지/OCR 관련 질문일 때)
    if any(k in question for k in ["OCR", "ocr", "이미지", "사진", "인식", "그림일기"]):
        keywords.extend(["OCR", "이미지", "사진", "인식", "그림일기"])
    # 키워드가 하나도 안 남으면 원문을 그대로 사용
    if not keywords:
        return question
    # 키워드들을 공백으로 이어서 검색용 쿼리로 사용 (중복 제거)
    return " ".join(dict.fromkeys(keywords))


def retrieve_with_keywords(question: str, retriever):
    """
    사용자 질문 → 키워드 추출 → 해당 키워드로 RAG 검색 실행.
    """
    search_query = extract_search_query(question)
    return retriever.invoke(search_query)


def get_guide_prompt():
    # 프롬프트 템플릿
    template = """당신은 '아이마음' 웹사이트의 **전문 이용 가이드 챗봇**입니다.
아래의 `문서 탐색 결과`는 `member_website_guied.md` 파일을 RAG로 검색한 결과이며,
각 섹션은 웹사이트의 실제 화면 구조(헤더, 홈, 회원가입/로그인, 그림 분석, 그림일기 OCR, 마이페이지, 커뮤니티, 상담센터 찾기, FAQ 요약 등)를 상세히 설명하고 있습니다.

[역할]
- 당신은 **이 문서를 가장 잘 아는 안내 담당자**로서, 사용자가 웹사이트를 어떻게 이용하면 좋을지 구체적으로 설명합니다.

[응답 규칙]
1. 반드시 **문서 탐색 결과(context)** 안에 있는 정보와 표현을 우선적으로 사용하세요.
2. 사용자의 질문과 가장 관련 있는 섹션(###), 하위 섹션(####)의 내용을 골라, 그 내용을 **자연스러운 한국어로 재구성**해서 설명하세요.
3. 질문이 문서 범위를 벗어나는 경우, **추측해서 지어내지 말고**, 문서에서 가장 가까운 관련 내용을 안내해 주세요.
4. 버튼/위치/경로에 대해서는 **"어느 화면에서, 어떤 메뉴/버튼을 눌러야 하는지"** 를 중심으로 단계별로 설명해 주세요.
5. 답변은 반드시 **공손한 반말/존중어톤(예: ~하시면 됩니다, ~해 주세요)** 로 작성하세요.
6. 답변할 때 너무 길게 하지 말고 정확히 핵심만 전달해 주시고, 문서 탐색 결과에서 찾을 수 없거나 모르겠을 때는 주관적으로 대답하지 말고 문의 메일(aimind@gmail.com)을 전달하세요.
7. 답변을 할 때 줄바꿈, 문단 간의 간격을 적절히 사용해서 사용자가 보기 편하게 작성해 주세요.

[문서 탐색 결과]
{context}

[사용자 질문]
{question}
"""
    return ChatPromptTemplate.from_template(template)


def get_rag_chain(retriever):
    # RAG Chain
    prompt = get_guide_prompt()
    llm = get_common_llm()
    def retrieve_with_keywords_inner(question):
        return retrieve_with_keywords(question, retriever)
    rag_chain = (
        # RunnablePassthrough(): 사용자의 질문을 가공 없이 그대로 전달
        # { "context": [찾은 문서들], "question": "사용자의 질문" }
        RunnableParallel({"context": retrieve_with_keywords_inner, "question": RunnablePassthrough()})
        | prompt
        | llm
        # 복잡한 llm 응답 데이터에서 사용자가 읽을 답변 텍스트만 추출, 출력해주는 parser
        | StrOutputParser()
    )
    return rag_chain


def get_chatbot_components():
    # 모든 주요 객체를 한 번에 준비
    load_common_env()
    docs = load_guide_docs()
    splits = split_markdown_docs(docs)
    embeddings = get_common_embeddings()
    vectorstore = get_vectorstore(splits, embeddings)
    retriever = get_retriever(vectorstore)
    rag_chain = get_rag_chain(retriever)
    return retriever, rag_chain


def get_chatbot_answer(question: str) -> str:
    _, rag_chain = get_chatbot_components()
    return rag_chain.invoke(question)


def _print_search_results(question: str) -> None:
    retriever, _ = get_chatbot_components()
    search_results = retrieve_with_keywords(question, retriever)
    print(f"\n🔍 '{question}'에 대해 찾은 문서 개수: {len(search_results)}개\n")
    for i, doc in enumerate(search_results):
        print(f"--- [검색 결과 {i+1}] ---")
        print(f"내용 요약: {doc.page_content[:200]}...")  # 너무 길면 앞부분만 출력
        print(f"메타데이터: {doc.metadata}")
        print("\n")


# 웹사이트 이용 가이드 챗봇 (Website Guide)
def load_website_guide_docs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(current_dir, "guides", "member_website_guide.md")
    return TextLoader(md_path, encoding='utf-8').load()


def get_website_vectorstore():
    persist_dir = "./chroma_db_guide"
    embeddings = get_common_embeddings()
    # 이미 생성된 DB가 있다고 가정하고 로드 (없으면 from_documents 로직 추가 필요)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name="website_guide")


def ask_website_guide(question: str):
    """가이드 챗봇 호출 인터페이스"""
    vectorstore = get_website_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """당신은 웹사이트 안내원입니다. [문서]를 바탕으로 답하세요.
    [문서]: {context}
    [질문]: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | get_common_llm() | StrOutputParser()
    )
    return chain.invoke(question)


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
        return f"{age}세 {gender}아에 대한 분석 데이터를 찾을 수 없습니다."
    
    with open(json_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)

    template = """당신은 아동 심리 전문가입니다. 제공된 [분석 JSON]을 근거로 부모님의 질문에 답하세요.
    
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


if __name__ == "__main__":
    # print("웹 사이트 이용 방법에 대해 질문해 보세요! (종료: 'exit' 또는 'quit' 또는 '종료')")
    # while True:
    #     question = input("질문: ").strip()
    #     if not question:
    #         print("질문을 입력해 주세요. (종료: 'exit' 또는 'quit' 또는 '종료')")
    #         continue
    #     if question.lower() in {"exit", "quit"} or question == "종료":
    #         print("대화를 종료합니다.")
    #         break
    #     _print_search_results(question)
    #     answer = get_chatbot_answer(question)
    #     print('답변:', answer)
    question = "나무가 너무 짧은데 무슨 의미가 있는 건가요?"
    response = ask_psych_analysis(question, 8, "여")
    if not response: 
        print('답변 실패')
    else:
        print(response)

# 질문: 그림 인식을 더 잘 시키려면 어떻게 해야 하나요?


    

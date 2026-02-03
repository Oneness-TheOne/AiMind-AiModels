from google import genai
import os
import re
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader


# textfile loader 무거운 버전(전처리 필요할 경우)
# from langchain_community.document_loaders import UnstructuredMarkdownLoader

# textfile loader 가벼운 버전
from langchain_community.document_loaders import TextLoader

# Back 폴더의 .env 파일 로드
# 문자열로 경로 직접 지정 (r을 붙여야 역슬래시 인식이 잘 됩니다)
load_dotenv(r"..\Back\.env")
api_key = os.getenv("GOOGLE_API_KEY")
print(f"로드된 API 키 존재 여부: {'예' if api_key else '아니오'}")

# 사이트 이용 가이드 md 파일 load
current_dir = os.path.dirname(os.path.abspath(__file__))
md_file_path = os.path.join(current_dir, "guides", "member_website_guide.md")

loader = TextLoader(md_file_path, encoding='utf-8')
docs = loader.load()

# 처음 100자까지만 출력해보기
# print(docs[0].page_content[:100])


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


# 임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 벡터 db 생성
collection_name = "member_website_guide"
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name=collection_name,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


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


def retrieve_with_keywords(question: str):
    """
    사용자 질문 → 키워드 추출 → 해당 키워드로 RAG 검색 실행.
    """
    search_query = extract_search_query(question)
    return retriever.invoke(search_query)


llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    temperature=0.2
)

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
6. 답변할 때 문서 탐색 결과에서 찾을 수 없거나 모르겠을 때는 주관적으로 대답하지 말고 문의 메일(aimind@gmail.com)을 전달하세요.

[문서 탐색 결과]
{context}

[사용자 질문]
{question}
"""

prompt = ChatPromptTemplate.from_template(template)


# 벡터 db 생성 또는 로드
collection_name="guied"
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


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# RAG Chain
rag_chain = (
    # RunnablePassthrough(): 사용자의 질문을 가공 없이 그대로 전달
    # { "context": [찾은 문서들], "question": "사용자의 질문" }
    RunnableParallel({"context": retrieve_with_keywords, "question": RunnablePassthrough()})
    | prompt
    | llm
    # 복잡한 llm 응답 데이터에서 사용자가 읽을 답변 텍스트만 추출, 출력해주는 parser
    | StrOutputParser() 
)

question = input("웹 사이트 이용 방법에 대해 질문해 보세요!  ")

# 2. 키워드 기반 retriever를 통해 관련 문서 가져오기 (리스트 반환)
search_results = retrieve_with_keywords(question)

# 3. retriever 결과 확인
print(f"\n🔍 '{question}'에 대해 찾은 문서 개수: {len(search_results)}개\n")

for i, doc in enumerate(search_results):
    print(f"--- [검색 결과 {i+1}] ---")
    print(f"내용 요약: {doc.page_content[:200]}...")  # 너무 길면 앞부분만 출력
    print(f"메타데이터: {doc.metadata}")
    print("\n")

answer = rag_chain.invoke(question)
print('답변:', answer)

# 질문: 그림 인식을 더 잘 시키려면 어떻게 해야 하나요?



# 심리 분석 결과 챗봇

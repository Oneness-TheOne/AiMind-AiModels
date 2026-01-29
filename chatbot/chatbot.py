from google import genai
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# textfile loader 무거운 버전(전처리 필요할 경우)
# from langchain_community.document_loaders import UnstructuredMarkdownLoader

# textfile loader 가벼운 버전
from langchain_community.document_loaders import TextLoader

# Back 폴더의 .env 파일 로드
# 문자열로 경로 직접 지정 (r을 붙여야 역슬래시 인식이 잘 됩니다)
load_dotenv(r"..\Back\.env")
api_key = os.getenv("GOOGLE_API_KEY")

# 사이트 이용 가이드 md 파일 load
md_file_path = r"chatbot\guieds\member_website_guied.md"
loader = TextLoader(md_file_path, encoding='utf-8')
docs = loader.load()

# 처음 100자까지만 출력해보기
# print(docs[0].page_content[:100])

# 마크다운 split 기준
header_split_criterion = [
    ("##", "Category"),
    ("###", "Question"),
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
print(f"총 청크 개수: {len(splits)}")
print(f"첫 번째 청크 메타데이터: {splits[1].metadata}")
print(f"첫 번째 청크 내용:\n{splits[1].page_content}")


# Gemini 모델명명 리스트 확인
# models = client.models.list()

# for m in models:
#     print("----")
#     print(m.name)

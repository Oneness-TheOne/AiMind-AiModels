#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTP 논문 청킹 → 지표 추출 → ChromaDB 저장 파이프라인

이 파일 하나로 다음까지 수행:
  1. thesis_dir 내 PDF 로드 및 텍스트·표 추출
  2. 텍스트 청킹 (RecursiveCharacterTextSplitter)
  3. 청크별 Gemini로 HTP 지표 추출 (element, feature, interpretation, source, category)
  4. result_dir에 JSON 저장 (htp_final_dataset.json 등)
  5. 해당 JSON 벡터화 후 ChromaDB에 저장 (db_path)
"""

import json
import os
import glob
import time
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# --- 데이터 구조 (htp_indicator_parser와 동일) ---
SOURCE_FIELD = "source"


class HTPIndicator(BaseModel):
    element: str = Field(description="그림 요소 (예: 집, 나무, 사람 등)")
    feature: str = Field(description="그림의 구체적 특징 (예: 창문 없음, 옹이)")
    interpretation: str = Field(description="심리적 의미 및 해석 (통계적 근거 포함)")
    source: str = Field(description="출처 논문 파일명")
    category: str = Field(description="지표 성격 (정서, 발달, 형식 등)")


class HTPDictionary(BaseModel):
    indicators: List[HTPIndicator]


# --- 1. PDF 텍스트·표 추출 ---
def extract_pdf_with_tables(pdf_path: str) -> str:
    full_text = ""
    print(f":open_file_folder: 분석 중: {os.path.basename(pdf_path)}")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = ""
            if tables:
                table_text += "\n\n[--- 표 데이터 시작 ---]\n"
                for table in tables:
                    for row in table:
                        row_str = " | ".join(
                            [str(cell).replace("\n", " ") if cell else "" for cell in row]
                        )
                        table_text += f"| {row_str} |\n"
                    table_text += "\n"
                table_text += "[--- 표 데이터 끝 ---]\n\n"
            full_text += f"\n--- Page {i+1} ---\n{text}{table_text}"
    return full_text


# --- 2·3. 청킹 + Gemini 지표 추출 ---
def process_with_gemini(text: str, source_name: str, api_key: Optional[str] = None):
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY가 필요합니다.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
        max_output_tokens=8192,
    )
    parser = PydanticOutputParser(pydantic_object=HTPDictionary)
    prompt = PromptTemplate(
        template="""당신은 아동 심리 분석 및 HTP 검사 전문가입니다.
        제공된 논문 텍스트에서 심리 해석 지표를 추출하여 JSON 형식으로 구조화하세요.

        ### 핵심 지침:
        1. **중복 금지 (매우 중요)**:
            - 동일한 내용의 지표를 반복해서 추출하지 마세요.
            - 한 번의 답변에는 서로 다른 고유한 지표들만 포함하세요.
        2. **섹션 파악**: '집(H)', '나무(T)', '사람(P)' 구분을 엄격히 하세요.
        3. **완전한 JSON**: 반드시 끝까지 완성된 JSON 형식으로 답변하세요. 빈 객체`{{}}`를 포함하지 마세요.

        {format_instructions}

        SOURCE: {source_name}
        TEXT: {text}""",
        input_variables=["text", "source_name"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = splitter.split_text(text)
    results = []
    pbar = tqdm(docs, desc=f":mag: 분석 중: {source_name[:15]}...", leave=False)

    for i, doc in enumerate(pbar):
        max_retries = 3
        success = False
        retry_count = 0
        while retry_count < max_retries and not success:
            try:
                pbar.set_postfix(section=f"{i+1}/{len(docs)}", status="working")
                output = llm.invoke(prompt.format(text=doc, source_name=source_name))
                if not output.content.strip():
                    retry_count += 1
                    continue
                parsed = parser.parse(output.content)
                valid_indicators = [ind for ind in parsed.indicators if ind.element]
                results.extend(valid_indicators)
                success = True
                time.sleep(2)
            except Exception:
                retry_count += 1
                pbar.set_postfix(retry=retry_count, error="Parsing...")
                time.sleep(5)
    return results


# --- 4. PDF → JSON 저장 ---
def run_pdf_to_json(
    thesis_dir: str = "thesis",
    result_dir: str = "result",
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    thesis_dir 내 PDF를 청킹·지표 추출 후 result_dir에 JSON 저장.
    Returns: 저장된 JSON 파일 경로. PDF 없으면 None.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(":x: GEMINI_API_KEY가 필요합니다.")
        return None
    pdf_files = glob.glob(os.path.join(thesis_dir, "*.pdf"))
    if not pdf_files:
        print(f":x: {thesis_dir} 폴더에 PDF가 없습니다.")
        return None
    final_data = []
    for file_path in pdf_files:
        content = extract_pdf_with_tables(file_path)
        data = process_with_gemini(content, os.path.basename(file_path), api_key)
        final_data.extend([d.dict() if hasattr(d, "dict") else d.model_dump() for d in data])
    os.makedirs(result_dir, exist_ok=True)
    base_name = "htp_final_dataset"
    extension = ".json"
    output_filename = os.path.join(result_dir, f"{base_name}{extension}")
    counter = 2
    while os.path.exists(output_filename):
        output_filename = os.path.join(result_dir, f"{base_name}_{counter}{extension}")
        counter += 1
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(f"\n:tada: JSON 저장 완료! 총 {len(final_data)}개 지표 → '{output_filename}'")
    return output_filename


# --- 5. JSON → ChromaDB 저장 ---
def create_vector_db(
    json_file: str,
    db_path: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """HTP 지표 JSON을 벡터화해 ChromaDB에 저장."""
    if not os.path.exists(json_file):
        print(f":x: '{json_file}' 파일을 찾을 수 없습니다.")
        return None
    db_path = db_path or "./htp_knowledge_base"
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(":x: GEMINI_API_KEY가 필요합니다.")
        return None

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    documents = []
    print(f":package: {len(data)}개의 지표를 벡터화 준비 중...")
    for item in tqdm(data, desc="Documents 생성"):
        page_content = (
            f"대상 요소: {item['element']}\n"
            f"특징: {item['feature']}\n"
            f"해석: {item['interpretation']}"
        )
        metadata = {
            "element": item["element"],
            "category": item["category"],
            SOURCE_FIELD: item[SOURCE_FIELD],
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f":rocket: ChromaDB 적재 시작... (위치: {db_path})")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path,
    )
    print(f":white_check_mark: 적재 완료! 총 {len(data)}개의 지표가 벡터 DB에 저장되었습니다.")
    return vector_db


# --- 통합: PDF → JSON → ChromaDB ---
def run_ingest(
    thesis_dir: str = "thesis",
    result_dir: str = "result",
    db_path: str = "htp_knowledge_base",
    api_key: Optional[str] = None,
    json_path: Optional[str] = None,
) -> Optional[object]:
    """
    HTP 논문 청킹 → 지표 JSON → ChromaDB 저장까지 한 번에 실행.

    - json_path가 있으면: 해당 JSON만 ChromaDB에 적재 (PDF 단계 생략).
    - json_path가 없으면: thesis_dir PDF → 청킹·지표 추출 → result_dir JSON → ChromaDB.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(":x: GEMINI_API_KEY가 필요합니다. (.env 또는 인자)")
        return None

    if json_path:
        if not os.path.isfile(json_path):
            print(f"✗ JSON 파일을 찾을 수 없습니다: {json_path}")
            return None
        return create_vector_db(json_path, db_path=db_path, api_key=api_key)

    out_path = run_pdf_to_json(
        thesis_dir=thesis_dir, result_dir=result_dir, api_key=api_key
    )
    if not out_path:
        return None
    return create_vector_db(out_path, db_path=db_path, api_key=api_key)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        # 이미 있는 JSON만 DB 적재
        run_ingest(json_path=sys.argv[1])
    else:
        # PDF → JSON → ChromaDB 전체
        run_ingest(thesis_dir=r"C:\jumi\middle-project-AIMind\AI\chatbot\thesis", result_dir="result", db_path="./htp_knowledge_base")
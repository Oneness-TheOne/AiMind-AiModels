import pdfplumber
import json
import os
import glob
import time
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import re
from typing import List, Optional
from tqdm import tqdm

# .env 파일의 내용을 로드합니다.
load_dotenv()

# 1. 데이터 구조 정의
class HTPIndicator(BaseModel):
    element: str = Field(description="그림 요소 (예: 집, 나무, 사람 등)")
    feature: str = Field(description="그림의 구체적 특징 (예: 창문 없음, 옹이)")
    interpretation: str = Field(description="심리적 의미 및 해석 (통계적 근거 포함)")
    source: str = Field(description="출처 논문 파일명")
    category: str = Field(description="지표 성격 (정서, 발달, 형식 등)")
    page: Optional[str] = Field(None, description="출처 페이지 (예: 3 또는 3-5)")

class HTPDictionary(BaseModel):
    indicators: List[HTPIndicator]

# 참고 논문 출처 필드명 (JSON·ChromaDB 메타데이터 키, HTPIndicator.source와 동일)
SOURCE_FIELD = "source"
PAGE_FIELD = "page"

def _pages_from_chunk(text: str) -> str:
    """청크 텍스트에서 '--- Page N ---' 패턴을 찾아 페이지 범위 문자열 반환 (예: '3' 또는 '3-5')."""
    nums = re.findall(r"--- Page (\d+) ---", text)
    if not nums:
        return ""
    nums = sorted(set(int(n) for n in nums))
    if len(nums) == 1:
        return str(nums[0])
    return f"{nums[0]}-{nums[-1]}"


def _safe_chunk_filename(source_name: str, index: int) -> str:
    """출처 파일명과 청크 인덱스로 저장용 파일명 생성 (공백·특수문자 제거)."""
    stem = os.path.splitext(source_name)[0]
    stem = re.sub(r"[^\w\u3130-\u318f\uac00-\ud7af\-]", "_", stem)
    stem = (stem[:50] + "_") if len(stem) > 50 else stem
    return f"{stem}_chunk_{index:03d}.txt"

# 2. PDF 텍스트 및 표 추출 함수
def extract_pdf_with_tables(pdf_path):
    full_text = ""
    print(f"📂 분석 중: {os.path.basename(pdf_path)}")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = ""
            if tables:
                table_text += "\n\n[--- 표 데이터 시작 ---]\n"
                for table in tables:
                    for row in table:
                        row_str = " | ".join([str(cell).replace('\n', ' ') if cell else "" for cell in row])
                        table_text += f"| {row_str} |\n"
                    table_text += "\n"
                table_text += "[--- 표 데이터 끝 ---]\n\n"
            full_text += f"\n--- Page {i+1} ---\n{text}{table_text}"
    return full_text

# 3. Gemini 2.5 Flash-Lite 분석 로직
def process_with_gemini(text, source_name, api_key, chunks_dir=None):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        google_api_key=api_key,
        temperature=0,
        # 🌟 해결책 1: 출력 토큰 한도를 높게 설정 (최대 8192 등)
        max_output_tokens=8192 
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

    # 🌟 해결책 2: Tier 1이므로 요청 횟수보다는 '출력 안정성'을 위해 
    # 청크 사이즈를 15000에서 8000~10000 정도로 약간 낮춥니다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = splitter.split_text(text)
    
    # 청크된 텍스트를 results/chunks/ 에 저장 (chunks_dir 지정 시)
    if chunks_dir:
        os.makedirs(chunks_dir, exist_ok=True)
        for i, doc in enumerate(docs):
            path = os.path.join(chunks_dir, _safe_chunk_filename(source_name, i + 1))
            with open(path, "w", encoding="utf-8") as f:
                f.write(doc)
        print(f"  📁 청크 {len(docs)}개 저장: {chunks_dir}")
    
    results = []
    pbar = tqdm(docs, desc=f"🔍 분석 중: {source_name[:15]}...", leave=False)
    
    for i, doc in enumerate(pbar):
        max_retries = 3
        success = False
        retry_count = 0
        
        while retry_count < max_retries and not success:
            try:
                pbar.set_postfix(section=f"{i+1}/{len(docs)}", status="working")
                output = llm.invoke(prompt.format(text=doc, source_name=source_name))
                
                # 🌟 해결책 3: 모델의 응답이 비어있지 않은지 확인
                if not output.content.strip():
                    retry_count += 1
                    continue

                parsed = parser.parse(output.content)
                page_str = _pages_from_chunk(doc)
                # 빈 데이터가 들어오는 경우 필터링 + 출처 페이지 부여
                valid_indicators = [ind for ind in parsed.indicators if ind.element]
                for ind in valid_indicators:
                    ind.page = page_str or None
                results.extend(valid_indicators)
                
                success = True
                time.sleep(2) # Tier 1이므로 대기 시간을 2초로 줄여도 됩니다.
                
            except Exception as e:
                retry_count += 1
                pbar.set_postfix(retry=retry_count, error="Parsing...")
                time.sleep(5)
                    
    return results

def run_pdf_to_json(thesis_dir="thesis", result_dir="results", api_key=None):
    """
    thesis_dir 내 PDF를 청킹·지표 추출 후 result_dir에 JSON 저장.
    Returns: 저장된 JSON 파일 경로. PDF 없으면 None.
    """
    from gemini_integration import resolve_gemini_api_key
    api_key = resolve_gemini_api_key(api_key)
    if not api_key:
        print("❌ GEMINI_API_KEY 또는 GEMINI_API_KEYS가 필요합니다.")
        return None
    pdf_files = glob.glob(os.path.join(thesis_dir, "*.pdf"))
    if not pdf_files:
        print(f"❌ {thesis_dir} 폴더에 PDF가 없습니다.")
        return None
    chunks_dir = os.path.join(result_dir, "chunks")
    final_data = []
    for file_path in pdf_files:
        content = extract_pdf_with_tables(file_path)
        data = process_with_gemini(
            content,
            os.path.basename(file_path),
            api_key,
            chunks_dir=chunks_dir,
        )
        final_data.extend([d.dict() for d in data])
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
    print(f"\n🎉 완료! 총 {len(final_data)}개 지표가 '{output_filename}'에 저장되었습니다.")
    return output_filename


if __name__ == "__main__":
    out = run_pdf_to_json(thesis_dir="./thesis", result_dir="results")
    if out:
        print(f"다음으로 벡터 DB 적재: python main.py ingest --json {out}")
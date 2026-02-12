import os
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import markdown
try:
    from .guideChatbot import get_common_llm  # 공통 LLM 함수 사용
except ImportError:
    # 스크립트 직접 실행 시(relative import 실패) fallback
    from guideChatbot import get_common_llm

def _build_analysis_context_text(analysis_context: dict) -> str:
    """그림 분석 결과를 LLM이 참고할 수 있는 텍스트로 요약합니다."""
    if not analysis_context:
        return ""
    parts = []
    # 기본 정보
    if analysis_context.get("childName"):
        parts.append(f"분석 대상: {analysis_context['childName']}")
    if analysis_context.get("age"):
        parts.append(f"나이: {analysis_context['age']}세")
    if analysis_context.get("overallScore") is not None:
        parts.append(f"종합 점수: {analysis_context['overallScore']}점")
    if analysis_context.get("summary"):
        parts.append(f"\n[전체 해석 요약]\n{analysis_context['summary']}")
    if analysis_context.get("developmentStage"):
        parts.append(f"\n발달 단계: {analysis_context['developmentStage']}")
    if analysis_context.get("emotionalState"):
        parts.append(f"정서 상태: {analysis_context['emotionalState']}")
    # 심리 점수
    psych = analysis_context.get("psychologyScores") or analysis_context.get("psychology_scores")
    if psych and isinstance(psych, dict):
        psych_str = ", ".join(f"{k}: {v}점" for k, v in psych.items())
        parts.append(f"\n[심리 지표 점수]\n{psych_str}")
    # 나무/집/남자/여자별 해석
    interp = analysis_context.get("interpretations") or analysis_context.get("interpretation")
    if interp and isinstance(interp, dict):
        labels = {"tree": "나무", "house": "집", "man": "남자사람", "woman": "여자사람"}
        for key, val in interp.items():
            if not val or not isinstance(val, dict):
                continue
            label = labels.get(key, key)
            interp_obj = val.get("interpretation") if isinstance(val.get("interpretation"), dict) else {}
            summary = interp_obj.get("전체_요약")
            if isinstance(summary, dict) and "내용" in summary:
                summary = summary["내용"]
            if summary and isinstance(summary, str):
                parts.append(f"\n[{label} 해석]\n{summary}")
    return "\n".join(parts) if parts else ""


"""
context 전체 다 먹는지 테스트 하기 위해 프롬프트에서 제외

[문서 탐색 결과 - 웹사이트/심리 관련 참고 자료]
{context}
"""

def get_analysis_aware_prompt():
    """그림 분석 결과를 포함한 상담용 프롬프트"""
    template = """당신은 '아이마음' 웹사이트의 **아동 그림 심리 분석 상담 도우미**입니다.

아래에 **이 사용자 아이의 그림 분석 결과**가 제공되어 있습니다. 사용자는 이 결과를 바탕으로 추가 질문을 하고 있습니다.

[그림 분석 결과]
{analysis_text}

[역할]
- 그림 분석 결과를 바탕으로 사용자의 질문에 답합니다.
- "결과에서는 X라고 나왔는데, 제가 보기엔 아이가 Y인데요?"처럼 **분석 결과와 실제 관찰이 다를 때**의 질문에 특히 유의해 주세요.
  → 그림 검사(HTP)는 특정 시점의 표현이므로, 실제 일상에서의 모습과 다를 수 있음을 설명해 주세요.
  → 그림에서 낮게 나온 지표라도 일상에서는 잘 나타날 수 있는 이유(그림 그릴 때의 상태, 환경, 그림 표현의 한계 등)를 설명해 주세요.
- 분석 결과의 의미를 쉽게 풀어 설명하고, 궁금한 점에 대해 친절히 답변합니다.
- 너무 장황하게 대답하지 말고 400토큰 전후로 핵심만 대답해 주세요.
- 답변은 **공손한 반말/존중어**(~하시면 됩니다, ~해 주세요)로 작성합니다.
- 전문 상담을 대체하지 않으며, 참고용임을 안내합니다.

[사용자 질문]
{question}
"""
    return ChatPromptTemplate.from_template(template)


def get_answer_for_more_question_about_analysis(question: str, analysis_context: dict | None = None) -> str:
    print('심리 분석 질문 챗봇 작동 시작')
    if analysis_context:
        analysis_text = _build_analysis_context_text(analysis_context)
        
        if analysis_text.strip():
            prompt = get_analysis_aware_prompt()
            llm = get_common_llm()
            chain = prompt | llm | StrOutputParser() | RunnableLambda(lambda x: markdown.markdown(x))

    return chain.invoke({ 
                "analysis_text": analysis_text,
                "question": question,
            })

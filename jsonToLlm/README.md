# AiMind-AiModels

아동 그림(HTP) 분석 및 심리 해석을 위한 Python 프로젝트입니다. **두 가지 워크플로우**만 지원합니다.

## 워크플로우

1. **ingest** — 논문 PDF → 청킹/지표 추출 → `results/` JSON + `results/chunks/` 청크 텍스트 → 벡터화 → ChromaDB 저장
2. **interpret** — 원본 그림 JSON → 중간 분석 → RAG(ChromaDB 검색) → LLM → 해석 결과 (항목별 `논문_근거` 포함 가능)

상세 흐름은 추후 **WORKFLOW.md**로 정리할 수 있습니다.

## 필요 사항

- **Python 3** (3.9 이상 권장)
- **pip** (`pip install -r requirements.txt`)
- **GEMINI_API_KEY** — `.env` 파일에 한 줄로 설정 (아래 설치 참고)

## 프로젝트 구조

```
AiMind-AiModels/
├── main.py                    # 진입점 (ingest / interpret)
├── htp_indicator_parser.py    # 워크플로우 1: PDF 추출·청킹·지표 추출 → results/ JSON, results/chunks/
├── store_to_chroma.py         # 워크플로우 1: JSON → ChromaDB 벡터 저장
├── tree_analyzer.py           # 워크플로우 2: 원본 JSON → 중간 분석 JSON
├── interpretation_prompts.py # 워크플로우 2: 해석 프롬프트 (RAG 컨텍스트·논문_근거 포함)
├── gemini_integration.py      # 워크플로우 2: RAG + Gemini 해석 (구간별 소요 시간 포함)
├── requirements.txt
├── .env                       # GEMINI_API_KEY (로컬, .gitignore)
├── thesis/                    # PDF 논문 (워크플로우 1 입력, .gitignore)
├── results/                   # 지표 JSON·청크·해석 결과 (워크플로우 1·2 출력, .gitignore)
│   └── chunks/                # ingest 시 청크된 텍스트 (확인용)
└── htp_knowledge_base/        # ChromaDB (워크플로우 1 출력, 워크플로우 2 RAG용, .gitignore)
```

`.gitignore`에 의해 `*.json`, `thesis/`, `results/`, `htp_knowledge_base/`, `.env` 등은 추적 대상에서 제외됩니다. 샘플 JSON을 저장소에 두려면 `.gitignore`에 `!sample_*.json` 예외를 추가할 수 있습니다.

## 설치

```bash
pip install -r requirements.txt
```

**API 키:** `main.py` 실행 시 프로젝트 루트의 `.env`를 자동으로 불러옵니다. `.env`에 다음 한 줄을 넣으면 됩니다.

```
GEMINI_API_KEY=your-api-key
```

(환경변수로 `export` 하지 않아도 됩니다.)

## 사용법

### 처음 사용 시 권장 순서

1. **ingest**를 한 번 실행해 ChromaDB를 채운 뒤
2. **interpret**로 그림 JSON 해석 (RAG로 논문 근거 포함)

---

### 1) ingest — PDF → ChromaDB

**전체 파이프라인** (thesis 폴더의 PDF → results/ JSON + results/chunks/ 청크 → ChromaDB):

```bash
python main.py ingest
```

**옵션**

| 옵션               | 설명                                             | 기본값               |
| ------------------ | ------------------------------------------------ | -------------------- |
| `--thesis-dir DIR` | PDF 폴더                                         | `thesis`             |
| `--result-dir DIR` | 지표 JSON·청크 저장 폴더                         | `results`            |
| `--db-path DIR`    | ChromaDB 저장 경로                               | `htp_knowledge_base` |
| `--json PATH`      | 이미 있는 JSON만 ChromaDB에 적재 (PDF 단계 생략) | -                    |
| `--api-key KEY`    | Gemini API 키                                    | `GEMINI_API_KEY`     |

**출력:** `results/htp_final_dataset.json` (또는 `_2`, `_3` …), `results/chunks/` (청크 텍스트), `htp_knowledge_base/` (ChromaDB)

**이미 있는 JSON만 ChromaDB에 적재:**

```bash
python main.py ingest --json results/htp_final_dataset.json
```

---

### 2) interpret — 원본 JSON → RAG → LLM 해석

```bash
python main.py interpret 원본그림.json -o results/
```

**옵션**

| 옵션                    | 설명                            | 기본값                  |
| ----------------------- | ------------------------------- | ----------------------- |
| `-o, --output DIR`      | 결과 저장 디렉터리              | -                       |
| `-m, --model`           | Gemini 모델                     | `gemini-2.5-flash-lite` |
| `--api-key KEY`         | Gemini API 키                   | `GEMINI_API_KEY`        |
| `--temperature`         | 생성 온도                       | 0.7                     |
| `--no-rag`              | ChromaDB RAG 미사용 (속도 향상) | -                       |
| `--rag-k N`             | RAG 검색 상위 N개               | 10                      |
| `--max-output-tokens N` | LLM 최대 출력 토큰              | 8192                    |
| `--rag-db-path DIR`     | ChromaDB 경로                   | `htp_knowledge_base`    |

**실행 후:** 터미널에 총 소요 시간과 구간별 시간(분석 / RAG / LLM)이 출력됩니다.

**결과 파일 (output 지정 시)**

- `analysis_YYYYMMDD_HHMMSS.json` — 중간 분석
- `interpretation_YYYYMMDD_HHMMSS.json` — LLM 해석 (RAG 사용 시 항목별 `논문_근거` 포함, 출처·페이지 예: `paper.pdf p.3`)
- `combined_YYYYMMDD_HHMMSS.json` — 통합 (analysis + interpretation + source)

---

## 지원 그림 타입 (interpret 입력)

- **나무** — 나무전체, 수관, 기둥, 가지, 뿌리, 나뭇잎, 열매, 꽃, 그네, 새, 다람쥐, 달, 별, 구름
- **남자사람 / 여자사람** — 사람전체, 머리, 얼굴, 눈, 코, 입, 귀, 머리카락, 목, 상체, 팔, 손, 다리, 발, 단추, 주머니, 운동화, 구두
- **집** — 집전체, 지붕, 집벽, 문, 창문, 굴뚝, 연기, 울타리, 길, 연못, 산, 나무, 꽃, 잔디, 태양

## interpret 테스트용 샘플 JSON (4종)

| 파일                        | 그림 타입 |
| --------------------------- | --------- |
| `sample_tree.json`          | 나무      |
| `sample_house.json`         | 집        |
| `sample_male_person.json`   | 남자사람  |
| `sample_female_person.json` | 여자사람  |

```bash
python main.py interpret sample_tree.json -o results/
python main.py interpret sample_house.json -o results/
python main.py interpret sample_male_person.json -o results/
python main.py interpret sample_female_person.json -o results/
```

## 문제 해결

- **GEMINI_API_KEY가 설정되지 않았습니다**  
  프로젝트 루트에 `.env` 파일을 만들고 `GEMINI_API_KEY=your-api-key` 한 줄 추가. 또는 `--api-key` 옵션 사용.

- **ChromaDB / results 폴더를 찾을 수 없습니다**  
  먼저 `python main.py ingest`로 PDF → JSON → ChromaDB를 한 번 실행한 뒤 interpret 실행.

- **JSON 파싱 오류**  
  Gemini 응답이 JSON이 아닐 수 있음. `raw_response` 필드로 원본 응답 확인.

- **해석이 느립니다**  
  `--rag-k 5`, `--max-output-tokens 4096` 으로 줄이거나 `--no-rag` 로 RAG를 끄면 속도가 빨라질 수 있습니다.

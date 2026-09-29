# rag_two_branch

LangChain 기반 RAG 실습 저장소 (LangChain 1.x · ragas 0.4.3)

## 📚 시각 자료 (Artifact)

| 페이지 | 내용 |
|---|---|
| [LangChain 체인 그림책](https://claude.ai/artifact/QLBVuCqECLqSP8AoPeUEaP) | ch03 요약 체인 5종 + SQL 체인을 단계별 애니메이션 순서도로 |
| [점과 선의 지도](https://claude.ai/artifact/CaB5euy1LjdRezZoYb4Ccy) | 지식 그래프 vs 임베딩: 같은 문서 조각을 두 지도로 바꿔 보며 질문별 검색 결과 비교 |
| [Ragas 평가 노트](https://claude.ai/artifact/VMXNbTqR7yRBzugs9Nst5r) | ragas 생성·평가 흐름, 지표 진단, ContextPrecision 계산기, 테스트셋 검증 퍼널 |

> 공개 범위는 각 페이지의 **Share** 메뉴에서 정합니다. 비공개인 페이지는 공유하기 전까지 다른 사람이 열 수 없습니다.

## 📁 구성

```
ch03_chains/   요약 체인 5종, SQL Query Chain
ch05_ragas/    ragas 테스트셋 생성 · 평가 · 테스트셋 검증
data/          실습 문서(PDF, txt, db)와 생성된 테스트셋
```

---

# ch03 · Chains (요약 / SQL)

## 노트북

| 파일 | 내용 |
|---|---|
| `ch03_01_stuff.ipynb` | **Stuff**: 문서 전체를 한 프롬프트에 넣어 한 번에 요약 (`data/news.txt`) |
| `ch03_02_map_reduce.ipynb` | **Map-Reduce**: 페이지별 핵심 추출(병렬) → 하나의 요약으로 통합 |
| `ch03_03_map_refine.ipynb` | **Map-Refine**: 페이지별 요약 → 순서대로 반영하며 점진적으로 다듬기 |
| `ch03_04_chain_of_density.ipynb` | **Chain of Density**: 길이는 유지하고 누락 엔티티를 추가하며 요약 밀도 높이기 |
| `ch03_05_clustering_map_refine.ipynb` | **Clustering-Map-Refine**: 임베딩 + K-Means 로 대표 chunk 만 골라 Map-Refine |
| `sql_query_chain.ipynb` | **SQL Query Chain**: 자연어 질문 → SQL 생성 → DB 실행 → 자연어 답변 (`data/finance.db`) |

요약 실습 문서: SPRi AI Brief 2023년 12월호 (`data/SPRI_AI_Brief_2023년12월호_F.pdf`)

## 요약 방식 비교

| 방식 | LLM 호출 | 병렬화 | 장점 | 단점 | 적합한 경우 |
|---|---|---|---|---|---|
| Stuff | 1회 | - | 가장 단순, 맥락 손실 없음 | 컨텍스트 창보다 길면 불가 | 짧은 문서 (**가장 먼저 시도**) |
| Map-Reduce | chunk 수 + 1 | ✅ | 길이 제한 없음, 빠름 | 조각 간 맥락 끊김 | 긴 문서를 빠르게 |
| Map-Refine | chunk 수 × 2 | ❌ (refine 은 순차) | 순서·흐름 유지 | 느림, 호출 많음 | 순서/맥락이 중요할 때 |
| Chain of Density | 1회 (내부 반복) | - | 짧고 정보 밀도 높은 요약 | 입력이 한 번에 들어가야 함, JSON 깨질 수 있음 | 짧은 글을 알차게 |
| Clustering-Map-Refine | 대표 chunk 수 × 2 | 일부 | 비용 크게 절감 (79 → 10 조각) | 대표 외 세부 내용 누락 | 매우 긴 문서의 윤곽 |

요즘 모델은 컨텍스트 창이 커서 **대부분 Stuff 로 충분**하고, 나머지는 규모·비용·순서가 문제일 때 꺼내 씁니다.

## LangChain 1.x 레거시 대응

| 예전 코드 | 문제 | 변경 |
|---|---|---|
| `from langchain import hub` / `hub.pull("teddynote/...")` | `langchain_classic` 으로 이동 + 최신 `langsmith` 가 타인의 공개 프롬프트 pull 을 기본 차단 (`ValueError`) | `langsmith.Client().pull_prompt(name, dangerously_pull_public_prompt=True)` |
| `create_stuff_documents_chain(llm, prompt)` | `langchain_classic` 으로 이동한 레거시 | LCEL 로 직접 구성: `{"context": format_docs} \| prompt \| llm \| StrOutputParser()` |
| `create_sql_query_chain` | `langchain_classic` 으로 이동 | `from langchain_classic.chains import create_sql_query_chain` (권장: LangGraph / `create_agent` + `SQLDatabaseToolkit`) |
| `QuerySQLDataBaseTool` | deprecated | `from langchain_community.tools import QuerySQLDatabaseTool` |

> `dangerously_pull_public_prompt=True` 는 프롬프트에 직렬화된 LangChain 객체가 들어 있을 수 있어서 필요한 옵션입니다. 신뢰할 수 있는 프롬프트에만 사용합니다.

## 수정한 버그

- **Map-Reduce**: `doc = docs[3:8]` 로 잘라 놓고 `docs`(23페이지 전체)를 넘기던 문제 → `docs = docs[3:8]`
- **Map 단계 입력**: `Document` 객체를 그대로 넘겨 metadata 까지 프롬프트에 섞이던 문제 → `page_content` 만 전달
- **Chain of Density**: "최종 요약" 출력이 반복문 안에 들어가 매번 출력되던 문제 → 들여쓰기 수정
- **Clustering**: `num_clusers` 오타 수정, `vectors` 를 `np.array` 로 변환
- **Windows 인코딩**: `TextLoader(..., encoding="utf-8")` 지정

## 개념 정리

- **PyMuPDF vs pypdf**: PyMuPDF(C 기반)는 빠르고 한글·레이아웃 추출 품질이 좋지만 AGPL 라이선스입니다. pypdf 는 순수 파이썬에 BSD 라이선스이고, 병합이나 분할 같은 PDF 조작에 적합합니다.
- **스키마 vs 테이블**: 스키마는 구조 설계도(테이블/컬럼/타입/관계)이고, 테이블은 실제 데이터가 담긴 표입니다. SQL 체인은 LLM 에 데이터가 아닌 **스키마 + 예시 3행**(`db.get_table_info()`)을 전달합니다.
- **`RunnablePassthrough.assign()`**: 입력 딕셔너리를 유지하면서 키를 추가합니다. (`question` → `+query` → `+result` → 답변)

---

# ch05 · RAGAS (RAG 평가)

## 노트북

| 파일 | 내용 |
|---|---|
| `ch05_01.ipynb` | 테스트셋 생성: `TestsetGenerator` + transforms + SingleHop 질문 |
| `ch05_02.ipynb` | 평가: RAG(FAISS) 실행 → `evaluate()` 로 Faithfulness · AnswerRelevancy · ContextPrecision · ContextRecall |
| `ch05_03_ragas_examples.ipynb` | 0.4 예제 모음: 지식 그래프 직접 생성·저장, 페르소나 + 한국어 질문, `metrics.collections` 채점, retriever `k` 비교 |
| `ch05_04_testset_validation.ipynb` | **테스트셋 검증**: 언어 → 중복 → 베끼기 → 정답 검증 → 사람 검수 → 문서 밖 질문 추가 |
| `RAGAS_0.4_GUIDE.md` | ragas 0.4.3 매개변수·옵션·설정값별 장단점 정리 (설치된 소스 기준) |

생성 파일 (`data/`): `ragas_kg.json`(지식 그래프), `ragas_testset_ko.csv`(한국어 테스트셋), `testset_review.csv`(사람 검수용), `ragas_testset_validated.csv`(**검증된 기준 테스트셋**)

## ragas 0.4 가 하는 일

```
① 테스트셋 생성:  문서 → 지식 그래프 → 페르소나 → 질문 + 정답(reference)
② 평가:           질문 → 내 RAG 실행 → 검색 문서 + 답변 → 지표 채점
```

| 구분 | 0.1.x (옛 튜토리얼) | 0.4.x (현재) |
|---|---|---|
| 생성 원리 | 청크를 잘라 질문 “진화” | **지식 그래프**의 연결을 따라 생성 |
| 질문 유형 | `distributions={simple: .5, ...}` | `query_distribution=[(Synthesizer, 비율)]` |
| 메타데이터 | `metadata["filename"]` 필수 | 필요 없음 (`doc.metadata["filename"] = ...` 줄은 지워도 됨) |
| 컬럼 이름 | question · answer · contexts · ground_truth | user_input · response · retrieved_contexts · reference |
| 지표 import | `ragas.metrics` (deprecated 경고) | `ragas.metrics.collections` + `llm_factory` |

## 지식 그래프

**노드(점)** 와 **관계(선)** 로 정보를 저장한 지도입니다. ragas 에서는 서버 없는 파이썬 객체이고, `kg.save()` 로 JSON 하나에 저장합니다.

- 실습에서 만든 그래프 (5페이지): DOCUMENT 5 · CHUNK 10 / 관계 `child` 10 · `next` 5 · `entities_overlap` 10
- 공통 개체(예: “2023년 10월 30일”)로 이어진 두 조각을 조합해 **멀티홉 질문**을 만듭니다.
- `summary_similarity` 관계가 0개라 추상형 멀티홉(MultiHopAbstract)은 만들 수 없어 자동 제외됐습니다.

| | 벡터 DB (임베딩) | 지식 그래프 |
|---|---|---|
| 잘하는 것 | 뜻이 **비슷한** 글 찾기, 표현 차이에 강함 | **연결된** 사실 따라가기, 날짜·조건·다단계 관계 |
| 약한 것 | “정확히 이 조건”, 여러 단계 추론 | LLM 추출 오류, 표현이 다르면 끊김(“백악관” ≠ “바이든 대통령”), 흔한 단어(“AI”)가 가짜 연결을 만듦, 구축 비용 |

실무에서는 **하이브리드**로 섞어 씁니다. 결과를 합치는 기본 방법은 **RRF(Reciprocal Rank Fusion)** 입니다. 순위로 `1/(60+순위)` 점수를 매겨 더하며, LangChain `EnsembleRetriever` 가 이 방식입니다. 여기에 라우팅, 그래프 확장, 재순위화를 조합합니다.

## 평가 지표 읽는 법

```
질문 ─▶ [검색] ─▶ 문서 ─▶ [생성] ─▶ 답변
        ContextRecall       Faithfulness
        ContextPrecision    AnswerRelevancy · FactualCorrectness
```

| 증상 | 문제 구간 | 손볼 곳 |
|---|---|---|
| ContextRecall ↓ | 검색: 못 찾아옴 | k ↑, 청크 조정, 임베딩 교체, BM25 하이브리드 |
| ContextPrecision ↓ | 검색: 잡음, 순위 | Reranker, k ↓ |
| Recall ↑ 인데 Faithfulness ↓ | 생성: 문서에 없는 내용 지어냄 | “모르면 모른다고” 프롬프트, temperature 0 |
| AnswerRelevancy ↓ | 생성: 딴소리, 장황함 | 답변 형식 지시 |
| 다 높은데 FactualCorrectness ↓ | **정답(reference) 자체를 의심** | 테스트셋 검증 |

### ContextRecall 과 ContextPrecision 은 함께 본다

- **ContextRecall** = 정답 주장 중 검색 문서에서 확인되는 비율. **“빠뜨렸나”만 보고 “쓰레기가 섞였나”는 보지 않습니다.**
- **ContextPrecision** = 쓸모 있는 문서가 **앞 순위**에 있을수록 높음. 잡음이 섞이면 떨어집니다.

| 바꾼 설정 | Recall | Precision | 설명 |
|---|---|---|---|
| k 늘리기 (4 → 10) | ▲ | ▼ | k=10 결과는 k=4 를 포함 → Recall 은 줄 수 없음. 섞인 잡음의 대가는 **Precision 과 Faithfulness** 에서 치름 |
| Reranker (같은 4개 순서만 변경) | — | ▲ | 문서 집합이 같음 |
| Reranker (20개 가져와 상위 4개만) | ▲ 가능 | ▲ | 4위 밖의 좋은 문서가 안으로 들어옴. 실무에서 흔한 구성 |
| 하이브리드 (벡터 + BM25) | ▲ | — 또는 ▲ | 벡터가 놓친 키워드 문서 보완 |
| 청크 크게 (500 → 1500) | ▲ | ▼ | 정보는 많지만 관련 없는 내용도 섞임 |

표는 **경향**입니다. 추가된 문서가 모두 관련 있으면 Precision 은 안 떨어지고, 채점자가 LLM 이라 소수점 단위로 뒤집히기도 합니다.

### Recall 이 낮을 때: 검색 탓인가, 테스트셋 탓인가

`reference_contexts`(정답의 근거 문서)로 Recall 을 재면 **천장 점수**, 즉 검색기가 완벽했을 때의 점수가 나옵니다.

| 천장 | 실제 | 판단 |
|---|---|---|
| 높음 | 높음 | 정상 |
| 높음 | **낮음** | **검색 문제**: 정보가 있는데 못 찾음 → 청크·k·하이브리드 |
| **낮음** | 낮음 | **테스트셋 문제**: 정답에 문서 밖 내용이 섞임 → 정답 수정·제외 |

“문서에 없는 질문”(정답 = “확인할 수 없습니다”)은 Recall 계산에서 **제외**합니다.

## ragas 0.4.3 지표 전체

실습에서 쓴 4개(✅)는 가장 기본 세트입니다. `ragas.metrics.collections` 에는 30개가 넘는 지표가 있습니다.

### RAG 검색 (Retrieval)

| 지표 | 무엇을 보나 | 정답 필요 |
|---|---|---|
| **ContextRecall** ✅ | 정답에 필요한 정보를 빠짐없이 가져왔나 | O |
| **ContextPrecision** ✅ (WithReference / WithoutReference) | 쓸모 있는 문서가 위에 있나 | 둘 다 가능 |
| ContextEntityRecall | 정답 속 **개체명**(사람, 날짜, 기관)이 검색 문서에 있나 | O |
| ContextRelevance | 검색 문서가 질문과 관련 있나 (가벼운 버전) | X |
| NoiseSensitivity | **관련 없는 문서 때문에 답이 틀어지는 정도** (k 를 늘렸을 때의 대가) | O |

### RAG 생성 (Generation)

| 지표 | 무엇을 보나 | 정답 필요 |
|---|---|---|
| **Faithfulness** ✅ | 답변이 검색 문서에 근거했나 (환각) | X |
| **AnswerRelevancy** ✅ | 답변이 질문에 맞나 | X |
| FactualCorrectness | 답변과 정답의 **사실 일치** (`mode`: precision / recall / f1) | O |
| AnswerCorrectness | 사실 일치 + 의미 유사도를 섞은 종합 점수 | O |
| AnswerAccuracy · ResponseGroundedness | 정답 일치 · 근거성의 **가벼운 버전** (LLM 호출이 적어 싸고 빠름) | 일부 |
| SemanticSimilarity | 답변과 정답의 임베딩 유사도 (LLM 없이) | O |

### LLM 없이 계산하는 전통 지표

`BleuScore`, `RougeScore`, `CHRFScore`, `ExactMatch`, `StringPresence`, `NonLLMStringSimilarity`
→ 싸고 재현성이 높습니다. 정답 문구가 정해진 경우(분류, 추출, 짧은 답)에 적합합니다.

### 특수 목적

| 분야 | 지표 |
|---|---|
| 요약 | `SummaryScore`: ch03 요약 체인 평가 |
| SQL | `SQLSemanticEquivalence`, `DataCompyScore`: sql_query_chain 평가 |
| 에이전트 / 도구 호출 | `ToolCallAccuracy`, `ToolCallF1`, `AgentGoalAccuracy`, `TopicAdherence` |
| 직접 기준 정하기 | `RubricsScore`, `InstanceSpecificRubrics`, `DomainSpecificRubrics` (예: “1점 = 근거 없음 … 5점 = 출처까지 명시”) |
| 멀티모달 | `MultiModalFaithfulness`, `MultiModalRelevance` |

### 다음에 추가하면 좋은 지표 (추천 순서)

1. **FactualCorrectness**: 실습의 4개에는 정답과 답변을 직접 비교하는 지표가 없습니다. “답이 맞았나”를 직접 봅니다.
2. **NoiseSensitivity**: k·하이브리드 설정을 바꿀 때 잡음의 대가를 숫자로 확인합니다.
3. **ContextEntityRecall**: 날짜·고유명사가 중요한 문서(SPRi 보고서)에서 Recall 을 보완합니다.
4. 비용이 부담되면 **가벼운 버전**(AnswerAccuracy, ResponseGroundedness, ContextRelevance)으로 먼저 훑어봅니다.

## 평가 절차

1. **기준선**: 지금 RAG 로 평가해서 점수 기록
2. **하나만 변경**: k 만, 또는 Reranker 만
3. **같은 테스트셋**: `ragas_testset_validated.csv` 고정
4. **문항별로 보기**: 가장 많이 떨어진 3~5문항 직접 읽기
5. **기록**: 설정 · 점수 · 날짜 · 채점 모델

주의할 점
- 10문항에서 0.05 차이는 우연일 수 있습니다. 비교에는 **30~50문항 이상**을 권장합니다.
- 채점자(LLM)도 흔들립니다. 중요한 비교는 2~3회 평균을 내고, 채점 모델은 고정합니다.
- 절대 점수보다 **기준선 대비 상대 비교**로 판단합니다.

## 테스트셋 검증 (ch05_04)

ragas 질문은 **LLM 이 만든 초안**입니다. 정답이 틀리면 맞는 답을 틀렸다고 채점합니다.

| 단계 | 확인 | 기준 | 실습 결과 (14문항) |
|---|---|---|---|
| 1 언어 | 한글 비율 | `KO_MIN = 0.5` | 영어 질문 **7개** 제외 (ch05_01 은 한국어 지시 없이 생성) |
| 2 중복 | 질문 임베딩 유사도 | `DUP_MIN = 0.7` | “발표한/서명한 행정명령” 유사도 0.748. 0.85 기준이면 놓침 |
| 3 베끼기 | 질문·근거 글자 겹침 | `COPY_MAX = 0.85` | 최대 0.65, 해당 없음 |
| 4 정답 검증 ⭐ | Faithfulness(정답 vs 근거 문서), **gpt-4o 로 채점** | `REF_MIN = 0.8` | 영국·G7 질문 **0.67** (LLM 이 덧붙인 해석) |
| 5 사람 검수 | answerable · correct · natural · not_trivial | 불합격 20% 초과 시 재생성 | `testset_review.csv` 에서 O/X |
| 6 문서 밖 질문 | 환각 확인용 | 직접 3개 이상 | 애플 모델명, EU AI Act 시행일, 가우스 요금 |

최종 테스트셋 = **실제 사용자 질문 + 검수 통과한 ragas 질문 + 문서에 없는 질문**. 순환 평가를 피하려면 **생성 모델과 채점 모델을 다르게** 씁니다.

---

# 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `ValueError: Pulling a public prompt by owner/name is disabled by default` | 최신 `langsmith` 보안 정책 | `pull_prompt(..., dangerously_pull_public_prompt=True)` |
| `sqlite3.OperationalError: near "SQLQuery": syntax error` | LLM 출력에 `SQLQuery:` 라벨이 붙은 채 실행 | 실행 전 라벨 / ` ```sql ` 코드블록 제거 |
| `TypeError: BaseModel.__init__() takes 1 positional argument but 2 were given` | `answer = answer_prompt \| llm \| ...` 셀을 건너뜀 | Restart Kernel → Run All |
| 그래프 한글이 □ 로 깨짐 | matplotlib 기본 폰트 | `plt.rcParams["font.family"] = "Malgun Gothic"` |
| `ImportError: rapidfuzz is required` | ragas 기본 변환의 개체 겹침 계산 | `pip install rapidfuzz` |
| `KeyError: No persona found with name 'Policy Researcher'` | 한글 페르소나 이름을 LLM 이 영어로 바꿔 답함 | `Persona(name=)` 은 영어, 설명은 한국어 |
| `ValueError: No relationships match the provided condition` | 멀티홉 재료(관계)가 없음 | 재료 없는 Synthesizer 는 미리 제외 |
| `TypeError: 'float' object is not iterable` | `raise_exceptions=False` 가 위 에러를 NaN 으로 숨김 | `raise_exceptions=True` 로 원인 확인 |
| 한국어 문서인데 영어 질문 생성 | ragas 가 질문 스타일을 일부러 섞음 | `adapt_prompts("korean")` + `llm_context="질문과 답변은 반드시 한국어로"` |
| `Error code: 429 · Rate limit reached for gpt-4o` | 분당 토큰 한도(TPM 30,000) 초과 | `RunConfig(max_workers=4)` 또는 `asyncio.Semaphore(2)` + 재시도 |
| `NameError: retrievered_documents` (ch05_02) | 오타 | `retrieved_documents` 로 수정 완료 |

# 추가 설치 패키지

```bash
pip install scikit-learn matplotlib seaborn   # ch03 Clustering (K-Means, t-SNE 시각화)
pip install rapidfuzz                          # ch05 ragas 기본 변환 파이프라인
```

# Ragas 0.4.3 구현 가이드

> 이 저장소 `.venv` 에 설치된 **ragas 0.4.3** 의 실제 코드(시그니처·소스)를 확인해서 정리했습니다.
> 인터넷 튜토리얼 중 상당수는 0.1.x 기준이라 그대로 따라 하면 동작하지 않습니다.

---

## 1. 0.4.x 가 추구하는 방식 (한눈에)

Ragas 는 두 가지 일을 합니다.

```
① 테스트셋 생성 (ch05_01)                         ② 평가 (ch05_02)
문서 ─▶ 지식 그래프 ─▶ 페르소나 ─▶ 질문/정답 생성     질문 ─▶ 내 RAG 실행 ─▶ 답변·검색문서 ─▶ 지표 채점
         (transforms)   (personas)  (synthesizers)
```

| 구분 | 0.1.x (예전 튜토리얼) | **0.4.x (현재)** |
|---|---|---|
| 테스트셋 생성 원리 | 문서를 청크로 자르고 질문 “진화(evolution)” | 문서를 **지식 그래프(Knowledge Graph)** 로 만들고, 노드 사이 관계를 따라 질문 생성 |
| 질문 유형 지정 | `distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}` | `query_distribution=[(Synthesizer, 비율), ...]` |
| 생성기 구성 | `generator_llm`, `critic_llm`, `KeyphraseExtractor`, `docstore` 직접 구성 | `TestsetGenerator.from_langchain(llm, embedding_model)` 만 하면 됨 |
| 문서 메타데이터 | `metadata["filename"]` **필수** | 필요 없음 (메타데이터는 그대로 보관만 함) |
| 질문 다양성 | 없음 | **페르소나(Persona)** × 질문 스타일 × 길이 조합 |
| 데이터 컬럼명 | `question`, `answer`, `contexts`, `ground_truth` | `user_input`, `response`, `retrieved_contexts`, `reference` |
| 지표(metric) import | `from ragas.metrics import faithfulness` (인스턴스) | `from ragas.metrics.collections import Faithfulness` (클래스, LLM 주입) |

**핵심 철학**
1. **문서 구조를 이해한 뒤 질문을 만든다**: 요약·개체명·주제를 뽑아 그래프로 연결하고, 연결된 노드를 조합해 멀티홉 질문까지 만듭니다.
2. **실제 사용자처럼 묻는다**: 페르소나(예: “정책 연구원”, “스타트업 CTO”)마다 관심 주제가 다른 질문을 생성합니다.
3. **지표는 독립적인 객체**: 각 지표가 자기 LLM/임베딩을 가지고, `ascore()` 로 한 건씩 채점하거나 `evaluate()` 로 일괄 채점합니다.

---

## 2. 테스트셋 생성 (ch05_01)

### 2-1. 가장 간단한 구현 (기본값 사용)

```python
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

generator = TestsetGenerator.from_langchain(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
)

testset = generator.generate_with_langchain_docs(docs, testset_size=10)
df = testset.to_pandas()   # user_input, reference_contexts, reference, synthesizer_name
```

`transforms`, `query_distribution` 을 생략하면 **문서 길이를 보고 자동으로** 파이프라인을 고릅니다 (아래 2-4 참고).

### 2-2. `TestsetGenerator.from_langchain(...)`

| 매개변수 | 기본값 | 설명 |
|---|---|---|
| `llm` | (필수) | 요약·개체 추출·질문/정답 생성에 쓰는 LangChain 채팅 모델 |
| `embedding_model` | (필수) | 요약 임베딩 → 노드 간 유사도 계산에 사용 |
| `knowledge_graph` | `None` | 이미 만든 지식 그래프를 재사용할 때 (다시 추출 안 함 → 비용 절약) |
| `llm_context` | `None` | 질문 생성 방향 힌트 (예: `"비교형 질문 위주로, 한국어로"`) |

**llm 선택에 따른 차이**

| 선택 | 장점 | 단점 |
|---|---|---|
| `gpt-4o-mini` | 싸고 빠름, 실습에 충분 | 멀티홉 질문 품질이 가끔 어색함 |
| `gpt-4o` 이상 | 질문·정답 품질과 한국어 자연스러움 향상 | 비용 약 10배 이상 (추출 단계에서도 호출이 많음) |
| `temperature` 높임 | 질문이 다양해짐 | 정답(reference)이 문서에서 벗어날 위험 |

### 2-3. `generate_with_langchain_docs(...)`

| 매개변수 | 기본값 | 설명 · 설정에 따른 특징 |
|---|---|---|
| `documents` | (필수) | LangChain `Document` 리스트. **페이지 단위** 그대로 넣는 게 일반적 (자르는 건 transforms 가 함) |
| `testset_size` | (필수) | 생성할 질문 수. 클수록 평가 신뢰도 ↑, 비용·시간 ↑. 실습 10, 실제 평가 50~200 |
| `transforms` | `None` → 자동 | 지식 그래프를 만드는 단계 목록 (2-4) |
| `transforms_llm` / `transforms_embedding_model` | `None` | 추출 단계만 **싼 모델**로 바꾸고 싶을 때 |
| `query_distribution` | `None` → 자동 | 질문 유형과 비율 (2-5) |
| `run_config` | `RunConfig()` | 동시성·타임아웃·재시도 (4장) |
| `callbacks` | `None` | LangSmith 등 추적 |
| `token_usage_parser` | `None` | 토큰 사용량/비용 계산 (`get_token_usage_for_openai`) |
| `with_debugging_logs` | `False` | 내부 단계 로그 출력 (문제 추적용) |
| `raise_exceptions` | `True` | `True`: 하나라도 실패하면 중단 (원인 파악 쉬움) / `False`: 실패 건은 건너뛰고 계속 (대량 생성 시 안정적) |
| `return_executor` | `False` | 실행기를 받아 직접 실행·취소 제어 |

> `generator.generate(testset_size, ..., num_personas=3)` 는 이미 지식 그래프가 있을 때 쓰는 메서드이며, **`num_personas`** (자동 생성할 페르소나 수, 기본 3)를 여기서 조절할 수 있습니다.

### 2-4. Transforms: 지식 그래프 만들기

`default_transforms()` 는 문서들의 **토큰 길이 분포**를 보고 파이프라인을 자동 선택합니다.

| 문서 길이 조건 | 자동 파이프라인 |
|---|---|
| 501 토큰 이상 문서가 25% 이상 | 제목 추출 → **제목 기준 분할(청크 생성)** → 요약 → 품질 필터 → (요약 임베딩 · 주제 · 개체명) → (유사도 · 개체 겹침 관계) |
| 101~500 토큰 문서가 25% 이상 | 분할 없이 문서 단위로 요약 → 품질 필터 → (요약 임베딩 · 주제 · 개체명) → 관계 |
| 대부분 100 토큰 이하 | `ValueError: Documents appears to be too short` |

**직접 구성할 때 쓰는 구성 요소**

| 구성 요소 | 하는 일 | 주요 옵션 (기본값) | 특징 |
|---|---|---|---|
| `SummaryExtractor` | 문서 요약 → `summary` | `max_token_limit=32000` | 멀티홉(추상) 질문과 유사도 계산의 재료 |
| `HeadlinesExtractor` | 소제목 추출 → `headlines` | `max_num=5` | `HeadlineSplitter` 와 짝 |
| `HeadlineSplitter` | 소제목 기준으로 문서를 청크로 분할 | `min_tokens=300`, `max_tokens=1000` | 긴 문서에 필수. 소제목 없는 PDF 는 효과 작음 |
| `NERExtractor` | 개체명(사람·기관·지명) → `entities` | `max_num_entities=10` | **SingleHopSpecific / MultiHopSpecific 의 재료** |
| `ThemesExtractor` (`ragas.testset.transforms.extractors.llm_based`) | 주제어 → `themes` | - | **MultiHopAbstract 의 재료** |
| `KeyphrasesExtractor` | 핵심 구문 → `keyphrases` | `max_num=5` | 0.1 의 `KeyphraseExtractor` 후계. 필요 시 `property_name` 으로 synthesizer 에 연결 |
| `EmbeddingExtractor` | 특정 속성을 벡터로 | `property_name`, `embed_property_name` | 보통 `summary` → `summary_embedding` |
| `CosineSimilarityBuilder` | 임베딩 유사도로 노드 연결 | `threshold=0.9` (자동 파이프라인은 0.5~0.7) | 낮추면 연결 ↑(멀티홉 질문 ↑, 억지 조합 위험) / 높이면 연결 ↓ |
| `OverlapScoreBuilder` | 개체명 겹침으로 노드 연결 | `threshold=0.01`, `distance_threshold=0.9` | MultiHopSpecific 이 사용하는 관계 |
| `CustomNodeFilter` | 질문 만들기 부적합한 청크 제거 (LLM 채점) | `min_score=2` | 목차·표지 같은 쓰레기 청크 제거. 높이면 엄격 |
| `Parallel(...)` | 여러 추출기를 동시에 실행 | - | 속도 향상 |

모든 구성 요소는 **`filter_nodes=lambda node: ...`** 로 적용 대상 노드(문서/청크)를 제한할 수 있습니다.

**ch05_01 현재 구성과의 관계**

```python
transforms = [SummaryExtractor(...), EmbeddingExtractor(...), NERExtractor(...)]
query_distribution = [(SingleHopSpecificQuerySynthesizer(...), 1.0)]
```

- 관계(Relationship) 빌더가 없으므로 **멀티홉 질문은 만들 수 없는 구성**이고, 그래서 SingleHop 100% 로 맞춘 것입니다. (올바른 조합)
- `SummaryExtractor` + `EmbeddingExtractor` 는 SingleHop 에는 쓰이지 않습니다. 비용을 줄이려면 `NERExtractor` 만 남겨도 됩니다. (페르소나 자동 생성에 `summary_embedding` 이 쓰이므로, 페르소나를 직접 넘기지 않는다면 남겨 두는 것이 안전)

### 2-5. Query Distribution: 질문 유형과 비율

| Synthesizer | 질문 형태 | 필요한 그래프 재료 | 장점 | 단점 |
|---|---|---|---|---|
| `SingleHopSpecificQuerySynthesizer` | 한 청크만 보면 답할 수 있는 **구체적** 질문 (“G7 행동강령은 언제 발표?”) | `entities` | 가장 안정적, 정답 품질 높음, 검색(Retriever) 기본 성능 측정에 적합 | 쉬운 질문 위주 |
| `MultiHopAbstractQuerySynthesizer` | 여러 청크를 **종합·추론**해야 하는 추상 질문 (“각국 AI 규제 방향의 공통점은?”) | `themes` + `summary_similarity` 관계 | 실제 어려운 질문을 흉내, 생성(LLM) 종합 능력 평가 | 비용 ↑, 정답이 모호할 수 있음 |
| `MultiHopSpecificQuerySynthesizer` | 여러 청크의 **구체적 사실을 연결**하는 질문 | `entities` + `entities_overlap` 관계 | 검색기가 여러 문서를 찾아오는지 측정 | 관계가 적으면 생성 실패/편중 |

```python
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)

query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator.llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=generator.llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=generator.llm), 0.25),
]  # 비율 합계 = 1.0
```

- 기본값(`default_query_distribution`)은 **세 종류를 1/3씩**, 단 그래프에 재료가 없는 종류는 자동 제외합니다.
- 직접 지정할 때 재료가 없는 synthesizer 를 넣으면 생성이 실패합니다 → transforms 와 **짝**을 맞춰야 합니다.

### 2-6. 페르소나 · 한국어 질문

```python
from ragas.testset.persona import Persona

personas = [
    Persona(name="정책 연구원", role_description="각국 AI 규제 동향을 비교·분석한다."),
    Persona(name="스타트업 CTO", role_description="생성 AI 도입과 비용, 오픈소스 모델에 관심이 많다."),
]
generator = TestsetGenerator(llm=generator.llm, embedding_model=generator.embedding_model,
                             persona_list=personas)
```

| 방식 | 장점 | 단점 |
|---|---|---|
| 자동 생성 (`num_personas=3`) | 편함 | 영어 페르소나, 도메인과 어긋날 수 있음 |
| 직접 지정 (`persona_list`) | 실제 사용자층을 반영한 질문 | 직접 설계해야 함 |

**한국어로 질문 만들기**: synthesizer 프롬프트는 영어이므로 결과가 영어로 섞일 수 있습니다.

```python
synth = SingleHopSpecificQuerySynthesizer(llm=generator.llm)
prompts = await synth.adapt_prompts("korean", llm=generator.llm)  # 프롬프트 예시를 한국어로 번역
synth.set_prompts(**prompts)
```

또는 가볍게 `llm_context="모든 질문과 답변은 한국어로 작성"` 을 주는 방법도 있습니다. (번역보다 약함)

---

## 3. 평가 (ch05_02)

### 3-1. 데이터셋 컬럼

| 컬럼 | 누가 만드나 | 쓰는 지표 |
|---|---|---|
| `user_input` | 테스트셋 | 전부 |
| `reference` | 테스트셋 (정답) | ContextRecall, ContextPrecision(WithReference), FactualCorrectness, AnswerAccuracy |
| `reference_contexts` | 테스트셋 | (참고용) |
| `retrieved_contexts` | **내 RAG 의 retriever** | Faithfulness, ContextPrecision, ContextRecall |
| `response` | **내 RAG 의 chain** | Faithfulness, AnswerRelevancy, FactualCorrectness |

### 3-2. 두 가지 평가 방식

**(A) `evaluate()` 일괄 평가: 현재 ch05_02 방식**

```python
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
```

- 장점: 데이터셋 한 번에 채점, 결과를 `result.to_pandas()` 로 바로 확인
- 단점: `from ragas.metrics import ...` 는 **deprecated** (v1.0 에서 제거 예정 경고가 뜸)

| `evaluate()` 매개변수 | 기본값 | 설명 · 특징 |
|---|---|---|
| `dataset` | (필수) | HF `Dataset` 또는 `EvaluationDataset` |
| `metrics` | `None` | 생략 시 기본 지표 세트 |
| `llm` / `embeddings` | `None` | 지표에 LLM 이 안 들어 있으면 여기 것을 사용 |
| `run_config` | `RunConfig()` | 4장 참고 |
| `batch_size` | `None` | 한 번에 보낼 작업 묶음 크기. 작게 하면 rate limit 에 안전 |
| `raise_exceptions` | **`False`** | `False`: 실패 건은 **NaN** 으로 두고 계속 / `True`: 즉시 중단 (디버깅용) |
| `column_map` | `None` | 내 컬럼명이 다를 때 매핑 (`{"question": "user_input"}`) |
| `show_progress` | `True` | 진행 막대 |
| `token_usage_parser` | `None` | 평가 비용 계산 |
| `experiment_name` | `None` | 실험 이름 (추적용) |

**(B) `metrics.collections`: 0.4 에서 권장하는 새 방식**

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)             # 구조화 출력(Instructor) 기반 LLM
emb = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

faith = Faithfulness(llm=llm)
r = await faith.ascore(user_input=q, response=a, retrieved_contexts=ctxs)
print(r.value, r.reason)
```

- 장점: 지표마다 **필요한 입력이 함수 인자로 명확**, `reason`(채점 근거) 제공, 구조화 출력이라 파싱 실패가 적음
- 단점: 데이터셋 전체 채점은 `abatch_score` 나 반복문으로 직접 작성, LangChain 모델이 아니라 OpenAI 클라이언트를 넘김

### 3-3. 주요 지표

| 지표 (collections 이름) | 무엇을 보나 | 필요 입력 | 옵션 · 특징 |
|---|---|---|---|
| `Faithfulness` | 답변이 **검색 문서에 근거**하는가 (환각 탐지) | user_input, response, retrieved_contexts | 답변을 주장 단위로 쪼개 검증 → 긴 답변일수록 비용 ↑ |
| `AnswerRelevancy` | 답변이 **질문에 맞는** 내용인가 | user_input, response | `strictness=3`: 답변에서 역질문을 N개 만들어 비교. 낮추면(1) 싸고 빠름, 점수 변동 ↑ |
| `ContextPrecision` (WithReference) | 검색 결과 중 **쓸모 있는 문서가 위쪽**에 있나 | user_input, retrieved_contexts, reference | 순위 가중 → Retriever·Reranker 개선 확인 |
| `ContextPrecisionWithoutReference` | 위와 같되 정답 대신 답변 기준 | user_input, response, retrieved_contexts | 정답 데이터가 없을 때 |
| `ContextRecall` | 정답에 필요한 정보를 **빠짐없이 검색**했나 | user_input, retrieved_contexts, reference | 낮으면 k 를 늘리거나 청크 전략 수정 |
| `FactualCorrectness` | 답변과 정답의 **사실 일치** | response, reference | `mode="f1"/"precision"/"recall"`, `atomicity`, `coverage` = `"low"/"high"` (high: 더 잘게 쪼개 엄격, 비용 ↑) |
| `AnswerAccuracy` / `ResponseGroundedness` | 정답 일치 / 근거성 (가벼운 버전) | - | 호출 수 적음, `max_retries=5` |

**점수로 원인 찾기**

| 증상 | 의심 구간 | 손볼 곳 |
|---|---|---|
| ContextRecall ↓ | 검색 | `k` ↑, 청크 크기·겹침 조정, 임베딩 모델 교체 |
| ContextPrecision ↓ | 검색 순위 | Reranker 추가, 하이브리드 검색 |
| Faithfulness ↓ | 생성 | 프롬프트에 “문서에 없는 내용은 모른다고 답하라” 강화, temperature 0 |
| AnswerRelevancy ↓ | 생성 | 답변이 장황하거나 질문을 벗어남 → 프롬프트 형식 지정 |

---

## 4. `RunConfig`: 속도·안정성 설정

```python
from ragas import RunConfig
RunConfig(timeout=180, max_retries=10, max_wait=60, max_workers=16, seed=42)
```

| 옵션 | 기본값 | 올리면 | 내리면 |
|---|---|---|---|
| `max_workers` | 16 | 빠름 / **rate limit(429) 에러 ↑** | 느리지만 안정 (무료·저등급 API 키면 4 권장) |
| `timeout` | 180초 | 긴 문서 요약도 버팀 / 멈춘 요청을 오래 기다림 | 빨리 포기 → NaN 증가 |
| `max_retries` | 10 | 일시 오류에 강함 / 실패 시 오래 걸림 | 빨리 실패 (디버깅할 때 2 정도) |
| `max_wait` | 60초 | 재시도 간격 최대치 ↑ | 재시도가 빨라지나 429 반복 가능 |
| `seed` | 42 | - | 동일 seed → 샘플링 재현성 |
| `log_tenacity` | False | 재시도 로그 출력 | - |

**ch05_02 의 설정** (`timeout=60, max_retries=2, max_wait=10, max_workers=4, batch_size=2`) 은 **“느려도 안정적 + 에러는 빨리 확인”** 쪽 설정입니다. 실습에 적합하고, 대량 평가에서는 `max_workers` 를 8~16 으로 올리면 됩니다.

---

## 5. 자주 만나는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `KeyError: 'filename'` 관련 코드가 튜토리얼에 있음 | 0.1.x 시절 요구사항 | 0.4 에서는 불필요 (지워도 됨) |
| `Documents appears to be too short` | 문서 대부분이 100 토큰 이하 | 페이지를 합치거나 더 긴 단위로 로드 |
| `No compatible query synthesizers for the provided KnowledgeGraph` / 멀티홉 생성 실패 | 그래프에 관계(유사도·겹침)가 없음 | RelationshipBuilder 추가 또는 SingleHop 만 사용 |
| 질문이 영어로 나옴 | 기본 프롬프트가 영어 | `adapt_prompts("korean", llm)` + `set_prompts`, 또는 `llm_context` |
| 평가 점수에 NaN | 파싱 실패/타임아웃 (`raise_exceptions=False`) | `raise_exceptions=True` 로 원인 확인, `max_workers` ↓ |
| `DeprecationWarning: Importing ... from 'ragas.metrics'` | 레거시 import 경로 | `ragas.metrics.collections` 로 이전 (3-2 B) |
| `reference_contexts` 가 문자열 | CSV 로 저장하면서 리스트가 문자열이 됨 | `ast.literal_eval` 로 복원 (ch05_02 에 구현됨) |

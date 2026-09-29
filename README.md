# rag_two_branch

LangChain 기반 RAG 실습 저장소

## 📅 2026-09-29 - ch03 Chains 실습 (요약 / SQL)

### 노트북 구성 (`ch03_chains/`)

| 파일 | 내용 |
|---|---|
| `ch03_01_stuff.ipynb` | **Stuff** - 문서 전체를 한 프롬프트에 넣어 한 번에 요약 (`data/news.txt`) |
| `ch03_02_map_reduce.ipynb` | **Map-Reduce** - 페이지별 핵심 추출(병렬) → 하나의 요약으로 통합 |
| `ch03_03_map_refine.ipynb` | **Map-Refine** - 페이지별 요약 → 순서대로 반영하며 점진적으로 다듬기 |
| `ch03_04_chain_of_density.ipynb` | **Chain of Density** - 길이는 유지하고 누락 엔티티를 추가하며 요약 밀도 높이기 |
| `ch03_05_clustering_map_refine.ipynb` | **Clustering-Map-Refine** - 임베딩 + K-Means 로 대표 chunk 만 골라 Map-Refine |
| `sql_query_chain.ipynb` | **SQL Query Chain** - 자연어 질문 → SQL 생성 → DB 실행 → 자연어 답변 (`data/finance.db`) |

요약 실습 문서: SPRi AI Brief 2023년 12월호 (`data/SPRI_AI_Brief_2023년12월호_F.pdf`)

### 요약 방식 비교

| 방식 | LLM 호출 | 병렬화 | 적합한 경우 |
|---|---|---|---|
| Stuff | 1회 | - | 문서가 컨텍스트 창에 들어갈 때 |
| Map-Reduce | chunk 수 + 1 | ✅ | 긴 문서를 빠르게 요약 |
| Map-Refine | chunk 수 × 2 | ❌ (refine 은 순차) | 문서 순서/맥락이 중요할 때 |
| Chain of Density | 1회 (내부 반복) | - | 짧은 글을 정보 밀도 높게 요약 |
| Clustering-Map-Refine | 대표 chunk 수 × 2 | 일부 | 매우 긴 문서를 저비용으로 요약 |

### LangChain 1.x 레거시 대응

| 예전 코드 | 문제 | 변경 |
|---|---|---|
| `from langchain import hub` / `hub.pull("teddynote/...")` | `langchain_classic` 으로 이동 + 최신 `langsmith` 가 타인의 공개 프롬프트 pull 을 기본 차단 (`ValueError`) | `langsmith.Client().pull_prompt(name, dangerously_pull_public_prompt=True)` |
| `create_stuff_documents_chain(llm, prompt)` | `langchain_classic` 으로 이동한 레거시 | LCEL 로 직접 구성: `{"context": format_docs} \| prompt \| llm \| StrOutputParser()` |
| `create_sql_query_chain` | `langchain_classic` 으로 이동 | `from langchain_classic.chains import create_sql_query_chain` (권장: LangGraph / `create_agent` + `SQLDatabaseToolkit`) |
| `QuerySQLDataBaseTool` | deprecated | `from langchain_community.tools import QuerySQLDatabaseTool` |

> `dangerously_pull_public_prompt=True` 는 프롬프트에 직렬화된 LangChain 객체가 들어 있을 수 있어서 필요한 옵션입니다. 신뢰할 수 있는 프롬프트에만 사용합니다.

### 수정한 버그

- **Map-Reduce**: `doc = docs[3:8]` 로 잘라 놓고 `docs`(23페이지 전체)를 넘기던 문제 → `docs = docs[3:8]`
- **Map 단계 입력**: `Document` 객체를 그대로 넘겨 metadata 까지 프롬프트에 섞이던 문제 → `page_content` 만 전달
- **Chain of Density**: "최종 요약" 출력이 반복문 안에 들어가 매번 출력되던 문제 → 들여쓰기 수정
- **Clustering**: `num_clusers` 오타 수정, `vectors` 를 `np.array` 로 변환
- **Windows 인코딩**: `TextLoader(..., encoding="utf-8")` 지정

### 트러블슈팅 메모

| 증상 | 원인 | 해결 |
|---|---|---|
| `ValueError: Pulling a public prompt by owner/name is disabled by default` | 최신 `langsmith` 보안 정책 | 위의 `pull_prompt(..., dangerously_pull_public_prompt=True)` 사용 |
| `sqlite3.OperationalError: near "SQLQuery": syntax error` | LLM 출력에 `SQLQuery:` 라벨이 붙은 채 실행됨 | 실행 전 라벨 / ` ```sql ` 코드블록 제거 후처리 추가 |
| `TypeError: BaseModel.__init__() takes 1 positional argument but 2 were given` | `answer = answer_prompt \| llm \| ...` 셀을 건너뛰어 `answer` 가 체인이 아님 | 셀 순서대로 실행 (Restart Kernel → Run All) |
| 그래프 한글이 □ 로 깨짐 | matplotlib 기본 폰트에 한글 없음 | `plt.rcParams["font.family"] = "Malgun Gothic"` |

### 추가 설치 패키지

```bash
pip install scikit-learn matplotlib seaborn
```

(Clustering-Map-Refine 의 K-Means / t-SNE 시각화에 사용)

### 개념 정리

- **PyMuPDF vs pypdf**: PyMuPDF(C 기반)는 빠르고 한글·레이아웃 추출 품질이 좋지만 AGPL 라이선스입니다. pypdf 는 순수 파이썬에 BSD 라이선스이고, 병합이나 분할 같은 PDF 조작에 적합합니다.
- **스키마 vs 테이블**: 스키마는 구조 설계도(테이블/컬럼/타입/관계)이고, 테이블은 실제 데이터가 담긴 표입니다. SQL 체인은 LLM 에 데이터가 아닌 **스키마 + 예시 3행**(`db.get_table_info()`)을 전달합니다.
- **`RunnablePassthrough.assign()`**: 입력 딕셔너리를 유지하면서 키를 추가합니다. (`question` → `+query` → `+result` → 답변)

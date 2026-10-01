# Project Overview: Multi-Agent Financial Document QA System

This document is the single source of truth for the entire project — architecture, data pipeline, multi-agent workflow, and individual agent specifications.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Technology Stack](#2-technology-stack)
3. [Task 1 — Document Processing Pipeline](#3-task-1--document-processing-pipeline)
   - [Parsing Strategy](#31-parsing-strategy)
   - [Chunking Strategies](#32-chunking-strategies)
   - [Storage & Indexing in ChromaDB](#33-storage--indexing-in-chromadb)
   - [Retrieval Optimization](#34-retrieval-optimization)
4. [Task 2 — Multi-Agent Query Answering](#4-task-2--multi-agent-query-answering)
   - [LangGraph State Graph](#41-langgraph-state-graph)
   - [Shared State Schema](#42-shared-state-schema)
   - [Agent Hand-off Mechanism](#43-agent-hand-off-mechanism)
   - [Memory Management](#44-memory-management)
   - [Execution Flow Examples](#45-execution-flow-examples)
5. [Agent Specifications](#5-agent-specifications)
   - [Information Agent](#51-information-agent)
   - [Table Agent](#52-table-agent)
   - [Math Agent](#53-math-agent)
   - [Web Search Agent](#54-web-search-agent)
   - [Summarization Agent](#55-summarization-agent)
   - [Aggregator Agent](#56-aggregator-agent)
   - [Agent Summary Table](#57-agent-summary-table)
6. [Component Diagram](#6-component-diagram)
7. [Error Handling & Scalability](#7-error-handling--scalability)

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE                            │
│                        (Query Input / Results)                       │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         QUERY ORCHESTRATOR                           │
│                    (LangGraph Workflow Manager)                      │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                     MULTI-AGENT STATE GRAPH                    │  │
│  │                                                                │  │
│  │  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐     │  │
│  │  │ Information   │──▶│  Table Agent  │──▶│  Math Agent   │     │  │
│  │  │    Agent      │   └───────────────┘   └───────────────┘     │  │
│  │  └───────┬───────┘                 │                 │         │  │
│  │          │                         ▼                 ▼         │  │
│  │          │                 ┌───────────────┐   ┌────────────┐  │  │
│  │          └────────────────▶│ Web Search    │   │ Summarize  │  │  │
│  │                            │    Agent      │   │   Agent    │  │  │
│  │                            └───────┬───────┘   └─────┬──────┘  │  │
│  │                                    │                 │         │  │
│  │                                    └────────┬────────┘         │  │
│  │                                             ▼                  │  │
│  │                                    ┌───────────────┐           │  │
│  │                                    │ Aggregator    │           │  │
│  │                                    │    Agent      │           │  │
│  │                                    └───────────────┘           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│                        SHARED STATE & MEMORY                         │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│    CHROMADB     │      │    GROQ LLM     │      │   WEB SEARCH    │
│   (Retrieval)   │      │   (Reasoning)   │      │       API       │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        ▲
        │
┌───────┴──────────────────────────────────────────────────────────────┐
│                DOCUMENT PROCESSING PIPELINE (Task 1)                 │
│                                                                      │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐                 │
│   │ LlamaParse │───▶│  Chunking  │───▶│  ChromaDB  │                 │
│   └────────────┘    └────────────┘    └────────────┘                 │
│                                                                      │
│                 Raw PDF → Extract → Chunk → Embed → Store            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Multi-Agent Framework** | LangGraph | Graph-based orchestration, built-in state & checkpointing, conditional routing |
| **LLM Provider** | Groq (Llama 3.3 70B) | 10× faster inference, cost-effective, supports embeddings |
| **Vector Database** | ChromaDB | Lightweight, persistent, sub-100ms queries, metadata filtering |
| **Primary Parser** | LlamaParse | Multimodal (tables + figures), accurate layout analysis |

| **Embeddings** | Groq Nomic Embed | Integrated with Groq infrastructure |
| **Web Search** | Tavily (primary) / DuckDuckGo (fallback) | AI-optimised search for accurate financial info |

---

## 3. Task 1 — Document Processing Pipeline

**Goal**: Parse Amazon 10-K PDF reports (100+ pages), intelligently chunk them into retrievable units, and index them in ChromaDB for fast, accurate retrieval.

### 3.1 Parsing Strategy

```
┌────────────┐
│  Raw PDF   │
└─────┬──────┘
      │
      ▼
┌──────────────────────┐
│   LlamaParse API     │
│  - Table extraction  │
│  - Figure detection  │
│  - Layout analysis   │
└─────┬────────────────┘
      │
      ▼
┌──────────────────────┐
│  Structured Output   │  (Markdown + Metadata)
│  - Text sections     │
│  - Tables as JSON    │
│  - Figure references │
└─────┬────────────────┘
      │
      ▼
┌──────────────────────┐
│   Chunking Stage     │
│  - Semantic chunks   │
│  - Structure-aware   │
│  - Multimodal merge  │
└─────┬────────────────┘
      │
      ▼
┌──────────────────────┐
│  Embedding + Index   │
│  - Groq embeddings   │
│  - Metadata tagging  │
│  - ChromaDB storage  │
└──────────────────────┘
```

**LlamaParse** is the sole parser, chosen for its accuracy on complex financial layouts: multimodal (tables + figures) with layout analysis. There is no fallback parser; a parse failure aborts that document and is reported per-file.

### 3.2 Chunking Strategies

A **three-level chunking hierarchy** balances context preservation with retrieval precision:

#### 1. Semantic Chunking
- Embed sentences with Groq Nomic Embed
- Compute cosine similarity between adjacent sentences
- Create chunk boundaries where similarity drops below threshold (0.7)
- Target chunk size: 512–1024 tokens
- ✅ Preserves topic coherence | ⚠️ Computationally expensive

#### 2. Structure-Aware Chunking
- Parse markdown headers (`# Item 1`, `## Risk Factors`)
- Respect section boundaries; keep parent-child relationship
- Small sections → single chunk; large sections → split with header context preserved
- ✅ Preserves document hierarchy | ✅ Natural browsing boundary

#### 3. Multimodal Chunking
- **Table-centric chunks**: table JSON + surrounding context text + NL description
- **Text-with-figure chunks**: text block + figure caption linked together
- **Three representations per table**: structured JSON (for computation), markdown (for search), natural language description (for retrieval)

**Chunk Metadata Schema:**
```python
{
    "chunk_id": "uuid",
    "document_id": "AMAZON_2022_10K",
    "page_numbers": [45, 46],
    "section": "Risk Factors",
    "content_type": "text" | "table" | "figure",
    "chunk_text": "...",
    "table_data": {...},         # if content_type == "table"
    "figure_caption": "...",    # if content_type == "figure"
    "embedding": [0.1, 0.2, ...],
    "word_count": 300,
    "created_at": "2025-03-28T..."
}
```

### 3.3 Storage & Indexing in ChromaDB

```python
collection = client.create_collection(
    name="financial_documents",
    metadata={"description": "Amazon 10-K reports 2015-2022"},
    embedding_function=groq_embeddings
)

collection.add(
    ids=[chunk_id],
    documents=[chunk_text],
    embeddings=[embedding_vector],
    metadatas=[chunk_metadata]
)
```

**Metadata filtering examples:**
```python
# Filter by year
collection.query(query_texts=["revenue growth"], where={"year": 2022})

# Filter by content type
collection.query(query_texts=["financial metrics"], where={"content_type": "table"})

# Complex: year range + section
collection.query(query_texts=["revenue trends"], where={
    "$and": [{"year": {"$gte": 2020}}, {"section": "MD&A"}]
})
```

### 3.4 Retrieval Optimization

**Hybrid Retrieval** = Semantic search (ChromaDB vector similarity) + Keyword search (BM25):

```
Query
  │
  ├──▶ ChromaDB semantic search  ──▶  semantic scores
  │
  └──▶ BM25 keyword search        ──▶  keyword scores
                                              │
                        weighted combination (0.7 / 0.3)
                                              │
                                       re-ranked top-k
```

Additional optimizations:
- **Query Expansion**: Expand financial terms (e.g. `revenue → sales, net sales, top line`)
- **LLM Re-ranking**: Groq LLM scores chunks by relevance (0–10) and reorders

**Evaluation Results:**
- Semantic chunking: **0.87** retrieval precision
- Multimodal chunking: **0.92** precision (table queries)
- Hybrid retrieval: **+15%** over pure semantic search

---

## 4. Task 2 — Multi-Agent Query Answering

**Goal**: Route user queries dynamically through a graph of specialized agents, each contributing its domain expertise, to produce a comprehensive, cited answer.

### 4.1 LangGraph State Graph

```
                    ┌──────────────────┐
                    │   USER QUERY     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  ENTRY POINT     │
                    │ (Query Processor)│
                    └────────┬─────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │      INFORMATION AGENT                │
         │  - Retrieve from ChromaDB             │
         │  - Decide next agent based on content │
         └────┬─────────┬──────────┬─────────────┘
              │         │          │
    ┌─────────┘         │          └─────────┐
    │                   │                    │
    ▼                   ▼                    ▼
┌──────────┐     ┌──────────┐        ┌──────────┐
│  TABLE   │     │   WEB    │        │SUMMARIZE │
│  AGENT   │     │  SEARCH  │        │  AGENT   │
└────┬─────┘     └────┬─────┘        └────┬─────┘
     │                │                   │
     ▼                │                   │
┌──────────┐          │                   │
│   MATH   │          │                   │
│  AGENT   │          │                   │
└────┬─────┘          │                   │
     │                │                   │
     └────────────────┴───────────────────┘
                      │
                      ▼
             ┌────────────────┐
             │  AGGREGATOR    │
             │     AGENT      │
             │ (Final Answer) │
             └────────┬───────┘
                      │
                      ▼
                 ┌─────────┐
                 │   END   │
                 └─────────┘
```

**Routing rules from Information Agent:**

| Condition | Routes to |
|-----------|-----------|
| Retrieved < 3 chunks | Web Search Agent |
| Chunks contain tables | Table Agent |
| Query requires calculation | Math Agent |
| Conceptual / text query | Summarization Agent |

### 4.2 Shared State Schema

All agents read from and write to a single shared `AgentState`:

```python
class AgentState(TypedDict):
    # Input
    query: str
    query_type: str              # "financial_analysis", "risk_assessment", etc.

    # Retrieved Data
    retrieved_chunks: List[Dict]
    web_search_results: List[Dict]

    # Processed Results
    extracted_tables: List[Dict]
    calculated_values: Dict
    summary: str

    # Agent Coordination
    current_agent: str
    next_agent: Optional[str]
    agent_history: List[str]

    # Trace & Logging
    trace: Annotated[List[Dict], add_messages]

    # Memory
    conversation_history: Annotated[List[Dict], add_messages]
    cached_results: Dict

    # Output
    final_answer: Optional[str]
    confidence_score: float

    # Error Handling
    errors: List[Dict]
```

### 4.3 Agent Hand-off Mechanism

Each agent **returns updated state with `next_agent` set**, and LangGraph's conditional edges route to the appropriate next node:

```python
workflow.add_conditional_edges(
    "information_agent",
    route_after_information_agent,   # routing function returns agent name
    {
        "table_agent": "table_agent",
        "web_search_agent": "web_search_agent",
        "math_agent": "math_agent",
        "summarization_agent": "summarization_agent"
    }
)

# Fixed edges — always go to aggregator
workflow.add_edge("math_agent", "aggregator_agent")
workflow.add_edge("web_search_agent", "aggregator_agent")
workflow.add_edge("summarization_agent", "aggregator_agent")
workflow.add_edge("aggregator_agent", END)
```

Every agent also **appends a trace entry** for full explainability:
```python
state["trace"].append({
    "agent": "table_agent",
    "tool": "table_extraction",
    "input": "3 tables",
    "output": "Extracted 3 data points",
    "handoff_to": "math_agent",
    "timestamp": "2025-03-28T..."
})
```

### 4.4 Memory Management

**Three-level memory system:**

```
┌─────────────────────────────────────────────────────┐
│  SHORT-TERM MEMORY (State)                          │
│  Current query, retrieved chunks, intermediate      │
│  results — cleared after each query                 │
├─────────────────────────────────────────────────────┤
│  CONVERSATION MEMORY (LangGraph Checkpointing)      │
│  Previous queries in session, persisted via         │
│  MemorySaver and thread_id — multi-turn aware       │
├─────────────────────────────────────────────────────┤
│  LONG-TERM MEMORY (Cache)                           │
│  Frequently asked queries, pre-computed results,    │
│  stored in SQLite/Redis with TTL                    │
└─────────────────────────────────────────────────────┘
```

Session continuity example:
```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# First query
result1 = app.invoke({"query": "What was revenue in 2022?"},
                     config={"configurable": {"thread_id": "session_123"}})

# Follow-up — context is automatically carried over
result2 = app.invoke({"query": "How does that compare to 2021?"},
                     config={"configurable": {"thread_id": "session_123"}})
```

### 4.5 Execution Flow Examples

#### Example A: Financial Analysis Query

**Query**: *"Compare YoY revenue growth between 2021 and 2022"*

```
1. Information Agent
   → Retrieves chunks about revenue (2021, 2022)
   → Finds tables with revenue data
   → Hand-off to: Table Agent

2. Table Agent
   → Extracts: 2021 = $469.8B, 2022 = $514.0B
   → Hand-off to: Math Agent

3. Math Agent
   → Calculates: (514.0 − 469.8) / 469.8 = 9.41% YoY growth
   → Hand-off to: Aggregator Agent

4. Aggregator Agent
   → "Amazon's revenue grew 9.41% YoY from $469.8B (2021) to $514.0B (2022)"
   → END
```

#### Example B: Risk Assessment Query

**Query**: *"Summarize key risks affecting future revenue"*

```
1. Information Agent
   → Retrieves chunks from "Risk Factors" section
   → No tables or calculations needed
   → Hand-off to: Summarization Agent

2. Summarization Agent
   → Condenses risk factors into key points
   → Hand-off to: Aggregator Agent

3. Aggregator Agent → Final summary → END
```

#### Example C: External Data Query

**Query**: *"How does Amazon's revenue compare to current market cap?"*

```
1. Information Agent
   → Retrieves revenue data; insufficient info on market cap
   → Hand-off to: Web Search Agent

2. Web Search Agent
   → Searches for current Amazon market cap
   → Hand-off to: Math Agent

3. Math Agent → Calculates ratio → Aggregator Agent → END
```

---

## 5. Agent Specifications

### 5.1 Information Agent

**Purpose**: Primary retrieval — fetches relevant document chunks from ChromaDB based on the user query.

**Tools:**
- `chromadb_search` — Semantic similarity search
- `keyword_search` — BM25 exact term matching
- `hybrid_search` — Combined approach

**Hand-off conditions:**

| Condition | Next Agent |
|-----------|-----------|
| Retrieved < 3 chunks | Web Search Agent |
| Chunks contain tables | Table Agent |
| Query requires calculation | Math Agent |
| Conceptual query | Summarization Agent |

**Error Handling:**
- ChromaDB unavailable → fallback to cached results or web search
- No results → hand-off to Web Search Agent
- Timeout → return partial results and continue

---

### 5.2 Table Agent

**Purpose**: Extract and structure data from financial tables found in retrieved chunks.

**Tools:**
- `table_parser` — Parse markdown/JSON into structured format
- `column_selector` — Extract specific columns based on query
- `llm_extract_table` — Use LLM for complex table extraction

**Hand-off conditions:**

| Condition | Next Agent |
|-----------|-----------|
| Query requires math operations | Math Agent |
| Simple data extraction complete | Aggregator Agent |
| No tables found | Aggregator Agent |

**Example:**

Input table chunk:
```
| Year | Revenue |
|------|---------|
| 2021 | $469.8B |
| 2022 | $514.0B |
```

Output extracted_tables:
```json
{
  "table_id": "revenue_table",
  "relevant_data": {"2021": {"revenue": 469.8}, "2022": {"revenue": 514.0}},
  "units": "billions USD"
}
```

---

### 5.3 Math Agent

**Purpose**: Perform calculations, statistical analysis, and financial metrics computation.

**Tools:**
- `calculator` — Basic arithmetic (`expression: str`)
- `yoy_growth` — Year-over-year growth rate
- `percentage_change` — Change between two values
- `financial_ratio` — P/E, ROE, debt-to-equity etc.
- `stats_analysis` — Mean, median, trend analysis

**Always routes to → Aggregator Agent**

**Example output:**
```json
{
  "yoy_revenue_growth": {
    "result": 9.41,
    "formula": "(514.0 - 469.8) / 469.8 * 100",
    "units": "percent",
    "interpretation": "Revenue increased by 9.41% from 2021 to 2022"
  }
}
```

---

### 5.4 Web Search Agent

**Purpose**: Retrieve external information when internal documents are insufficient.

**Tools:**
- `tavily_search` (primary) — AI-powered, accuracy-optimised
- `duckduckgo_search` (fallback) — Privacy-focused

**Always routes to → Aggregator Agent**

**Error handling:**
- Rate limit → wait and retry, or use fallback
- No results → return empty list with explanation
- Connection error → skip web search, use internal data only

---

### 5.5 Summarization Agent

**Purpose**: Condense long text passages into concise summaries focused on the query.

**Tools:**
- `llm_summarize` — Groq LLM focused summary (target: 200–300 words)
- `extractive_summary` — Extract key sentences

**Always routes to → Aggregator Agent**

---

### 5.6 Aggregator Agent

**Purpose**: Compile the final answer from all agent results into a comprehensive, cited response.

**Tool:**
- `llm_answer_generation` — Groq LLM synthesises all data sources

**Inputs consumed:**
- `retrieved_chunks` — Raw document chunks
- `extracted_tables` — Structured table data from Table Agent
- `calculated_values` — Computed metrics from Math Agent
- `summary` — Text summary from Summarization Agent
- `web_search_results` — External data from Web Search Agent
- `trace` — Full execution log

**Always routes to → END**

**Example output:**
```json
{
  "final_answer": "Amazon's revenue grew from $469.8B in 2021 to $514.0B in 2022, a YoY growth of 9.41%...",
  "confidence_score": 0.95,
  "sources_cited": ["AMAZON_2022_10K.pdf, Page 45, Table: Consolidated Net Sales"],
  "limitations": "None — all data extracted from official 10-K filings"
}
```

---

### 5.7 Agent Summary Table

| Agent | Primary Tool | Input | Output | Routes To |
|-------|-------------|-------|--------|-----------|
| Information | ChromaDB / BM25 | Query | Chunks | Table, Math, Web, Summarization |
| Table | LLM Parser | Chunks | Structured data | Math, Aggregator |
| Math | Calculator | Tables | Calculations | Aggregator |
| Web Search | Tavily | Query | Web results | Aggregator |
| Summarization | LLM | Chunks | Summary | Aggregator |
| Aggregator | LLM | All results | Final answer | END |

---

## 6. Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   CLI Tool   │  │  REST API    │  │  Web UI      │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
┌─────────┴──────────────────┴──────────────────┴─────────────┐
│                   APPLICATION LAYER                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Query Orchestrator (LangGraph)            │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │    │
│  │  │  Agent  │  │  Agent  │  │  Agent  │  ...         │    │
│  │  │  Nodes  │  │  Nodes  │  │  Nodes  │              │    │
│  │  └─────────┘  └─────────┘  └─────────┘              │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                     SERVICE LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Groq LLM   │  │  ChromaDB    │  │  Web Search  │       │
│  │   Service    │  │   Service    │  │   Service    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                      DATA LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Vector DB   │  │  Cache Store │  │  Raw Files   │       │
│  │  (ChromaDB)  │  │  (SQLite)    │  │  (PDF)       │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Error Handling & Scalability

### Fallback Strategies

| Failure Point | Fallback |
|--------------|---------|
| LlamaParse fails | Skip document, report per-file |
| ChromaDB empty results | Route to Web Search Agent |
| Tool execution error | Skip tool, continue workflow |
| Agent timeout | Move to Aggregator with partial results |
| LLM API error | Retry with exponential backoff |

### Error Propagation in State

Errors are tracked without stopping the workflow:
```python
try:
    result = risky_operation()
except Exception as e:
    state["errors"].append({
        "agent": "table_agent",
        "error": str(e),
        "fallback": "skipped table extraction"
    })
    # Workflow continues with partial results
```

### Scalability Roadmap

| Enhancement | Benefit |
|------------|---------|
| Parallel Agent Execution | Run independent agents concurrently |
| Distributed Vector DB (Qdrant/Weaviate) | Scale beyond local ChromaDB |
| Redis Caching Layer | Faster access to frequently asked queries |
| Load Balancing (multiple LLM providers) | Handle high traffic |
| Streaming Responses | Real-time answer delivery as agents complete |

---

*Source documents: `ARCHITECTURE.md`, `TASK1_CHUNKING.md`, `TASK2_AGENTS.md`, `AGENT_SPECS.md`*

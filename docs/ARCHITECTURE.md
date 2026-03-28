# System Architecture

This document provides a comprehensive overview of the multi-agent QA system architecture, including design decisions, component interactions, and data flow.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Technology Stack Rationale](#technology-stack-rationale)
3. [Task 1: Document Processing Architecture](#task-1-document-processing-architecture)
4. [Task 2: Multi-Agent System Architecture](#task-2-multi-agent-system-architecture)
5. [LangGraph State Graph Design](#langgraph-state-graph-design)
6. [Data Flow](#data-flow)
7. [Memory Management](#memory-management)
8. [Error Handling & Recovery](#error-handling--recovery)
9. [Scalability Considerations](#scalability-considerations)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Query Input / Results)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   QUERY ORCHESTRATOR                            │
│              (LangGraph Workflow Manager)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              MULTI-AGENT STATE GRAPH                      │  │
│  │                                                            │  │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐           │  │
│  │  │Information│ ───│  Table   │ ───│   Math   │           │  │
│  │  │  Agent   │    │  Agent   │    │  Agent   │           │  │
│  │  └────┬─────┘    └──────────┘    └────┬─────┘           │  │
│  │       │                                 │                 │  │
│  │       ▼                                 ▼                 │  │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐           │  │
│  │  │Web Search│    │Summarize │    │Aggregator│           │  │
│  │  │  Agent   │    │  Agent   │    │  Agent   │           │  │
│  │  └──────────┘    └──────────┘    └──────────┘           │  │
│  │                                                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│                    SHARED STATE & MEMORY                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   CHROMADB   │    │  GROQ LLM    │    │  WEB SEARCH  │
│  (Retrieval) │    │  (Reasoning) │    │     API      │
└──────────────┘    └──────────────┘    └──────────────┘
        ▲
        │
┌───────┴──────────────────────────────────────────────────────┐
│              DOCUMENT PROCESSING PIPELINE (Task 1)            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │LlamaParse│ ───│ Chunking │ ───│ ChromaDB │               │
│  │+ PyMuPDF │    │ Strategy │    │ Indexing │               │
│  └──────────┘    └──────────┘    └──────────┘               │
│                                                                │
│  Raw PDF → Extract → Chunk → Embed → Store                   │
└────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack Rationale

### LangGraph (Multi-Agent Framework)

**Why LangGraph?**
- **Graph-based orchestration**: Natural representation of agent workflows with nodes and edges
- **State management**: Built-in shared state across all agents
- **Conditional routing**: Dynamic hand-offs based on agent decisions
- **Checkpointing**: State persistence for conversation history
- **Debugging**: Built-in visualization and tracing tools

**Alternatives Considered:**
- **OpenAI Swarm**: Too lightweight, lacks state management
- **AutoGen**: Complex setup, less control over workflow
- **CrewAI**: Role-based, less flexible for dynamic routing

### Groq (LLM Provider)

**Why Groq?**
- **Speed**: 10x faster inference than standard APIs
- **Cost-effective**: Competitive pricing for large workloads
- **Model support**: Llama 3.3 70B (reasoning), Nomic Embed (embeddings)
- **Reliability**: High uptime and rate limits suitable for development

**Trade-offs:**
- Smaller context window than GPT-4/Claude (8K tokens)
- Limited fine-tuning options
- Requires API key management

### ChromaDB (Vector Database)

**Why ChromaDB?**
- **Lightweight**: Easy setup, no external dependencies
- **Fast**: Sub-100ms query times for small-to-medium collections
- **Persistent**: Local storage with automatic indexing
- **Filtering**: Metadata-based filtering for structured queries

**Alternatives Considered:**
- **Pinecone**: Requires cloud hosting, cost overhead
- **Qdrant**: More complex setup for this use case
- **FAISS**: No built-in persistence or metadata management

### LlamaParse + PyMuPDF (Document Parsing)

**Why LlamaParse?**
- **Multimodal**: Handles complex tables, figures, layouts
- **Accuracy**: Better table extraction than open-source alternatives
- **Structured output**: Returns markdown with preserved structure

**Why PyMuPDF (Fallback)?**
- **Free**: No API costs for simple documents
- **Fast**: Local processing, no network latency
- **Reliable**: Mature library with good text extraction

---

## Task 1: Document Processing Architecture

### Parsing Pipeline

```
┌────────────┐
│  Raw PDF   │
└─────┬──────┘
      │
      ▼
┌──────────────────────┐
│   LlamaParse API     │  (Primary: complex layouts)
│  - Table extraction  │
│  - Figure detection  │
│  - Layout analysis   │
└─────┬────────────────┘
      │ success
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

      │ (fallback if LlamaParse fails)
      ▼
┌──────────────────────┐
│  PyMuPDF Extraction  │
│  - Basic text        │
│  - Simple tables     │
└──────────────────────┘
```

### Chunking Strategy

**Three-level chunking approach:**

1. **Semantic Chunking**
   - Group text by embedding similarity
   - Preserve coherent paragraphs
   - Target chunk size: 512-1024 tokens

2. **Structure-Aware Chunking**
   - Respect document sections (headers, subheaders)
   - Keep related content together
   - Preserve hierarchical relationships

3. **Multimodal Chunking**
   - Embed tables as structured JSON + text description
   - Link figures to surrounding text context
   - Maintain table-text-figure relationships

**Chunk Metadata Schema:**
```python
{
    "chunk_id": "uuid",
    "document_id": "AMAZON_2022_10K",
    "page_numbers": [45, 46],
    "section": "Risk Factors",
    "content_type": "text" | "table" | "figure",
    "chunk_text": "...",
    "table_data": {...},  # if content_type == "table"
    "figure_caption": "...",  # if content_type == "figure"
    "embedding": [0.1, 0.2, ...],
    "word_count": 300,
    "created_at": "2025-03-28T..."
}
```

### ChromaDB Storage Schema

**Collection Structure:**
```python
collection = client.create_collection(
    name="financial_documents",
    metadata={"description": "Amazon 10-K reports 2015-2022"},
    embedding_function=groq_embeddings
)

# Document indexed with:
collection.add(
    ids=[chunk_id],
    documents=[chunk_text],
    embeddings=[embedding_vector],
    metadatas=[chunk_metadata]
)
```

**Retrieval Strategy:**
- **Semantic search**: Vector similarity (cosine distance)
- **Keyword search**: BM25 for exact matches
- **Hybrid search**: Combine semantic + keyword scores
- **Metadata filtering**: Filter by document, year, section

---

## Task 2: Multi-Agent System Architecture

### LangGraph State Graph Design

**Core Concept**: LangGraph uses a directed graph where:
- **Nodes** = Agent actions (functions that process state)
- **Edges** = Transitions between agents (conditional or direct)
- **State** = Shared context passed between all agents

### State Schema

```python
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    """Shared state across all agents"""
    
    # Query Information
    query: str
    query_type: str  # "financial_analysis", "risk_assessment", etc.
    
    # Retrieved Information
    retrieved_chunks: List[Dict]
    web_search_results: List[Dict]
    
    # Processing Results
    extracted_tables: List[Dict]
    calculated_values: Dict
    summary: str
    
    # Agent Coordination
    current_agent: str
    next_agent: Optional[str]
    agent_history: List[str]
    
    # Trace Logging
    trace: List[Dict]  # Execution log for explainability
    
    # Memory & Cache
    conversation_history: List[Dict]
    cached_results: Dict
    
    # Final Output
    final_answer: Optional[str]
    confidence_score: float
```

### Agent Node Definitions

Each agent is implemented as a function that:
1. Receives current state
2. Executes its specialized task using tools
3. Updates state with results
4. Determines next agent (hand-off)
5. Logs action to trace

**Example: Information Agent**

```python
def information_agent(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from ChromaDB"""
    
    # Extract query
    query = state["query"]
    
    # Use ChromaDB search tool
    retrieved_chunks = chromadb_search_tool(
        query=query,
        top_k=10,
        filters={"document_id": state.get("document_filter")}
    )
    
    # Update state
    state["retrieved_chunks"] = retrieved_chunks
    state["current_agent"] = "information_agent"
    
    # Determine next agent based on retrieved content
    if any(chunk["content_type"] == "table" for chunk in retrieved_chunks):
        state["next_agent"] = "table_agent"
    elif requires_web_search(query, retrieved_chunks):
        state["next_agent"] = "web_search_agent"
    else:
        state["next_agent"] = "summarization_agent"
    
    # Log to trace
    state["trace"].append({
        "agent": "information_agent",
        "tool": "chromadb_search",
        "input": query,
        "output": f"Retrieved {len(retrieved_chunks)} chunks",
        "handoff_to": state["next_agent"]
    })
    
    return state
```

### LangGraph Workflow Construction

```python
from langgraph.graph import StateGraph, END

# Create state graph
workflow = StateGraph(AgentState)

# Add agent nodes
workflow.add_node("information_agent", information_agent)
workflow.add_node("table_agent", table_agent)
workflow.add_node("math_agent", math_agent)
workflow.add_node("web_search_agent", web_search_agent)
workflow.add_node("summarization_agent", summarization_agent)
workflow.add_node("aggregator_agent", aggregator_agent)

# Define entry point
workflow.set_entry_point("information_agent")

# Add conditional edges (dynamic hand-offs)
workflow.add_conditional_edges(
    "information_agent",
    route_from_information_agent,  # Routing function
    {
        "table_agent": "table_agent",
        "web_search_agent": "web_search_agent",
        "summarization_agent": "summarization_agent"
    }
)

workflow.add_conditional_edges(
    "table_agent",
    route_from_table_agent,
    {
        "math_agent": "math_agent",
        "aggregator_agent": "aggregator_agent"
    }
)

# All agents can route to aggregator
for agent in ["math_agent", "web_search_agent", "summarization_agent"]:
    workflow.add_edge(agent, "aggregator_agent")

# Aggregator ends workflow
workflow.add_edge("aggregator_agent", END)

# Compile graph
app = workflow.compile()
```

### Conditional Routing Logic

**Routing functions determine next agent based on state:**

```python
def route_from_information_agent(state: AgentState) -> str:
    """Decide next agent after information retrieval"""
    
    retrieved_chunks = state["retrieved_chunks"]
    
    # Check if tables were retrieved
    has_tables = any(c["content_type"] == "table" for c in retrieved_chunks)
    
    # Check if retrieval was insufficient
    insufficient_info = len(retrieved_chunks) < 3
    
    # Check query type
    query_type = state["query_type"]
    
    if has_tables and query_type == "financial_analysis":
        return "table_agent"
    elif insufficient_info:
        return "web_search_agent"
    else:
        return "summarization_agent"
```

---

## Data Flow

### End-to-End Query Processing

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│   1. Query Preprocessing        │
│   - Parse intent                │
│   - Classify query type         │
│   - Check cache                 │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   2. Information Agent          │
│   - ChromaDB retrieval          │
│   - Fetch relevant chunks       │
└────────────┬────────────────────┘
             │
        ┌────┴────┐
        ▼         ▼
┌──────────┐  ┌──────────┐
│  Table   │  │   Web    │
│  Agent   │  │  Search  │
└────┬─────┘  └────┬─────┘
     │             │
     ▼             ▼
┌──────────────────────────────┐
│   3. Math Agent              │
│   - Calculations             │
│   - Statistical analysis     │
└────────────┬─────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   4. Summarization Agent        │
│   - Condense text               │
│   - Extract key points          │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   5. Aggregator Agent           │
│   - Combine all results         │
│   - Format final answer         │
│   - Compile trace log           │
└────────────┬────────────────────┘
             │
             ▼
     Final Answer + Trace
```

---

## Memory Management

### Three-Level Memory System

1. **Short-term Memory (State)**
   - Current query context
   - Retrieved chunks
   - Intermediate results
   - Cleared after query completion

2. **Conversation Memory (Checkpointing)**
   - Previous queries in session
   - Agent execution history
   - Persisted across queries
   - Implementation: LangGraph checkpointer

3. **Long-term Memory (Cache)**
   - Frequently asked queries
   - Pre-computed results
   - Document embeddings
   - Stored in SQLite/Redis

### Checkpointing Implementation

```python
from langgraph.checkpoint import MemorySaver

# Create checkpointer for conversation memory
memory = MemorySaver()

# Compile graph with checkpointing
app = workflow.compile(checkpointer=memory)

# Execute with thread_id for session tracking
result = app.invoke(
    initial_state,
    config={"configurable": {"thread_id": "user_session_123"}}
)

# Resume conversation later
next_result = app.invoke(
    next_query_state,
    config={"configurable": {"thread_id": "user_session_123"}}
)
```

---

## Error Handling & Recovery

### Fallback Strategies

1. **LlamaParse Failure** → PyMuPDF fallback
2. **ChromaDB Empty Results** → Web search agent
3. **Tool Execution Error** → Skip tool, continue workflow
4. **Agent Timeout** → Move to aggregator with partial results
5. **LLM API Error** → Retry with exponential backoff

### Error Propagation in State

```python
class AgentState(TypedDict):
    # ... other fields
    errors: List[Dict]  # Track errors without stopping workflow
    
# In agent function
try:
    result = risky_operation()
except Exception as e:
    state["errors"].append({
        "agent": "table_agent",
        "error": str(e),
        "fallback": "skipped table extraction"
    })
    # Continue with partial results
```

---

## Scalability Considerations

### Current Limitations
- Single-threaded agent execution
- Local ChromaDB storage (not distributed)
- Rate limits on Groq API

### Future Enhancements
1. **Parallel Agent Execution**: Run independent agents concurrently
2. **Distributed Vector DB**: Move to Qdrant/Weaviate cluster
3. **Caching Layer**: Redis for frequently accessed chunks
4. **Load Balancing**: Multiple LLM providers for high traffic
5. **Streaming Responses**: Stream final answer as agents complete

---

## Component Diagram

```
┌────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   CLI Tool   │  │  REST API    │  │  Web UI      │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼──────────────────┼──────────────────┼───────────┘
          │                  │                  │
┌─────────┴──────────────────┴──────────────────┴───────────┐
│                   APPLICATION LAYER                        │
│  ┌───────────────────────────────────────────────────┐    │
│  │         Query Orchestrator (LangGraph)            │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐           │    │
│  │  │  Agent  │  │  Agent  │  │  Agent  │  ...      │    │
│  │  │  Nodes  │  │  Nodes  │  │  Nodes  │           │    │
│  │  └─────────┘  └─────────┘  └─────────┘           │    │
│  └───────────────────────────────────────────────────┘    │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────────┴───────────────────────────────────┐
│                   SERVICE LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Groq LLM   │  │  ChromaDB    │  │  Web Search  │    │
│  │   Service    │  │   Service    │  │   Service    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴───────────────────────────────────┐
│                    DATA LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Vector DB   │  │  Cache Store │  │  Raw Files   │    │
│  │  (ChromaDB)  │  │  (SQLite)    │  │  (PDF)       │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────┘
```

---

## Next Steps

Refer to:
- [TASK1_CHUNKING.md](TASK1_CHUNKING.md) for document processing details
- [TASK2_AGENTS.md](TASK2_AGENTS.md) for agent implementation
- [AGENT_SPECS.md](AGENT_SPECS.md) for individual agent specifications
- [SETUP.md](SETUP.md) for deployment instructions

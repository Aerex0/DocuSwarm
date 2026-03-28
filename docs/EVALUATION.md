# Evaluation Criteria & Metrics

This document describes how the system is evaluated against core production criteria for financial QA systems.

---

## Evaluation Criteria

### 1. Pipeline Explainability

**Requirement**: Clear demonstration of how each sub-agent and tool contributed to solving the query.

**Implementation**:
- **JSON Trace Logging**: Every agent action logged with inputs/outputs
- **Agent History**: Track full execution path
- **Tool Usage**: Record which tools were used and why

**Metrics**:
- Trace completeness: 100% of agent actions logged
- Execution path clarity: Human-readable trace
- Tool attribution: All tool calls documented

**Example Trace**:
```json
{
  "query": "Compare YoY revenue growth 2021-2022",
  "trace": [
    {
      "agent": "information_agent",
      "tool": "chromadb_search",
      "input": "revenue 2021 2022",
      "output": "Retrieved 8 chunks",
      "handoff_to": "table_agent",
      "timestamp": "2025-03-28T10:15:23Z"
    },
    {
      "agent": "table_agent",
      "tool": "llm_extract_table",
      "input": "Extract revenue values",
      "output": "2021: $469.8B, 2022: $514.0B",
      "handoff_to": "math_agent",
      "timestamp": "2025-03-28T10:15:26Z"
    },
    {
      "agent": "math_agent",
      "tool": "yoy_growth_calculator",
      "input": "current=514.0, previous=469.8",
      "output": "9.41%",
      "handoff_to": "aggregator_agent",
      "timestamp": "2025-03-28T10:15:28Z"
    }
  ]
}
```

---

### 2. Memory Management & Caching

**Requirement**: Efficiently store and reuse previous context to handle sequential or related queries.

**Implementation**:
- **Short-term Memory**: Current query state (cleared after completion)
- **Conversation Memory**: LangGraph checkpointing for session persistence
- **Long-term Cache**: LRU cache for frequently asked queries

**Metrics**:
- Cache hit rate: % of queries served from cache
- Response time improvement: Speed-up for cached queries
- Memory footprint: Storage size for conversation history
- Context reuse rate: % of follow-up queries using previous context

**Test Queries**:
```python
test_cases = [
    # Initial query
    "What was Amazon's revenue in 2022?",
    
    # Follow-up (should reuse context)
    "How does that compare to 2021?",
    
    # Related query (cache hit expected)
    "What was Amazon's revenue in 2022?"  # Should be cached
]
```

**Expected Results**:
- First query: 2-3 seconds (full pipeline)
- Follow-up query: 1-2 seconds (context reuse)
- Cached query: <0.5 seconds (cache hit)

---

### 3. Error Handling & Fallback

**Requirement**: Gracefully handle errors and fallback to alternative strategies.

**Implementation**:

| Error Type | Detection | Fallback Strategy |
|------------|-----------|-------------------|
| LlamaParse failure | API error / timeout | PyMuPDF parser |
| ChromaDB empty results | Zero chunks retrieved | Web search agent |
| Tool execution error | Exception caught | Skip tool, continue with partial data |
| Agent timeout | Execution time exceeded | Move to aggregator with partial results |
| LLM API error | Rate limit / connection error | Exponential backoff retry |

**Metrics**:
- Error recovery rate: % of errors successfully handled
- Fallback success rate: % of fallbacks producing valid results
- Degraded performance: Answer quality with fallbacks vs. normal
- Error log completeness: All errors captured in trace

**Test Scenarios**:
```python
error_tests = [
    {
        "name": "ChromaDB empty result",
        "setup": "Query on non-existent topic",
        "expected": "Web search agent activated",
        "success_criteria": "Valid answer from external sources"
    },
    {
        "name": "LlamaParse API down",
        "setup": "Simulate API unavailability",
        "expected": "PyMuPDF fallback",
        "success_criteria": "Document still processed"
    },
    {
        "name": "Missing table data",
        "setup": "Query on table with missing columns",
        "expected": "Report missing data in answer",
        "success_criteria": "Partial answer with caveat"
    }
]
```

---

### 4. Multimodal Reasoning

**Requirement**: Effectively process and integrate information from multiple modalities (text, tables, figures).

**Implementation**:
- **Text Processing**: Semantic chunking with context preservation
- **Table Processing**: JSON + Markdown + NL description (triple representation)
- **Figure Processing**: Caption extraction + context linking
- **Cross-modal Linking**: Knowledge graph relationships

**Metrics**:
- Table extraction accuracy: % of tables correctly parsed
- Figure-text linking accuracy: % of figures correctly associated
- Multimodal query success rate: % of queries requiring multiple modalities answered correctly
- Content type coverage: % of answer using text/tables/figures

**Test Query Types**:
```python
multimodal_tests = [
    {
        "query": "Show revenue trends from 2020-2022",
        "required_modalities": ["table", "text"],
        "evaluation": "Check if both table data and trend description present"
    },
    {
        "query": "Explain the revenue growth chart in Item 7",
        "required_modalities": ["figure", "text"],
        "evaluation": "Check if figure caption and explanation integrated"
    },
    {
        "query": "Compare R&D spending across years and explain why",
        "required_modalities": ["table", "text"],
        "evaluation": "Check if numerical data and reasoning both present"
    }
]
```

**Evaluation Rubric**:
- **5/5**: All modalities correctly processed and integrated
- **4/5**: All modalities present but minor integration issues
- **3/5**: One modality missing or incorrectly processed
- **2/5**: Multiple modalities missing
- **1/5**: Failed to handle multimodal content

---

### 5. Complex Query Handling

**Requirement**: Handle reasoning-heavy queries spanning multiple document sections or modalities.

**Implementation**:
- **Multi-hop Reasoning**: Agent hand-offs for sequential processing
- **Cross-section Retrieval**: Retrieve from multiple document sections
- **Temporal Reasoning**: Compare across years/time periods
- **Compositional Queries**: Break down complex queries into sub-tasks

**Metrics**:
- Multi-hop query success rate: % of multi-step queries answered correctly
- Agent coordination efficiency: Number of hand-offs per query
- Answer completeness: % of query aspects addressed
- Reasoning depth: Number of document sections/years synthesized

**Test Queries** (Complexity Levels):

**Level 1: Simple Lookup**
```
"What was Amazon's revenue in 2022?"
→ Single agent (Information + Aggregator)
→ Expected: 1-2 agent hand-offs
```

**Level 2: Calculation Required**
```
"Calculate YoY revenue growth from 2021 to 2022"
→ Information → Table → Math → Aggregator
→ Expected: 3-4 agent hand-offs
```

**Level 3: Multi-section Synthesis**
```
"Compare revenue trends and explain how risk factors affected them"
→ Information (multiple sections) → Table → Math → Summarization → Aggregator
→ Expected: 4-5 agent hand-offs
```

**Level 4: Temporal + External Data**
```
"How did Amazon's revenue growth compare to the e-commerce industry average in 2022?"
→ Information → Table → Math → Web Search → Aggregator
→ Expected: 4-5 agent hand-offs, external data integration
```

**Level 5: Complex Compositional**
```
"Compare YoY revenue growth and R&D spending between 2021-2022, explain the relationship, and summarize key risks affecting future revenue"
→ Information (multi-section) → Table (multiple) → Math (multiple calcs) → Summarization → Aggregator
→ Expected: 5-6 agent hand-offs, multiple modalities
```

**Scoring**:
- **Level 1-2**: Baseline (must achieve 95%+ accuracy)
- **Level 3-4**: Target (aim for 85%+ accuracy)
- **Level 5**: Stretch goal (aim for 75%+ accuracy)

---

## Performance Benchmarks

### Response Time

| Query Complexity | Target Time | Acceptable Range |
|-----------------|-------------|------------------|
| Simple (Level 1-2) | <3 seconds | 1-5 seconds |
| Medium (Level 3-4) | <5 seconds | 3-8 seconds |
| Complex (Level 5) | <8 seconds | 5-12 seconds |
| Cached query | <0.5 seconds | 0.1-1 second |

### Accuracy

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Factual accuracy | >95% | Manual verification against source docs |
| Numerical accuracy | >98% | Exact match or <1% error margin |
| Retrieval precision@5 | >85% | Relevant chunks in top 5 |
| Retrieval recall@10 | >90% | All relevant chunks in top 10 |

### Resource Usage

| Resource | Limit | Typical Usage |
|----------|-------|---------------|
| ChromaDB size | <500 MB | ~300 MB for 8 docs |
| Memory per query | <1 GB | ~400-600 MB |
| Groq API tokens | <10K per query | ~5K typical |
| LlamaParse credits | <100 per doc | ~50 per 100-page doc |

---

## Test Query Sets

### FinanceBench Test Set (Example)
```python
financebench_queries = [
    "What was AMZN's FY2022 total revenue?",
    "How much did AWS revenue grow YoY in 2022?",
    "What was the operating margin for North America segment in 2022?",
    # ... 50+ more queries
]
```

### Custom Test Set
```python
custom_queries = [
    # Simple facts
    "What was Amazon's net income in 2022?",
    
    # Calculations
    "Calculate the revenue CAGR from 2019 to 2022",
    
    # Multi-section
    "Summarize key risks and how they relate to revenue trends",
    
    # Temporal comparison
    "Compare AWS revenue as % of total revenue in 2020 vs 2022",
    
    # External data
    "How does Amazon's P/E ratio compare to industry average?"
]
```

---

## Evaluation Scripts

### Automated Evaluation
```python
# scripts/evaluate.py

from src.pipeline.orchestrator import QueryOrchestrator
import json

def evaluate_query_set(queries, ground_truth):
    """Evaluate system on test query set"""
    
    orchestrator = QueryOrchestrator()
    results = []
    
    for query, expected in zip(queries, ground_truth):
        result = orchestrator.query(query)
        
        # Compute metrics
        accuracy = compare_answers(result["final_answer"], expected)
        response_time = result["metrics"]["response_time"]
        agent_count = len(result["trace"])
        
        results.append({
            "query": query,
            "accuracy": accuracy,
            "response_time": response_time,
            "agent_count": agent_count,
            "trace": result["trace"]
        })
    
    # Aggregate metrics
    avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
    avg_time = sum(r["response_time"] for r in results) / len(results)
    
    print(f"Average Accuracy: {avg_accuracy:.2%}")
    print(f"Average Response Time: {avg_time:.2f}s")
    
    return results
```

---

## Submission Checklist

- [ ] All 5 evaluation criteria implemented
- [ ] JSON trace logging for all queries
- [ ] Memory management with caching
- [ ] Error handling with fallback strategies
- [ ] Multimodal content processing (text + tables + figures)
- [ ] Complex query handling (Level 1-5)
- [ ] Performance benchmarks measured
- [ ] Test query sets evaluated
- [ ] Results documented in report
- [ ] Statistics and diagrams in reports/statistics/

---

**Related Documentation**:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [AGENT_SPECS.md](AGENT_SPECS.md) - Agent details
- [REPORT.md](../reports/REPORT.md) - Final submission report

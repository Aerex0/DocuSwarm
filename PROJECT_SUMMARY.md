# DocuSwarm Project Summary

## Overview

DocuSwarm is a multi-agent QA system for financial reports. It ingests PDF filings, extracts multimodal content (text + tables), indexes chunks in ChromaDB, and answers analytical queries through a routed LangGraph workflow.

The current validated setup uses Amazon 2022 10-K data and supports both single-query and batch-query execution.

## Current Implementation Status

### Implemented and Working

- End-to-end preprocessing pipeline:
  - PDF parsing via LlamaParse (with PyMuPDF fallback path in code)
  - Multimodal chunk generation
  - ChromaDB indexing with custom embedding adapter
- Multi-agent query pipeline:
  - Information Agent
  - Table Agent
  - Math Agent
  - Web Search Agent
  - Summarization Agent
  - Aggregator Agent
- LangGraph workflow with conditional routing
- CLI scripts:
  - `scripts/preprocess_documents.py`
  - `scripts/run_pipeline.py`
- JSON trace capture and batch output export

### Recently Resolved Runtime Issues

- Chroma embedding function compatibility with newer Chroma interface:
  - Added `__call__(input=...)`, `name()`, `embed_query()`, `embed_documents()`
- Missing utility exports:
  - Added `call_groq_llm` in `src/utils/groq_client.py`
  - Added `get_trace_logger` and `setup_logging` in `src/utils/logging_utils.py`
- LangGraph message coercion issue:
  - Removed message-specific annotations from non-message state fields
- Batch runner compatibility:
  - `scripts/run_pipeline.py` now supports grouped query JSON format directly

## Observed Results (from `output/batch_results.json`)

- Total queries executed: **21**
- Average confidence score: **~0.83**
- Runs with explicit errors: **0**
- Queries using table extraction: **21/21**
- Queries using math branch: **7/21**
- Queries using web search branch: **6/21**

Most frequent workflow patterns:

1. `information_agent -> table_agent -> aggregator_agent`
2. `information_agent -> table_agent -> math_agent -> web_search_agent -> aggregator_agent`

## Repository Reality Check

### Active, Relevant Directories

- `src/` - application code
- `scripts/` - runnable entry points
- `configs/` - runtime configs
- `examples/` - query sets
- `data/` - source docs + vector DB files
- `output/` - generated run outputs
- `docs/` - technical docs

## How to Run

### 1) Preprocess data

```bash
python scripts/preprocess_documents.py --input data/Amazon/AMAZON_2022_10K.pdf --reset
```

### 2) Run a single query

```bash
python scripts/run_pipeline.py --query "What was Amazon's total net sales in 2022?"
```

### 3) Run batch queries

```bash
python scripts/run_pipeline.py --batch examples/example_queries.json --verbose
```

## Practical Notes

- You do not need to rebuild chunks after every `git pull` unless you reset/delete ChromaDB or intentionally re-index with different chunking logic.
- If LlamaParse quota is exhausted, switch to:

```bash
python scripts/preprocess_documents.py --input data/Amazon --parser pymupdf --reset
```

## Next High-Impact Improvements

1. **Reduce repeated model load overhead** in embedding path by using a singleton/shared model instance.
2. **Tighten retrieval-to-math handoff quality** so fewer math queries depend on web fallback.
3. **Add compact evaluation script** to compute exact-match/contains metrics against `expected_answer_contains` in examples.
4. **Add tighter answer-format constraints** in aggregation prompts to reduce overly generic narrative responses.

# DocuSwarm: Multi-Agent Financial Report QA

DocuSwarm is a production-style, multi-agent question-answering system for long financial documents (annual reports, 10-K filings). It combines document parsing, multimodal chunking, vector retrieval, and agent orchestration to answer analytical queries with traceable execution.

> 📖 **Architecture & Workflow** — for a full walkthrough of the system design, data pipeline, agent graph, and component diagrams, open [`PROJECT.md`](PROJECT.md).

## What it does

- Parses PDF financial reports with LlamaParse
- Extracts and chunks text + table-heavy sections for retrieval
- Stores embeddings in ChromaDB for semantic and hybrid search
- Routes each query dynamically through specialized agents (retrieval → table → math → summarization → aggregation)
- Produces structured traces so every answer can be audited step-by-step

## Core stack

| Component | Technology |
|-----------|-----------|
| Multi-agent orchestration | LangGraph |
| LLM inference & embeddings | Groq (Llama 3.3 70B + Nomic Embed) |
| Vector storage & retrieval | ChromaDB |
| Document parsing | LlamaParse |
| Web search | Tavily (fallback: DuckDuckGo) |

## Repository structure

```text
.
├── .env.example
├── .gitignore
├── PROJECT.md
├── QUICKSTART.md
├── README.md
├── requirements.txt
├── setup.py
├── NLP.pdf
├── configs/
│   ├── agents.yaml
│   ├── chromadb.yaml
│   ├── groq.yaml
│   └── llamaparse.yaml
├── data/
│   ├── Amazon/
│   ├── cache/
│   ├── chromadb/
│   └── processed/
├── docs/
├── examples/
│   └── example_queries.json
├── output/
│   └── batch_results.json
├── reports/
│   └── REPORT.md
├── scripts/
│   ├── preprocess_documents.py
│   ├── run_pipeline.py
│   └── test_system.py
├── src/
│   ├── pipeline/
│   │   ├── orchestrator.py
│   │   └── query_handler.py
│   ├── task1_chunking/
│   │   ├── chunkers/
│   │   │   └── multimodal_chunker.py
│   │   ├── parsers/
│   │   │   └── llamaparse_handler.py
│   │   └── storage/
│   │       └── chromadb_manager.py
│   ├── task2_agents/
│   │   ├── agents/
│   │   │   ├── aggregator_agent.py
│   │   │   ├── information_agent.py
│   │   │   ├── math_agent.py
│   │   │   ├── summarization_agent.py
│   │   │   ├── table_agent.py
│   │   │   └── web_search_agent.py
│   │   └── core/
│   │       ├── langgraph_workflow.py
│   │       └── state_schema.py
│   └── utils/
│       ├── config.py
│       ├── groq_client.py
│       └── logging_utils.py
└── tests/
```

## Quick start

**1. Create and activate a virtual environment**

```bash
python -m venv .venv && source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment**

```bash
cp .env.example .env
# Fill in: GROQ_API_KEY, LLAMAPARSE_API_KEY, TAVILY_API_KEY
```

**4. Preprocess documents**

```bash
# Single PDF
python scripts/preprocess_documents.py --input data/Amazon/AMAZON_2022_10K.pdf --reset

# Entire directory
python scripts/preprocess_documents.py --input data/Amazon --reset
```

**5. Run queries**

```bash
# Single query
python scripts/run_pipeline.py --query "What was Amazon's total net sales in 2022?"

# Batch from JSON
python scripts/run_pipeline.py --batch examples/example_queries.json --verbose

# Interactive mode
python scripts/run_pipeline.py --interactive
```

## Pipeline status

Based on `output/batch_results.json` (21 queries):

| Metric | Value |
|--------|-------|
| Average confidence score | ~0.83 |
| Explicit errors | 0 |
| Queries using table extraction | 21 / 21 |
| Queries using web search | 6 / 21 |
| Queries using math agent | 7 / 21 |
| Most common path | `information_agent → table_agent → aggregator_agent` |

> Retrieval and table extraction are stable. Math and web-search branches activate correctly for comparative and calculation queries.

## Notes

- Parsing requires a `LLAMAPARSE_API_KEY` and consumes LlamaParse credits; a failed parse aborts that document rather than falling back.
- The first run may be slower due to model loading and caching.

## Development

```bash
# Run tests
pytest tests/

# Syntax check
python -m compileall src scripts

# Pull with local changes
git stash push -m "temp"
git pull origin main
git stash pop
```

## License

This project is licensed under the MIT License. See `LICENSE`.

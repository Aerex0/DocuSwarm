# DocuSwarm: Multi-Agent Financial Report QA

DocuSwarm is a production-style, multi-agent question-answering system for long financial documents (for example, annual reports and 10-K filings). It combines document parsing, multimodal chunking, vector retrieval, and agent orchestration to answer analytical queries with traceable execution.

## What it does

- Parses PDF financial reports with a primary/fallback parser strategy.
- Extracts and chunks text + table-heavy sections for retrieval.
- Stores embeddings in ChromaDB for semantic search.
- Routes each query through specialized agents (retrieval, table, math, web, summarization, aggregation).
- Produces structured traces so each answer can be audited.

## Core stack

- LangGraph: multi-agent orchestration and state transitions
- Groq: LLM inference for reasoning and response generation
- ChromaDB: vector storage and retrieval
- LlamaParse + PyMuPDF: document parsing (primary + fallback)
- SentenceTransformers (`all-MiniLM-L6-v2`): local embedding backend

## Repository structure

```text
.
├── configs/
├── data/
│   ├── Amazon/
│   ├── chromadb/
│   └── processed/
├── docs/
├── examples/
│   └── example_queries.json
├── output/
│   └── batch_results.json
├── scripts/
│   ├── preprocess_documents.py
│   └── run_pipeline.py
└── src/
    ├── task1_chunking/
    ├── task2_agents/
    ├── pipeline/
    └── utils/
```

## Quick start

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Configure environment

```bash
cp .env.example .env
# set: GROQ_API_KEY, LLAMAPARSE_API_KEY, TAVILY_API_KEY
```

3) Preprocess documents (single file or directory)

```bash
# Single PDF
python scripts/preprocess_documents.py --input data/Amazon/AMAZON_2022_10K.pdf --reset

# Or a directory
python scripts/preprocess_documents.py --input data/Amazon --reset
```

Optional parser switch:

```bash
python scripts/preprocess_documents.py --input data/Amazon --parser pymupdf --reset
```

4) Run queries

```bash
# Single query
python scripts/run_pipeline.py --query "What was Amazon's total net sales in 2022?"

# Batch from JSON
python scripts/run_pipeline.py --batch examples/example_queries.json --verbose

# Interactive mode
python scripts/run_pipeline.py --interactive
```

## Current observed project status

Based on `output/batch_results.json`:

- Total batch queries executed: 21
- Average confidence score: ~0.83
- Runs with explicit errors: 0
- Queries using table extraction: 21/21
- Queries using web search: 6/21
- Queries using math agent: 7/21
- Most common workflow: `information_agent -> table_agent -> aggregator_agent`

Interpretation:

- Retrieval/table extraction path is stable and frequently used.
- Math/web branches are active for comparative and calculation prompts.
- End-to-end pipeline runs successfully on batch mode.

## Notes and caveats

- If LlamaParse credits are exhausted, use `--parser pymupdf`.
- First run can be slower due to model caching/downloads.
- If `git pull` fails with local changes, stash first:

```bash
git stash push -m "temp"
git pull origin main
git stash pop
```

## Development

Useful commands:

```bash
pytest tests/
python -m compileall src scripts
```

## License

This repository currently has no explicit license file. Add one before external distribution.

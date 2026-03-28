# Setup & Installation Guide

Complete guide to setting up the multi-agent financial QA system.

---

## Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows with WSL
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: At least 2GB free space

---

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd prepathon-ps
```

### 2. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate venv
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- LangGraph & LangChain
- Groq SDK
- ChromaDB
- LlamaParse
- PyMuPDF
- All utilities

---

## API Keys Setup

### Required API Keys

1. **Groq API** (Required)
   - Sign up: https://console.groq.com
   - Navigate to API Keys
   - Create new API key
   - Free tier: 14,400 tokens/minute

2. **LlamaParse API** (Required for Task 1)
   - Sign up: https://llamaparse.ai
   - Get API key from dashboard
   - Free tier: 1,000 pages/day

3. **Tavily Search API** (Optional for web search)
   - Sign up: https://tavily.com
   - Get API key
   - Free tier: 1,000 requests/month

### Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env  # or vim, code, etc.
```

Add your API keys:
```bash
# Groq API (Required)
GROQ_API_KEY=your_groq_api_key_here

# LlamaParse API (Required)
LLAMAPARSE_API_KEY=your_llamaparse_api_key_here

# Web Search API (Optional)
TAVILY_API_KEY=your_tavily_api_key_here

# Paths
DATA_DIR=./data
CHROMADB_PATH=./data/chromadb
CACHE_DIR=./data/cache
```

---

## Task 1: Document Processing Setup

### Parse Financial Documents

```bash
# Process Amazon 10-K reports
python scripts/preprocess_documents.py \
  --input data/raw/Amazon \
  --output data/processed \
  --parser llamaparse \
  --chunk-strategy multimodal
```

**Options**:
- `--parser`: `llamaparse` (default) or `pymupdf`
- `--chunk-strategy`: `semantic`, `structure_aware`, or `multimodal`
- `--chunk-size`: Token size per chunk (default: 1024)
- `--chunk-overlap`: Overlap between chunks (default: 128)

**Expected Output**:
```
Processing AMAZON_2022_10K.pdf...
  ├─ Parsing with LlamaParse... ✓ (45s)
  ├─ Extracting tables... ✓ (23 tables found)
  ├─ Extracting figures... ✓ (12 figures found)
  ├─ Chunking document... ✓ (287 chunks created)
  └─ Saved to data/processed/AMAZON_2022_10K/

Total: 8 documents, 2,145 chunks
```

### Index in ChromaDB

```bash
# Create vector database index
python scripts/setup_chromadb.py \
  --chunks-dir data/processed \
  --collection-name financial_documents
```

**Expected Output**:
```
Initializing ChromaDB...
  ├─ Collection: financial_documents
  ├─ Embedding model: nomic-embed-text (Groq)
  └─ Storage: data/chromadb/

Indexing chunks...
  ├─ AMAZON_2015_10K: 245 chunks ✓
  ├─ AMAZON_2016_10K: 251 chunks ✓
  ├─ AMAZON_2017_10K: 268 chunks ✓
  ├─ AMAZON_2018_10K: 273 chunks ✓
  ├─ AMAZON_2019_10K: 281 chunks ✓
  ├─ AMAZON_2020_10K: 289 chunks ✓
  ├─ AMAZON_2021_10K: 275 chunks ✓
  └─ AMAZON_2022_10K: 287 chunks ✓

Total: 2,145 chunks indexed
Index size: 287 MB
```

---

## Task 2: Multi-Agent System Setup

### Test LangGraph Workflow

```bash
# Run test query
python scripts/run_pipeline.py \
  --query "What was Amazon's revenue in 2022?"
```

**Expected Output**:
```
Running query: What was Amazon's revenue in 2022?

Execution trace:
  1. information_agent (1.2s)
     └─ Retrieved 8 chunks → table_agent
  
  2. table_agent (0.8s)
     └─ Extracted revenue: $514.0B → aggregator_agent
  
  3. aggregator_agent (1.5s)
     └─ Final answer generated

Total time: 3.5s

Final Answer:
Amazon's total revenue in 2022 was $514.0 billion, representing a 9.4% 
increase from $469.8 billion in 2021. This includes $315.9B from North 
America, $118.0B from International, and $80.1B from AWS.

Confidence: 0.95
Sources: AMAZON_2022_10K.pdf (Table on page 45)
```

### Interactive Mode

```bash
# Start interactive query session
python scripts/run_pipeline.py --interactive
```

```
Multi-Agent Financial QA System
Type 'quit' to exit

> What was Amazon's revenue in 2022?
[Answer displayed]

> How does that compare to 2021?
[Uses conversation context]

> quit
Session saved to data/cache/session_123.json
```

---

## Configuration

### Agent Configuration

Edit `configs/agents.yaml`:

```yaml
information_agent:
  retrieval:
    top_k: 10
    similarity_threshold: 0.7
    use_hybrid_search: true
  
table_agent:
  extraction_method: "llm"  # or "rule_based"
  preserve_formatting: true

math_agent:
  precision: 2  # decimal places
  units_display: true

# ... more agents
```

### Groq Model Configuration

Edit `configs/groq.yaml`:

```yaml
models:
  reasoning: llama-3.3-70b-versatile
  embedding: nomic-embed-text
  fallback: llama-3.1-8b-instant

parameters:
  temperature: 0.1
  max_tokens: 8000
  top_p: 0.95

rate_limits:
  requests_per_minute: 30
  tokens_per_minute: 14400
```

### ChromaDB Configuration

Edit `configs/chromadb.yaml`:

```yaml
storage:
  path: ./data/chromadb
  collection_name: financial_documents

retrieval:
  distance_metric: cosine
  default_top_k: 10

embedding:
  model: nomic-embed-text
  dimension: 768
```

---

## Verification

### Test Installation

```bash
# Run system tests
pytest tests/

# Test specific component
pytest tests/test_agents.py -v

# Test with coverage
pytest --cov=src tests/
```

### Verify API Keys

```bash
# Test Groq API
python -c "
from langchain_groq import ChatGroq
llm = ChatGroq(model='llama-3.3-70b-versatile')
print(llm.invoke('Hello').content)
"

# Test LlamaParse API
python -c "
from llama_parse import LlamaParse
parser = LlamaParse()
print('LlamaParse API key valid')
"
```

---

## Troubleshooting

### Issue: ChromaDB Import Error

**Error**: `ModuleNotFoundError: No module named 'chromadb'`

**Solution**:
```bash
pip install chromadb --upgrade
```

### Issue: Groq API Rate Limit

**Error**: `groq.error.RateLimitError: Rate limit exceeded`

**Solution**:
- Wait 60 seconds and retry
- Reduce `requests_per_minute` in `configs/groq.yaml`
- Use fallback model: `llama-3.1-8b-instant`

### Issue: LlamaParse Credits Exhausted

**Error**: `LlamaParse: Free tier limit reached`

**Solution**:
- Use PyMuPDF fallback: `--parser pymupdf`
- Wait for daily reset (midnight UTC)
- Upgrade LlamaParse plan

### Issue: Out of Memory

**Error**: `MemoryError` or system crash

**Solution**:
- Reduce chunk size: `--chunk-size 512`
- Process fewer documents at once
- Close other applications
- Increase system RAM

### Issue: Slow Query Response

**Problem**: Queries taking >10 seconds

**Solution**:
1. Check ChromaDB index size: `du -sh data/chromadb`
2. Reduce retrieval top_k: `top_k: 5`
3. Enable caching in production
4. Use Groq's faster model for simple queries

---

## Development Setup

### Install Dev Dependencies

```bash
pip install -r requirements-dev.txt
```

This adds:
- pytest (testing)
- black (code formatting)
- ruff (linting)
- mypy (type checking)
- jupyter (notebooks)

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in notebooks/ directory
# - 01_document_parsing_test.ipynb
# - 02_chunking_experiments.ipynb
# - etc.
```

---

## Docker Setup (Optional)

### Build Image

```bash
docker build -t prepathon-financial-qa .
```

### Run Container

```bash
docker run -it \
  -v $(pwd)/data:/app/data \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  -e LLAMAPARSE_API_KEY=$LLAMAPARSE_API_KEY \
  prepathon-financial-qa
```

---

## Next Steps

After setup:
1. **Process Documents**: Run Task 1 document processing
2. **Test System**: Run example queries
3. **Explore Notebooks**: Experiment with chunking strategies
4. **Read Docs**: Review architecture and agent specs
5. **Customize**: Modify configs for your use case

---

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Check troubleshooting section
- **Contact**: Refer to README.md for PrepaTHON contacts

---

**Related Documentation**:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [TASK1_CHUNKING.md](TASK1_CHUNKING.md) - Document processing
- [TASK2_AGENTS.md](TASK2_AGENTS.md) - Multi-agent system

# Multi-Agent QA System for Financial Documents

**Inter IIT Tech Meet 14.0 - PrepaTHON 2025 | IIT(BHU)**

A hierarchical multi-agent system for answering complex queries on financial documents using LangGraph, Groq, and ChromaDB.

---

## Problem Statement

Financial documents (annual reports, earnings statements, regulatory filings) are long, complex, and multimodal. This project builds a system that can:

1. **Parse and structure financial documents** (Task 1)
   - Handle multimodal content (text, tables, figures)
   - Chunk documents while preserving context
   - Store chunks optimally for retrieval

2. **Answer complex queries using multi-agent coordination** (Task 2)
   - Dynamic agent hand-offs based on query requirements
   - Explainable reasoning with transparent logs
   - Memory management for efficient query handling

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Multi-Agent Framework** | LangGraph | Graph-based agent orchestration with state management |
| **LLM Provider** | Groq | Fast inference with Llama 3.3 70B |
| **Vector Database** | ChromaDB | Lightweight, efficient document storage & retrieval |
| **Document Parsing** | LlamaParse + PyMuPDF | Complex layout parsing with fallback support |
| **Embeddings** | Nomic Embed Text (via Groq) | Text embeddings for semantic search |

---

## Project Structure

```
prepathon-ps/
├── README.md                          # This file
├── docs/                              # Comprehensive documentation
│   ├── ARCHITECTURE.md                # System architecture & design
│   ├── TASK1_CHUNKING.md             # Document processing approach
│   ├── TASK2_AGENTS.md               # Multi-agent system design
│   ├── AGENT_SPECS.md                # Detailed agent specifications
│   ├── EVALUATION.md                 # Evaluation criteria & metrics
│   ├── SETUP.md                      # Installation & setup guide
│   └── API_COSTS.md                  # Cost tracking & optimization
│
├── src/                               # Source code
│   ├── task1_chunking/               # Document processing (Task 1)
│   │   ├── parsers/                  # PDF, table, figure parsing
│   │   ├── chunkers/                 # Chunking strategies
│   │   └── storage/                  # ChromaDB integration
│   ├── task2_agents/                 # Multi-agent system (Task 2)
│   │   ├── core/                     # LangGraph workflow
│   │   ├── agents/                   # Specialized agents
│   │   ├── tools/                    # Agent tools
│   │   └── prompts/                  # Agent prompts
│   ├── utils/                        # Utilities
│   └── pipeline/                     # End-to-end orchestration
│
├── configs/                           # Configuration files
│   ├── agents.yaml                   # Agent configurations
│   ├── groq.yaml                     # Groq LLM settings
│   ├── chromadb.yaml                 # Vector DB settings
│   └── llamaparse.yaml               # Document parsing settings
│
├── data/                              # Data directory
│   ├── raw/Amazon/                   # Raw 10-K reports (2015-2022)
│   ├── processed/                    # Processed chunks
│   ├── chromadb/                     # Vector database files
│   └── cache/                        # Cached results
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_document_parsing_test.ipynb
│   ├── 02_chunking_experiments.ipynb
│   ├── 03_chromadb_indexing.ipynb
│   ├── 04_langgraph_workflow.ipynb
│   └── 05_evaluation.ipynb
│
├── tests/                             # Test suite
├── examples/                          # Example queries & outputs
├── reports/                           # Final submission report
└── scripts/                           # Utility scripts
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd prepathon-ps

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
# - GROQ_API_KEY=your_groq_api_key
# - LLAMAPARSE_API_KEY=your_llamaparse_api_key
```

### 3. Process Documents (Task 1)

```bash
# Parse and chunk financial documents
python scripts/preprocess_documents.py --input data/raw/Amazon --output data/processed

# Index chunks in ChromaDB
python scripts/setup_chromadb.py --chunks data/processed/chunks
```

### 4. Run Query System (Task 2)

```bash
# Start the multi-agent query system
python scripts/run_pipeline.py --query "Compare YoY revenue growth between 2021 and 2022"
```

Or use Python API:

```python
from src.pipeline.orchestrator import QueryOrchestrator

# Initialize system
orchestrator = QueryOrchestrator()

# Run query
result = orchestrator.query(
    "Compare the YoY revenue growth and R&D spending between 2021 and 2022"
)

# View results
print(result["final_answer"])
print(result["trace"])  # Detailed execution log
```

---

## System Architecture

### Task 1: Document Processing Pipeline

```
Raw PDF → LlamaParse → Extract Text/Tables/Figures → Chunking → ChromaDB
              ↓
         PyMuPDF (fallback)
```

**Key Features:**
- Multimodal content extraction (text, tables, charts)
- Structure-aware chunking preserving context
- Hybrid retrieval (semantic + keyword search)

### Task 2: Multi-Agent Workflow (LangGraph)

```
User Query → Information Agent → Table Agent → Math Agent → Aggregator Agent → Final Answer
                    ↓                              ↓
            Web Search Agent                  Summarization Agent
```

**Agent Coordination:**
- Dynamic hand-offs based on query requirements
- Shared state across all agents
- Memory management for conversation history
- Transparent reasoning logs (JSON trace)

---

## Example Query

**Input:**
```
"Compare the YoY revenue growth and R&D spending between 2021 and 2022, 
and summarize the risks affecting future revenue."
```

**Output:**
```json
{
  "query": "Compare the YoY revenue growth...",
  "final_answer": "Amazon's revenue grew 21.7% YoY (2021→2022)...",
  "trace": [
    {
      "agent": "InformationAgent",
      "tool": "chromadb_search",
      "input": "revenue 2021 2022",
      "output": "Retrieved 5 relevant chunks",
      "handoff_to": "TableAgent"
    },
    {
      "agent": "TableAgent",
      "tool": "table_parser",
      "input": "Extract revenue values from table",
      "output": "2021: $469.8B, 2022: $514.0B",
      "handoff_to": "MathAgent"
    },
    ...
  ]
}
```

---

## Evaluation Criteria

The system is evaluated on:

1. **Pipeline Explainability** - Clear agent/tool contribution logs
2. **Memory Management** - Efficient caching and context reuse
3. **Error Handling** - Graceful fallback strategies
4. **Multimodal Reasoning** - Accurate processing of text, tables, figures
5. **Complex Query Handling** - Multi-hop reasoning across documents

See [docs/EVALUATION.md](docs/EVALUATION.md) for detailed metrics.

---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design & component interactions
- **[TASK1_CHUNKING.md](docs/TASK1_CHUNKING.md)** - Document processing strategy
- **[TASK2_AGENTS.md](docs/TASK2_AGENTS.md)** - Multi-agent system design
- **[AGENT_SPECS.md](docs/AGENT_SPECS.md)** - Detailed agent specifications
- **[EVALUATION.md](docs/EVALUATION.md)** - Evaluation metrics & benchmarks
- **[SETUP.md](docs/SETUP.md)** - Detailed installation guide
- **[API_COSTS.md](docs/API_COSTS.md)** - API usage & cost tracking

---

## Dataset

**Amazon 10-K Reports (2015-2022)**
- Located in `data/raw/Amazon/`
- 8 annual reports totaling ~1000 pages
- Includes financial tables, risk factors, MD&A sections

Additional datasets supported:
- FinanceBench
- Financial Q&A - 10k dataset

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_agents.py

# Run with coverage
pytest --cov=src tests/
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in notebooks/ directory
```

---

## Submission

This project is submitted for Inter IIT Tech Meet 14.0 PrepaTHON.

**Deliverables:**
- ✅ Complete codebase with documentation
- ✅ Detailed approach report in `reports/REPORT.md`
- ✅ Architecture diagrams in `reports/diagrams/`
- ✅ Performance statistics in `reports/statistics/`
- ✅ Usage instructions (this README)

---

## Contact

**PrepaTHON 2025 - IIT(BHU)**

- **CM:** Tejbir - 9034705165
- **Contact:** Bhaagyesh - 7428647019
- **WhatsApp Community:** [Join Link]

---

## License

This project is developed for Inter IIT Tech Meet 14.0 PrepaTHON competition.

---

## Acknowledgments

- **LangGraph** - Agent orchestration framework
- **Groq** - Fast LLM inference
- **ChromaDB** - Vector database
- **LlamaParse** - Document parsing
- **Inter IIT Tech Meet** - Competition organizers

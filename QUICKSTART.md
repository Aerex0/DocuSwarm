# QUICKSTART GUIDE

## Quick Setup and Testing

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or use a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Preprocess Documents (Task 1)

```bash
# Process Amazon 10-K reports and create ChromaDB index
python scripts/preprocess_documents.py --input data/Amazon --output data/processed

# Check ChromaDB stats
python -c "from src.task1_chunking.storage.chromadb_manager import ChromaDBManager; db = ChromaDBManager(); print(db.get_stats())"
```

Expected output:
- Documents parsed: 8 (Amazon 10-K reports 2015-2022)
- Chunks stored: ~500-1000 (depends on document size)
- Tables extracted: ~100-200

### 3. Run Single Query (Task 2)

```bash
# Test with a simple query
python scripts/run_pipeline.py --query "What was Amazon's total revenue in 2022?"

# With verbose output (shows agent trace)
python scripts/run_pipeline.py --query "Calculate the YoY revenue growth from 2021 to 2022" --verbose
```

### 4. Run Batch Queries

```bash
# Process all example queries
python scripts/run_pipeline.py --batch examples/example_queries.json --verbose

# Results will be saved to: output/batch_results.json
```

### 5. Interactive Mode

```bash
# Launch interactive terminal
python scripts/run_pipeline.py --interactive

# Commands:
# - Enter any query to process
# - Type 'verbose on' to see agent traces
# - Type 'verbose off' to disable traces
# - Type 'exit', 'quit', or 'q' to exit
```

---

## Example Queries to Test

### Financial Metrics
- "What was Amazon's total revenue in 2022?"
- "What were Amazon's operating expenses in 2021?"
- "Show me Amazon's cash flow from operations"

### Calculations & Comparisons
- "Calculate the YoY revenue growth from 2021 to 2022"
- "What is the percentage change in net income from 2020 to 2021?"
- "Compare R&D expenses between 2019 and 2022"

### Risk Assessment
- "What are the main risk factors mentioned in the 2022 10-K?"
- "Describe Amazon's competitive risks"
- "What regulatory challenges does Amazon face?"

### Business Strategy
- "Summarize Amazon's acquisition strategy"
- "What are Amazon's key growth initiatives?"
- "Describe Amazon's international expansion plans"

### Complex Multi-Step Queries
- "How has Amazon's AWS revenue grown over the last 3 years and what percentage of total revenue does it represent?"
- "What were the major factors affecting Amazon's profitability in 2021 and 2022?"

---

## Verify System Components

### Check Agent Implementations

```bash
# List all implemented agents
ls -la src/task2_agents/agents/

# Should show:
# - information_agent.py
# - table_agent.py
# - math_agent.py
# - web_search_agent.py
# - summarization_agent.py
# - aggregator_agent.py
```

### Check Workflow

```python
# Verify LangGraph workflow
from src.task2_agents.core.langgraph_workflow import get_default_workflow

workflow = get_default_workflow()
print("Workflow compiled successfully!")
print(f"Nodes: {workflow.get_graph().nodes}")
```

### Check ChromaDB Collection

```python
from src.task1_chunking.storage.chromadb_manager import ChromaDBManager

db = ChromaDBManager()
stats = db.get_stats()

print(f"Collection Name: {stats['collection_name']}")
print(f"Total Documents: {stats['total_documents']}")
print(f"Embedding Model: {stats['embedding_model']}")
```

---

## Troubleshooting

### Import Errors (LSP warnings are expected before installation)

If you see import errors like "chromadb could not be resolved", run:
```bash
pip install -r requirements.txt
```

### API Key Errors

Verify API keys are set in `.env`:
```bash
cat .env

# Should show:
# GROQ_API_KEY=gsk_...
# LLAMA_CLOUD_API_KEY=llx_...
# TAVILY_API_KEY=tvly-dev-...
```

### ChromaDB Collection Not Found

Run preprocessing first:
```bash
python scripts/preprocess_documents.py --input data/Amazon
```

### Memory Issues

If running out of memory during document processing:
- Process documents one at a time
- Reduce chunk size in `configs/chunking_config.yaml`
- Use smaller batch sizes

---

## Expected Agent Workflow Examples

### Example 1: Simple Retrieval
Query: "What was Amazon's revenue in 2022?"

Workflow:
```
information_agent → summarization_agent → aggregator_agent
```

### Example 2: Calculation with Tables
Query: "Calculate YoY revenue growth from 2021 to 2022"

Workflow:
```
information_agent → table_agent → math_agent → aggregator_agent
```

### Example 3: Complex Multi-Step
Query: "Compare AWS revenue growth across 2020-2022 and explain key drivers"

Workflow:
```
information_agent → table_agent → math_agent → web_search_agent → summarization_agent → aggregator_agent
```

---

## Performance Benchmarks (Expected)

- **Document Preprocessing**: ~2-5 minutes for 8 Amazon 10-K PDFs
- **Single Query Processing**: ~5-15 seconds (depends on complexity)
- **Batch Processing (20 queries)**: ~3-5 minutes

---

## Next Steps After Testing

1. **Evaluate Results**: Check `output/batch_results.json` for accuracy
2. **Review Traces**: Examine `logs/traces/` for agent execution logs
3. **Tune Parameters**: Adjust chunk sizes, retrieval top_k in configs
4. **Add More Documents**: Process additional 10-K reports from other companies
5. **Optimize Performance**: Fine-tune agent routing logic

---

## File Structure Overview

```
prepathon-ps/
├── src/
│   ├── task1_chunking/          # Document processing (Task 1)
│   │   ├── parsers/             # LlamaParse + PyMuPDF
│   │   ├── chunkers/            # Multimodal chunking
│   │   └── storage/             # ChromaDB manager
│   ├── task2_agents/            # Multi-agent system (Task 2)
│   │   ├── agents/              # 6 specialized agents
│   │   └── core/                # LangGraph workflow
│   ├── pipeline/                # Query orchestrator
│   └── utils/                   # Groq client, config, logging
├── scripts/
│   ├── preprocess_documents.py  # Task 1 pipeline
│   └── run_pipeline.py          # Task 2 pipeline
├── configs/                     # YAML configurations
├── data/                        # Input documents
├── logs/                        # Execution logs & traces
└── output/                      # Results & reports
```

---

For detailed information, see `docs/` directory.

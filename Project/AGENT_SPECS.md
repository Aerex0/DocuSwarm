# Agent Specifications

Detailed specifications for each agent in the multi-agent system, including responsibilities, tools, prompts, and hand-off conditions.

---

## Table of Contents

1. [Information Agent](#information-agent)
2. [Table Agent](#table-agent)
3. [Math Agent](#math-agent)
4. [Web Search Agent](#web-search-agent)
5. [Summarization Agent](#summarization-agent)
6. [Aggregator Agent](#aggregator-agent)

---

## Information Agent

### Purpose
Primary retrieval agent responsible for fetching relevant document chunks from ChromaDB based on user queries.

### Input State Fields
- `query` (str): User's query
- `conversation_history` (List[Dict]): Previous queries in session
- `cached_results` (Dict): Cached query results

### Output State Updates
- `retrieved_chunks` (List[Dict]): List of relevant document chunks
- `next_agent` (str): Next agent to execute
- `trace` (List[Dict]): Updated execution log

### Tools Available

#### 1. ChromaDB Vector Search
```python
Tool(
    name="chromadb_search",
    description="Semantic search over financial document chunks",
    parameters={
        "query": str,
        "top_k": int,
        "filters": Dict  # e.g., {"year": 2022, "section": "Risk Factors"}
    }
)
```

#### 2. Keyword Search (BM25)
```python
Tool(
    name="keyword_search",
    description="Exact keyword matching for specific terms",
    parameters={
        "query": str,
        "top_k": int
    }
)
```

#### 3. Hybrid Search
```python
Tool(
    name="hybrid_search",
    description="Combine semantic and keyword search",
    parameters={
        "query": str,
        "semantic_weight": float,  # 0.0-1.0
        "top_k": int
    }
)
```

### Prompt Template

```python
INFORMATION_AGENT_PROMPT = """
You are an Information Retrieval Agent for a financial document QA system.

Your role:
1. Analyze the user query to understand information needs
2. Retrieve the most relevant document chunks from ChromaDB
3. Determine which specialized agent should process the results next

Query: {query}

Available tools:
- chromadb_search: Semantic similarity search
- keyword_search: Exact term matching
- hybrid_search: Combined approach

Instructions:
- For specific financial metrics (e.g., "revenue 2022"), use keyword search
- For conceptual queries (e.g., "business strategy"), use semantic search
- Retrieve 5-10 chunks to provide sufficient context
- Apply filters when query mentions specific years or sections

After retrieval, analyze the chunks to determine the next agent:
- If tables found → TableAgent
- If calculation needed → MathAgent
- If insufficient information → WebSearchAgent
- Otherwise → SummarizationAgent

Return:
{{
  "retrieved_chunks": [...],
  "next_agent": "agent_name",
  "reasoning": "explanation"
}}
"""
```

### Hand-off Conditions

| Condition | Next Agent | Reasoning |
|-----------|-----------|-----------|
| Retrieved < 3 chunks | WebSearchAgent | Insufficient internal information |
| Chunks contain tables | TableAgent | Structured data needs extraction |
| Query requires calculation | MathAgent | Mathematical operations needed |
| Query is conceptual | SummarizationAgent | Text-based answer sufficient |

### Error Handling

- **ChromaDB unavailable**: Fallback to cached results or web search
- **No results found**: Hand-off to WebSearchAgent
- **Timeout**: Return partial results and continue

### Example Execution

**Input State**:
```json
{
  "query": "What was Amazon's revenue in 2022?",
  "conversation_history": []
}
```

**Actions**:
1. Use `keyword_search` with filters: `{"year": 2022, "content_type": "table"}`
2. Retrieve 8 chunks containing revenue data
3. Detect tables in chunks
4. Decide next_agent = "table_agent"

**Output State**:
```json
{
  "retrieved_chunks": [
    {
      "chunk_id": "abc123",
      "text": "...",
      "content_type": "table",
      "metadata": {...}
    },
    ...
  ],
  "next_agent": "table_agent",
  "trace": [{
    "agent": "information_agent",
    "tool": "keyword_search",
    "input": "revenue 2022",
    "output": "8 chunks retrieved",
    "handoff_to": "table_agent"
  }]
}
```

---

## Table Agent

### Purpose
Extract and structure data from financial tables found in retrieved chunks.

### Input State Fields
- `query` (str): User's query
- `retrieved_chunks` (List[Dict]): Chunks from InformationAgent
- `extracted_tables` (List[Dict]): Previously extracted tables (if any)

### Output State Updates
- `extracted_tables` (List[Dict]): Structured table data
- `next_agent` (str): Next agent to execute

### Tools Available

#### 1. Table Parser
```python
Tool(
    name="table_parser",
    description="Parse table markdown/JSON into structured format",
    parameters={
        "table_data": Union[str, Dict],
        "extract_columns": List[str]
    }
)
```

#### 2. Column Selector
```python
Tool(
    name="column_selector",
    description="Extract specific columns based on query",
    parameters={
        "table": Dict,
        "query": str,
        "columns": List[str]
    }
)
```

#### 3. LLM Table Extraction
```python
Tool(
    name="llm_extract_table",
    description="Use LLM to extract relevant data from complex tables",
    parameters={
        "table_markdown": str,
        "query": str
    }
)
```

### Prompt Template

```python
TABLE_AGENT_PROMPT = """
You are a Table Processing Agent for a financial document QA system.

Your role:
1. Identify tables in the retrieved chunks
2. Extract relevant data based on the user query
3. Structure the data for downstream processing

Query: {query}

Retrieved Tables:
{tables}

Instructions:
- Parse each table to understand its structure (headers, rows, data types)
- Extract only the data relevant to answering the query
- Preserve numerical precision (don't round unnecessarily)
- Note any missing data or inconsistencies
- If calculation is needed, hand-off to MathAgent
- Otherwise, hand-off to AggregatorAgent

Return structured data as:
{{
  "extracted_tables": [
    {{
      "table_id": "...",
      "title": "...",
      "relevant_data": {{...}},
      "columns": [...],
      "rows": [...]
    }}
  ],
  "next_agent": "math_agent" or "aggregator_agent"
}}
"""
```

### Hand-off Conditions

| Condition | Next Agent | Reasoning |
|-----------|-----------|-----------|
| Query requires math operations | MathAgent | Need calculations on extracted data |
| Simple data extraction | AggregatorAgent | Ready for final answer |
| No tables found | AggregatorAgent | Skip table processing |

### Error Handling

- **Malformed table**: Attempt parsing with multiple strategies
- **Missing columns**: Report missing data in output
- **LLM extraction fails**: Return raw table with warning

### Example Execution

**Input State**:
```json
{
  "query": "Compare revenue between 2021 and 2022",
  "retrieved_chunks": [
    {
      "content_type": "table",
      "table_markdown": "| Year | Revenue |\n|------|--------|\n| 2021 | $469.8B |\n| 2022 | $514.0B |"
    }
  ]
}
```

**Output State**:
```json
{
  "extracted_tables": [
    {
      "table_id": "revenue_table",
      "relevant_data": {
        "2021": {"revenue": 469.8},
        "2022": {"revenue": 514.0}
      },
      "units": "billions USD"
    }
  ],
  "next_agent": "math_agent"
}
```

---

## Math Agent

### Purpose
Perform calculations, statistical analysis, and financial metrics computation.

### Input State Fields
- `query` (str): User's query
- `extracted_tables` (List[Dict]): Structured table data
- `retrieved_chunks` (List[Dict]): Additional context

### Output State Updates
- `calculated_values` (Dict): Computation results
- `next_agent` (str): Always "aggregator_agent"

### Tools Available

#### 1. Calculator
```python
Tool(
    name="calculator",
    description="Basic arithmetic operations",
    parameters={
        "expression": str  # e.g., "(514.0 - 469.8) / 469.8"
    }
)
```

#### 2. YoY Growth Calculator
```python
Tool(
    name="yoy_growth",
    description="Calculate year-over-year growth rate",
    parameters={
        "current_value": float,
        "previous_value": float
    }
)
```

#### 3. Percentage Change
```python
Tool(
    name="percentage_change",
    description="Calculate percentage change between values",
    parameters={
        "values": List[float]
    }
)
```

#### 4. Financial Ratios
```python
Tool(
    name="financial_ratio",
    description="Calculate common financial ratios",
    parameters={
        "ratio_type": str,  # "P/E", "ROE", "debt_to_equity", etc.
        "numerator": float,
        "denominator": float
    }
)
```

#### 5. Statistical Analysis
```python
Tool(
    name="stats_analysis",
    description="Mean, median, std dev, trend analysis",
    parameters={
        "values": List[float],
        "analysis_type": str  # "mean", "median", "trend", etc.
    }
)
```

### Prompt Template

```python
MATH_AGENT_PROMPT = """
You are a Math & Analysis Agent for a financial document QA system.

Your role:
1. Identify required calculations from the query
2. Execute mathematical operations on extracted data
3. Provide accurate, well-formatted results

Query: {query}

Extracted Data:
{extracted_tables}

Available operations:
- YoY growth rate
- Percentage change
- Financial ratios (P/E, ROE, margins, etc.)
- Statistical analysis (mean, trend, etc.)
- Custom calculations

Instructions:
- Identify all calculations needed to answer the query
- Preserve precision (use at least 2 decimal places)
- Show intermediate steps for complex calculations
- Include units in results (e.g., "$B", "percentage points")
- Always hand-off to AggregatorAgent when done

Return:
{{
  "calculated_values": {{
    "calculation_1": {{
      "result": value,
      "formula": "...",
      "units": "..."
    }},
    ...
  }},
  "next_agent": "aggregator_agent"
}}
"""
```

### Hand-off Conditions

| Condition | Next Agent | Reasoning |
|-----------|-----------|-----------|
| Calculations complete | AggregatorAgent | Ready for final answer compilation |

### Error Handling

- **Missing data**: Report which values are missing
- **Division by zero**: Return error with explanation
- **Invalid operation**: Suggest alternative approach

### Example Execution

**Input State**:
```json
{
  "query": "Calculate YoY revenue growth from 2021 to 2022",
  "extracted_tables": [{
    "relevant_data": {
      "2021": {"revenue": 469.8},
      "2022": {"revenue": 514.0}
    }
  }]
}
```

**Output State**:
```json
{
  "calculated_values": {
    "yoy_revenue_growth": {
      "result": 9.41,
      "formula": "(514.0 - 469.8) / 469.8 * 100",
      "units": "percent",
      "interpretation": "Revenue increased by 9.41% from 2021 to 2022"
    }
  },
  "next_agent": "aggregator_agent"
}
```

---

## Web Search Agent

### Purpose
Retrieve external information when internal documents are insufficient.

### Input State Fields
- `query` (str): User's query
- `retrieved_chunks` (List[Dict]): Internal retrieval results

### Output State Updates
- `web_search_results` (List[Dict]): External search results
- `next_agent` (str): Always "aggregator_agent"

### Tools Available

#### 1. Tavily Search (Primary)
```python
Tool(
    name="tavily_search",
    description="AI-powered web search optimized for accuracy",
    parameters={
        "query": str,
        "max_results": int,
        "search_depth": str  # "basic" or "advanced"
    }
)
```

#### 2. DuckDuckGo Search (Fallback)
```python
Tool(
    name="duckduckgo_search",
    description="Privacy-focused web search",
    parameters={
        "query": str,
        "max_results": int
    }
)
```

### Prompt Template

```python
WEB_SEARCH_AGENT_PROMPT = """
You are a Web Search Agent for a financial document QA system.

Your role:
1. Identify what external information is needed
2. Formulate effective search queries
3. Retrieve and validate web results

Original Query: {query}

Internal Search Results: {retrieved_chunks}

Why web search is needed:
{search_reason}

Instructions:
- Formulate search query to find missing information
- Prioritize authoritative sources (SEC, company websites, financial news)
- Retrieve 3-5 relevant results
- Validate information credibility
- Always hand-off to AggregatorAgent

Return:
{{
  "web_search_results": [
    {{
      "url": "...",
      "title": "...",
      "content": "...",
      "source_credibility": "high/medium/low"
    }}
  ],
  "next_agent": "aggregator_agent"
}}
"""
```

### Hand-off Conditions

| Condition | Next Agent | Reasoning |
|-----------|-----------|-----------|
| Search complete | AggregatorAgent | External data ready for integration |

### Error Handling

- **API rate limit**: Wait and retry or use fallback search
- **No results**: Return empty list with explanation
- **Connection error**: Skip web search, use only internal data

---

## Summarization Agent

### Purpose
Condense long text passages into concise summaries focused on the query.

### Input State Fields
- `query` (str): User's query
- `retrieved_chunks` (List[Dict]): Text chunks to summarize

### Output State Updates
- `summary` (str): Concise summary
- `next_agent` (str): Always "aggregator_agent"

### Tools Available

#### 1. LLM Summarization
```python
Tool(
    name="llm_summarize",
    description="Use LLM to generate focused summary",
    parameters={
        "text": str,
        "query": str,
        "max_length": int  # words
    }
)
```

#### 2. Extractive Summarization
```python
Tool(
    name="extractive_summary",
    description="Extract key sentences",
    parameters={
        "text": str,
        "num_sentences": int
    }
)
```

### Prompt Template

```python
SUMMARIZATION_AGENT_PROMPT = """
You are a Summarization Agent for a financial document QA system.

Your role:
1. Read retrieved text chunks
2. Extract information relevant to the query
3. Generate a concise, accurate summary

Query: {query}

Text Chunks:
{text_chunks}

Instructions:
- Focus only on information relevant to the query
- Preserve important facts, numbers, and dates
- Keep summary to 200-300 words
- Maintain original meaning and context
- Cite specific document sections when possible
- Always hand-off to AggregatorAgent

Return:
{{
  "summary": "...",
  "key_points": [...],
  "next_agent": "aggregator_agent"
}}
"""
```

### Hand-off Conditions

| Condition | Next Agent | Reasoning |
|-----------|-----------|-----------|
| Summary complete | AggregatorAgent | Ready for final answer |

---

## Aggregator Agent

### Purpose
Compile final answer from all agent results and generate comprehensive response.

### Input State Fields
- `query` (str): Original user query
- `retrieved_chunks` (List[Dict]): All retrieved chunks
- `extracted_tables` (List[Dict]): Extracted table data
- `calculated_values` (Dict): Math results
- `summary` (str): Text summary
- `web_search_results` (List[Dict]): External data

### Output State Updates
- `final_answer` (str): Comprehensive answer
- `confidence_score` (float): Answer confidence (0-1)
- `next_agent` (None): End of workflow

### Tools Available

#### 1. LLM Answer Generation
```python
Tool(
    name="llm_answer_generation",
    description="Generate comprehensive answer from all sources",
    parameters={
        "query": str,
        "all_data": Dict
    }
)
```

### Prompt Template

```python
AGGREGATOR_AGENT_PROMPT = """
You are the Aggregator Agent - the final agent responsible for compiling a comprehensive answer.

Query: {query}

Available Information:
1. Retrieved Chunks: {retrieved_chunks}
2. Extracted Tables: {extracted_tables}
3. Calculated Values: {calculated_values}
4. Text Summary: {summary}
5. Web Search Results: {web_search_results}

Agent Execution Trace:
{trace}

Your role:
1. Synthesize all information from previous agents
2. Generate a clear, accurate, comprehensive answer
3. Cite specific sources and data points
4. Note any limitations or uncertainties
5. Provide confidence score

Answer Format:
- Direct answer to the query (1-2 sentences)
- Supporting details with citations
- Relevant numbers and calculations
- Any caveats or limitations

Return:
{{
  "final_answer": "...",
  "confidence_score": 0.0-1.0,
  "sources_cited": [...],
  "limitations": "..."
}}
"""
```

### Hand-off Conditions

| Condition | Next Agent | Reasoning |
|-----------|-----------|-----------|
| Answer complete | END | Workflow finished |

### Example Output

```json
{
  "final_answer": "Amazon's revenue grew from $469.8 billion in 2021 to $514.0 billion in 2022, representing a year-over-year growth rate of 9.41%. This was driven primarily by North America segment growth of 13% and AWS growth of 29%, partially offset by a 8% decline in International segment revenue. The growth rate represents a deceleration from 2021's 21.7% YoY growth.",
  "confidence_score": 0.95,
  "sources_cited": [
    "AMAZON_2022_10K.pdf, Page 45, Table: Consolidated Net Sales",
    "AMAZON_2021_10K.pdf, Page 42, MD&A Section"
  ],
  "limitations": "None - all data extracted from official 10-K filings"
}
```

---

## Summary Table

| Agent | Primary Tool | Input | Output | Next Agent(s) |
|-------|--------------|-------|--------|---------------|
| Information | ChromaDB | Query | Chunks | Table, Math, Web, Summarization |
| Table | LLM Parser | Chunks | Structured data | Math, Aggregator |
| Math | Calculator | Tables | Calculations | Aggregator |
| Web Search | Tavily | Query | Web results | Aggregator |
| Summarization | LLM | Chunks | Summary | Aggregator |
| Aggregator | LLM | All results | Final answer | END |

---

**Related Documentation**:
- [TASK2_AGENTS.md](TASK2_AGENTS.md) - Multi-agent workflow
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design

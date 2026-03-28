# Task 2: Dynamic Multi-Agent Query Answering

This document details the design and implementation of the multi-agent system for handling complex queries on financial documents using LangGraph.

---

## Table of Contents

1. [Overview](#overview)
2. [LangGraph Workflow Design](#langgraph-workflow-design)
3. [Agent Architecture](#agent-architecture)
4. [State Management](#state-management)
5. [Hand-off Mechanism](#hand-off-mechanism)
6. [Memory Management](#memory-management)
7. [Tool Integration](#tool-integration)
8. [Execution Flow Examples](#execution-flow-examples)
9. [Implementation Guide](#implementation-guide)

---

## Overview

**Goal**: Build a dynamic multi-agent system that can:
- Handle unpredictable, user-driven queries
- Coordinate specialized agents through hand-offs
- Maintain conversation history and context
- Generate transparent reasoning logs

**Key Requirements**:
1. **Dynamic routing**: Agents decide which agent to hand-off to
2. **Shared state**: All agents access common context
3. **Memory management**: Store conversation history for query reuse
4. **Explainability**: Log all agent actions and hand-offs

---

## LangGraph Workflow Design

### What is LangGraph?

LangGraph is a framework for building stateful, multi-agent applications as directed graphs:
- **Nodes** = Agent functions (process state)
- **Edges** = Transitions between agents (conditional or direct)
- **State** = Shared context (passed through all nodes)

**Why LangGraph for this task?**
- Native support for conditional routing (agent hand-offs)
- Built-in state management and checkpointing
- Visualization and debugging tools
- Easy to trace execution flow

### State Graph Architecture

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

### Node Definitions

**6 Specialized Agent Nodes**:

1. **Information Agent**: Retrieves relevant chunks from ChromaDB
2. **Table Agent**: Processes and extracts data from tables
3. **Math Agent**: Performs calculations and statistical analysis
4. **Web Search Agent**: Fetches external information (if needed)
5. **Summarization Agent**: Condenses long text passages
6. **Aggregator Agent**: Compiles final answer from all results

---

## Agent Architecture

### Base Agent Pattern

All agents follow a consistent pattern:

```python
from typing import TypedDict
from langgraph.graph import StateGraph

def agent_function(state: AgentState) -> AgentState:
    """
    Agent function pattern:
    1. Read current state
    2. Execute specialized task using tools
    3. Update state with results
    4. Determine next agent (hand-off)
    5. Log action to trace
    6. Return updated state
    """
    
    # 1. Read state
    query = state["query"]
    context = state.get("context", {})
    
    # 2. Execute task
    result = execute_agent_task(query, context)
    
    # 3. Update state
    state["agent_results"][agent_name] = result
    state["current_agent"] = agent_name
    
    # 4. Determine next agent
    next_agent = decide_next_agent(state, result)
    state["next_agent"] = next_agent
    
    # 5. Log to trace
    state["trace"].append({
        "agent": agent_name,
        "tool": tool_used,
        "input": input_data,
        "output": result_summary,
        "handoff_to": next_agent,
        "timestamp": datetime.now().isoformat()
    })
    
    # 6. Return updated state
    return state
```

### Agent Specializations

#### 1. Information Agent

**Role**: Primary retrieval from vector database

```python
def information_agent(state: AgentState) -> AgentState:
    """Retrieve relevant document chunks"""
    
    query = state["query"]
    
    # Use ChromaDB retrieval tool
    retriever = ChromaDBRetriever(collection_name="financial_documents")
    chunks = retriever.retrieve(
        query=query,
        top_k=10,
        filters=extract_filters_from_query(query)
    )
    
    # Update state
    state["retrieved_chunks"] = chunks
    state["current_agent"] = "information_agent"
    
    # Analyze retrieved content to determine next agent
    has_tables = any(c["content_type"] == "table" for c in chunks)
    has_numbers = contains_numerical_data(chunks)
    insufficient_info = len(chunks) < 3
    
    # Routing logic
    if insufficient_info:
        next_agent = "web_search_agent"
    elif has_tables:
        next_agent = "table_agent"
    elif has_numbers and is_calculation_query(query):
        next_agent = "math_agent"
    else:
        next_agent = "summarization_agent"
    
    state["next_agent"] = next_agent
    
    # Log
    state["trace"].append({
        "agent": "information_agent",
        "tool": "chromadb_retrieval",
        "input": query,
        "output": f"Retrieved {len(chunks)} chunks",
        "handoff_to": next_agent
    })
    
    return state
```

#### 2. Table Agent

**Role**: Extract and structure data from tables

```python
def table_agent(state: AgentState) -> AgentState:
    """Process table data"""
    
    chunks = state["retrieved_chunks"]
    query = state["query"]
    
    # Extract tables from chunks
    tables = [c for c in chunks if c["content_type"] == "table"]
    
    # Use LLM to extract relevant data
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    
    extracted_data = []
    for table in tables:
        prompt = f"""
        Query: {query}
        
        Table:
        {table["table_markdown"]}
        
        Extract the relevant data from this table to answer the query.
        Return as structured JSON.
        """
        
        response = llm.invoke(prompt)
        extracted_data.append(json.loads(response.content))
    
    # Update state
    state["extracted_tables"] = extracted_data
    state["current_agent"] = "table_agent"
    
    # Determine if calculations needed
    if requires_calculation(query):
        state["next_agent"] = "math_agent"
    else:
        state["next_agent"] = "aggregator_agent"
    
    # Log
    state["trace"].append({
        "agent": "table_agent",
        "tool": "table_extraction",
        "input": f"{len(tables)} tables",
        "output": f"Extracted {len(extracted_data)} data points",
        "handoff_to": state["next_agent"]
    })
    
    return state
```

#### 3. Math Agent

**Role**: Perform calculations and statistical analysis

```python
def math_agent(state: AgentState) -> AgentState:
    """Perform mathematical computations"""
    
    query = state["query"]
    extracted_tables = state.get("extracted_tables", [])
    
    # Parse calculation requirements from query
    calculation_type = identify_calculation_type(query)
    
    results = {}
    
    if calculation_type == "yoy_growth":
        # Calculate year-over-year growth
        for table in extracted_tables:
            if "years" in table and "values" in table:
                growth = calculate_yoy_growth(
                    table["years"],
                    table["values"]
                )
                results["yoy_growth"] = growth
    
    elif calculation_type == "percentage_change":
        # Calculate percentage change
        values = extract_numerical_values(extracted_tables)
        pct_change = calculate_percentage_change(values)
        results["percentage_change"] = pct_change
    
    elif calculation_type == "ratio":
        # Calculate financial ratios
        numerator = extract_value(extracted_tables, "numerator_field")
        denominator = extract_value(extracted_tables, "denominator_field")
        results["ratio"] = numerator / denominator
    
    # Update state
    state["calculated_values"] = results
    state["current_agent"] = "math_agent"
    state["next_agent"] = "aggregator_agent"
    
    # Log
    state["trace"].append({
        "agent": "math_agent",
        "tool": "calculator",
        "input": calculation_type,
        "output": str(results),
        "handoff_to": "aggregator_agent"
    })
    
    return state
```

#### 4. Web Search Agent

**Role**: Retrieve external information when needed

```python
def web_search_agent(state: AgentState) -> AgentState:
    """Search web for additional information"""
    
    query = state["query"]
    
    # Use web search tool (Tavily/DuckDuckGo)
    search_tool = TavilySearchTool()
    search_results = search_tool.search(
        query=query,
        max_results=5
    )
    
    # Update state
    state["web_search_results"] = search_results
    state["current_agent"] = "web_search_agent"
    state["next_agent"] = "aggregator_agent"
    
    # Log
    state["trace"].append({
        "agent": "web_search_agent",
        "tool": "tavily_search",
        "input": query,
        "output": f"Found {len(search_results)} results",
        "handoff_to": "aggregator_agent"
    })
    
    return state
```

#### 5. Summarization Agent

**Role**: Condense long text passages

```python
def summarization_agent(state: AgentState) -> AgentState:
    """Summarize retrieved text"""
    
    chunks = state["retrieved_chunks"]
    query = state["query"]
    
    # Combine relevant text
    text_chunks = [c["text"] for c in chunks if c["content_type"] == "text"]
    combined_text = "\n\n".join(text_chunks)
    
    # Use LLM to summarize
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    
    prompt = f"""
    Query: {query}
    
    Text:
    {combined_text[:4000]}  # Limit to context window
    
    Provide a concise summary focused on answering the query.
    """
    
    summary = llm.invoke(prompt).content
    
    # Update state
    state["summary"] = summary
    state["current_agent"] = "summarization_agent"
    state["next_agent"] = "aggregator_agent"
    
    # Log
    state["trace"].append({
        "agent": "summarization_agent",
        "tool": "llm_summarization",
        "input": f"{len(combined_text)} characters",
        "output": summary[:200] + "...",
        "handoff_to": "aggregator_agent"
    })
    
    return state
```

#### 6. Aggregator Agent

**Role**: Compile final answer from all agent results

```python
def aggregator_agent(state: AgentState) -> AgentState:
    """Aggregate results and generate final answer"""
    
    query = state["query"]
    
    # Gather all results
    retrieved_chunks = state.get("retrieved_chunks", [])
    extracted_tables = state.get("extracted_tables", [])
    calculated_values = state.get("calculated_values", {})
    summary = state.get("summary", "")
    web_results = state.get("web_search_results", [])
    
    # Build comprehensive prompt
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    
    prompt = f"""
    Query: {query}
    
    Available Information:
    
    1. Retrieved Chunks: {len(retrieved_chunks)} chunks from financial documents
    2. Extracted Tables: {extracted_tables}
    3. Calculated Values: {calculated_values}
    4. Summary: {summary}
    5. Web Results: {web_results}
    
    Based on all this information, provide a comprehensive, accurate answer to the query.
    
    Format your response as:
    - Clear, concise answer
    - Cite specific numbers and sources
    - Mention any limitations or uncertainties
    """
    
    final_answer = llm.invoke(prompt).content
    
    # Update state
    state["final_answer"] = final_answer
    state["current_agent"] = "aggregator_agent"
    state["next_agent"] = None  # End of workflow
    
    # Final log entry
    state["trace"].append({
        "agent": "aggregator_agent",
        "tool": "llm_aggregation",
        "input": "all_agent_results",
        "output": final_answer[:200] + "...",
        "handoff_to": "END"
    })
    
    return state
```

---

## State Management

### State Schema Definition

```python
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    """Shared state across all agents"""
    
    # Input
    query: str
    query_type: str  # "financial_analysis", "risk_assessment", etc.
    
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
    trace: Annotated[List[Dict], add_messages]  # Execution log
    
    # Memory
    conversation_history: Annotated[List[Dict], add_messages]
    cached_results: Dict
    
    # Output
    final_answer: Optional[str]
    confidence_score: float
    
    # Error Handling
    errors: List[Dict]
```

### State Updates

**State is immutable but updated through returns**:

```python
# Agents receive state and return updated state
def agent(state: AgentState) -> AgentState:
    # Modify state
    state["field"] = new_value
    return state  # LangGraph merges updates
```

**Annotations for list fields**:
```python
from langgraph.graph import add_messages

# add_messages: Append to list instead of replacing
trace: Annotated[List[Dict], add_messages]
```

---

## Hand-off Mechanism

### Conditional Edge Routing

**Routing functions decide next agent based on state**:

```python
def route_after_information_agent(state: AgentState) -> str:
    """Determine next agent after information retrieval"""
    
    chunks = state["retrieved_chunks"]
    query = state["query"]
    
    # Insufficient information → web search
    if len(chunks) < 3:
        return "web_search_agent"
    
    # Tables found → table agent
    if any(c["content_type"] == "table" for c in chunks):
        return "table_agent"
    
    # Calculation needed → math agent
    if is_calculation_query(query):
        return "math_agent"
    
    # Default → summarization
    return "summarization_agent"

# Add conditional edge to graph
workflow.add_conditional_edges(
    "information_agent",
    route_after_information_agent,
    {
        "web_search_agent": "web_search_agent",
        "table_agent": "table_agent",
        "math_agent": "math_agent",
        "summarization_agent": "summarization_agent"
    }
)
```

### Direct Edges

**Fixed transitions (no conditional logic)**:

```python
# All agents eventually route to aggregator
workflow.add_edge("math_agent", "aggregator_agent")
workflow.add_edge("summarization_agent", "aggregator_agent")

# Aggregator ends workflow
workflow.add_edge("aggregator_agent", END)
```

---

## Memory Management

### Three-Level Memory

#### 1. Short-term Memory (State)

Current query context, cleared after completion:
```python
state = {
    "query": "What was revenue in 2022?",
    "retrieved_chunks": [...],
    "final_answer": "..."
}
```

#### 2. Conversation Memory (Checkpointing)

Persisted across queries in a session:
```python
from langgraph.checkpoint import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# First query
result1 = app.invoke(
    {"query": "What was revenue in 2022?"},
    config={"configurable": {"thread_id": "session_123"}}
)

# Follow-up query (context aware)
result2 = app.invoke(
    {"query": "How does that compare to 2021?"},
    config={"configurable": {"thread_id": "session_123"}}
)
```

#### 3. Long-term Memory (Cache)

Frequently asked queries cached:
```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_result(query_hash: str):
    """Check cache for previous query results"""
    return cache.get(query_hash)

def query_with_cache(query: str) -> Dict:
    """Check cache before running full workflow"""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    
    # Check cache
    cached = get_cached_result(query_hash)
    if cached:
        return cached
    
    # Run workflow
    result = app.invoke({"query": query})
    
    # Cache result
    cache.set(query_hash, result, expire=3600)
    
    return result
```

---

## Tool Integration

### Tool Definition

```python
from langchain.tools import Tool

# ChromaDB retrieval tool
chromadb_tool = Tool(
    name="chromadb_search",
    func=chromadb_retriever.retrieve,
    description="Search financial document chunks by semantic similarity"
)

# Web search tool
web_search_tool = Tool(
    name="tavily_search",
    func=tavily_search,
    description="Search the web for current financial information"
)

# Calculator tool
calculator_tool = Tool(
    name="calculator",
    func=calculate_expression,
    description="Perform mathematical calculations on extracted values"
)
```

### Tool Execution in Agents

```python
def agent_with_tools(state: AgentState, tools: List[Tool]) -> AgentState:
    """Agent that uses multiple tools"""
    
    for tool in tools:
        if should_use_tool(tool, state):
            result = tool.run(get_tool_input(state))
            state = update_state_with_tool_result(state, tool.name, result)
    
    return state
```

---

## Execution Flow Examples

### Example 1: Financial Analysis Query

**Query**: "Compare YoY revenue growth between 2021 and 2022"

**Execution Flow**:
```
1. Information Agent
   - Retrieves chunks about revenue (2021, 2022)
   - Finds tables with revenue data
   - Hand-off to: Table Agent

2. Table Agent
   - Extracts revenue values:
     * 2021: $469.8B
     * 2022: $514.0B
   - Hand-off to: Math Agent

3. Math Agent
   - Calculates YoY growth:
     * (514.0 - 469.8) / 469.8 = 9.4%
   - Hand-off to: Aggregator Agent

4. Aggregator Agent
   - Compiles final answer:
     "Amazon's revenue grew 9.4% YoY from $469.8B in 2021 to $514.0B in 2022"
   - END
```

### Example 2: Risk Assessment Query

**Query**: "Summarize key risks affecting future revenue"

**Execution Flow**:
```
1. Information Agent
   - Retrieves chunks from "Risk Factors" section
   - No tables or calculations needed
   - Hand-off to: Summarization Agent

2. Summarization Agent
   - Condenses risk factors into key points
   - Hand-off to: Aggregator Agent

3. Aggregator Agent
   - Formats final summary
   - END
```

### Example 3: External Data Query

**Query**: "How does Amazon's revenue compare to current market cap?"

**Execution Flow**:
```
1. Information Agent
   - Retrieves revenue data from documents
   - Insufficient info on market cap
   - Hand-off to: Web Search Agent

2. Web Search Agent
   - Searches web for current Amazon market cap
   - Hand-off to: Math Agent

3. Math Agent
   - Calculates revenue-to-market-cap ratio
   - Hand-off to: Aggregator Agent

4. Aggregator Agent
   - Combines internal + external data
   - END
```

---

## Implementation Guide

### Step 1: Define State Schema

```python
# src/task2_agents/core/state_schema.py

from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[Dict]
    # ... (full schema above)
```

### Step 2: Implement Agent Nodes

```python
# src/task2_agents/agents/information_agent.py

def information_agent(state: AgentState) -> AgentState:
    # Implementation (see above)
    pass
```

### Step 3: Create Routing Functions

```python
# src/task2_agents/core/langgraph_workflow.py

def route_after_information_agent(state: AgentState) -> str:
    # Routing logic (see above)
    pass
```

### Step 4: Build LangGraph Workflow

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("information_agent", information_agent)
workflow.add_node("table_agent", table_agent)
workflow.add_node("math_agent", math_agent)
workflow.add_node("web_search_agent", web_search_agent)
workflow.add_node("summarization_agent", summarization_agent)
workflow.add_node("aggregator_agent", aggregator_agent)

# Set entry point
workflow.set_entry_point("information_agent")

# Add edges
workflow.add_conditional_edges(
    "information_agent",
    route_after_information_agent,
    {
        "table_agent": "table_agent",
        "web_search_agent": "web_search_agent",
        "math_agent": "math_agent",
        "summarization_agent": "summarization_agent"
    }
)

workflow.add_conditional_edges(
    "table_agent",
    lambda s: "math_agent" if requires_calculation(s) else "aggregator_agent",
    {
        "math_agent": "math_agent",
        "aggregator_agent": "aggregator_agent"
    }
)

workflow.add_edge("math_agent", "aggregator_agent")
workflow.add_edge("web_search_agent", "aggregator_agent")
workflow.add_edge("summarization_agent", "aggregator_agent")
workflow.add_edge("aggregator_agent", END)

# Compile
app = workflow.compile()
```

### Step 5: Execute Query

```python
# Run query
initial_state = {
    "query": "Compare YoY revenue growth between 2021 and 2022",
    "trace": [],
    "conversation_history": []
}

result = app.invoke(initial_state)

print(result["final_answer"])
print(json.dumps(result["trace"], indent=2))
```

---

## Next Steps

- Implement parallel agent execution for independent tasks
- Add support for multi-turn conversations
- Optimize agent prompts for better routing decisions
- Add visualization for execution graphs

**Related Documentation**:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview
- [AGENT_SPECS.md](AGENT_SPECS.md) - Detailed agent specifications
- [TASK1_CHUNKING.md](TASK1_CHUNKING.md) - Document retrieval

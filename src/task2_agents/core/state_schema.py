"""LangGraph state schema for multi-agent system"""

from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Shared state across all agents"""

    # Input
    query: str
    query_type: Optional[str]  # "financial_analysis", "risk_assessment", etc.

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
    trace: Annotated[List[Dict], add_messages]

    # Memory
    conversation_history: Annotated[List[Dict], add_messages]
    cached_results: Dict

    # Output
    final_answer: Optional[str]
    confidence_score: float

    # Error Handling
    errors: List[Dict]

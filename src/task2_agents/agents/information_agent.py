"""Information Retrieval Agent"""

from typing import Dict
from datetime import datetime
from src.task2_agents.core.state_schema import AgentState
from src.task1_chunking.storage.chromadb_manager import ChromaDBManager
from src.utils.logging_utils import log_info


class InformationAgent:
    """Agent for retrieving relevant chunks from ChromaDB"""

    def __init__(self):
        self.db_manager = ChromaDBManager()

    def __call__(self, state: AgentState) -> AgentState:
        """Execute information retrieval"""
        log_info("InformationAgent: Retrieving relevant chunks")

        query = state["query"]

        # Retrieve chunks from ChromaDB
        chunks = self.db_manager.query(query_text=query, top_k=10)

        # Update state
        state["retrieved_chunks"] = chunks
        state["current_agent"] = "information_agent"
        state["agent_history"] = state.get("agent_history", []) + ["information_agent"]

        # Determine next agent based on retrieved content
        next_agent = self._determine_next_agent(state, chunks)
        state["next_agent"] = next_agent

        # Log to trace
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": "information_agent",
            "tool": "chromadb_search",
            "input": query[:100],
            "output": f"Retrieved {len(chunks)} chunks",
            "handoff_to": next_agent,
        }

        if "trace" not in state:
            state["trace"] = []
        state["trace"].append(trace_entry)

        log_info(
            f"InformationAgent: Retrieved {len(chunks)} chunks, next: {next_agent}"
        )

        return state

    def _determine_next_agent(self, state: AgentState, chunks: list) -> str:
        """Determine which agent to hand-off to"""

        if len(chunks) < 3:
            # Insufficient information, try web search
            return "web_search_agent"

        # Check if tables are present
        has_tables = any(
            chunk.get("metadata", {}).get("content_type") == "table" for chunk in chunks
        )

        if has_tables:
            return "table_agent"

        # Check if query requires calculation
        query_lower = state["query"].lower()
        calc_keywords = [
            "calculate",
            "growth",
            "change",
            "percentage",
            "ratio",
            "compare",
        ]

        if any(keyword in query_lower for keyword in calc_keywords):
            return "math_agent"

        # Default to summarization
        return "summarization_agent"


# Create agent instance
information_agent = InformationAgent()

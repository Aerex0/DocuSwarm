"""Query handler for processing user queries"""

import json
from typing import Dict, Any, Optional
from datetime import datetime
from src.task2_agents.core.state_schema import AgentState
from src.utils.logging_utils import log_info, log_error, get_trace_logger


class QueryHandler:
    """Handles query classification and preprocessing"""

    def __init__(self):
        self.trace_logger = get_trace_logger()

    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """Preprocess and classify the query"""

        query = query.strip()

        # Classify query type
        query_type = self._classify_query(query)

        log_info(f"Query classified as: {query_type}")

        return {
            "original_query": query,
            "processed_query": query,
            "query_type": query_type,
            "timestamp": datetime.now().isoformat(),
        }

    def _classify_query(self, query: str) -> str:
        """Classify query into categories"""

        query_lower = query.lower()

        # Financial calculation queries
        if any(
            keyword in query_lower
            for keyword in ["growth", "change", "percentage", "yoy", "year-over-year"]
        ):
            return "financial_calculation"

        # Comparison queries
        if any(
            keyword in query_lower
            for keyword in ["compare", "versus", "vs", "difference"]
        ):
            return "comparison"

        # Risk assessment queries
        if any(
            keyword in query_lower
            for keyword in ["risk", "threat", "challenge", "concern", "exposure"]
        ):
            return "risk_assessment"

        # Market/business strategy queries
        if any(
            keyword in query_lower
            for keyword in [
                "strategy",
                "plan",
                "initiative",
                "acquisition",
                "expansion",
                "market",
            ]
        ):
            return "business_strategy"

        # Financial metrics queries
        if any(
            keyword in query_lower
            for keyword in [
                "revenue",
                "profit",
                "income",
                "expense",
                "cash flow",
                "margin",
                "ratio",
            ]
        ):
            return "financial_metrics"

        # Summarization queries
        if any(
            keyword in query_lower
            for keyword in ["summarize", "summary", "overview", "describe"]
        ):
            return "summarization"

        # Default
        return "general"

    def initialize_state(
        self, query: str, query_metadata: Dict[str, Any]
    ) -> AgentState:
        """Initialize AgentState for workflow"""

        state: AgentState = {
            # Input
            "query": query,
            "query_type": query_metadata.get("query_type"),
            # Retrieved Data
            "retrieved_chunks": [],
            "web_search_results": [],
            # Processed Results
            "extracted_tables": [],
            "calculated_values": {},
            "summary": "",
            # Agent Coordination
            "current_agent": "",
            "next_agent": None,
            "agent_history": [],
            # Trace & Logging
            "trace": [],
            # Memory
            "conversation_history": [],
            "cached_results": {},
            # Output
            "final_answer": None,
            "confidence_score": 0.0,
            # Error Handling
            "errors": [],
        }

        return state

    def postprocess_results(self, state: AgentState) -> Dict[str, Any]:
        """Postprocess and format final results"""

        # Extract key information
        result = {
            "query": state.get("query", ""),
            "query_type": state.get("query_type", "general"),
            "final_answer": state.get("final_answer", "No answer generated"),
            "confidence_score": state.get("confidence_score", 0.0),
            "agent_workflow": " → ".join(state.get("agent_history", [])),
            "trace": state.get("trace", []),
            "errors": state.get("errors", []),
            "metadata": {
                "num_chunks_retrieved": len(state.get("retrieved_chunks", [])),
                "num_tables_extracted": len(state.get("extracted_tables", [])),
                "web_search_performed": len(state.get("web_search_results", [])) > 0,
                "calculations_performed": bool(state.get("calculated_values", {})),
            },
        }

        # Save trace to JSON
        self._save_trace(result)

        return result

    def _save_trace(self, result: Dict[str, Any]) -> None:
        """Save execution trace to JSON file"""

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_file = f"logs/traces/trace_{timestamp}.json"

            import os

            os.makedirs("logs/traces", exist_ok=True)

            with open(trace_file, "w") as f:
                json.dump(result, f, indent=2)

            log_info(f"Trace saved to: {trace_file}")

        except Exception as e:
            log_error(f"Failed to save trace: {str(e)}")

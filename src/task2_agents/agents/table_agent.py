"""Table Processing Agent"""

import json
from datetime import datetime
from src.task2_agents.core.state_schema import AgentState
from src.utils.groq_client import GroqClient
from src.utils.logging_utils import log_info


class TableAgent:
    """Agent for extracting and processing table data"""

    def __init__(self):
        self.llm = GroqClient()

    def __call__(self, state: AgentState) -> AgentState:
        """Execute table processing"""
        log_info("TableAgent: Processing tables")

        query = state["query"]
        chunks = state.get("retrieved_chunks", [])

        # Filter table chunks
        table_chunks = [
            c for c in chunks if c.get("metadata", {}).get("content_type") == "table"
        ]

        extracted_tables = []

        for chunk in table_chunks[:3]:  # Limit to top 3 tables
            table_text = chunk.get("text", "")

            # Use LLM to extract relevant data
            prompt = f"""Extract relevant data from this table to answer the query.

Query: {query}

Table:
{table_text}

Return only the extracted values as JSON. For example:
{{"year_2021": 469.8, "year_2022": 514.0}}
"""

            try:
                response = self.llm.invoke(prompt)
                # Try to parse JSON from response
                extracted_data = self._parse_json_response(response)
                extracted_tables.append(extracted_data)
            except Exception as e:
                log_info(f"TableAgent: Error extracting from table: {e}")

        # Update state
        state["extracted_tables"] = extracted_tables
        state["current_agent"] = "table_agent"
        state["agent_history"] = state.get("agent_history", []) + ["table_agent"]

        # Determine next agent
        if self._requires_calculation(query):
            state["next_agent"] = "math_agent"
        else:
            state["next_agent"] = "aggregator_agent"

        # Log to trace
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": "table_agent",
            "tool": "llm_extract_table",
            "input": f"{len(table_chunks)} tables",
            "output": f"Extracted {len(extracted_tables)} data points",
            "handoff_to": state["next_agent"],
        }

        if "trace" not in state:
            state["trace"] = []
        state["trace"].append(trace_entry)

        log_info(
            f"TableAgent: Extracted {len(extracted_tables)} tables, next: {state['next_agent']}"
        )

        return state

    def _parse_json_response(self, response: str) -> dict:
        """Try to parse JSON from LLM response"""
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        return {}

    def _requires_calculation(self, query: str) -> bool:
        """Check if query requires mathematical calculations"""
        calc_keywords = ["calculate", "growth", "change", "percentage", "compare"]
        return any(kw in query.lower() for kw in calc_keywords)


# Create agent instance
table_agent = TableAgent()

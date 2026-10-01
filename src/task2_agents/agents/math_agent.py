"""Math Agent for calculations and YoY growth analysis"""

import re
from typing import Dict, List, Any
from datetime import datetime
from src.task2_agents.core.state_schema import AgentState
from src.utils.groq_client import call_groq_llm
from src.utils.logging_utils import log_info, log_error


class MathAgent:
    """Agent for performing calculations, YoY growth, ratios, and percentage changes"""

    def __init__(self):
        self.model = "llama-3.3-70b-versatile"

    def __call__(self, state: AgentState) -> AgentState:
        """Execute mathematical calculations"""
        log_info("MathAgent: Performing calculations")

        try:
            query = state["query"]
            retrieved_chunks = state.get("retrieved_chunks", [])
            extracted_tables = state.get("extracted_tables", [])

            # Build context from chunks and tables
            context = self._build_context(retrieved_chunks, extracted_tables)

            # Perform calculation using LLM
            calculation_result = self._perform_calculation(query, context)

            # Update state
            state["calculated_values"] = calculation_result
            state["current_agent"] = "math_agent"
            state["agent_history"] = state.get("agent_history", []) + ["math_agent"]

            # Determine next agent
            next_agent = self._determine_next_agent(state)
            state["next_agent"] = next_agent

            # Log to trace
            trace_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "math_agent",
                "tool": "llm_calculation",
                "input": query[:100],
                "output": str(calculation_result)[:200],
                "handoff_to": next_agent,
            }

            if "trace" not in state:
                state["trace"] = []
            state["trace"].append(trace_entry)

            log_info(f"MathAgent: Calculations complete, next: {next_agent}")

        except Exception as e:
            log_error(f"MathAgent error: {str(e)}")
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "math_agent",
                "error": str(e),
            }
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_entry)

            # Fallback to aggregator
            state["next_agent"] = "aggregator_agent"

        return state

    def _build_context(
        self, retrieved_chunks: List[Dict], extracted_tables: List[Dict]
    ) -> str:
        """Build context from retrieved chunks and extracted tables"""
        context_parts = []

        # Add text chunks
        for i, chunk in enumerate(retrieved_chunks[:5]):
            content = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            year = metadata.get("year", "unknown")
            context_parts.append(f"[Chunk {i + 1} - Year {year}]: {content[:500]}")

        # Add table data
        for i, table in enumerate(extracted_tables[:3]):
            table_data = table.get("data", {})
            context_parts.append(f"[Table {i + 1}]: {str(table_data)[:500]}")

        return "\n\n".join(context_parts)

    def _perform_calculation(self, query: str, context: str) -> Dict[str, Any]:
        """Use LLM to perform calculations based on context"""

        prompt = f"""You are a financial calculation expert. Given the query and context below, perform the requested calculations.

Query: {query}

Context:
{context}

Instructions:
1. Extract relevant numerical values from the context
2. Perform the requested calculation (YoY growth, percentage change, ratio, etc.)
3. Show your work step-by-step
4. Return the result in JSON format

Return your answer in this JSON structure:
{{
    "extracted_values": {{"year_1": value1, "year_2": value2, ...}},
    "calculation_steps": ["step 1", "step 2", ...],
    "final_result": {{
        "value": numeric_value,
        "unit": "percentage/dollar/ratio",
        "interpretation": "brief explanation"
    }}
}}

If you cannot find the required values, return:
{{
    "error": "Insufficient data to perform calculation",
    "missing_values": ["list", "of", "missing", "data"]
}}
"""

        try:
            response = call_groq_llm(
                prompt=prompt,
                model=self.model,
                temperature=0.1,
                max_tokens=1000,
            )

            # Try to parse JSON from response
            result = self._extract_json_from_response(response)

            if result:
                return result
            else:
                # Fallback: return raw response
                return {
                    "raw_calculation": response,
                    "final_result": {
                        "value": None,
                        "unit": "text",
                        "interpretation": response[:500],
                    },
                }

        except Exception as e:
            log_error(f"Calculation error: {str(e)}")
            return {
                "error": str(e),
                "final_result": {
                    "value": None,
                    "unit": "error",
                    "interpretation": "Calculation failed",
                },
            }

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON object from LLM response"""
        import json

        # Try to find JSON in response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _determine_next_agent(self, state: AgentState) -> str:
        """Determine next agent in workflow"""

        # Check if we need additional web search
        calculated_values = state.get("calculated_values", {})
        if "error" in calculated_values or "missing_values" in calculated_values:
            # Try web search for missing data
            if "web_search_agent" not in state.get("agent_history", []):
                return "web_search_agent"

        # Check if we need summarization
        query_lower = state["query"].lower()
        if any(
            keyword in query_lower
            for keyword in ["summarize", "overview", "explain", "describe"]
        ):
            return "summarization_agent"

        # Otherwise, go to aggregator
        return "aggregator_agent"


# Create agent instance
math_agent = MathAgent()

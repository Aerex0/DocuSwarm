"""Summarization Agent for text condensation"""

from typing import Dict, List, Any
from datetime import datetime
from src.task2_agents.core.state_schema import AgentState
from src.utils.groq_client import call_groq_llm
from src.utils.logging_utils import log_info, log_error


class SummarizationAgent:
    """Agent for summarizing retrieved information and preparing concise answers"""

    def __call__(self, state: AgentState) -> AgentState:
        """Execute summarization"""
        log_info("SummarizationAgent: Summarizing information")

        try:
            query = state["query"]

            # Gather all available information
            context = self._build_context(state)

            # Generate summary
            summary = self._generate_summary(query, context)

            # Update state
            state["summary"] = summary
            state["current_agent"] = "summarization_agent"
            state["agent_history"] = state.get("agent_history", []) + [
                "summarization_agent"
            ]

            # Always go to aggregator after summarization
            next_agent = "aggregator_agent"
            state["next_agent"] = next_agent

            # Log to trace
            trace_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "summarization_agent",
                "tool": "llm_summarization",
                "input": query[:100],
                "output": summary[:200],
                "handoff_to": next_agent,
            }

            if "trace" not in state:
                state["trace"] = []
            state["trace"].append(trace_entry)

            log_info(f"SummarizationAgent: Summary complete, next: {next_agent}")

        except Exception as e:
            log_error(f"SummarizationAgent error: {str(e)}")
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "summarization_agent",
                "error": str(e),
            }
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_entry)

            # Continue to aggregator even if summarization fails
            state["next_agent"] = "aggregator_agent"

        return state

    def _build_context(self, state: AgentState) -> str:
        """Build context from all available information"""
        context_parts = []

        # Add retrieved chunks
        retrieved_chunks = state.get("retrieved_chunks", [])
        if retrieved_chunks:
            context_parts.append("=== RETRIEVED DOCUMENTS ===")
            for i, chunk in enumerate(retrieved_chunks[:10]):
                content = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                year = metadata.get("year", "unknown")
                doc_name = metadata.get("document_name", "unknown")
                context_parts.append(
                    f"\n[Document {i + 1} - {doc_name} ({year})]\n{content[:800]}"
                )

        # Add extracted tables
        extracted_tables = state.get("extracted_tables", [])
        if extracted_tables:
            context_parts.append("\n\n=== EXTRACTED TABLES ===")
            for i, table in enumerate(extracted_tables[:5]):
                table_data = table.get("data", {})
                context_parts.append(f"\n[Table {i + 1}]\n{str(table_data)[:500]}")

        # Add calculated values
        calculated_values = state.get("calculated_values", {})
        if calculated_values and "final_result" in calculated_values:
            context_parts.append("\n\n=== CALCULATIONS ===")
            result = calculated_values.get("final_result", {})
            context_parts.append(
                f"Value: {result.get('value')}\n"
                f"Unit: {result.get('unit')}\n"
                f"Interpretation: {result.get('interpretation')}"
            )

        # Add web search results
        web_search_results = state.get("web_search_results", [])
        if web_search_results:
            context_parts.append("\n\n=== WEB SEARCH RESULTS ===")
            for i, result in enumerate(web_search_results[:5]):
                title = result.get("title", "")
                content = result.get("content", "")
                context_parts.append(f"\n[Result {i + 1} - {title}]\n{content[:500]}")

        return "\n".join(context_parts)

    def _generate_summary(self, query: str, context: str) -> str:
        """Generate a comprehensive summary using LLM"""

        prompt = f"""You are a financial analysis expert. Given the query and context below, provide a clear, comprehensive, and well-structured summary.

Query: {query}

Context:
{context}

Instructions:
1. Directly answer the query using information from the context
2. Structure your response with clear sections if the query is complex
3. Include specific numerical values, dates, and financial metrics when relevant
4. Cite sources when possible (e.g., "According to Amazon's 2022 10-K report...")
5. If information is incomplete, clearly state what data is missing
6. Be concise but thorough - aim for 200-400 words
7. Use professional financial language

Provide your summary now:"""

        try:
            summary = call_groq_llm(
                prompt=prompt,
                temperature=0.3,
            )

            return summary.strip()

        except Exception as e:
            log_error(f"Summary generation error: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _determine_next_agent(self, state: AgentState) -> str:
        """Always go to aggregator after summarization"""
        return "aggregator_agent"


# Create agent instance
summarization_agent = SummarizationAgent()

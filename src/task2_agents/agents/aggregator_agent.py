"""Aggregator Agent for final answer compilation"""

from typing import Dict, List, Any
from datetime import datetime
from src.task2_agents.core.state_schema import AgentState
from src.utils.groq_client import call_groq_llm
from src.utils.logging_utils import log_info, log_error


class AggregatorAgent:
    """Agent for compiling final answer from all agent outputs"""

    def __call__(self, state: AgentState) -> AgentState:
        """Execute final answer aggregation"""
        log_info("AggregatorAgent: Compiling final answer")

        try:
            query = state["query"]

            # Gather all information from state
            all_info = self._gather_all_information(state)

            # Generate final answer
            final_answer = self._generate_final_answer(query, all_info, state)

            # Calculate confidence score
            confidence_score = self._calculate_confidence(state)

            # Update state
            state["final_answer"] = final_answer
            state["confidence_score"] = confidence_score
            state["current_agent"] = "aggregator_agent"
            state["agent_history"] = state.get("agent_history", []) + [
                "aggregator_agent"
            ]
            state["next_agent"] = None  # End of workflow

            # Log to trace
            trace_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "aggregator_agent",
                "tool": "final_compilation",
                "input": query[:100],
                "output": final_answer[:200],
                "confidence": confidence_score,
                "handoff_to": None,
            }

            if "trace" not in state:
                state["trace"] = []
            state["trace"].append(trace_entry)

            log_info(
                f"AggregatorAgent: Final answer complete (confidence: {confidence_score:.2f})"
            )

        except Exception as e:
            log_error(f"AggregatorAgent error: {str(e)}")
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "aggregator_agent",
                "error": str(e),
            }
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_entry)

            # Provide fallback answer
            state["final_answer"] = (
                f"Error compiling final answer: {str(e)}. "
                f"Please review the agent trace for partial results."
            )
            state["confidence_score"] = 0.0
            state["next_agent"] = None

        return state

    def _gather_all_information(self, state: AgentState) -> Dict[str, Any]:
        """Gather all information from various agents"""
        return {
            "query": state.get("query", ""),
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "extracted_tables": state.get("extracted_tables", []),
            "calculated_values": state.get("calculated_values", {}),
            "web_search_results": state.get("web_search_results", []),
            "summary": state.get("summary", ""),
            "agent_history": state.get("agent_history", []),
            "errors": state.get("errors", []),
        }

    def _generate_final_answer(
        self, query: str, all_info: Dict[str, Any], state: AgentState
    ) -> str:
        """Generate final comprehensive answer"""

        # Check if we already have a summary
        if all_info.get("summary"):
            # If summarization agent already ran, use its output as base
            base_answer = all_info["summary"]
        else:
            # Generate answer from scratch
            base_answer = self._generate_from_scratch(query, all_info)

        # Add source citations and metadata
        final_answer = self._add_citations_and_metadata(base_answer, all_info, state)

        return final_answer

    def _generate_from_scratch(self, query: str, all_info: Dict[str, Any]) -> str:
        """Generate answer from scratch if no summary exists"""

        # Build context
        context_parts = []

        # Retrieved chunks
        if all_info.get("retrieved_chunks"):
            context_parts.append("=== RETRIEVED INFORMATION ===")
            for i, chunk in enumerate(all_info["retrieved_chunks"][:8]):
                content = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                year = metadata.get("year", "unknown")
                context_parts.append(
                    f"\n[Source {i + 1} - Year {year}]: {content[:600]}"
                )

        # Tables
        if all_info.get("extracted_tables"):
            context_parts.append("\n\n=== TABLES ===")
            for i, table in enumerate(all_info["extracted_tables"][:3]):
                context_parts.append(f"\n[Table {i + 1}]: {str(table)[:400]}")

        # Calculations
        if all_info.get("calculated_values"):
            context_parts.append("\n\n=== CALCULATIONS ===")
            context_parts.append(str(all_info["calculated_values"])[:400])

        # Web results
        if all_info.get("web_search_results"):
            context_parts.append("\n\n=== WEB SEARCH ===")
            for i, result in enumerate(all_info["web_search_results"][:3]):
                context_parts.append(
                    f"\n[{result.get('title', '')}]: {result.get('content', '')[:400]}"
                )

        context = "\n".join(context_parts)

        prompt = f"""You are a financial analysis expert. Compile a comprehensive, accurate answer to the query below using the provided information.

Query: {query}

Available Information:
{context}

Instructions:
1. Provide a direct, clear answer to the query
2. Use specific data, numbers, and dates from the context
3. Structure your answer logically (use sections/bullet points if needed)
4. If information is incomplete, state what's missing
5. Be professional and concise (300-500 words)
6. Include relevant financial metrics and analysis

Provide your final answer now:"""

        try:
            answer = call_groq_llm(
                prompt=prompt,
                temperature=0.2,
            )
            return answer.strip()

        except Exception as e:
            log_error(f"Answer generation error: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def _add_citations_and_metadata(
        self, base_answer: str, all_info: Dict[str, Any], state: AgentState
    ) -> str:
        """Add source citations and metadata to answer"""

        # Extract unique sources
        sources = set()
        retrieved_chunks = all_info.get("retrieved_chunks", [])
        for chunk in retrieved_chunks[:10]:
            metadata = chunk.get("metadata", {})
            doc_name = metadata.get("document_name", "")
            year = metadata.get("year", "")
            if doc_name and year:
                sources.add(f"{doc_name} ({year})")

        # Build final answer with metadata
        final_parts = [base_answer]

        # Add sources section
        if sources:
            final_parts.append("\n\n---\n**Sources:**")
            for i, source in enumerate(sorted(sources), 1):
                final_parts.append(f"{i}. {source}")

        # Add agent workflow info
        agent_history = all_info.get("agent_history", [])
        if agent_history:
            final_parts.append(f"\n**Agents Used:** {' → '.join(agent_history)}")

        # Add warnings if any errors occurred
        errors = all_info.get("errors", [])
        if errors:
            final_parts.append(
                f"\n**Note:** {len(errors)} warning(s) occurred during processing. "
                "Please review the trace for details."
            )

        return "\n".join(final_parts)

    def _calculate_confidence(self, state: AgentState) -> float:
        """Calculate confidence score based on available information"""

        confidence = 0.5  # Base confidence

        # Boost for retrieved chunks
        retrieved_chunks = state.get("retrieved_chunks", [])
        if len(retrieved_chunks) >= 5:
            confidence += 0.2
        elif len(retrieved_chunks) >= 3:
            confidence += 0.1

        # Boost for tables
        extracted_tables = state.get("extracted_tables", [])
        if extracted_tables:
            confidence += 0.1

        # Boost for calculations
        calculated_values = state.get("calculated_values", {})
        if calculated_values and "final_result" in calculated_values:
            confidence += 0.1

        # Boost for web search
        web_search_results = state.get("web_search_results", [])
        if web_search_results:
            confidence += 0.1

        # Penalty for errors
        errors = state.get("errors", [])
        if errors:
            confidence -= 0.1 * len(errors)

        # Clamp between 0 and 1
        confidence = max(0.0, min(1.0, confidence))

        return confidence


# Create agent instance
aggregator_agent = AggregatorAgent()

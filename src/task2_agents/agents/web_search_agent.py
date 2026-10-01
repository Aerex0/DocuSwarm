"""Web Search Agent using Tavily API"""

import os
from typing import Dict, List, Any
from datetime import datetime
from src.task2_agents.core.state_schema import AgentState
from src.utils.logging_utils import log_info, log_error


class WebSearchAgent:
    """Agent for retrieving external information using Tavily search"""

    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            log_error("TAVILY_API_KEY not found in environment")

    def __call__(self, state: AgentState) -> AgentState:
        """Execute web search"""
        log_info("WebSearchAgent: Performing web search")

        try:
            query = state["query"]

            # Perform web search
            search_results = self._search_tavily(query)

            # Update state
            state["web_search_results"] = search_results
            state["current_agent"] = "web_search_agent"
            state["agent_history"] = state.get("agent_history", []) + [
                "web_search_agent"
            ]

            # Determine next agent
            next_agent = self._determine_next_agent(state)
            state["next_agent"] = next_agent

            # Log to trace
            trace_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "web_search_agent",
                "tool": "tavily_search",
                "input": query[:100],
                "output": f"Found {len(search_results)} results",
                "handoff_to": next_agent,
            }

            if "trace" not in state:
                state["trace"] = []
            state["trace"].append(trace_entry)

            log_info(
                f"WebSearchAgent: Found {len(search_results)} results, next: {next_agent}"
            )

        except Exception as e:
            log_error(f"WebSearchAgent error: {str(e)}")
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent": "web_search_agent",
                "error": str(e),
            }
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_entry)

            # Continue to aggregator even if search fails
            state["next_agent"] = "aggregator_agent"

        return state

    def _search_tavily(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Tavily API"""
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.tavily_api_key)

            # Perform search
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",  # Use advanced search for better quality
                include_answer=True,  # Get AI-generated answer
                include_raw_content=False,  # Don't need full page content
            )

            results = []

            # Process results
            for result in response.get("results", []):
                results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0.0),
                    }
                )

            # Add AI answer if available
            if "answer" in response:
                results.insert(
                    0,
                    {
                        "title": "AI Summary",
                        "url": "",
                        "content": response["answer"],
                        "score": 1.0,
                    },
                )

            return results

        except ImportError:
            log_error("tavily-python not installed. Install with: pip install tavily")
            return []
        except Exception as e:
            log_error(f"Tavily search error: {str(e)}")
            return []

    def _determine_next_agent(self, state: AgentState) -> str:
        """Determine next agent in workflow"""

        search_results = state.get("web_search_results", [])

        # If no results found, go directly to aggregator
        if not search_results:
            return "aggregator_agent"

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
            # Allow one math run per web search. Math may have failed on missing
            # values that this search just supplied, so a retry is worthwhile -
            # but capping runs per search keeps math and web from ping-ponging.
            history = state.get("agent_history", [])
            if history.count("math_agent") <= history.count("web_search_agent"):
                return "math_agent"

        # Check if we need summarization
        if any(
            keyword in query_lower
            for keyword in ["summarize", "overview", "explain", "describe"]
        ):
            if "summarization_agent" not in state.get("agent_history", []):
                return "summarization_agent"

        # Default to aggregator
        return "aggregator_agent"


# Create agent instance
web_search_agent = WebSearchAgent()

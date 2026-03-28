"""Main orchestrator for the multi-agent QA system"""

from typing import Dict, Any, Optional
from src.task2_agents.core.langgraph_workflow import get_default_workflow
from src.pipeline.query_handler import QueryHandler
from src.utils.logging_utils import log_info, log_error
from src.utils.config import load_config


class Orchestrator:
    """Main orchestrator that ties together the entire pipeline"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize orchestrator with configuration"""

        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = load_config("configs/agents_config.yaml")

        # Initialize components
        self.query_handler = QueryHandler()
        self.workflow = None

        log_info("Orchestrator initialized")

    def process_query(
        self, query: str, thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the entire pipeline

        Args:
            query: User query string
            thread_id: Optional thread ID for conversation memory

        Returns:
            Dict containing final answer, trace, and metadata
        """

        log_info(f"Processing query: {query[:100]}...")

        try:
            # Step 1: Preprocess and classify query
            query_metadata = self.query_handler.preprocess_query(query)

            # Step 2: Initialize state
            initial_state = self.query_handler.initialize_state(query, query_metadata)

            # Step 3: Get compiled workflow
            if self.workflow is None:
                self.workflow = get_default_workflow()

            # Step 4: Execute workflow
            config = {"configurable": {"thread_id": thread_id or "default"}}
            final_state = self.workflow.invoke(initial_state, config)

            # Step 5: Postprocess results
            result = self.query_handler.postprocess_results(final_state)

            log_info("Query processing complete")

            return result

        except Exception as e:
            log_error(f"Orchestrator error: {str(e)}")

            # Return error result
            return {
                "query": query,
                "query_type": "error",
                "final_answer": f"Error processing query: {str(e)}",
                "confidence_score": 0.0,
                "agent_workflow": "error",
                "trace": [],
                "errors": [{"error": str(e)}],
                "metadata": {},
            }

    def process_batch_queries(
        self, queries: list[str], thread_id: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """
        Process multiple queries in batch

        Args:
            queries: List of query strings
            thread_id: Optional thread ID for conversation memory

        Returns:
            List of result dicts
        """

        log_info(f"Processing batch of {len(queries)} queries")

        results = []
        for i, query in enumerate(queries):
            log_info(f"Processing query {i + 1}/{len(queries)}")
            result = self.process_query(query, thread_id)
            results.append(result)

        log_info("Batch processing complete")

        return results

    def get_conversation_history(self, thread_id: str) -> list[Dict[str, Any]]:
        """
        Get conversation history for a thread

        Args:
            thread_id: Thread ID

        Returns:
            List of conversation messages
        """

        try:
            # LangGraph's MemorySaver maintains conversation history
            # This would require accessing the checkpointer directly
            log_info(f"Retrieving conversation history for thread: {thread_id}")

            # TODO: Implement retrieval from checkpointer
            # For now, return empty list
            return []

        except Exception as e:
            log_error(f"Failed to retrieve conversation history: {str(e)}")
            return []

    def clear_conversation_history(self, thread_id: str) -> bool:
        """
        Clear conversation history for a thread

        Args:
            thread_id: Thread ID

        Returns:
            Success status
        """

        try:
            log_info(f"Clearing conversation history for thread: {thread_id}")

            # TODO: Implement clearing from checkpointer
            # For now, return True
            return True

        except Exception as e:
            log_error(f"Failed to clear conversation history: {str(e)}")
            return False


# Create default orchestrator instance
def get_orchestrator() -> Orchestrator:
    """Get default orchestrator instance"""
    return Orchestrator()

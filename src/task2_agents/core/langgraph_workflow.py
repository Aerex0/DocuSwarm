"""LangGraph workflow for multi-agent orchestration"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.task2_agents.core.state_schema import AgentState
from src.task2_agents.agents.information_agent import information_agent
from src.task2_agents.agents.table_agent import table_agent
from src.task2_agents.agents.math_agent import math_agent
from src.task2_agents.agents.web_search_agent import web_search_agent
from src.task2_agents.agents.summarization_agent import summarization_agent
from src.task2_agents.agents.aggregator_agent import aggregator_agent
from src.utils.logging_utils import log_info


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow with all agents and routing logic"""

    # Initialize workflow with state schema
    workflow = StateGraph(AgentState)

    # Add all agent nodes
    workflow.add_node("information_agent", information_agent)
    workflow.add_node("table_agent", table_agent)
    workflow.add_node("math_agent", math_agent)
    workflow.add_node("web_search_agent", web_search_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("aggregator_agent", aggregator_agent)

    # Set entry point - always start with information agent
    workflow.set_entry_point("information_agent")

    # Add conditional edges based on next_agent in state
    workflow.add_conditional_edges(
        "information_agent",
        router,
        {
            "table_agent": "table_agent",
            "math_agent": "math_agent",
            "web_search_agent": "web_search_agent",
            "summarization_agent": "summarization_agent",
            "aggregator_agent": "aggregator_agent",
            "END": END,
        },
    )

    workflow.add_conditional_edges(
        "table_agent",
        router,
        {
            "math_agent": "math_agent",
            "web_search_agent": "web_search_agent",
            "summarization_agent": "summarization_agent",
            "aggregator_agent": "aggregator_agent",
            "END": END,
        },
    )

    workflow.add_conditional_edges(
        "math_agent",
        router,
        {
            "table_agent": "table_agent",
            "web_search_agent": "web_search_agent",
            "summarization_agent": "summarization_agent",
            "aggregator_agent": "aggregator_agent",
            "END": END,
        },
    )

    workflow.add_conditional_edges(
        "web_search_agent",
        router,
        {
            "table_agent": "table_agent",
            "math_agent": "math_agent",
            "summarization_agent": "summarization_agent",
            "aggregator_agent": "aggregator_agent",
            "END": END,
        },
    )

    workflow.add_conditional_edges(
        "summarization_agent",
        router,
        {
            "aggregator_agent": "aggregator_agent",
            "END": END,
        },
    )

    # Aggregator always ends the workflow
    workflow.add_edge("aggregator_agent", END)

    return workflow


def router(
    state: AgentState,
) -> Literal[
    "information_agent",
    "table_agent",
    "math_agent",
    "web_search_agent",
    "summarization_agent",
    "aggregator_agent",
    "END",
]:
    """Route to the next agent based on state.next_agent"""

    next_agent = state.get("next_agent")

    # End workflow if no next agent
    if next_agent is None:
        log_info("Router: Ending workflow")
        return "END"

    # Prevent infinite loops - max 10 agent hops
    agent_history = state.get("agent_history", [])
    if len(agent_history) >= 10:
        log_info(
            f"Router: Max agent hops reached ({len(agent_history)}), going to aggregator"
        )
        return "aggregator_agent"

    # Route to next agent
    log_info(f"Router: Routing to {next_agent}")

    if next_agent == "information_agent":
        return "information_agent"
    elif next_agent == "table_agent":
        return "table_agent"
    elif next_agent == "math_agent":
        return "math_agent"
    elif next_agent == "web_search_agent":
        return "web_search_agent"
    elif next_agent == "summarization_agent":
        return "summarization_agent"
    elif next_agent == "aggregator_agent":
        return "aggregator_agent"
    else:
        # Default to aggregator if unknown
        log_info(f"Router: Unknown next_agent '{next_agent}', defaulting to aggregator")
        return "aggregator_agent"


def compile_workflow(checkpointer=None) -> Any:
    """Compile the workflow into an executable graph"""

    workflow = create_workflow()

    # Use memory checkpointer if none provided
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Compile with checkpointing for memory management
    compiled = workflow.compile(checkpointer=checkpointer)

    log_info("LangGraph workflow compiled successfully")

    return compiled


# Initialize default compiled workflow
def get_default_workflow():
    """Get the default compiled workflow with memory"""
    return compile_workflow()

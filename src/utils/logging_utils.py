"""Logging utilities for structured logging"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from src.utils.config import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class TraceLogger:
    """Logger for agent execution traces"""

    def __init__(self):
        self.trace: List[Dict[str, Any]] = []
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

    def log_agent_action(
        self,
        agent: str,
        tool: str,
        input_data: Any,
        output: Any,
        handoff_to: str,
        **kwargs,
    ):
        """Log agent action to trace"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "tool": tool,
            "input": str(input_data)[:500],  # Limit size
            "output": str(output)[:500],
            "handoff_to": handoff_to,
            **kwargs,
        }
        self.trace.append(entry)

        if config.debug:
            logger.debug(f"Agent Action: {json.dumps(entry, indent=2)}")

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get full execution trace"""
        return self.trace

    def save_trace(self, query: str, final_answer: str):
        """Save trace to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.logs_dir / f"trace_{timestamp}.json"

        trace_data = {
            "query": query,
            "trace": self.trace,
            "final_answer": final_answer,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filename, "w") as f:
            json.dump(trace_data, f, indent=2)

        logger.info(f"Trace saved to {filename}")

    def clear(self):
        """Clear trace"""
        self.trace = []


def log_error(error: Exception, context: str = ""):
    """Log error with context"""
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)


def log_info(message: str):
    """Log info message"""
    logger.info(message)


def log_debug(message: str):
    """Log debug message"""
    logger.debug(message)

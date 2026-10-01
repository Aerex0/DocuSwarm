#!/usr/bin/env python3
"""
Test script to verify all components work without needing PDFs
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("TESTING MULTI-AGENT QA SYSTEM COMPONENTS")
print("=" * 80)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from src.utils.config import load_config
    from src.utils.groq_client import call_groq_llm
    from src.utils.logging_utils import log_info, log_error
    from src.task2_agents.core.state_schema import AgentState
    from src.task2_agents.agents.information_agent import information_agent
    from src.task2_agents.agents.table_agent import table_agent
    from src.task2_agents.agents.math_agent import math_agent
    from src.task2_agents.agents.web_search_agent import web_search_agent
    from src.task2_agents.agents.summarization_agent import summarization_agent
    from src.task2_agents.agents.aggregator_agent import aggregator_agent
    from src.task2_agents.core.langgraph_workflow import get_default_workflow
    from src.pipeline.orchestrator import get_orchestrator

    print("   ✓ All imports successful!")
except Exception as e:
    print(f"   ✗ Import failed: {str(e)}")
    sys.exit(1)

# Test 2: Check environment variables
print("\n2. Testing environment variables...")
import os

try:
    groq_key = os.getenv("GROQ_API_KEY")
    llama_key = os.getenv("LLAMAPARSE_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if groq_key:
        print(f"   ✓ GROQ_API_KEY: {groq_key[:20]}...")
    else:
        print("   ⚠ GROQ_API_KEY not found")

    if llama_key:
        print(f"   ✓ LLAMAPARSE_API_KEY: {llama_key[:20]}...")
    else:
        print("   ⚠ LLAMAPARSE_API_KEY not found")

    if tavily_key:
        print(f"   ✓ TAVILY_API_KEY: {tavily_key[:20]}...")
    else:
        print("   ⚠ TAVILY_API_KEY not found")

except Exception as e:
    print(f"   ✗ Error checking env vars: {str(e)}")

# Test 3: Test Groq API
print("\n3. Testing Groq API connection...")
try:
    response = call_groq_llm(
        prompt="Say 'Hello' if you can hear me.",
        temperature=0.1,
    )
    print(f"   ✓ Groq API working! Response: {response[:100]}")
except Exception as e:
    print(f"   ✗ Groq API failed: {str(e)}")

# Test 4: Test LangGraph workflow compilation
print("\n4. Testing LangGraph workflow...")
try:
    workflow = get_default_workflow()
    print("   ✓ Workflow compiled successfully!")

    # Get graph info
    graph = workflow.get_graph()
    nodes = list(graph.nodes.keys())
    print(f"   ✓ Workflow has {len(nodes)} nodes: {', '.join(nodes)}")
except Exception as e:
    print(f"   ✗ Workflow compilation failed: {str(e)}")

# Test 5: Test orchestrator initialization
print("\n5. Testing orchestrator...")
try:
    orchestrator = get_orchestrator()
    print("   ✓ Orchestrator initialized successfully!")
except Exception as e:
    print(f"   ✗ Orchestrator failed: {str(e)}")

# Test 6: Test query handler
print("\n6. Testing query handler...")
try:
    from src.pipeline.query_handler import QueryHandler

    handler = QueryHandler()

    test_query = "What was Amazon's revenue in 2022?"
    metadata = handler.preprocess_query(test_query)

    print(f"   ✓ Query classified as: {metadata['query_type']}")
except Exception as e:
    print(f"   ✗ Query handler failed: {str(e)}")

print("\n" + "=" * 80)
print("COMPONENT TEST COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Upload PDF files to data/Amazon/ directory")
print("2. Run: python scripts/preprocess_documents.py --input data/Amazon")
print("3. Run: python scripts/run_pipeline.py --interactive")
print("=" * 80 + "\n")

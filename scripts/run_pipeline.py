#!/usr/bin/env python3
"""
Script to run the multi-agent QA pipeline

Usage:
    python scripts/run_pipeline.py --query "What was Amazon's revenue in 2022?"
    python scripts/run_pipeline.py --batch examples/example_queries.json
    python scripts/run_pipeline.py --interactive
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.orchestrator import get_orchestrator
from src.utils.logging_utils import setup_logging, log_info, log_error


def print_result(result: dict, verbose: bool = False):
    """Pretty print the result"""

    print("\n" + "=" * 80)
    print("QUERY:", result.get("query", ""))
    print("=" * 80)

    print("\nQUERY TYPE:", result.get("query_type", "unknown"))
    print("AGENT WORKFLOW:", result.get("agent_workflow", ""))
    print("CONFIDENCE SCORE:", f"{result.get('confidence_score', 0.0):.2f}")

    print("\n" + "-" * 80)
    print("FINAL ANSWER:")
    print("-" * 80)
    print(result.get("final_answer", "No answer generated"))
    print("-" * 80)

    # Metadata
    metadata = result.get("metadata", {})
    print("\nMETADATA:")
    print(f"  • Chunks Retrieved: {metadata.get('num_chunks_retrieved', 0)}")
    print(f"  • Tables Extracted: {metadata.get('num_tables_extracted', 0)}")
    print(f"  • Web Search Performed: {metadata.get('web_search_performed', False)}")
    print(
        f"  • Calculations Performed: {metadata.get('calculations_performed', False)}"
    )

    # Errors
    errors = result.get("errors", [])
    if errors:
        print(f"\n⚠️  WARNINGS/ERRORS: {len(errors)}")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error.get('error', 'Unknown error')}")

    # Trace (if verbose)
    if verbose:
        trace = result.get("trace", [])
        print(f"\n📊 EXECUTION TRACE ({len(trace)} steps):")
        for i, step in enumerate(trace, 1):
            agent = step.get("agent", "unknown")
            tool = step.get("tool", "unknown")
            handoff = step.get("handoff_to", "END")
            print(f"  {i}. {agent} → {tool} → {handoff}")

    print("=" * 80 + "\n")


def run_single_query(query: str, verbose: bool = False):
    """Run a single query"""

    log_info(f"Running single query: {query}")

    orchestrator = get_orchestrator()
    result = orchestrator.process_query(query)

    print_result(result, verbose)

    return result


def run_batch_queries(batch_file: str, verbose: bool = False):
    """Run batch queries from JSON file"""

    log_info(f"Running batch queries from: {batch_file}")

    # Load queries
    try:
        with open(batch_file, "r") as f:
            data = json.load(f)

        queries = data.get("queries", [])

        # Backward-compatible support for grouped query files
        if not queries:
            grouped_keys = [
                "simple_queries",
                "calculation_queries",
                "multi_section_queries",
                "temporal_comparison_queries",
                "external_data_queries",
                "multimodal_queries",
                "conversational_queries",
                "edge_case_queries",
            ]

            for key in grouped_keys:
                section_items = data.get(key, [])
                if not isinstance(section_items, list):
                    continue

                for item in section_items:
                    if isinstance(item, dict) and item.get("query"):
                        queries.append({"query": item["query"], "category": key})
                    elif isinstance(item, dict) and isinstance(
                        item.get("conversation"), list
                    ):
                        for turn in item["conversation"]:
                            if isinstance(turn, dict) and turn.get("query"):
                                queries.append(
                                    {
                                        "query": turn["query"],
                                        "category": f"{key}_conversation",
                                    }
                                )

        if not queries:
            log_error("No queries found in batch file")
            return

        log_info(f"Found {len(queries)} queries")

    except Exception as e:
        log_error(f"Failed to load batch file: {str(e)}")
        return

    # Process queries
    orchestrator = get_orchestrator()
    results = []

    for i, query_item in enumerate(queries, 1):
        if isinstance(query_item, dict):
            query = query_item.get("query", "")
            category = query_item.get("category", "unknown")
            print(f"\n{'=' * 80}")
            print(f"QUERY {i}/{len(queries)} - Category: {category}")
            print(f"{'=' * 80}")
        else:
            query = str(query_item)

        if not query:
            continue

        result = orchestrator.process_query(query)
        print_result(result, verbose)
        results.append(result)

    # Save results
    output_file = "output/batch_results.json"
    Path("output").mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    log_info(f"Batch results saved to: {output_file}")


def run_interactive(verbose: bool = False):
    """Run in interactive mode"""

    print("\n" + "=" * 80)
    print("MULTI-AGENT QA SYSTEM - INTERACTIVE MODE")
    print("=" * 80)
    print("Enter your queries below. Type 'exit', 'quit', or 'q' to exit.")
    print("Type 'verbose on' or 'verbose off' to toggle verbose mode.")
    print("=" * 80 + "\n")

    orchestrator = get_orchestrator()
    thread_id = "interactive_session"

    while True:
        try:
            # Get query from user
            query = input("\n🔍 Query: ").strip()

            # Check for exit commands
            if query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!\n")
                break

            # Check for verbose toggle
            if query.lower() == "verbose on":
                verbose = True
                print("✓ Verbose mode enabled")
                continue
            elif query.lower() == "verbose off":
                verbose = False
                print("✓ Verbose mode disabled")
                continue

            # Skip empty queries
            if not query:
                continue

            # Process query
            result = orchestrator.process_query(query, thread_id=thread_id)
            print_result(result, verbose)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            log_error(f"Error in interactive mode: {str(e)}")
            print(f"\n❌ Error: {str(e)}\n")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Run the multi-agent QA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--query", "-q", type=str, help="Single query to process")

    parser.add_argument(
        "--batch", "-b", type=str, help="Path to JSON file with batch queries"
    )

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output with trace"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Run based on mode
    if args.query:
        run_single_query(args.query, args.verbose)
    elif args.batch:
        run_batch_queries(args.batch, args.verbose)
    elif args.interactive:
        run_interactive(args.verbose)
    else:
        parser.print_help()
        print("\n❌ Error: Please specify --query, --batch, or --interactive\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

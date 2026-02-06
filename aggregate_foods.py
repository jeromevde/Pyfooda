#!/usr/bin/env python3
"""
CLI tool to run the agentic food aggregation workflow.

Usage:
    python aggregate_foods.py --sample 1000 --threshold 0.85 --use-llm
"""

import argparse
from pyfooda.agentic_aggregator import run_aggregation_workflow


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate similar food items using agentic workflow'
    )
    
    parser.add_argument(
        '--input',
        default='pyfooda/data/foods.csv',
        help='Input CSV file with food data'
    )
    
    parser.add_argument(
        '--output',
        default='pyfooda/data/foods_aggregated.csv',
        help='Output CSV file for aggregated data'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Number of items to sample (default: use all)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Similarity threshold for merge candidates (default: 0.85)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum iterations before stopping (default: 10)'
    )
    
    parser.add_argument(
        '--use-llm',
        action='store_true',
        default=False,
        help='Use LLM for merge validation (requires OPENROUTER_API_KEY)'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        default=False,
        help='Do NOT use LLM (use heuristics only)'
    )
    
    args = parser.parse_args()
    
    # Determine LLM usage
    use_llm = args.use_llm and not args.no_llm
    
    print("="*60)
    print("Agentic Food Aggregation Workflow")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Sample size: {args.sample or 'ALL'}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"LLM validation: {'YES' if use_llm else 'NO'}")
    print("="*60)
    
    # Run workflow
    run_aggregation_workflow(
        input_csv=args.input,
        output_csv=args.output,
        sample_size=args.sample,
        similarity_threshold=args.threshold,
        max_iterations=args.max_iterations,
        use_llm=use_llm
    )


if __name__ == "__main__":
    main()

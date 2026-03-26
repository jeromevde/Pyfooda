#!/usr/bin/env python3
"""
Aggregate USDA foods into a compact everyday nutrition database.

Usage (from repo root):
    python scripts/aggregate.py test          # first 1000 items
    python scripts/aggregate.py full          # all ~296k items
    python scripts/aggregate.py full --resume  # resume from checkpoint
    
"""


import argparse
import os
import sys

import pandas as pd

# Allow running from repo root: python scripts/aggregate.py
sys.path.insert(0, os.path.dirname(__file__))
from aggregator import FoodAggregator


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate USDA foods into a compact everyday database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["test", "full"],
        default="test",
        help="'test' = first 1000 items, 'full' = everything (default: test)",
    )
    parser.add_argument("--model", default="google/gemini-2.0-flash-lite-001", help="LLM model")
    parser.add_argument("--batch-size", type=int, default=1, help="deprecated (streaming is always used)")
    parser.add_argument("--streaming", action="store_true", help="deprecated no-op (streaming is default and only mode)")
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default="openrouter", help="LLM provider")
    parser.add_argument("--base-url", default=None, help="override OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="override API key")
    parser.add_argument("--search-top-k", type=int, default=8, help="search results shown to LLM")
    parser.add_argument("--prompt", default=None, help="path to custom prompt file")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_input = os.path.join(repo_root, "pyfooda/data/fooddata.csv")
    default_output = os.path.join(repo_root, "pyfooda/data/foods_aggregated.json")
    default_checkpoint = os.path.join(repo_root, "checkpoints")
    parser.add_argument("--input", default=default_input, help="input CSV")
    parser.add_argument("--output", default=default_output, help="output JSON")
    parser.add_argument("--checkpoint-dir", default=default_checkpoint, help="checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--limit", type=int, default=None, help="custom item limit")
    parser.add_argument("--timeout-seconds", type=int, default=180, help="LLM request timeout per item")

    args = parser.parse_args()

    # Determine limit and input file
    if args.limit:
        limit = args.limit
    elif args.mode == "test":
        limit = None  # test uses a small curated dataset, no limit needed
    else:
        limit = None

    effective_batch_size = 1

    if args.provider == "ollama":
        if not args.base_url:
            args.base_url = "http://localhost:11434/v1"
        if not args.api_key:
            args.api_key = "ollama"

    # Test mode uses a curated test dataset if available
    input_path = args.input
    output_path = args.output
    checkpoint_dir = args.checkpoint_dir
    if args.mode == "test":
        test_csv = os.path.join(repo_root, "tests", "test_fooddata.csv")
        if os.path.exists(test_csv):
            input_path = test_csv
        else:
            print(f"  Warning: {test_csv} not found, run build_test_dataset.py first")
            print(f"  Falling back to first 2000 items from fooddata.csv")
            limit = 2000

        if args.output == default_output:
            output_path = os.path.join(repo_root, "tests", "test_aggregated_streaming.json" if args.streaming else "test_aggregated.json")
        if args.checkpoint_dir == default_checkpoint:
            checkpoint_dir = os.path.join(repo_root, "tests", "checkpoints_streaming" if args.streaming else "checkpoints")

    print("=" * 60)
    print("  Food Aggregation — Tool-based approach")
    print("=" * 60)
    print(f"  Mode           : {args.mode}")
    print(f"  Items          : {limit or 'ALL'}")
    print(f"  Model          : {args.model}")
    print("  Decision mode  : streaming (1 item/call)")
    print(f"  Batch size     : {effective_batch_size}")
    print(f"  Provider       : {args.provider}")
    print(f"  Base URL       : {args.base_url or '(provider default)'}")
    print(f"  Search top-k   : {args.search_top_k}")
    print(f"  Timeout (sec)  : {args.timeout_seconds}")
    print(f"  Custom prompt  : {args.prompt or '(default)'}")
    print(f"  Input          : {input_path}")
    print(f"  Output         : {output_path}")
    print("=" * 60)

    # Load source data
    print("\nLoading source data …")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} foods loaded")

    # Create aggregator
    agg = FoodAggregator(
        df,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        batch_size=effective_batch_size,
        search_top_k=args.search_top_k,
        prompt_path=args.prompt,
        checkpoint_dir=checkpoint_dir,
        timeout_seconds=args.timeout_seconds,
    )

    # Resume?
    resume_from = 0
    if args.resume:
        if agg.load_checkpoint():
            resume_from = agg.processed_count
        else:
            print("  (no checkpoint found, starting fresh)")

    # Run
    agg.run(limit=limit, resume_from=resume_from)

    # Save
    agg.save(output_path)


if __name__ == "__main__":
    main()

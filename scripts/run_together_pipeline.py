#!/usr/bin/env python3
"""
Standalone Together AI streaming pipeline runner.

Example:
  export TOGETHER_API_KEY=...
  python scripts/run_together_pipeline.py --mode test
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd

from aggregator import FoodAggregator


def main():
    parser = argparse.ArgumentParser(description="Run Together AI streaming aggregation pipeline")
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    parser.add_argument("--api-key", default=None, help="Together API key (or use TOGETHER_API_KEY env)")
    parser.add_argument("--base-url", default="https://api.together.xyz/v1")
    parser.add_argument("--search-top-k", type=int, default=8)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--estimate-full-size", type=int, default=296000)
    parser.add_argument("--estimated-cost-per-call", type=float, default=0.0, help="optional rough USD estimate per LLM API call")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "pyfooda/data/fooddata.csv"
    if args.mode == "test":
        test_csv = repo_root / "tests/test_fooddata.csv"
        if test_csv.exists():
            input_path = test_csv

    output_path = Path(args.output) if args.output else (
        repo_root / "tests/test_aggregated_together.json" if args.mode == "test" else repo_root / "pyfooda/data/foods_aggregated_together.json"
    )
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        repo_root / "tests/checkpoints_together" if args.mode == "test" else repo_root / "checkpoints_together"
    )

    df = pd.read_csv(input_path)
    api_key = args.api_key or __import__("os").getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Missing Together API key. Set TOGETHER_API_KEY or pass --api-key")

    agg = FoodAggregator(
        df,
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        search_top_k=args.search_top_k,
        prompt_path=args.prompt,
        checkpoint_dir=str(checkpoint_dir),
        timeout_seconds=args.timeout_seconds,
    )

    resume_from = 0
    if args.resume and agg.load_checkpoint():
        resume_from = agg.processed_count

    limit = args.limit
    start = time.time()
    agg.run(limit=limit, resume_from=resume_from)
    elapsed = time.time() - start
    agg.save(str(output_path))

    processed = agg.processed_count
    rate = processed / elapsed if elapsed > 0 else 0
    eta_seconds = args.estimate_full_size / rate if rate > 0 else math.inf

    api_calls = agg.stats.get("api_calls", 0)
    est_cost = api_calls * args.estimated_cost_per_call
    est_full_calls = (args.estimate_full_size / processed * api_calls) if processed > 0 else 0
    est_full_cost = est_full_calls * args.estimated_cost_per_call
    summary = {
        "mode": args.mode,
        "provider": "together",
        "model": args.model,
        "input_rows": int(len(df if limit is None else df.iloc[:limit])),
        "processed": processed,
        "api_calls": api_calls,
        "final_groups": len(agg.db),
        "elapsed_seconds": round(elapsed, 2),
        "items_per_second": round(rate, 4),
        "eta_full_dataset_seconds": round(eta_seconds, 2) if math.isfinite(eta_seconds) else None,
        "eta_full_dataset_hms": time.strftime("%Hh %Mm %Ss", time.gmtime(eta_seconds)) if math.isfinite(eta_seconds) else None,
        "estimated_cost_per_call_usd": args.estimated_cost_per_call,
        "estimated_cost_usd": round(est_cost, 6),
        "estimated_full_api_calls": round(est_full_calls, 2),
        "estimated_full_cost_usd": round(est_full_cost, 6),
        "output_json": str(output_path),
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    print("\nTogether metrics:")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Standalone local Ollama streaming pipeline runner.

Example:
  python scripts/run_ollama_pipeline.py --mode test --model qwen2.5:3b
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
    parser = argparse.ArgumentParser(description="Run local Ollama streaming aggregation pipeline")
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--search-top-k", type=int, default=8)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--estimate-full-size", type=int, default=296000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "pyfooda/data/fooddata.csv"
    if args.mode == "test":
        test_csv = repo_root / "tests/test_fooddata.csv"
        if test_csv.exists():
            input_path = test_csv

    output_path = Path(args.output) if args.output else (
        repo_root / "tests/test_aggregated_ollama_stream.json" if args.mode == "test" else repo_root / "pyfooda/data/foods_aggregated_ollama.json"
    )
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        repo_root / "tests/checkpoints_ollama_stream" if args.mode == "test" else repo_root / "checkpoints_ollama_stream"
    )

    df = pd.read_csv(input_path)
    agg = FoodAggregator(
        df,
        model=args.model,
        api_key="ollama",
        base_url=args.base_url,
        search_top_k=args.search_top_k,
        prompt_path=args.prompt,
        checkpoint_dir=str(checkpoint_dir),
        timeout_seconds=args.timeout_seconds,
    )

    resume_from = 0
    if args.resume and agg.load_checkpoint():
        resume_from = agg.processed_count

    start = time.time()
    agg.run(limit=args.limit, resume_from=resume_from)
    elapsed = time.time() - start
    agg.save(str(output_path))

    processed = agg.processed_count
    rate = processed / elapsed if elapsed > 0 else 0
    eta_seconds = args.estimate_full_size / rate if rate > 0 else math.inf

    summary = {
        "mode": args.mode,
        "provider": "ollama",
        "model": args.model,
        "input_rows": int(len(df if args.limit is None else df.iloc[:args.limit])),
        "processed": processed,
        "api_calls": agg.stats.get("api_calls", 0),
        "final_groups": len(agg.db),
        "elapsed_seconds": round(elapsed, 2),
        "items_per_second": round(rate, 4),
        "eta_full_dataset_seconds": round(eta_seconds, 2) if math.isfinite(eta_seconds) else None,
        "eta_full_dataset_hms": time.strftime("%Hh %Mm %Ss", time.gmtime(eta_seconds)) if math.isfinite(eta_seconds) else None,
        "estimated_cost_usd": 0.0,
        "output_json": str(output_path),
    }

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    print("\nOllama metrics:")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

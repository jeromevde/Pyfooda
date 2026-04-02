#!/usr/bin/env python3
"""
Standalone aggregation runner (batched multi-item LLM calls).

Example:
  python scripts/run_aggregation.py --mode test --batch-size 8
  python scripts/run_aggregation.py --mode full --batch-size 8 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd

from aggregator import (
    FoodAggregator,
    _build_item_prompt,
    _call_llm_batch,
    _data_type_rank,
    _nutrient_fingerprint,
    _parse_llm_decision,
)


def _resolve_input_output(repo_root: Path, mode: str, output: str | None, checkpoint_dir: str | None):
    default_input = repo_root / "pyfooda/data/fooddata.csv"
    default_output = repo_root / "tests/batch_test_aggregated.json" if mode == "test" else repo_root / "pyfooda/data/foods_aggregated_batch.json"
    default_ckpt = repo_root / ("tests/checkpoints_batch" if mode == "test" else "checkpoints_batch")

    input_path = default_input
    if mode == "test":
        test_csv = repo_root / "tests/test_fooddata.csv"
        if test_csv.exists():
            input_path = test_csv

    return input_path, Path(output) if output else default_output, Path(checkpoint_dir) if checkpoint_dir else default_ckpt


def _apply_decision(agg: FoodAggregator, food: dict, row: pd.Series, decision: dict, create_aliases: dict[str, str] | None = None):
    name = food["name"]
    if decision["action"] == "CREATE":
        create_name = decision.get("name", name)
        if create_aliases is not None:
            key = create_name.strip().lower()
            create_name = create_aliases.setdefault(key, create_name)
        name_key = create_name.strip().lower()
        if name_key in agg._name_to_id:
            existing_id = agg._name_to_id[name_key]
            result = agg._do_add(existing_id, row)
            if result == "ok":
                agg.stats["added"] += 1
            else:
                agg._do_create(create_name, row)
                agg.stats["created"] += 1
        else:
            agg._do_create(create_name, row)
            agg.stats["created"] += 1

    elif decision["action"] == "ADD":
        target = decision.get("target_id")
        target_name = decision.get("target_name")
        valid_add_ids = food.get("_valid_add_ids", set())

        if target is not None and target not in valid_add_ids:
            target = None

        if target is None and target_name:
            matches = agg.index.search(target_name, top_k=1)
            if matches and matches[0]["score"] > 0.75 and matches[0]["id"] in valid_add_ids:
                target = matches[0]["id"]

        if target and target in agg.db:
            result = agg._do_add(target, row)
            if result == "ok":
                agg.stats["added"] += 1
            elif result == "foundation_locked":
                agg.stats["ignored"] += 1
            else:
                agg._do_create(target_name or name, row)
                agg.stats["created"] += 1
        else:
            agg._do_create(target_name or name, row)
            agg.stats["created"] += 1

    elif decision["action"] == "IGNORE":
        agg.stats["ignored"] += 1
    else:
        agg.stats["errors"] += 1
        agg.stats["ignored"] += 1


def run_batched(
    agg: FoodAggregator,
    *,
    limit: int | None,
    resume_from: int,
    batch_size: int,
):
    df = agg.source_df
    df = df[df["foodName"].notna() & (df["foodName"].astype(str) != "nan")].reset_index(drop=True)
    df["_dtype_rank"] = df["data_type"].apply(_data_type_rank)
    df = df.sort_values(["_dtype_rank", "food_category"], ascending=[True, True], na_position="last").reset_index(drop=True)
    if limit:
        df = df.iloc[:limit]

    total = len(df)
    start_ts = time.time()

    i = resume_from
    while i < total:
        chunk = []
        end = min(i + batch_size, total)
        for idx in range(i, end):
            row = df.iloc[idx]
            name = str(row.get("foodName", ""))
            cat = str(row.get("food_category", ""))
            search_results = agg.index.search(name, top_k=agg.search_top_k)
            for m in search_results:
                entry = agg.db.get(m["id"])
                if entry:
                    m["nutrients"] = _nutrient_fingerprint(entry["nutrients"], is_dict=True)
                    m["count"] = entry.get("count", 1)

            food = {
                "idx": idx,
                "name": name,
                "category": cat,
                "nutrients_str": _nutrient_fingerprint(row),
                "search_results": search_results,
                "_valid_add_ids": {m["id"] for m in search_results},
            }
            chunk.append((idx, row, food))

        user_msg = "\n\n".join(_build_item_prompt(food) for _, _, food in chunk)
        user_msg += (
            "\n\nReturn exactly one decision per item, format: [idx] CREATE <name> | ADD <id> | IGNORE"
            "\nImportant: if multiple incoming items should end up in the SAME group, use the EXACT same CREATE name text for all of them."
            "\nDo not invent slightly different synonyms for the same target group in one batch."
        )

        try:
            raw = _call_llm_batch(
                agg.system_prompt,
                user_msg,
                api_key=agg.api_key,
                base_url=agg.base_url,
                model=agg.model,
                timeout_seconds=agg.timeout_seconds,
            )
            agg.stats["api_calls"] += 1
        except Exception:
            # fallback: mark all as ignored on batch failure
            for _idx, _row, _food in chunk:
                agg.stats["errors"] += 1
                agg.stats["ignored"] += 1
                agg.processed_count += 1
            i = end
            continue

        # parse per line by idx, fallback per-item parse on whole raw
        line_map = {}
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("[") and "]" in ln:
                try:
                    idx = int(ln[1:ln.index("]")])
                    line_map[idx] = ln
                except Exception:
                    pass

        create_aliases: dict[str, str] = {}
        for idx, row, food in chunk:
            decision = None
            if idx in line_map:
                decision = _parse_llm_decision(line_map[idx], idx)
            if decision is None:
                decision = _parse_llm_decision(raw, idx)
            if decision is None:
                agg.stats["errors"] += 1
                agg.stats["ignored"] += 1
            else:
                _apply_decision(agg, food, row, decision, create_aliases=create_aliases)
            agg.processed_count += 1

        if agg.processed_count % 500 == 0:
            agg._save_checkpoint()
        i = end

    agg._apply_canonical_postpass()
    agg._save_checkpoint()
    agg._print_summary()

    elapsed = time.time() - start_ts
    return {"elapsed_seconds": elapsed, "processed": agg.processed_count, "api_calls": agg.stats.get("api_calls", 0), "final_groups": len(agg.db)}


def main():
    parser = argparse.ArgumentParser(description="Run aggregation pipeline (batched)")
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--model", default="google/gemini-2.0-flash-lite-001")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
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
    input_path, output_path, checkpoint_dir = _resolve_input_output(repo_root, args.mode, args.output, args.checkpoint_dir)

    args.base_url = args.base_url or "https://openrouter.ai/api/v1"
    args.api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not args.api_key:
        raise ValueError("Missing OPENROUTER_API_KEY (or pass --api-key)")

    df = pd.read_csv(input_path)

    agg = FoodAggregator(
        df,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        batch_size=args.batch_size,
        search_top_k=args.search_top_k,
        prompt_path=args.prompt,
        checkpoint_dir=str(checkpoint_dir),
        timeout_seconds=args.timeout_seconds,
    )

    resume_from = 0
    if args.resume and agg.load_checkpoint():
        resume_from = agg.processed_count

    limit = args.limit
    if args.mode == "full":
        limit = limit or None

    result = run_batched(agg, limit=limit, resume_from=resume_from, batch_size=args.batch_size)
    agg.save(str(output_path))

    rate = result["processed"] / result["elapsed_seconds"] if result["elapsed_seconds"] > 0 else 0
    eta_seconds = args.estimate_full_size / rate if rate > 0 else math.inf

    est_cost = result["api_calls"] * args.estimated_cost_per_call
    est_full_calls = (args.estimate_full_size / result["processed"] * result["api_calls"]) if result["processed"] > 0 else 0
    est_full_cost = est_full_calls * args.estimated_cost_per_call
    summary = {
        "mode": args.mode,
        "provider": "openrouter",
        "model": args.model,
        "batch_size": args.batch_size,
        "input_rows": int(len(df if limit is None else df.iloc[:limit])),
        "processed": result["processed"],
        "api_calls": result["api_calls"],
        "final_groups": result["final_groups"],
        "elapsed_seconds": round(result["elapsed_seconds"], 2),
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
    print("\nBatching metrics:")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

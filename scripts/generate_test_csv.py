#!/usr/bin/env python3
"""
Regenerate tests/test_fooddata.csv from tests/test_set.json.

The test CSV is a purpose-built input file containing exactly the food rows
needed to evaluate every case in test_set.json.  It is NOT a random slice of
fooddata.csv — the sorted full CSV would require processing 10k+ rows before
all test foods appear.

Usage:
    python scripts/generate_test_csv.py
    python scripts/generate_test_csv.py --test-set tests/test_set.json \
        --source pyfooda/data/fooddata.csv --output tests/test_fooddata.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate tests/test_fooddata.csv")
    parser.add_argument("--test-set", default=str(REPO_ROOT / "tests/test_set.json"))
    parser.add_argument("--source", default=str(REPO_ROOT / "pyfooda/data/fooddata.csv"))
    parser.add_argument("--output", default=str(REPO_ROOT / "tests/test_fooddata.csv"))
    args = parser.parse_args()

    with open(args.test_set) as f:
        ts = json.load(f)

    # Collect every food name referenced in any test case
    needed: set[str] = set()
    for case in ts["cases"]:
        if "foods" in case:
            needed.update(case["foods"])
        elif "food" in case:
            needed.add(case["food"])

    print(f"Food names needed by test_set.json: {len(needed)}")

    df = pd.read_csv(args.source)
    df = df[df["foodName"].notna()].copy()

    # Prioritise higher-quality data types so we keep the best row per name
    _DTYPE_RANK = {
        "foundation_food": 0,
        "sr_legacy_food": 1,
        "survey_fndds_food": 1,
        "sub_sample_food": 2,
        "agricultural_acquisition": 2,
        "branded_food": 3,
    }
    df["_rank"] = df["data_type"].apply(lambda d: _DTYPE_RANK.get(str(d).strip(), 9))
    df = df.sort_values("_rank").drop_duplicates(subset=["foodName"], keep="first").drop(columns=["_rank"])

    matched = df[df["foodName"].isin(needed)]
    missing = needed - set(matched["foodName"])

    print(f"Matched in source CSV: {len(matched)}")
    if missing:
        print(f"Not found in source ({len(missing)}):")
        for m in sorted(missing):
            print(f"  {m}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out, index=False)
    print(f"Saved {out}  ({len(matched)} rows)")


if __name__ == "__main__":
    main()

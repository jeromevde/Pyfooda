#!/usr/bin/env python3
"""Recompute per-nutrient support/CI stats and per-source nutrients in ingredients_meta.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nutrient_stats import nutrient_stats, row_nutrients

REPO_ROOT = Path(__file__).resolve().parents[1]
USDA_CSV = REPO_ROOT / "pyfooda/data/fooddata.csv"
META_JSON = REPO_ROOT / "pyfooda/data/ingredients_meta.json"
NUTRIENTS_CSV = REPO_ROOT / "pyfooda/data/nutrients.csv"


def recompute(meta_path: Path = META_JSON, usda_path: Path = USDA_CSV) -> int:
    nutrient_cols = pd.read_csv(NUTRIENTS_CSV)["nutrientName"].tolist()
    meta = json.loads(meta_path.read_text())

    print(f"Loading USDA data from {usda_path}")
    usda = pd.read_csv(usda_path)
    usda["_key"] = usda["foodName"].astype(str).str.lower()
    usda_by_name = usda.groupby("_key", sort=False)

    updated = 0
    for item in meta:
        sources = item.get("sources") or []
        source_count = len(sources)
        if source_count == 0:
            item["nutrient_stats"] = {}
            continue

        rows = []
        for src in sources:
            key = str(src.get("foodName", "")).lower()
            if key not in usda_by_name.groups:
                src["nutrients"] = {}
                continue
            row = usda_by_name.get_group(key).iloc[0]
            src["nutrients"] = row_nutrients(row, nutrient_cols)
            rows.append(row)

        if not rows:
            item["nutrient_stats"] = {}
            continue

        selected = pd.DataFrame(rows)
        item["nutrient_stats"] = nutrient_stats(selected, nutrient_cols, source_count)
        updated += 1

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Updated nutrient stats for {updated}/{len(meta)} ingredients")
    return 0


if __name__ == "__main__":
    raise SystemExit(recompute())

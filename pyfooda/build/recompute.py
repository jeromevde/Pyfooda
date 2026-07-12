#!/usr/bin/env python3
"""Recompute per-nutrient stats and per-source nutrients in ingredients_meta.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pyfooda.build.download_usda import ensure_fooddata
from pyfooda.build.nutrient_stats import average_nutrients, nutrient_stats, row_nutrients
from pyfooda.build.paths import INGREDIENTS_CSV, META_JSON, NUTRIENTS_CSV, USDA_CSV


def recompute(
    meta_path: Path = META_JSON,
    usda_path: Path = USDA_CSV,
    ingredients_path: Path = INGREDIENTS_CSV,
) -> int:
    nutrient_cols = pd.read_csv(NUTRIENTS_CSV)["nutrientName"].tolist()
    meta = json.loads(meta_path.read_text())
    ingredients = pd.read_csv(ingredients_path)

    print(f"Loading USDA data from {usda_path}")
    usda = pd.read_csv(usda_path)
    usda["_key"] = usda["foodName"].astype(str).str.lower()
    usda_by_name = usda.groupby("_key", sort=False)

    updated = 0
    averages: dict[str, dict] = {}

    for item in meta:
        ingredient_id = item["ingredient_id"]
        sources = item.get("sources") or []
        source_count = len(sources)

        if source_count == 0:
            item["nutrient_stats"] = {}
            averages[ingredient_id] = {col: None for col in nutrient_cols}
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
            averages[ingredient_id] = {col: None for col in nutrient_cols}
            continue

        selected = pd.DataFrame(rows)
        item["nutrient_stats"] = nutrient_stats(selected, nutrient_cols, source_count)
        averages[ingredient_id] = average_nutrients(selected, nutrient_cols)
        updated += 1

    meta_path.write_text(json.dumps(meta, indent=2))

    for idx, row in ingredients.iterrows():
        ingredient_id = row["ingredient_id"]
        avg = averages.get(ingredient_id, {})
        for col in nutrient_cols:
            ingredients.at[idx, col] = avg.get(col)

    ingredients.to_csv(ingredients_path, index=False)
    print(f"Updated nutrient stats for {updated}/{len(meta)} ingredients")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute nutrient stats for existing USDA sources")
    p.add_argument("--meta", type=Path, default=META_JSON)
    p.add_argument("--usda", type=Path, default=USDA_CSV)
    p.add_argument("--ingredients", type=Path, default=INGREDIENTS_CSV)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        ensure_fooddata(args.usda)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return recompute(
        meta_path=args.meta,
        usda_path=args.usda,
        ingredients_path=args.ingredients,
    )


if __name__ == "__main__":
    raise SystemExit(main())

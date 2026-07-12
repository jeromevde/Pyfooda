#!/usr/bin/env python3
"""Recompute per-nutrient support/CI stats and per-source nutrients in ingredients_meta.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nutrient_stats import (
    average_nutrients,
    filter_source_records,
    nutrient_stats,
    row_nutrients,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
USDA_CSV = REPO_ROOT / "pyfooda/data/fooddata.csv"
META_JSON = REPO_ROOT / "pyfooda/data/ingredients_meta.json"
INGREDIENTS_CSV = REPO_ROOT / "pyfooda/data/ingredients.csv"
NUTRIENTS_CSV = REPO_ROOT / "pyfooda/data/nutrients.csv"


def recompute(
    meta_path: Path = META_JSON,
    usda_path: Path = USDA_CSV,
    ingredients_path: Path = INGREDIENTS_CSV,
    *,
    top_sources: int = 5,
    min_source_similarity: float = 0.80,
    max_similarity_drop: float = 0.08,
) -> int:
    nutrient_cols = pd.read_csv(NUTRIENTS_CSV)["nutrientName"].tolist()
    meta = json.loads(meta_path.read_text())
    ingredients = pd.read_csv(ingredients_path)

    print(f"Loading USDA data from {usda_path}")
    usda = pd.read_csv(usda_path)
    usda["_key"] = usda["foodName"].astype(str).str.lower()
    usda_by_name = usda.groupby("_key", sort=False)

    updated = 0
    source_counts: dict[str, int] = {}
    top_similarities: dict[str, float | None] = {}
    averages: dict[str, dict] = {}

    for item in meta:
        ingredient_id = item["ingredient_id"]
        sources = filter_source_records(
            item.get("sources") or [],
            top_sources=top_sources,
            min_source_similarity=min_source_similarity,
            max_similarity_drop=max_similarity_drop,
        )
        item["sources"] = sources
        source_count = len(sources)
        source_counts[ingredient_id] = source_count

        if source_count == 0:
            item["nutrient_stats"] = {}
            top_similarities[ingredient_id] = None
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
            top_similarities[ingredient_id] = float(sources[0].get("similarity") or 0)
            averages[ingredient_id] = {col: None for col in nutrient_cols}
            continue

        selected = pd.DataFrame(rows)
        item["nutrient_stats"] = nutrient_stats(selected, nutrient_cols, source_count)
        top_similarities[ingredient_id] = float(sources[0].get("similarity") or 0)
        averages[ingredient_id] = average_nutrients(selected, nutrient_cols)
        updated += 1

    meta_path.write_text(json.dumps(meta, indent=2))

    for idx, row in ingredients.iterrows():
        ingredient_id = row["ingredient_id"]
        ingredients.at[idx, "source_count"] = source_counts.get(ingredient_id, 0)
        top_sim = top_similarities.get(ingredient_id)
        ingredients.at[idx, "top_similarity"] = top_sim
        avg = averages.get(ingredient_id, {})
        for col in nutrient_cols:
            ingredients.at[idx, col] = avg.get(col)

    ingredients.to_csv(ingredients_path, index=False)

    dist = ingredients["source_count"].value_counts().sort_index()
    print(f"Updated nutrient stats for {updated}/{len(meta)} ingredients")
    print(f"Source count distribution: {dict(dist)}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute nutrient stats and filter USDA sources")
    p.add_argument("--meta", type=Path, default=META_JSON)
    p.add_argument("--usda", type=Path, default=USDA_CSV)
    p.add_argument("--ingredients", type=Path, default=INGREDIENTS_CSV)
    p.add_argument("--top-sources", type=int, default=5)
    p.add_argument("--min-source-similarity", type=float, default=0.80)
    p.add_argument("--max-similarity-drop", type=float, default=0.08)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        recompute(
            meta_path=args.meta,
            usda_path=args.usda,
            ingredients_path=args.ingredients,
            top_sources=args.top_sources,
            min_source_similarity=args.min_source_similarity,
            max_similarity_drop=args.max_similarity_drop,
        )
    )

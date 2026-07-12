"""Nutrient coverage statistics across the ingredient database."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def compute_nutrient_coverage(
    meta: list[dict],
    nutrient_cols: list[str],
    nutrient_meta: pd.DataFrame | None = None,
) -> dict:
    """Summarize which nutrients are measured across ingredients with USDA sources."""
    ingredients_with_sources = [item for item in meta if item.get("sources")]
    total = len(ingredients_with_sources)

    if nutrient_meta is None:
        nutrient_meta = pd.DataFrame({"nutrientName": nutrient_cols})

    meta_by_name = {
        row["nutrientName"]: row
        for _, row in nutrient_meta.iterrows()
        if pd.notna(row.get("nutrientName"))
    }

    nutrients = []
    for col in nutrient_cols:
        with_value = 0
        for item in ingredients_with_sources:
            stat = (item.get("nutrient_stats") or {}).get(col, {})
            if stat.get("support", 0) > 0:
                with_value += 1

        info = meta_by_name.get(col, {})
        nutrients.append(
            {
                "name": col,
                "category": info.get("nutrient_category"),
                "unit": info.get("unit_name"),
                "ingredients_with_value": with_value,
                "ingredients_total": total,
                "coverage_pct": round(with_value / total, 4) if total else 0.0,
            }
        )

    nutrients.sort(key=lambda row: (row["coverage_pct"], row["name"]))

    measured = [n for n in nutrients if n["coverage_pct"] >= 0.5]
    sparse = [n for n in nutrients if 0 < n["coverage_pct"] < 0.5]
    missing = [n for n in nutrients if n["coverage_pct"] == 0]

    return {
        "ingredients_with_sources": total,
        "nutrient_count": len(nutrient_cols),
        "nutrients": nutrients,
        "summary": {
            "well_measured": len(measured),
            "partially_measured": len(sparse),
            "rarely_measured": len(missing),
        },
    }


def ingredient_coverage(item: dict, nutrient_cols: list[str]) -> dict:
    """Per-ingredient nutrient coverage from stored nutrient_stats."""
    stats = item.get("nutrient_stats") or {}
    source_count = len(item.get("sources") or [])
    measured = []
    missing = []

    for col in nutrient_cols:
        support = (stats.get(col) or {}).get("support", 0)
        if support > 0:
            measured.append(col)
        else:
            missing.append(col)

    total = len(nutrient_cols)
    return {
        "ingredient_id": item.get("ingredient_id"),
        "display_name": item.get("display_name"),
        "source_count": source_count,
        "nutrients_measured": len(measured),
        "nutrients_total": total,
        "coverage_pct": round(len(measured) / total, 4) if total else 0.0,
        "measured": measured,
        "missing": missing,
    }


def write_coverage_report(
    meta_path: Path,
    nutrients_path: Path,
    output_path: Path,
) -> dict:
    meta = json.loads(meta_path.read_text())
    nutrient_cols = pd.read_csv(nutrients_path)["nutrientName"].tolist()
    nutrients_df = pd.read_csv(nutrients_path)
    report = compute_nutrient_coverage(meta, nutrient_cols, nutrients_df)
    output_path.write_text(json.dumps(report, indent=2))
    return report

#!/usr/bin/env python3
"""Export ingredients database to JSON for the GitHub Pages frontend."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pyfooda.build.paths import INGREDIENTS_CSV, META_JSON, NUTRIENTS_CSV, VERSION_FILE, WEB_JSON


def package_version() -> str:
    return VERSION_FILE.read_text().strip()


def export_web_data(output: Path = WEB_JSON) -> Path:
    meta_items = json.loads(META_JSON.read_text())
    meta_by_id = {item["ingredient_id"]: item for item in meta_items}
    nutrient_cols = pd.read_csv(NUTRIENTS_CSV)["nutrientName"].tolist()
    df = pd.read_csv(INGREDIENTS_CSV)

    foods = []
    for _, row in df.iterrows():
        ingredient_id = row["ingredient_id"]
        nutrients = {}
        for col in nutrient_cols:
            val = row.get(col)
            nutrients[col] = None if pd.isna(val) else float(val)

        meta = meta_by_id.get(ingredient_id, {})
        top_sim = row.get("top_similarity")
        foods.append(
            {
                "ingredient_id": ingredient_id,
                "display_name": row["display_name"],
                "source_count": int(row["source_count"]),
                "top_similarity": None if pd.isna(top_sim) else float(top_sim),
                "nutrients": nutrients,
                "nutrient_stats": meta.get("nutrient_stats", {}),
                "sources": meta.get("sources", []),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "count": len(foods),
            "vocabulary": "epicure",
            "version": package_version(),
        },
        "foods": foods,
    }
    output.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {len(foods)} ingredients to {output}")
    return output


def main() -> int:
    export_web_data()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

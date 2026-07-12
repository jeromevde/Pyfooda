#!/usr/bin/env python3
"""Build the Pyfooda ingredient database from Epicure vocabulary + USDA FoodData Central."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pyfooda.build.download_usda import ensure_fooddata
from pyfooda.build.nutrient_stats import average_nutrients, nutrient_stats, select_source_rows, source_record
from pyfooda.build.paths import INGREDIENTS_CSV, META_JSON, NUTRIENTS_CSV, USDA_CSV, VOCAB_JSON

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_vocabulary(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_nutrient_columns(path: Path) -> list[str]:
    return pd.read_csv(path)["nutrientName"].tolist()


def nutrient_coverage(row: pd.Series, nutrient_cols: list[str]) -> int:
    if "number_of_nutrients" in row.index and pd.notna(row["number_of_nutrients"]):
        return int(row["number_of_nutrients"])
    return int(row[nutrient_cols].notna().sum())


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def build_database(
    usda_path: Path,
    vocab_path: Path,
    nutrients_path: Path,
    output_path: Path,
    meta_path: Path,
    top_k: int,
    top_sources: int,
    min_similarity: float,
    min_source_similarity: float,
    max_similarity_drop: float,
    min_macro_agreement: float,
    batch_size: int,
) -> pd.DataFrame:
    vocab = load_vocabulary(vocab_path)
    nutrient_cols = load_nutrient_columns(nutrients_path)

    print(f"Loading USDA data from {usda_path}")
    usda = pd.read_csv(usda_path)
    usda = usda[usda["foodName"].notna() & (usda["foodName"].astype(str).str.strip() != "")]
    usda = usda.reset_index(drop=True)
    print(f"  {len(usda):,} USDA rows")

    print(f"Loading embedding model {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    usda_names = usda["foodName"].astype(str).tolist()
    print("Embedding USDA food names...")
    usda_emb = embed_texts(model, usda_names, batch_size)
    usda_emb = np.nan_to_num(usda_emb, nan=0.0)

    ingredient_texts = [item["name"] for item in vocab]
    print("Embedding Epicure ingredient names...")
    ing_emb = embed_texts(model, ingredient_texts, batch_size)
    ing_emb = np.nan_to_num(ing_emb, nan=0.0)

    print("Computing similarity...")
    sims = ing_emb @ usda_emb.T

    rows = []
    meta = []

    for i, item in enumerate(tqdm(vocab, desc="Building ingredients")):
        ingredient_id = item["id"]
        display_name = item["name"]
        scores = sims[i]
        candidate_idx = np.argpartition(scores, -top_k)[-top_k:]
        candidate_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        candidate_idx = [j for j in candidate_idx if scores[j] >= min_similarity]

        if not candidate_idx:
            rows.append(
                {
                    "ingredient_id": ingredient_id,
                    "display_name": display_name,
                    "source_count": 0,
                    "top_similarity": None,
                    **{col: None for col in nutrient_cols},
                }
            )
            meta.append(
                {
                    "ingredient_id": ingredient_id,
                    "display_name": display_name,
                    "sources": [],
                    "nutrient_stats": {},
                }
            )
            continue

        candidates = usda.iloc[candidate_idx].copy()
        candidates["_similarity"] = [float(scores[j]) for j in candidate_idx]
        candidates["_coverage"] = candidates.apply(
            lambda r: nutrient_coverage(r, nutrient_cols), axis=1
        )
        selected = select_source_rows(
            candidates,
            top_sources=top_sources,
            min_source_similarity=min_source_similarity,
            max_similarity_drop=max_similarity_drop,
            min_macro_agreement=min_macro_agreement,
        )
        avg = average_nutrients(selected, nutrient_cols)
        stats = nutrient_stats(selected, nutrient_cols, len(selected))

        rows.append(
            {
                "ingredient_id": ingredient_id,
                "display_name": display_name,
                "source_count": len(selected),
                "top_similarity": float(selected["_similarity"].iloc[0]),
                **avg,
            }
        )
        meta.append(
            {
                "ingredient_id": ingredient_id,
                "display_name": display_name,
                "sources": [
                    source_record(
                        r,
                        nutrient_cols,
                        float(r["_similarity"]),
                        int(r["_coverage"]),
                    )
                    for _, r in selected.iterrows()
                ],
                "nutrient_stats": stats,
            }
        )

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    matched = int((out_df["source_count"] > 0).sum())
    print(f"Wrote {len(out_df)} ingredients to {output_path}")
    print(f"  {matched}/{len(out_df)} ingredients have USDA sources")
    print(f"  Source details: {meta_path}")
    return out_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Pyfooda ingredient database")
    p.add_argument("--usda", type=Path, default=USDA_CSV)
    p.add_argument("--vocab", type=Path, default=VOCAB_JSON)
    p.add_argument("--nutrients", type=Path, default=NUTRIENTS_CSV)
    p.add_argument("--output", type=Path, default=INGREDIENTS_CSV)
    p.add_argument("--meta", type=Path, default=META_JSON)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-sources", type=int, default=5)
    p.add_argument("--min-similarity", type=float, default=0.50)
    p.add_argument("--min-source-similarity", type=float, default=0.80)
    p.add_argument("--max-similarity-drop", type=float, default=0.08)
    p.add_argument("--min-macro-agreement", type=float, default=0.85)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        ensure_fooddata(args.usda)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    for path in (args.usda, args.vocab, args.nutrients):
        if not path.exists():
            print(f"Missing required file: {path}", file=sys.stderr)
            return 1

    build_database(
        usda_path=args.usda,
        vocab_path=args.vocab,
        nutrients_path=args.nutrients,
        output_path=args.output,
        meta_path=args.meta,
        top_k=args.top_k,
        top_sources=args.top_sources,
        min_similarity=args.min_similarity,
        min_source_similarity=args.min_source_similarity,
        max_similarity_drop=args.max_similarity_drop,
        min_macro_agreement=args.min_macro_agreement,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
LLM-driven curation of USDA source entries for Pyfooda ingredients.

Retrieves top-K USDA candidates for each ingredient using TF-IDF similarity on
food names, then asks a language model to select which entries represent the
correct whole/plain food.  Selected sources are written to ingredients_meta.json;
run `python -m pyfooda.build.recompute` afterward to rebuild ingredients.csv.

Usage:
    # Curate specific ingredients (recommended for targeted fixes):
    python -m pyfooda.build.curate --ids fennel,lard,potato,beef,chicken

    # Curate all ingredients (full database rebuild via LLM):
    python -m pyfooda.build.curate --all

    # Preview candidates without making LLM calls:
    python -m pyfooda.build.curate --ids fennel --dry-run

    # Use a different model or provider:
    python -m pyfooda.build.curate --all --model gpt-4o --base-url https://models.inference.ai.azure.com

Environment:
    OPENAI_API_KEY   API key for OpenAI (or compatible provider)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

from pyfooda.build.download_usda import ensure_fooddata
from pyfooda.build.nutrient_stats import (
    _is_nutrient_outlier,
    average_nutrients,
    coerce_usda_nutrients,
    nutrient_stats,
    row_nutrients,
)
from pyfooda.build.paths import (
    COVERAGE_JSON,
    INGREDIENTS_CSV,
    META_JSON,
    NUTRIENTS_CSV,
    USDA_CSV,
    VOCAB_JSON,
)

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a nutritional database curator. For each food ingredient you will be
shown USDA FoodData Central candidate entries. Select the ones that best
represent the plain, whole form of that ingredient as typically purchased at a
grocery store.

Rules:
- Prefer raw or minimally processed entries (raw > plain cooked > processed).
- Reject entries for a different part of the food (fennel seed ≠ fennel bulb,
  chicken skin ≠ chicken meat, beef liver ≠ beef).
- Reject composite dishes, fast food, baby food, and fortified supplements.
- Prefer data types: sr_legacy_food, foundation_food, survey_fndds_food.
- Avoid exact duplicates; pick the most representative form.
- Select between 1 and 5 entries. Return ONLY a JSON array of 0-based indices.
  Example response:  [0, 2]"""

_USER = """\
Ingredient: {display_name}  (id: {ingredient_id})

Candidates:
{candidates_text}

Which indices represent "{display_name}" as a plain grocery store ingredient?
Return only a JSON array of indices, no explanation."""

_KEY_NUTRIENTS = ["Energy", "Protein", "Total fat", "Carbohydrate", "Iron", "Calcium"]


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

_AUTHORITATIVE = frozenset({"sr_legacy_food", "foundation_food", "survey_fndds_food"})


def build_tfidf_index(usda: pd.DataFrame) -> tuple[TfidfVectorizer, object, pd.DataFrame]:
    """
    Fit a TF-IDF index over authoritative USDA rows only (sr_legacy, foundation,
    survey).  Branded products (284K+ rows) are excluded because their names are
    often generic labels ("BEEF", "CHICKEN") that swamp keyword-based retrieval.
    Returns (vectorizer, matrix, authoritative_subset).
    """
    auth = usda[usda["data_type"].isin(_AUTHORITATIVE)].reset_index(drop=True)
    names = auth["foodName"].fillna("").astype(str).tolist()
    vect = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    mat = vect.fit_transform(names)
    return vect, mat, auth


def get_candidates(
    auth: pd.DataFrame,
    vect: TfidfVectorizer,
    mat,
    query: str,
    top_k: int,
) -> pd.DataFrame:
    """Return up to top_k authoritative USDA rows ranked by TF-IDF cosine similarity."""
    q_vec = vect.transform([query])
    scores = cosine_similarity(q_vec, mat)[0]
    k = min(top_k, len(scores))
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    result = auth.iloc[idx].copy()
    result["_tfidf"] = scores[idx]
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_val(row: pd.Series, col: str) -> str:
    v = row.get(col)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.1f}"


def format_candidates(df: pd.DataFrame) -> str:
    lines = []
    for i, (_, r) in enumerate(df.iterrows()):
        dt = str(r.get("data_type", "?"))[:4]
        name = str(r.get("foodName", ""))[:65]
        cat = str(r.get("food_category", ""))[:30]
        nutrients = "  ".join(
            f"{c.split()[0]}={_fmt_val(r, c)}" for c in _KEY_NUTRIENTS
        )
        lines.append(f"{i:2d}. [{dt}] {name:<65} | {nutrients} | {cat}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def ask_llm(
    client: "OpenAI",
    model: str,
    ingredient_id: str,
    display_name: str,
    candidates: pd.DataFrame,
) -> list[int]:
    """Ask the LLM to select the best candidate indices. Returns a list of ints."""
    candidates_text = format_candidates(candidates)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER.format(
                ingredient_id=ingredient_id,
                display_name=display_name,
                candidates_text=candidates_text,
            )},
        ],
        temperature=0,
        max_tokens=64,
    )
    raw = response.choices[0].message.content.strip()
    m = re.search(r"\[[\d\s,]*\]", raw)
    if not m:
        return []
    try:
        indices = json.loads(m.group())
        return [int(i) for i in indices if isinstance(i, (int, float)) and 0 <= int(i) < len(candidates)]
    except (json.JSONDecodeError, ValueError):
        return []


async def ask_llm_async(
    client,
    model: str,
    ingredient_id: str,
    display_name: str,
    candidates: pd.DataFrame,
) -> list[int]:
    """Async wrapper — runs the blocking OpenAI call in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, ask_llm, client, model, ingredient_id, display_name, candidates
    )


# ---------------------------------------------------------------------------
# Meta update
# ---------------------------------------------------------------------------

def update_meta_entry(
    meta: list[dict],
    ingredient_id: str,
    display_name: str,
    selected: pd.DataFrame,
    nutrient_cols: list[str],
) -> None:
    """Replace sources for ingredient_id in-place in the meta list."""
    sources = [
        {
            "foodName": str(r["foodName"]),
            "data_type": str(r.get("data_type", "")),
            "similarity": float(r.get("_tfidf", 0.0)),
            "nutrient_coverage": int(r.get("number_of_nutrients") or 0),
            "nutrients": row_nutrients(r, nutrient_cols),
        }
        for _, r in selected.iterrows()
    ]

    if not selected.empty:
        coerced = coerce_usda_nutrients(selected, nutrient_cols)
        stats = nutrient_stats(coerced, nutrient_cols, len(selected))
    else:
        stats = {}

    for item in meta:
        if item["ingredient_id"] == ingredient_id:
            item["sources"] = sources
            item["nutrient_stats"] = stats
            return

    meta.append({
        "ingredient_id": ingredient_id,
        "display_name": display_name,
        "sources": sources,
        "nutrient_stats": stats,
    })


# ---------------------------------------------------------------------------
# Main curation loop
# ---------------------------------------------------------------------------

def curate(
    ids: list[str] | None,
    *,
    model: str,
    base_url: str | None,
    api_key: str | None,
    top_k: int,
    max_sources: int,
    dry_run: bool,
    workers: int = 1,
) -> int:
    if not dry_run and OpenAI is None:
        print("openai package not installed.  Run: pip install openai", file=sys.stderr)
        return 1

    print("Loading USDA data…")
    ensure_fooddata(USDA_CSV)
    usda = pd.read_csv(USDA_CSV)
    usda = usda[usda["foodName"].notna() & (usda["foodName"].str.strip() != "")].reset_index(drop=True)

    nutrient_cols = pd.read_csv(NUTRIENTS_CSV)["nutrientName"].tolist()
    vocab: list[dict] = json.loads(VOCAB_JSON.read_text())
    meta: list[dict] = json.loads(META_JSON.read_text())

    if ids is not None:
        id_set = set(ids)
        vocab = [v for v in vocab if v["id"] in id_set]
        if not vocab:
            print(f"No vocabulary entries found for ids: {ids}", file=sys.stderr)
            return 1

    print(f"Building TF-IDF index over authoritative USDA rows…")
    vect, mat, auth = build_tfidf_index(usda)
    print(f"  {len(auth):,} sr_legacy / foundation / survey rows indexed")

    client = None
    if not dry_run:
        client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    # Build candidate lists upfront (fast, CPU-only)
    tasks: list[tuple[dict, pd.DataFrame]] = []
    for item in vocab:
        candidates = get_candidates(auth, vect, mat, item["name"], top_k)
        outlier_mask = candidates.apply(_is_nutrient_outlier, axis=1)
        candidates = candidates[~outlier_mask].reset_index(drop=True)
        tasks.append((item, candidates))

    if dry_run:
        for item, candidates in tasks:
            print(f"\n{'─' * 70}")
            print(f"  {item['name']}  ({item['id']})")
            if candidates.empty:
                print("  No candidates found.")
            else:
                print(format_candidates(candidates))
        return 0

    # Run LLM selection — sequential or concurrent
    results: dict[str, list[int]] = {}

    if workers <= 1:
        for item, candidates in tasks:
            iid = item["id"]
            name = item["name"]
            if candidates.empty:
                print(f"  {name}: no candidates — skipping")
                continue
            selected_idx = ask_llm(client, model, iid, name, candidates)
            results[iid] = selected_idx
            names = [str(candidates.iloc[i]["foodName"]) for i in selected_idx[:max_sources]]
            print(f"  {name}: {names}")
    else:
        sem = asyncio.Semaphore(workers)

        async def _process(item: dict, candidates: pd.DataFrame) -> None:
            iid = item["id"]
            name = item["name"]
            if candidates.empty:
                return
            async with sem:
                selected_idx = await ask_llm_async(client, model, iid, name, candidates)
            results[iid] = selected_idx
            names = [str(candidates.iloc[i]["foodName"]) for i in selected_idx[:max_sources]]
            print(f"  {name}: {names}")

        async def _run_all() -> None:
            await asyncio.gather(*[_process(item, cands) for item, cands in tasks])

        asyncio.run(_run_all())

    # Apply results to meta
    changed = 0
    task_map = {item["id"]: (item, cands) for item, cands in tasks}
    for iid, selected_idx in results.items():
        if not selected_idx:
            continue
        item, candidates = task_map[iid]
        selected_idx = selected_idx[:max_sources]
        selected = candidates.iloc[selected_idx].reset_index(drop=True)
        update_meta_entry(meta, iid, item["name"], selected, nutrient_cols)
        changed += 1

    if changed > 0:
        META_JSON.write_text(json.dumps(meta, indent=2))
        print(f"\n✓ Updated {changed} ingredient(s) in {META_JSON.name}")
        skipped = len(tasks) - changed
        if skipped:
            print(f"  {skipped} skipped (no candidates or no LLM selection)")
        print("\nNext step:  python -m pyfooda.build.recompute")
    else:
        print("\nNo changes written.")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-driven curation of USDA source entries for Pyfooda ingredients."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", help="Comma-separated ingredient IDs to curate (e.g. fennel,lard)")
    group.add_argument("--all", action="store_true", help="Curate all ingredients")

    p.add_argument("--model", default="gpt-4o-mini", help="LLM model name (default: gpt-4o-mini)")
    p.add_argument("--base-url", default=None, help="Override OpenAI base URL")
    p.add_argument("--api-key", default=None, help="API key (falls back to OPENAI_API_KEY env var)")
    p.add_argument("--top-k", type=int, default=20, help="USDA candidates per ingredient (default: 20; use 50 for common/short names)")
    p.add_argument("--max-sources", type=int, default=5, help="Max sources to select per ingredient (default: 5)")
    p.add_argument("--workers", type=int, default=1, help="Concurrent LLM requests (default: 1; use 20 for a fast full rebuild)")
    p.add_argument("--dry-run", action="store_true", help="Print candidates without making LLM calls")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    return curate(
        ids=ids,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        top_k=args.top_k,
        max_sources=args.max_sources,
        dry_run=args.dry_run,
        workers=args.workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())

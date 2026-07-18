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
- Prefer entries with complete nutrient profiles (Energy present, high coverage).
- Reject entries for a different part of the food (fennel seed ≠ fennel bulb,
  chicken skin ≠ chicken meat, beef liver ≠ beef, potato skin ≠ whole potato).
- Reject a different product class that only shares a word
  (wine vinegar ≠ wine; fruit cereal ≠ fruit; flavored milk ≠ milk;
  oil blends/sprays ≠ plain cooking oil; baby produce ≠ ordinary produce).
- Reject composite dishes, candies, sauces, condiments, fast food, baby food,
  and fortified supplements unless the ingredient itself is that product.
- Prefer data types: sr_legacy_food, foundation_food, survey_fndds_food.
- Avoid exact duplicates; pick the most representative form.
- When a close generic equivalent exists, prefer it over returning empty
  (e.g. sparkling wine for prosecco, phyllo for yufka, zucchini for courgette).
- Return [] ONLY if every candidate is clearly the wrong food.
- Otherwise select between 1 and 5 entries.
- Return ONLY a JSON array of 0-based indices. Example: [0, 2]"""

_USER = """\
Ingredient: {display_name}  (id: {ingredient_id})

Candidates:
{candidates_text}

Which indices best represent "{display_name}" (or its closest plain grocery equivalent)?
Return [] only if all candidates are clearly wrong. Return only a JSON array of indices."""

_SYSTEM_BATCH = """\
You are a nutritional database curator. You will receive several ingredients at once.
For each ingredient, select USDA candidate indices that best represent its plain,
whole grocery-store form (or closest generic equivalent).

Rules:
- Prefer raw / minimally processed; reject wrong parts, dishes, and product classes
  that only share a word.
- Prefer sr_legacy_food / foundation_food / survey_fndds_food.
- Use a close generic equivalent when needed (prosecco→sparkling wine).
- Return [] for an ingredient ONLY if every candidate is clearly wrong.
- Return ONLY a JSON object mapping ingredient_id → array of 0-based indices.
  Example: {"potato":[0,2],"fennel":[1]}"""

_USER_BATCH = """\
Select indices for each ingredient below.

{blocks}

Return ONLY a JSON object mapping each ingredient_id to an array of indices."""

_KEY_NUTRIENTS = ["Energy", "Protein", "Total fat", "Carbohydrate", "Fiber", "Iron", "Calcium"]


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


def _query_variants(query: str) -> list[str]:
    """Expand a short ingredient name with simple plural/singular variants for retrieval."""
    q = query.strip()
    variants = [q]
    lower = q.lower()
    if lower.endswith("ies") and len(lower) > 4:
        variants.append(q[:-3] + "y")
    elif lower.endswith("oes") and len(lower) > 4:
        variants.append(q[:-2])  # potatoes -> potato
    elif lower.endswith("es") and len(lower) > 3:
        variants.append(q[:-2])
        variants.append(q[:-1])
    elif lower.endswith("s") and len(lower) > 3 and not lower.endswith("ss"):
        variants.append(q[:-1])
    else:
        # strawberry -> strawberries; potato -> potatoes; oil -> oils
        if re.search(r"[^aeiou]y$", lower):
            variants.append(q[:-1] + "ies")
        elif lower.endswith(("s", "x", "z", "ch", "sh")):
            variants.append(q + "es")
        elif lower.endswith("o"):
            variants.append(q + "es")
            variants.append(q + "s")
        else:
            variants.append(q + "s")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _headword_boost(food_name: str, variants: list[str]) -> float:
    """
    Boost USDA rows whose primary name tokens are the ingredient itself.

    This keeps 'Strawberries, raw' ahead of 'Strawberry milk' / 'Pie, strawberry'
    and 'Squash, summer, zucchini…' findable from 'zucchini', without category lists.
    """
    name = str(food_name or "").lower()
    if not name:
        return 0.0
    # Tokens from the descriptive name before preparation notes.
    primary = re.split(r"\b(raw|cooked|frozen|canned|boiled|baked|roasted)\b", name, maxsplit=1)[0]
    primary_tokens = re.findall(r"[a-z0-9]+", primary)
    head = re.split(r"[,(/]", name, maxsplit=1)[0].strip()
    head_tokens = head.split()
    boost = 0.0
    for v in variants:
        vl = v.lower()
        vl_stem = vl.rstrip("s")
        if head == vl or head.rstrip("s") == vl_stem:
            boost = max(boost, 0.55)
        elif head_tokens and (head_tokens[0] == vl or head_tokens[0].rstrip("s") == vl_stem):
            boost = max(boost, 0.40)
        elif any(t == vl or t.rstrip("s") == vl_stem for t in primary_tokens):
            boost = max(boost, 0.45)
        elif re.search(rf"\b{re.escape(vl)}\b", name):
            boost = max(boost, 0.08)
    # Prefer plain forms in the name.
    if re.search(r"\b(raw|fresh|nfs)\b", name):
        boost += 0.05
    if re.search(
        r"\b(milk|yogurt|juice|pie|cake|candy|syrup|sauce|soup|cereal|bread|muffin|"
        r"pickle|pickled|topping|sundae|shake)\b",
        name,
    ):
        boost -= 0.25
    return boost


def get_candidates(
    auth: pd.DataFrame,
    vect: TfidfVectorizer,
    mat,
    query: str,
    top_k: int,
) -> pd.DataFrame:
    """Return up to top_k authoritative USDA rows ranked by TF-IDF + headword match."""
    variants = _query_variants(query)
    q_vec = vect.transform(variants)
    tfidf_scores = cosine_similarity(q_vec, mat).max(axis=0)

    # Pull a wider TF-IDF pool, then re-rank with headword boost so short names
    # like "strawberry" surface 'Strawberries, raw' instead of flavored products.
    pool = min(max(top_k * 8, 200), len(tfidf_scores))
    idx = np.argpartition(tfidf_scores, -pool)[-pool:]
    names = auth.iloc[idx]["foodName"].astype(str).tolist()
    boosts = np.array([_headword_boost(n, variants) for n in names], dtype=float)
    combined = tfidf_scores[idx] + boosts
    order = np.argsort(combined)[::-1][:top_k]
    picked = idx[order]
    result = auth.iloc[picked].copy()
    result["_tfidf"] = combined[order]
    return result.reset_index(drop=True)


def filter_candidate_quality(candidates: pd.DataFrame) -> pd.DataFrame:
    """Drop incomplete / outlier rows before the LLM sees them."""
    if candidates.empty:
        return candidates
    energy = pd.to_numeric(candidates.get("Energy"), errors="coerce")
    coverage = pd.to_numeric(candidates.get("number_of_nutrients"), errors="coerce").fillna(0)
    # Authoritative rows without Energy are usually incomplete assay stubs.
    keep = energy.notna() & (energy > 0) & (coverage >= 10)
    outlier_mask = candidates.apply(_is_nutrient_outlier, axis=1)
    return candidates[keep & ~outlier_mask].reset_index(drop=True)


def retrieval_is_weak(candidates: pd.DataFrame, query: str) -> bool:
    """True when TF-IDF/headword retrieval likely missed the plain food form."""
    if candidates.empty:
        return True
    variants = _query_variants(query)
    boosts = [_headword_boost(str(n), variants) for n in candidates["foodName"].tolist()]
    return max(boosts) < 0.35


def expand_search_queries(client: "OpenAI", model: str, display_name: str) -> list[str]:
    """Ask the LLM for alternate USDA-style search phrases when retrieval is weak."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You help search USDA FoodData Central. Given an ingredient, return up to 5 "
                    "alternate English food names that might appear in USDA for the plain grocery "
                    "form. Prefer generic names (e.g. sparkling wine for prosecco, zucchini for "
                    "courgette). Return ONLY a JSON array of strings."
                ),
            },
            {
                "role": "user",
                "content": f"Ingredient: {display_name}\nReturn JSON array of search names.",
            },
        ],
        temperature=0,
        max_tokens=128,
    )
    content = response.choices[0].message.content if response.choices else None
    raw = (content or "").strip()
    if not raw:
        return []
    m = re.search(r"\[[^\]]*\]", raw, flags=re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out[:5]


def merge_candidate_frames(frames: list[pd.DataFrame], top_k: int) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        return merged
    merged = merged.drop_duplicates(subset=["foodName", "data_type"], keep="first")
    merged = merged.sort_values("_tfidf", ascending=False).head(top_k)
    return merged.reset_index(drop=True)


def headword_fallback_indices(candidates: pd.DataFrame, query: str, limit: int = 3) -> list[int]:
    """If the LLM returns [], keep strong headword matches instead of leaving a gap."""
    if candidates.empty:
        return []
    variants = _query_variants(query)
    scored: list[tuple[float, int]] = []
    for i, name in enumerate(candidates["foodName"].astype(str).tolist()):
        boost = _headword_boost(name, variants)
        if boost >= 0.40:
            scored.append((boost, i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:limit]]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_val(row: pd.Series, col: str) -> str:
    v = row.get(col)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.1f}"


def format_candidates(df: pd.DataFrame, *, compact: bool = False) -> str:
    lines = []
    for i, (_, r) in enumerate(df.iterrows()):
        dt = str(r.get("data_type", "?"))
        name = str(r.get("foodName", ""))[:70]
        cov = int(r.get("number_of_nutrients") or 0)
        if compact:
            energy = _fmt_val(r, "Energy")
            lines.append(f"{i}. [{dt}] E={energy} {name}")
        else:
            cat = str(r.get("food_category", "") or "")[:35]
            nutrients = "  ".join(
                f"{c.split()[0]}={_fmt_val(r, c)}" for c in _KEY_NUTRIENTS
            )
            lines.append(
                f"{i:2d}. [{dt}] cov={cov:2d} {name:<70} | {nutrients} | {cat}"
            )
    return "\n".join(lines)


def _parse_index_list(raw: str, n_candidates: int) -> list[int]:
    m = re.search(r"\[[\d\s,]*\]", raw)
    if not m:
        return []
    try:
        indices = json.loads(m.group())
        return [
            int(i)
            for i in indices
            if isinstance(i, (int, float)) and 0 <= int(i) < n_candidates
        ]
    except (json.JSONDecodeError, ValueError):
        return []


def _parse_batch_object(raw: str, expected: dict[str, int]) -> dict[str, list[int]]:
    """Parse `{id: [indices…]}` from model output; clamp indices per ingredient."""
    out: dict[str, list[int]] = {iid: [] for iid in expected}
    if not raw:
        return out
    # Prefer fenced / whole-object JSON; fall back to first {...} span.
    blob = raw
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        blob = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            blob = brace.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        # Last resort: per-id arrays if the model ignored the schema.
        for iid, n in expected.items():
            m = re.search(rf'"{re.escape(iid)}"\s*:\s*(\[[\d\s,]*\])', raw)
            if m:
                out[iid] = _parse_index_list(m.group(1), n)
        return out
    if not isinstance(data, dict):
        return out
    for iid, n in expected.items():
        val = data.get(iid, data.get(str(iid)))
        if not isinstance(val, list):
            continue
        out[iid] = [
            int(i)
            for i in val
            if isinstance(i, (int, float)) and 0 <= int(i) < n
        ]
    return out


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
    content = response.choices[0].message.content if response.choices else None
    return _parse_index_list((content or "").strip(), len(candidates))


def ask_llm_batch(
    client: "OpenAI",
    model: str,
    batch: list[tuple[dict, pd.DataFrame]],
) -> dict[str, list[int]]:
    """Select indices for many ingredients in one LLM call."""
    blocks: list[str] = []
    expected: dict[str, int] = {}
    for item, candidates in batch:
        iid = item["id"]
        # Cap candidates shown to keep prompts short/fast.
        shown = candidates.head(15)
        expected[iid] = len(shown)
        blocks.append(
            f"### {item['name']}  (id: {iid})\n"
            f"Candidates:\n{format_candidates(shown, compact=True)}"
        )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_BATCH},
                {"role": "user", "content": _USER_BATCH.format(blocks="\n\n".join(blocks))},
            ],
            temperature=0,
            max_tokens=min(2048, 40 * len(batch) + 128),
        )
    except Exception as exc:  # noqa: BLE001 — timeout / rate-limit / network
        print(f"  BATCH ERROR {exc!r}", flush=True)
        return {iid: [] for iid in expected}
    content = response.choices[0].message.content if response.choices else None
    return _parse_batch_object((content or "").strip(), expected)


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


async def ask_llm_batch_async(
    client,
    model: str,
    batch: list[tuple[dict, pd.DataFrame]],
) -> dict[str, list[int]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, ask_llm_batch, client, model, batch)


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

def _apply_selection(
    meta: list[dict],
    item: dict,
    candidates: pd.DataFrame,
    selected_idx: list[int],
    nutrient_cols: list[str],
    max_sources: int,
) -> tuple[str, list[str]]:
    """Write one ingredient into meta. Returns (status, food names)."""
    idx = selected_idx[:max_sources]
    used_fallback = False
    if not idx:
        idx = headword_fallback_indices(candidates, item["name"], max_sources)
        used_fallback = bool(idx)
    if idx:
        selected = candidates.iloc[idx].reset_index(drop=True)
        names = [str(r["foodName"]) for _, r in selected.iterrows()]
        status = "fallback" if used_fallback else "ok"
    else:
        selected = candidates.iloc[0:0].copy()
        names = []
        status = "empty"
    update_meta_entry(meta, item["id"], item["name"], selected, nutrient_cols)
    return status, names


def curate(
    ids: list[str] | None,
    *,
    model: str,
    base_url: str | None,
    api_key: str | None,
    top_k: int,
    max_sources: int,
    dry_run: bool,
    workers: int = 15,
    batch_size: int = 20,
    expand: bool = False,
) -> int:
    if not dry_run and OpenAI is None:
        print("openai package not installed.  Run: pip install openai", file=sys.stderr)
        return 1

    print("Loading USDA data…", flush=True)
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

    print("Building TF-IDF index over authoritative USDA rows…", flush=True)
    vect, mat, auth = build_tfidf_index(usda)
    print(f"  {len(auth):,} sr_legacy / foundation / survey rows indexed", flush=True)

    client = None
    if not dry_run:
        client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=60.0,
            max_retries=2,
        )

    # Build candidate lists upfront (CPU-only).
    tasks: list[tuple[dict, pd.DataFrame]] = []
    weak_ids: set[str] = set()
    for item in vocab:
        candidates = get_candidates(auth, vect, mat, item["name"], top_k)
        candidates = filter_candidate_quality(candidates)
        if retrieval_is_weak(candidates, item["name"]):
            weak_ids.add(item["id"])
        tasks.append((item, candidates))

    if expand:
        print(f"  {len(weak_ids)}/{len(tasks)} marked weak (query expansion ON)", flush=True)
    else:
        print(f"  {len(weak_ids)}/{len(tasks)} weak headword (expansion OFF — use --expand)", flush=True)

    if dry_run:
        for item, candidates in tasks:
            print(f"\n{'─' * 70}")
            flag = "  [weak retrieval]" if item["id"] in weak_ids else ""
            print(f"  {item['name']}  ({item['id']}){flag}")
            if candidates.empty:
                print("  No candidates found.")
            else:
                print(format_candidates(candidates))
        return 0

    def _expand_candidates(item: dict, candidates: pd.DataFrame) -> pd.DataFrame:
        if not expand or item["id"] not in weak_ids:
            return candidates
        alts = expand_search_queries(client, model, item["name"])
        if not alts:
            return candidates
        frames = [candidates]
        for alt in alts:
            extra = filter_candidate_quality(get_candidates(auth, vect, mat, alt, top_k))
            frames.append(extra)
        merged = merge_candidate_frames(frames, top_k)
        print(f"  expand {item['name']}: {alts} -> {len(merged)} candidates", flush=True)
        return merged

    # Optional expansion pass (slow; off by default).
    if expand and weak_ids:
        expanded: list[tuple[dict, pd.DataFrame]] = []
        for item, candidates in tasks:
            expanded.append((item, _expand_candidates(item, candidates)))
        tasks = expanded

    # Triage BEFORE calling any LLM:
    #   auto   — clear primary-name hit (no API)
    #   llm    — middling headword signal (ambiguous; ask model)
    #   skip   — no credible USDA row (specialty names); leave unmatched, no API
    # This is what makes a ~1800-item rebuild take seconds, not hang on OpenRouter.
    results: dict[str, list[int]] = {}
    need_llm: list[tuple[dict, pd.DataFrame]] = []
    auto_n = 0
    skip_n = 0
    empty_cand_n = 0
    for item, candidates in tasks:
        if candidates.empty:
            results[item["id"]] = []
            empty_cand_n += 1
            continue
        variants = _query_variants(item["name"])
        best_boost = max(
            (_headword_boost(str(n), variants) for n in candidates["foodName"].astype(str)),
            default=0.0,
        )
        auto = headword_fallback_indices(candidates, item["name"], max_sources)
        # Exact primary-name hits only (avoid salt→salt pork, ice→ice cream).
        if auto and best_boost >= 0.55 and len(item["name"]) >= 5:
            results[item["id"]] = auto[:max_sources]
            auto_n += 1
            continue
        # No real headword signal → don't burn an LLM call; USDA simply lacks it.
        if best_boost < 0.25:
            results[item["id"]] = []
            skip_n += 1
            continue
        need_llm.append((item, candidates.head(15).reset_index(drop=True)))

    print(
        f"  auto={auto_n}  llm={len(need_llm)}  skip_no_usda={skip_n}  "
        f"no_candidates={empty_cand_n}  batch_size={batch_size} workers={workers}",
        flush=True,
    )

    # Persist auto/skip immediately so a killed LLM pass still keeps them.
    for item, candidates in tasks:
        iid = item["id"]
        if iid in results and iid not in {x["id"] for x, _ in need_llm}:
            status, names = _apply_selection(
                meta, item, candidates, results[iid], nutrient_cols, max_sources
            )
            tag = "auto" if names else "skip"
            print(f"  {item['name']}: {names}  [{tag}]", flush=True)
    if auto_n or skip_n or empty_cand_n:
        META_JSON.write_text(json.dumps(meta, separators=(",", ":")))
        print(
            f"  …wrote {auto_n + skip_n + empty_cand_n} auto/skip entries (no LLM)",
            flush=True,
        )

    # Batch the rest: ~12 ingredients per request, many requests in parallel.
    batches = [
        need_llm[i : i + batch_size]
        for i in range(0, len(need_llm), batch_size)
    ] if need_llm else []

    done_llm = 0
    lock = asyncio.Lock() if workers > 1 else None
    # Pretty-printing the full 10MB+ meta on every batch serializes the run.
    # Keep in-memory updates; compact-checkpoint rarely on a background path.
    _CHECKPOINT_EVERY = 120

    def _checkpoint_meta() -> None:
        META_JSON.write_text(json.dumps(meta, separators=(",", ":")))

    def _consume_batch_result(batch: list[tuple[dict, pd.DataFrame]], picked: dict[str, list[int]]) -> None:
        nonlocal done_llm
        for item, candidates in batch:
            iid = item["id"]
            results[iid] = picked.get(iid, [])
            names = [
                str(candidates.iloc[i]["foodName"])
                for i in results[iid][:max_sources]
                if i < len(candidates)
            ]
            print(f"  {item['name']}: {names}", flush=True)
            _apply_selection(
                meta, item, candidates, results[iid], nutrient_cols, max_sources
            )
            done_llm += 1
        if done_llm % _CHECKPOINT_EVERY == 0 or done_llm == len(need_llm):
            _checkpoint_meta()
            print(f"  …checkpoint llm {done_llm}/{len(need_llm)}", flush=True)

    if batches and workers <= 1:
        for batch in batches:
            try:
                picked = ask_llm_batch(client, model, batch)
            except Exception as exc:  # noqa: BLE001
                print(f"  BATCH ERROR {exc!r}", flush=True)
                picked = {item["id"]: [] for item, _ in batch}
            _consume_batch_result(batch, picked)
    elif batches:
        sem = asyncio.Semaphore(workers)

        async def _process_batch(batch: list[tuple[dict, pd.DataFrame]]) -> None:
            async with sem:
                try:
                    picked = await ask_llm_batch_async(client, model, batch)
                except Exception as exc:  # noqa: BLE001
                    print(f"  BATCH ERROR {exc!r}", flush=True)
                    picked = {item["id"]: [] for item, _ in batch}
            assert lock is not None
            async with lock:
                _consume_batch_result(batch, picked)

        async def _run_all() -> None:
            await asyncio.gather(*[_process_batch(b) for b in batches])

        asyncio.run(_run_all())

    # Final consistent write (includes auto-accepted + LLM + empties).
    llm_cands = {item["id"]: cands for item, cands in need_llm}
    changed = empty_ok = fallback_n = 0
    for item, candidates in tasks:
        iid = item["id"]
        if iid not in results:
            continue
        frame = llm_cands.get(iid, candidates)
        status, names = _apply_selection(
            meta, item, frame, results[iid], nutrient_cols, max_sources
        )
        changed += 1
        if status == "empty":
            empty_ok += 1
        elif status == "fallback":
            fallback_n += 1

    if changed > 0:
        META_JSON.write_text(json.dumps(meta, indent=2))
        print(f"\n✓ Updated {changed} ingredient(s) in {META_JSON.name}", flush=True)
        print(f"  auto-accepted={auto_n}  llm-batches={len(batches)}", flush=True)
        if fallback_n:
            print(f"  {fallback_n} filled via headword fallback after empty LLM pick", flush=True)
        if empty_ok:
            print(f"  {empty_ok} left unmatched (no credible USDA candidate)", flush=True)
        print("\nNext step:  python -m pyfooda.build.recompute", flush=True)
    else:
        print("\nNo changes written.", flush=True)

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
    group.add_argument(
        "--unmatched",
        action="store_true",
        help="Curate only ingredients that currently have zero USDA sources",
    )

    p.add_argument("--model", default="gpt-4o-mini", help="LLM model name (default: gpt-4o-mini)")
    p.add_argument("--base-url", default=None, help="Override OpenAI base URL")
    p.add_argument("--api-key", default=None, help="API key (falls back to OPENAI_API_KEY env var)")
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="USDA candidates per ingredient (default: 50 with --all/--unmatched, else 20)",
    )
    p.add_argument("--max-sources", type=int, default=5, help="Max sources to select per ingredient (default: 5)")
    p.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Concurrent LLM batch requests (default: 15)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Ingredients per LLM request (default: 20)",
    )
    p.add_argument(
        "--expand",
        action="store_true",
        help="Enable slow LLM query-expansion for weak headword retrieval",
    )
    p.add_argument("--dry-run", action="store_true", help="Print candidates without making LLM calls")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    ids = None
    if args.ids:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    elif args.unmatched:
        meta = json.loads(META_JSON.read_text())
        ids = [m["ingredient_id"] for m in meta if not m.get("sources")]
        if not ids:
            print("No unmatched ingredients found.")
            return 0
        print(f"Retrying {len(ids)} unmatched ingredients…")
    top_k = args.top_k if args.top_k is not None else (50 if (args.all or args.unmatched) else 20)
    return curate(
        ids=ids,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        top_k=top_k,
        max_sources=args.max_sources,
        dry_run=args.dry_run,
        workers=args.workers,
        batch_size=args.batch_size,
        expand=args.expand,
    )


if __name__ == "__main__":
    raise SystemExit(main())

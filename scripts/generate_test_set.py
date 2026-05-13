#!/usr/bin/env python3
"""
Generate a data-driven test_set.json from the actual fooddata.csv.

Strategy
--------
All food names used in the test set are exact strings from the CSV —
no guessing, no fabrication.

Case generation:
  MERGE  — pairs/groups of foods that share an edit-distance-close
            normalized name AND have similar energy (<20% relative diff).
            These are the "trivial descriptor variants" the aggregator
            should consolidate.

  SPLIT  — pairs of foods from the same broad category whose energy
            differs by >50% OR whose name contains a cooking-state or
            fat-level keyword that makes merging nutritionally wrong.

  NAME_QUALITY — one food per category from sr_legacy/survey data types,
            providing a diverse cross-section of name patterns.

Usage
-----
  python scripts/generate_test_set.py
  python scripts/generate_test_set.py --output tests/test_set.json --n-merge 50 --n-split 80
  python scripts/generate_test_set.py --seed 42 --n-name 30
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "pyfooda/data/fooddata.csv"
DEFAULT_OUTPUT = REPO_ROOT / "tests/test_set.json"

# Data types to use for case generation (the actual aggregation targets)
TARGET_DTYPES = {"sr_legacy_food", "survey_fndds_food", "foundation_food",
                 "sub_sample_food"}

# Cooking-state keywords: if two foods differ on these, they MUST split
COOKING_STATE_TOKENS = {
    "raw", "cooked", "dried", "smoked", "canned", "frozen", "pickled",
    "roasted", "baked", "fried", "boiled", "grilled", "salted", "dry",
    "fresh",
}

# Fat-level keywords: explicit fat-level difference → must split
FAT_LEVEL_TOKENS = {
    "nonfat", "fat free", "fat-free", "lowfat", "low fat", "low-fat",
    "reduced fat", "reduced-fat", "light", "whole", "2%", "1%", "skim",
    "part skim", "heavy", "half and half",
}

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, drop punctuation."""
    s = str(text).strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_set(text: str) -> frozenset:
    return frozenset(_norm(text).split())


def _edit_ratio(a: str, b: str) -> float:
    """Jaccard similarity on token sets (fast proxy for edit distance)."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# Descriptor suffixes that carry no food-identity signal.
# Two foods that differ ONLY in these tokens may still be completely different foods.
_DESCRIPTOR_TOKENS = frozenset({
    "from", "canned", "dried", "fresh", "frozen", "added", "no", "without",
    "with", "reduced", "sodium", "fat", "ns", "as", "to", "nfs", "and",
    "or", "in", "the", "a", "of", "type", "style", "made", "prepared",
    "restaurant", "fast", "food", "store",
})


def _core_tokens(name: str) -> frozenset:
    """Token set minus generic descriptor words — represents the food identity."""
    return _token_set(name) - _DESCRIPTOR_TOKENS


def _core_ratio(a: str, b: str) -> float:
    """Jaccard similarity on core (non-descriptor) tokens only."""
    ca, cb = _core_tokens(a), _core_tokens(b)
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    return len(ca & cb) / len(ca | cb)


def _cooking_tokens(name: str) -> frozenset:
    tokens = _token_set(name)
    return tokens & COOKING_STATE_TOKENS


def _fat_tokens(name: str) -> set:
    nl = _norm(name)
    return {t for t in FAT_LEVEL_TOKENS if t in nl}


def _energy_ratio(e1: float, e2: float) -> float:
    """Relative difference, using smaller as denominator."""
    if e1 <= 0 or e2 <= 0:
        return 0.0
    return abs(e1 - e2) / min(e1, e2)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["foodName"].notna() & (df["foodName"].astype(str) != "nan")]
    df = df[df["data_type"].isin(TARGET_DTYPES)].copy()
    df["foodName"] = df["foodName"].astype(str).str.strip()
    df = df[df["foodName"].str.len() > 3]
    df["_energy"] = pd.to_numeric(df.get("Energy", pd.Series(dtype=float)), errors="coerce")
    df["_fat"] = pd.to_numeric(df.get("Total fat", pd.Series(dtype=float)), errors="coerce")
    df["_protein"] = pd.to_numeric(df.get("Protein", pd.Series(dtype=float)), errors="coerce")
    df["_norm"] = df["foodName"].apply(_norm)
    df = df.drop_duplicates(subset=["foodName"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merge case generation
#
# Strategy: within each food_category, find pairs of foods whose
# normalised-token Jaccard similarity is high (>0.55) AND whose energy
# is within 20%.  These are trivially "same food, different descriptor".
# Extend pairs into groups of up to 5 where all members are close to each
# other.
# ---------------------------------------------------------------------------

def _generate_merge_cases(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> list[dict]:

    # Require high core-token similarity AND overall Jaccard — prevents
    # "Black beans, from canned" matching "Chickpeas, from canned" purely
    # on shared descriptor suffixes.
    CORE_THRESH = 0.65    # core (food-identity) token Jaccard
    JACCARD_THRESH = 0.50  # overall token Jaccard
    ENERGY_DIFF_THRESH = 0.20  # 20% relative difference

    cases: list[dict] = []
    seen_sets: set[frozenset] = set()

    categories = [c for c in df["food_category"].dropna().unique()
                  if df[df["food_category"] == c].shape[0] >= 4]
    rng.shuffle(categories)

    for cat in categories:
        if len(cases) >= n:
            break
        sub = df[df["food_category"] == cat].reset_index(drop=True)
        if len(sub) < 2:
            continue

        # Build adjacency: pairs that are similar AND energy-compatible
        adj: dict[int, list[int]] = {i: [] for i in range(len(sub))}
        for i, j in combinations(range(len(sub)), 2):
            na, nb = sub.at[i, "foodName"], sub.at[j, "foodName"]
            # Both overall AND core-token similarity must be high
            if _edit_ratio(na, nb) < JACCARD_THRESH:
                continue
            if _core_ratio(na, nb) < CORE_THRESH:
                continue
            ea, eb = sub.at[i, "_energy"], sub.at[j, "_energy"]
            if pd.notna(ea) and pd.notna(eb) and ea > 0 and eb > 0:
                if _energy_ratio(ea, eb) > ENERGY_DIFF_THRESH:
                    continue
            # Check they don't have conflicting cooking-state tokens
            ct_a = _cooking_tokens(na)
            ct_b = _cooking_tokens(nb)
            if ct_a and ct_b and ct_a != ct_b:
                continue
            # Same fat-level tokens required
            fl_a = _fat_tokens(na)
            fl_b = _fat_tokens(nb)
            if fl_a != fl_b:
                continue
            adj[i].append(j)
            adj[j].append(i)

        # Collect connected components (cliques up to 5)
        visited = set()
        for seed in rng.permutation(len(sub)):
            if seed in visited:
                continue
            cluster = [seed]
            visited.add(seed)
            for nb in adj[seed]:
                if nb not in visited and len(cluster) < 5:
                    cluster.append(nb)
                    visited.add(nb)
            if len(cluster) < 2:
                continue
            foods = [sub.at[i, "foodName"] for i in cluster]
            key = frozenset(foods)
            if key in seen_sets:
                continue
            seen_sets.add(key)

            # Assign severity: foundation foods = important, others = normal
            dtypes = {sub.at[i, "data_type"] for i in cluster}
            sev = "important" if "foundation_food" in dtypes else "normal"

            cases.append({
                "id": f"m{len(cases)+1:03d}",
                "type": "merge",
                "severity": sev,
                "foods": sorted(foods),
                "note": (
                    f"category={cat!r}  "
                    f"core_jaccard≈{_core_ratio(foods[0], foods[1]):.2f}  "
                    f"auto-generated from CSV"
                ),
            })
            if len(cases) >= n:
                break

    return cases


# ---------------------------------------------------------------------------
# Split case generation
#
# Three sub-strategies:
#   1. Cooking-state splits: same base name, one has "raw", other has "cooked"
#   2. Fat-level splits: same category, same base product, different fat level
#   3. Energy-cliff splits: same category, energy differs >50%, similar tokens
# ---------------------------------------------------------------------------

def _generate_split_cases(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> list[dict]:

    cases: list[dict] = []
    seen_pairs: set[frozenset] = set()

    # --- Strategy 1: cooking-state pairs ---
    cooking_pairs: list[tuple[str, str, str]] = []
    for cat, sub in df.groupby("food_category"):
        sub = sub.reset_index(drop=True)
        for i, j in combinations(range(len(sub)), 2):
            na, nb = sub.at[i, "foodName"], sub.at[j, "foodName"]
            ct_a = _cooking_tokens(na)
            ct_b = _cooking_tokens(nb)
            if not ct_a or not ct_b or ct_a == ct_b:
                continue
            # Core food identity must overlap (same base food, different preparation)
            if _core_ratio(na, nb) < 0.35:
                continue
            cooking_pairs.append((na, nb, str(cat)))

    rng.shuffle(cooking_pairs)
    for na, nb, cat in cooking_pairs:
        if len(cases) >= n // 2:
            break
        key = frozenset([na, nb])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        ct_a = _cooking_tokens(na)
        ct_b = _cooking_tokens(nb)
        cases.append({
            "id": f"s{len(cases)+1:03d}",
            "type": "split",
            "severity": "important",
            "foods": sorted([na, nb]),
            "note": (
                f"cooking-state conflict: {sorted(ct_a)} vs {sorted(ct_b)}  "
                f"category={cat!r}  auto-generated"
            ),
        })

    # --- Strategy 2: fat-level pairs ---
    fat_pairs: list[tuple[str, str, str]] = []
    for cat, sub in df.groupby("food_category"):
        sub = sub.reset_index(drop=True)
        for i, j in combinations(range(len(sub)), 2):
            na, nb = sub.at[i, "foodName"], sub.at[j, "foodName"]
            fl_a = _fat_tokens(na)
            fl_b = _fat_tokens(nb)
            if not fl_a or not fl_b or fl_a == fl_b:
                continue
            # Core food identity must overlap (same product, different fat level)
            if _core_ratio(na, nb) < 0.35:
                continue
            fat_pairs.append((na, nb, str(cat)))

    rng.shuffle(fat_pairs)
    for na, nb, cat in fat_pairs:
        if len(cases) >= int(n * 0.75):
            break
        key = frozenset([na, nb])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        fl_a = _fat_tokens(na)
        fl_b = _fat_tokens(nb)
        cases.append({
            "id": f"s{len(cases)+1:03d}",
            "type": "split",
            "severity": "important",
            "foods": sorted([na, nb]),
            "note": (
                f"fat-level conflict: {sorted(fl_a)} vs {sorted(fl_b)}  "
                f"category={cat!r}  auto-generated"
            ),
        })

    # --- Strategy 3: energy-cliff pairs (same category, very different energy) ---
    # Exclude junk/miscellaneous categories that produce meaningless pairings
    _SKIP_CATS = {
        "not included in a food category", "not included", "american indian/alaska native foods",
        "baby foods", "infant formula",
    }
    energy_pairs: list[tuple[str, str, str, float]] = []
    for cat, sub in df.groupby("food_category"):
        if str(cat).lower().strip() in _SKIP_CATS:
            continue
        sub = sub[sub["_energy"].notna() & (sub["_energy"] > 10)].reset_index(drop=True)
        if len(sub) < 2:
            continue
        for i, j in combinations(range(len(sub)), 2):
            ea, eb = sub.at[i, "_energy"], sub.at[j, "_energy"]
            ratio = _energy_ratio(ea, eb)
            if ratio < 0.50:
                continue
            na, nb = sub.at[i, "foodName"], sub.at[j, "foodName"]
            # Core food identity must overlap (same broad food family)
            # but not too similar (that would be a merge candidate, not a split)
            core_sim = _core_ratio(na, nb)
            if core_sim < 0.20 or core_sim > 0.75:
                continue
            energy_pairs.append((na, nb, str(cat), ratio))

    # Sort by highest energy ratio first (most extreme = most important),
    # then deduplicate anchors so no single food appears > 3 times
    energy_pairs.sort(key=lambda x: x[3], reverse=True)
    anchor_counts: dict[str, int] = {}
    deduped: list[tuple[str, str, str, float]] = []
    for entry in energy_pairs:
        na, nb = entry[0], entry[1]
        if anchor_counts.get(na, 0) >= 3 or anchor_counts.get(nb, 0) >= 3:
            continue
        anchor_counts[na] = anchor_counts.get(na, 0) + 1
        anchor_counts[nb] = anchor_counts.get(nb, 0) + 1
        deduped.append(entry)
    rng.shuffle(deduped[:500])
    for na, nb, cat, ratio in deduped[:500]:
        if len(cases) >= n:
            break
        key = frozenset([na, nb])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        ea = df.loc[df["foodName"] == na, "_energy"].iloc[0]
        eb = df.loc[df["foodName"] == nb, "_energy"].iloc[0]
        cases.append({
            "id": f"s{len(cases)+1:03d}",
            "type": "split",
            "severity": "normal",
            "foods": sorted([na, nb]),
            "note": (
                f"energy cliff: {ea:.0f} vs {eb:.0f} kcal "
                f"(ratio={ratio:.1f}x)  category={cat!r}  auto-generated"
            ),
        })

    # Re-number ids
    for i, c in enumerate(cases):
        c["id"] = f"s{i+1:03d}"

    return cases


# ---------------------------------------------------------------------------
# Name-quality case generation
#
# Pick one food per major category from sr_legacy/survey whose name
# contains a pattern the aggregator should clean: USDA comma format,
# brand tokens, or lab/spec tokens.
# ---------------------------------------------------------------------------

_USDA_COMMA_RE = re.compile(r"^[A-Za-z][a-z]+,\s")
_BRAND_RE = re.compile(
    r"\b(BURGER KING|McDONALD|CHOBANI|DANNON|KRAFT|PIZZA HUT|TACO BELL"
    r"|WENDY|ARBY|KFC|POPEYES|SUBWAY|DOMINO|LITTLE CAESAR|PAPA JOHN"
    r"|CHICK.FIL.A|STARBUCKS)\b",
    re.IGNORECASE,
)
_LAB_RE = re.compile(r"\b(NFS|NS as to|0% moisture|\d+% moisture)\b", re.IGNORECASE)


def _generate_name_cases(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> list[dict]:
    cases: list[dict] = []
    seen_cats: set[str] = set()

    # Priority: USDA comma format (very common), then brand, then lab tokens
    def _pattern_type(name: str) -> tuple[str, list[str], list[str]]:
        if _USDA_COMMA_RE.search(name):
            # Extract the main food noun (before first comma)
            main = name.split(",")[0].strip()
            return (
                "usda_format",
                [","],          # forbidden_patterns
                [main],         # required_patterns
            )
        if _BRAND_RE.search(name):
            brand = _BRAND_RE.search(name).group()
            return (
                "brand",
                [brand],
                [],
            )
        if _LAB_RE.search(name):
            token = _LAB_RE.search(name).group()
            return (
                "lab_token",
                [token],
                [],
            )
        return ("", [], [])

    rows = df[df["data_type"].isin(["sr_legacy_food", "survey_fndds_food"])].copy()
    rows = rows[rows["foodName"].notna()].reset_index(drop=True)
    # Shuffle so we get variety
    idx = rng.permutation(len(rows))

    for i in idx:
        if len(cases) >= n:
            break
        row = rows.iloc[i]
        name = str(row["foodName"])
        cat = str(row.get("food_category", ""))
        if cat in seen_cats:
            continue
        ptype, forbidden, required = _pattern_type(name)
        if not ptype:
            continue
        seen_cats.add(cat)
        cases.append({
            "id": f"n{len(cases)+1:03d}",
            "type": "name_quality",
            "severity": "important" if ptype == "brand" else "normal",
            "food": name,
            "forbidden_patterns": forbidden,
            "required_patterns": required,
            "note": f"pattern={ptype}  category={cat!r}  auto-generated",
        })

    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(
    csv_path: Path,
    output_path: Path,
    n_merge: int,
    n_split: int,
    n_name: int,
    seed: int,
) -> None:
    print(f"Loading {csv_path} ...")
    df = load_csv(csv_path)
    print(f"  {len(df)} foods loaded (target dtypes only)")

    rng = np.random.default_rng(seed)

    print("Generating merge cases ...")
    merge_cases = _generate_merge_cases(df, n_merge, rng)
    print(f"  {len(merge_cases)} merge cases")

    print("Generating split cases ...")
    split_cases = _generate_split_cases(df, n_split, rng)
    print(f"  {len(split_cases)} split cases")

    print("Generating name-quality cases ...")
    name_cases = _generate_name_cases(df, n_name, rng)
    print(f"  {len(name_cases)} name-quality cases")

    all_cases = merge_cases + split_cases + name_cases

    # Severity summary
    sev_counts: dict[str, int] = {}
    for c in all_cases:
        sev_counts[c.get("severity", "normal")] = sev_counts.get(c.get("severity", "normal"), 0) + 1

    # Category coverage
    cats_covered: set[str] = set()
    for c in all_cases:
        foods = c.get("foods", []) + ([c["food"]] if "food" in c else [])
        for f in foods:
            rows = df[df["foodName"] == f]
            if not rows.empty:
                cats_covered.add(str(rows.iloc[0]["food_category"]))

    output = {
        "version": "2.0",
        "generated_by": "generate_test_set.py",
        "seed": seed,
        "csv_source": str(csv_path),
        "description": (
            "Data-driven test set: all food names are exact strings from "
            "fooddata.csv.  Merge cases = similar-name/similar-energy pairs. "
            "Split cases = cooking-state, fat-level, or energy-cliff conflicts. "
            "Name-quality cases = USDA/brand/lab patterns that must be cleaned."
        ),
        "scoring": {
            "merge_weight": 0.35,
            "split_weight": 0.45,
            "name_quality_weight": 0.20,
        },
        "stats": {
            "total_cases": len(all_cases),
            "merge": len(merge_cases),
            "split": len(split_cases),
            "name_quality": len(name_cases),
            "severity": sev_counts,
            "categories_covered": len(cats_covered),
        },
        "cases": all_cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(all_cases)} cases -> {output_path}")
    print(f"  merge={len(merge_cases)}  split={len(split_cases)}  name={len(name_cases)}")
    print(f"  severity: {sev_counts}")
    print(f"  categories covered: {len(cats_covered)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a data-driven test_set.json from fooddata.csv"
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to fooddata.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output path")
    parser.add_argument("--n-merge", type=int, default=60, help="Target merge cases")
    parser.add_argument("--n-split", type=int, default=100, help="Target split cases")
    parser.add_argument("--n-name", type=int, default=40, help="Target name-quality cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate(
        csv_path=Path(args.csv),
        output_path=Path(args.output),
        n_merge=args.n_merge,
        n_split=args.n_split,
        n_name=args.n_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

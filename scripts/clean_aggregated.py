"""
Post-hoc cleaning of foods_aggregated.json
==========================================
1. Re-map source_names → fooddata.csv rows (source_ids are stale after rebuild)
2. Apply smart outlier filtering on Energy per aggregated food:
   a) If Foundation/SR anchor exists ("raw", ", " in name → USDA reference food),
      keep sources whose energy is within a band of the anchor.
   b) Otherwise, apply MAD-based symmetric outlier removal.
3. Recompute nutrient averages from clean (non-outlier) sources
4. Update source_ids, source_names, count, portion data

Foundation/SR anchor rationale: USDA Foundation and SR Legacy foods have names
like "Banana, raw" with scientifically measured nutrient values. Branded foods
("CRUNCHY BANANA", 350 kcal) are often processed variants that would contaminate
the average of the canonical form.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAD_THRESHOLD = 3.5     # multiples of MAD for symmetric filtering
FIXED_BAND = 150        # kcal fallback when MAD < 10

# Anchor-based filtering: keep sources within this factor of the anchor energy
# e.g., anchor=89 kcal → keep 89 * 0.5 .. 89 * 2.0 → 44.5 .. 178 kcal
ANCHOR_LOW_FACTOR = 0.3
ANCHOR_HIGH_FACTOR = 2.5
ANCHOR_MIN_BAND = 80    # minimum band width (kcal) around anchor

MIN_SOURCES_AFTER = 1

NUTRIENT_COLS = [
    'Energy', 'Carbohydrate', 'Protein', 'Total fat',
    'Fiber', 'Sugars, Total', 'Sodium', 'Calcium', 'Iron',
    'Cholesterol', 'Fatty acids, total saturated',
    'Vitamin A, RAE', 'Vitamin C', 'Vitamin D (D2 + D3)',
    'Vitamin E', 'Vitamin B-6', 'Vitamin B-12',
    'Folate, total', 'Zinc', 'Magnesium', 'Potassium', 'Phosphorus',
]


def _is_reference_food(name: str) -> bool:
    """Detect if a source food name is a Foundation/SR Legacy reference food.

    USDA reference foods follow patterns like:
      "Banana, raw"
      "Apples, raw, without skin"
      "Rice, white, long-grain, regular, cooked"
    They tend to have: lowercase, comma-separated qualifiers, "raw"/"cooked".
    Branded foods are typically ALL CAPS with marketing language.
    """
    if re.search(r'\braw\b', name, re.IGNORECASE):
        return True
    if re.search(r'\bcooked\b', name, re.IGNORECASE):
        return True
    if ', ' in name and name[0].islower():
        return True
    return False


def _name_matches_generic(source_name: str, generic_name: str) -> bool:
    """Check if a reference source name actually relates to the generic food.

    Prevents false anchoring, e.g. "Tomatoes, orange, raw" should NOT anchor
    the "Orange" food group — its primary food is "Tomatoes", not "Orange".
    """
    # Get the first segment of the source name (before first comma)
    first_seg = source_name.split(',')[0].strip().lower()
    # Check if any word from the generic name appears in the first segment
    generic_words = generic_name.lower().split()
    for word in generic_words:
        if len(word) >= 3 and word in first_seg:
            return True
    return False


def _find_anchor_energy(source_names: list[str], generic_name: str,
                        name_to_rows: dict,
                        fdf: pd.DataFrame) -> float | None:
    """Find the energy anchor from reference (Foundation/SR) sources.

    Only considers reference sources whose name matches the generic food name,
    preventing false anchoring from unrelated reference foods.
    Prefers "cooked" sources over "raw" when both exist (cooked form is
    more representative for foods consumed cooked, e.g. rice, chicken).
    """
    cooked_energies = []
    raw_energies = []
    other_ref_energies = []

    for sname in source_names:
        if not _is_reference_food(sname):
            continue
        if not _name_matches_generic(sname, generic_name):
            continue
        key = sname.strip().upper()
        rows = name_to_rows.get(key, [])
        if rows:
            e = fdf.iloc[rows[0]]['Energy']
            if pd.notna(e) and float(e) > 0:
                if re.search(r'\bcooked\b', sname, re.IGNORECASE):
                    cooked_energies.append(float(e))
                elif re.search(r'\braw\b', sname, re.IGNORECASE):
                    raw_energies.append(float(e))
                else:
                    other_ref_energies.append(float(e))

    # Prefer cooked > raw > other reference energies
    for pool in [cooked_energies, raw_energies, other_ref_energies]:
        if pool:
            return float(np.median(pool))
    return None


def _detect_bimodal_lower_cluster(energies_arr: np.ndarray,
                                  gap_threshold: float = 0.25
                                  ) -> Optional[tuple]:
    """Detect bimodal distribution and return the lower cluster bounds.

    Looks for the largest gap between consecutive sorted values.
    If that gap > gap_threshold * full_range, the distribution is bimodal.
    Returns (lo, hi) bounds of the lower cluster, or None.

    NOTE: Only returns a cluster if the lower cluster is the MAJORITY,
    preventing false positives for high-energy foods (butter, oils) where
    a few low-calorie spray/extract entries create a small lower cluster.
    """
    if len(energies_arr) < 6:
        return None
    sorted_e = np.sort(energies_arr)
    full_range = sorted_e[-1] - sorted_e[0]
    if full_range < 80:
        return None  # tight distribution

    diffs = np.diff(sorted_e)
    max_gap_idx = int(np.argmax(diffs))
    max_gap = diffs[max_gap_idx]

    if max_gap < max(100, gap_threshold * full_range):
        return None

    lower = sorted_e[:max_gap_idx + 1]
    upper = sorted_e[max_gap_idx + 1:]

    # Only use lower cluster if it's the MAJORITY (>50% of sources)
    if len(lower) < len(upper):
        return None
    if len(lower) < 3:
        return None
    return (float(lower[0]), float(lower[-1]))


def _filter_sources(src_df: pd.DataFrame, source_names: list[str],
                    generic_name: str,
                    name_to_rows: dict, fdf: pd.DataFrame) -> tuple:
    """Return a boolean mask of sources to keep and the method used.

    Strategy:
    1. If validated reference foods exist → anchor-based band filtering
    2. Else if bimodal → keep lower cluster (+ margin)
    3. Else → MAD-based symmetric filtering

    Returns: (keep_mask: pd.Series, method: str)
    """
    energies = src_df['Energy'].dropna()
    keep = pd.Series(True, index=src_df.index)

    if len(energies) < 3:
        return keep, 'skip'

    energies_arr = energies.values.astype(float)

    # --- Tier 1: Anchor-based filtering ---
    anchor = _find_anchor_energy(source_names, generic_name, name_to_rows, fdf)
    if anchor is not None and anchor > 0:
        lo = anchor * ANCHOR_LOW_FACTOR
        hi = anchor * ANCHOR_HIGH_FACTOR
        if hi - lo < ANCHOR_MIN_BAND:
            lo = anchor - ANCHOR_MIN_BAND / 2
            hi = anchor + ANCHOR_MIN_BAND / 2

        for i in src_df.index:
            e = src_df.loc[i, 'Energy']
            if pd.notna(e) and (e < lo or e > hi):
                keep[i] = False
        return keep, 'anchor'

    # --- Tier 2: Bimodal detection → lower cluster ---
    cluster = _detect_bimodal_lower_cluster(energies_arr)
    if cluster is not None:
        lo, hi = cluster
        margin = max(50, (hi - lo) * 0.3)
        for i in src_df.index:
            e = src_df.loc[i, 'Energy']
            if pd.notna(e) and (e < lo - margin or e > hi + margin):
                keep[i] = False
        return keep, 'bimodal'

    # --- Tier 3: MAD-based symmetric filtering ---
    median_e = float(np.median(energies_arr))
    deviations = np.abs(energies_arr - median_e)
    mad = float(np.median(deviations))

    if mad < 10:
        threshold = FIXED_BAND
    else:
        threshold = MAD_THRESHOLD * mad

    for i in src_df.index:
        e = src_df.loc[i, 'Energy']
        if pd.notna(e) and abs(e - median_e) > threshold:
            keep[i] = False

    return keep, 'mad'


def main():
    base = Path(__file__).resolve().parent.parent

    # Load data
    print("Loading fooddata.csv ...")
    fdf = pd.read_csv(base / "pyfooda/data/fooddata.csv")
    print(f"  {len(fdf)} rows")

    print("Loading foods_aggregated.json ...")
    with open(base / "pyfooda/data/foods_aggregated.json") as f:
        foods = json.load(f)
    print(f"  {len(foods)} aggregated foods")

    # Build name → row index lookup
    print("Building name index ...")
    name_to_rows = {}
    for idx, row in fdf.iterrows():
        key = str(row['foodName']).strip().upper()
        if key not in name_to_rows:
            name_to_rows[key] = []
        name_to_rows[key].append(idx)

    # Process
    total_before = sum(f['count'] for f in foods)
    total_after = 0
    total_outliers_removed = 0
    total_unmatched = 0
    total_anchor_filtered = 0
    total_bimodal_filtered = 0
    total_mad_filtered = 0
    foods_removed = 0

    cleaned = []
    for food in foods:
        # Re-map source_names to current fooddata.csv
        matched_indices = []
        matched_names = []
        for sname in food['source_names']:
            key = sname.strip().upper()
            candidates = name_to_rows.get(key, [])
            if candidates:
                matched_indices.append(candidates[0])
                matched_names.append(sname)
            else:
                total_unmatched += 1

        if not matched_indices:
            foods_removed += 1
            continue

        src_df = fdf.iloc[matched_indices].copy().reset_index(drop=True)
        local_to_fdf = {i: matched_indices[i] for i in range(len(matched_indices))}
        local_names = matched_names

        # Pre-filter: remove sources with 0 energy (extracts, data errors)
        for i in src_df.index:
            e = src_df.loc[i, 'Energy']
            if pd.notna(e) and float(e) <= 0:
                src_df.drop(i, inplace=True)
                del local_to_fdf[i]
        # Re-index after pre-filter
        old_to_new = {}
        new_indices = []
        new_fdf_map = {}
        new_names = []
        for j, old_i in enumerate(src_df.index):
            old_to_new[old_i] = j
            new_fdf_map[j] = local_to_fdf[old_i]
            new_names.append(local_names[old_i])
        src_df = src_df.reset_index(drop=True)
        local_to_fdf = new_fdf_map
        local_names = new_names

        if len(src_df) == 0:
            foods_removed += 1
            continue

        # Filter
        keep, method = _filter_sources(src_df, local_names, food['generic_name'],
                                       name_to_rows, fdf)

        n_removed = (~keep).sum()
        total_outliers_removed += n_removed
        if method == 'anchor':
            total_anchor_filtered += n_removed
        elif method == 'bimodal':
            total_bimodal_filtered += n_removed
        else:
            total_mad_filtered += n_removed

        clean_df = src_df[keep]
        if len(clean_df) < MIN_SOURCES_AFTER:
            foods_removed += 1
            continue

        # Recompute nutrients
        nutrients = {}
        for col in NUTRIENT_COLS:
            if col in clean_df.columns:
                vals = clean_df[col].dropna()
                nutrients[col] = round(float(vals.mean()), 4) if len(vals) > 0 else None
            else:
                nutrients[col] = None

        clean_local = clean_df.index.tolist()
        clean_fdf_ids = [local_to_fdf[i] for i in clean_local]
        clean_names = [local_names[i] for i in clean_local]

        # Portion data
        pw = clean_df['portion_gram_weight'].dropna().tolist() if 'portion_gram_weight' in clean_df.columns else []
        pu = clean_df['portion_unit_name'].dropna().tolist() if 'portion_unit_name' in clean_df.columns else []

        cleaned.append({
            'id': food['id'],
            'generic_name': food['generic_name'],
            'food_category': food['food_category'],
            'nutrients': nutrients,
            'source_ids': [int(i) for i in clean_fdf_ids],
            'source_names': clean_names,
            'count': len(clean_df),
            'portion_gram_weight': round(float(np.median(pw)), 1) if pw else None,
            'portion_unit_name': Counter(pu).most_common(1)[0][0] if pu else None,
        })
        total_after += len(clean_df)

    # Summary
    print(f"\n{'='*60}")
    print(f"  CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"  Foods before:          {len(foods)}")
    print(f"  Foods after:           {len(cleaned)}  (removed {foods_removed})")
    print(f"  Sources before:        {total_before}")
    print(f"  Sources after:         {total_after}")
    print(f"  Outliers removed:      {total_outliers_removed}")
    print(f"    via anchor (ref):    {total_anchor_filtered}")
    print(f"    via bimodal (lower): {total_bimodal_filtered}")
    print(f"    via MAD (fallback):  {total_mad_filtered}")
    print(f"  Unmatched names:       {total_unmatched}")
    print(f"{'='*60}")

    # Spot-check
    expected = {
        "Banana": 89, "Apple": 52, "Orange": 47, "White Rice": 350,
        "Bread": 265, "Butter": 717, "Yogurt": 80, "Whole Milk": 61,
        "Honey": 304, "Olive Oil": 884, "Chicken Breast": 120,
        "Salmon": 142, "Tomato": 18, "Potato": 77, "Egg": 155,
        "Carrot": 41,
    }
    print("\n  Spot check (Energy kcal/100g):")
    for item in cleaned:
        if item['generic_name'] in expected:
            actual = item['nutrients'].get('Energy', 0) or 0
            exp = expected[item['generic_name']]
            diff = (actual - exp) / exp * 100
            flag = " ***" if abs(diff) > 30 else ""
            print(f"    {item['generic_name']:20s}: {actual:6.1f} (expected ~{exp}, {diff:+.0f}%){flag}")

    # Save
    out_json = base / "pyfooda/data/foods_aggregated.json"
    with open(out_json, 'w') as f:
        json.dump(cleaned, f, indent=2)
    print(f"\n  Saved {len(cleaned)} foods -> {out_json}")

    rows = []
    for food in cleaned:
        row = {
            'foodName': food['generic_name'],
            'category': food['food_category'],
            'portion_gram_weight': food.get('portion_gram_weight'),
            'portion_unit_name': food.get('portion_unit_name'),
        }
        row.update(food['nutrients'])
        rows.append(row)
    csv_path = base / "pyfooda/data/foods_aggregated.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved CSV -> {csv_path}")


if __name__ == '__main__':
    main()

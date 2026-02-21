#!/usr/bin/env python3
"""
Build a small test dataset (~2000 foods) from fooddata.csv for fast
aggregation testing. Focuses on known problem cases (bimodal energy
distributions, naming collisions) plus a random sample for diversity.

Output: tests/test_fooddata.csv
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "tests"
OUTPUT_FILE = OUTPUT_DIR / "test_fooddata.csv"

# Problem-case search terms: foods where fresh/dried/cooked/raw
# distinctions or brand vs generic naming caused issues
PROBLEM_QUERIES = {
    # Bimodal energy: fresh vs dried
    "banana": 30,
    "apple": 30,
    "mango": 15,
    "cranberr": 15,
    "coconut": 15,
    "fig": 10,
    "date": 10,
    # Bimodal energy: cooked vs dry
    "rice": 30,
    "pasta": 20,
    "lentil": 10,
    "bean": 20,
    "oat": 15,
    # High vs low calorie variants
    "butter": 25,
    "olive oil": 15,
    "cream": 20,
    "milk": 25,
    "yogurt": 25,
    # Protein foods (raw vs cooked)
    "chicken breast": 25,
    "salmon": 25,
    "egg": 20,
    "ground beef": 15,
    "tuna": 15,
    # Produce (easy cases)
    "tomato": 20,
    "potato": 20,
    "orange": 20,
    "carrot": 10,
    "broccoli": 10,
    "spinach": 10,
    # Pantry staples
    "bread": 25,
    "honey": 15,
    "sugar": 15,
    "flour": 10,
    # Brand-name → generic mapping test cases
    "nutella": 10,
    "hazelnut": 15,
    "oreo": 10,
    "cheerios": 10,
    "gatorade": 10,
}

# Total targeted ≈ 700; we'll add random to reach ~2000
RANDOM_SAMPLE = 1300
SEED = 42


def main():
    print("Loading fooddata.csv ...")
    df = pd.read_csv(REPO_ROOT / "pyfooda/data/fooddata.csv")
    print(f"  {len(df)} rows")

    selected_indices = set()
    rng = np.random.RandomState(SEED)

    # Targeted sampling: for each query, sample N matching rows
    print("\nTargeted sampling:")
    for query, n in PROBLEM_QUERIES.items():
        mask = df["foodName"].str.contains(query, case=False, na=False)
        candidates = df[mask].index.tolist()
        # Remove already-selected
        candidates = [i for i in candidates if i not in selected_indices]
        take = min(n, len(candidates))
        if candidates:
            chosen = rng.choice(candidates, size=take, replace=False)
            selected_indices.update(chosen)
            print(f"  {query:20s}: {take:3d} / {len(df[mask])} available")
        else:
            print(f"  {query:20s}:   0 (no matches)")

    # Ensure we include Foundation/SR reference foods (the anchors)
    print("\nAdding Foundation/SR reference foods ...")
    ref_mask = (
        df["foodName"].str.contains(", raw", case=False, na=False)
        | df["foodName"].str.contains(", cooked", case=False, na=False)
    ) & df["foodName"].str[0].str.islower()
    ref_indices = [i for i in df[ref_mask].index if i not in selected_indices]
    ref_take = min(200, len(ref_indices))
    ref_chosen = rng.choice(ref_indices, size=ref_take, replace=False)
    selected_indices.update(ref_chosen)
    print(f"  {ref_take} reference foods added (from {len(df[ref_mask])} available)")

    # Random sample for diversity
    remaining = [i for i in df.index if i not in selected_indices]
    random_take = min(RANDOM_SAMPLE, len(remaining))
    random_chosen = rng.choice(remaining, size=random_take, replace=False)
    selected_indices.update(random_chosen)
    print(f"\nRandom sample: {random_take}")

    # Build output
    test_df = df.loc[sorted(selected_indices)].copy()
    print(f"\nTotal test dataset: {len(test_df)} foods")

    # Show category diversity
    n_cats = test_df["food_category"].nunique()
    print(f"  Categories: {n_cats}")

    # Show energy range
    e = test_df["Energy"].dropna()
    print(f"  Energy: min={e.min():.0f}, median={e.median():.0f}, max={e.max():.0f} kcal")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    test_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

"""Shared nutrient statistics across USDA source rows."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Two-sided 95% t critical values for small sample sizes (n-1 df).
_T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

KEY_MACRO_COLS = ["Energy", "Protein", "Carbohydrate", "Total fat"]

# Nutrients commonly present on US branded food labels even when coverage is low.
_LABEL_CORE = {
    "Energy",
    "Protein",
    "Carbohydrate",
    "Total fat",
    "Cholesterol",
    "Sodium",
    "Sugars, Total",
    "Fiber",
    "Fatty acids, total saturated",
}


def normalize_nutrient_value(row: pd.Series, col: str, val) -> float | None:
    """Treat sparse branded-food zeros as missing (USDA pads unreported fields with 0)."""
    if pd.isna(val):
        return None
    v = float(val)
    if v != 0:
        return v
    if row.get("data_type") != "branded_food":
        return v
    coverage = row.get("number_of_nutrients")
    if pd.isna(coverage):
        return v
    if int(coverage) >= 25 or col in _LABEL_CORE:
        return v
    return None


def row_nutrients(row: pd.Series, nutrient_cols: list[str]) -> dict:
    out = {}
    for col in nutrient_cols:
        out[col] = normalize_nutrient_value(row, col, row.get(col))
    return out


def coerce_usda_nutrients(df: pd.DataFrame, nutrient_cols: list[str]) -> pd.DataFrame:
    """Return a copy with normalized nutrient columns for averaging and stats."""
    out = df.copy()
    for col in nutrient_cols:
        if col not in out.columns:
            continue
        out[col] = out.apply(lambda r, c=col: normalize_nutrient_value(r, c, r.get(c)), axis=1)
    return out


def values_conflict(
    vals: list[float],
    *,
    rel_tol: float = 0.15,
    abs_tol: float = 0.5,
) -> bool:
    """True only when 2+ sources report meaningfully different values."""
    if len(vals) <= 1:
        return False
    lo, hi = min(vals), max(vals)
    span = hi - lo
    mean = sum(vals) / len(vals)
    mean_abs = abs(mean)

    # Small absolute values: allow more spread before calling it a conflict.
    if mean_abs < 10:
        if span <= max(3.0, 0.5 * max(mean_abs, 1.0)):
            return False
    elif span <= abs_tol:
        return False

    if mean_abs <= abs_tol:
        return span > max(abs_tol, 0.25 * max(hi, 1.0))

    std = float(np.std(vals, ddof=1))
    return (std / mean_abs) > rel_tol


def macro_agreement(row: pd.Series, reference: pd.Series, key_cols: list[str]) -> float:
    """1.0 = macros match; lower = more disagreement on shared fields."""
    scores: list[float] = []
    for col in key_cols:
        a, b = row.get(col), reference.get(col)
        if pd.isna(a) or pd.isna(b):
            continue
        a, b = float(a), float(b)
        denom = max(abs(a), abs(b), 1.0)
        scores.append(1.0 - min(1.0, abs(a - b) / denom))
    return float(np.mean(scores)) if scores else 1.0


DATA_TYPE_BONUS = {
    "survey_fndds_food": 0.06,
    "sr_legacy_food": 0.06,
    "foundation_food": 0.05,
    "branded_food": 0.0,
    "sub_sample_food": -0.05,
}

SPARSE_BRANDED_COVERAGE = 18
SPARSE_BRANDED_PENALTY = 0.15

# Approximate adult DRV values in the same units as USDA FoodData Central columns.
# Used to detect source rows that are almost certainly data-entry errors.
_NUTRIENT_DRV: dict[str, float] = {
    "Vitamin B-12": 2.4,        # µg
    "Thiamin": 1.2,              # mg
    "Riboflavin": 1.3,           # mg
    "Niacin": 16.0,              # mg
    "Vitamin B-6": 1.7,          # mg
    "Folate, total": 400.0,      # µg
    "Vitamin C": 90.0,           # mg
    "Vitamin D (D2 + D3)": 20.0, # µg
    "Vitamin E": 15.0,           # mg
    "Vitamin A, RAE": 900.0,     # µg
    "Iron": 18.0,                # mg
    "Zinc": 11.0,                # mg
}
# Source rows where any tracked nutrient exceeds this multiple of DRV per 100 g
# are excluded before averaging — real foods almost never exceed ~50× DRV.
_OUTLIER_DRV_CAP = 100.0


def _is_nutrient_outlier(row: pd.Series) -> bool:
    """Return True when any micronutrient value exceeds the DRV-based outlier cap."""
    for col, drv in _NUTRIENT_DRV.items():
        val = row.get(col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        try:
            if float(val) > _OUTLIER_DRV_CAP * drv:
                return True
        except (TypeError, ValueError):
            continue
    return False


def source_quality_score(row: pd.Series) -> float:
    """Rank USDA rows: prefer survey/sr/foundation and full nutrient profiles over sparse branded."""
    sim = float(row["_similarity"])
    coverage = int(row.get("_coverage", 0))
    data_type = str(row.get("data_type") or "")
    bonus = DATA_TYPE_BONUS.get(data_type, 0.0)
    coverage_bonus = min(coverage / 35.0, 0.06)
    penalty = (
        SPARSE_BRANDED_PENALTY
        if data_type == "branded_food" and coverage < SPARSE_BRANDED_COVERAGE
        else 0.0
    )
    return sim + bonus + coverage_bonus - penalty


def select_source_rows(
    candidates: pd.DataFrame,
    *,
    top_sources: int = 5,
    min_source_similarity: float = 0.80,
    max_similarity_drop: float = 0.08,
    min_macro_agreement: float = 0.85,
    key_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Pick up to top_sources USDA rows: quality-ranked anchor + agreeing macros."""
    if candidates.empty:
        return candidates

    key_cols = key_cols or KEY_MACRO_COLS
    ranked = candidates.copy()
    ranked["_quality"] = ranked.apply(source_quality_score, axis=1)

    # Exclude source rows whose micronutrient values exceed the DRV-based outlier cap.
    # This catches data-entry errors (e.g. 480 µg B-12/100 g in a pie crust entry)
    # before they can become an anchor and skew the average.
    non_outlier = ranked[~ranked.apply(_is_nutrient_outlier, axis=1)]
    if not non_outlier.empty:
        ranked = non_outlier

    ordered = ranked.sort_values(["_quality", "_similarity"], ascending=False)
    anchor = ordered.iloc[0]
    top_sim = float(anchor["_similarity"])
    sim_floor = max(min_source_similarity, top_sim - max_similarity_drop)

    picked: list[pd.Series] = [anchor]
    for _, row in ordered.iloc[1:].iterrows():
        if len(picked) >= top_sources:
            break
        if float(row["_similarity"]) < sim_floor:
            continue
        if macro_agreement(row, anchor, key_cols) >= min_macro_agreement:
            picked.append(row)

    return pd.DataFrame(picked)


def source_record(row: pd.Series, nutrient_cols: list[str], similarity: float, coverage: int) -> dict:
    return {
        "foodName": row["foodName"],
        "data_type": row.get("data_type"),
        "similarity": float(similarity),
        "nutrient_coverage": int(coverage),
        "nutrients": row_nutrients(row, nutrient_cols),
    }


def nutrient_stats(rows: pd.DataFrame, nutrient_cols: list[str], source_count: int) -> dict:
    """Compute mean/support/CI per nutrient from selected USDA source rows."""
    out: dict[str, dict] = {}
    denom = max(int(source_count), 1)

    for col in nutrient_cols:
        vals = pd.to_numeric(rows[col], errors="coerce").dropna().astype(float)
        n = int(len(vals))
        if n == 0:
            out[col] = {
                "support": 0,
                "support_pct": 0.0,
                "std": None,
                "ci_low": None,
                "ci_high": None,
                "cv": None,
                "conflicting": False,
            }
            continue

        val_list = vals.tolist()
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            sem = std / math.sqrt(n)
            t_crit = _T_95.get(n, 1.96)
            ci_low = mean - t_crit * sem
            ci_high = mean + t_crit * sem
        else:
            ci_low = ci_high = mean

        cv = (std / abs(mean)) if mean != 0 else None
        out[col] = {
            "support": n,
            "support_pct": round(n / denom, 3),
            "std": round(std, 4) if n > 1 else 0.0,
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "cv": round(cv, 4) if cv is not None else None,
            "conflicting": values_conflict(val_list),
        }
    return out


def average_nutrients(rows: pd.DataFrame, nutrient_cols: list[str]) -> dict:
    out = {}
    for col in nutrient_cols:
        vals = pd.to_numeric(rows[col], errors="coerce").dropna()
        out[col] = float(vals.mean()) if not vals.empty else None
    return out

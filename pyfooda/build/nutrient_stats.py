"""Shared nutrient statistics across USDA source rows."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Two-sided 95% t critical values for small sample sizes (n-1 df).
_T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

KEY_MACRO_COLS = ["Energy", "Protein", "Carbohydrate", "Total fat"]


def row_nutrients(row: pd.Series, nutrient_cols: list[str]) -> dict:
    out = {}
    for col in nutrient_cols:
        val = row.get(col)
        out[col] = None if pd.isna(val) else float(val)
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
    if hi - lo <= abs_tol:
        return False
    mean = sum(vals) / len(vals)
    if abs(mean) <= abs_tol:
        return hi - lo > abs_tol
    std = float(np.std(vals, ddof=1))
    return (std / abs(mean)) > rel_tol


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


def select_source_rows(
    candidates: pd.DataFrame,
    *,
    top_sources: int = 5,
    min_source_similarity: float = 0.80,
    max_similarity_drop: float = 0.08,
    min_macro_agreement: float = 0.85,
    key_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Pick up to top_sources USDA rows: best embedding match + agreeing macros."""
    if candidates.empty:
        return candidates

    key_cols = key_cols or KEY_MACRO_COLS
    ordered = candidates.sort_values("_similarity", ascending=False)
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

"""Shared nutrient statistics across USDA source rows."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Two-sided 95% t critical values for small sample sizes (n-1 df).
_T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def row_nutrients(row: pd.Series, nutrient_cols: list[str]) -> dict:
    out = {}
    for col in nutrient_cols:
        val = row.get(col)
        out[col] = None if pd.isna(val) else float(val)
    return out


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
            }
            continue

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
        }
    return out


def average_nutrients(rows: pd.DataFrame, nutrient_cols: list[str]) -> dict:
    out = {}
    for col in nutrient_cols:
        vals = pd.to_numeric(rows[col], errors="coerce").dropna()
        out[col] = float(vals.mean()) if not vals.empty else None
    return out

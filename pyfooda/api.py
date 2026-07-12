"""Offline recipe-ingredient nutrition API backed by Epicure + USDA."""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

import pandas as pd

_ingredients_df: Optional[pd.DataFrame] = None
_vocab: Optional[list[dict]] = None
_nutrients_df: Optional[pd.DataFrame] = None
_meta: Optional[dict[str, dict]] = None


def _data_path(filename: str) -> str:
    return str(resources.files("pyfooda").joinpath("data").joinpath(filename))


def ensure_data_loaded() -> None:
    global _ingredients_df, _vocab, _nutrients_df, _meta
    if _ingredients_df is None:
        _ingredients_df = pd.read_csv(_data_path("ingredients.csv"))
    if _vocab is None:
        with open(_data_path("epicure_vocabulary.json")) as f:
            _vocab = json.load(f)
    if _nutrients_df is None:
        _nutrients_df = pd.read_csv(_data_path("nutrients.csv"))
    if _meta is None:
        meta_path = _data_path("ingredients_meta.json")
        try:
            with open(meta_path) as f:
                items = json.load(f)
            _meta = {item["ingredient_id"]: item for item in items}
        except FileNotFoundError:
            _meta = {}


def _nutrient_columns() -> list[str]:
    ensure_data_loaded()
    assert _nutrients_df is not None
    return _nutrients_df["nutrientName"].tolist()


def _resolve_ingredient_id(name: str) -> Optional[str]:
    ensure_data_loaded()
    assert _ingredients_df is not None

    key = name.strip().lower()
    by_id = _ingredients_df[_ingredients_df["ingredient_id"].str.lower() == key]
    if not by_id.empty:
        return str(by_id["ingredient_id"].iloc[0])

    by_display = _ingredients_df[_ingredients_df["display_name"].str.lower() == key]
    if not by_display.empty:
        return str(by_display["ingredient_id"].iloc[0])

    snake = key.replace(" ", "_")
    by_snake = _ingredients_df[_ingredients_df["ingredient_id"].str.lower() == snake]
    if not by_snake.empty:
        return str(by_snake["ingredient_id"].iloc[0])

    return None


def list_ingredients() -> list[str]:
    """Return all canonical ingredient ids."""
    ensure_data_loaded()
    assert _ingredients_df is not None
    return _ingredients_df["ingredient_id"].tolist()


def get_display_name(ingredient_id: str) -> Optional[str]:
    """Return the human-readable ingredient name."""
    ensure_data_loaded()
    assert _ingredients_df is not None
    row = _ingredients_df[_ingredients_df["ingredient_id"] == ingredient_id]
    if row.empty:
        return None
    return str(row["display_name"].iloc[0])


def find_ingredients(partial_name: str, limit: int = 10) -> list[str]:
    """Search ingredient ids by partial match on id or display name."""
    ensure_data_loaded()
    assert _ingredients_df is not None
    q = partial_name.lower()
    mask = (
        _ingredients_df["ingredient_id"].str.lower().str.contains(q, na=False)
        | _ingredients_df["display_name"].str.lower().str.contains(q, na=False)
    )
    matches = _ingredients_df[mask]
    matches = matches[matches["source_count"].fillna(0) > 0]
    return matches["ingredient_id"].head(limit).tolist()


def get_nutrients(name: str) -> Optional[dict]:
    """Return averaged nutrient values for an ingredient id or display name."""
    ensure_data_loaded()
    assert _ingredients_df is not None

    ingredient_id = _resolve_ingredient_id(name)
    if ingredient_id is None:
        return None

    row = _ingredients_df[_ingredients_df["ingredient_id"] == ingredient_id]
    if row.empty or int(row["source_count"].fillna(0).iloc[0]) == 0:
        return None

    nutrient_cols = _nutrient_columns()
    values = row[nutrient_cols].iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in values.items()}


def get_sources(name: str) -> Optional[list[dict]]:
    """Return USDA source rows used to build an ingredient profile."""
    ensure_data_loaded()
    ingredient_id = _resolve_ingredient_id(name)
    if ingredient_id is None:
        return None
    item = _meta.get(ingredient_id)
    if not item:
        return None
    return item.get("sources", [])


def get_ingredients_df() -> pd.DataFrame:
    """Return the full ingredients DataFrame."""
    ensure_data_loaded()
    assert _ingredients_df is not None
    return _ingredients_df.copy()


def get_vocabulary() -> list[dict]:
    """Return the Epicure vocabulary bundled with Pyfooda."""
    ensure_data_loaded()
    assert _vocab is not None
    return list(_vocab)


def get_drv_df() -> pd.DataFrame:
    """Return dietary reference values."""
    ensure_data_loaded()
    assert _nutrients_df is not None
    return _nutrients_df.copy()


# Backward-compatible aliases
find_closest_matches = find_ingredients

"""Turn USDA nutrient columns into one usable value per measurement.

The wide FoodData Central export names columns by nutrientName, and several
USDA nutrient ids share a name or a meaning. This module applies three rules
to every row. None of them mentions a food.

1. Same analyte. Two ids that USDA uses for one quantity, in one unit, become
   one value. The current FoodData Central id wins when both are present and
   they disagree. A missing side is filled from the other.

2. A component cannot outweigh its parent. Sugars and fiber are part of
   carbohydrate. Saturated fat and the named fatty acids are part of total fat.
   A value that exceeds its parent is not a measurement, so it is dropped.
   A different assay is not a parent: AOAC fiber is not checked against
   Prosky fiber, only against carbohydrate.

3. Calories have to come from somewhere. When protein, carbohydrate, and fat
   explain less than a quarter of the stated energy, a macro that itself
   supplies under 2 kcal is a padded zero, not a measured zero. Energy stays.
   Alcohol explains this gap for drinks, and this rule does not invent an
   alcohol figure or replace the measured energy with an Atwater estimate.
"""

from __future__ import annotations

import pandas as pd

from pyfooda.build.paths import NUTRIENTS_CSV

# (member ids, preferred id). Preferred is the id FoodData Central publishes now.
SAME_ANALYTE: tuple[tuple[tuple[int, ...], int], ...] = (
    ((2000, 1063), 2000),  # total sugars, grams
)

# (component id, parent id). Both are grams.
WITHIN_PARENT: tuple[tuple[int, int], ...] = (
    (2000, 1005),  # total sugars ⊆ carbohydrate
    (1063, 1005),
    (1079, 1005),  # fiber ⊆ carbohydrate
    (2033, 1005),  # AOAC 2011.25 fiber ⊆ carbohydrate
    (1258, 1004),  # saturated fat ⊆ total fat
    (1272, 1004),  # DHA
    (1278, 1004),  # EPA
    (1404, 1004),  # ALA
)

ENERGY_NAME = "Energy"
PROTEIN_ID = 1003
CARB_ID = 1005
FAT_ID = 1004

# A component may sit this far above its parent before it is rejected.
_ABS_SLACK_G = 0.5
_REL_SLACK = 0.02

# Below this, stated energy is too small for the incomplete-macro test.
_ENERGY_FLOOR_KCAL = 20.0
# Macros must explain at least this share of stated energy.
_EXPLAINED_FLOOR = 0.25
# A macro contributing less than this, inside an unexplained energy total, is a pad.
_PAD_KCAL = 2.0


def unique_nutrient_names(names: list[str]) -> list[str]:
    """Drop repeated names. The catalog lists several ids under one name."""
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def unique_sources(sources: list[dict]) -> list[dict]:
    """One vote per USDA food name. The wide export repeats some rows."""
    seen: set[str] = set()
    out: list[dict] = []
    for src in sources:
        key = str(src.get("foodName", "")).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(src)
    return out


def _names_for(catalog: pd.DataFrame, nutrient_id: int) -> list[str]:
    hit = catalog.loc[catalog["nutrient_id"] == nutrient_id, "nutrientName"]
    return [str(name) for name in hit.tolist()]


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    column = frame[name]
    if isinstance(column, pd.DataFrame):
        column = column.iloc[:, 0]
    return pd.to_numeric(column, errors="coerce")


def _series_or_nan(frame: pd.DataFrame, name: str | None) -> pd.Series:
    if name is None or name not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return _numeric(frame, name)


def resolve_nutrient_frame(
    rows: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a copy with alias columns reconciled and impossible values removed."""
    if catalog is None:
        catalog = pd.read_csv(NUTRIENTS_CSV)
    out = rows.copy().reset_index(drop=True)
    if out.empty:
        return out

    _coalesce_aliases(out, catalog)
    _drop_components_over_parent(out, catalog)
    _drop_padded_macros(out, catalog)
    return out


def _coalesce_aliases(frame: pd.DataFrame, catalog: pd.DataFrame) -> None:
    for member_ids, preferred_id in SAME_ANALYTE:
        columns: list[str] = []
        for nutrient_id in member_ids:
            for name in _names_for(catalog, nutrient_id):
                if name in frame.columns and name not in columns:
                    columns.append(name)
        if not columns:
            continue
        preferred_names = [n for n in _names_for(catalog, preferred_id) if n in columns]
        preferred = preferred_names[0] if preferred_names else columns[0]
        pooled = _series_or_nan(frame, preferred)
        for name in columns:
            if name == preferred:
                continue
            other = _numeric(frame, name)
            # Fill only the gaps. A present preferred value is a different
            # publication of the same analyte, and it is the one we keep.
            pooled = pooled.combine_first(other)
        for name in columns:
            frame[name] = pooled


def _drop_components_over_parent(frame: pd.DataFrame, catalog: pd.DataFrame) -> None:
    for component_id, parent_id in WITHIN_PARENT:
        component_names = [n for n in _names_for(catalog, component_id) if n in frame.columns]
        parent_names = [n for n in _names_for(catalog, parent_id) if n in frame.columns]
        if not component_names or not parent_names:
            continue
        parent = _numeric(frame, parent_names[0])
        for name in component_names:
            component = _numeric(frame, name)
            slack = (parent.abs() * _REL_SLACK).clip(lower=_ABS_SLACK_G)
            impossible = component.notna() & parent.notna() & (component > parent + slack)
            cleaned = component.mask(impossible)
            frame[name] = cleaned


def _drop_padded_macros(frame: pd.DataFrame, catalog: pd.DataFrame) -> None:
    protein_name = next(iter(_names_for(catalog, PROTEIN_ID)), None)
    carb_name = next(iter(_names_for(catalog, CARB_ID)), None)
    fat_name = next(iter(_names_for(catalog, FAT_ID)), None)
    if ENERGY_NAME not in frame.columns:
        return
    energy = _numeric(frame, ENERGY_NAME)
    protein = _series_or_nan(frame, protein_name)
    carb = _series_or_nan(frame, carb_name)
    fat = _series_or_nan(frame, fat_name)
    explained = 4.0 * protein.fillna(0) + 4.0 * carb.fillna(0) + 9.0 * fat.fillna(0)
    incomplete = energy.notna() & (energy >= _ENERGY_FLOOR_KCAL) & (explained < _EXPLAINED_FLOOR * energy)
    for name, series, kcal_per_g in (
        (protein_name, protein, 4.0),
        (carb_name, carb, 4.0),
        (fat_name, fat, 9.0),
    ):
        if name is None or name not in frame.columns:
            continue
        padded = incomplete & series.notna() & (series * kcal_per_g < _PAD_KCAL)
        frame[name] = series.mask(padded)

"""Choose USDA rows that are the ingredient named, and that agree with each other.

A row qualifies when the ingredient's main word is the USDA food (the text
before the comma, or the first specific term after it), not a later modifier.
Rows that add a preparation the ingredient name does not use — seed, dried,
flour, tonic, condensed — lose to a row that does not add it. Rows whose
calories are several times the plain row are left out of the average.

A stored row is kept when USDA has no food under this name. Bourbon stays
whiskey. The function does not invent that equivalence, and it does not
replace it with a fuzzy neighbor.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

_STOP = {
    "and", "or", "with", "of", "the", "a", "an", "no", "not", "to", "in",
    "from", "for", "as", "on", "ns", "nfs", "type", "regular", "all",
}

# A token here is a different food or preparation when the ingredient
# name does not already contain it.
_PREP = {
    "seed", "seeds", "oil", "flour", "powder", "dried", "juice", "tonic",
    "flavored", "enhanced", "cube", "cubes", "bouillon", "condensed",
    "baby", "babyfood", "fried", "breaded", "pickled", "smoked", "malt",
    "ground", "chip", "chips", "latte", "cappuccino", "macchiato",
    "mocha", "dehydrated", "frozen",
}

_PLAIN = ("raw", "nfs", "brewed", "fluid", "drinking", "tap", "plain", "fresh")

# A trailing word that names the form of the food, not a different food.
_QUALIFIER = {
    "table", "iodized", "kosher", "granulated", "bottled", "generic",
    "municipal", "carbonated", "unsweetened", "sweetened",
}


def _stems(word: str) -> set[str]:
    out = {word}
    if word.endswith("ies") and len(word) > 4:
        out.add(word[:-3] + "y")
    if word.endswith("es") and len(word) > 4:
        out.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        out.add(word[:-1])
    return out


def _tokens(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOP and len(t) > 1
    ]


def _prep_stems() -> set[str]:
    stems: set[str] = set()
    for word in _PREP:
        stems |= _stems(word)
    return stems


_PREP_STEMS = _prep_stems()


def name_score(query: str, food_name: str) -> float | None:
    """How clearly this USDA name is the ingredient. None when the main word is absent."""
    q_tokens = _tokens(query)
    if not q_tokens or not food_name:
        return None
    # English compounds are head-final: "chicken broth" is a broth, "apricot brandy" a brandy.
    principal = q_tokens[-1]
    segments = [part.strip() for part in food_name.split(",")]
    name_stems: set[str] = set()
    for token in _tokens(food_name):
        name_stems |= _stems(token)

    def hits(token: str, place: list[str]) -> bool:
        want = _stems(token)
        for word in place:
            if want & _stems(word):
                return True
        return False

    score = None
    for index, words in enumerate(_tokens(part) for part in segments):
        if not hits(principal, words):
            continue
        raw_first = re.findall(r"[a-z0-9]+", segments[index].lower())
        starts = bool(raw_first) and bool(_stems(raw_first[0]) & _stems(principal))
        if index == 0:
            score = 3.0
        elif index == 1 or starts:
            # "Pork, cured, bacon": the segment is the food.
            # "with salt" and "fresh water" do not start with the food.
            score = 2.0
        else:
            break
        # "salt pork" starts with salt and is still pork.
        query_stems: set[str] = set()
        for token in q_tokens:
            query_stems |= _stems(token)
        for word in words:
            stems = _stems(word)
            if stems & _stems(principal) or stems & query_stems or word in _PLAIN:
                continue
            score -= 2.0
        break
    if score is None:
        return None

    for token in q_tokens:
        if token == principal:
            continue
        score += 1.0 if _stems(token) & name_stems else -0.5

    for token in _tokens(food_name):
        if _stems(token) & _PREP_STEMS and not (_stems(token) & set().union(*(_stems(t) for t in q_tokens))):
            score -= 2.0

    lowered = food_name.lower()
    if any(re.search(rf"\b{hint}\b", lowered) for hint in _PLAIN):
        score += 0.4
    return score


def _drop_added_preparations(
    query: str,
    rows: list[tuple[str, float | None]],
) -> list[tuple[str, float | None]]:
    """Drop seed, dried, or ground rows when a row without that preparation is present."""
    query_stems: set[str] = set()
    for token in _tokens(query):
        query_stems |= _stems(token)

    def preparations(food_name: str) -> set[str]:
        found = set()
        for token in _tokens(food_name):
            stems = _stems(token)
            if stems & _PREP_STEMS and not (stems & query_stems):
                found |= stems & _PREP_STEMS
        return found

    marked = [(row, preparations(row[0])) for row in rows]
    if any(not prep for _, prep in marked) and any(prep for _, prep in marked):
        return [row for row, prep in marked if not prep]
    return rows


def _anchor(query: str, rows: list[tuple[str, float | None]]) -> float | None:
    energies = [energy for _, energy in rows if energy is not None]
    median = sorted(energies)[len(energies) // 2] if energies else 0.0
    ranked: list[tuple[tuple[float, int, int, float], float | None]] = []
    for name, energy in rows:
        score = name_score(query, name)
        plain = 1 if any(re.search(rf"\b{hint}\b", name.lower()) for hint in _PLAIN) else 0
        extras = len(_extra_tokens(query, name))
        distance = abs((energy if energy is not None else median) - median)
        key = (score if score is not None else -1.0, -extras, plain, -distance)
        ranked.append((key, energy))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _near_anchor(energy: float | None, anchor: float | None) -> bool:
    if energy is None or anchor is None:
        return True
    if anchor < 1:
        return energy < 15
    lo, hi = min(energy, anchor), max(energy, anchor)
    if lo < 1:
        return hi < 15
    return hi / lo <= 5.0


def _extra_tokens(query: str, food_name: str) -> list[str]:
    """Words in the USDA name that the ingredient did not use and that are not a plain-form hint."""
    query_stems: set[str] = set()
    for token in _tokens(query):
        query_stems |= _stems(token)
    extras = []
    for token in _tokens(food_name):
        if _stems(token) & query_stems or token in _PLAIN:
            continue
        extras.append(token)
    return extras


def _all_query_tokens(query: str, food_name: str) -> bool:
    name_stems: set[str] = set()
    for token in _tokens(food_name):
        name_stems |= _stems(token)
    for token in _tokens(query):
        if not (_stems(token) & name_stems):
            return False
    return bool(_tokens(query))


def choose_names(
    query: str,
    current: list[tuple[str, float | None]],
    catalog: list[tuple[str, float | None]],
    *,
    limit: int = 5,
) -> list[str]:
    """Return USDA food names to average for this ingredient."""
    def scored(rows: list[tuple[str, float | None]], floor: float) -> list[tuple[str, float | None]]:
        return [
            row
            for row in rows
            if (sc := name_score(query, row[0])) is not None and sc >= floor
        ]

    pool = scored(current, 2.0)
    if not pool:
        # The stored rows are a different food. Take a USDA row whose name is
        # this ingredient and nothing else, such as "Water, tap" for water.
        pool = [
            row
            for row in catalog
            if (sc := name_score(query, row[0])) is not None
            and sc >= 2.5
            and _all_query_tokens(query, row[0])
            and all(token in _QUALIFIER or token in _PLAIN for token in _extra_tokens(query, row[0]))
        ]
    if not pool:
        pool = list(current)
    pool = _drop_added_preparations(query, pool)

    # One row per USDA name. Repeated exports of the same food are not more evidence.
    deduped: dict[str, tuple[str, float | None]] = {}
    for name, energy in pool:
        key = name.strip().lower()
        if key not in deduped:
            deduped[key] = (name, energy)
    pool = list(deduped.values())

    anchor = _anchor(query, pool)
    pool = [row for row in pool if _near_anchor(row[1], anchor)]

    def sort_key(row: tuple[str, float | None]) -> tuple:
        score = name_score(query, row[0])
        return (score if score is not None else -1.0, row[0])

    pool.sort(key=sort_key, reverse=True)
    return [name for name, _ in pool[:limit]]


_AUTHORITATIVE = ("sr_legacy_food", "foundation_food", "survey_fndds_food")


def apply_alignment(
    meta_path: Path | None = None,
    usda_path: Path | None = None,
) -> int:
    """Rewrite ingredients_meta.json so each ingredient's USDA rows pass choose_names.

    Returns the number of ingredients whose source list changed.
    """
    from pyfooda.build.paths import META_JSON, USDA_CSV

    meta_path = meta_path or META_JSON
    usda_path = usda_path or USDA_CSV
    usda = pd.read_csv(usda_path)
    auth = usda[usda["data_type"].isin(_AUTHORITATIVE)].copy()
    auth["Energy"] = pd.to_numeric(auth["Energy"], errors="coerce")
    auth = auth.sort_values("number_of_nutrients", ascending=False)
    auth = auth.drop_duplicates("foodName")

    catalog: list[tuple[str, float | None]] = []
    energy: dict[str, float | None] = {}
    data_types: dict[str, str] = {}
    for food_name, kcal, data_type in zip(
        auth["foodName"].astype(str),
        auth["Energy"],
        auth["data_type"].astype(str),
    ):
        value = None if pd.isna(kcal) else float(kcal)
        catalog.append((food_name, value))
        energy[food_name] = value
        data_types[food_name] = data_type

    def relevant(query: str) -> list[tuple[str, float | None]]:
        tokens = _tokens(query)
        if not tokens:
            return []
        stems = _stems(tokens[-1])
        return [(food_name, kcal) for food_name, kcal in catalog if any(stem in food_name.lower() for stem in stems)]

    meta = json.loads(meta_path.read_text())
    changed = 0
    for item in meta:
        display = item.get("display_name") or ""
        current: list[tuple[str, float | None]] = []
        by_name: dict[str, dict] = {}
        for src in item.get("sources") or []:
            food_name = src.get("foodName") or ""
            by_name[food_name] = src
            kcal = energy.get(food_name)
            if kcal is None:
                stored = (src.get("nutrients") or {}).get("Energy")
                kcal = float(stored) if isinstance(stored, (int, float)) else None
            current.append((food_name, kcal))
        chosen = choose_names(display, current, relevant(display))
        if [food_name for food_name, _ in current] == chosen:
            continue
        new_sources = []
        for food_name in chosen:
            if food_name in by_name:
                new_sources.append(by_name[food_name])
            else:
                new_sources.append(
                    {
                        "foodName": food_name,
                        "data_type": data_types.get(food_name),
                        "similarity": None,
                        "nutrient_coverage": None,
                        "nutrients": {},
                    }
                )
        item["sources"] = new_sources
        changed += 1

    if changed:
        meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Aligned sources for {changed} ingredients")
    return changed

import pandas as pd
from importlib import resources
from rank_bm25 import BM25Okapi

# Global variables to store DataFrames, initialized as None
foods_df = None
nutrients_df = None
_bm25_index = None
_food_names = None
_food_names_lower = None

def _data_path(filename: str) -> str:
    """Return the filesystem path to a data file shipped with the package."""
    return str(resources.files('pyfooda').joinpath('data', filename))

def ensure_data_loaded():
    """Load the CSV files into DataFrames if not already loaded."""
    global foods_df, nutrients_df, _bm25_index, _food_names, _food_names_lower
    if foods_df is None:
        foods_df = pd.read_csv(_data_path('foods_aggregated.csv'))
        # Build BM25 search index over food names
        _food_names = foods_df['foodName'].dropna().tolist()
        _food_names_lower = [n.lower() for n in _food_names]
        _bm25_index = BM25Okapi([n.split() for n in _food_names_lower])
    if nutrients_df is None:
        nutrients_df = pd.read_csv(_data_path('nutrients.csv'))

def get_category(foodName):
    """Return the category of the specified food item."""
    ensure_data_loaded()
    row = foods_df[foods_df['foodName'].str.lower() == foodName.lower()]
    if row.empty:
        return None
    return row['category'].iloc[0]

def get_nutrients(foodName):
    """Return a dictionary of nutrient values for the specified food item."""
    ensure_data_loaded()
    row = foods_df[foods_df['foodName'].str.lower() == foodName.lower()]
    if row.empty:
        return None
    # Only return nutrients that exist in both nutrients.csv and the food table
    nutrient_columns = [
        c for c in nutrients_df['nutrientName'].unique()
        if c in foods_df.columns
    ]
    nutrients = row[nutrient_columns].iloc[0].to_dict()
    return nutrients

def get_portion_gram_weight(foodName):
    """Return the portion gram weight of the specified food item as a float."""
    ensure_data_loaded()
    if 'portion_gram_weight' not in foods_df.columns:
        return None
    row = foods_df[foods_df['foodName'].str.lower() == foodName.lower()]
    if row.empty:
        return None
    val = row['portion_gram_weight'].iloc[0]
    return float(val) if pd.notna(val) else None

def get_portion_unit_name(foodName):
    """Return the portion unit name of the specified food item."""
    ensure_data_loaded()
    if 'portion_unit_name' not in foods_df.columns:
        return None
    row = foods_df[foods_df['foodName'].str.lower() == foodName.lower()]
    if row.empty:
        return None
    val = row['portion_unit_name'].iloc[0]
    return val if pd.notna(val) else None

def find_closest_matches(partialName, n=10):
    """Return up to *n* food names ranked by relevance (BM25 + brevity).

    Uses BM25 for term-importance scoring, then re-ranks with bonuses
    for exact matches, prefix matches, and shorter (more generic) names.
    """
    ensure_data_loaded()
    q = partialName.lower()
    q_tokens = q.split()
    scores = _bm25_index.get_scores(q_tokens)

    # Gather all candidates with a positive BM25 score
    candidates = []
    for i in range(len(_food_names)):
        if scores[i] <= 0:
            continue
        nl = _food_names_lower[i]
        s = float(scores[i])
        # Exact match bonus
        if nl == q:
            s += 100
        # Name starts with query
        elif nl.startswith(q + ' ') or nl == q:
            s += 20
        # All query tokens present in name
        elif all(t in nl for t in q_tokens):
            s += 10
        # Brevity: prefer shorter, more generic names
        word_count = len(nl.split())
        s += 8.0 / (1 + word_count)
        candidates.append((s, i))

    candidates.sort(key=lambda x: -x[0])
    return [_food_names[i] for _, i in candidates[:n]]

def get_fooddata_df():
    """Return the fooddata DataFrame."""
    ensure_data_loaded()
    return foods_df

def get_drv_df():
    """Return the DRV DataFrame."""
    ensure_data_loaded()
    return nutrients_df
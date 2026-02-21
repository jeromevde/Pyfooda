# Pyfooda

A compact Python API for **USDA FoodData Central** with an LLM-powered
pipeline that compresses ~296k raw food items into a clean everyday
nutrition database.

## Repository structure

```
pyfooda/                       # Installable Python package (end-user API)
  api.py                       #   lookup functions
  data/
    fooddata.csv               #   296k preprocessed USDA foods
    nutrients.csv              #   nutrient metadata + daily reference values

scripts/                       # Data pipeline (not part of the package)
  build_fooddata.py            #   Step 1 — raw USDA CSV → fooddata.csv
  aggregate.py                 #   Step 2 — fooddata.csv → foods_aggregated.json
  aggregator.py                #   aggregation engine (used by aggregate.py)
  aggregation_prompt.txt       #   tweakable LLM prompt
  nutrients_drv.py             #   nutrient definitions + DRVs
  requirements.txt             #   pipeline dependencies
```

---

## 1 · Install the package

```bash
pip install pyfooda          # from PyPI
# or
pip install -e .             # editable install from source
```

The package ships with the preprocessed USDA data — no downloads needed.

## 2 · Use the lookup API

```python
import pyfooda as pf

pf.find_closest_matches("cheddar")           # ranked by relevance (BM25)
pf.get_nutrients("Cheddar Cheese")           # dict of nutrient values
pf.get_category("Cheddar Cheese")            # "Cheese"
pf.get_portion_gram_weight("Cheddar Cheese") # grams per portion
pf.get_portion_unit_name("Cheddar Cheese")   # e.g. "cup, shredded"

df  = pf.get_fooddata_df()   # full 296k × 44 DataFrame
drv = pf.get_drv_df()        # daily reference values per nutrient
```

| Function | Returns |
|----------|---------|
| `get_category(name)` | Food category (`str`) |
| `get_nutrients(name)` | `dict[nutrient → value]` or `None` |
| `get_portion_gram_weight(name)` | `float` or `None` |
| `get_portion_unit_name(name)` | `str` or `None` |
| `find_closest_matches(partial, n=10)` | `list[str]` ranked by relevance |
| `get_fooddata_df()` | Full food DataFrame |
| `get_drv_df()` | Nutrient DRV DataFrame |

---

## 3 · Data pipeline (for contributors / rebuilding)

The `scripts/` directory contains the full pipeline that produces the
data shipped with the package. You only need this if you want to
rebuild from a newer USDA release or re-run the aggregation.

### Prerequisites

```bash
pip install -r scripts/requirements.txt
```

### Step 1 — Build `fooddata.csv` from raw USDA download

1. Download the CSV bundle from
   [FoodData Central](https://fdc.nal.usda.gov/download-datasets)
2. Extract it (e.g. `~/Downloads/FoodData_Central_csv_2024-10-31/`)
3. Run:

```bash
python scripts/build_fooddata.py ~/Downloads/FoodData_Central_csv_2024-10-31
```

This reads the raw USDA tables (`food.csv`, `food_nutrient.csv`,
`food_category.csv`, etc.), joins and pivots them, and writes the
result to `pyfooda/data/fooddata.csv` + `pyfooda/data/nutrients.csv`.

### Step 2 — Aggregate into a compact everyday database

The raw database has **295,943 items** — dozens of entries for
"cheddar cheese" alone. The aggregator uses an LLM to classify each
food as:

| Action | Meaning |
|--------|---------|
| **CREATE** | Start a new generic food (e.g. "Cheddar Cheese") |
| **ADD** | Merge into an existing generic (nutrients averaged) |
| **IGNORE** | Skip (supplements, additives, unidentifiable) |

The LLM sees each food's name, category, **nutrient profile**, and
the closest existing entries, so it makes nutritionally-informed
decisions (e.g. "Tonic Water" ≠ "Lime Juice").

```bash
export OPENROUTER_API_KEY="sk-or-..."

# Quick test — first 1000 items (~20 API calls, ~2 min)
python scripts/aggregate.py test

# Full run — all 296k items
python scripts/aggregate.py full

# Resume after interruption
python scripts/aggregate.py full --resume
```

**Output:**

| File | Description |
|------|-------------|
| `pyfooda/data/foods_aggregated.json` | Generic name, averaged nutrients, source USDA IDs |
| `pyfooda/data/foods_aggregated.csv` | Flat CSV for quick inspection |

### Tweaking the aggregation

Edit `scripts/aggregation_prompt.txt` to change how the LLM classifies
foods. For example you could add:

- *"Merge all yogurt flavors into a single Yogurt entry"*
- *"Keep organic and conventional separate"*
- *"Ignore all baby food"*

## License

MIT

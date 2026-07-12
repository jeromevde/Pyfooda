# Pyfooda

Offline Python API for recipe ingredient nutrition — built from the [Epicure](https://huggingface.co/datasets/Kaikaku/epicure-corpus-resources) ingredient vocabulary and [USDA FoodData Central](https://fdc.nal.usda.gov/) nutrient data.

---

## Using the library

```bash
pip install pyfooda
```

```python
import pyfooda as pf

# Search canonical recipe ingredients
pf.find_ingredients("hazelnut")
# -> ['chocolate_hazelnut_spread', 'hazelnut', 'hazelnut_oil', ...]

# Nutrients per 100g (averaged from top USDA matches)
pf.get_nutrients("chocolate_hazelnut_spread")
pf.get_nutrients("Chocolate Hazelnut Spread")  # display name also works

# USDA evidence used for the profile
pf.get_sources("chocolate_hazelnut_spread")

# Full tables
pf.get_ingredients_df()
pf.get_vocabulary()
pf.get_drv_df()
```

---

## Building the database

The library ships a pre-built `ingredients.csv`. To rebuild from USDA source data:

```bash
pip install -r requirements-build.txt
python scripts/build_database.py
```

For each of the 1,790 Epicure ingredients, the build script:

1. Embeds the ingredient name and all USDA food names (`BAAI/bge-small-en-v1.5`)
2. Retrieves the top 50 USDA matches by cosine similarity
3. Keeps the top 5 with the best nutrient coverage
4. Averages their nutrient values into one profile per ingredient

Outputs:

- `pyfooda/data/ingredients.csv` — nutrient profiles
- `pyfooda/data/ingredients_meta.json` — USDA source traceability

---

## Data sources

| Source | Role | License |
|---|---|---|
| [Epicure vocabulary](https://huggingface.co/datasets/Kaikaku/epicure-corpus-resources) | 1,790 canonical recipe ingredient names | CC BY 4.0 |
| USDA FoodData Central | Nutrient measurements | Public domain |
| `BAAI/bge-small-en-v1.5` | Embedding model for USDA matching | MIT |

Epicure vocabulary attribution: Radzikowski & Chen, KAIKAKU.AI.

---

## Project layout

```
pyfooda/
  api.py                         # public API
  data/
    epicure_vocabulary.json      # 1,790 ingredient names
    ingredients.csv              # built nutrient database
    ingredients_meta.json        # USDA source mapping
    fooddata.csv                 # raw USDA input (build only)
    nutrients.csv                # nutrient metadata + DRV
scripts/
  build_database.py              # database build pipeline
```

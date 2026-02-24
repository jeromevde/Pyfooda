# Pyfooda

Python API + LLM pipeline that compresses **~296k USDA FoodData Central**
items into a clean everyday nutrition database. Nutrients per 100g,
with portion-size conversion.

## Structure

```
pyfooda/                       # Installable package (end-user API)
  api.py                       #   lookup functions
  data/
    fooddata.csv               #   296k preprocessed USDA foods
    foods_aggregated.json      #   aggregated generic foods
    nutrients.csv              #   nutrient metadata + DRVs

scripts/                       # Data pipeline
  build_fooddata.py            #   Step 1 — raw USDA CSV → fooddata.csv
  aggregate.py                 #   Step 2 — fooddata.csv → foods_aggregated.json
  aggregator.py                #   aggregation engine (embedding search + LLM)
  aggregation_prompt.txt       #   tweakable LLM prompt
  nutrients_drv.py             #   nutrient definitions + DRVs

docs/                          # Static website (GitHub Pages)
  index.html                   #   search UI with portion-size dropdown
```

---

## Install

```bash
pip install pyfooda          # from PyPI
pip install -e .             # or editable from source
```

## API

```python
import pyfooda as pf

pf.find_closest_matches("cheddar")           # ranked by BM25
pf.get_nutrients("Cheddar Cheese")           # dict of nutrient values
pf.get_category("Cheddar Cheese")            # "Cheese"
pf.get_portion_gram_weight("Cheddar Cheese") # grams per portion
pf.get_portion_unit_name("Cheddar Cheese")   # e.g. "cup, shredded"
pf.get_fooddata_df()                         # full 296k DataFrame
pf.get_drv_df()                              # daily reference values
```

---

## Data pipeline

Rebuild the aggregated database from a newer USDA release or with
different LLM settings. Requires `pip install -r scripts/requirements.txt`.

### Step 1 — Build `fooddata.csv`

```bash
python scripts/build_fooddata.py ~/Downloads/FoodData_Central_csv_2024-10-31
```

### Step 2 — Aggregate

The aggregator uses **sentence-transformers** (BAAI/bge-small-en-v1.5) +
**FAISS** for semantic search, then sends batches to an LLM that decides
CREATE / ADD / IGNORE / RENAME for each food. Nutrients are averaged
per 100g; all source portion sizes are preserved.

```bash
export OPENROUTER_API_KEY="sk-or-..."

python scripts/aggregate.py test --batch-size 30   # curated test set
python scripts/aggregate.py full                    # all ~296k items
python scripts/aggregate.py full --resume           # resume from checkpoint
```

**Interrupt anytime** — the checkpoint uses the same `{meta, foods}`
JSON format as the output, so you can open it, inspect the results,
and resume when ready.

### Quality controls

| Mechanism | Purpose |
|-----------|---------|
| Foundation lock | Foundation foods create groups; branded items can't dilute them |
| Energy gate | `max(3.5 × MAD, 200 kcal)` rejects nutritional outliers |
| Embedding dedup | Cosine > 0.88 auto-merges duplicate CREATE names |
| Intra-batch dedup | Same generic name within a batch → merge, not duplicate |

### Tuning

Edit `scripts/aggregation_prompt.txt` to change LLM behavior, e.g.:
*"Merge all yogurt flavors"*, *"Keep organic separate"*, *"Ignore baby food"*.

## Output format

Both checkpoint and output share the same JSON structure:

```json
{
  "meta": { "processed_count": 296000, "stats": { "created": 30000, ... } },
  "foods": [
    {
      "id": 1,
      "generic_name": "Cheddar Cheese",
      "food_category": "Cheese",
      "nutrients": { "Energy": 403, "Protein": 24.9, ... },
      "source_ids": [101, 205, ...],
      "source_names": ["Cheese, cheddar", ...],
      "portion_gram_weights": [132.0, 28.35, ...],
      "portion_unit_names": ["1 cup, shredded", "1 oz", ...],
      "count": 47
    }
  ]
}
```

## Website

`docs/index.html` — single-file static site (GitHub Pages). Features:
BM25 + prefix + source-name search, nutrient bars with %DV, and a
**portion-size dropdown** that rescales all nutrient values.

## License

MIT

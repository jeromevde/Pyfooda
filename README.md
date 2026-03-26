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
**FAISS** for semantic search, then sends each food (streaming, 1 item/call)
to an LLM that decides CREATE / ADD / IGNORE / RENAME. Nutrients are averaged
per 100g; all source portion sizes are preserved.

```bash
export OPENROUTER_API_KEY="sk-or-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
python scripts/aggregate.py test --timeout-seconds 240
python scripts/aggregate.py full --timeout-seconds 240
python scripts/aggregate.py full --resume

# Local Ollama (OpenAI-compatible endpoint)
python scripts/aggregate.py test --provider ollama --model qwen2.5:14b
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
| Streaming decisions | Exactly one food per LLM call (no intra-batch confusion possible) |
| ADD-id guard | ADD ids must be from shown nearest-neighbor candidates |

### Tuning

Edit `scripts/aggregation_prompt.txt` to change LLM behavior, e.g.:
*"Merge all yogurt flavors"*, *"Keep organic separate"*, *"Ignore baby food"*.

## Benchmark trace (2026-03-26)

All runs used streaming mode (1 item/call), same prompt, and same test set.

### Full curated test set (244 items)

| Provider / Model | Time | Final groups | Created | Added | Ignored | Parse errors | Renamed |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenRouter `google/gemini-2.0-flash-lite-001` | ~4m10s | 32 | 36 | 175 | 33 | 0 | 80 |
| Ollama `qwen2.5:3b` | ~27m | 58 | 60 | 151 | 33 | 23 | 43 |

### Controlled speed sample (30 items)

| Provider / Model | Real time | Final groups | Parse errors |
|---|---:|---:|---:|
| OpenRouter `google/gemini-2.0-flash-lite-001` | 45.63s | 10 | 0 |
| Ollama `qwen2.5:3b` | 137.07s | 5 | 11 |

### Stored result artifacts

- OpenRouter full test output:
  - `tests/test_aggregated.json`
  - `tests/test_aggregated.csv`
- Ollama full test output:
  - `tests/test_aggregated_ollama.json`
  - `tests/test_aggregated_ollama.csv`
- 30-item comparison outputs:
  - `tests/test_aggregated_or_30.json`
  - `tests/test_aggregated_or_30.csv`
  - `tests/test_aggregated_ollama_30b.json`
  - `tests/test_aggregated_ollama_30b.csv`

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

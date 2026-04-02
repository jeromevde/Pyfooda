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
  run_aggregation.py           #   single standalone batched pipeline (argparse)
  aggregator.py                #   aggregation engine (search + LLM decision logic)
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

### Step 2 — Aggregate (single standalone script)

Use only:

```bash
export OPENROUTER_API_KEY="sk-or-..."
python scripts/run_aggregation.py --mode test --batch-size 16
python scripts/run_aggregation.py --mode full --limit 1000 --batch-size 16
```

Why batched mode only:
- streaming variants were tested and kept as historical trace below,
- but were slower / less reliable for this repo.

#### Batch-size strategy

For `google/gemini-2.0-flash-lite-001`, practical context fit and quality/speed tradeoff on the 244-item curated test set:
- `batch_size=8`  → slower, more calls
- `batch_size=24` → fastest but higher parse/noise risk
- **`batch_size=16` (recommended)** → good quality with strong speed/cost

### Quality controls

| Mechanism | Purpose |
|-----------|---------|
| Foundation lock | Foundation foods create groups; branded items cannot dilute them |
| Energy gate | `max(3.5 × MAD, 200 kcal)` rejects nutritional outliers |
| Embedding dedup | Cosine > 0.88 auto-merges duplicate CREATE names |
| ADD-id guard | ADD ids must come from shown nearest neighbors |
| Intra-batch naming rule | model must reuse the exact same CREATE name for same target group |

### Expected aggregation shape for the curated test dataset

Reference intent (used to evaluate quality):

- **Yogurt family**: nonfat / whole-plain / greek / flavored as distinct practical groups
- **Apple family**: fresh apple separate from pie/dessert forms
- **Lentil family**: dry lentils separate from cooked lentils
- **Lemon family**: lemon juice separate from lemonade/drink-mix style products
- **Ham family**: ham variants grouped when nutritionally aligned

### Experiments in this PR (kept concise)

#### A) Curated test set (244 items), OpenRouter + Gemini Flash-Lite

| Batch size | Time | API calls | Final groups | Parse errors | Estimated full cost (from test metrics) |
|---|---:|---:|---:|---:|---:|
| 8  | 138.78s | 31 | 18 | 23 | ~$18.80 |
| 16 | 107.56s | 16 | 17 | 8  | ~$9.70  |
| 24 | 85.73s  | 11 | 19 | 26 | ~$6.67  |

**Finding:** `batch_size=16` gives the best quality/speed compromise.

#### B) Real database slice (1000 rows, offset 5000), OpenRouter + Gemini Flash-Lite, batch=16

- Processed: 1000
- Time: 148.07s
- API calls: 63
- Final groups: 565
- Parse errors: 35
- Estimated full runtime: ~12h10m
- Estimated full cost: ~$9.32

Assessment:
- this offset is a harder mixed-category segment (meats/dishes/restaurant-style items),
- output labels are mostly readable,
- parse reliability is still the main weakness on hard slices.

Pipeline improvements for high-quality production DB:
1. **Strict JSON decision schema (no retry)** to reduce parser ambiguity.
2. **Stronger IGNORE policy** for irrelevant over-specific/noisy items.
3. **Adaptive batching** (default 16; fallback to 8 on hard chunks).
4. **Candidate compacting** (shorter prompt + top-k=5) to reduce context pressure.

### Historical trace (streaming attempts, not kept as active pipeline)

| Variant | 30-item speed | Quality summary | Cost expectation |
|---|---:|---|---:|
| Together streaming (`Qwen2.5-7B-Turbo`) | ~26.7s | usable but more aggressive merges | higher (~$207 full est. with prior assumptions) |
| Ollama local streaming (`qwen2.5:3b`) | ~101s | slower + noisier labels/parse behavior | $0 API, but low throughput |

Conclusion from trace: streaming was explored, benchmarked, and documented, but batched OpenRouter is more practical for a usable database.

### Stored result artifacts

- `tests/exp_latest_1000_off5000_b16.json/.csv/.metrics.json`
- `tests/exp_latest_1000_off5000_b16.groups.txt` (quick quality review format: `group: item1, item2, ...`)

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

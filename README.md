# Pyfooda

Offline Python API for the [USDA FoodData Central](https://fdc.nal.usda.gov/) database — no API key required.

---

## 1. Using the library

```bash
pip install pyfooda
```

```python
import pyfooda as pf

# Search by partial name
pf.find_closest_matches('apple')

# Category, nutrients, portion
pf.get_category('Apple, raw')
pf.get_nutrients('Apple, raw')             # dict of nutrient → value
pf.get_portion_gram_weight('Apple, raw')   # e.g. 138.0
pf.get_portion_unit_name('Apple, raw')     # e.g. "medium"

# Raw DataFrames for custom analysis
df     = pf.get_fooddata_df()   # all foods × nutrients
drv_df = pf.get_drv_df()        # dietary reference values
```

| Function | Returns |
|---|---|
| `find_closest_matches(partial)` | up to 10 food names containing the string |
| `get_category(name)` | category string, `'Other'` if not found |
| `get_nutrients(name)` | `dict[nutrient → float]` or `None` |
| `get_portion_gram_weight(name)` | `float` or `None` |
| `get_portion_unit_name(name)` | `str` or `None` |
| `get_fooddata_df()` | full foods DataFrame |
| `get_drv_df()` | DRV reference DataFrame |

---

## 2. Aggregation pipeline

The pipeline collapses ~296 k USDA food entries into a compact generic food database using an LLM.
Scripts are in `scripts/` and must be run from that directory.

**LLM backend** — set `OPENROUTER_API_KEY` to use OpenRouter.
Without a key the scripts automatically fall back to the local [VS Code Copilot proxy](https://github.com/hyorman/copilot-proxy) at `http://127.0.0.1:3000`.

```bash
cd scripts

# ── 1. TEST SET EVALUATION ────────────────────────────────────────────────────
# Run the aggregator on the 200-row stratified sample, then score it.
# Do this after every prompt or model change to track quality.

python run_aggregation.py --mode test --batch-size 24
python score_aggregation.py ../tests/batch_test_aggregated.json

# Regenerate the ground-truth test set from the real CSV:
# python generate_test_set.py


# ── 2. SLICE RUN (qualitative check before committing to a full run) ──────────
# Process 1 000 foods at a non-zero offset.
# The dataset is sorted by data type then category, so neighbouring rows are
# correlated — a non-zero offset samples a different region of the data.

python run_aggregation.py --mode full --batch-size 24 --limit 1000 --offset 5000 \
    --output ../tests/slice_5000.json

# Human-friendly group listing:  tests/slice_5000.groups.txt
# Throughput + cost estimate:    tests/slice_5000.metrics.json


# ── 3. FULL RUN ───────────────────────────────────────────────────────────────
# Processes all ~296 k foods. Resumes automatically from the last checkpoint.

python run_aggregation.py --mode full --batch-size 24 --resume

# Output:       pyfooda/data/foods_aggregated.json  (and .csv)
# Checkpoints:  checkpoints_batch/  (saved every 500 items)
```

---

## 3. Experiments

Scoring is computed by `scripts/score_aggregation.py` against `tests/test_set.json`.
Weights: merge 0.35 · split 0.45 · name\_quality 0.20.

| Date | Test set | Items | Model | Overall | Merge | Split | Name quality | Notes |
|------|----------|-------|-------|---------|-------|-------|--------------|-------|
| 2026-05-17 | v2.0 (200 cases) | 355 | gpt-5-mini | 81.1 % | 79.0 % | 80.4 % | 87.5 % | Baseline run |
| 2026-05-17 | v2.1 (204 cases) | 355 | gpt-5-mini | 84.2 % | 79.0 % | 86.7 % | 87.5 % | +energy extreme-ratio gate (10×); +fat-tier conflict in postpass; +prompt: fat tier for all foods, fish species guidance, category noun rule |

**Test set changelog**

| Version | Cases | Changes |
|---------|-------|---------|
| v2.0 | 200 | Initial stratified set (62 merge, 98 split, 40 name\_quality) |
| v2.1 | 204 | +m061 OJ trivial form modifier; +m062 grape juice fortification; +s101 brewed vs instant coffee energy cliff; +s102 fat-free vs regular salad dressing |

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

When running the experiment loop improvements, the goal is to make the test set very representative to measure the quality we will get on a full run on the full dataset. This can be done by trying small test runs real world samples (by doing the aggreagation with LIMIT and OFFSET), and if you notice cases we should add to the test set, do it. Then, based on that, tweak the prompt and pipeline to improve accuracy on the test set. Create like this a loop of continuous improvements of both the test set and the pipeline (prompt, batch, ...) untill you are satisfied we will get good results out of a full run (which will be done on openrouter and is costly). PLEASE PLEASE PLEASE prefer a prompt based approach to hardcoding rules or postprocessing...

Scoring is computed by `scripts/score_aggregation.py` against `tests/test_set.json`.
Weights: merge 0.35 · split 0.45 · name\_quality 0.20.

| Date | Test set | Items | Model | Overall | Merge | Split | Name quality | Notes |
|------|----------|-------|-------|---------|-------|-------|--------------|-------|
| 2026-05-17 | v2.0 (200 cases) | 355 | gpt-5-mini | 81.1 % | 79.0 % | 80.4 % | 87.5 % | Baseline run |
| 2026-05-17 | v2.1 (204 cases) | 355 | gpt-5-mini | 84.2 % | 79.0 % | 86.7 % | 87.5 % | +energy extreme-ratio gate (10×); +fat-tier conflict in postpass; +prompt: fat tier for all foods, fish species guidance, category noun rule |
| 2026-05-18 | v2.1 (204 cases) | 355 | gpt-4o (0×) | 84.5 % | 79.0 % | 91.8 % | 77.5 % | +fat-tier ADD gate (source-name check); 592 s vs 933 s; NQ regresses — gpt-4o omits category prefix more often |
| 2026-05-18 | v2.2 (216 cases) | 377 | gpt-5-mini | 80.8 % | 72.0 % | 88.3 % | 77.5 % | v2.2 test set (+12 empirical cases: dry-form, cheese varieties, grain flours, brand cookies, baked-goods types); +dry-form cooking-state gate; +6 new prompt rules; merge regressed due to overly broad rules |
| 2026-05-18 | v2.2 (216 cases) | 377 | gpt-5-mini | **88.4 %** | **93.4 %** | **86.1 %** | **85.0 %** | +postpass dry-form conflict gate; softened prompt rule 10 (baked goods); fixed fat-tier gate (incoming non-regular now conflicts with regular sources); fat-tier fallback path fix; energy extreme ratio lowered to 3× |

**Test set changelog**

| Version | Cases | Changes |
|---------|-------|---------|
| v2.0 | 200 | Initial stratified set (62 merge, 98 split, 40 name\_quality) |
| v2.1 | 204 | +m061 OJ trivial form modifier; +m062 grape juice fortification; +s101 brewed vs instant coffee energy cliff; +s102 fat-free vs regular salad dressing |
| v2.2 | 216 | +s103–s112 (baked-goods types, cheese varieties, grain flours, dry-form vs cooked, brand flavors, fast-food products); +m063 bagel toasted variant; +m064 English muffin variants |

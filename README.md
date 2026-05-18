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

### Test format (v3 — active)

The test set is `tests/test_groups.txt` — a plain pipe-delimited file:

```
group_name | food_item_1 | food_item_2 | ...
```

Each group declares items that **should merge** together.  
**Confusable pairs** (group-name Jaccard ≥ 0.30) are auto-detected and tested for correct **separation**.  
One file covers both merge and split checks with no manual case IDs.

```bash
# Run aggregation, then score
python scripts/run_aggregation.py --mode test --batch-size 24
python scripts/score_groups.py                         # uses tests/output/batch_test_aggregated.json
python scripts/score_groups.py -v                      # show passing checks too
python scripts/score_groups.py --threshold 0.40        # stricter pair detection
```

Scoring: `score = 1 − (failures / total_checks)` where each check is one merge or split test.

| Date | Test set | Groups | Pairs | Checks | Model | Score | Merge | Split | Notes |
|------|----------|--------|-------|--------|-------|-------|-------|-------|-------|
| 2026-05-17 | v2.2 (216 cases) | — | — | 216 | gpt-5-mini | 88.4 % | 93.4 % | 86.1 % | Heaviest rule-gate version (cooking-state + fat-tier + dry-form gates hardcoded) |
| 2026-05-18 | v3 (41 groups) | 41 | 31 | 45 | gpt-5-mini | **88.9 %** | **78.6 %** | **93.5 %** | Simplified: removed all hardcoded cooking/fat/dry gates; prompt-only rules; new group-based test format |

**Open failures on v3 (5/45)**

| Type | Groups | Issue |
|------|--------|-------|
| merge | Wine Red Varietals | Claret / Gamay / Merlot / Syrah kept as separate groups — prompt should say "wine varietals of the same type merge" |
| merge | Tea Green | Decaf vs regular brewed green tea split — should merge (same nutritional profile) |
| merge | Beer Regular | BUDWEISER brand variant split from generic regular beer — should merge |
| split | French Dressing Regular ↔ Reduced-Fat | Merged despite fat-tier difference — prompt fat-tier rule not firing |
| split | Biscuits Dry ↔ Biscuits Baked | Dry mix merged with baked — energy gate didn't catch this pair |

**Legacy test set (v2.x)** — kept for reference

`tests/test_set.json` (216 cases: 64 merge · 112 split · 40 name\_quality) and  
`scripts/score_aggregation.py` (weighted scoring) are still runnable but no longer the primary benchmark.

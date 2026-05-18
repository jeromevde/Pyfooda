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
Scripts are run from the repo root.

**LLM backend** — set `OPENROUTER_API_KEY` to use OpenRouter.
Without a key the scripts fall back to the local [VS Code Copilot proxy](https://github.com/hyorman/copilot-proxy) at `http://127.0.0.1:3000`.

---

## 3. Experiments

**Goal:** iterate on `DEFAULT_PROMPT` in `scripts/run_aggregation.py` until the test suite passes cleanly, then do a full run on OpenRouter (costly — only when confident).

**Iteration loop:**
1. Edit the prompt in `run_aggregation.py` (`DEFAULT_PROMPT`)
2. Run the targeted test suite — each cluster is 1–3 LLM calls, results in seconds
3. Fix failures, repeat

```bash
python scripts/run_test.py                 # all 22 clusters
python scripts/run_test.py --only NAME     # rerun specific failures by name fragment
python scripts/run_test.py -v              # show PASSes + actual output group names
```

**Test set** — `scripts/test_groups.txt`, pipe-delimited:
```
group_name | food_item_1 | food_item_2 | ...
```
Each group = items that must merge. Confusable pairs (group-name Jaccard ≥ 0.30) are auto-detected and tested for correct separation. Add new groups to this file whenever a real-world slice run reveals a new failure pattern.

**Slice runs** — quick sanity check before a full run:
```bash
python scripts/run_aggregation.py --mode full --batch-size 24 --limit 1000 --offset 5000 \
    --output scripts/output/slice_5000.json
```

**Full run** — only when the test suite is green:
```bash
python scripts/run_aggregation.py --mode full --batch-size 24 --resume
# Output: pyfooda/data/foods_aggregated.json + .csv
```



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

The pipeline collapses ~296 k USDA food entries (subset of USDA database based on name length and other filters) into a compact generic food database using an LLM.
Scripts are run from the repo root.

**LLM backend** — set `OPENROUTER_API_KEY` to use OpenRouter.
Without a key the scripts fall back to the local [VS Code Copilot proxy](https://github.com/hyorman/copilot-proxy) at `http://127.0.0.1:3000`.

---

## 3. Experiments
Using python scripts/run_test.py --batch-size 20 run a loop to ruthlessly simplify the aggregation pipeline and improve the score without adding rule based pre and postprocessing but by optimizing the pipeline and the prompts
potentially we could choose to put forwards certain types of nutrients or some preprocessing when it really helps and shows useful info the model would otherwise miss but we have to build a simple elegant and robust aggregation pipeline
as we move forward we will ad more and more dificult examples to the test dataset and continue to improve
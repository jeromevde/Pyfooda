# pyfooda — Technical Summary

Dense reference covering architecture, algorithms, data flow, constants, bugs found, and design rationale.

**Research logging policy:** maintain `.github/EXPERIMENTS.md` as a living experiment diary.
Any agent making pipeline/prompt/model changes must append a structured experiment entry there.

---

## Architecture Overview

**Goal:** Collapse ~296k USDA FoodData Central entries into ~2-5k generic everyday foods (e.g. 47 branded lentil products → "Dry Lentils") with averaged nutrients.

**Pipeline:** `fooddata.csv` → sort by data-type priority → batch items → embedding search for existing matches → LLM classifies CREATE / ADD / IGNORE / RENAME → running nutrient average → checkpoint → final JSON + CSV output.

**Core engine:** `scripts/aggregator.py` (~918 lines). CLI runner: `scripts/aggregate.py`.

---

## Data Flow

```
pyfooda/data/fooddata.csv   (296k rows, built by scripts/build_fooddata.py from USDA csvs)
        │
        ▼
  Sort by data_type priority:
    foundation_food(0) > sr_legacy_food(1) = survey_fndds_food(1) > branded_food(3)
  Within same priority: sort by food_category (batches similar foods)
        │
        ▼
  Process in batches (default 150 per LLM call):
    1. For each item → FoodSearchIndex.search(name, top_k=8)
    2. Build prompt with nutrient fingerprints + ⚠️ ENERGY MISMATCH warnings
    3. Call LLM → one decision line per item
    4. Parse response → {action, idx, name/target_id}
    5. Apply: _do_create / _do_add / rename / ignore
    6. Checkpoint every 500 items
        │
        ▼
  pyfooda/data/foods_aggregated.json   (unified {meta, foods} format)
  pyfooda/data/foods_aggregated.csv    (flat, median serving size)
```

---

## Embedding Search — `FoodSearchIndex`

| Parameter | Value |
|-----------|-------|
| Model | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Dimensions | 384 |
| Index | `faiss.IndexFlatIP` (inner product = cosine on L2-normed vectors) |
| Min score threshold | 0.25 (results below this are filtered out) |
| Top-k | 8 (configurable via `--search-top-k`) |

**Mechanics:** All embeddings are L2-normalized before indexing → inner product = cosine similarity. On `add()`, the new vector is appended to a numpy matrix and faiss index is rebuilt (full rebuild; fine for <10k entries). `rename()` replaces the embedding in-place and rebuilds. `search()` returns `[{id, name, score}]`.

**Lazy loading:** The SentenceTransformer model is loaded once on first use (`_get_embedding_model()` with module-global `_embedding_model`).

---

## LLM Integration

| Parameter | Value |
|-----------|-------|
| Model | `google/gemini-2.0-flash-lite-001` (via OpenRouter) |
| API | OpenAI-compatible client (`openai.OpenAI`) |
| Base URL | `https://openrouter.ai/api/v1` |
| Temperature | 0.05 |
| Max tokens | 4096 |
| Retries | 3 (with 5s sleep between) |
| Auth | `OPENROUTER_API_KEY` env var |

**Prompt structure:** System prompt from `scripts/aggregation_prompt.txt` (102 lines, aggressive merge guidance). User message built by `_build_batch_prompt()`:
```
[0] "ORGANIC FUJI APPLE" | Fruits | kcal=52 carb=14 prot=0 fat=0 sugar=10
  Closest existing entries:
    id=12 "Apple" (count=8) [kcal=52 carb=14 prot=0 fat=0 sugar=10]
    id=45 "Apple Pie" (count=3) [kcal=237 carb=34 prot=2 fat=11 sugar=15] ⚠️ ENERGY MISMATCH
```

**Energy mismatch flag:** Added to search results when `|incoming_kcal - existing_kcal| / max(incoming, existing, 1) > 0.50` (>50% relative difference). Helps the LLM avoid merging nutritionally incompatible foods.

**Valid ADD ids:** Each food tracks `_valid_add_ids` — the set of DB ids shown in its search results. If the LLM returns an ADD with an id not in this set (common failure: confusing batch item indices with DB ids), the decision is rerouted to CREATE.

---

## LLM Response Parsing — `_parse_llm_response()`

1. **Try JSON first:** If the response is a JSON array of objects, parse directly.
2. **Line-by-line regex fallback:** Matches patterns like `[0] CREATE Apple` or `[2] ADD 12`.
3. **Supported actions:**
   - `CREATE <generic_name>` — start new group
   - `ADD <id>` or `ADD <name>` — merge into existing
   - `RENAME <id> <new_name_existing> CREATE <new_name_incoming>` — split an overbroad group
   - `IGNORE` — skip

---

## Quality Gates

### 1. Foundation Lock (`_do_add`)
Groups created by `foundation_food` data type reject all non-foundation sources. Foundation foods have lab-measured nutrients — averaging in branded/estimated values would degrade quality.

Return: `'foundation_locked'` → item is silently ignored. Stat: `stats['foundation_locked']`.

### 2. Energy Gate (`_is_energy_compatible`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `_ENERGY_MAD_THRESHOLD` | 3.5 | Multiples of MAD |
| `_ENERGY_FIXED_BAND` | 200 kcal | Floor when MAD is small |
| `_ENERGY_MIN_COUNT` | 3 | Don't gate until N sources |

**Algorithm:**
```python
arr = np.array(entry['_source_energies'])
median = np.median(arr)
mad = np.median(np.abs(arr - median))
threshold = max(3.5 * mad, 200)  # whichever is MORE lenient
return abs(new_energy - median) <= threshold
```

**Design rationale:** Pure MAD with a cliff (`if mad < 10: use fixed band, else: use MAD`) caused a discontinuity — a group with MAD=9 got a ±200 kcal band, but MAD=11 got only ±38.5 kcal. The `max()` formulation ensures the fixed band always serves as a floor, so homogeneous groups (low MAD) aren't overly restrictive. The 200 kcal band accommodates branded variants that may differ from foundation/legacy values.

**Energy rejected** items are not ignored — they're rerouted to CREATE a new group.

### 3. Embedding Dedup (CREATE path)
Before creating a new group, check if a near-identical name already exists:
```python
existing = self.index.search(name, top_k=1)
if existing and existing[0]['score'] > 0.88:
    self._do_add(existing[0]['id'], row)  # merge instead of create
```
Threshold: **0.88** cosine similarity. Catches cases where the LLM creates "Rolled Oats" when "Oats" already exists.

### 4. Intra-Batch Dedup
Within a single batch, if the LLM says CREATE for the same generic name twice, the second is automatically merged into the first via `batch_created` dict (lowercase name → db id).

### 5. Name-to-ID Exact Lookup
`_name_to_id` dict maps lowercase generic names to DB ids. Catches exact duplicates that embedding search might miss (different embedding but same string).

### 6. ADD ID Rejection
If the LLM returns `ADD <id>` where `<id>` wasn't in the search results shown to that food item, the decision is rerouted to CREATE with `stats['add_id_rejected']` incremented. Common failure mode: LLM confuses batch item indices `[0], [1], [2]` with database IDs.

---

## `_do_create` — New Group Creation

Creates a DB entry:
```python
{
    'id': int,
    'generic_name': str,
    'food_category': str,
    'nutrients': {col: float|None for col in NUTRIENT_COLS},  # 22 nutrients
    'source_ids': [int],
    'source_names': [str],
    'portion_gram_weights': [float],    # all source portion weights collected
    'portion_unit_names': [str],        # parallel array of unit names
    '_source_energies': [float],        # for energy gate (internal, stripped on output)
    '_created_by': str,                 # data_type of first source (internal)
    'count': 1,
}
```
Also adds to `FoodSearchIndex` and `_name_to_id`.

## `_do_add` — Merge Into Existing Group

Order of checks:
1. **Foundation lock** → reject if group is foundation-locked and incoming isn't foundation
2. **Energy gate** → reject if `_is_energy_compatible()` returns False
3. **Running average** → `_avg_nutrients(existing, new_row, count)` — weighted by count
4. **Collect portions** → append `portion_gram_weight` and `portion_unit_name` to arrays
5. **Track energy** → append to `_source_energies` for future gating
6. Increment `count`, append to `source_ids` and `source_names`

---

## Nutrients Tracked (22 columns, per 100g)

Energy, Carbohydrate, Protein, Total fat, Fiber, Sugars Total, Sodium, Calcium, Iron, Cholesterol, Fatty acids total saturated, Vitamin A RAE, Vitamin C, Vitamin D (D2+D3), Vitamin E, Vitamin B-6, Vitamin B-12, Folate total, Zinc, Magnesium, Potassium, Phosphorus.

**Prompt fingerprint** (shown to LLM, compact): Energy, Carbohydrate, Protein, Total fat, Sugars Total → `kcal=52 carb=14 prot=0 fat=0 sugar=10`.

---

## Unified JSON Format

Checkpoint and output use the same structure:
```json
{
  "meta": {
    "processed_count": 244,
    "next_id": 20,           // checkpoint only
    "stats": { "created": 19, "added": 109, "ignored": 116, ... }
  },
  "foods": [
    {
      "id": 1,
      "generic_name": "Nonfat Yogurt",
      "food_category": "Dairy and Egg Products",
      "nutrients": { "Energy": 56.0, "Protein": 5.7, ... },
      "source_ids": [0, 1, ...],
      "source_names": ["CHOBANI NONFAT YOGURT", ...],
      "portion_gram_weights": [156.0, 170.0, ...],
      "portion_unit_names": ["container", "container", ...],
      "count": 12
    }
  ]
}
```
**Checkpoint extras:** Entries include `_source_energies` (float[]) and `_created_by` (str) for energy gate and foundation lock on resume. Output strips all `_*` fields.

**Backward compat:** `load_checkpoint()` supports legacy format `{processed_count, next_id, stats, db: {id_str: entry}}`.

---

## Data-Type Priority & Sorting

```python
def _data_type_rank(dt):
    return {'foundation_food': 0, 'sr_legacy_food': 1, 'survey_fndds_food': 1, 'branded_food': 3}.get(dt, 5)
```

Input is sorted by `(_dtype_rank, food_category)` before processing. This ensures foundation foods create groups first (getting the foundation lock), then legacy, then branded. Within same rank, sorting by category puts similar foods in the same batch → better LLM context.

---

## Website — `docs/index.html`

**Search engine (client-side):** BM25-based with trigram tokenization + prefix matching. Also builds a reverse index on `source_names` so searching "Chobani" finds "Nonfat Yogurt". Scores are normalized, BM25 results are combined with source-name matches.

**Data loading:** Fetches `foods_aggregated.json`, supports both `{meta, foods}` wrapper and legacy flat array: `foods = Array.isArray(data) ? data : (data.foods || [])`.

**Portion selector:** Dropdown with "Per 100g" default. Builds unique portion options from `portion_gram_weights[]` / `portion_unit_names[]` arrays (deduped by gram weight, sorted ascending). Selecting a portion scales all nutrient values and %DV by `portionGrams / 100`.

**Nutrient display:** Shows macros as a bar chart (Carbs/Protein/Fat proportions), 22 nutrients with %DV where applicable, source count, and food category.

---

## Key Bugs Found & Fixed

### 1. Missing `elif` for ADD handler
**Symptom:** 116 of 244 test items silently dropped (not created, not added, not ignored).
**Root cause:** The ADD action branch was an `if` instead of `elif`, so it fell through to the IGNORE/error path. Only CREATE and IGNORE worked.
**Fix:** Changed `if decision['action'] == 'ADD':` to `elif decision['action'] == 'ADD':`.
**Impact:** Test results went from 35-38 generic groups to 19 (correct aggressive merging).

### 2. Energy gate cliff effect
**Symptom:** Groups with `MAD ≈ 10` had wildly different thresholds depending on which side of the `if mad < 10` boundary they fell on. `MAD=9` → 200 kcal band. `MAD=11` → 38.5 kcal band.
**Root cause:** Original code: `if mad < 10: threshold = _ENERGY_FIXED_BAND else: threshold = _ENERGY_MAD_THRESHOLD * mad`. Discontinuity at the boundary.
**Fix:** `threshold = max(_ENERGY_MAD_THRESHOLD * mad, _ENERGY_FIXED_BAND)` — the fixed band is always a floor.
**Impact:** Smoother energy gating, branded variants (which often have slightly different kcal) no longer rejected from legitimate groups.

---

## File Map

| File | Purpose |
|------|---------|
| `scripts/aggregator.py` | Core engine: FoodAggregator class, FoodSearchIndex, all quality gates |
| `scripts/aggregate.py` | CLI runner: arg parsing, test/full mode, resume support |
| `scripts/aggregation_prompt.txt` | System prompt for LLM (102 lines, merge rules, examples) |
| `scripts/build_fooddata.py` | Builds `fooddata.csv` from raw USDA CSV downloads |
| `scripts/build_test_dataset.py` | Creates 244-row curated test set (lentils, ham, apple pie, yogurt, lemon) |
| `scripts/clean_aggregated.py` | Post-processing cleanup |
| `scripts/nutrients_drv.py` | Daily reference values for %DV calculations |
| `pyfooda/api.py` | Python API for consuming the aggregated data |
| `pyfooda/data/fooddata.csv` | Full USDA extract (~296k rows) |
| `pyfooda/data/foods_aggregated.json` | Production output |
| `docs/index.html` | GitHub Pages website with search + portion scaling |
| `tests/test_fooddata.csv` | 244-row test dataset |
| `tests/test_aggregated.json` | Test output |

---

## CLI Quick Reference

```bash
# Test run (244 curated items)
python scripts/aggregate.py test --batch-size 30

# Full run
python scripts/aggregate.py full

# Resume from checkpoint
python scripts/aggregate.py full --resume

# Custom settings
python scripts/aggregate.py full --model google/gemini-2.0-flash-lite-001 \
  --batch-size 150 --search-top-k 8 --prompt scripts/aggregation_prompt.txt
```

---

## Tunable Constants

| Constant / Flag | Location | Default | Effect |
|-----------------|----------|---------|--------|
| `_ENERGY_MAD_THRESHOLD` | aggregator.py:193 | 3.5 | MAD multiplier for energy gate |
| `_ENERGY_FIXED_BAND` | aggregator.py:194 | 200 kcal | Floor for energy gate threshold |
| `_ENERGY_MIN_COUNT` | aggregator.py:195 | 3 | Min sources before energy gate activates |
| Embedding dedup threshold | aggregator.py (CREATE path) | 0.88 | Cosine sim above which CREATE → ADD |
| `search_top_k` | CLI `--search-top-k` | 8 | Search results shown to LLM per item |
| `batch_size` | CLI `--batch-size` | 150 | Foods per LLM call |
| `min_score` | FoodSearchIndex | 0.25 | Min cosine sim to include in results |
| Energy mismatch flag | `_build_batch_prompt` | 0.50 | Relative diff to show ⚠️ in prompt |
| Name-based ADD score | ADD resolution | 0.75 | Min cosine for name-based ADD resolution |
| Temperature | `_call_llm_batch` | 0.05 | LLM temperature |
| Checkpoint interval | `run()` | 500 items | How often checkpoint is saved |

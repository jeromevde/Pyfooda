# Food Aggregation Plan - Agentic Workflow

## Problem Statement

The current PyFooda database contains **295,943 food items** from USDA FoodData Central. This is too granular for everyday use:
- Multiple entries for essentially the same food (e.g., "ORGANIC APPLE", "APPLE, RAW", "FUJI APPLE")
- Branded variations that are nutritionally similar
- Overwhelming for users who just want "Apple" nutrition info

**Goal:** Create a lightweight, everyday-use nutrition database by intelligently merging similar items.

---

## Solution: Agentic Aggregation Workflow

### Architecture

The workflow uses an **iterative agent** that:
1. **Finds** similar food items using semantic + nutrient similarity
2. **Validates** merge candidates (optionally using LLM)
3. **Merges** items by averaging nutrients and using generic names
4. **Repeats** until no more meaningful merges are found (fixed point)

### Key Components

#### 1. **Similarity Detection (Hybrid)**
- **Semantic Similarity (70%):** Uses SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
  - Captures meaning: "ORGANIC APPLE" ≈ "APPLE, RAW"
- **Nutrient Profile Similarity (30%):** Cosine similarity on key nutrients
  - Ensures merged items are nutritionally similar
  - Prevents merging "APPLE PIE" with "APPLE"

#### 2. **LLM Validation (Optional)**
- Uses cheap LLM (default: `qwen/qwen-2.5-7b-instruct` via OpenRouter)
- Validates that merges make sense for everyday use
- Generates simple, generic names ("Apple" instead of "FUJI APPLE, ORGANIC")
- Fallback to heuristics if no API key available

#### 3. **Iterative Merging**
- Each iteration:
  1. Computes similarity matrix for current items
  2. Finds top candidates above threshold
  3. Validates and merges
  4. Updates database
- Stops when no more candidates found (convergence)

#### 4. **Nutrient Averaging**
- Merged items have averaged nutrient values
- Handles missing values gracefully
- Preserves category information

---

## Implementation Details

### Files Added

1. **`pyfooda/agentic_aggregator.py`**
   - `FoodAggregationAgent` class - core workflow
   - Embedding computation, similarity matrix, LLM validation
   - Merge tracking and history

2. **`aggregate_foods.py`**
   - CLI tool to run the workflow
   - Configurable parameters (sample size, threshold, iterations)

### Usage

#### Quick Start (No LLM)
```bash
# Aggregate 1000 items using heuristics only
python aggregate_foods.py --sample 1000
```

#### With LLM Validation
```bash
# Set API key
export OPENROUTER_API_KEY="your-key-here"

# Run with LLM validation
python aggregate_foods.py --sample 1000 --use-llm
```

#### Full Database (Warning: Slow!)
```bash
# Process all 295k items (will take hours)
python aggregate_foods.py --use-llm
```

#### Custom Parameters
```bash
python aggregate_foods.py \
  --sample 5000 \
  --threshold 0.90 \
  --max-iterations 20 \
  --use-llm
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sample` | None (all) | Number of items to sample for testing |
| `--threshold` | 0.85 | Similarity threshold (0-1, higher = stricter) |
| `--max-iterations` | 10 | Max iterations before stopping |
| `--use-llm` | False | Use LLM for validation (requires API key) |
| `--input` | `pyfooda/data/foods.csv` | Input CSV file |
| `--output` | `pyfooda/data/foods_aggregated.csv` | Output CSV file |

### Output Files

1. **`foods_aggregated.csv`** - Lightweight database with merged items
2. **`foods_aggregated_history.json`** - Full merge history for transparency

---

## Expected Results

### Test Run (1000 items, threshold 0.85)
- **Before:** 1000 food items
- **After:** ~600-700 items (30-40% reduction)
- **Time:** ~5-10 minutes (with LLM), ~1-2 minutes (without)

### Full Run (295k items, threshold 0.85)
- **Expected:** ~150k-200k items (30-50% reduction)
- **Time:** Several hours (with LLM), ~1 hour (without)

### Quality Metrics
- Similar foods merged (e.g., "ORGANIC APPLE" + "APPLE, RAW" → "Apple")
- Nutritionally distinct foods preserved (e.g., "APPLE PIE" stays separate)
- Generic names are simple and everyday-friendly

---

## Algorithm Flow

```
START
  ↓
Load food database (295k items)
  ↓
[ITERATION LOOP]
  ↓
Compute embeddings for all food names
  ↓
Compute similarity matrix (semantic + nutrient)
  ↓
Find candidate pairs (similarity > threshold)
  ↓
For each candidate pair:
  ├─ Validate with LLM (optional)
  ├─ If merge approved:
  │   ├─ Average nutrient values
  │   ├─ Update to generic name
  │   └─ Remove duplicate
  └─ Track merge in history
  ↓
Any merges performed?
  ├─ YES → Continue iteration
  └─ NO → CONVERGENCE (stop)
  ↓
Save aggregated database + history
  ↓
END
```

---

## Cost Estimate (with LLM)

Using `qwen/qwen-2.5-7b-instruct` (OpenRouter):
- **Input:** ~100 tokens per validation
- **Output:** ~50 tokens per validation
- **Cost:** ~$0.0001 per validation

**For 1000 items:**
- ~300-500 validations → ~$0.03-0.05

**For full 295k items:**
- ~50k-100k validations → ~$5-10

**Note:** Much cheaper than GPT-4! Can also run without LLM using heuristics.

---

## Next Steps

### Phase 1: Validate Approach (Now)
1. ✅ Implement agentic workflow
2. ⏳ Run on 1000-item sample
3. ⏳ Manually review results for quality
4. ⏳ Tune threshold and parameters

### Phase 2: Optimize (After validation)
1. Add category-aware merging (only merge within same category)
2. Implement confidence scores for merges
3. Add manual review interface for borderline cases
4. Optimize embedding caching for speed

### Phase 3: Production (After optimization)
1. Run on full 295k dataset
2. Publish lightweight database
3. Update PyFooda API to use aggregated database
4. Add option to toggle between full/aggregated databases

### Phase 4: Maintenance
1. Re-run aggregation when USDA updates FoodData Central
2. Allow user feedback to refine merges
3. Build community-curated merge rules

---

## Alternative Approaches Considered

### 1. **Rule-Based Clustering**
- **Pros:** Fast, deterministic
- **Cons:** Hard to capture semantic similarity, requires manual rule tuning
- **Verdict:** Too rigid for 295k diverse items

### 2. **HDBSCAN Clustering (Current `cluster.py`)**
- **Pros:** Automatic cluster discovery
- **Cons:** No control over cluster quality, opaque merges, doesn't converge to fixed point
- **Verdict:** Good for exploration, not for production database

### 3. **Pure LLM (No Embeddings)**
- **Pros:** Highest quality decisions
- **Cons:** Too expensive (50k+ API calls), too slow
- **Verdict:** Use LLM only for validation, not discovery

### 4. **Agentic Workflow (Chosen)**
- **Pros:** Balances quality + cost + speed, transparent, iterative refinement
- **Cons:** More complex implementation
- **Verdict:** Best approach for this use case

---

## Testing & Validation

### Manual Review Checklist
After running on sample, check:
- [ ] Are similar foods merged correctly? (e.g., "ORGANIC APPLE" + "APPLE, RAW")
- [ ] Are distinct foods preserved? (e.g., "APPLE" vs "APPLE PIE")
- [ ] Are generic names simple and intuitive?
- [ ] Are nutrient values reasonable (no extreme outliers)?
- [ ] Is the reduction meaningful (30-50%)?

### Automated Metrics
1. **Reduction rate:** (original - final) / original
2. **Avg similarity within merges:** Should be high (>0.85)
3. **Nutrient variance:** Should be low within merged groups
4. **Category preservation:** Merged items should share categories

---

## Dependencies

New dependencies to add to `requirements.txt`:
```
sentence-transformers>=2.0.0
openai>=1.0.0
tqdm>=4.65.0
scikit-learn>=1.0.0
```

Existing dependencies (already in project):
- pandas
- numpy

---

## FAQ

**Q: Why not just use the existing `cluster.py` approach?**  
A: HDBSCAN is great for exploration but doesn't converge to a stable database. The agentic workflow iteratively refines until no more merges make sense.

**Q: Do I need an API key?**  
A: No! The workflow works without LLM using heuristics. LLM just improves quality.

**Q: How do I know if merges are correct?**  
A: Check the merge history JSON file. It logs every merge with original names and generic name.

**Q: Can I undo merges?**  
A: Yes, the history file lets you trace back to original items. You can also re-run with different parameters.

**Q: What if I want more/less aggregation?**  
A: Adjust `--threshold`:
- Higher (0.90+) → fewer merges, more items preserved
- Lower (0.75-0.80) → more merges, smaller database

**Q: How long does it take?**  
A: 
- 1000 items: ~5 min (with LLM), ~1 min (without)
- 10k items: ~30 min (with LLM), ~5 min (without)
- 295k items: ~4-6 hours (with LLM), ~30-60 min (without)

---

## References

- [USDA FoodData Central](https://fdc.nal.usda.gov/)
- [SentenceTransformers](https://www.sbert.net/)
- [HDBSCAN](https://hdbscan.readthedocs.io/)
- [OpenRouter](https://openrouter.ai/)

---

**Status:** ✅ Implementation complete, ready for testing!

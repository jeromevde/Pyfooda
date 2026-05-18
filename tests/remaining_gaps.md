# Remaining Aggregation Score Gaps

**Baseline (run 1)**: 82.1% overall (35 failures) — purpose-built test CSV (547 rows), full LLM pass.  
**Run 2**: 81.1% overall (37 failures, 3 skipped) — reconstructed test CSV (546 rows), partial LLM pass (384/546 foods processed; 162 items ignored due to proxy dropout on batches 17-23).  
Weights: merge=35%, split=45%, name_quality=20%.

---

## Split failures (14 IMPORTANT, ~7.1% of score)

Two foods wrongly share the same group. Two sub-types:

### A. Semantically different food, near-identical nutrients
Nutrients are too similar to trigger the gate (threshold: kcal >40%, fat >60%), so only the *name* signals they're different. The LLM collapses them because shared tokens dominate.

| Foods | kcal diff | fat diff |
|---|---|---|
| Cheese, cheddar, nonfat vs Cheese, mozzarella, nonfat | 11% | 0% |
| Turkey, drumstick, roasted vs Turkey, thigh, cooked | 12% | 8% |
| Fish, salmon, baked/broiled vs Fish, salmon, fried | 12% | 11% |
| Green peas, canned vs Peas and corn, cooked | 16% | 41% |
| Beans, from canned vs Beans, from dried | ~low | ~low |
| Fish, white, baked/broiled vs Fish, white, grilled | ~low | ~low |
| Turkey, dark meat, raw vs Turkey, light meat, raw | 6% | 41% |
| Turkey, whole, meat and skin, raw vs Turkey, light meat, raw | ~low | ~low |

**Fix direction**: Add name-level conflict tokens — e.g. body-part pairs (drumstick/thigh), cheese type (cheddar/mozzarella), cooking method distinctions (fried ≠ baked), and canned/dried distinctions (already in PREP_CONFLICT but not triggering).

### B. Fat-level marketing tier conflation
"Light" ≠ "reduced fat" ≠ "fat-free" are treated as equivalent by the LLM.

| Foods | kcal diff | fat diff |
|---|---|---|
| Potato chips, lightly salted vs reduced fat | 19% | 41% |
| Sour cream, light vs reduced fat | 33% | 25% |
| Ice creams, vanilla, light vs BREYERS 98% Fat Free | 32% | 54% |
| Tortilla chips, low fat, unsalted vs reduced fat, flavored | ~low | ~low |
| Almonds, dry roasted salted (specific lot) vs general | ~low | ~low |

**Fix direction**: Lower fat gate threshold to ~30% (would catch potato chips at 41%, ice cream at 54%). Add fat-tier conflict tokens: `{'light', 'reduced fat', 'low fat', 'fat free', 'nonfat'}` as a conflict set (none of these should merge with each other or with regular).

---

## Merge failures (13 NORMAL, ~5.9% of score)

Foods that *should* share a group are in separate ones.

### A. Fish species fragmentation (10/13)
Test expects taxonomically/nutritionally similar white fish and raw fish species to be grouped (e.g., burbot + cusk + ling + cod → "White fish, cooked dry heat"). They have near-identical nutrients but different species names → near-zero token jaccard → similarity postpass skips them (food_jaccard gate ≥0.2 blocks).

| Split groups | Expected merge |
|---|---|
| burbot, cusk / ling / lingcod / mahi-mahi | White fish cooked dry heat |
| bluefish+carp / scup+sheepshead+spot+swordfish | Cooked fish |
| butterfish / milkfish | Cooked fish |
| haddock / monkfish | Cooked white fish |
| grouper / perch / seatrout+sturgeon / snapper (raw) | Raw fish |
| cod+pollock / wolffish | Cooked white fish |
| pike northern / pike walleye | Pike cooked |
| catfish farm-raised / tilapia farm-raised | Farm-raised fish raw |

**Fix direction**: Relax or remove the food_jaccard gate specifically for the "Finfish and Shellfish Products" category. Fish species names share almost no tokens by design, but nutrient similarity is the right merge signal there.

### B. Beverage/product variants (3/13)

| Split groups | Expected merge |
|---|---|
| Budweiser (brand) / Beer, regular, all | Regular beer |
| BUD LIGHT (brand) / Beer, light | Light beer |
| Soy protein powder / Whey protein powder | Protein powder |

**Fix direction**: Soy vs whey protein powder is genuinely debatable (different food). For beer, brand-specific entries could be merged with generics if brand tokens are identified as non-discriminative.

---

## Name quality failures (8 NORMAL, ~2.4% of score)

| Case | Group name assigned | Required keyword | Fix direction |
|---|---|---|---|
| n004 | Cranberry Cherry Juice | 'Water' | Prompt: always include category noun |
| n015 | Reduced Fat Monterey Jack | 'Cheese' | Prompt: include food class |
| n017 | '93' (not title case) | title case | Sanitizer: force title case |
| n020 | Frozen Collard Greens Cooked | 'Collards' | Prompt: use common name |
| n022 | Flavored Non Dairy Latte | 'Coffee' | Prompt: include base ingredient |
| n033 | Whole Grain White Bun | 'Roll' | Prompt: include product type |
| n037 | Scup Cooked | 'Mollusks' | Wrong category (scup is finfish) — test case quirk |
| n040 | Ground Turmeric | 'Spices' | Prompt: spice names should start with "Spices," |

**Fix direction**: Add to system prompt: *"Group names must include the food's primary category noun (e.g., 'Cheese', 'Juice', 'Coffee', 'Roll', 'Spices'). Names must be title case including numbers."*

---

## s064 — CHOBANI not found (skip/not-scored)
`Yogurt, Greek, nonfat, vanilla, CHOBANI` was not present in `test_fooddata.csv` at scoring time. Verify the food is in the test CSV.

---

## Summary of fix priority

| Fix | Failures addressed | Estimated gain |
|---|---|---|
| Lower fat gate to 30% + fat-tier conflict tokens | s021, s051, s058, s069, s071 | ~2–3% |
| Body-part / cheese-type / prep-method name conflicts | s022, s026, s032, s038, s043, s053, s061 | ~2–3% |
| Relax fish food_jaccard gate in Finfish category | m030, m032, m034, m035, m037, m040, m042, m043 | ~3–4% |
| Name quality prompt improvements | n004, n015, n017, n020, n022, n033, n040 | ~1–2% |

---

## Empirical gaps confirmed in Run 2 (not previously documented)

### New split patterns

**C. Dry/powder-format beverage vs brewed liquid**  
`Coffee, brewed` (1 kcal) grouped with `Beverages, coffee, instant, regular, powder` (353 kcal) → 353x energy cliff. The energy gate should catch this, but shared tokens ("coffee") dominate the similarity signal when the gate is applied at group-creation time rather than at merge time. This pattern is a beverage-domain analog of s093/s098 (soup broth vs dry soup mix).  
→ **New test case added**: `s101`  
→ **Fix direction**: Extend the "dry mix / powder" token blocklist to include beverages, not just soups.

**D. Fat-free vs regular of the same exact dressing**  
`Blue or roquefort cheese dressing` (484 kcal) vs `Blue or roquefort cheese dressing, fat free` (115 kcal). This is a 4x energy ratio on an identical base food. Current fat-level conflict detection fires when comparing two *different* foods with fat-tier tokens, but may miss same-food variants if tokens partially overlap and calorie ratio falls below the soft gate.  
→ **New test case added**: `s102`  
→ **Fix direction**: Lower fat gate threshold to ~30% (4x ratio = 75% reduction) or add `fat free` as an explicit split token against any non-fat-free variant of the same named food.

### New merge patterns

**E. Trivial format modifier prevents merge (juice)**  
`Orange juice, 100%, NFS` vs `Orange juice, canned, unsweetened` (47 vs 47 kcal — identical energy) end up in separate groups because "canned" triggers a cooking-state-like discriminator.  
→ **New test case added**: `m061`  
→ **Fix direction**: Treat "canned" as non-discriminative for juices; the food_jaccard similarity should be sufficient.

**F. Added-nutrient fortification variant prevents merge**  
`Grape juice, 100%` vs `Grape juice, 100%, with calcium added` (66 vs 62 kcal — ~6% diff). The "with calcium added" clause acts as a discriminating token even though it represents a minor formulation difference.  
→ **New test case added**: `m062`  
→ **Fix direction**: Add `with [nutrient] added` as a non-discriminative suffix pattern (i.e., strip it from token comparison).

### Confirmed name quality patterns (newly observed in Run 2)

| Case | Group name assigned | Issue | Notes |
|---|---|---|---|
| n013 | Chicken Broth Low Sodium | missing 'Soup' | category='Soups, Sauces, and Gravies' — category noun dropped entirely |
| n024 | Ribeye Steak | missing 'Beef' | category='Beef, excludes ground' — protein class noun dropped |
| n037 | Cooked Blue Mussel | missing 'Mollusks' | category='Finfish and Shellfish Products' — phylum noun dropped |

All three are instances of the same systemic problem: the LLM omits the food-class category noun when it constructs the group name, using only the specific food descriptor. The prompt currently says "concise", which the model interprets as removing the category prefix.  
→ **Fix direction**: Add an explicit instruction: *"Always include the primary category noun in the group name (e.g., 'Beef Ribeye Steak', 'Chicken Soup Low Sodium', 'Mollusks Blue Mussel Cooked')."*

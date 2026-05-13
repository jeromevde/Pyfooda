# Remaining Aggregation Score Gaps

**Baseline**: 82.1% overall (35 failures) after run with source-name-aware conflict detection.
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

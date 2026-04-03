# EXPERIMENTS.md — Ongoing Experiment Trace

Purpose: keep a durable, chronological research log so future agents and humans can understand what was tried, what worked, and what failed.

## Logging rules (mandatory)

For every meaningful experiment, append one new entry with:

1. **Date/time**
2. **Goal / hypothesis**
3. **Code / prompt version** (commit hash, key config)
4. **Dataset slice** (e.g., test 244, offset 5000 + first 1000)
5. **Runtime and API usage**
6. **Cost estimate assumptions**
7. **Quality observations** (good merges, bad merges, naming issues)
8. **Artifacts produced** (`.json`, `.csv`, `.metrics.json`, `.groups.txt`)
9. **Decision** (keep / reject / follow-up)

## Evaluation checklist

- Parse error rate
- Group-name quality (short, generic, non-brand)
- Nutrition-aware separation (e.g., pizza subtypes, dry vs cooked)
- Duplicate/near-duplicate group leakage
- Compression behavior (observed, not hardcoded target)

## Current best-known baseline (update when superseded)

- Pipeline: `scripts/run_aggregation.py`
- Model: `google/gemini-2.0-flash-lite-001`
- Decision mode: `line` (JSON-first parsing)
- Representative slice: offset 5000, limit 1000
- Main review artifact: `tests/exp_latest_1000_off5000_b16.groups.txt`

---

## Experiment log

### 2026-03-29 — Standalone pipeline split + model probes (30-item slice)
- Goal: compare cheap/fast/reliable options across providers.
- Config: separate runners (batching, Together streaming, Ollama streaming).
- Runtime/API snapshot:
  - OpenRouter batching (Gemini flash-lite, b=8): ~5–11s on 30 items, 4 calls.
  - Together streaming (Qwen2.5-7B): ~19–26s on 30 items, 30 calls.
  - Ollama local streaming (qwen2.5:3b): ~101–109s on 30 items, 30 calls.
- Cost assumptions:
  - cloud runs used per-call estimate in metrics; local Ollama API cost = 0.
- Quality:
  - batching produced cleaner names and fewer malformed merges.
  - together usable but more aggressive merges.
  - local 3b slower and noisier naming.
- Decision: prefer batched OpenRouter baseline.

### 2026-04-02 — Repository simplification to single pipeline
- Goal: remove non-working streaming paths from active pipeline.
- Config: removed `run_together_pipeline.py`, `run_ollama_pipeline.py`, old `aggregate.py`; kept `run_aggregation.py`.
- Quality/cost context moved to README as historical trace.
- Decision: keep one production path; preserve historical results only for reference.

### 2026-04-02 — Batch-size sweep on curated test set (244)
- Goal: find best quality/speed/cost batch size.
- Config: OpenRouter Gemini flash-lite, decision mode line.
- Results:
  - b=8: 138.78s, 31 calls, 18 groups, 23 parse errors.
  - b=16: 107.56s, 16 calls, 17 groups, 8 parse errors.
  - b=24: 85.73s, 11 calls, 19 groups, 26 parse errors.
- Decision: b=16 selected (quality/speed compromise).
- Artifacts: `tests/exp_test244_b*.{json,csv,metrics.json}` (later cleaned during artifact pruning).

### 2026-04-02 — Real-data 1000-row run (offset 0)
- Goal: smoke test on realistic volume.
- Config: b=16, OpenRouter Gemini flash-lite.
- Result: 1000 processed, 513 groups, 4 parse errors, 244.79s.
- Estimation: full-runtime ~20h, full-cost ~$9.32 (per-call estimate based scaling).
- Decision: workable but not representative enough due to ordering concentration.

### 2026-04-02 / 2026-04-03 — Offset robustness runs (offset 5000)
- Goal: avoid first-block bias and test harder mixed categories.
- Config: b=16, offset=5000, limit=1000.
- Observed variance across reruns due to prompt/parser revisions:
  - run A: 407 groups, 39 parse errors, 139s.
  - run B: 550 groups, 12 parse errors, 261s.
  - run C: 565 groups, 51 parse errors, 271s.
- Quality: label readability mostly acceptable; parse stability and over-fragmentation remained key risks.
- Decision: make parser stricter (JSON-first) and improve naming constraints.

### 2026-04-03 — Prompt hardening iterations
- Goal: improve generic naming quality.
- Changes:
  - removed canonical hardcoded target list (anti-cheat request).
  - added nutrition-first merge rule.
  - enforced pizza subtype separation (no generic Pizza bucket).
  - added anti-brand naming guidance (e.g., prefer Fat Free Salad Dressing labels).
  - added short-name target (2–4 words, hard max 6).
  - added trivial variant normalization guidance (Non Dairy vs Non-Dairy).
- Decision: keep prompt generic + nutrition-aware without fixed canonical answer key.

### 2026-04-03 — Human review artifact addition
- Goal: faster quality inspection.
- Change: pipeline now writes `*.groups.txt` in format `Group: item1, item2, ...`.
- Decision: keep as mandatory artifact for manual review.

### 2026-04-03 — JSON-first parser hardening rerun
- Goal: reduce parse errors while keeping speed/cost acceptable.
- Config: batch=16, offset=5000, limit=1000, OpenRouter Gemini flash-lite, decision mode line.
- Result: parse errors dropped to ~10 on rerun with JSON-first path; groups.txt used for naming review.
- Artifacts:
  - `tests/exp_latest_1000_off5000_b16.json`
  - `tests/exp_latest_1000_off5000_b16.csv`
  - `tests/exp_latest_1000_off5000_b16.metrics.json`
  - `tests/exp_latest_1000_off5000_b16.groups.txt`
- Decision: keep JSON-first parsing baseline.

### 2026-04-03 — Group-only decision mode trial (rejected)
- Goal: test simplified model output (`GROUP <name>|IGNORE`) and parser-managed create/add.
- Config: `decision_mode=group` on offset-5000/1000.
- Result: model output didn’t follow schema reliably; effectively 100% parse failures in first trial.
- Decision: reject for now; keep line mode with strict JSON schema prompts.

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

### 2026-04-03 — JSON-first parser hardening rerun
- Goal: reduce parse errors while keeping speed/cost acceptable.
- Config: batch=16, offset=5000, limit=1000, OpenRouter Gemini flash-lite.
- Result: parse errors reduced materially versus previous noisy run; quality review via groups.txt.
- Artifacts:
  - `tests/exp_latest_1000_off5000_b16.json`
  - `tests/exp_latest_1000_off5000_b16.csv`
  - `tests/exp_latest_1000_off5000_b16.metrics.json`
  - `tests/exp_latest_1000_off5000_b16.groups.txt`
- Decision: keep JSON-first parsing as baseline.

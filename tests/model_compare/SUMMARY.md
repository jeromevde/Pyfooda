# Model comparison on problematic offsets (prompt-improved, no new hard sanitizer)

Offsets tested:
- 2000 (mixed noisy slice)
- 7000 (known bean/moisture label failure slice)

Settings:
- limit=1000 per offset
- batch_size=16
- decision_mode=line
- timeout=60s

## Key metrics

| Model | Offset | Final groups | Parse errors | Added | Ignored | Bad-name hits* | Long names (>6 words) | Elapsed (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| google/gemini-2.0-flash-lite-001 | 2000 | 639 | 35 | 321 | 37 | 2 | 3 | 278.93 |
| qwen/qwen-2.5-7b-instruct | 2000 | 423 | 2 | 509 | 57 | 2 | 2 | 290.50 |
| openai/gpt-4o-mini | 2000 | 82 | 0 | 882 | 0 | 0 | 2 | 149.80 |
| google/gemini-2.0-flash-lite-001 | 7000 | 343 | 3 | 635 | 16 | 26 | 14 | 231.03 |
| qwen/qwen-2.5-7b-instruct | 7000 | 226 | 21 | 674 | 97 | 1 | 0 | 334.95 |
| openai/gpt-4o-mini | 7000 | 96 | 0 | 870 | 16 | 1 | 2 | 129.81 |

\* bad-name hits = heuristic count of generic names containing noisy patterns (e.g., moisture/code tails).

## Readout

- **Gemini flash-lite**: best continuity with existing behavior, but still leaks noisy names on offset 7000 and shows high parse errors on offset 2000.
- **Qwen 2.5 7B**: reduces noisy names on offset 7000 but has high parse error/ignore behavior there.
- **GPT-4o-mini**: cleanest naming and zero parse errors in these tests, but likely over-merges aggressively (very low final group counts), so quality must be manually audited before switching.

## Practical recommendation before full run

1. Keep current model as default unless over-merge audit passes for gpt-4o-mini.
2. Use **model fallback policy** for hard slices:
   - first pass: gemini-2.0-flash-lite
   - retry hard chunks with gpt-4o-mini (only when parse/noisy-name thresholds are exceeded)
3. Continue prompt iteration (already improved) and re-test only hard offsets (7000-like) before full run.
4. `.groups.txt` now uses `|` separators for readability.

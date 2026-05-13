#!/usr/bin/env python3
"""
Autosearch: greedy hill-climb over prompt and batch-instruction variants.

Strategy
--------
1. Score the baseline (DEFAULT_PROMPT + default batch instruction).
2. For each round, try every remaining patch on top of the current best config.
3. Keep the patch that most improves the overall score (if any).
4. Repeat until no improvement or --rounds exhausted.
5. Write the winning prompt to scripts/aggregation_prompt.txt.

Usage
-----
  cd scripts
  python autosearch.py
  python autosearch.py --rounds 3 --batch-size 24 --model gpt-4o-mini
  python autosearch.py --rounds 3 --batch-size 24 --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from aggregator import FoodAggregator, _resolve_llm_backend
from run_aggregation import (
    DEFAULT_PROMPT,
    _DEFAULT_BATCH_INSTRUCTION,
    run_batched,
)
from score_aggregation import score as compute_score

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
SCRIPTS_DIR = REPO_ROOT / "scripts"
BEST_PROMPT_PATH = SCRIPTS_DIR / "aggregation_prompt.txt"

# ---------------------------------------------------------------------------
# Patch library
#
# Each entry: (patch_id, description, prompt_old, prompt_new, batch_inst_new)
#   - prompt_old=None → append prompt_new at end of system prompt
#   - prompt_old not found in current prompt → skip (not applicable)
#   - batch_inst_new=None → keep current batch instruction unchanged
# ---------------------------------------------------------------------------

_PATCHES: list[tuple[str, str, str | None, str | None, str | None]] = [
    # --- System-prompt patches ---
    (
        "cooking_state_examples",
        "Add kcal ratio example for dry vs cooked split",
        "Dry vs cooked legumes",
        "Dry vs cooked legumes (raw lentils ~350 kcal/100g vs cooked ~115 kcal — always separate groups; caloric ratio ≥2:1 is a hard signal)",
        None,
    ),
    (
        "fat_level_kcal",
        "Add numeric kcal threshold for dairy fat-level splits",
        "Fat-level differences for yogurt (whole/low-fat/nonfat/greek)",
        "Fat-level differences for dairy: whole-milk yogurt (~60 kcal/100g), low-fat (~40 kcal), nonfat (~30 kcal) — always separate groups when kcal differs by >20 kcal/100g",
        None,
    ),
    (
        "composite_examples",
        "Add examples for composite dish vs ingredient split",
        "Composite dish vs plain ingredient (e.g., Ham Sub != Ham)",
        "Composite dish vs plain ingredient: 'Chicken Soup' ≠ 'Chicken Breast'; 'Ham Sandwich' ≠ 'Ham'; 'Beef Stew' ≠ 'Beef Chuck'. Never merge a recipe/dish with its primary raw ingredient.",
        None,
    ),
    (
        "energy_gate_explicit",
        "Explicit kcal-ratio ADD gate",
        "Use the nutrient fingerprint in the prompt as a primary signal; avoid merging items with clearly different calories/macros.",
        "Use the nutrient fingerprint as the primary merge signal. Hard rule: if |kcal_incoming − kcal_candidate| / min(kcal_incoming, kcal_candidate) > 0.40, do NOT ADD — CREATE a separate group instead.",
        None,
    ),
    (
        "shorter_names",
        "Tighten name length to max 4 words",
        "Target name length: 2-4 words (hard max 6 words).",
        "Target name length: 2-3 words (hard max 4 words). Drop filler adjectives. Bad: 'Grilled Chicken Breast Piece Cooked'. Good: 'Grilled Chicken Breast'.",
        None,
    ),
    (
        "brand_merge_aggressive",
        "More aggressive brand variant merging when macros match",
        "Brand variants should merge when nutrients/use are similar.",
        "Brand variants ALWAYS merge when kcal/macros are within 15%. Store-brand vs name-brand of the same food = same group. Different flavors of the same base product (strawberry vs blueberry yogurt of same fat level) = same group.",
        None,
    ),
    (
        "relax_ignore",
        "Prefer grouping over ignoring for edge cases",
        "IGNORE only if clearly non-food/unclear/supplement/baby/pet.",
        "IGNORE only if genuinely non-food (pet food, cleaning supplies) or completely unrecognizable noise. Supplements, baby food, ethnic specialties — give them a generic group name. When uncertain, CREATE rather than IGNORE.",
        None,
    ),
    # --- Batch-instruction patches ---
    (
        "batch_inst_verbose_sequential",
        "More explicit within-batch sequential consistency instruction",
        None,
        None,
        (
            "\n\nReturn exactly one decision per item as strict JSON object lines."
            "\nAllowed JSON schemas:"
            '\n{"idx": <idx>, "action": "CREATE", "name": "<generic_name>"}'
            '\n{"idx": <idx>, "action": "ADD", "target_id": <id>}'
            '\n{"idx": <idx>, "action": "IGNORE"}'
            "\nWithin-batch sequential rule: process items in index order. Track every CREATE name you output."
            " If a later item belongs to a group you already named in this batch, output CREATE with the EXACT same"
            " name string (same words, same capitalization, same order) — NOT a synonym or reordering."
            " They will be automatically merged. Only ADD to ids shown in each item's candidate list."
        ),
    ),
    (
        "batch_inst_compact",
        "Compact batch instruction — less token overhead",
        None,
        None,
        (
            "\n\nOne JSON line per item. Schema:"
            '\n{"idx":N,"action":"CREATE","name":"<name>"}  or'
            '\n{"idx":N,"action":"ADD","target_id":<id>}  or'
            '\n{"idx":N,"action":"IGNORE"}'
            "\nFor items in the same group: identical CREATE name (exact string). ADD only to ids in that item's candidate list."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Run + score helpers
# ---------------------------------------------------------------------------

def _run_and_score(
    prompt: str,
    *,
    batch_size: int,
    model: str,
    api_key: str,
    base_url: str,
    batch_instruction: str | None,
    run_id: str,
) -> tuple[float | None, dict]:
    """Run the full test pipeline with the given prompt and score it."""
    input_path = TESTS_DIR / "test_fooddata.csv"
    output_path = TESTS_DIR / f"autosearch_{run_id}.json"
    ckpt_dir = TESTS_DIR / f"autosearch_ckpt_{run_id}"
    test_set_path = TESTS_DIR / "test_set.json"

    # Clean previous checkpoint for this run slot
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_path = f.name

    try:
        agg = FoodAggregator(
            df,
            model=model,
            api_key=api_key,
            base_url=base_url,
            batch_size=batch_size,
            search_top_k=8,
            prompt_path=prompt_path,
            checkpoint_dir=str(ckpt_dir),
        )
        run_batched(
            agg,
            limit=None,
            offset=0,
            resume_from=0,
            batch_size=batch_size,
            batch_instruction=batch_instruction,
        )
        agg.save(str(output_path))
    finally:
        os.unlink(prompt_path)

    if not output_path.exists():
        return None, {}

    report = compute_score(output_path, test_set_path)
    overall = report["scores"]["overall"]
    return overall, report


def _fmt(score: float | None) -> str:
    if score is None:
        return "N/A"
    label = "GOOD" if score >= 90 else "OK" if score >= 70 else "POOR"
    return f"{score:.1f}% [{label}]"


def _component_scores(report: dict) -> str:
    s = report.get("scores", {})
    m = s.get("merge", {}).get("score")
    sp = s.get("split", {}).get("score")
    n = s.get("name_quality", {}).get("score")
    return f"merge={m:.0f}% split={sp:.0f}% name={n:.0f}%" if all(
        v is not None for v in [m, sp, n]
    ) else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Autosearch: hill-climb over prompt variants")
    parser.add_argument("--rounds", type=int, default=2, help="max hill-climb rounds (default: 2)")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--model", default=None, help="override model name (e.g. gpt-4o-mini, gpt-5-mini)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output-prompt", default=str(BEST_PROMPT_PATH))
    args = parser.parse_args()

    api_key, base_url, model = _resolve_llm_backend(
        args.api_key, args.base_url, args.model or "gpt-4o-mini"
    )

    print(f"\nAutosearch")
    print(f"  model      : {model}")
    print(f"  backend    : {base_url}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  rounds     : {args.rounds}")
    print(f"  patches    : {len(_PATCHES)}")

    run_kwargs = dict(
        batch_size=args.batch_size,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    # ---- Baseline ----
    print("\n[baseline] running ...")
    t0 = time.time()
    best_score, best_report = _run_and_score(
        DEFAULT_PROMPT,
        **run_kwargs,
        batch_instruction=None,
        run_id="baseline",
    )
    print(f"[baseline] {_fmt(best_score)}  {_component_scores(best_report)}  ({time.time()-t0:.0f}s)")

    best_prompt: str = DEFAULT_PROMPT
    best_batch_inst: str | None = None
    applied: list[str] = []

    history: list[dict] = [{"tag": "baseline", "score": best_score, "applied": []}]

    remaining = list(_PATCHES)

    for round_idx in range(args.rounds):
        print(f"\n---- Round {round_idx + 1}/{args.rounds} (best so far: {_fmt(best_score)}) ----")

        round_best_score = best_score
        round_best_prompt = best_prompt
        round_best_inst = best_batch_inst
        round_best_tag: str | None = None

        still_remaining = []
        for patch in remaining:
            patch_id, desc, p_old, p_new, b_new = patch

            # Build candidate prompt
            if p_old is not None:
                if p_old not in best_prompt:
                    still_remaining.append(patch)
                    continue  # patch text not found — skip
                candidate_prompt = best_prompt.replace(p_old, p_new, 1)
            elif p_new is not None:
                candidate_prompt = best_prompt.rstrip() + "\n" + p_new
            else:
                candidate_prompt = best_prompt

            # Build candidate batch instruction
            candidate_inst = b_new if b_new is not None else best_batch_inst

            print(f"  [{patch_id}] {desc} ...", end="", flush=True)
            t0 = time.time()
            s, rep = _run_and_score(
                candidate_prompt,
                **run_kwargs,
                batch_instruction=candidate_inst,
                run_id=patch_id,
            )
            elapsed = time.time() - t0
            delta = f"{s - best_score:+.1f}%" if (s is not None and best_score is not None) else ""
            print(f"  {_fmt(s)}  {delta}  {_component_scores(rep)}  ({elapsed:.0f}s)")

            if s is not None and (round_best_score is None or s > round_best_score):
                round_best_score = s
                round_best_prompt = candidate_prompt
                round_best_inst = candidate_inst
                round_best_tag = patch_id

            still_remaining.append(patch)

        remaining = [p for p in still_remaining if p[0] != round_best_tag]

        if round_best_tag is not None and (best_score is None or round_best_score > best_score):
            delta = round_best_score - (best_score or 0)
            print(f"\n  ✓ Kept '{round_best_tag}'  {_fmt(best_score)} → {_fmt(round_best_score)}  (+{delta:.1f}%)")
            best_score = round_best_score
            best_prompt = round_best_prompt
            best_batch_inst = round_best_inst
            applied.append(round_best_tag)
            history.append({"tag": round_best_tag, "score": best_score, "applied": list(applied)})
        else:
            print(f"\n  No improvement this round. Stopping early.")
            break

    # ---- Save results ----
    out_path = Path(args.output_prompt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(best_prompt)

    # If a custom batch instruction was found, append it as a comment block
    if best_batch_inst is not None and best_batch_inst != _DEFAULT_BATCH_INSTRUCTION:
        inst_path = SCRIPTS_DIR / "autosearch_best_batch_instruction.txt"
        inst_path.write_text(best_batch_inst)
        print(f"\nCustom batch instruction → {inst_path}")

    hist_path = TESTS_DIR / "autosearch_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest prompt saved → {out_path}")
    print(f"Final score       : {_fmt(best_score)}")
    print(f"Applied patches   : {applied or ['none (baseline was best)']}")
    print(f"History           → {hist_path}")


if __name__ == "__main__":
    main()

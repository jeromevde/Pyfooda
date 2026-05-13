#!/usr/bin/env python3
"""
Standalone aggregation runner (batched multi-item LLM calls).

Example:
  python scripts/run_aggregation.py --mode test --batch-size 8
  python scripts/run_aggregation.py --mode full --batch-size 8 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from aggregator import (
    FoodAggregator,
    _build_item_prompt,
    _call_llm_batch,
    _data_type_rank,
    _nutrient_fingerprint,
    _parse_llm_decision,
    _resolve_llm_backend,
)

DEFAULT_PROMPT = """You are curating a compact food database for everyday tracking.

You receive ONE incoming food item and nearest existing entries.
Return EXACTLY one line, and nothing else:

[<idx>] CREATE <generic_name>
[<idx>] ADD <id>
[<idx>] RENAME <id> <new_name_for_existing> CREATE <new_name_for_incoming>
[<idx>] IGNORE

Hard constraints:
- Prefer ADD over CREATE only when the candidate group is the same generic food. When in doubt, CREATE.
- ADD id must be one of the shown candidate ids.
- Use the nutrient fingerprint in the prompt as a primary signal; avoid merging items with clearly different calories/macros.
- Generic names must be short, generic, title case, no brands, no commas.
- Strip brand/proprietary style tokens from group names (e.g., prefer \"Fat Free Salad Dressing\" over \"Thousand Island Dressing Fat Free\").
- Target name length: 2-4 words (hard max 6 words).
- Never output raw USDA names as CREATE names.
- Never include lab/sample/spec tokens in CREATE names: no lot IDs, no assay strings, no moisture percentages, no numeric code tails (examples to avoid: "11F 8119 0 Moisture", "NFS", "NS as to").
- If the incoming name is noisy/encoded, map it to the nearest plain-language food name.
- IGNORE only if clearly non-food/unclear/supplement/baby/pet.

Merge policy (aggressive but nutrition-aware):
- Brand variants should merge when nutrients/use are similar.
- Flavored fruit yogurts may merge into one flavored-yogurt bucket (apple/cherry/strawberry can merge).
- Keep distinctions when nutrition changes materially:
  1) Dry vs cooked legumes
  2) Fat-level differences for yogurt (whole/low-fat/nonfat/greek)
  3) Composite dish vs plain ingredient (e.g., Ham Sub != Ham)
  4) Lemon juice vs lemonade/soda drinks
  5) Pizza types MUST stay separate by category/style (e.g., cheese pizza, meat pizza, veggie pizza, deep-dish/frozen etc.) when nutrients differ noticeably
- Do NOT collapse all pizzas into a generic \"Pizza\" group.
- Do NOT create over-specific single-SKU style names; choose a broader nutrition-relevant generic bucket.
- Normalize trivial wording variants into one canonical name (hyphenation/singular/plural/word order), e.g. \"Non Dairy\" and \"Non-Dairy\" must be the same group.- IMPORTANT: different specific foods must NOT share a group. Corn ≠ collard greens. Chestnuts ≠ hummus. Salmon ≠ cod. Asparagus ≠ peas. Each distinct food type needs its own group.
Return one decision line only.
"""


def _normalized_name_key(name: str) -> str:
    import re
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_generic_name(name: str) -> str:
    import re
    s = re.sub(r"\s+", " ", str(name)).strip()
    # hard cap to reduce overly specific labels
    words = s.split()
    if len(words) > 6:
        s = " ".join(words[:6])

    # normalize dressing labels to generic class when needed
    low = s.lower()
    if "dressing" in low and "salad dressing" not in low:
        qualifier = ""
        for q in ["fat free", "low fat", "reduced fat", "light"]:
            if q in low:
                qualifier = q.title() + " "
                break
        s = f"{qualifier}Salad Dressing".strip()

    return s


def _find_existing_id_by_name(agg: FoodAggregator, candidate: str):
    key = _normalized_name_key(candidate)
    # fast path: exact mapping
    if key in agg._name_to_id:
        return agg._name_to_id[key]
    # fallback scan to catch legacy keys
    for k, v in agg._name_to_id.items():
        if _normalized_name_key(k) == key:
            return v
    return None


def _resolve_input_output(repo_root: Path, mode: str, output: str | None, checkpoint_dir: str | None):
    default_input = repo_root / "pyfooda/data/fooddata.csv"
    default_output = repo_root / "tests/batch_test_aggregated.json" if mode == "test" else repo_root / "pyfooda/data/foods_aggregated_batch.json"
    default_ckpt = repo_root / ("tests/checkpoints_batch" if mode == "test" else "checkpoints_batch")

    input_path = default_input
    if mode == "test":
        test_csv = repo_root / "tests/test_fooddata.csv"
        if test_csv.exists():
            input_path = test_csv

    return input_path, Path(output) if output else default_output, Path(checkpoint_dir) if checkpoint_dir else default_ckpt


def _apply_decision(agg: FoodAggregator, food: dict, row: pd.Series, decision: dict, create_aliases: dict[str, str] | None = None):
    name = food["name"]
    if decision["action"] == "CREATE":
        create_name = _sanitize_generic_name(decision.get("name", name))
        if create_aliases is not None:
            key = _normalized_name_key(create_name)
            create_name = create_aliases.setdefault(key, create_name)
        existing_id = _find_existing_id_by_name(agg, create_name)
        if existing_id is not None:
            result = agg._do_add(existing_id, row)
            if result == "ok":
                agg.stats["added"] += 1
            else:
                agg._do_create(create_name, row)
                # register normalized key for stronger dedup gateway
                agg._name_to_id[_normalized_name_key(create_name)] = agg._next_id - 1
                agg.stats["created"] += 1
        else:
            agg._do_create(create_name, row)
            # register normalized key for stronger dedup gateway
            agg._name_to_id[_normalized_name_key(create_name)] = agg._next_id - 1
            agg.stats["created"] += 1

    elif decision["action"] == "ADD":
        target = decision.get("target_id")
        target_name = decision.get("target_name")
        valid_add_ids = food.get("_valid_add_ids", set())

        if target is not None and target not in valid_add_ids:
            target = None

        if target is None and target_name:
            matches = agg.index.search(target_name, top_k=1)
            if matches and matches[0]["score"] > 0.75 and matches[0]["id"] in valid_add_ids:
                target = matches[0]["id"]

        if target and target in agg.db:
            result = agg._do_add(target, row)
            if result == "ok":
                agg.stats["added"] += 1
            elif result == "foundation_locked":
                agg.stats["ignored"] += 1
            else:
                cname = _sanitize_generic_name(target_name or name)
                agg._do_create(cname, row)
                agg._name_to_id[_normalized_name_key(cname)] = agg._next_id - 1
                agg.stats["created"] += 1
        else:
            cname = _sanitize_generic_name(target_name or name)
            agg._do_create(cname, row)
            agg._name_to_id[_normalized_name_key(cname)] = agg._next_id - 1
            agg.stats["created"] += 1

    elif decision["action"] == "IGNORE":
        agg.stats["ignored"] += 1
    else:
        agg.stats["errors"] += 1
        agg.stats["ignored"] += 1


def _parse_group_only(line: str, expected_idx: int):
    import re
    s = line.strip().lstrip('-').strip()
    if not s:
        return None
    # Accept no-index format
    if re.match(r'^(GROUP|IGNORE)\b', s, re.IGNORECASE):
        s = f'[{expected_idx}] ' + s
    m = re.match(r'^\[(\d+)\]\s*(GROUP|IGNORE)\s*(.*)$', s, re.IGNORECASE)
    if not m:
        return None
    idx = int(m.group(1))
    if idx != expected_idx:
        return None
    act = m.group(2).upper()
    rest = m.group(3).strip().strip('"').strip("'")
    if act == 'IGNORE':
        return {'idx': expected_idx, 'action': 'IGNORE'}
    if act == 'GROUP' and rest:
        return {'idx': expected_idx, 'action': 'CREATE', 'name': rest}
    return None


def _parse_json_line(raw: str, expected_idx: int):
    try:
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = '\n'.join(cleaned.split('\n')[1:])
        if cleaned.endswith('```'):
            cleaned = '\n'.join(cleaned.split('\n')[:-1])
        obj = json.loads(cleaned)
    except Exception:
        return None

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict) and int(it.get('idx', -1)) == expected_idx:
                obj = it
                break
        else:
            return None

    if not isinstance(obj, dict):
        return None
    idx_val = int(obj.get('idx', -1))
    if expected_idx >= 0 and idx_val != expected_idx:
        return None

    action = str(obj.get('action', '')).upper()
    if action == 'IGNORE':
        return {'idx': idx_val, 'action': 'IGNORE'}
    if action == 'ADD' and obj.get('target_id') is not None:
        return {'idx': idx_val, 'action': 'ADD', 'target_id': int(obj['target_id'])}
    if action == 'CREATE' and obj.get('name'):
        return {'idx': idx_val, 'action': 'CREATE', 'name': str(obj['name'])}
    if action == 'GROUP' and obj.get('name'):
        return {'idx': idx_val, 'action': 'CREATE', 'name': str(obj['name'])}
    return None


_DEFAULT_BATCH_INSTRUCTION = (
    "\n\nReturn exactly one decision per item as strict JSON object lines."
    "\nAllowed JSON schemas:"
    "\n{\"idx\": <idx>, \"action\": \"CREATE\", \"name\": \"<generic_name>\"}"
    "\n{\"idx\": <idx>, \"action\": \"ADD\", \"target_id\": <id>}"
    "\n{\"idx\": <idx>, \"action\": \"IGNORE\"}"
    "\n"
    "\nCRITICAL — anti-catch-all rule: each item in this batch is a DIFFERENT food. Expect to CREATE a distinct group"
    " for MOST items. Only reuse an identical CREATE name or issue ADD if you are CERTAIN two foods are the exact same"
    " generic food (e.g. two brands of plain white bread, or two preparations of the same bean variety)."
    " Grouping all or most items under one name is ALWAYS wrong."
    "\n"
    "\nWithin-batch name reuse: if item [3] creates 'Red Kidney Beans' and item [9] is also red kidney beans,"
    " output CREATE with the EXACT same name string — they will be auto-merged. But corn ≠ collard greens ≠ asparagus"
    " even if all are cooked vegetables."
    "\nADD is only valid using an id shown in that item's candidate list."
)


def run_batched(
    agg: FoodAggregator,
    *,
    limit: int | None,
    offset: int,
    resume_from: int,
    batch_size: int,
    decision_mode: str = "line",
    batch_instruction: str | None = None,
):
    df = agg.source_df
    df = df[df["foodName"].notna() & (df["foodName"].astype(str) != "nan")].reset_index(drop=True)
    df["_dtype_rank"] = df["data_type"].apply(_data_type_rank)
    df = df.sort_values(["_dtype_rank", "food_category"], ascending=[True, True], na_position="last").reset_index(drop=True)
    if offset:
        df = df.iloc[offset:].reset_index(drop=True)
    if limit:
        df = df.iloc[:limit]

    total = len(df)
    start_ts = time.time()

    i = resume_from
    while i < total:
        chunk = []
        end = min(i + batch_size, total)
        for idx in range(i, end):
            row = df.iloc[idx]
            name = str(row.get("foodName", ""))
            cat = str(row.get("food_category", ""))
            search_results = agg.index.search(name, top_k=agg.search_top_k)
            for m in search_results:
                entry = agg.db.get(m["id"])
                if entry:
                    m["nutrients"] = _nutrient_fingerprint(entry["nutrients"], is_dict=True)
                    m["count"] = entry.get("count", 1)

            food = {
                "idx": idx,
                "name": name,
                "category": cat,
                "nutrients_str": _nutrient_fingerprint(row),
                "search_results": search_results,
                "_valid_add_ids": {m["id"] for m in search_results},
            }
            chunk.append((idx, row, food))

        user_msg = "\n\n".join(_build_item_prompt(food) for _, _, food in chunk)
        if decision_mode == "group":
            user_msg += (
                "\n\nReturn exactly one decision per item as strict JSON object lines."
                "\nAllowed JSON schemas:"
                "\n{\"idx\": <idx>, \"action\": \"GROUP\", \"name\": \"<generic_name>\"}"
                "\n{\"idx\": <idx>, \"action\": \"IGNORE\"}"
                "\nUse the EXACT same GROUP name for items that belong together in this batch."
            )
        else:
            user_msg += batch_instruction if batch_instruction is not None else _DEFAULT_BATCH_INSTRUCTION

        try:
            raw = _call_llm_batch(
                agg.system_prompt,
                user_msg,
                api_key=agg.api_key,
                base_url=agg.base_url,
                model=agg.model,
                timeout_seconds=agg.timeout_seconds,
            )
            agg.stats["api_calls"] += 1
        except Exception:
            # fallback: mark all as ignored on batch failure
            for _idx, _row, _food in chunk:
                agg.stats["errors"] += 1
                agg.stats["ignored"] += 1
                agg.processed_count += 1
            i = end
            continue

        # parse per line JSON first; fallback to legacy parser only for backward compatibility
        line_map = {}
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parsed = _parse_json_line(ln, expected_idx=-1)
            # expected_idx=-1 means disabled strict check in this stage; we parse idx later
            if parsed and parsed.get('idx') is not None:
                line_map[int(parsed['idx'])] = ln
            elif ln.startswith("[") and "]" in ln:
                try:
                    idx = int(ln[1:ln.index("]")])
                    line_map[idx] = ln
                except Exception:
                    pass

        create_aliases: dict[str, str] = {}
        for idx, row, food in chunk:
            decision = None
            if idx in line_map:
                decision = _parse_json_line(line_map[idx], idx)
                if decision is None:
                    decision = _parse_group_only(line_map[idx], idx) if decision_mode == "group" else _parse_llm_decision(line_map[idx], idx)
            if decision is None:
                decision = _parse_json_line(raw, idx)
            if decision is None:
                decision = _parse_group_only(raw, idx) if decision_mode == "group" else _parse_llm_decision(raw, idx)
            if decision is None:
                agg.stats["errors"] += 1
                agg.stats["ignored"] += 1
            else:
                _apply_decision(agg, food, row, decision, create_aliases=create_aliases)
            agg.processed_count += 1

        if agg.processed_count % 500 == 0:
            agg._save_checkpoint()
        i = end

    agg._apply_canonical_postpass()
    agg._save_checkpoint()
    agg._print_summary()

    elapsed = time.time() - start_ts
    return {"elapsed_seconds": elapsed, "processed": agg.processed_count, "api_calls": agg.stats.get("api_calls", 0), "final_groups": len(agg.db)}


def main():
    parser = argparse.ArgumentParser(description="Run aggregation pipeline (batched)")
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--model", default="google/gemini-2.0-flash-lite-001")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--search-top-k", type=int, default=8)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="skip first N sorted rows before processing")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--estimate-full-size", type=int, default=296000)
    parser.add_argument("--estimated-cost-per-call", type=float, default=0.0, help="optional rough USD estimate per LLM API call")
    parser.add_argument("--decision-mode", choices=["line", "group"], default="line", help="line=create/add/ignore, group=group-name-or-ignore")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path, output_path, checkpoint_dir = _resolve_input_output(repo_root, args.mode, args.output, args.checkpoint_dir)

    # Resolve LLM backend — falls back to local Copilot proxy if no key is set.
    args.api_key, args.base_url, args.model = _resolve_llm_backend(
        args.api_key, args.base_url, args.model
    )

    df = pd.read_csv(input_path)

    prompt_path = args.prompt
    tmp_prompt = None
    if not prompt_path:
        tmp_prompt = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        tmp_prompt.write(DEFAULT_PROMPT)
        tmp_prompt.flush()
        prompt_path = tmp_prompt.name

    agg = FoodAggregator(
        df,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        batch_size=args.batch_size,
        search_top_k=args.search_top_k,
        prompt_path=prompt_path,
        checkpoint_dir=str(checkpoint_dir),
        timeout_seconds=args.timeout_seconds,
    )

    resume_from = 0
    if args.resume and agg.load_checkpoint():
        resume_from = agg.processed_count

    limit = args.limit
    if args.mode == "full":
        limit = limit or None

    result = run_batched(
        agg,
        limit=limit,
        offset=args.offset,
        resume_from=resume_from,
        batch_size=args.batch_size,
        decision_mode=args.decision_mode,
    )
    agg.save(str(output_path))

    # Human-review friendly grouping trace: Group Name: item1, item2, ...
    groups_txt = output_path.with_suffix('.groups.txt')
    lines = []
    for entry in sorted(agg.db.values(), key=lambda e: str(e.get('generic_name', '')).lower()):
        gname = entry.get('generic_name', 'Unknown Group')
        src_names = entry.get('source_names', [])
        unique_src = []
        seen = set()
        for s in src_names:
            t = str(s).strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            unique_src.append(t)
        lines.append(f"{gname}: " + " | ".join(unique_src))
    groups_txt.write_text("\n".join(lines) + "\n")

    rate = result["processed"] / result["elapsed_seconds"] if result["elapsed_seconds"] > 0 else 0
    eta_seconds = args.estimate_full_size / rate if rate > 0 else math.inf

    est_cost = result["api_calls"] * args.estimated_cost_per_call
    est_full_calls = (args.estimate_full_size / result["processed"] * result["api_calls"]) if result["processed"] > 0 else 0
    est_full_cost = est_full_calls * args.estimated_cost_per_call
    summary = {
        "mode": args.mode,
        "provider": "openrouter",
        "model": args.model,
        "batch_size": args.batch_size,
        "offset": args.offset,
        "decision_mode": args.decision_mode,
        "input_rows": int(len(df if limit is None else df.iloc[:limit])),
        "processed": result["processed"],
        "api_calls": result["api_calls"],
        "final_groups": result["final_groups"],
        "elapsed_seconds": round(result["elapsed_seconds"], 2),
        "items_per_second": round(rate, 4),
        "eta_full_dataset_seconds": round(eta_seconds, 2) if math.isfinite(eta_seconds) else None,
        "eta_full_dataset_hms": time.strftime("%Hh %Mm %Ss", time.gmtime(eta_seconds)) if math.isfinite(eta_seconds) else None,
        "estimated_cost_per_call_usd": args.estimated_cost_per_call,
        "estimated_cost_usd": round(est_cost, 6),
        "estimated_full_api_calls": round(est_full_calls, 2),
        "estimated_full_cost_usd": round(est_full_cost, 6),
        "output_json": str(output_path),
    }

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2))
    print("\nBatching metrics:")
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved groups  -> {groups_txt}")

    if tmp_prompt is not None:
        try:
            os.unlink(tmp_prompt.name)
        except OSError:
            pass


if __name__ == "__main__":
    main()

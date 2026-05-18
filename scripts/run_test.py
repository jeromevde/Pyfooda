#!/usr/bin/env python3
"""
run_test.py — Aggregation test runner.

Runs run_batched on all items from test_groups.txt with the real pipeline
— no seeding, no clusters, no special ordering.

Scoring (per item, per group):
  items[0] of each group  → PASS if LLM created a group whose name matches
                             the canonical group name (token Jaccard >= threshold)
  items[1+] of each group → PASS if they ended up in the same group as items[0]

Multiple LLM calls happen naturally: batch_size controls how many items go
per call, so FAISS accumulates between batches just like the real pipeline.
Cascading errors (items[0] wrong → items[1+] wrong) are expected and real.

Usage:
  python scripts/run_test.py
  python scripts/run_test.py --batch-size 20
  python scripts/run_test.py -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from run_aggregation import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PROMPT,
    FoodAggregator,
    _resolve_llm_backend,
    run_batched,
)

# ── test-set I/O ──────────────────────────────────────────────────────────────

def load_groups(path: Path) -> dict[str, list[str]]:
    """Parse test_groups.txt → {group_name: [item, ...]}."""
    groups: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 2:
            groups[parts[0]] = parts[1:]
    return groups


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Aggregation test runner — runs real pipeline on test items and scores per item',
    )
    parser.add_argument('--groups',        default='scripts/test_groups.txt')
    parser.add_argument('--data',          default=None, help='Path to fooddata CSV (auto-detected)')
    parser.add_argument('--batch-size',    type=int,   default=DEFAULT_BATCH_SIZE,
                        help='Items per LLM call (fewer = more FAISS batches = more realistic)')
    parser.add_argument('--search-top-k',  type=int,   default=8)
    parser.add_argument('--model',         default=None)
    parser.add_argument('--base-url',      default=None)
    parser.add_argument('--api-key',       default=None)
    parser.add_argument('--prompt',        default=None, help='Path to prompt .txt file')
    parser.add_argument('--only',       action='append', default=[], metavar='PATTERN',
                        help='Only include groups whose name contains PATTERN (repeatable)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show PASS lines and output DB group names')
    args = parser.parse_args()

    repo_root   = Path(__file__).resolve().parents[1]
    groups_path = Path(args.groups) if Path(args.groups).is_absolute() else repo_root / args.groups
    data_path   = (
        Path(args.data) if args.data
        else repo_root / 'scripts/test_fooddata.csv'
    )
    if not data_path.exists():
        data_path = repo_root / 'pyfooda/data/fooddata.csv'

    api_key, base_url, model = _resolve_llm_backend(args.api_key, args.base_url, args.model)
    system_prompt = Path(args.prompt).read_text() if args.prompt else DEFAULT_PROMPT

    groups = load_groups(groups_path)
    if args.only:
        pats   = [p.lower() for p in args.only]
        groups = {g: items for g, items in groups.items()
                  if any(p in g.lower() for p in pats)}
    df     = pd.read_csv(data_path)

    # Build mini_df from all items listed in test_groups.txt
    rows_by_item: dict[str, pd.Series] = {}
    missing: list[str] = []
    for gname, items in groups.items():
        for item in items:
            if item in rows_by_item:
                continue
            match = df[df['foodName'] == item]
            if match.empty:
                missing.append(item)
            else:
                rows_by_item[item] = match.iloc[0]

    if not rows_by_item:
        print('ERROR: no test items found in CSV')
        sys.exit(1)

    mini_df = pd.DataFrame(list(rows_by_item.values())).reset_index(drop=True)
    n_items   = len(mini_df)
    n_batches = (n_items + args.batch_size - 1) // args.batch_size

    # Rough cost estimate for OSS 120B on OpenRouter (~$0.40/M input tokens)
    est_input_tokens = n_items * 200 + n_batches * (len(system_prompt) // 4 + 100)
    est_cost_usd     = est_input_tokens * 0.40 / 1_000_000
    print(f'Groups: {len(groups)}  |  Items: {n_items}  |  batch_size={args.batch_size}'
          f'  →  {n_batches} LLM call(s)  |  est. cost OSS-120B: ${est_cost_usd:.4f}\n')

    # ── per-batch scoring state ────────────────────────────────────────────────
    item_to_group: dict[str, tuple[str, int]] = {
        item: (gname, pos)
        for gname, items in groups.items()
        for pos, item in enumerate(items)
    }
    scored:       set[str]       = set()
    anchor_gids:  dict[str, int] = {}   # gname → db gid of items[0]
    printed_grps: set[str]       = set()
    total_pass    = 0
    total_fail    = 0
    failures: list[str] = []

    def _print_group_header(gname: str) -> None:
        if gname not in printed_grps:
            printed_grps.add(gname)
            print(f'\n{gname}')

    def _record(item: str, gname: str, passed: bool, actual: str, note: str) -> None:
        nonlocal total_pass, total_fail
        tag = f'[{note}]'
        _print_group_header(gname)
        if passed:
            total_pass += 1
            line = f'  PASS  {tag}  "{item}"'
            if args.verbose:
                line += f'  → "{actual}"'
            print(line, flush=True)
        else:
            total_fail += 1
            if actual not in ('?', '(not found)', '(not in output)', '(not in CSV)'):
                suffix = f'  → got "{actual}"'
            else:
                suffix = f'  → {actual}'
            print(f'  FAIL  {tag}  "{item}"{suffix}', flush=True)
            failures.append(f'{gname}  {tag}  "{item}"')
        scored.add(item)

    def on_batch_complete(foods: list) -> None:
        """Score any newly-processable items after each batch."""
        source_map = {
            src: gid
            for gid, entry in agg.db.items()
            for src in entry.get('source_names', [])
        }
        # Walk groups in stable order so anchor (pos=0) is always visited before dependents
        for gname, items in groups.items():
            for pos, item in enumerate(items):
                if item in scored:
                    continue
                if item not in source_map:
                    continue  # not yet processed by any batch
                if pos > 0 and gname not in anchor_gids:
                    continue  # anchor not yet seen
                gid = source_map[item]
                if pos == 0:
                    anchor_gids[gname] = gid
                    created_name = agg.db[gid]['generic_name']
                    faiss_hits = agg.index.search(gname, top_k=1)
                    name_ok = bool(faiss_hits and faiss_hits[0]['id'] == gid)
                    _record(item, gname, name_ok, created_name, 'CREATE')
                else:
                    anchor_gid  = anchor_gids[gname]
                    actual_gid  = source_map.get(item)
                    passed      = (actual_gid == anchor_gid)
                    actual_name = (
                        agg.db.get(actual_gid, {}).get('generic_name', '(not found)')
                        if actual_gid is not None else '(not found)'
                    )
                    _record(item, gname, passed, actual_name, 'ADD')

    print('Loading sentence-transformer model...', flush=True)
    agg = FoodAggregator(
        mini_df,
        api_key=api_key,
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        batch_size=args.batch_size,
        search_top_k=args.search_top_k,
        checkpoint_dir='/tmp/pyfooda_test',
    )
    # Disable postpass — test LLM decisions only, not post-hoc heuristics
    agg._apply_similarity_merge_postpass = lambda *a, **kw: None
    agg._apply_canonical_postpass        = lambda *a, **kw: None

    run_batched(agg, limit=None, offset=0, resume_from=0, batch_size=args.batch_size,
                on_batch_complete=on_batch_complete)

    # Final pass: catch items deferred across batch boundaries
    on_batch_complete([])

    # Items missing from CSV entirely
    for item in missing:
        if item in item_to_group and item not in scored:
            gname, pos = item_to_group[item]
            _record(item, gname, False, '(not in CSV)', 'CREATE' if pos == 0 else 'ADD')

    total = total_pass + total_fail
    pct   = 100 * total_pass / total if total else 0.0
    print('\n' + '─' * 55)
    print(f'Score: {total_pass}/{total}  ({pct:.1f}%)')

    if missing:
        print(f'\nMissing from CSV ({len(missing)}): {missing}')

    if failures and not args.verbose:
        print(f'\nFailed ({len(failures)}):')
        for f in failures:
            print(f'  {f}')


if __name__ == '__main__':
    main()

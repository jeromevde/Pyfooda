#!/usr/bin/env python3
"""
run_test.py — Targeted aggregation test runner.

For each group in test_groups.txt, finds confusable partner groups
(Jaccard similarity on name tokens >= threshold) and runs a mini-pipeline
with just those 2-3 groups' items against a fresh empty DB.

Scores each cluster as PASS/FAIL for:
  merge  — all items from the same group land in the same output group
  split  — items from different groups land in different output groups

This lets you iterate on the prompt and aggregation logic quickly:
one cluster = 1-3 LLM calls, not hundreds.

Usage:
  python scripts/run_test.py
  python scripts/run_test.py --only Pasta --only Beer
  python scripts/run_test.py -v
  python scripts/run_test.py --threshold 0.25 --top-k 2
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import pandas as pd

# same-directory import
sys.path.insert(0, str(Path(__file__).parent))

from run_aggregation import (
    DEFAULT_PROMPT,
    FoodAggregator,
    _resolve_llm_backend,
    run_batched,
)

# ── token helpers ──────────────────────────────────────────────────────────────

_STOP = {
    'a', 'an', 'and', 'as', 'at', 'by', 'for', 'from', 'in', 'n', 'no', 'of',
    'on', 'or', 's', 'the', 'to', 'with', 'without',
    'added', 'all', 'any', 'baked', 'boiled', 'broiled', 'canned', 'cooked',
    'dark', 'dried', 'dry', 'eaten', 'fat', 'form', 'free', 'fresh', 'fried',
    'frozen', 'grilled', 'heat', 'high', 'light', 'lightly', 'low', 'meat',
    'mild', 'moist', 'not', 'nfs', 'ns', 'only', 'plain', 'raw', 'reduced',
    'regular', 'roasted', 'skin', 'smoked', 'species', 'steamed', 'stewed',
    'type', 'unsalted', 'whole',
}


def _tokens(name: str) -> set[str]:
    return set(re.findall(r'\b[a-z]{2,}\b', name.lower())) - _STOP


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


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


# ── cluster builder ────────────────────────────────────────────────────────────

def build_clusters(
    groups: dict[str, list[str]],
    threshold: float,
    top_k: int = 2,
) -> list[list[str]]:
    """
    For each group, pair it with up to top_k most confusable partners
    (Jaccard >= threshold).  Returns deduplicated clusters, each a list
    with the focal group first.
    """
    all_names = list(groups.keys())
    seen: set[frozenset] = set()
    clusters: list[list[str]] = []

    for name in all_names:
        scored = sorted(
            [(n, _jaccard(name, n)) for n in all_names if n != name],
            key=lambda x: -x[1],
        )
        partners = [n for n, s in scored[:top_k] if s >= threshold]
        cluster = [name] + partners
        key = frozenset(cluster)
        if key not in seen:
            seen.add(key)
            clusters.append(cluster)

    return clusters


# ── mini pipeline ─────────────────────────────────────────────────────────────

def run_cluster(
    cluster: list[str],
    groups: dict[str, list[str]],
    df: pd.DataFrame,
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    batch_size: int,
    verbose: bool = False,
) -> dict:
    """
    Run a fresh mini-aggregation on the items from `cluster` groups.

    Returns:
        merge  {group_name: True/False/None}   all items in same output group?
        split  {(g1, g2): True/False}          items from each pair in different groups?
        db_names  {group_id: generic_name}     what the model actually created
        missing   [item_name, ...]             items not found in CSV
    """
    # Collect items and match to CSV rows
    item_to_group: dict[str, str] = {
        item: gname
        for gname in cluster
        for item in groups[gname]
    }
    mini_rows, missing = [], []
    for item in item_to_group:
        match = df[df['foodName'] == item]
        if match.empty:
            missing.append(item)
        else:
            mini_rows.append(match.iloc[0])

    if not mini_rows:
        return {'error': 'no rows found in CSV', 'missing': missing,
                'merge': {}, 'split': {}, 'db_names': {}}

    mini_df = pd.DataFrame(mini_rows).reset_index(drop=True)
    bs = max(batch_size, len(mini_df))   # single batch — all items in one LLM call

    agg = FoodAggregator(
        mini_df,
        api_key=api_key,
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        batch_size=bs,
        checkpoint_dir=f'/tmp/pyfooda_test_{abs(hash(frozenset(cluster))) % 0xFFFFFF:06x}',
    )

    # Disable postpass: operates on tiny DBs and can merge things that should split
    agg._apply_similarity_merge_postpass = lambda *a, **kw: None
    agg._apply_canonical_postpass = lambda *a, **kw: None

    sink = io.StringIO()
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)
    with ctx:
        run_batched(agg, limit=None, offset=0, resume_from=0, batch_size=bs)

    # Map source food name → output group id
    source_map: dict[str, int] = {
        src: gid
        for gid, entry in agg.db.items()
        for src in entry.get('source_names', [])
    }

    # ── merge check ──────────────────────────────────────────────────────────
    merge: dict[str, bool | None] = {}
    for gname in cluster:
        items = groups[gname]
        gids = {source_map.get(it) for it in items} - {None}
        if not gids:
            merge[gname] = None          # couldn't evaluate (all missing)
        elif len(items) < 2:
            merge[gname] = True          # single item, trivially merged
        else:
            merge[gname] = len(gids) == 1

    # ── split check ──────────────────────────────────────────────────────────
    split: dict[tuple[str, str], bool] = {}
    for i, g1 in enumerate(cluster):
        for g2 in cluster[i + 1:]:
            gids1 = {source_map.get(it) for it in groups[g1]} - {None}
            gids2 = {source_map.get(it) for it in groups[g2]} - {None}
            if gids1 and gids2:
                split[(g1, g2)] = gids1.isdisjoint(gids2)

    return {
        'merge': merge,
        'split': split,
        'db_names': {gid: e['generic_name'] for gid, e in agg.db.items()},
        'missing': missing,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Targeted aggregation test runner (mini-pipeline per confusable cluster)',
    )
    parser.add_argument('--groups',     default='tests/test_groups.txt')
    parser.add_argument('--data',       default=None,  help='Path to fooddata CSV (auto-detected)')
    parser.add_argument('--threshold',  type=float, default=0.30,
                        help='Jaccard threshold for confusable pair detection')
    parser.add_argument('--top-k',      type=int,   default=2,
                        help='Max confusable partners per focal group')
    parser.add_argument('--batch-size', type=int,   default=8)
    parser.add_argument('--model',      default=None)
    parser.add_argument('--base-url',   default=None)
    parser.add_argument('--api-key',    default=None)
    parser.add_argument('--prompt',     default=None, help='Path to prompt .txt file')
    parser.add_argument('--only',       action='append', default=[], metavar='PATTERN',
                        help='Only run clusters whose group names contain PATTERN (repeatable)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show PASS lines too, and DB group names on failures')
    args = parser.parse_args()

    repo_root   = Path(__file__).resolve().parents[1]
    groups_path = Path(args.groups) if Path(args.groups).is_absolute() else repo_root / args.groups
    data_path   = (
        Path(args.data) if args.data
        else repo_root / 'tests/test_fooddata.csv'
    )
    if not data_path.exists():
        data_path = repo_root / 'pyfooda/data/fooddata.csv'

    api_key, base_url, model = _resolve_llm_backend(args.api_key, args.base_url, args.model)
    system_prompt = Path(args.prompt).read_text() if args.prompt else DEFAULT_PROMPT

    groups   = load_groups(groups_path)
    df       = pd.read_csv(data_path)
    clusters = build_clusters(groups, threshold=args.threshold, top_k=args.top_k)

    if args.only:
        pats = [p.lower() for p in args.only]
        clusters = [c for c in clusters if any(p in g.lower() for g in c for p in pats)]

    total_pass = total_checks = 0
    failures: list[str] = []

    print(f'Groups: {len(groups)}  |  Clusters: {len(clusters)}'
          f'  |  threshold={args.threshold}  top_k={args.top_k}\n')

    for cluster in clusters:
        result = run_cluster(
            cluster, groups, df,
            api_key=api_key, base_url=base_url, model=model,
            system_prompt=system_prompt,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        if 'error' in result and not result.get('merge'):
            print(f'ERROR  [{" | ".join(cluster)}]: {result["error"]}')
            continue

        merge     = result['merge']
        split     = result['split']
        label     = ' | '.join(cluster)
        ok_lines:   list[str] = []
        fail_lines: list[str] = []

        for gname in cluster:
            n = len(groups[gname])
            v = merge.get(gname)
            total_checks += 1
            if v is True:
                total_pass += 1
                ok_lines.append(f'  PASS  merge: {gname} ({n} item{"s" if n != 1 else ""})')
            elif v is False:
                fail_lines.append(f'  FAIL  merge: {gname} ({n} items)')
                failures.append(f'merge  {gname}')
            else:
                fail_lines.append(f'  ?     merge: {gname} (no items found in CSV)')

        for (g1, g2), v in split.items():
            total_checks += 1
            if v:
                total_pass += 1
                ok_lines.append(f'  PASS  split: {g1} ↔ {g2}')
            else:
                fail_lines.append(f'  FAIL  split: {g1} ↔ {g2}')
                failures.append(f'split  {g1} ↔ {g2}')

        cluster_ok = not fail_lines
        print(f'{"PASS" if cluster_ok else "FAIL"}  {label}')

        if fail_lines:
            for ln in fail_lines:
                print(ln)
            if args.verbose:
                db = result.get('db_names', {})
                print(f'       output groups: {list(db.values())}')
        if args.verbose and ok_lines:
            for ln in ok_lines:
                print(ln)

        if result.get('missing') and (fail_lines or args.verbose):
            print(f'       missing from CSV: {result["missing"]}')

        if fail_lines or args.verbose:
            print()

    pct = 100 * total_pass / total_checks if total_checks else 0.0
    print('─' * 55)
    print(f'Score: {total_pass}/{total_checks}  ({pct:.1f}%)')
    if failures:
        print(f'\nFailed ({len(failures)}):')
        for f in failures:
            print(f'  {f}')


if __name__ == '__main__':
    main()

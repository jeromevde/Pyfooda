#!/usr/bin/env python3
"""Score aggregation output against tests/test_groups.txt.

Groups file format
------------------
  group_name | food_item_1 | food_item_2 | ...
  Lines starting with # are comments.

How it works
------------
  1. Each group defines items that SHOULD land in the same output group
     → merge_check (one check per group with ≥2 found items)

  2. Confusable pairs are auto-detected: any two groups whose name-token
     Jaccard ≥ threshold (default 0.30) are tested for correct separation
     → split_check (one check per confusable pair)

  3. score = 1 - (failures / total_checks)

Usage
-----
  # After running: python run_aggregation.py --mode test --batch-size 24
  python score_groups.py
  python score_groups.py tests/output/batch_test_aggregated.json
  python score_groups.py --groups tests/test_groups.txt --threshold 0.30 -v
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "for", "with", "to", "from",
    "is", "as", "by", "at", "on", "vs", "no", "not",
}


def _name_tokens(name: str) -> set[str]:
    words = set(re.findall(r"\b[a-z]+\b", name.lower()))
    return words - _STOP


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_groups(path: str) -> dict[str, list[str]]:
    """Parse pipe-delimited groups file → {group_name: [item, ...]}."""
    groups: dict[str, list[str]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            name = parts[0]
            items = [p for p in parts[1:] if p]
            if items:
                groups[name] = items
    return groups


def load_output(path: str) -> tuple[dict[str, int], dict[int, str]]:
    """Load aggregation JSON → (source_name → group_id, group_id → generic_name)."""
    with open(path) as fh:
        data = json.load(fh)
    source_to_gid: dict[str, int] = {}
    gid_to_name: dict[int, str] = {}
    for entry in data.get("foods", []):
        gid = int(entry["id"])
        gid_to_name[gid] = str(entry.get("generic_name", ""))
        for sname in entry.get("source_names", []):
            key = _norm(sname)
            if key:
                source_to_gid[key] = gid
    return source_to_gid, gid_to_name


def find_confusable_pairs(
    groups: dict[str, list[str]], threshold: float
) -> list[tuple[str, str, float]]:
    """Return (name_a, name_b, jaccard) for all pairs above threshold."""
    names = list(groups.keys())
    tokens = {n: _name_tokens(n) for n in names}
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            j_val = _jaccard(tokens[names[i]], tokens[names[j]])
            if j_val >= threshold:
                pairs.append((names[i], names[j], j_val))
    pairs.sort(key=lambda x: -x[2])
    return pairs


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score(
    groups_path: str,
    output_path: str,
    threshold: float = 0.30,
    verbose: bool = False,
    json_report: Optional[str] = None,
) -> float:
    groups = parse_groups(groups_path)
    s2g, g2n = load_output(output_path)
    pairs = find_confusable_pairs(groups, threshold)

    print(f"Groups: {len(groups)}, Confusable pairs: {len(pairs)} (threshold={threshold:.2f})")

    if verbose:
        print("\nAuto-detected confusable pairs:")
        for a, b, j in pairs:
            shared = _name_tokens(a) & _name_tokens(b)
            print(f"  [{j:.2f}] {a!r} ↔ {b!r}  (shared: {sorted(shared)})")

    failures: list[str] = []
    passes:   list[str] = []

    # ── merge checks ─────────────────────────────────────────────────────────
    merge_pass = merge_fail = merge_skip = 0

    for gname, items in groups.items():
        found  = {i: s2g[_norm(i)] for i in items if _norm(i) in s2g}
        missed = [i for i in items if _norm(i) not in s2g]

        if missed and verbose:
            print(f"  WARN {gname!r}: not in output → {missed}")

        if len(found) < 2:
            merge_skip += 1
            continue  # can't test merge with fewer than 2 found items

        gids = set(found.values())
        if len(gids) == 1:
            merge_pass += 1
            gid = next(iter(gids))
            passes.append(f"  PASS merge  {gname!r} → group {gid} ({g2n.get(gid, '?')!r})")
        else:
            merge_fail += 1
            lines = [f"  FAIL merge  {gname!r} → split into {len(gids)} groups:"]
            for item, gid in sorted(found.items(), key=lambda kv: kv[1]):
                lines.append(f"      gid {gid} ({g2n.get(gid, '?')!r}): {item!r}")
            failures.append("\n".join(lines))

    # ── split checks ─────────────────────────────────────────────────────────
    split_pass = split_fail = split_skip = 0

    for a_name, b_name, j_val in pairs:
        found_a = {i: s2g[_norm(i)] for i in groups[a_name] if _norm(i) in s2g}
        found_b = {i: s2g[_norm(i)] for i in groups[b_name] if _norm(i) in s2g}

        if not found_a or not found_b:
            split_skip += 1
            continue

        gids_a = set(found_a.values())
        gids_b = set(found_b.values())
        overlap = gids_a & gids_b

        if not overlap:
            split_pass += 1
            passes.append(
                f"  PASS split  {a_name!r} ↔ {b_name!r}  [j={j_val:.2f}]"
            )
        else:
            split_fail += 1
            lines = [
                f"  FAIL split  {a_name!r} ↔ {b_name!r}  [j={j_val:.2f}]"
                f" — merged into group(s) {sorted(overlap)}"
            ]
            all_found = {**found_a, **found_b}
            for gid in sorted(overlap):
                merged = [i for i, g in all_found.items() if g == gid]
                lines.append(f"      gid {gid} ({g2n.get(gid, '?')!r}): {merged}")
            failures.append("\n".join(lines))

    # ── summary ──────────────────────────────────────────────────────────────
    merge_total = merge_pass + merge_fail
    split_total = split_pass + split_fail
    total       = merge_total + split_total

    print(f"\nMerge  {merge_pass}/{merge_total} passed  ({merge_skip} skipped — single-item groups)")
    print(f"Split  {split_pass}/{split_total} passed  ({split_skip} skipped — items not found)")

    if failures:
        print(f"\n── Failures ({len(failures)}) ──────────────────────────────────────────")
        for f in failures:
            print(f)

    if verbose and passes:
        print(f"\n── Passes ({len(passes)}) ───────────────────────────────────────────────")
        for p in passes:
            print(p)

    if total == 0:
        print("\nNo checks performed — check that items exist in the output JSON.")
        return 0.0

    overall = (merge_pass + split_pass) / total
    loss    = 1 - overall
    n_fail  = merge_fail + split_fail
    print(f"\nScore  {overall:.1%}   (loss {loss:.1%}, {n_fail} failures / {total} checks)")

    if json_report:
        report = {
            "groups_file": groups_path,
            "output_file": output_path,
            "threshold": threshold,
            "merge": {"pass": merge_pass, "fail": merge_fail, "skip": merge_skip},
            "split": {"pass": split_pass, "fail": split_fail, "skip": split_skip},
            "score": round(overall, 4),
            "loss":  round(loss, 4),
        }
        with open(json_report, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"JSON report → {json_report}")

    return overall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).parent.parent
    default_output = repo_root / "tests" / "output" / "batch_test_aggregated.json"
    default_groups = repo_root / "tests" / "test_groups.txt"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output",    nargs="?", default=str(default_output),
                    help="Aggregation output JSON (default: tests/output/batch_test_aggregated.json)")
    ap.add_argument("--groups",  default=str(default_groups),
                    help="Groups file (default: tests/test_groups.txt)")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="Jaccard threshold for confusable-pair detection (default: 0.30)")
    ap.add_argument("--json-report", metavar="PATH",
                    help="Write JSON report to this path")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show passing checks and detected pair details")
    args = ap.parse_args()

    if not Path(args.output).exists():
        print(f"Error: output file not found: {args.output}", file=sys.stderr)
        print("Run:  python scripts/run_aggregation.py --mode test --batch-size 24", file=sys.stderr)
        sys.exit(1)
    if not Path(args.groups).exists():
        print(f"Error: groups file not found: {args.groups}", file=sys.stderr)
        sys.exit(1)

    score(
        groups_path=args.groups,
        output_path=args.output,
        threshold=args.threshold,
        verbose=args.verbose,
        json_report=args.json_report,
    )


if __name__ == "__main__":
    main()

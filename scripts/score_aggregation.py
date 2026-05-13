#!/usr/bin/env python3
"""
Score an aggregated foods JSON against the curated test set.

Metrics
-------
  merge   (35%) – fraction of merge cases where ALL listed foods land in the same group
  split   (45%) – fraction of split cases where ALL listed foods land in DIFFERENT groups
  name    (20%) – fraction of name-quality cases where the group name passes all rules

Overall = weighted average of the three.

Usage
-----
  python scripts/score_aggregation.py tests/batch_test_aggregated.json
  python scripts/score_aggregation.py path/to/foods.json --test-set tests/test_set.json
  python scripts/score_aggregation.py path/to/foods.json --json-report out/report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Name-quality helpers
# ---------------------------------------------------------------------------

# Brand names that must never appear in a generic group name
_BRAND_RE = re.compile(
    r"\b(KRAFT|BREAKSTONE'?S?|McDONALD'?S?|McFLURRY|BURGER\s+KING|PIZZA\s+HUT"
    r"|PAPA\s+JOHN'?S?|CHOBANI|DANNON(\s+OIKOS)?|CHICK.FIL.A|POPEYES"
    r"|WENDY'?S?|ARBY'?S?|TACO\s+BELL|KFC|DOMINO'?S?|LITTLE\s+CAESARS?"
    r"|LIFEWAY|BENECOL|Ritz\b|Cheez.It|Goldfish|Triscuit|Wheat\s+Thins"
    r"|OREO|M&M'?S?|BREAKSTONE)\b",
    re.IGNORECASE,
)

# USDA-style "Food, descriptor, ..." pattern
_USDA_FORMAT_RE = re.compile(r"^[A-Za-z][a-z]+,\s")

# Lab / spec tokens that must not appear in names
_LAB_TOKEN_RE = re.compile(
    r"\b(NFS|NS\s+as\s+to|moisture|lot\s+id|assay)\b", re.IGNORECASE
)

# Minor words that are expected in lowercase (not-first position)
_MINOR = {
    "a", "an", "the", "and", "or", "but", "for", "nor", "so", "yet",
    "at", "by", "in", "of", "on", "to", "up", "as", "vs", "with",
}


def _check_name_quality(name: str, case: dict) -> tuple[bool, list[str]]:
    """
    Return (passed, list_of_failure_messages).

    Rules applied:
      1. Word count <= max_words (default 6)
      2. Title case (minor words excepted after first position)
      3. No known brand names
      4. No USDA comma-descriptor format
      5. No lab/spec tokens
      6. case-level forbidden_patterns
      7. case-level required_patterns
    """
    failures: list[str] = []

    # 1. Word count
    words = name.split()
    max_words = int(case.get("max_words", 6))
    if len(words) > max_words:
        failures.append(
            f"too long ({len(words)} words, max {max_words}): '{name}'"
        )

    # 2. Title case
    bad_case = []
    for i, w in enumerate(words):
        if not w:
            continue
        is_minor = w.lower() in _MINOR
        if i == 0 and not w[0].isupper():
            bad_case.append(w)
        elif i > 0 and not is_minor and not w[0].isupper():
            bad_case.append(w)
    if bad_case:
        failures.append(f"not title case: {bad_case}")

    # 3. Known brand names
    m = _BRAND_RE.search(name)
    if m:
        failures.append(f"brand name detected '{m.group()}': '{name}'")

    # 4. USDA comma-descriptor format
    if _USDA_FORMAT_RE.search(name):
        failures.append(f"USDA comma-format detected: '{name}'")

    # 5. Lab tokens
    m2 = _LAB_TOKEN_RE.search(name)
    if m2:
        failures.append(f"lab/spec token '{m2.group()}': '{name}'")

    # 6. Case-level forbidden patterns
    for pattern in case.get("forbidden_patterns", []):
        if pattern.lower() in name.lower():
            failures.append(
                f"contains forbidden pattern '{pattern}': '{name}'"
            )

    # 7. Case-level required patterns
    for pattern in case.get("required_patterns", []):
        if pattern.lower() not in name.lower():
            failures.append(
                f"missing required pattern '{pattern}': '{name}'"
            )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def _load_aggregated(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """
    Parse aggregated JSON produced by FoodAggregator.save().

    Returns
    -------
    source_to_gid : normalized source name -> group id
    gid_to_name   : group id -> generic_name
    """
    with open(path) as f:
        data = json.load(f)

    foods = data.get("foods", [])
    source_to_gid: dict[str, int] = {}
    gid_to_name: dict[int, str] = {}

    for entry in foods:
        gid = int(entry["id"])
        gname = str(entry.get("generic_name", ""))
        gid_to_name[gid] = gname
        for sname in entry.get("source_names", []):
            key = _normalize(sname)
            if key:
                source_to_gid[key] = gid

    return source_to_gid, gid_to_name


# ---------------------------------------------------------------------------
# Per-case runners
# ---------------------------------------------------------------------------

def _run_merge(
    case: dict,
    source_to_gid: dict[str, int],
    gid_to_name: dict[int, str],
) -> dict:
    foods = case["foods"]
    lookup = {f: source_to_gid.get(_normalize(f)) for f in foods}
    found = {f: gid for f, gid in lookup.items() if gid is not None}
    not_found = [f for f, gid in lookup.items() if gid is None]

    base = {
        "id": case["id"],
        "type": "merge",
        "severity": case.get("severity", "normal"),
        "note": case.get("note", ""),
        "foods_total": len(foods),
        "foods_found": len(found),
        "not_found": not_found,
    }

    if len(found) < 2:
        return {
            **base,
            "pass": None,
            "reason": (
                f"fewer than 2 foods found in output "
                f"(not found: {not_found})"
            ),
        }

    unique_gids = set(found.values())
    passed = len(unique_gids) == 1
    reason = ""
    if not passed:
        per_group: dict[str, list[str]] = {}
        for f, gid in found.items():
            gname = gid_to_name.get(gid, f"gid={gid}")
            per_group.setdefault(gname, []).append(f)
        reason = f"split across {len(unique_gids)} groups: {per_group}"

    return {
        **base,
        "pass": passed,
        "groups": [gid_to_name.get(g, f"gid={g}") for g in unique_gids],
        "reason": reason,
    }


def _run_split(
    case: dict,
    source_to_gid: dict[str, int],
    gid_to_name: dict[int, str],
) -> dict:
    foods = case["foods"]
    lookup = {f: source_to_gid.get(_normalize(f)) for f in foods}
    # Foods absent from output are treated as their own null group
    # (they are not in ANY existing group, so they trivially don't conflict)
    found_pairs = [(f, gid) for f, gid in lookup.items() if gid is not None]
    not_found = [f for f, gid in lookup.items() if gid is None]

    base = {
        "id": case["id"],
        "type": "split",
        "severity": case.get("severity", "normal"),
        "note": case.get("note", ""),
        "foods_total": len(foods),
        "foods_found": len(found_pairs),
        "not_found": not_found,
    }

    if len(found_pairs) < 2:
        return {
            **base,
            "pass": None,
            "reason": (
                f"fewer than 2 foods found in output "
                f"(not found: {not_found})"
            ),
        }

    found_gids = [gid for _, gid in found_pairs]
    passed = len(set(found_gids)) == len(found_gids)
    reason = ""
    if not passed:
        gid_to_foods: dict[int, list[str]] = {}
        for f, gid in found_pairs:
            gid_to_foods.setdefault(gid, []).append(f)
        conflicts = {
            gid_to_name.get(gid, f"gid={gid}"): fs
            for gid, fs in gid_to_foods.items()
            if len(fs) > 1
        }
        reason = f"wrongly co-grouped: {conflicts}"

    return {**base, "pass": passed, "reason": reason}


def _run_name_quality(
    case: dict,
    source_to_gid: dict[str, int],
    gid_to_name: dict[int, str],
) -> dict:
    food = case["food"]
    gid = source_to_gid.get(_normalize(food))

    base = {
        "id": case["id"],
        "type": "name_quality",
        "severity": case.get("severity", "normal"),
        "note": case.get("note", ""),
        "food": food,
    }

    if gid is None:
        return {
            **base,
            "pass": None,
            "reason": f"food not found in output: '{food}'",
        }

    group_name = gid_to_name.get(gid, "")
    passed, failures = _check_name_quality(group_name, case)
    return {
        **base,
        "pass": passed,
        "group_name": group_name,
        "failures": failures,
        "reason": "; ".join(failures) if not passed else "",
    }


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score(aggregated_path: Path, test_set_path: Path) -> dict:
    source_to_gid, gid_to_name = _load_aggregated(aggregated_path)

    with open(test_set_path) as f:
        test_set = json.load(f)

    results = []
    for case in test_set["cases"]:
        t = case["type"]
        if t == "merge":
            results.append(_run_merge(case, source_to_gid, gid_to_name))
        elif t == "split":
            results.append(_run_split(case, source_to_gid, gid_to_name))
        elif t == "name_quality":
            results.append(_run_name_quality(case, source_to_gid, gid_to_name))

    def cat_stats(type_: str) -> tuple[int, int, int]:
        """(passed, failed, skipped)"""
        cases = [r for r in results if r["type"] == type_]
        p = sum(1 for r in cases if r["pass"] is True)
        f = sum(1 for r in cases if r["pass"] is False)
        s = sum(1 for r in cases if r["pass"] is None)
        return p, f, s

    m_p, m_f, m_s = cat_stats("merge")
    s_p, s_f, s_s = cat_stats("split")
    n_p, n_f, n_s = cat_stats("name_quality")

    cfg = test_set.get("scoring", {})
    w_merge = float(cfg.get("merge_weight", 0.35))
    w_split = float(cfg.get("split_weight", 0.45))
    w_name = float(cfg.get("name_quality_weight", 0.20))

    def pct(passed: int, failed: int) -> Optional[float]:
        total = passed + failed
        return round(100.0 * passed / total, 1) if total > 0 else None

    merge_score = pct(m_p, m_f)
    split_score = pct(s_p, s_f)
    name_score = pct(n_p, n_f)

    # Weighted average (only over categories that have at least one scored case)
    scored = [
        (merge_score, w_merge),
        (split_score, w_split),
        (name_score, w_name),
    ]
    total_w = sum(w for s, w in scored if s is not None)
    overall = (
        round(sum(s * w for s, w in scored if s is not None) / total_w, 1)
        if total_w > 0
        else None
    )

    # Per-severity breakdown (for split + merge cases)
    severity_breakdown: dict[str, dict] = {}
    for r in results:
        if r["type"] not in ("merge", "split"):
            continue
        sev = r.get("severity", "normal")
        sb = severity_breakdown.setdefault(sev, {"passed": 0, "failed": 0, "skipped": 0})
        if r["pass"] is True:
            sb["passed"] += 1
        elif r["pass"] is False:
            sb["failed"] += 1
        else:
            sb["skipped"] += 1

    return {
        "aggregated_path": str(aggregated_path),
        "test_set_path": str(test_set_path),
        "test_set_version": test_set.get("version", "unknown"),
        "scores": {
            "overall": overall,
            "merge": {
                "score": merge_score,
                "passed": m_p,
                "failed": m_f,
                "skipped": m_s,
                "weight": w_merge,
            },
            "split": {
                "score": split_score,
                "passed": s_p,
                "failed": s_f,
                "skipped": s_s,
                "weight": w_split,
            },
            "name_quality": {
                "score": name_score,
                "passed": n_p,
                "failed": n_f,
                "skipped": n_s,
                "weight": w_name,
            },
        },
        "severity_breakdown": severity_breakdown,
        "coverage": {
            "total_source_names_indexed": len(source_to_gid),
            "total_groups": len(gid_to_name),
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def _fmt_score(s: Optional[float]) -> str:
    if s is None:
        return "N/A"
    if s >= 90:
        return f"{s:.1f}%  [GOOD]"
    if s >= 70:
        return f"{s:.1f}%  [OK]"
    return f"{s:.1f}%  [POOR]"


def _print_report(report: dict) -> None:
    scores = report["scores"]
    sev = report.get("severity_breakdown", {})

    print()
    print("=" * 65)
    print("  AGGREGATION QUALITY REPORT")
    print("=" * 65)
    print(f"  Output:        {report['aggregated_path']}")
    print(f"  Test set:      {report['test_set_path']}  (v{report['test_set_version']})")
    cov = report["coverage"]
    print(f"  Groups:        {cov['total_groups']}  ({cov['total_source_names_indexed']} source names indexed)")
    print()

    overall = scores["overall"]
    print(f"  OVERALL SCORE:   {_fmt_score(overall)}")
    print()

    def row(label, d):
        return (
            f"  {label:<14} (w={d['weight']:.0%}):  {_fmt_score(d['score'])}  "
            f"pass={d['passed']}  fail={d['failed']}  skip={d['skipped']}"
        )

    print(row("Merge", scores["merge"]))
    print(row("Split", scores["split"]))
    print(row("Name quality", scores["name_quality"]))

    if sev:
        print()
        print("  Severity breakdown (merge + split cases):")
        for s_label in ("critical", "important", "normal"):
            if s_label in sev:
                sb = sev[s_label]
                total = sb["passed"] + sb["failed"]
                pct_str = (
                    f"{100*sb['passed']//total}%"
                    if total > 0 else "N/A"
                )
                print(
                    f"    {s_label:<10}  pass={sb['passed']}  "
                    f"fail={sb['failed']}  skip={sb['skipped']}  "
                    f"({pct_str})"
                )

    failures = [r for r in report["results"] if r["pass"] is False]
    if failures:
        print()
        print(f"  FAILURES ({len(failures)}):")
        # Show critical first, then important, then normal
        ordered = sorted(
            failures,
            key=lambda r: {"critical": 0, "important": 1, "normal": 2}.get(
                r.get("severity", "normal"), 2
            ),
        )
        for r in ordered:
            sev_label = r.get("severity", "normal").upper()
            tid = r["id"]
            reason = r.get("reason", "") or "; ".join(r.get("failures", []))
            note = r.get("note", "")
            gname = r.get("group_name") or (
                ", ".join(r.get("groups", [])) if r.get("groups") else ""
            )
            print(f"    [{tid}] [{sev_label}] {reason}")
            if note:
                print(f"           note: {note}")
            if gname:
                print(f"           group: {gname}")

    skipped = [r for r in report["results"] if r["pass"] is None]
    if skipped:
        print()
        print(f"  SKIPPED — food not found in output ({len(skipped)}):")
        for r in skipped:
            print(f"    [{r['id']}] {r.get('reason', '')}")

    print("=" * 65)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score aggregation output against the curated test set"
    )
    parser.add_argument(
        "aggregated",
        help="Path to aggregated foods JSON (output of FoodAggregator.save())",
    )
    parser.add_argument(
        "--test-set",
        default=None,
        help="Path to test_set.json (default: tests/test_set.json in repo root)",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        help="Also write the full structured report to this JSON file",
    )
    args = parser.parse_args()

    agg_path = Path(args.aggregated)
    if not agg_path.exists():
        print(f"error: aggregated file not found: {agg_path}", file=sys.stderr)
        sys.exit(1)

    if args.test_set:
        ts_path = Path(args.test_set)
    else:
        ts_path = Path(__file__).resolve().parents[1] / "tests" / "test_set.json"

    if not ts_path.exists():
        print(f"error: test set not found: {ts_path}", file=sys.stderr)
        sys.exit(1)

    report = score(agg_path, ts_path)
    _print_report(report)

    if args.json_report:
        out = Path(args.json_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report saved -> {out}")


if __name__ == "__main__":
    main()

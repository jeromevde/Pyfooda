#!/usr/bin/env python3
"""Print nutrient coverage report for the Pyfooda database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyfooda.build.coverage import write_coverage_report
from pyfooda.build.paths import COVERAGE_JSON, META_JSON, NUTRIENTS_CSV


def print_report(report: dict, *, limit: int = 15) -> None:
    total = report["ingredients_with_sources"]
    summary = report["summary"]
    print(f"Ingredients with USDA sources: {total}")
    print(
        f"Nutrients: {summary['well_measured']} well measured (≥50%), "
        f"{summary['partially_measured']} partial, "
        f"{summary['rarely_measured']} rarely measured"
    )
    print()

    nutrients = report["nutrients"]
    print(f"Best covered (top {limit}):")
    for row in nutrients[-limit:][::-1]:
        pct = row["coverage_pct"] * 100
        print(
            f"  {pct:5.1f}%  {row['name']} "
            f"({row['ingredients_with_value']}/{row['ingredients_total']})"
        )
    print()
    print(f"Worst covered (bottom {limit}):")
    for row in nutrients[:limit]:
        pct = row["coverage_pct"] * 100
        print(
            f"  {pct:5.1f}%  {row['name']} "
            f"({row['ingredients_with_value']}/{row['ingredients_total']})"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nutrient coverage report for Pyfooda")
    p.add_argument("--meta", type=Path, default=META_JSON)
    p.add_argument("--nutrients", type=Path, default=NUTRIENTS_CSV)
    p.add_argument("--output", type=Path, default=COVERAGE_JSON)
    p.add_argument("--json", action="store_true", help="Print full JSON to stdout")
    p.add_argument("--limit", type=int, default=15, help="Rows to show per section")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.meta.exists():
        print(f"Missing meta file: {args.meta}", file=sys.stderr)
        return 1
    report = write_coverage_report(args.meta, args.nutrients, args.output)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report, limit=args.limit)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

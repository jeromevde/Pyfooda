#!/usr/bin/env python3
"""Rebuild the ingredient database from the stored USDA matches.

This is the reproducible path. It does not call an embedding model or a
language model. Starting from ingredients_meta.json and the USDA export:

1. align.py keeps or replaces each ingredient's USDA rows by name and calories.
2. recompute.py resolves nutrient columns and writes ingredients.csv.
3. export_web.py writes the GitHub Pages catalog.

Usage:
    python -m pyfooda.build.rebuild
"""

from __future__ import annotations

import sys

from pyfooda.build.align import apply_alignment
from pyfooda.build.download_usda import ensure_fooddata
from pyfooda.build.export_web import export_web_data
from pyfooda.build.recompute import recompute


def rebuild() -> int:
    try:
        ensure_fooddata()
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    apply_alignment()
    recompute()
    export_web_data()
    return 0


def main() -> int:
    return rebuild()


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared paths for build tooling."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
REPO_ROOT = PACKAGE_ROOT.parent

USDA_CSV = DATA_DIR / "fooddata.csv"
USDA_GZ = DATA_DIR / "fooddata.csv.gz"
USDA_SHA256 = DATA_DIR / "fooddata.csv.sha256"
VOCAB_JSON = DATA_DIR / "epicure_vocabulary.json"
NUTRIENTS_CSV = DATA_DIR / "nutrients.csv"
INGREDIENTS_CSV = DATA_DIR / "ingredients.csv"
META_JSON = DATA_DIR / "ingredients_meta.json"
VERSION_FILE = PACKAGE_ROOT / "VERSION"
WEB_JSON = REPO_ROOT / "docs" / "data" / "ingredients.json"

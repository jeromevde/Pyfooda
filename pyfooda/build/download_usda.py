#!/usr/bin/env python3
"""Decompress bundled USDA fooddata for local database rebuilds."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

from pyfooda.build.paths import USDA_CSV, USDA_GZ, USDA_SHA256


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_fooddata(
    output: Path = USDA_CSV,
    *,
    gz_path: Path = USDA_GZ,
    sha_path: Path = USDA_SHA256,
    force: bool = False,
) -> Path:
    """Ensure fooddata.csv exists, decompressing the bundled .gz if needed."""
    if output.exists() and not force:
        return output

    if not gz_path.exists():
        raise FileNotFoundError(
            f"Missing {gz_path}. Clone the full repo — USDA data ships as a compressed bundle."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Decompressing {gz_path.name} → {output.name}")
    with gzip.open(gz_path, "rb") as src, open(output, "wb") as dst:
        shutil.copyfileobj(src, dst)

    if sha_path.exists():
        expected = sha_path.read_text().strip().split()[0]
        actual = _sha256(output)
        if actual != expected:
            output.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {output.name}: expected {expected}, got {actual}")

    print(f"Ready: {output} ({output.stat().st_size / 1_000_000:.1f} MB)")
    return output


def main() -> int:
    p = argparse.ArgumentParser(description="Decompress bundled USDA fooddata.csv")
    p.add_argument("--output", type=Path, default=USDA_CSV)
    p.add_argument("--force", action="store_true", help="Re-decompress even if CSV exists")
    args = p.parse_args()
    try:
        ensure_fooddata(args.output, force=args.force)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

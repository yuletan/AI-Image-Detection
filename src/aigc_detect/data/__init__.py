"""Dataset manifest, splits, dedupe, leak check (Day 1: A1 builds here)."""

from __future__ import annotations

import csv
from pathlib import Path

MANIFEST_COLUMNS = ["image_path", "label", "source", "generator", "split"]
VALID_SPLITS = {"train", "val", "test", "heldout", "demo"}


def write_manifest(rows: list[dict], out: Path) -> Path:
    """Write manifest.csv with the CONTRACTS.md schema. Day 0: minimal writer."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})
    return out


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))

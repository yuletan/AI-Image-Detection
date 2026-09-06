"""Pack a Kaggle-uploadable payload: manifest-referenced images + manifest.csv.

Reads data/processed/manifest.csv, copies every referenced file (preserving
repo-relative paths) plus the manifest itself into --out-dir, then zips it.
Upload the zip as one private Kaggle dataset; the training notebook unzips
so that image_path values resolve again.

Usage:
    python scripts/make_kaggle_payload.py --out data/kaggle_payload
    # -> data/kaggle_payload.zip (+ data/kaggle_payload/ tree)
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "processed" / "manifest.csv"


def build_payload(manifest: Path, out_dir: Path, root: Path = REPO_ROOT) -> dict:
    with manifest.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["image_path", "label", "source", "generator", "split"]:
            raise ValueError(f"bad manifest columns: {reader.fieldnames}")
        rows = list(reader)
    missing = [r["image_path"] for r in rows if not (root / r["image_path"]).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} manifest files missing (first: {missing[:5]})")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, r in enumerate(rows, 1):
        src, dst = root / r["image_path"], out_dir / r["image_path"]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        total += dst.stat().st_size
        if i % 5000 == 0:
            print(f"[payload] {i}/{len(rows)} copied ({total / 2**30:.2f} GB)", flush=True)
    shutil.copyfile(manifest, out_dir / "manifest.csv")
    print(f"[payload] {len(rows)} files, {total / 2**30:.2f} GB -> {out_dir}", flush=True)
    return {"n_files": len(rows), "bytes": total, "out_dir": str(out_dir)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pack manifest payload for Kaggle upload")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "kaggle_payload")
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip via shutil")
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    a = ap.parse_args(argv)
    info = build_payload(a.manifest, a.out, a.root)
    if a.zip:
        archive = shutil.make_archive(str(a.out), "zip", root_dir=a.out.parent,
                                      base_dir=a.out.name)
        print(f"[payload] archive: {archive} "
              f"({Path(archive).stat().st_size / 2**30:.2f} GB)", flush=True)
    print(f"[payload] done: {info}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

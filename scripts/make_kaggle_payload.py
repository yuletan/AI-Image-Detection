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
import time
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


def write_zip(out_dir: Path, archive: Path) -> Path:
    """Zip the payload dir with per-file progress (shutil is silent for minutes).

    JPEGs don't recompress, so ZIP_STORED is ~as small and much faster.
    Writes to a temp name + atomic rename: a killed run never leaves a
    fake-complete zip behind.
    """
    import zipfile

    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    tmp = archive.with_suffix(".zip.partial")
    if tmp.exists():
        tmp.unlink()  # leftover from a killed run
    total_in, t0 = 0, time.perf_counter()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for i, p in enumerate(files, 1):
            z.write(p, p.relative_to(out_dir.parent).as_posix())
            total_in += p.stat().st_size
            if i % 2000 == 0 or i == len(files):
                el = time.perf_counter() - t0
                print(f"[payload] zip {i}/{len(files)} "
                      f"({total_in / 2**30:.2f} GB, {el:.0f}s)", flush=True)
    print(f"[payload] zip wrote {tmp.stat().st_size / 2**30:.2f} GB", flush=True)
    tmp.replace(archive)  # atomic on the same volume
    return archive


def verify_archive(archive: Path, min_files: int) -> int:
    """Open the zip (reads its central directory) and sanity-check it.

    Catches truncated/interleaved writes immediately instead of on Kaggle.
    Returns the entry count. Raises RuntimeError on any problem.
    """
    import zipfile

    try:
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
    except Exception as e:
        raise RuntimeError(f"archive {archive} is not a valid zip: {e}") from e
    if len(names) < min_files:
        raise RuntimeError(f"archive {archive}: only {len(names)} entries, "
                           f"expected >= {min_files}")
    if not any(n.endswith("manifest.csv") for n in names):
        raise RuntimeError(f"archive {archive}: manifest.csv missing")
    return len(names)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pack manifest payload for Kaggle upload")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "kaggle_payload")
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip via shutil")
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    a = ap.parse_args(argv)
    info = build_payload(a.manifest, a.out, a.root)
    if a.zip:
        archive = write_zip(a.out, a.out.with_suffix(".zip"))
        print(f"[payload] archive: {archive} "
              f"({archive.stat().st_size / 2**30:.2f} GB)", flush=True)
        n = verify_archive(archive, info["n_files"] + 1)
        print(f"[payload] archive OK: {n} entries, central directory valid",
              flush=True)
    print(f"[payload] done: {info}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Day 0 dataset bootstrap.

Downloads are the #1 hidden time sink — start NOW, keep GPU busy later.

Sources (from the Day 0 board):
- WildFake: ModelScope (translate page), take a SUBSET first (~20k balanced).
- SID_Set: Hugging Face, streaming subset (good first-24h fallback if ModelScope is slow).
- CIFAKE: sanity only (32x32, single generator — never for robustness claims).
- Demo-only quarantine (NEVER in train):
    data/demo_benchmark/real <- COCO val2017
    data/demo_benchmark/fake <- DALL-E Advanced (or other DALL-E photoreal)

Usage:
    python scripts/download_datasets.py --only cifake --out data/raw
    python scripts/download_datasets.py --only sid_set --out data/raw --limit 2000
    python scripts/download_datasets.py --all --out data/raw
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

CIFAKE_URL = "https://github.com/junyanz/CIFAKE/releases/download/v1.0/CIFAKE.zip"
SID_SET_HF = "HaoxuanLi/SID_Set"  # use with datasets.load_dataset(..., streaming=True)
WILDFAKE_MODELSCOPE = "WildFake/WildFake"  # modelscope: translate page; mirror if blocked


def download_cifake(out: Path, limit: int | None = None) -> Path:
    dest = out / "cifake"
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "CIFAKE.zip"
    if not any(dest.iterdir()):
        print(f"[cifake] downloading {CIFAKE_URL} ...")
        urllib.request.urlretrieve(CIFAKE_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        zpath.unlink(missing_ok=True)
    print(f"[cifake] ready at {dest} (sanity only — do NOT train robustness here)")
    return dest


def fetch_sid_set(out: Path, limit: int = 2000) -> Path:
    """Streaming subset via HF datasets (no full download on Day 0)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[sid_set] pip install datasets to stream; writing instructions only.")
        (out / "sid_set_STREAMING.txt").write_text(
            f"Use: load_dataset({SID_SET_HF!r}, streaming=True).take({limit})\n",
            encoding="utf-8",
        )
        return out
    ds = load_dataset(SID_SET_HF, streaming=True, split="train")
    dest = out / "sid_set_subset"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for _ex in ds:
        if n >= limit:
            break
        # Day 1 A1 converts ex -> {image, label, generator}; Day 0 just counts.
        n += 1
    print(f"[sid_set] streamed {n} examples (subset manifest: Day 1 A1).")
    return dest


def fetch_wildfake(out: Path, limit: int = 2000) -> Path:
    """WildFake lives on ModelScope. Try modelscope SDK; else print manual steps."""
    dest = out / "wildfake_subset"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope.hub.snapshot_download import snapshot_download  # type: ignore

        snapshot_download(WILDFAKE_MODELSCOPE, cache_dir=str(dest))
        print(f"[wildfake] snapshot at {dest}")
    except Exception as e:
        (dest / "MANUAL.txt").write_text(
            "WildFake is on ModelScope (translate page).\n"
            f"Repo: {WILDFAKE_MODELSCOPE}\n"
            f"Auto-download failed ({e}). Manual: install modelscope, "
            "snapshot_download, take a balanced ~20k subset (>=4 generators).\n"
            "If blocked/slow: use SID_Set streaming for the first 24h.\n",
            encoding="utf-8",
        )
        print(f"[wildfake] manual steps written to {dest / 'MANUAL.txt'}")
    return dest


def make_demo_quarantine(root: Path) -> Path:
    dest = root / "demo_benchmark"
    (dest / "real").mkdir(parents=True, exist_ok=True)
    (dest / "fake").mkdir(parents=True, exist_ok=True)
    (dest / "DO_NOT_TRAIN.txt").write_text(
        "QUARANTINE: COCO val2017 (real) vs DALL-E Advanced (fake).\n"
        "Demo-only benchmark. NEVER in train. Leak-checked with pHash.\n",
        encoding="utf-8",
    )
    return dest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument(
        "--only", choices=["cifake", "sid_set", "wildfake", "demo", "all"], default="all"
    )
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=2000)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    only = "all" if args.all else args.only
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if only in ("cifake", "all"):
        download_cifake(out)
    if only in ("sid_set", "all"):
        fetch_sid_set(out, limit=args.limit)
    if only in ("wildfake", "all"):
        fetch_wildfake(out, limit=args.limit)
    if only in ("demo", "all"):
        make_demo_quarantine(Path("data"))
    print("done. Next: Day 1 A1 builds manifest (20k/2k/4k + held-out).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

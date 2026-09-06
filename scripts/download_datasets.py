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
from pathlib import Path

CIFAKE_HF_CANDIDATES = [
    "yanbax/CIFAKE_autotrain_compatible",  # train-only, fewer files — try first
    "Hemg/cifake-real-and-ai-generated-synthetic-images",
    "dragonintelligence/CIFAKE-image-dataset",
    # NOTE: batgre/CIFAKE excluded — 19,996 file refs, hangs on resolve
]
CIFAKE_KAGGLE = "birdy654/cifake-real-and-ai-generated-synthetic-images"
SID_SET_HF = "saberzl/SID_Set"  # 240k rows; 0 real, 1 synth, 2 tampered
SID_SET_FALLBACK = "HaoxuanLi/SID_Set"
WILDFAKE_MODELSCOPE = (
    "hy2628982280/WildFake"  # https://modelscope.cn/datasets/hy2628982280/WildFake/summary
)
WILDFAKE_FALLBACK = "WildFake/WildFake"


def download_cifake(out: Path, limit: int | None = None) -> Path:
    """CIFAKE via HuggingFace (sanity only). Falls back to Kaggle instructions."""
    dest = out / "cifake"
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"[cifake] already present at {dest}")
        return dest
    try:
        from datasets import load_dataset
    except ImportError:
        (dest / "MANUAL.txt").write_text(
            "CIFAKE (sanity only, 32x32):\n"
            f"1) HF: load_dataset({CIFAKE_HF_CANDIDATES[0]!r})\n"
            f"2) Kaggle: kaggle datasets download -d {CIFAKE_KAGGLE} -p data/raw/cifake --unzip\n"
            "pip install datasets kagglehub to auto-fetch.\n",
            encoding="utf-8",
        )
        print(f"[cifake] pip install datasets for auto-fetch; manual at {dest / 'MANUAL.txt'}")
        return dest
    last_err = None
    for repo in CIFAKE_HF_CANDIDATES:
        try:
            print(f"[cifake] trying HF {repo} ...")
            ds = load_dataset(repo, split="train", streaming=True)
            n = 0
            cap = limit or 2000
            for ex in ds:
                if n >= cap:
                    break
                # Most mirrors have {image, label}; save a small subset for sanity tests
                img = ex.get("image") or ex.get("img")
                if img is not None:
                    try:
                        img.save(dest / f"{n:05d}_{ex.get('label', 'x')}.jpg")
                    except Exception:
                        pass
                n += 1
            print(f"[cifake] saved {n} sanity images to {dest} (full set not needed)")
            return dest
        except Exception as e:  # try next mirror
            last_err = e
            continue
    (dest / "MANUAL.txt").write_text(
        f"Auto-fetch failed ({last_err}).\nKaggle: kaggle datasets download -d {CIFAKE_KAGGLE}\n",
        encoding="utf-8",
    )
    print(f"[cifake] all HF mirrors failed; see {dest / 'MANUAL.txt'}")
    return dest


def fetch_sid_set(out: Path, limit: int = 2000) -> Path:
    """Streaming subset via HF datasets (no full download on Day 0).

    Source: saberzl/SID_Set (240k rows). Labels: 0 real, 1 full_synthetic, 2 tampered.
    Binary mapping for Day 1 A1: label==0 -> 0 (real), label in (1,2) -> 1 (fake).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[sid_set] pip install datasets to stream; writing instructions only.")
        (out / "sid_set_STREAMING.txt").write_text(
            f"Use: load_dataset({SID_SET_HF!r}, streaming=True).take({limit})\n"
            "Labels: 0 real, 1 full_synthetic->fake, 2 tampered->fake.\n",
            encoding="utf-8",
        )
        return out
    last_err = None
    for repo in (SID_SET_HF, SID_SET_FALLBACK):
        try:
            ds = load_dataset(repo, streaming=True, split="train")
            dest = out / "sid_set_subset"
            dest.mkdir(parents=True, exist_ok=True)
            n = 0
            for _ex in ds:
                if n >= limit:
                    break
                # Day 1 A1 converts ex -> {image, label, generator}; Day 0 just counts.
                n += 1
            print(f"[sid_set] {repo}: streamed {n} (subset manifest: Day 1 A1).")
            return dest
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"[sid_set] all mirrors failed: {last_err}")


def fetch_wildfake(out: Path, limit: int = 2000) -> Path:
    """WildFake lives on ModelScope CN. Try modelscope SDK; else print manual steps.

    Source: https://modelscope.cn/datasets/hy2628982280/WildFake/summary
    """
    dest = out / "wildfake_subset"
    dest.mkdir(parents=True, exist_ok=True)
    for repo in (WILDFAKE_MODELSCOPE, WILDFAKE_FALLBACK):
        try:
            from modelscope.hub.snapshot_download import snapshot_download  # type: ignore

            snapshot_download(repo, cache_dir=str(dest))
            print(f"[wildfake] {repo}: snapshot at {dest}")
            return dest
        except Exception as e:
            last_err = e
            continue
    (dest / "MANUAL.txt").write_text(
        "WildFake is on ModelScope CN (translate page).\n"
        "URL: https://modelscope.cn/datasets/hy2628982280/WildFake/summary\n"
        f"Repos tried: {WILDFAKE_MODELSCOPE}, {WILDFAKE_FALLBACK}\n"
        f"Auto-download failed ({last_err}). Manual: pip install modelscope, "
        "snapshot_download('hy2628982280/WildFake'), take balanced ~20k (>=4 generators).\n"
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
    failed = []
    for name, fn in [
        ("cifake", lambda: download_cifake(out, limit=args.limit)),
        ("sid_set", lambda: fetch_sid_set(out, limit=args.limit)),
        ("wildfake", lambda: fetch_wildfake(out, limit=args.limit)),
        ("demo", lambda: make_demo_quarantine(Path("data"))),
    ]:
        if only not in (name, "all"):
            continue
        try:
            fn()
        except Exception as e:
            print(f"[{name}] FAILED: {e} — continuing with next dataset")
            failed.append(name)
    print("done. Next: Day 1 A1 builds manifest (20k/2k/4k + held-out).")
    if failed:
        print(f"failed: {failed} (see data/raw/*/MANUAL.txt)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

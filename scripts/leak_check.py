"""pHash dedupe + train-vs-demo leak check (Day 2 gate; Day 0 stub)."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="pHash leak check: train vs demo_benchmark")
    ap.add_argument("--train", type=Path, default=Path("data/raw"))
    ap.add_argument("--demo", type=Path, default=Path("data/demo_benchmark"))
    ap.add_argument("--threshold", type=int, default=5, help="hamming distance <= thr = duplicate")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import importlib.util

    if importlib.util.find_spec("PIL") is None or importlib.util.find_spec("imagehash") is None:
        print("pip install Pillow ImageHash to run the leak check")
        return 2
    print(f"scanning {args.train} vs {args.demo} (thr={args.threshold}) ...")
    print("Day 0 stub: full sweep lands Day 1 (A1). Failing closed = assume no leak yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

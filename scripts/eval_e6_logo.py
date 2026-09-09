"""Day-2 E6: leave-one-generator-out sweep on cached clean features.

For each train generator G: train the v0-recipe linear probe
(LogisticRegression C=1.0) on train_clean minus G, then report:
  - acc@0.5 on G's test-clean slice (single-class -> accuracy only),
  - acc@0.5 of the same probe on every other test generator slice,
  - val-clean AUROC sanity (full val, both classes),
  - DDIM held-out acc@0.5.
Plus a full-train baseline probe (same seed) for apples-to-apples deltas.

CPU-only. Test slices are single-class, so per-slice AUROC is undefined
by construction -- accuracy at thr 0.5 is the metric (see hypothesis.md).

Usage:
  python scripts/eval_e6_logo.py --cache-dir data/kaggle_cache/cache
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "kaggle_cache" / "cache"
DEFAULT_OUT = REPO_ROOT / "results"

THR = 0.5


def load_with_meta(npy: Path):
    import numpy as np

    from aigc_detect.models.probe import load_cache

    idx = npy.parent / (npy.name + ".index.csv")
    X, y = load_cache(npy, idx)
    with idx.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gens = [r["generator"] for r in rows]
    assert len(gens) == len(y) == len(X)
    return np.asarray(X, dtype=np.float64), y, gens


def main() -> None:
    ap = argparse.ArgumentParser(description="E6 leave-one-generator-out (CPU-only)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from aigc_detect.eval import (
        _roc_auc,  # private but pinned; same impl as summarize
        accuracy_at_threshold,
    )
    from aigc_detect.models.probe import LinearProbe

    t0 = time.perf_counter()
    cd = args.cache_dir
    Xtr, ytr, gtr = load_with_meta(cd / "train_clean_None.npy")
    Xva, yva, _ = load_with_meta(cd / "val_clean_None.npy")
    Xte, yte, gte = load_with_meta(cd / "test_clean_None.npy")
    Xho, yho, _ = load_with_meta(cd / "heldout_clean_None.npy")
    train_gens = sorted(set(gtr))
    print(f"[e6] train gens: {train_gens}", flush=True)

    def fit(mask) -> LinearProbe:
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
        clf.fit(Xtr[mask], [y for y, m in zip(ytr, mask, strict=False) if m])
        return LinearProbe(clf.coef_.ravel(), float(clf.intercept_[0]), thr=THR)

    def acc(X, y, probe) -> float:
        return accuracy_at_threshold(y, probe.predict_proba(np.asarray(X)), THR)

    results = {}
    # full-train baseline first
    full = fit(np.ones(len(ytr), dtype=bool))
    base = {
        "val_auroc": _roc_auc(yva, full.predict_proba(Xva)),
        "heldout_acc": acc(Xho, yho, full),
        "slices": {},
    }
    for g in sorted(set(gte)):
        m = np.array([gg == g for gg in gte])
        base["slices"][g] = acc(Xte[m], [y for y, k in zip(yte, m, strict=False) if k], full)
    results["full_train"] = base
    print(
        f"[e6] full: val_auroc={base['val_auroc']:.4f} "
        f"heldout={base['heldout_acc']:.4f} "
        + " ".join(f"{g}={a:.4f}" for g, a in base["slices"].items()),
        flush=True,
    )

    for g in train_gens:
        mask = np.array([gg != g for gg in gtr])
        probe = fit(mask)
        left_m = np.array([gg == g for gg in gte])
        entry = {
            "train_n": int(mask.sum()),
            "val_auroc": _roc_auc(yva, probe.predict_proba(Xva)),
            "heldout_acc": acc(Xho, yho, probe),
            "leftout_slice_acc": None,
            "leftout_slice_n": int(left_m.sum()),
            "other_slices": {},
        }
        if left_m.any():
            entry["leftout_slice_acc"] = acc(
                Xte[left_m],
                [y for y, k in zip(yte, left_m, strict=False) if k],
                probe,
            )
        for h in sorted(set(gte)):
            if h == g:
                continue
            m = np.array([gg == h for gg in gte])
            entry["other_slices"][h] = acc(
                Xte[m], [y for y, k in zip(yte, m, strict=False) if k], probe
            )
        results[f"leave_{g}"] = entry
        print(
            f"[e6] leave_{g}: train_n={entry['train_n']} "
            f"val_auroc={entry['val_auroc']:.4f} "
            f"leftout={entry['leftout_slice_acc']} (n={entry['leftout_slice_n']}) "
            f"heldout={entry['heldout_acc']:.4f}",
            flush=True,
        )

    out = args.out / "e6_logo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[e6] done in {time.perf_counter() - t0:.1f}s -> {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

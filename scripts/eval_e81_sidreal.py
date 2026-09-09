"""Day-3 E8.1: sid_real real-vs-real probe (CPU seconds).

Question: are the SID "real" images actually real? Train a linear probe
(LogisticRegression C=1.0) on CLIP features to tell appart the two *real*
families the model was trained on: celebahq+ffhq vs sid_real.

Decision rule (pre-registered in EXPERIMENTS.md):
  val/test AUROC ~= 0.5  -> sid_real is indistinguishable from genuine reals
                            -> random label noise (they ARE real, E6 collapse is
                            scatter, not systematic).
  AUROC >= 0.9           -> sid_real sits in a distinct feature region
                            -> systematic distribution (recapture / screen / fake)
                            -> quarantine before v-final.

Only real rows (label==0) are used; fakes are excluded by construction so the
signature is purely "is sid_real drawn from the same real manifold".

Usage:
  python scripts/eval_e81_sidreal.py --cache-dir data/kaggle_cache/cache
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
GENUINE = ("celebahq", "ffhq")
SUSPECT = "sid_real"


def load_reals(npy: Path):
    import numpy as np

    idx = npy.parent / (npy.name + ".index.csv")
    X = np.load(npy)
    with idx.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(X), f"{npy}: {len(X)} rows vs {len(rows)} labels"
    Xr, yr, gens, paths = [], [], [], []
    for feat, r in zip(X, rows, strict=False):
        if r["label"] != "0":
            continue
        gen = r["generator"]
        if gen in GENUINE or gen == SUSPECT:
            Xr.append(feat)
            yr.append(1 if gen == SUSPECT else 0)
            gens.append(gen)
            paths.append(r["image_path"])
    return np.asarray(Xr, dtype=np.float64), yr, gens, paths


def cv_auc(X, y, C: float, seed: int, folds: int) -> tuple[list[float], float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    from aigc_detect.eval import _roc_auc

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        clf.fit(X[tr], [y[i] for i in tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(_roc_auc([y[i] for i in te], p.tolist()))
    import numpy as np

    return aucs, float(np.mean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser(description="E8.1 sid_real real-vs-real probe (CPU-only)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from aigc_detect.eval import _roc_auc

    t0 = time.perf_counter()
    X, y, gens, paths = load_reals(args.cache_dir / "train_clean_None.npy")
    n_suspect = sum(y)
    n_genuine = len(y) - n_suspect
    assert n_suspect and n_genuine, f"no rows: suspect={n_suspect} genuine={n_genuine}"

    # honest train/test split on the reals only
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    clf = LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed)
    clf.fit(Xtr, ytr)
    ptr = clf.predict_proba(Xtr)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    auc_tr = _roc_auc(ytr, ptr.tolist())
    auc_te = _roc_auc(yte, pte.tolist())

    # 5-fold CV on the full reals for stability
    fold_aucs, auc_cv = cv_auc(X, y, args.C, args.seed, args.cv_folds)
    fold_std = float(np.std(fold_aucs))

    # odds direction: what fraction of sid_real get a suspect-score >= 0.5
    from aigc_detect.eval import accuracy_at_threshold

    acc = accuracy_at_threshold(yte, pte.tolist(), 0.5)

    # interpret
    if auc_te < 0.6:
        verdict = "NOISE: sid_real sits inside the genuine-real manifold (AUROC ~0.5)"
    elif auc_te < 0.9:
        verdict = f"UNRESOLVED: partial separation ({auc_te:.3f}) - needs E8.2/E8.4 before a call"
    else:
        verdict = "SYSTEMATIC: sid_real is a distinct distribution vs genuine reals (AUROC >= 0.9)"

    result = {
        "train_n": int(len(ytr)),
        "test_n": int(len(yte)),
        "n_genuine": n_genuine,
        "genuine_families": list(GENUINE),
        "n_suspect": n_suspect,
        "suspect_family": SUSPECT,
        "C": args.C,
        "seed": args.seed,
        "test_split": args.test_size,
        "auroc_train": auc_tr,
        "auroc_test": auc_te,
        "acc_test_at_0.5": acc,
        "cv_folds": args.cv_folds,
        "auroc_cv_mean": auc_cv,
        "auroc_cv_std": fold_std,
        "auroc_cv": fold_aucs,
        "verdict": verdict,
    }
    out = args.out / "e8_sidreal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e8_1_real_vs_real.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[e8.1] genuine={n_genuine} (celebahq+ffhq) suspect={n_suspect} (sid_real)\n"
        f"[e8.1] test AUROC={auc_te:.4f} | CV {args.cv_folds}-fold AUROC="
        f"{auc_cv:.4f}+-{fold_std:.4f} | acc@0.5={acc:.4f}\n"
        f"[e8.1] VERDICT: {verdict}\n"
        f"[e8.1] done in {time.perf_counter() - t0:.1f}s -> {out / 'e8_1_real_vs_real.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

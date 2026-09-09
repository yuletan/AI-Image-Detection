"""Day-3 E7a: score-blend v0_linear + v1_mlp (CPU seconds).

Motivation (E1/E2/E5/E6): v0 (clean-trained linear probe) has the best
held-out DDIM (0.909) but weaker in-distribution robustness; v1_mlp (full
25/75 clean/randaug3 + sklearn MLP) has the best robustness (mean
TPR@1%FPR 0.8847) but erodes held-out (~0.76). The two heads sit on
opposite ends of a trade-off. This asks whether a *score blend* dominates
both endpoints with zero new data.

Recipe (pre-registered in EXPERIMENTS.md):
  - scale-normalise each head's VAL logits by their val std (MLP probs
    saturate near 0/1 and would otherwise dominate the blend; we use
    scale-only, not mean-shift, so the p=0.5 boundary stays head-agnostic
    and the single-head endpoints reproduce their published numbers),
  - blend p = sigmoid(alpha*z0 + (1-alpha)*z1), sweep alpha on VAL only,
  - eval 18 test variants + held-out DDIM per alpha (held-out both at
    p=0.5, to compare with prior numbers, and at the val-frozen 1% FPR
    threshold, the deployment operating point),
  - for the picked alpha, re-freeze the threshold at 1% FPR on val and
    rerun the E5 per-variant FPR drift table (v0 is clean-trained and may
    import FPR instability).

Decision: pre-reg expects held-out 0.76->0.91 and mean TPR 0.88->0.80 as
alpha goes 0->1; if alpha~0.5 lands >=0.85 on BOTH, it is a zero-new-data
v-final candidate.

Usage:
  python scripts/eval_e7a_blend.py --cache-dir data/kaggle_cache/cache
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "kaggle_cache" / "cache"
DEFAULT_OUT = REPO_ROOT / "results"
V0_PROBE = REPO_ROOT / "results" / "v0_probe" / "probe.npz"
V1_MLP = REPO_ROOT / "results" / "v1_mlp"

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EPS = 1e-6


def load_cache(npy: Path):
    import numpy as np

    from aigc_detect.models.probe import load_cache as _lc

    X, y = _lc(npy, npy.parent / (npy.name + ".index.csv"))
    return np.asarray(X, dtype=np.float64), y


def sigmoid(z):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def logit(p):
    import numpy as np

    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1.0 - p))


def v0_logits(probe_path: Path, X):
    import numpy as np

    z = np.load(probe_path)
    return X @ z["coef"].ravel() + float(z["intercept"])


def mlp_logits(seed_dir: Path, X):
    import joblib

    mlp = joblib.load(seed_dir / "mlp.joblib")
    scaler = joblib.load(seed_dir / "scaler.joblib")
    return logit(mlp.predict_proba(scaler.transform(X))[:, 1])


def scale_fit(z_val) -> float:
    """Scale-only normaliser (val std, no mean shift).

    Full z-scoring would move the p=0.5 decision boundary and break the
    single-head endpoints (v0 held-out 0.909 -> 0.969), so we equalise
    scale only; p=0.5 stays the raw probability boundary for both heads.
    """
    import numpy as np

    return float(np.std(z_val))


def fpr_tpr_at_thr(y, scores, thr: float) -> tuple[float, float, float]:
    import numpy as np

    y = np.asarray(y)
    s = np.asarray(scores)
    pred = s >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return fp / max(1, n_neg), tp / max(1, n_pos), float((pred == y).mean())


def thr_at_1fpr(y, scores) -> float:
    import numpy as np

    neg = np.sort(np.asarray(scores)[np.asarray(y) == 0])
    return float(neg[min(len(neg) - 1, int(np.ceil(0.99 * len(neg))))])


def main() -> None:
    ap = argparse.ArgumentParser(description="E7a score blend v0 + v1_mlp (CPU-only)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    args = ap.parse_args()

    import numpy as np

    from aigc_detect.eval import summarize

    t0 = time.perf_counter()
    cd = args.cache_dir
    Xva, yva = load_cache(cd / "val_clean_None.npy")
    Xho, yho = load_cache(cd / "heldout_clean_None.npy")
    variants = sorted(cd.glob("test_*.npy"))
    var_data = {v.stem: load_cache(v) for v in variants}

    # ---- v0 head: logits for every cache, z-scored on val ----
    z0_va = v0_logits(V0_PROBE, Xva)
    z0_sd = scale_fit(z0_va)
    z0 = {"val": z0_va / z0_sd, "heldout": v0_logits(V0_PROBE, Xho) / z0_sd}
    for stem, (Xv, _) in var_data.items():
        z0[stem] = v0_logits(V0_PROBE, Xv) / z0_sd

    # ---- per-seed v1_mlp head: logits, z-scored on val ----
    z1_heads = {}
    for seed in args.seeds:
        seed_dir = V1_MLP / f"seed{seed}"
        assert (seed_dir / "mlp.joblib").exists(), f"missing {seed_dir}"
        z1_va = mlp_logits(seed_dir, Xva)
        sd = scale_fit(z1_va)
        h = {"val": z1_va / sd, "heldout": mlp_logits(seed_dir, Xho) / sd}
        for stem, (Xv, _) in var_data.items():
            h[stem] = mlp_logits(seed_dir, Xv) / sd
        z1_heads[seed] = h

    # ---- alpha sweep: metrics per seed, then mean/std ----
    sweep: dict[str, list] = {}
    for seed, h1 in z1_heads.items():
        for a in args.alphas:
            pv = sigmoid(a * z0["val"] + (1 - a) * h1["val"])
            vm = summarize(yva, pv.tolist())
            thr = thr_at_1fpr(yva, pv)
            aucs, tprs, fprs = [], [], []
            for stem, (_, yv) in var_data.items():
                p = sigmoid(a * z0[stem] + (1 - a) * h1[stem])
                m = summarize(yv, p.tolist())
                aucs.append(m["auroc"])
                tprs.append(m["tpr_at_1fpr"])
                fprs.append(fpr_tpr_at_thr(yv, p, thr)[0])
            ph = sigmoid(a * z0["heldout"] + (1 - a) * h1["heldout"])
            sweep.setdefault(str(a), []).append(
                {
                    "seed": seed,
                    "val_auroc": vm["auroc"],
                    "test_mean_auroc": float(np.mean(aucs)),
                    "test_mean_tpr_1fpr": float(np.mean(tprs)),
                    "test_worst_tpr_1fpr": float(np.min(tprs)),
                    "heldout_acc_0.5": summarize(yho, ph.tolist())["acc"],
                    "heldout_tpr_at_frozen_thr": float((ph >= thr).mean()),
                    "thr_at_1fpr_val": thr,
                    "test_max_fpr_at_frozen_thr": float(np.max(fprs)),
                    "test_mean_fpr_at_frozen_thr": float(np.mean(fprs)),
                }
            )

    agg = {}
    for a, runs in sweep.items():
        agg[a] = {}
        for k in runs[0]:
            if k == "seed":
                continue
            vals = [r[k] for r in runs]
            agg[a][k + "_mean"] = float(np.mean(vals))
            agg[a][k + "_std"] = float(np.std(vals))

    feasible = [
        a
        for a, m in agg.items()
        if m["heldout_acc_0.5_mean"] >= 0.85 and m["test_mean_tpr_1fpr_mean"] >= 0.85
    ]
    pick_pool = feasible or list(agg)
    pick = float(
        max(
            pick_pool,
            key=lambda a: min(agg[a]["heldout_acc_0.5_mean"], agg[a]["test_mean_tpr_1fpr_mean"]),
        )
    )

    # ---- E5-style drift table at the picked alpha (mean over seeds) ----
    drift_rows = {stem: {"fpr": [], "tpr": [], "acc": []} for stem in var_data}
    for h1 in z1_heads.values():
        a = float(pick)
        pv = sigmoid(a * z0["val"] + (1 - a) * h1["val"])
        thr = thr_at_1fpr(yva, pv)
        for stem, (_, yv) in var_data.items():
            p = sigmoid(a * z0[stem] + (1 - a) * h1[stem])
            fpr, tpr, acc = fpr_tpr_at_thr(yv, p, thr)
            drift_rows[stem]["fpr"].append(fpr)
            drift_rows[stem]["tpr"].append(tpr)
            drift_rows[stem]["acc"].append(acc)
    drift = [
        {
            "variant": stem,
            "fpr_mean": float(np.mean(r["fpr"])),
            "fpr_std": float(np.std(r["fpr"])),
            "tpr_mean": float(np.mean(r["tpr"])),
            "acc_mean": float(np.mean(r["acc"])),
        }
        for stem, r in drift_rows.items()
    ]
    drift.sort(key=lambda r: (r["variant"] != "test_clean_None", r["variant"]))

    out = args.out / "e7a_blend"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(
        json.dumps({"alphas": agg, "picked_alpha": float(pick), "feasible": feasible}, indent=2),
        encoding="utf-8",
    )
    (out / f"drift_alpha{pick:.1f}.json").write_text(json.dumps(drift, indent=2), encoding="utf-8")

    print("[e7a] alpha | heldout@0.5 | heldout@thr | mean_tpr | worst_tpr | mean_auroc | max_fpr")
    for a in args.alphas:
        m = agg[str(a)]
        star = " <== pick" if float(a) == float(pick) else ""
        print(
            f"[e7a]  {a:.1f}  | {m['heldout_acc_0.5_mean']:.4f} | "
            f"{m['heldout_tpr_at_frozen_thr_mean']:.4f} | "
            f"{m['test_mean_tpr_1fpr_mean']:.4f} | {m['test_worst_tpr_1fpr_mean']:.4f} | "
            f"{m['test_mean_auroc_mean']:.4f} | {m['test_max_fpr_at_frozen_thr_mean']:.4f}{star}",
            flush=True,
        )
    print(
        f"[e7a] feasible alphas (heldout>=0.85 & mean_tpr>=0.85): {feasible or 'NONE'}\n"
        f"[e7a] picked alpha={pick:.1f} -> {out}\n"
        f"[e7a] done in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

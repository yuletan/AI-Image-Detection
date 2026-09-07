"""Day-2 E4/E5 on cached features (CPU-only, no GPU).

E4 (TTA + abstain): for the 5 TTA views pinned in configs/train.yaml
(clean, jpeg70, resize0.5, centercrop80, flip) score the same test-clean
images under each view, average -> TTA-mean score, std -> uncertainty.
Reports single-clean vs TTA-mean metrics plus an abstain sweep: flag the
top-q most-uncertain images, report accuracy-of-the-rest.

E5 (threshold drift + temperature scaling): pick the operating threshold
at 1% FPR on clean val (99th percentile of negative scores), freeze it,
report per-variant FPR/TPR/acc drift on all test caches. Fit a 1-param
temperature T on val NLL; report ECE before/after (scaling is monotonic
so ranking metrics are unchanged -- only calibration moves).

Covers configs with sklearn-MLP heads (results/{cfg}/seed{N}/mlp.joblib
+ scaler.joblib); defaults: v1_mlp + v1b_mlp, seeds 0/1/2.

Usage:
  python scripts/eval_e4_e5.py --cache-dir data/kaggle_cache/cache
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "kaggle_cache" / "cache"
DEFAULT_OUT = REPO_ROOT / "results"

# (cache stem suffix after "test_") for the 5 pinned TTA views.
TTA_VIEWS = ["clean_None", "jpeg_70", "resize_0.5", "centercrop_80", "flip_h"]
ABSTAIN_QS = [0.0, 0.05, 0.10, 0.20]
T_GRID = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
    1.10,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
    4.00,
]
EPS = 1e-6


def load_model(cfg_dir: Path):
    """(proba_fn) for a seed dir holding mlp.joblib + scaler.joblib."""
    import joblib
    import numpy as np

    mlp = joblib.load(cfg_dir / "mlp.joblib")
    scaler = joblib.load(cfg_dir / "scaler.joblib")

    def proba(X) -> list[float]:
        Xs = scaler.transform(np.asarray(X, dtype=np.float64))
        return mlp.predict_proba(Xs)[:, 1].tolist()

    return proba


def load_npy_labels(npy: Path):
    import csv

    import numpy as np

    from aigc_detect.models.probe import load_cache

    X, y = load_cache(npy, npy.parent / (npy.name + ".index.csv"))
    with (npy.parent / (npy.name + ".index.csv")).open(encoding="utf-8") as f:
        paths = [r["image_path"] for r in csv.DictReader(f)]
    return np.asarray(X, dtype=np.float64), y, paths


def fpr_tpr_at_thr(y, scores, thr: float) -> tuple[float, float, float]:
    import numpy as np

    y = np.asarray(y)
    s = np.asarray(scores)
    pred = s >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    fpr = fp / max(1, n_neg)
    tpr = tp / max(1, n_pos)
    acc = (pred == y).mean().item()
    return fpr, tpr, acc


def nll(y, scores) -> float:
    import numpy as np

    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(scores, dtype=np.float64), EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def run_e4(proba, cache_dir: Path) -> dict:
    """TTA-mean vs single-clean on test-clean images + abstain sweep."""
    import numpy as np

    from aigc_detect.eval import summarize

    views = [(f"test_{v}.npy") for v in TTA_VIEWS]
    Xs, y0, paths0 = load_npy_labels(cache_dir / views[0])
    Ps = [np.asarray(proba(Xs), dtype=np.float64)]
    for v in views[1:]:
        Xv, yv, pv = load_npy_labels(cache_dir / v)
        assert yv == y0 and pv == paths0, f"row order mismatch in {v}"
        Ps.append(np.asarray(proba(Xv), dtype=np.float64))
    P = np.stack(Ps)  # (5, n)
    mean, std = P.mean(axis=0), P.std(axis=0)
    single = summarize(y0, Ps[0].tolist())
    tta = summarize(y0, mean.tolist())
    sweep = []
    for q in ABSTAIN_QS:
        if q == 0.0:
            kept = np.ones(len(y0), dtype=bool)
        else:
            kept = std <= np.quantile(std, 1 - q)
        yr = [y for y, k in zip(y0, kept, strict=False) if k]
        sr = [s for s, k in zip(mean.tolist(), kept, strict=False) if k]
        m = summarize(yr, sr) if yr and sum(yr) not in (0, len(yr)) else {"acc": 0.0, "n": 0}
        sweep.append(
            {
                "flagged_q": q,
                "kept_n": int(kept.sum()),
                "acc_rest": m["acc"],
                "mean_std_rest": float(std[kept].mean()) if kept.any() else 0.0,
                "mean_std_flagged": float(std[~kept].mean()) if (~kept).any() else 0.0,
            }
        )
    return {
        "views": TTA_VIEWS,
        "n": len(y0),
        "single_clean": single,
        "tta_mean": tta,
        "d_auroc": tta["auroc"] - single["auroc"],
        "d_acc": tta["acc"] - single["acc"],
        "abstain_sweep": sweep,
    }


def run_e5(proba, cache_dir: Path) -> dict:
    """Fixed thr@1%FPR from val-clean -> per-variant drift + temp scaling."""
    import numpy as np

    from aigc_detect.eval import expected_calibration_error
    from aigc_detect.eval.evaluate import parse_cache_stem

    Xv, yv, _ = load_npy_labels(cache_dir / "val_clean_None.npy")
    sv = np.asarray(proba(Xv), dtype=np.float64)
    yv_arr = np.asarray(yv)
    neg = np.sort(sv[yv_arr == 0])
    thr = float(neg[min(len(neg) - 1, int(np.ceil(0.99 * len(neg))))])
    val_fpr, val_tpr, val_acc = fpr_tpr_at_thr(yv, sv, thr)
    # temperature on val NLL
    logit = np.log(np.clip(sv, EPS, 1 - EPS) / (1 - np.clip(sv, EPS, 1 - EPS) + EPS))
    nlls = [nll(yv, 1 / (1 + np.exp(-logit / t))) for t in T_GRID]
    t_star = T_GRID[int(np.argmin(nlls))]
    ece_before = expected_calibration_error(yv, sv.tolist())
    ece_after = expected_calibration_error(yv, (1 / (1 + np.exp(-logit / t_star))).tolist())
    variants = []
    for npy in sorted(cache_dir.glob("test_*.npy")):
        transform, param = parse_cache_stem(npy.stem, "test")
        Xt, yt, _ = load_npy_labels(npy)
        st = np.asarray(proba(Xt), dtype=np.float64)
        fpr, tpr, acc = fpr_tpr_at_thr(yt, st, thr)
        eb = expected_calibration_error(yt, st.tolist())
        lt = np.log(np.clip(st, EPS, 1 - EPS) / (1 - np.clip(st, EPS, 1 - EPS) + EPS))
        ea = expected_calibration_error(yt, (1 / (1 + np.exp(-lt / t_star))).tolist())
        variants.append(
            {
                "transform": transform,
                "param": param,
                "n": len(yt),
                "fpr_at_fixed_thr": fpr,
                "tpr_at_fixed_thr": tpr,
                "acc_at_fixed_thr": acc,
                "ece_before": eb,
                "ece_after": ea,
            }
        )
    variants.sort(key=lambda r: (r["transform"] != "clean", r["transform"], str(r["param"])))
    return {
        "thr_at_1fpr_val": thr,
        "val_fpr": val_fpr,
        "val_tpr": val_tpr,
        "val_acc": val_acc,
        "t_star": t_star,
        "val_nll_before": nll(yv, sv),
        "val_nll_after": float(min(nlls)),
        "val_ece_before": ece_before,
        "val_ece_after": ece_after,
        "variants": variants,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="E4 TTA + E5 threshold drift (CPU-only)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--configs", nargs="+", default=["v1_mlp", "v1b_mlp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    import numpy as np

    t0 = time.perf_counter()
    e4_agg: dict[str, list] = {}
    e5_agg: dict[str, list] = {}
    for cfg in args.configs:
        for seed in args.seeds:
            seed_dir = args.out / cfg / f"seed{seed}"
            assert (seed_dir / "mlp.joblib").exists(), f"missing {seed_dir}"
            proba = load_model(seed_dir)
            e4 = run_e4(proba, args.cache_dir)
            e5 = run_e5(proba, args.cache_dir)
            (args.out / "e4_tta" / cfg / f"seed{seed}").mkdir(parents=True, exist_ok=True)
            (args.out / "e5_threshold" / cfg / f"seed{seed}").mkdir(parents=True, exist_ok=True)
            (args.out / "e4_tta" / cfg / f"seed{seed}" / "results.json").write_text(
                json.dumps(e4, indent=2), encoding="utf-8"
            )
            (args.out / "e5_threshold" / cfg / f"seed{seed}" / "results.json").write_text(
                json.dumps(e5, indent=2), encoding="utf-8"
            )
            e4_agg.setdefault(cfg, []).append(e4)
            e5_agg.setdefault(cfg, []).append(e5)
            a10 = next(s for s in e4["abstain_sweep"] if s["flagged_q"] == 0.10)
            print(
                f"[{cfg} seed{seed}] E4 single={e4['single_clean']['auroc']:.4f} "
                f"tta={e4['tta_mean']['auroc']:.4f} (d={e4['d_auroc']:+.4f}) "
                f"acc_rest@10%={a10['acc_rest']:.4f} | E5 "
                f"thr={e5['thr_at_1fpr_val']:.4f} T*={e5['t_star']:.2f} "
                f"valECE {e5['val_ece_before']:.4f}->{e5['val_ece_after']:.4f}",
                flush=True,
            )
    # mean±std over seeds
    for name, agg, keys in (
        ("e4", e4_agg, ["d_auroc", "d_acc"]),
        ("e5", e5_agg, ["t_star", "val_ece_before", "val_ece_after"]),
    ):
        summ = {}
        for cfg, runs in agg.items():
            summ[cfg] = {}
            for k in keys:
                vs = [r[k] for r in runs]
                summ[cfg][k + "_mean"] = float(np.mean(vs))
                summ[cfg][k + "_std"] = float(np.std(vs))
            if name == "e5":
                # mean FPR drift per variant
                base = runs[0]["variants"]
                rows = []
                for i, r0 in enumerate(base):
                    fprs = [r["variants"][i]["fpr_at_fixed_thr"] for r in runs]
                    rows.append(
                        {
                            "transform": r0["transform"],
                            "param": r0["param"],
                            "fpr_mean": float(np.mean(fprs)),
                            "fpr_std": float(np.std(fprs)),
                        }
                    )
                summ[cfg]["fpr_drift"] = rows
        (args.out / f"{name}_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(f"[e4+e5] done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

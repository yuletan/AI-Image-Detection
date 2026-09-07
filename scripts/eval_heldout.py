"""Held-out DDIM check for v1 models (fake-only cache -> report acc@0.5).

v0 reference: acc 0.909 at thr 0.5 (EXPERIMENTS.md).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "kaggle_cache" / "cache"


def main() -> None:
    import argparse

    import joblib
    import numpy as np

    from aigc_detect.eval import accuracy_at_threshold
    from aigc_detect.models.probe import LinearProbe, load_cache

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["v1_linear", "v1_mlp", "v1b_linear"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    npy = CACHE / "heldout_clean_None.npy"
    X, y = load_cache(npy, npy.parent / (npy.name + ".index.csv"))
    print(f"heldout rows={len(y)} pos_rate={sum(y) / len(y):.3f} (expect 1.000, fake-only DDIM)")

    for cfg in args.configs:
        for seed in args.seeds:
            d = REPO_ROOT / "results" / cfg / f"seed{seed}"
            if not d.exists():
                print(f"{cfg} seed{seed}: MISSING, skipped")
                continue
            if (d / "probe.npz").exists():
                scores = LinearProbe.load(d / "probe.npz").predict_proba(X)
            else:
                mlp = joblib.load(d / "mlp.joblib")
                sc = joblib.load(d / "scaler.joblib")
                scores = mlp.predict_proba(sc.transform(np.asarray(X, dtype=np.float64)))[
                    :, 1
                ].tolist()
            acc = accuracy_at_threshold(y, scores, 0.5)
            print(f"{cfg} seed{seed}: heldout acc@0.5 = {acc:.4f} (v0: 0.9090)")


if __name__ == "__main__":
    main()

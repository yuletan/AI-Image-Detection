"""Day-2 E1/E2: v1 heads on clean + random-augmented caches.

Trains, for seeds [0, 1, 2]:
  - v1_linear: LogisticRegression on clean train + train_randaug3 (80k x 768)
  - v1_mlp:    sklearn MLP(512, relu, adam, lr=1e-3, 30 epochs) on the same
               features standardized with a train-fit StandardScaler.
               (Spec wants dropout 0.2 + AdamW; sklearn has neither on CPU,
               so L2 alpha=1e-4 substitutes for dropout. Noted in output meta.)

v0 sanity: linear probe on clean-only, seed 0 (expect val AUROC ~0.9933).

Each model is evaluated on val_clean + every test_*.npy in the cache dir
with the CONTRACTS §5 metric set; outputs mirror the v0 layout:
  results/v1_linear/seed{N}/{probe.npz, results.json, results.md}
  results/v1_mlp/seed{N}/{mlp.joblib, scaler.joblib, results.json, results.md}
plus results/v1_summary.json with mean±std over seeds per variant.

Usage:
  python scripts/train_v1.py --cache-dir data/kaggle_cache/cache
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "kaggle_cache" / "cache"
DEFAULT_OUT = REPO_ROOT / "results"

SEEDS = [0, 1, 2]
THR = 0.5


def load_concat(parts: list[tuple[Path, Path]]):
    import numpy as np

    from aigc_detect.models.probe import load_cache

    Xs, ys = [], []
    for npy, idx in parts:
        X, y = load_cache(npy, idx)
        Xs.append(np.asarray(X, dtype=np.float64))
        ys.extend(y)

    return np.concatenate(Xs), ys


def eval_all(model_proba, cache_dir: Path, thr: float = THR) -> list[dict]:
    """Evaluate predict-proba fn on val_clean + every test_*.npy cache."""
    from aigc_detect.eval import summarize
    from aigc_detect.eval.evaluate import parse_cache_stem
    from aigc_detect.models.probe import load_cache

    rows = []
    # val clean
    vnpy = cache_dir / "val_clean_None.npy"
    X, y = load_cache(vnpy, vnpy.parent / (vnpy.name + ".index.csv"))
    rows.append(
        {
            "split": "val",
            "transform": "clean",
            "param": None,
            **summarize(y, model_proba(X), thr=thr),
        }
    )
    for npy in sorted(cache_dir.glob("test_*.npy")):
        transform, param = parse_cache_stem(npy.stem, "test")
        X, y = load_cache(npy, npy.parent / (npy.name + ".index.csv"))
        rows.append(
            {
                "split": "test",
                "transform": transform,
                "param": param,
                **summarize(y, model_proba(X), thr=thr),
            }
        )
        print(
            f"[eval] {transform}={param} auroc={rows[-1]['auroc']:.4f} "
            f"acc={rows[-1]['acc']:.4f} tpr@1fpr={rows[-1]['tpr_at_1fpr']:.4f} "
            f"ece={rows[-1]['ece']:.4f} n={rows[-1]['n']}",
            flush=True,
        )
    return rows


def train_linear(X, y, seed: int):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def train_mlp(X, y, seed: int):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    mlp = MLPClassifier(
        hidden_layer_sizes=(512,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=30,
        random_state=seed,
        verbose=False,
    )
    mlp.fit(Xs, y)
    return mlp, scaler


def main() -> None:
    ap = argparse.ArgumentParser(description="Train v1 heads (E1/E2)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument(
        "--skip-v0", action="store_true", help="skip the v0 clean-only repro (already recorded)"
    )
    ap.add_argument(
        "--half-mix",
        action="store_true",
        help="50/50 mix: clean + first aug view per image (outputs to results/v1b_*)",
    )
    args = ap.parse_args()

    import joblib
    import numpy as np

    from aigc_detect.eval.evaluate import write_markdown
    from aigc_detect.models.probe import LinearProbe

    cd = args.cache_dir
    clean = (cd / "train_clean_None.npy", cd / "train_clean_None.npy.index.csv")
    aug = (cd / "train_randaug3_seed42.npy", cd / "train_randaug3_seed42.npy.index.csv")
    t0 = time.perf_counter()
    X_clean, y_clean = load_concat([clean])
    if args.half_mix:
        import csv

        import numpy as np

        Xa, idx_path = aug
        seen: set[str] = set()
        keep: list[int] = []
        with idx_path.open(encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                if r["image_path"] not in seen:
                    seen.add(r["image_path"])
                    keep.append(i)
        A = np.load(Xa, mmap_mode="r")
        X_aug, y_aug = np.asarray(A[keep], dtype=np.float64), None
        with idx_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        y_aug = [int(rows[i]["label"]) for i in keep]
        tag = "v1b"
        print(
            f"[v1b] half-mix: {len(keep)} first-views kept ({len(seen)} unique images)", flush=True
        )
    else:
        X_aug, y_aug = load_concat([clean, aug])
        tag = "v1"
    print(
        f"[v1] clean {X_clean.shape} pos={sum(y_clean)} | "
        f"clean+aug {X_aug.shape} pos={sum(y_aug)} "
        f"({time.perf_counter() - t0:.1f}s load)",
        flush=True,
    )

    summary: dict = {
        "seeds": args.seeds,
        "variants": {},
        "mix": "half (50/50 clean/first-view)" if args.half_mix else "full (25/75 clean/randaug3)",
    }

    lin_name, mlp_name = f"{tag}_linear", f"{tag}_mlp"
    train_desc = "clean+first-randaug-view" if args.half_mix else "clean+randaug3_seed42"

    # v0 sanity: linear on clean-only, seed 0
    if not args.skip_v0:
        clf = train_linear(X_clean, y_clean, seed=0)
        probe = LinearProbe(
            clf.coef_.ravel(),
            float(clf.intercept_[0]),
            thr=THR,
            meta={"config": "v0_repro_clean_only", "seed": 0},
        )
        out = args.out / "v0_repro" / "seed0"
        probe.save(out / "probe.npz")
        rows = eval_all(probe.predict_proba, cd)
        (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        write_markdown([{**r, "param": r["param"]} for r in rows], out / "results.md")
        val_auroc = rows[0]["auroc"]
        print(f"[v0_repro] val clean auroc={val_auroc:.4f} (expect ~0.9933)", flush=True)

    for name, trainer in ((lin_name, train_linear),):
        for seed in args.seeds:
            t1 = time.perf_counter()
            clf = trainer(X_aug, y_aug, seed)
            probe = LinearProbe(
                clf.coef_.ravel(),
                float(clf.intercept_[0]),
                thr=THR,
                meta={"config": name, "seed": seed, "train_rows": len(y_aug), "train": train_desc},
            )
            out = args.out / name / f"seed{seed}"
            probe.save(out / "probe.npz")
            rows = eval_all(probe.predict_proba, cd)
            (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            write_markdown(rows, out / "results.md")
            print(
                f"[{name}] seed={seed} val={rows[0]['auroc']:.4f} "
                f"({time.perf_counter() - t1:.1f}s)",
                flush=True,
            )

    if not args.skip_mlp:
        for seed in args.seeds:
            t1 = time.perf_counter()
            mlp, scaler = train_mlp(X_aug, y_aug, seed)
            out = args.out / mlp_name / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(mlp, out / "mlp.joblib")
            joblib.dump(scaler, out / "scaler.joblib")
            meta = {
                "config": mlp_name,
                "seed": seed,
                "train_rows": len(y_aug),
                "train": train_desc,
                "hidden": [512],
                "dropout_spec": 0.2,
                "dropout_actual": "n/a (sklearn CPU) -> L2 alpha=1e-4",
                "optimizer": "adam lr=1e-3 (AdamW unavailable in sklearn)",
                "epochs": 30,
                "standardized": True,
            }
            (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            def proba(X, _mlp=mlp, _sc=scaler):
                return _mlp.predict_proba(_sc.transform(np.asarray(X, dtype=np.float64)))[
                    :, 1
                ].tolist()

            rows = eval_all(proba, cd)
            (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            write_markdown(rows, out / "results.md")
            print(
                f"[v1_mlp] seed={seed} val={rows[0]['auroc']:.4f} "
                f"loss={mlp.loss_:.4f} iters={mlp.n_iter_} "
                f"({time.perf_counter() - t1:.1f}s)",
                flush=True,
            )

    # mean±std over seeds per (split, transform, param), per config
    for config in [lin_name] + ([] if args.skip_mlp else [mlp_name]):
        key0 = f"{config}/seed{args.seeds[0]}/results.json"
        base = json.loads((args.out / key0).read_text(encoding="utf-8"))
        agg = []
        for i, r0 in enumerate(base):
            accs = {m: [] for m in ("auroc", "acc", "tpr_at_1fpr", "ece")}
            for seed in args.seeds:
                r = json.loads(
                    (args.out / config / f"seed{seed}" / "results.json").read_text(encoding="utf-8")
                )[i]
                for m in accs:
                    accs[m].append(r[m])
            entry = {
                "split": r0["split"],
                "transform": r0["transform"],
                "param": r0["param"],
                "n": r0["n"],
            }
            for m, vs in accs.items():
                entry[m + "_mean"] = float(np.mean(vs))
                entry[m + "_std"] = float(np.std(vs))
            agg.append(entry)
        summary["variants"][config] = agg
    summary_path = args.out / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{tag}] done in {time.perf_counter() - t0:.1f}s -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()

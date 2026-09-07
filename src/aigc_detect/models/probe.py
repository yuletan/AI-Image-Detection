"""Linear probe on frozen features (Day 1 Block 4: v0 baseline).

Trains sklearn LogisticRegression on a cached train split, reports the full
CONTRACTS §5 metric set on a val cache. Feature-level API: Day-2 predict.py
will compose backbone + probe for raw-image Model.score(images).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_OUT = REPO_ROOT / "results" / "v0_probe"


def load_cache(npy_path: Path, index_path: Path) -> tuple[object, list[int]]:
    import numpy as np

    X = np.load(npy_path)
    with index_path.open(encoding="utf-8") as f:
        y = [int(r["label"]) for r in csv.DictReader(f)]
    if len(y) != len(X):
        raise ValueError(f"{npy_path}: {len(X)} rows vs {len(y)} labels")
    return X, y


def train_probe(X, y, C: float = 1.0, seed: int = 0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


class LinearProbe:
    """Calibrated P(AIGC) head: predict_proba -> [0,1], label at thr."""

    def __init__(self, coef, intercept, thr: float = 0.5, meta: dict | None = None):
        import numpy as np

        self.coef_ = np.asarray(coef, dtype=np.float64).ravel()
        self.intercept_ = float(intercept)
        self.thr = float(thr)
        self.meta = meta or {}

    def predict_proba(self, X) -> list[float]:
        import numpy as np

        X = np.asarray(X, dtype=np.float64)
        z = X @ self.coef_ + self.intercept_
        return (1.0 / (1.0 + np.exp(-z))).tolist()

    def predict(self, X) -> list[int]:
        return [1 if p >= self.thr else 0 for p in self.predict_proba(X)]

    def save(self, path: Path) -> Path:
        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, coef=self.coef_, intercept=self.intercept_, thr=self.thr)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2),
                                             encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> LinearProbe:
        import numpy as np

        path = Path(path)
        z = np.load(path)
        meta_p = path.with_suffix(".json")
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        return cls(z["coef"], float(z["intercept"]), float(z["thr"]), meta)


def run(args: argparse.Namespace) -> dict:
    from aigc_detect.eval import summarize

    t0 = time.perf_counter()
    Xtr, ytr = load_cache(Path(args.train_npy), Path(args.train_index))
    Xva, yva = load_cache(Path(args.val_npy), Path(args.val_index))
    print(f"[probe] train {Xtr.shape} pos={sum(ytr)} | val {Xva.shape} pos={sum(yva)}",
          flush=True)
    clf = train_probe(Xtr, ytr, C=args.C, seed=args.seed)
    probe = LinearProbe(clf.coef_.ravel(), float(clf.intercept_[0]), thr=args.thr,
                        meta={"C": args.C, "seed": args.seed, "thr": args.thr,
                              "train_rows": len(ytr), "train_npy": str(args.train_npy)})
    scores = probe.predict_proba(Xva)
    metrics = summarize(yva, scores, thr=args.thr)
    out = Path(args.out)
    probe.save(out / "probe.npz")
    print(f"[probe] val auroc={metrics['auroc']:.4f} acc={metrics['acc']:.4f} "
          f"tpr@1fpr={metrics['tpr_at_1fpr']:.4f} ece={metrics['ece']:.4f} "
          f"({time.perf_counter() - t0:.1f}s) -> {out / 'probe.npz'}", flush=True)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Linear probe on frozen caches (v0)")
    ap.add_argument("--train-npy", required=True, type=Path)
    ap.add_argument("--train-index", required=True, type=Path)
    ap.add_argument("--val-npy", required=True, type=Path)
    ap.add_argument("--val-index", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    return ap


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

"""Robustness eval: caches + probe -> results.json + table + heatmap.

CONTRACTS §5 row: {transform, param, auroc, acc, tpr_at_1fpr, ece}.
One row per cached ``{split}_*.npy`` (default split: test). Filenames parse as
``{split}_{transform}_{param}`` with transform allowed underscores
(``screenshot_repost``) and param ``None``/``chain`` preserved (None -> null).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_OUT = REPO_ROOT / "results" / "v0_robustness"


def parse_cache_stem(stem: str, split: str) -> tuple[str, object]:
    """'test_screenshot_repost_chain' -> ('screenshot_repost', 'chain')."""
    from aigc_detect.features.extract import parse_param

    body = stem[len(split) + 1:] if stem.startswith(split + "_") else stem
    transform, _, raw = body.rpartition("_")
    if not transform:  # no underscore left: whole body is the transform
        transform, raw = body, "None"
    return transform, parse_param(transform, raw)


def evaluate_cache(npy_path: Path, probe, thr: float) -> dict:
    from aigc_detect.eval import summarize
    from aigc_detect.models.probe import load_cache

    index = npy_path.parent / (npy_path.name + ".index.csv")
    X, y = load_cache(npy_path, index)
    return summarize(y, probe.predict_proba(X), thr=thr)


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = ["| transform | param | n | auroc | acc | tpr@1fpr | ece |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['transform']} | {r['param']} | {r['n']} | "
                     f"{r['auroc']:.4f} | {r['acc']:.4f} | "
                     f"{r['tpr_at_1fpr']:.4f} | {r['ece']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_heatmap(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{r['transform']}={r['param']}" for r in rows]
    vals = np.array([[r["auroc"], r["acc"], r["tpr_at_1fpr"], 1.0 - r["ece"]]
                     for r in rows])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(rows) + 1.5)))
    im = ax.matshow(vals, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(4), ["auroc", "acc", "tpr@1fpr", "1-ece"])
    ax.set_yticks(range(len(rows)), labels)
    for i in range(len(rows)):
        for j in range(4):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="higher is better")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> list[dict]:
    from aigc_detect.models.probe import LinearProbe

    cache_dir, out = Path(args.cache_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probe = LinearProbe.load(Path(args.probe))
    thr = args.thr if args.thr is not None else probe.thr
    npys = sorted(cache_dir.glob(f"{args.split}_*.npy"))
    if not npys:
        raise RuntimeError(f"no {args.split}_*.npy in {cache_dir}")
    rows: list[dict] = []
    for npy in npys:
        transform, param = parse_cache_stem(npy.stem, args.split)
        m = evaluate_cache(npy, probe, thr)
        rows.append({"transform": transform, "param": param, **m})
        print(f"[eval] {transform}={param} auroc={m['auroc']:.4f} acc={m['acc']:.4f} "
              f"tpr@1fpr={m['tpr_at_1fpr']:.4f} ece={m['ece']:.4f} n={m['n']}",
              flush=True)
    rows.sort(key=lambda r: (r["transform"] != "clean", r["transform"], str(r["param"])))
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(rows, out / "results.md")
    write_heatmap(rows, out / "heatmap.png")
    print(f"[eval] wrote {out / 'results.json'} + results.md + heatmap.png "
          f"({len(rows)} variants, thr={thr})", flush=True)
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Robustness table from caches (CONTRACTS §5)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--probe", type=Path, required=True, help="probe.npz from models.probe")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--thr", type=float, default=None, help="override probe threshold")
    return ap


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

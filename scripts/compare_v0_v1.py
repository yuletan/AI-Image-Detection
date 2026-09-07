"""Print v0 vs v1_linear vs v1_mlp comparison (test split, mean±std over seeds)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/v1_summary.json")
    ap.add_argument("--configs", nargs="+", default=["v1_linear", "v1_mlp"])
    args = ap.parse_args()

    res = REPO_ROOT / "results"
    v0 = {
        (r["transform"], str(r["param"])): r
        for r in load_rows(res / "v0_robustness" / "results.json")
    }
    summ = json.loads((REPO_ROOT / args.summary).read_text(encoding="utf-8"))
    cfgs = [c for c in args.configs if c in summ["variants"]]
    print(
        f"{'variant':24s} {'v0_auc':>8s} "
        + " ".join(f"{c + '_auc':>16s}" for c in cfgs)
        + "  "
        + f"{'v0_tpr':>7s} "
        + " ".join(f"{c + '_tpr':>16s}" for c in cfgs)
    )
    base = summ["variants"][cfgs[0]]
    for r0 in base:
        if r0["split"] != "test":
            continue
        key = (r0["transform"], str(r0["param"]))
        o = v0[key]
        line = f"{r0['transform'] + '=' + str(r0['param']):24s} {o['auroc']:8.4f} "
        tprs = f"{o['tpr_at_1fpr']:7.4f} "
        for c in cfgs:
            r = next(
                x
                for x in summ["variants"][c]
                if (x["transform"], str(x["param"])) == key and x["split"] == "test"
            )
            line += f"{r['auroc_mean']:8.4f}+-{r['auroc_std']:.4f} "
            tprs += f"{r['tpr_at_1fpr_mean']:8.4f}+-{r['tpr_at_1fpr_std']:.4f} "
        print(line + " " + tprs)
    # means over test variants
    print()
    for c in cfgs:
        for m in ("auroc", "tpr_at_1fpr", "acc", "ece"):
            vals = [r[m + "_mean"] for r in summ["variants"][c] if r["split"] == "test"]
            print(f"{c} mean_test_{m} = {sum(vals) / len(vals):.4f}")
    o_auc = sum(r["auroc"] for r in v0.values()) / len(v0)
    o_tpr = sum(r["tpr_at_1fpr"] for r in v0.values()) / len(v0)
    print(f"v0   mean_test_auroc = {o_auc:.4f}")
    print(f"v0   mean_test_tpr@1fpr = {o_tpr:.4f}")


if __name__ == "__main__":
    main()

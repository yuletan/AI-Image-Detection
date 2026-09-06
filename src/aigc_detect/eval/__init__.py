"""Eval metrics: AUROC, acc@thr, TPR@1%FPR, ECE (Day 1: A4 builds here)."""

from __future__ import annotations


def _roc_auc(y_true: list[int], y_score: list[float]) -> float:
    # Lightweight pure-python AUROC (no sklearn needed for Day 0 tests).
    pairs = sorted(zip(y_score, y_true, strict=False), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    prev_tp = prev_fp = 0
    auc = 0.0
    prev_score = None
    for s, y in pairs:
        if prev_score is not None and s != prev_score:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp, prev_fp = tp, fp
        prev_score = s
        if y:
            tp += 1
        else:
            fp += 1
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


def accuracy_at_threshold(y_true: list[int], y_score: list[float], thr: float = 0.5) -> float:
    preds = [1 if s >= thr else 0 for s in y_score]
    return sum(p == y for p, y in zip(preds, y_true, strict=False)) / max(1, len(y_true))


def tpr_at_fpr(y_true: list[int], y_score: list[float], fpr: float = 0.01) -> float:
    """TPR at a fixed FPR (interpolated on the ROC convex hull of thresholds)."""
    pairs = sorted(zip(y_score, y_true, strict=False), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # ROC points (fpr, tpr) walking down the ranked scores, ties grouped.
    pts = [(0.0, 0.0)]
    tp = fp = 0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        pts.append((fp / n_neg, tp / n_pos))
        i = j
    for (f0, t0), (f1, t1) in zip(pts, pts[1:], strict=False):
        if f1 >= fpr:
            w = 0.0 if f1 == f0 else (fpr - f0) / (f1 - f0)
            return t0 + w * (t1 - t0)
    return 1.0


def expected_calibration_error(y_true: list[int], y_score: list[float],
                               n_bins: int = 15) -> float:
    """ECE with uniform-width bins on [0,1]."""
    if not y_score:
        return float("nan")
    edges = [i / n_bins for i in range(n_bins + 1)]
    ece, n = 0.0, len(y_score)
    for b in range(n_bins):
        idx = [i for i, s in enumerate(y_score)
               if (edges[b] < s <= edges[b + 1] if b else s <= edges[b + 1])]
        if not idx:
            continue
        acc = sum(y_true[i] for i in idx) / len(idx)
        conf = sum(y_score[i] for i in idx) / len(idx)
        ece += len(idx) / n * abs(acc - conf)
    return ece


def summarize(y_true: list[int], y_score: list[float], thr: float = 0.5) -> dict:
    return {
        "auroc": _roc_auc(y_true, y_score),
        "acc": accuracy_at_threshold(y_true, y_score, thr),
        "tpr_at_1fpr": tpr_at_fpr(y_true, y_score, 0.01),
        "ece": expected_calibration_error(y_true, y_score),
        "n": len(y_true),
    }

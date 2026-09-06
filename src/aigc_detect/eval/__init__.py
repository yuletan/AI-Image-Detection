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


def summarize(y_true: list[int], y_score: list[float], thr: float = 0.5) -> dict:
    return {
        "auroc": _roc_auc(y_true, y_score),
        "acc": accuracy_at_threshold(y_true, y_score, thr),
        "n": len(y_true),
    }

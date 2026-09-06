# PROBLEM.md — framing (Day 0 · locked)

## One paragraph

We build a **robust post-upload detector** for AI-generated images on a social platform:
every uploaded image passes through a moderation queue that scores it
`pred ∈ [0,1]` (probability it is AIGC) before human review or ranking.
**False positives hurt individual creators more than false negatives hurt the platform:**
a wrongly flagged real photo suppresses reach, triggers appeals, and erodes trust,
while a missed fake is one of thousands and can still be caught by downstream reports,
retrieval, or the abstain band. So we optimise for **low FPR at high TPR**
(TPR@1%FPR), calibrated probabilities, and an explicit **abstain / send-to-human**
band under image degradations (JPEG, resize, blur, noise, crop, jitter, repost chains) —
not just clean accuracy.

## Deployment assumption

- Point of use: **post-upload moderation queue** (batch + online scoring via `predict.py`).
- Input: arbitrary jpg/jpeg/png/webp/bmp/tiff, nested dirs, possibly corrupt.
- Output: `[{image_path, pred}]` with `pred` calibrated; separate `label` at a
  published threshold (chosen at 1% FPR on clean val) + `uncertainty` (TTA std).
- Constraints: <2B params total (we use ~0.43B frozen CLIP ViT-L/14 + tiny head),
  CPU fallback, weights auto-download, `--lite` ViT-B/16 option.

## Why frozen CLIP (UnivFD-style)

Ojha et al. 2023 (UnivFD) showed frozen CLIP features generalise to unseen
generators far better than CNNs trained from scratch; DIRE / GenImage document
the benchmark practice (multi-generator train, held-out generator test).
We add: (1) augmentation-as-training for the head, (2) a degradation-gated
low-level (SRM residual) branch, (3) TTA + consistency uncertainty.

## Papers skimmed (Day 0)

1. Ojha et al. 2023 — *Towards Universal Fake Image Detectors that Generalize
   Across Generative Models* (UnivFD, frozen CLIP + linear probe).
2. Wang et al. 2023 — DIRE + GenImage benchmark (multi-generator eval, cross-generator protocol).
3. Robustness / augmentation study notes — JPEG/blur/resize destroy low-level
   fingerprints first; semantic features survive; report clean vs. transformed
   tables + degradation curves, never a single number.

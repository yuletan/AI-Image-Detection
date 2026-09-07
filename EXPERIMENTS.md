# EXPERIMENTS.md — Day-1 v0 findings + Day-2 plan

## v0 baseline (linear probe on frozen CLIP ViT-L/14, clean train)

Val clean (n=2000): AUROC **0.9933**, acc 0.9620, TPR@1%FPR 0.8550, ECE 0.0339.
Test clean (n=4000): AUROC **0.9932**, acc 0.9625.
Held-out DDIM, never trained (n=1000 fake-only): acc **0.909** at thr 0.5.

Full test table (`results/v0_robustness/`): 18 variants, worst AUROC 0.9874
(noise 0.10). No collapse on blur σ2 (0.9895), resize 0.25× (0.9898), JPEG 30
(0.9934) — the expected Day-1 drops did NOT happen on AUROC.

**Key split:** ranking holds everywhere; the damage is at strict operating
points — TPR@1%FPR falls 0.834 (clean) → 0.696 (noise 0.10). Calibration
(ECE ≤ 0.05) survives. So Day 2 is about TPR@FPR + thresholds, not the backbone.

## Day-2 experiments

- **E1 (tonight, GPU ~15 min):** K=3 random-augmented views per train image
  (`extract --split train --random-aug 3 --aug-seed 42` → 60k rows). Then v1 =
  head on clean + aug. Expect: clean −0.5pt, blur/resize/noise TPR@1%FPR +10–20pt.
- **E2:** 2-layer MLP head (dropout 0.2, AdamW, 30 epochs, 3 seeds) vs linear
  probe on the same caches. Keep whichever wins mean robustness; report mean±std.
- **E3 (GATE):** LoRA on last 4 ViT blocks with online aug, 1–2 h. SHIP only if
  > +2pt mean robustness over v1 AND no held-out regression. Else stay frozen.
- **E4:** TTA over {clean, jpeg70, resize0.5, crop80, flip} + uncertainty = std
  across views → abstain band. Measure flagged-% vs accuracy-of-rest.
- **E5:** Threshold at 1% FPR on clean val; report FPR drift per transform at
  that fixed threshold (the real moderation failure mode) + temperature scaling.
- **E6:** Leave-one-generator-out sweep on cached features (~1 h). Decides how
  much generator diversity v-final needs.

## Overnight queue (in order)

1. `train_randaug3_seed42` extraction (~60k forwards, ~15 min).
2. v1 head training (CPU seconds after caches land).
3. 3-seed runs of the chosen config if E1–E2 conclude before morning.

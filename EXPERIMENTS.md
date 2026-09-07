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

1. `train_randaug3_seed42` extraction (~60k forwards, ~15 min). ✅ DONE
2. v1 head training (CPU seconds after caches land). ✅ DONE 2026-09-07
   (`scripts/train_v1.py`; linear ~2 s/seed, sklearn-MLP ~2 min/seed on CPU).
3. 3-seed runs of the chosen config if E1–E2 conclude before morning.

## Day-2 results: E1 + E2 (v1 = clean + aug, 3 seeds; commit `feat/data`)

2×2: mix {full 25/75 clean/randaug3, half 50/50 clean/first-view} ×
head {linear probe (C=1.0), MLP 512-relu (sklearn adam lr=1e-3, 30 epochs,
standardized; NO dropout/AdamW in sklearn — L2 alpha=1e-4 substitutes)}.
Eval: val clean + 18 test variants + held-out DDIM (fake-only, acc@0.5).
Full tables: `results/{v1,v1b}_{linear,mlp}/seed{0,1,2}/`,
`results/{v1,v1b}_summary.json` (mean±std; local, gitignored).

| config | mix | mean test AUROC | mean test TPR@1%FPR | held-out DDIM acc |
|---|---|---|---|---|
| v0 linear (ref) | clean only | 0.9918 | 0.8029 | **0.909** |
| v1b_linear | 50/50 | 0.9920 | 0.8090 | **0.909** |
| v1b_mlp | 50/50 | 0.9933 | 0.8555 | 0.839–0.854 |
| v1_linear | 25/75 | 0.9933 | 0.8357 | 0.762 |
| v1_mlp | 25/75 | **0.9957** | **0.8847** | 0.707–0.798 |

Findings:
- E1 expectation was half-right: no clean cost anywhere (clean AUROC
  0.9932 → 0.9942–0.9965 — aug training HELPED clean). Robustness gains
  are real but concentrate at TPR@1%FPR: v1_mlp noise 0.10
  0.696 → **0.849** (+15pt), blur σ2 0.775 → 0.850, resize 0.25×
  0.779 → 0.853.
- **Held-out trade-off (new):** aug fraction trades unseen-generator
  generalisation for in-distribution robustness, monotonically:
  0.909 (clean, 50/50-linear) → ~0.847 (50/50-MLP) → 0.762 (full-linear)
  → ~0.76 (full-MLP). Aug-as-training erodes the frozen-CLIP
  generalisation story that motivated the architecture.
- E2 verdict: **MLP > linear at fixed mix** (v1_mlp beats v1_linear on all
  18 variants). BUT v1b_mlp REGRESSES vs v0 on noise (0.9855 vs 0.9874)
  and resize 0.25× (0.9829 vs 0.9898) — high-capacity head + 30 epochs to
  near-zero loss overfits the small-mix cache. No single winner:
  robustness (v1_mlp) vs generalisation (v1b_linear) is now the explicit
  trade-off for the Day-3 Trade-offs section.
- Caveat: linear seeds are bit-identical (lbfgs deterministic, std=0.0000);
  seed-spread is meaningful only for the MLP (±0.0000–0.0004 AUROC —
  stable).

Implications:
- E3 bar is now v1_mlp (mean 0.9957 / 0.8847) AND held-out ≥ ~0.85.
  LoRA must beat BOTH; given aug already costs held-out, LoRA + online
  aug is likely to regress DDIM further — skip unless it clearly wins.
- Next cheapest wins (no GPU): E4 TTA + E5 threshold/FPR-drift on the
  v1b_mlp and v1_mlp heads; E6 LOGO to test whether generator diversity
  (not mix) recovers held-out.

## Backbone B: DINOv2 ViT-L/14 v0 (linear probe, clean train; 2026-09-07)

Setup: `timm vit_large_patch14_dinov2.lvd142m`, shared 224 pipeline with
ImageNet norm, 1024-d L2-normed, fp16 on CUDA (commit `612151c`,
`--batch-size 128 --workers 4`). 21 caches in `cache_dinov2/` (clean
train/val/test/heldout + 15 test params + 2 chains): train (20000, 1024),
val (2000, 1024). Head: same `LogisticRegression(C=1.0)` recipe as CLIP v0.
Kaggle: `dinov2_v0_probe/` + `dinov2_v0_robustness/` (saved notebook output).

- Val clean: AUROC **0.9806**, acc 0.9255, TPR@1%FPR 0.5900, ECE 0.0227.
- Test clean (n=4000): AUROC **0.9802**, acc 0.9287, TPR@1%FPR 0.5940.
- Test mean (18 variants): AUROC **0.9800**, TPR@1%FPR **0.5885**;
  worst AUROC 0.9787 (noise 0.10), best 0.9808 (`screenshot_repost`).
  Flat table: every variant within ±0.002 of clean — same "no collapse"
  shape as CLIP, one level lower.
- Held-out DDIM (fake-only, acc@0.5): **0.912** vs CLIP v0 0.909.

Verdict: CLIP stays primary (clean +1.3pt AUROC, TPR@1%FPR +26pt:
0.855 → 0.590). DINOv2's only win is held-out (+0.3pt) — second data
point for the Day-3 trade-off: backbones, like aug-mix, trade
in-distribution strict-operating-point power for unseen-generator
generalisation. Keep as insurance; no E1/E2 repeat on DINOv2 unless
CLIP held-out regresses further.

## Day-2 results: E4 (TTA+abstain) + E5 (threshold drift + temp scaling)

CPU-only on cached CLIP features (`scripts/eval_e4_e5.py`, 17 s laptop;
`results/e4_tta/`, `results/e5_threshold/`, `results/{e4,e5}_summary.json`).
Heads: v1_mlp (full 25/75 mix) vs v1b_mlp (half 50/50 mix), seeds 0/1/2.

E4 — TTA-mean over the 5 pinned views (clean, jpeg70, resize0.5, crop80,
flip) on test-clean (n=4000): AUROC delta **+0.0002–0.0004** (v1_mlp),
**−0.0002–−0.0004** (v1b_mlp). Ranking is saturated — TTA-mean is not
worth 5× inference. Abstain band works: flag top-10% by cross-view std
→ accuracy-of-rest 0.974 → **0.994** (v1_mlp; 0.992 v1b), top-20% →
0.9975. Flagged images carry ~0.2 mean std vs ~0.0001 for the rest —
uncertainty cleanly separates the errors. Ship abstain, skip TTA-mean.

E5 — threshold frozen at 1% FPR on clean val, per-variant FPR drift:
- v1_mlp threshold is **stable**: FPR 0.008–0.017 on all 18 variants
  (clean 0.010). Full-mix aug training bought threshold stability.
- v1b_mlp **drifts 2–4.75×** where it matters: resize 0.25× 0.0475,
  `filter_app` 0.0300, noise 0.10 0.0275, colorjitter 0.0225, blur σ2
  0.0212, JPEG 30/50 ~0.019–0.020. This is the real moderation failure
  mode (AUROC hides it), and it decides the mix: v1_mlp over v1b_mlp
  for anything deployed at a fixed threshold. (centercrop / screenshot
  chains deflate FPR — conservative direction, no action.)
- Calibration: MLP heads are overconfident (thr ≈ 0.994–1.000,
  T\* = 2.5–3.0); temperature scaling halves val ECE (~0.02–0.03 →
  ~0.01). Scaling is monotonic — ranking/FPR-drift numbers are
  unaffected, only probabilities move. Apply T\* before any abstain
  band or human-review queue.

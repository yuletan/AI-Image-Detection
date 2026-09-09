# EXPERIMENTS.md — full experiment log (v0 → E6) + state of play

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
0.855 → 0.590). DINOv2's only apparent win is held-out (+0.3pt) —
CORRECTION 2026-09-09: at p≈0.9, n=1000, binomial SE≈±1pt, so 0.912 vs
0.909 is noise-level, not a data point. Demoted: the backbone trade-off
stands on TPR@1%FPR (26pt gap, solid) only. Pool more held-out rows
before claiming any generalisation edge. Keep as insurance; no E1/E2
repeat on DINOv2 unless CLIP held-out regresses further.

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

## Day-2 results: E6 (leave-one-generator-out; pre-registered, 2026-09-09)

Question: does leaving one generator family out of training collapse
detection of that family? (`scripts/eval_e6_logo.py`, 2 s laptop CPU;
`results/e6_logo/results.json`, seed 0, v0-recipe linear probe on clean
features. Hypothesis pre-registered in `hypothesis.md`, merged here
after grading — predictions below are verbatim pre-run.)

Setup: train fakes = 2 families (`sid_synth` 5000, `sid_tampered` 5000);
train reals = `celebahq`/`ffhq` 4000 each, `sid_real` 2000. Test slices
are single-class (400–1000 rows → ±2–3 pts noise), so the metric is
acc@0.5, not AUROC. Reference: full-train v0 → DDIM 0.909.

Pre-run predictions: reals interchangeable (unseen real ~0.94–0.95,
DDIM ~0.90); losing half the fakes costs 10–15 pts on the unseen fake;
`leave-synth-out` worse than `leave-tampered-out`; no total collapse
(P(~0.6) ≈ 15%). Decision rule: fake-LOGO dip ≥ 10 pts → diversity is
v-final's binding constraint.

| Left out       | Left-out slice (pred) | DDIM held-out (pred) | Hit? |
|----------------|-----------------------|----------------------|------|
| `celebahq`     | 1.000 (~0.95)         | 0.932 (~0.90)        | 0/2  |
| `ffhq`         | 0.991 (~0.95)         | 0.946 (~0.90)        | 0/2  |
| `sid_real`     | **0.030** (~0.94)     | **0.997** (~0.90)    | 0/2  |
| `sid_synth`    | 0.531 (0.80–0.85)     | **0.854** (0.83–0.87)| 1/2  |
| `sid_tampered` | **0.058** (0.85–0.90) | **0.089** (0.85–0.88)| 0/2  |

Score **1/10 numeric, 0/2 directional** (needed real-dip < fake-dip and
synth-out ≤ tampered-out; got the opposite). Full-train slice baseline
for context: celebahq 1.000, ffhq 0.9988, sid_real **0.710** (weak even
with full training), synth 0.998, tampered 0.969.

Findings (all unexpected):
1. `sid_tampered` is load-bearing for ALL non-synth fake detection.
   Without it: tampered → 0.058 AND DDIM → 0.089. Synth-only training
   learns a "fake" concept covering only synth; tampered-family
   artifacts bridge to DDIM, synth artifacts don't. With 2 fake
   families, generalisation hangs on ONE of them.
2. `sid_real` is suspect data: 0.710 even full-train (vs 1.000/0.999 on
   the other reals), 0.030 unseen — these "real" images look fake to
   the model. Removing it IMPROVES everything else (tampered 0.997,
   DDIM 0.997): it trains as label noise. Open: what IS sid_real
   (recaptures? screenshots?) — inspect before v-final.
3. Dropping celebahq/ffhq RAISES DDIM (0.932/0.946): fewer clean reals
   → tighter real cluster → trigger-happier fake boundary (same
   threshold-shift physics as E5).
4. val AUROC stays 0.947–0.9935 in every run while slice accuracy
   collapses — ranking hides the failure, third instance after v0-noise
   and E5 (always report operating-point metrics).

E6 verdict: diversity, not mix-tuning, is v-final's binding constraint.
(a) Add fake families; (b) quarantine/investigate sid_real; (c) never
ship a head trained without tampered. Head choice (v1_mlp) stands, but
its held-out 0.71–0.80 now reads as tampered-dependence + aug erosion
combined — E1/E2's trade-off has a named cause.

## Day-3 plan (2026-09-09; external review incorporated, adapted to repo)

E3 closure — DROPPED, no run: "E6 shows errors are diversity-bound
(LOGO collapse without sid_tampered: 0.058/0.089), not capacity-bound;
prior work reports fine-tuning CLIP reduces cross-generator transfer
(Ojha et al. CVPR 2023 — VERIFY before citing). Expected LoRA gain
concentrates in-distribution, where v1_mlp already saturates (mean
0.9957/0.8847)." Optional 10-min capacity sanity to close properly:
MLP width sweep {128, 512, 2048} on cached features — if width doesn't
matter, the not-capacity-bound claim is sealed.

E6b — rerun LOGO with the v1_mlp recipe (not just v0 linear;
`scripts/eval_e6_logo.py` + MLP branch, ~2 min CPU). MLP+aug may amplify
tampered-dependence; must know before v-final commits to the recipe.

E7a — score-blend v0_linear + v1_mlp (CPU seconds; new
`scripts/eval_e7a_blend.py`). v0 head = `results/v0_probe/probe.npz`
(best held-out 0.909), v1_mlp heads = `results/v1_mlp/seed{N}/`
(best robustness 0.8847 mean TPR@1%FPR). Z-normalise each head's val
logits first (MLP probabilities saturate near 0/1 and would dominate),
sweep blend α on VAL only, eval 18 variants + held-out, then re-freeze
the threshold and re-run the E5 drift table on the blend (v0 is
clean-trained and may import FPR instability). Pre-reg: held-out
interpolates 0.76→0.91 and mean TPR 0.88→0.80 roughly monotonically; if
α≈0.5 lands ≥0.85 on both, zero-new-data v-final candidate.

E7b — CLIP+DINOv2 feature concat (CPU minutes, AFTER pulling the
Kaggle `cache_dinov2/` outputs — ~95k rows × 1024 × 4B ≈ **390 MB**,
hundreds not tens; download as a versioned dataset, don't re-extract).
Per-backbone StandardScaler → concat (768+1024=1792-d) → linear + MLP
on both mixes + trivial score-average; align rows by `index.csv`
`image_path` (both extractors write manifest order — assert, don't
assume). Tests whether the backbone trade-off is a plane or a point
(precedent: AIDE CVPR 2024 hybrid frozen-CLIP features — VERIFY).
Reads with the DINOv2 noise correction above: expect the TPR gap to
persist, judge only on operating-point metrics + pooled held-out.

E8 — sid_real audit battery (CPU ~30 min), then quarantine-retrain.
Cheapest decisive order: E8.0 check the data card/forum threads for
this dataset FIRST (fastest resolution path if SID reals are already
characterised); E8.1 real-vs-real probe on `train_clean_None.npy`
(celebahq+ffhq vs sid_real, LR AUROC ≈ 0.5 → random label noise,
≥ 0.9 → systematic distribution); E8.2 cleanlab (`pip install
cleanlab`, 5-fold OOF over 20k train = 5× logreg fits, minutes) for a
principled suspect list; E8.3 third-party vote — 400 sid_real test rows
+ 400 celebahq control through off-the-shelf HF AI-image detectors, if
independent detectors also cry fake the noise is externally confirmed;
E8.4 cheap forensics (PIL EXIF, radial FFT spectra, Laplacian variance,
ImageHash pHash NN vs fake families — all deps already present).
Contact sheet: no script on THIS branch (one lives on `feat/features`)
→ write `scripts/make_sidreal_sheet.py` or port it. Then v2 = v1_mlp
recipe minus sid_real (or minus relabelled rows) → rerun E5 + E6 +
held-out. Pre-reg: DDIM 0.85–0.95 (E6 removal effect minus aug
erosion), FPR stability preserved, sid_real slice drops (expected, now
justified).

E9 — scoped family expansion (Kaggle GPU; reuse
`notebooks/kaggle_extract.ipynb` MANIFEST + skip-exists loop. New
GENERATORS, not new views, so extend the manifest pipeline +
`data/` payload, not `configs/transforms.yaml`.) At most three: one diffusion family (SD1.5 —
GenImage or self-generate with diffusers), one GAN family
(BigGAN/StyleGAN ex-GenImage), one local-edit family (SD-inpaint ~2k,
random box masks on ffhq/celebahq crops — directly tests the E6
hypothesis that tamper-style artifacts bridge to DDIM: with tamper
diversity, the leave-tampered-out 0.089 collapse should shrink).
Generate/download 1–2k images from a NEVER-trained generator (SDXL or
SD2.1) as a SECOND held-out — "diversity helps" must be tested on two
points, DDIM stays untouched. Keep tampered in, family-balance fakes.

E10 — if idle: manifold mixup on cached features (α 0.2–0.4, CPU
minutes; known to help strict operating points + calibration);
cross-tab the E4 flagged-10% against transform/family (free figure, may
expose a systematic pocket).

Hygiene (do with v2): bootstrap CIs — 1000 resamples on the 400-row
slice accuracies (seconds); the ±2–3pt noise bands asserted in E6 are
currently unquantified, and E7b/gate calls need real error bars.

Acceptance gate for v-final (PRE-REGISTERED — numbers frozen before
training; note tension: DDIM≥0.90 and LOGO-fake≥0.60 are FAILED by
current v1_mlp 0.71–0.80 / 0.058 — that is intentional, the gate forces
the E8+E9 resolution rather than blessing the status quo):

| Metric | Pass |
|---|---|
| val AUROC | ≥ 0.99 |
| mean TPR@1%FPR (18 variants) | ≥ 0.85 |
| max FPR @ frozen threshold, any variant | ≤ 0.02 |
| held-out DDIM | ≥ 0.90 |
| new held-out family | ≥ 0.85 |
| LOGO: any fake family | ≥ 0.60 |
| LOGO: tampered-out DDIM dip | < 15pt (was ~82pt) |
| abstain: acc-of-rest at 10% flagged | ≥ 0.99 |

Literature search handles (UNVERIFIED — external-model memory,
early-2025 cutoff; verify identity + claims before citing anything):
Ojha/Li/Lee universal fake detectors (CVPR 2023, frozen CLIP-L probe
lineage + fine-tuning-transfer warning); GenImage (NeurIPS 2023 D&B, 8
families — E9 source); AIDE (CVPR 2024, hybrid frozen-CLIP — E7b
precedent); DeepfakeBench (NeurIPS 2023 D&B, TPR@FPR conventions);
cleanlab (E8.2); Wang et al. 2020 CNN-generated-images (aug-training
lineage); DIRE (ICCV 2023) / LGrad / NPR (CVPR 2024) (non-CLIP features,
related-work only — GPU-heavy, not Day-3); DiffusionDB (realism-heavy
held-out alternative); Guo et al. 2017 (temperature scaling — done,
cite it); HF "AI image detector" spaces/models (E8.3 votes).

Day-3 shape: AM (CPU) E8.0→E8.1 + E7a script, kick off Kaggle E9
extraction before lunch so GPU runs during CPU work; midday v2 train +
E5/E6/E6b gate rerun + bootstrap CIs + write-up skeleton; PM v-final =
best recipe on cleaned (+diversified if E9 landed) data, full gate,
freeze, finish write-up. Keep the E6 1/10 scoreboard visible — asset,
not embarrassment.

## State of play + open decisions (for next agent, 2026-09-09)

Decided (don't relitigate without new data):
- Backbone: frozen CLIP ViT-L/14 primary; DINOv2 insurance only
  (better held-out +0.3pt, worse everywhere else, esp. TPR@1%FPR).
- Head/mix for fixed-threshold deployment: v1_mlp (full 25/75
  clean/randaug3 + sklearn MLP 512-relu). Only config with stable FPR
  across all 18 variants (E5).
- Ship abstain band (flag 10% by TTA-std → 99.4% accuracy-of-rest),
  skip TTA-mean (±0.0004 AUROC, not worth 5× inference).
- Temperature T\*≈2.5–3.0 before any review queue (halves ECE).
- Never train without `sid_tampered`; treat `sid_real` as guilty until
  inspected.

Open (pick up here):
1. E3 LoRA: DROPPED, see Day-3 plan closure paragraph (diversity-bound
   per E6 + fine-tuning-transfer warning). Optional MLP width sweep
   {128, 512, 2048} only if someone wants the capacity question sealed.
2. `sid_real` inspection: contact-sheet the 400 test + 2000 train rows;
   decide relabel / exclude / keep. Blocks v-final data card.
3. More fake families for v-final: which datasets, who extracts (needs
   GPU — Kaggle loop pattern in notebooks/kaggle_extract.ipynb reuses
   MANIFEST + skip-exists resume).
4. v-final recipe: v1_mlp + T\* + abstain band, retrained on
   cleaned/diversified data; re-run E4–E6 as the acceptance gate.
5. Day-3 write-up: Trade-offs section now has three named data points
   (aug-mix E1/E2, backbone DINOv2, diversity E6) + the recurring
   methods lesson (AUROC hides operating-point failures).

Artifact map (all local paths repo-relative; `results/` is gitignored):
- Caches `data/kaggle_cache/cache/`: 22 CLIP `.npy` (11.7 MB each:
  train/val/test/heldout clean + 18 test variants incl. 2 chains) +
  `train_randaug3_seed42.npy` (60k×768).
- Heads/tables: `results/v0_robustness/`, `results/v0_probe/`,
  `results/v0_heldout/`, `results/v0_repro/`,
  `results/{v1,v1b}_{linear,mlp}/seed{0,1,2}/`,
  `results/{v1,v1b}_summary.json`, `results/{e4,e5}_summary.json`,
  `results/e4_tta/`, `results/e5_threshold/`, `results/e6_logo/`.
- Kaggle-only (saved notebook outputs, NOT in repo): 21 DINOv2 caches
  (`cache_dinov2/`), `dinov2_v0_probe/`, `dinov2_v0_robustness/`.
- Scripts: `scripts/train_v1.py` (E1/E2), `scripts/eval_e4_e5.py` (E4/E5),
  `scripts/eval_e6_logo.py` (E6), `scripts/eval_heldout.py`,
  `scripts/compare_v0_v1.py`. Reruns: E4+E5 ~17 s, E6 ~2 s, tests 34
  passed 1 skipped (~6 s). Linear seeds deterministic (lbfgs).
- Branch `feat/data` (ahead of origin until pushed); laptop CPU-only,
  GPU via Kaggle notebook (`notebooks/kaggle_extract.ipynb`).
- Recurring gotcha: test slices are single-class → acc@0.5, never AUROC;
  always report fixed-threshold metrics alongside ranking ones.

# Robust AIGC Image Detection (TikTok TechJam 2026 · PS#5 · solo)

Frozen **CLIP ViT-L/14** + augmentation-trained head + TTA, with a degradation-aware
low-level branch. Deployment assumption: **post-upload moderation queue** (see `PROBLEM.md`).

> Day 1 status (2026-09-06): end-to-end v0 done — 27k manifest, contact sheet,
> 21 CLIP caches, linear-probe baseline. See Results below + `EXPERIMENTS.md`
> (lands on `feat/features` until merge).

## Quickstart (Day 0)

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check src tests scripts

# 1) see what datasets you need (downloads are the #1 hidden time sink — start NOW)
uv run python scripts/download_datasets.py --help
uv run python scripts/download_datasets.py --only cifake --out data/raw

# 2) transforms registry (exact official table lives in configs/transforms.yaml)
uv run python -c "from aigc_detect.transforms import list_transforms; print(list_transforms())"

# 3) CLI stub (full model lands Day 1-2)
uv run python -m aigc_detect.cli.predict --help
```

## Layout

```
src/aigc_detect/
  data/        manifest.csv builder, pHash dedupe, splits, leak check
  transforms/  registry: apply(img, name, param) + chains, SRM residual
  features/    frozen backbone extraction + caching (.npy + index.csv)
  models/      MLP head, LoRA/partial FT, fusion head
  eval/        AUROC / acc@thr / TPR@1%FPR / ECE, tables, curves
  cli/         predict.py : dir -> JSON [{image_path, pred}]
configs/       transforms.yaml (official table), train.yaml
scripts/       download_datasets.py, leak_check.py
data/          raw/ processed/ cache/ demo_benchmark/  (gitignored except README)
```

## Data (Day 1 manifest: `data/processed/manifest.csv`, 27,000 rows)

- WildFake subset (ModelScope): Real/celebahq 30k pool → 5,200 used,
  Real/ffhq 70k pool → 5,200 used, Diffusion DDIM 65,713 pool → held-out only.
- SID_Set (HF `saberzl/SID_Set`, streaming): label 0 → real (`sid_real` 2,600),
  1/2 → fake (`sid_synth`/`sid_tampered` 6,500 each).
- Splits: train 20k (10k real / 10k fake, 5 generators), val 2k, test 4k,
  heldout 1k (DDIM only — never in train, asserted in code).
- Dedupe: 42 pHash duplicates removed before split (seed 42, deterministic).
- Leak check vs `data/demo_benchmark/`: clean (0 hits; quarantine dirs were
  still empty at check time — re-run once COCO/DALL·E land).
- Licences/subset doc: see `data/README.md` + scripts provenance headers.

## Results

### v0 baseline (Day 1): frozen CLIP ViT-L/14 + logistic probe on clean train

| split | n | AUROC | acc@0.5 | TPR@1%FPR | ECE |
|---|---|---|---|---|---|
| val clean | 2000 | 0.9933 | 0.9620 | 0.8550 | 0.0339 |
| test clean | 4000 | 0.9932 | 0.9625 | 0.8340 | 0.0319 |
| heldout DDIM (fake-only) | 1000 | n/a | 0.9090 | n/a | — |

Robustness (test, n=4000 each; full table in `results/v0_robustness/`):

| transform | AUROC range | worst TPR@1%FPR |
|---|---|---|
| jpeg 90→30 | 0.9932–0.9943 | 0.8300 |
| blur σ0.5→2.0 | 0.9932→0.9895 | 0.7750 |
| resize 0.5 / 0.25 | 0.9922 / 0.9898 | 0.7785 |
| noise 0.02→0.10 | 0.9909→0.9874 | **0.6955** |
| colorjitter / centercrop / flip | 0.9912 / 0.9921 / 0.9933 | 0.8020 |
| chains (repost / filter_app) | 0.9910 / 0.9930 | 0.8085 |

Reading: no AUROC collapse anywhere (worst −0.006) — ranking survives all
transforms. Damage concentrates at strict operating points (noise 0.10 costs
14pt of TPR@1%FPR). Day 2 = augmented-head training (E1) + threshold/abstain
analysis, not backbone surgery. No Day-0 numbers exist (skeleton only) —
v0 is the first measured point.

## Parameter budget

- Primary: CLIP ViT-L/14 frozen (~0.43B) + MLP head (~0.5M) → ~0.43B total, ≪ 2B cap.
- Lite: ViT-B/16 (~0.15B) variant for the feasibility story.

## Docs

- `PROBLEM.md` — framing (Day 0, you)
- `CONTRACTS.md` — interfaces (freeze Day 1 10:30)
- `configs/transforms.yaml` — exact eval parameters

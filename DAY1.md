# DAY 1 — Foundation & baseline (exit: end-to-end robustness table exists)

Source: Track 3 board (`src/lib/seed-tasks.ts`). Solo adaptation below.
Interfaces are frozen (`CONTRACTS.md`) — build against them, don't change them.

## Block 1 — 09:00-10:30 · Contracts + manifest

- [x] `CONTRACTS.md` frozen (c300326)
- [x] Build subset manifest → `data/processed/manifest.csv` (27,000 rows: 20k train / 2k val / 4k test / 1k heldout=DDIM-only; schema `image_path,label,source,generator,split`)
  - Scan: `data/raw/wildfake_subset/Images/` (165k local) + SID_Set stream (`saberzl/SID_Set`, 0 real / 1+2 fake)
  - Target: 20k train (balanced real/fake, ≥4 generators), 2k val, 4k test, +1 held-out generator (never trained)
  - Dedupe by pHash; leak-check vs `data/demo_benchmark/` (must stay empty of train images)
  - Schema: `image_path,label,source,generator,split`
- [x] Sanity: `data/processed/manifest.csv` = 27,000 rows + header (verified 2026-09-06)

## Block 2 — 10:30-13:00 · Transforms first

- [ ] Visual contact sheet from `configs/transforms.yaml` params:
  JPEG q90/70/50/30 · blur σ0.5/1/2 · resize 0.5×/0.25× (upscale back) ·
  noise σ0.02/0.05/0.10 · jitter ±20% · center-crop 80%
- [ ] `pytest tests/test_transforms.py -q` green (already is — re-run after any edit)
- [ ] Chains work: `screenshot_repost`, `filter_app`
- [ ] GPU (Kaggle/Colab): smoke-test `open_clip ViT-L/14` fp16 on 512 imgs → record img/s, pick batch/workers

## Block 3 — 14:00-18:00 · Feature extraction (needs GPU)

- [ ] Frozen CLIP ViT-L/14 → `data/cache/{split}_{transform}_{param}.npy + index.csv`
- [ ] Coverage: clean train/val/test + 22 transformed test variants (~115k forwards, ~20-30 min on T4)
- [ ] Meanwhile (CPU): write data section of README (sources, licences, subset sizes, split rules, held-out generator)
- [ ] Queue DINOv2 ViT-L/14 extraction as backup backbone (cheap insurance)

## Block 4 — 18:00-19:30 · GATE: v0 baseline

- [ ] Linear probe on clean features → first full robustness table (`results.json` + markdown + heatmap)
- [ ] Expect: clean AUROC ≥0.95 in-distribution; big drops on blur σ2 / resize 0.25× / JPEG 30
- [ ] If clean < 0.90: STOP — check manifest labels, CLIP preprocessing (224 center-crop + CLIP mean/std), subset skew. Do NOT "fix" with fine-tuning.

## Block 5 — 20:00-22:00 · Analyze + overnight

- [ ] Write `EXPERIMENTS.md`: which transforms hurt most? AUROC drop (ranking) vs threshold-only (calibration)?
- [x] Queue overnight GPU: K=3 random-augmented views per TRAIN image (~60k forwards) — DONE 2026-09-06: `data/kaggle_cache/cache/train_randaug3_seed42.npy` (60,000×768, 63.6 img/s T4, 0 broken)
- [ ] Overnight CPU: tests, docstrings, CI green, README install verified from fresh venv

## Exit criteria (Day 1 done when ALL true)

- [x] `data/processed/manifest.csv` exists with 5 splits incl. heldout
- [ ] Contact sheet visually reviewed (reviewed per README; artifact not stored on this branch)
- [x] `results/v0_robustness.{json,md,png}` exists (local, gitignored; table mirrored in README)
- [ ] `EXPERIMENTS.md` lists numbered Day-2 experiments (lives on `feat/features` @ b91d8d1, not this branch)
- [x] Overnight augmentation job running, GPU queue non-empty → completed (see Block 5)

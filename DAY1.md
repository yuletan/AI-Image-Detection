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

- [x] Visual contact sheet from `configs/transforms.yaml` params:
  JPEG q90/70/50/30 · blur σ0.5/1/2 · resize 0.5×/0.25× (upscale back) ·
  noise σ0.02/0.05/0.10 · jitter ±20% · center-crop 80%
  - Reviewed per README 2026-09-06; script (`scripts/make_contact_sheet.py`) lives on `feat/features`, artifact ephemeral (not stored on `feat/data`)
- [x] `pytest tests/test_transforms.py -q` green (verified 2026-09-07: 11 passed, full suite)
- [x] Chains work: `screenshot_repost`, `filter_app` (in `transforms.yaml` + yaml-locked tests, green)
- [x] GPU (Kaggle/Colab): smoke-test `open_clip ViT-L/14` fp16 on 512 imgs → record img/s, pick batch/workers
  - Done: 21 CLIP caches landed + `train_randaug3_seed42` @ 63.6 img/s T4 (see Block 5)

## Block 3 — 14:00-18:00 · Feature extraction (needs GPU)

- [x] Frozen CLIP ViT-L/14 → `data/cache/{split}_{transform}_{param}.npy + index.csv`
  - Done 2026-09-06: 21 caches in `data/kaggle_cache/cache/` (train/val/test/heldout clean + 17 transformed test variants)
- [x] Coverage: clean train/val/test + 22 transformed test variants (~115k forwards, ~20-30 min on T4)
  - Done as 18 test variants (yaml defines 18 incl. clean + 2 chains; "22" was pre-audit estimate) + randaug train cache
- [x] Meanwhile (CPU): write data section of README (sources, licences, subset sizes, split rules, held-out generator)
- [ ] Queue DINOv2 ViT-L/14 extraction as backup backbone (cheap insurance)
  - Code support LANDED 2026-09-07 (`extract.py --backbone dinov2`, timm, ImageNet norm, 224 shared pipeline, 1024-d; CPU-plumbing verified). Still needs the GPU run — see Kaggle commands below. Not blocking Day 2 (E1/E2 done on CLIP).

## Block 4 — 18:00-19:30 · GATE: v0 baseline

- [x] Linear probe on clean features → first full robustness table (`results.json` + markdown + heatmap)
  - Done: `results/v0_robustness/{results.json,results.md,heatmap.png}` + `v0_probe/` + `v0_heldout/` (verified 2026-09-07)
- [x] Expect: clean AUROC ≥0.95 in-distribution; big drops on blur σ2 / resize 0.25× / JPEG 30
  - PASS with surprise: clean 0.9932; worst AUROC 0.9874 (noise 0.10, −0.006) — NO collapse; damage is at TPR@1%FPR (0.834→0.696 noise 0.10)
- [x] If clean < 0.90: STOP — check manifest labels, CLIP preprocessing (224 center-crop + CLIP mean/std), subset skew. Do NOT "fix" with fine-tuning.
  - N/A — gate passed (0.9932), no stop needed

## Block 5 — 20:00-22:00 · Analyze + overnight

- [x] Write `EXPERIMENTS.md`: which transforms hurt most? AUROC drop (ranking) vs threshold-only (calibration)?
  - Done: lives on `feat/features` @ b91d8d1 (not this branch) — v0 findings + E1–E6 Day-2 plan
- [x] Queue overnight GPU: K=3 random-augmented views per TRAIN image (~60k forwards) — DONE 2026-09-06: `data/kaggle_cache/cache/train_randaug3_seed42.npy` (60,000×768, 63.6 img/s T4, 0 broken)
- [x] Overnight CPU: tests, docstrings, CI green, README install verified from fresh venv
  - DONE 2026-09-07: root-caused (unused `albumentations` dep pulled `stringzilla`, which needs MSVC on Windows) → dropped dep (verified unused); rebuilt `.venv` (py3.11); `ruff check` + `ruff format --check` + `pytest` (35 passed) all green; full fresh-clone repro (`file://` clone → `uv sync --all-extras` → lint → 33 passed) green. Real GitHub Actions run fires on push.

## Exit criteria (Day 1 done when ALL true)

- [x] `data/processed/manifest.csv` exists with 5 splits incl. heldout
  - Verified 2026-09-07: 27,000 rows (train 20k balanced 10k/10k, val 2k, test 4k, heldout 1k DDIM-only, never in train)
- [x] Contact sheet visually reviewed (reviewed per README; artifact not stored on this branch)
- [x] `results/v0_robustness.{json,md,png}` exists (local, gitignored; table mirrored in README)
- [x] `EXPERIMENTS.md` lists numbered Day-2 experiments (lives on `feat/features` @ b91d8d1, not this branch)
- [x] Overnight augmentation job running, GPU queue non-empty → completed (see Block 5)

> Day-1 close-out 2026-09-07: effectively DONE except (a) DINOv2 backup extraction (Block 3, optional insurance) and
> (b) CI + fresh-venv README verification (Block 5). Neither blocks Day 2 — E1 input cache is already local.

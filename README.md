# Robust AIGC Image Detection (TikTok TechJam 2026 · PS#5 · solo)

Frozen **CLIP ViT-L/14** + augmentation-trained head + TTA, with a degradation-aware
low-level branch. Deployment assumption: **post-upload moderation queue** (see `PROBLEM.md`).

> Day 0 status: repo skeleton + dataset download starters + contracts placeholder.
> Interfaces freeze Day 1 10:30 in `CONTRACTS.md`.

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

## Data hygiene (read this)

- Train: WildFake subset + SID_Set (HF streaming) + ≥4 generators, balanced.
- Sanity only: CIFAKE (32×32, single generator — never evaluate robustness on it).
- **Quarantine:** `data/demo_benchmark/` = COCO val2017 (real) vs DALL·E Advanced (fake).
  NEVER in train. Leak-checked with pHash (`scripts/leak_check.py`).

## Parameter budget

- Primary: CLIP ViT-L/14 frozen (~0.43B) + MLP head (~0.5M) → ~0.43B total, ≪ 2B cap.
- Lite: ViT-B/16 (~0.15B) variant for the feasibility story.

## Docs

- `PROBLEM.md` — framing (Day 0, you)
- `CONTRACTS.md` — interfaces (freeze Day 1 10:30)
- `configs/transforms.yaml` — exact eval parameters

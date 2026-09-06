# CONTRACTS.md — interfaces (FROZEN 2026-09-06)

> **FROZEN 2026-09-06 by yuletan.** No interface changes without an explicit
> gate review. All agents/modules build against this file.

## 1) manifest.csv

```
image_path,label,source,generator,split
data/raw/wildfake/.../0001.jpg,1,wildfake,stylegan3,train
```

- `label`: 0 = real, 1 = fake (AIGC). `split` ∈ {train, val, test, heldout, demo}.
- Builder: `src/aigc_detect/data/manifest.py`. Dedupe by pHash before split.

## 2) transforms.apply(img, name, param) -> PIL.Image

```python
from aigc_detect.transforms import apply
out = apply(img, "jpeg", 70)   # exact semantics in configs/transforms.yaml
```

- Registry must implement the full official table + `chains` (repost combos).
- Deterministic, parameterised, unit-tested (`tests/test_transforms.py`).

## 3) features cache

```
data/cache/{split}_{transform}_{param}.npy + index.csv
```

- One row per image in manifest order; `extract_features.py --split --transform --param`.
- Frozen backbones only (CLIP ViT-L/14 primary, DINOv2-L/14 backup).

## 4) Model.score(images) -> probs[0..1]

- Calibrated P(AIGC). Threshold published separately; CLI also emits `label` + `uncertainty`.

## 5) eval output: results.json

```json
[{"transform": "jpeg", "param": 70, "auroc": 0.97, "acc": 0.92, "tpr_at_1fpr": 0.71, "ece": 0.03}]
```

- `eval.py` reads caches → `results.json` + markdown table + heatmap + curves.

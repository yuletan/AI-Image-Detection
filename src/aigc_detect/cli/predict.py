"""predict.py — dir -> JSON [{image_path, pred}]. Day 0 stub; full model Day 1-3.

Alt-requirements covered from Day 0:
- recurse nested dirs; accept jpg/jpeg/png/webp/bmp/tiff; skip non-images
- corrupt files -> pred=null + error field (never crash)
- --format jsonl, --absolute-paths, --device/--batch-size/--tta/--lite flags exist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from aigc_detect.transforms import apply

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

TTA_VIEWS: list[tuple[str, object]] = [
    ("clean", None),
    ("jpeg", 70),
    ("resize", 0.5),
    ("centercrop", 80),
    ("flip", "h"),
]


def _iter_images(input_dir: Path):
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUFFIXES:
            yield p


def _dummy_score(img: Image.Image) -> float:
    # Day 0 placeholder: mean brightness -> [0,1]. Replaced by CLIP+head Day 1.
    g = img.convert("L").resize((32, 32))
    px = g.tobytes()
    return sum(px) / (len(px) * 255.0)


def score_image(path: Path, tta: bool = False) -> dict:
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:  # corrupt -> pred=null, never crash
        return {"image_path": "", "pred": None, "error": f"unreadable: {e}"}
    try:
        if not tta:
            pred = _dummy_score(img)
            return {"image_path": "", "pred": round(float(pred), 4)}
        scores = []
        for name, param in TTA_VIEWS:
            v = apply(img, name, param)
            scores.append(_dummy_score(v))
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        return {
            "image_path": "",
            "pred": round(float(mean), 4),
            "uncertainty": round(float(var**0.5), 4),
        }
    except Exception as e:
        return {"image_path": "", "pred": None, "error": f"failed: {e}"}


def predict_dir(
    input_dir: Path,
    output: Path,
    tta: bool = False,
    absolute_paths: bool = False,
    fmt: str = "json",
) -> list[dict]:
    results: list[dict] = []
    for p in _iter_images(input_dir):
        r = score_image(p, tta=tta)
        rel = str(p.resolve()) if absolute_paths else str(p.relative_to(input_dir))
        r["image_path"] = rel
        # Day 0: also emit label at 0.5 + keep pred calibrated in [0,1]
        if r.get("pred") is not None:
            r["label"] = int(r["pred"] >= 0.5)
        results.append(r)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        output.write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")
    else:
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AIGC detect: image dir -> JSON")
    ap.add_argument("--input", required=True, type=Path, help="input image directory")
    ap.add_argument("--output", required=True, type=Path, help="output .json / .jsonl")
    ap.add_argument("--device", default="cpu", help="(Day 1+) cuda/cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tta", action="store_true", help="test-time augmentation")
    ap.add_argument("--lite", action="store_true", help="(Day 3) ViT-B/16 lite model")
    ap.add_argument("--format", dest="fmt", choices=["json", "jsonl"], default="json")
    ap.add_argument("--absolute-paths", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    predict_dir(
        args.input, args.output, tta=args.tta, absolute_paths=args.absolute_paths, fmt=args.fmt
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

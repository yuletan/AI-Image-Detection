"""Build a visual contact sheet proving transforms match configs/transforms.yaml.

Parses the official table (no hardcoded params) and renders one panel per
(name, param) plus one panel per chain, using aigc_detect.transforms.apply
and apply_chain. Saves a labeled PNG grid via matplotlib (Agg, CPU-only).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from aigc_detect.transforms import apply, apply_chain  # noqa: E402


def load_sample(path: str | None) -> tuple[Image.Image, str]:
    """Load sample image or fall back to a synthetic PIL image."""
    if path:
        try:
            img = Image.open(path).convert("RGB")
            return img, path
        except Exception as e:
            print(f"WARN: cannot read --sample {path!r} ({e}); using synthetic image")
    return make_synthetic(), "<synthetic>"


def make_synthetic(size: int = 256) -> Image.Image:
    """High-detail synthetic image (edges/texture/asymmetry) for eyeball checks."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (x % 256, y % 256, (x * 2 + y) % 256)
    d = ImageDraw.Draw(img)
    # off-center shapes so flip/crop are obvious
    d.ellipse([30, 40, 110, 130], fill=(220, 30, 30), outline=(255, 255, 255))
    d.rectangle([150, 60, 230, 200], fill=(30, 60, 220), outline=(255, 255, 0))
    d.line([0, 0, size - 1, size - 1], fill=(255, 255, 255), width=3)
    for gx in range(0, size, 16):
        d.line([gx, 0, gx, size], fill=(0, 0, 0), width=1)
    return img


def build_panels(img: Image.Image, cfg: dict) -> list[tuple[str, Image.Image]]:
    """One panel per (name, param) in yaml order + one per chain."""
    panels: list[tuple[str, Image.Image]] = []
    transforms: dict = cfg.get("transforms", {})
    for name, params in transforms.items():
        for param in params:
            out = apply(img, name, param)
            panels.append((f"{name} {param}", out))
    chains: dict = cfg.get("chains", {})
    for chain in chains:
        out = apply_chain(img, chain)
        panels.append((chain, out))
    return panels


def main() -> None:
    ap = argparse.ArgumentParser(description="Render transforms contact sheet from yaml.")
    ap.add_argument("--sample", default=None, help="Sample image path (else synthetic).")
    ap.add_argument("--out", default="results/contact_sheet/contact_sheet.png")
    ap.add_argument("--config", default="configs/transforms.yaml")
    ap.add_argument("--panels-dir", default=None, help="Optional per-panel PNG dir.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img, used = load_sample(args.sample)
    print(f"sample: {used} size={img.size} mode={img.mode}")

    panels = build_panels(img, cfg)
    print(f"panels: {len(panels)}")

    # Optional per-panel PNGs next to the grid.
    panels_dir = Path(args.panels_dir) if args.panels_dir else Path(args.out).parent / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    for label, panel in panels:
        fname = label.replace(" ", "_").replace("None", "orig") + ".png"
        panel.save(panels_dir / fname)

    cols = 6
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows + 1))
    axes = axes.flat if len(panels) > 1 else [axes]
    for ax, (label, panel) in zip(axes, panels, strict=False):
        ax.imshow(panel)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    for ax in list(axes)[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"contact sheet — sample: {used}", fontsize=11)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out} + {len(panels)} panels in {panels_dir}")


if __name__ == "__main__":
    main()

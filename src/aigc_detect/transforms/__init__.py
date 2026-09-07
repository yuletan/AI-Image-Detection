"""Transform registry implementing the official table exactly (Pillow/numpy).

Contract: apply(img, name, param) -> PIL.Image (see CONTRACTS.md).
No albumentations dependency: every op is Pillow/numpy so `uv sync` works
on hosts without a C++ build toolchain. (Day-2 SRM residual branch will
use torch/numpy instead.)
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageEnhance, ImageFilter

SUPPORTED: dict[str, list] = {
    "clean": [None],
    "jpeg": [90, 70, 50, 30],
    "blur": [0.5, 1.0, 2.0],
    "resize": [0.5, 0.25],  # downscale then upscale back to original size
    "noise": [0.02, 0.05, 0.10],  # gaussian sigma on [0,1] floats
    "colorjitter": [20],  # ±20% brightness/contrast/saturation
    "centercrop": [80],  # 80% center crop then resize back
    "flip": ["h"],
}

CHAINS: dict[str, list[tuple[str, object]]] = {
    "screenshot_repost": [("resize", 0.5), ("jpeg", 70), ("centercrop", 80)],
    "filter_app": [("colorjitter", 20), ("jpeg", 50)],
}


def list_transforms() -> dict[str, list]:
    return {k: list(v) for k, v in SUPPORTED.items()}


def apply(img: Image.Image, name: str, param) -> Image.Image:
    if name not in SUPPORTED:
        raise KeyError(f"unknown transform {name!r}; choose from {sorted(SUPPORTED)}")
    if name == "clean":
        return img.copy()
    if name == "jpeg":
        q = int(param)
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "blur":
        return img.filter(ImageFilter.GaussianBlur(float(param)))
    if name == "resize":
        s = float(param)
        w, h = img.size
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)
    if name == "noise":
        import hashlib

        import numpy as np

        sigma = float(param)
        px = img.convert("RGB")
        # deterministic per image (content hash), varying across images —
        # the old stdlib loop seeded on (size, sigma) only, stamping the
        # SAME noise field on every same-size image, ~100x slower.
        seed = int.from_bytes(hashlib.md5(px.tobytes()).digest()[:8], "little")
        seed ^= hash(("noise", sigma)) & 0xFFFFFFFFFFFFFFFF
        rng = np.random.default_rng(seed)
        arr = np.asarray(px, dtype=np.float32)
        noisy = arr + rng.standard_normal(arr.shape, dtype=np.float32) * (sigma * 255.0)
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    if name == "colorjitter":
        amt = float(param) / 100.0
        out = img
        for enh_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            out = enh_cls(out).enhance(1.0 + amt)
        return out
    if name == "centercrop":
        pct = float(param) / 100.0
        w, h = img.size
        cw, ch = int(w * pct), int(h * pct)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)
    if name == "flip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    raise KeyError(name)


def apply_chain(img: Image.Image, chain: str) -> Image.Image:
    if chain not in CHAINS:
        raise KeyError(f"unknown chain {chain!r}; choose from {sorted(CHAINS)}")
    out = img
    for name, param in CHAINS[chain]:
        out = apply(out, name, param)
    return out

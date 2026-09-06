from pathlib import Path

import yaml
from PIL import Image

from aigc_detect.transforms import CHAINS, SUPPORTED, apply, apply_chain, list_transforms

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "transforms.yaml"


def _img():
    return Image.new("RGB", (64, 64), (128, 64, 32))


def _yaml():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def _mse(a: Image.Image, b: Image.Image) -> float:
    ba, bb = a.tobytes(), b.tobytes()
    return sum((x - y) ** 2 for x, y in zip(ba, bb, strict=True)) / len(ba)


def test_registry_covers_official_table():
    t = list_transforms()
    for k in ["jpeg", "blur", "resize", "noise", "colorjitter", "centercrop"]:
        assert k in t
    assert 30 in t["jpeg"] and 2.0 in t["blur"] and 0.25 in t["resize"]


def test_registry_matches_yaml():
    cfg = _yaml()
    assert set(SUPPORTED) == set(cfg["transforms"])
    for name, params in cfg["transforms"].items():
        assert list(SUPPORTED[name]) == list(params), name
    yaml_chains = {c: [tuple(step) for step in steps] for c, steps in cfg["chains"].items()}
    norm_chains = {c: [tuple(step) for step in steps] for c, steps in CHAINS.items()}
    assert norm_chains == yaml_chains


def test_apply_roundtrip_all():
    img = _img()
    for name, params in SUPPORTED.items():
        for p in params:
            out = apply(img, name, p)
            assert out.size == img.size
            assert out.mode == "RGB"


def test_yaml_params_size_and_mode():
    img = _img()
    cfg = _yaml()
    for name, params in cfg["transforms"].items():
        for p in params:
            out = apply(img, name, p)
            assert out.size == img.size, (name, p)
            assert out.mode == "RGB", (name, p)


def test_chains():
    img = _img()
    for c in CHAINS:
        out = apply_chain(img, c)
        assert out.size == img.size


def test_chains_size_yaml():
    img = _img()
    cfg = _yaml()
    for c in cfg["chains"]:
        out = apply_chain(img, c)
        assert out.size == img.size, c
        assert out.mode == "RGB", c


def test_determinism_spot_check():
    img = _img()
    clean = apply(img, "clean", None)
    assert clean.tobytes() == img.tobytes()
    assert clean is not img
    assert _mse(img, apply(img, "jpeg", 90)) < _mse(img, apply(img, "jpeg", 30))
    assert apply(img, "noise", 0.05).tobytes() == apply(img, "noise", 0.05).tobytes()

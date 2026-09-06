from PIL import Image

from aigc_detect.transforms import CHAINS, SUPPORTED, apply, apply_chain, list_transforms


def _img():
    return Image.new("RGB", (64, 64), (128, 64, 32))


def test_registry_covers_official_table():
    t = list_transforms()
    for k in ["jpeg", "blur", "resize", "noise", "colorjitter", "centercrop"]:
        assert k in t
    assert 30 in t["jpeg"] and 2.0 in t["blur"] and 0.25 in t["resize"]


def test_apply_roundtrip_all():
    img = _img()
    for name, params in SUPPORTED.items():
        for p in params:
            out = apply(img, name, p)
            assert out.size == img.size
            assert out.mode == "RGB"


def test_chains():
    img = _img()
    for c in CHAINS:
        out = apply_chain(img, c)
        assert out.size == img.size

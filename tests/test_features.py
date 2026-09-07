import csv
from pathlib import Path

import pytest

from aigc_detect.features import cache_path
from aigc_detect.features.extract import (
    DINOV2_DIM,
    build_parser,
    load_preprocessing_cfg,
    parse_param,
    read_split_rows,
    resolve_backbone,
    sample_aug_views,
    write_index,
)

REPO = Path(__file__).resolve().parents[1]


def _rows_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label", "source", "generator", "split"])
        w.writeheader()
        w.writerows(rows)
    return path


def test_parse_param():
    assert parse_param("jpeg", "70") == 70
    assert parse_param("colorjitter", "20") == 20
    assert parse_param("blur", "0.5") == 0.5
    assert parse_param("noise", "0.02") == 0.02
    assert parse_param("flip", "h") == "h"
    assert parse_param("clean", "null") is None
    assert parse_param("clean", "None") is None
    assert parse_param("clean", None) is None


def test_resolve_backbone_specs():
    clip = resolve_backbone("clip")
    assert clip["kind"] == "open_clip" and clip["dim"] == 768
    assert clip["mean"] == [0.48145466, 0.4578275, 0.40821073]
    dino = resolve_backbone("dinov2")
    assert dino["kind"] == "timm" and dino["dim"] == DINOV2_DIM == 1024
    assert dino["mean"] == [0.485, 0.456, 0.406] and len(dino["std"]) == 3
    with pytest.raises(ValueError):
        resolve_backbone("resnet")


def test_parser_backbone_flag_defaults_clip():
    ap = build_parser()
    assert ap.parse_args(["--split", "test"]).backbone == "clip"
    assert ap.parse_args(["--split", "test", "--backbone", "dinov2"]).backbone == "dinov2"


def test_cache_naming_contract():
    p = cache_path(Path("data/cache"), "test", "jpeg", 70)
    assert p.name == "test_jpeg_70.npy"
    p = cache_path(Path("data/cache"), "train", "clean", None)
    assert p.parent.name == "cache" and p.suffix == ".npy"
    assert cache_path("data/cache", "test", "jpeg", 70).name == "test_jpeg_70.npy"


def test_preproc_cfg_matches_yaml():
    cfg = load_preprocessing_cfg(REPO / "configs" / "transforms.yaml")
    assert cfg["size"] == 224 and cfg["center_crop"] is True
    assert cfg["mean"] == [0.48145466, 0.4578275, 0.40821073]
    assert len(cfg["std"]) == 3


def test_build_preprocess_shape():
    torchvision = pytest.importorskip("torchvision")
    torch = pytest.importorskip("torch")
    _ = torchvision  # imported for its side-effect-free builder below
    from PIL import Image

    from aigc_detect.features.extract import build_preprocess

    t = build_preprocess(
        {"size": 224, "center_crop": True, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    )
    out = t(Image.new("RGB", (300, 200), (10, 20, 30)))
    assert isinstance(out, torch.Tensor) and out.shape == (3, 224, 224)


def test_read_split_rows_and_limit(tmp_path: Path):
    rows = [
        {
            "image_path": "a.jpg",
            "label": "0",
            "source": "wildfake",
            "generator": "ffhq",
            "split": "train",
        },
        {
            "image_path": "b.jpg",
            "label": "1",
            "source": "wildfake",
            "generator": "ddim",
            "split": "test",
        },
    ]
    mp = _rows_csv(tmp_path / "m.csv", rows)
    got = read_split_rows(mp, "train")
    assert [r["image_path"] for r in got] == ["a.jpg"]  # manifest order kept
    assert len(read_split_rows(mp, "test", limit=1)) == 1
    with pytest.raises(RuntimeError):
        read_split_rows(mp, "heldout")


def test_write_index_roundtrip(tmp_path: Path):
    rows = [{"image_path": "a.jpg", "label": 0, "source": "s", "generator": "g", "split": "train"}]
    out = tmp_path / "x.index.csv"
    write_index(rows, out)
    with out.open(encoding="utf-8") as f:
        back = list(csv.DictReader(f))
    assert back[0]["image_path"] == "a.jpg" and back[0]["split"] == "train"


def test_parser_contract_args():
    ap = build_parser()
    a = ap.parse_args(["--split", "test", "--transform", "jpeg", "--param", "70"])
    assert (a.split, a.transform, a.param) == ("test", "jpeg", "70")
    assert a.model is None and a.backbone == "clip" and a.precision == "auto"
    b = ap.parse_args(["--split", "test", "--chain", "screenshot_repost"])
    assert b.chain == "screenshot_repost" and b.transform == "clean"
    with pytest.raises(SystemExit):
        ap.parse_args([])  # --split required


def test_chain_cache_naming():
    p = cache_path(Path("data/cache"), "test", "screenshot_repost", "chain")
    assert p.name == "test_screenshot_repost_chain.npy"


def test_sample_aug_views_deterministic_and_mixed():
    a = sample_aug_views(60, seed=42)
    b = sample_aug_views(60, seed=42)
    assert a == b and len(a) == 60  # reproducible
    assert sample_aug_views(60, seed=7) != a  # seed matters
    kinds = {(v["transform"], v["chain"]) for v in a}
    assert any(c for _, c in kinds) and any(not c for _, c in kinds)  # chains + singles
    with pytest.raises(ValueError):
        sample_aug_views(0, seed=42)


def test_write_index_extra_cols(tmp_path: Path):
    rows = [
        {
            "image_path": "a.jpg",
            "label": 1,
            "source": "s",
            "generator": "g",
            "split": "train",
            "transform": "jpeg",
            "param": "70",
        }
    ]
    out = tmp_path / "r.index.csv"
    write_index(rows, out, extra=("transform", "param"))
    with out.open(encoding="utf-8") as f:
        back = list(csv.DictReader(f))
    assert back[0]["transform"] == "jpeg" and back[0]["param"] == "70"

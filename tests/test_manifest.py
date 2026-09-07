from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aigc_detect.data import MANIFEST_COLUMNS, VALID_SPLITS, read_manifest, write_manifest
from aigc_detect.data.manifest import (
    check_demo_leak,
    select_split,
    sid_label_and_generator,
    validate_rows,
)


def test_manifest_roundtrip(tmp_path: Path):
    rows = [
        {
            "image_path": "a.jpg",
            "label": 0,
            "source": "coco",
            "generator": "real",
            "split": "train",
        },
        {
            "image_path": "b.jpg",
            "label": 1,
            "source": "wildfake",
            "generator": "sd",
            "split": "train",
        },
    ]
    out = tmp_path / "manifest.csv"
    write_manifest(rows, out)
    back = read_manifest(out)
    assert len(back) == 2
    assert back[0]["label"] == "0"


def _noise_img(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(path)
    return path


def _row(img: Path, label: int, gen: str, split: str, source: str = "wildfake") -> dict:
    return {
        "image_path": img.as_posix(),
        "label": label,
        "source": source,
        "generator": gen,
        "split": split,
    }


def test_schema_columns(tmp_path: Path):
    out = tmp_path / "m.csv"
    write_manifest([_row(Path("x.jpg"), 0, "g", "train")], out)
    with out.open() as f:
        assert f.readline().strip() == ",".join(MANIFEST_COLUMNS)


def test_validate_rows_ok_and_bad():
    validate_rows([_row(Path("x.jpg"), 0, "g", "train"), _row(Path("y.jpg"), "1", "g", "demo")])
    with pytest.raises(ValueError):
        validate_rows([_row(Path("x.jpg"), 2, "g", "train")])
    with pytest.raises(ValueError):
        validate_rows([_row(Path("x.jpg"), 0, "g", "staging")])
    bad = {"image_path": "", "label": 0, "source": "s", "generator": "g", "split": "train"}
    with pytest.raises(ValueError):
        validate_rows([bad])


def test_sid_label_map():
    assert sid_label_and_generator("abc123", 0) == (0, "sid_real")
    assert sid_label_and_generator("full_synthetic_1", 1) == (1, "sid_synth")
    assert sid_label_and_generator("tampered_1", 2) == (1, "sid_tampered")
    with pytest.raises(ValueError):
        sid_label_and_generator("x", 3)


def test_select_split_balanced_heldout_disjoint_and_dedupes(tmp_path: Path):
    pools: dict[str, list] = {}
    for gen, label, base in (("celebahq", 0, 0), ("ffhq", 0, 100), ("sid_synth", 1, 200)):
        d = tmp_path / gen
        d.mkdir()
        pools[gen] = [(_noise_img(d / f"{i}.png", base + i), label) for i in range(12)]
    dup_src = tmp_path / "celebahq" / "0.png"  # planted exact pHash dupes
    for k in range(3):
        Image.open(dup_src).save(tmp_path / "celebahq" / f"dup{k}.png")
        pools["celebahq"].append((tmp_path / "celebahq" / f"dup{k}.png", 0))
    quotas = {
        "train": {"celebahq": 4, "ffhq": 4, "sid_synth": 8},
        "val": {"celebahq": 1, "ffhq": 1, "sid_synth": 2},
        "heldout": {"sid_synth_probe": 0},
    }
    rows, removed = select_split(pools, quotas, seed=42)
    assert removed >= 1  # the planted duplicate was caught by pHash
    labs = [r["label"] for r in rows if r["split"] == "train"]
    assert labs.count(0) == labs.count(1) == 8  # balanced train
    assert {r["split"] for r in rows} <= VALID_SPLITS
    assert all(r["label"] in (0, 1) for r in rows)
    # heldout generator never in train (use a dedicated probe generator here)
    gdir = tmp_path / "ddim"
    gdir.mkdir()
    q2 = {"train": {"celebahq": 4, "ffhq": 4}, "heldout": {"ddim": 2}}
    pools["ddim"] = [(_noise_img(gdir / f"{i}.png", 500 + i), 1) for i in range(4)]
    rows2, _ = select_split(pools, q2, seed=42)
    train_gens = {r["generator"] for r in rows2 if r["split"] == "train"}
    held_gens = {r["generator"] for r in rows2 if r["split"] == "heldout"}
    assert held_gens == {"ddim"} and not (held_gens & train_gens)


def test_demo_leak_check_detects_and_cleans(tmp_path: Path):
    img = _noise_img(tmp_path / "a.png", 7)
    other = _noise_img(tmp_path / "b.png", 8)
    rows = [_row(img, 0, "g", "train")]
    demo = tmp_path / "demo"
    (demo / "real").mkdir(parents=True)
    ok = check_demo_leak(rows, demo, root=tmp_path, threshold=5)
    assert ok["exact_hits"] == ok["near_hits"] == 0  # empty demo dir: vacuously clean
    import shutil

    shutil.copy(img, demo / "real" / "leak.png")  # exact-byte quarantine violation
    with pytest.raises(RuntimeError):
        check_demo_leak(rows, demo, root=tmp_path.parent, threshold=5)
    ok2 = check_demo_leak([_row(other, 1, "g", "test")], demo, root=tmp_path.parent, threshold=5)
    assert ok2["n_demo"] == 1 and ok2["exact_hits"] == ok2["near_hits"] == 0

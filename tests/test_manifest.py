from pathlib import Path

from aigc_detect.data import read_manifest, write_manifest


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

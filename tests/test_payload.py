import csv
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_kaggle_payload import build_payload, main, verify_archive  # noqa: E402


def _img(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (1, 2, 3)).save(path)
    return path


def _manifest(root: Path, rels: list[str]) -> Path:
    rows = [{"image_path": r, "label": "0", "source": "wildfake",
             "generator": "ffhq", "split": "train"} for r in rels]
    for r in rels:
        _img(root / r)
    mp = root / "data" / "processed" / "manifest.csv"
    mp.parent.mkdir(parents=True, exist_ok=True)
    with mp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return mp


def test_build_payload_copies_tree_and_manifest(tmp_path: Path):
    rels = ["data/raw/wildfake_subset/Images/Real/ffhq/a.jpg",
            "data/raw/sid_set_subset/b.jpg"]
    mp = _manifest(tmp_path, rels)
    out = tmp_path / "payload"
    info = build_payload(mp, out, root=tmp_path)
    assert info["n_files"] == 2 and info["bytes"] > 0
    for r in rels:
        assert (out / r).is_file()
    with (out / "manifest.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_build_payload_fail_closed_on_missing(tmp_path: Path):
    mp = _manifest(tmp_path, ["data/raw/x.jpg"])
    (tmp_path / "data" / "raw" / "x.jpg").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        build_payload(mp, tmp_path / "payload", root=tmp_path)


def test_main_writes_zip(tmp_path: Path):
    mp = _manifest(tmp_path, ["data/raw/y.jpg"])
    out = tmp_path / "payload"
    assert main(["--manifest", str(mp), "--out", str(out),
                 "--root", str(tmp_path), "--zip"]) == 0
    zf = out.with_suffix(".zip")
    assert zf.is_file()
    with zipfile.ZipFile(zf) as z:
        names = z.namelist()
    assert any(n.endswith("manifest.csv") for n in names)


def test_verify_archive_rejects_garbage(tmp_path: Path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip" * 100)
    with pytest.raises(RuntimeError, match="not a valid zip"):
        verify_archive(bad, 1)


def test_verify_archive_checks_entry_count(tmp_path: Path):
    mp = _manifest(tmp_path, ["data/raw/y.jpg"])
    out = tmp_path / "payload"
    assert main(["--manifest", str(mp), "--out", str(out),
                 "--root", str(tmp_path), "--zip"]) == 0
    with pytest.raises(RuntimeError, match="expected >="):
        verify_archive(out.with_suffix(".zip"), 10**9)

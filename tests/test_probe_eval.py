import csv
import json
from pathlib import Path

import numpy as np
import pytest

from aigc_detect.eval import (
    expected_calibration_error,
    summarize,
    tpr_at_fpr,
)
from aigc_detect.eval.evaluate import parse_cache_stem
from aigc_detect.eval.evaluate import run as eval_run
from aigc_detect.models.probe import LinearProbe, load_cache, train_probe


def test_tpr_at_fpr():
    assert tpr_at_fpr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert tpr_at_fpr([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert np.isnan(tpr_at_fpr([1, 1], [0.5, 0.6]))


def test_ece():
    assert expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert expected_calibration_error([0, 1], [0.5, 0.5]) == pytest.approx(0.0)
    assert np.isnan(expected_calibration_error([], []))


def test_summarize_has_contract_keys():
    r = summarize([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert r["auroc"] == r["acc"] == r["tpr_at_1fpr"] == 1.0
    assert r["ece"] >= 0.0 and r["n"] == 4


def _blobs(seed: int = 0, n: int = 60):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(-2.0, 1.0, (n, 8))
    X1 = rng.normal(2.0, 1.0, (n, 8))
    return np.vstack([X0, X1]).astype(np.float32), [0] * n + [1] * n


def test_probe_separates_and_roundtrips(tmp_path: Path):
    Xtr, ytr = _blobs(0)
    Xva, yva = _blobs(1)
    clf = train_probe(Xtr, ytr)
    probe = LinearProbe(clf.coef_.ravel(), float(clf.intercept_[0]))
    s = probe.predict_proba(Xva)
    assert all(0.0 <= p <= 1.0 for p in s)
    assert summarize(yva, s)["auroc"] == pytest.approx(1.0)
    probe.save(tmp_path / "probe.npz")
    back = LinearProbe.load(tmp_path / "probe.npz")
    assert back.predict_proba(Xva) == pytest.approx(s)


def test_load_cache_rejects_mismatch(tmp_path: Path):
    np.save(tmp_path / "x.npy", np.zeros((3, 4), dtype=np.float32))
    with (tmp_path / "x.npy.index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label", "source",
                                          "generator", "split"])
        w.writeheader()
        w.writerow({"image_path": "a", "label": "0", "source": "s",
                    "generator": "g", "split": "train"})
    with pytest.raises(ValueError, match="labels"):
        load_cache(tmp_path / "x.npy", tmp_path / "x.npy.index.csv")


def test_parse_cache_stem():
    assert parse_cache_stem("test_jpeg_70", "test") == ("jpeg", 70)
    assert parse_cache_stem("test_clean_None", "test") == ("clean", None)
    assert parse_cache_stem("test_screenshot_repost_chain", "test") == (
        "screenshot_repost", "chain")
    assert parse_cache_stem("test_blur_0.5", "test") == ("blur", 0.5)


def _fake_cache(d: Path, stem: str, seed: int) -> None:
    X, y = _blobs(seed)
    np.save(d / f"{stem}.npy", X)
    with (d / f"{stem}.npy.index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label", "source",
                                          "generator", "split"])
        w.writeheader()
        for i, lab in enumerate(y):
            w.writerow({"image_path": f"{i}.jpg", "label": str(lab), "source": "s",
                        "generator": "g", "split": "test"})


def test_evaluate_end_to_end(tmp_path: Path):
    pytest.importorskip("matplotlib")
    cache, out = tmp_path / "cache", tmp_path / "out"
    cache.mkdir()
    _fake_cache(cache, "test_clean_None", 0)
    _fake_cache(cache, "test_jpeg_70", 1)
    Xtr, ytr = _blobs(2)
    clf = train_probe(Xtr, ytr)
    LinearProbe(clf.coef_.ravel(), float(clf.intercept_[0])).save(cache / "probe.npz")
    import argparse

    args = argparse.Namespace(cache_dir=cache, probe=cache / "probe.npz",
                              split="test", out=out, thr=None)
    rows = eval_run(args)
    assert [r["transform"] for r in rows] == ["clean", "jpeg"]  # clean first
    assert rows[1]["param"] == 70 and rows[0]["param"] is None  # null in json
    saved = json.loads((out / "results.json").read_text())
    assert saved[0]["auroc"] == pytest.approx(1.0)
    assert (out / "results.md").exists() and (out / "heatmap.png").exists()

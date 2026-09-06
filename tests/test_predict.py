from pathlib import Path

from PIL import Image

from aigc_detect.cli.predict import predict_dir


def test_predict_dir_handles_nested_and_corrupt(tmp_path: Path):
    inp = tmp_path / "in"
    (inp / "sub").mkdir(parents=True)
    Image.new("RGB", (16, 16), (255, 0, 0)).save(inp / "a.jpg")
    Image.new("RGB", (16, 16), (0, 255, 0)).save(inp / "sub" / "b.png")
    (inp / "corrupt.jpg").write_bytes(b"not an image")
    (inp / "notes.txt").write_text("skip me")

    out = tmp_path / "preds.json"
    results = predict_dir(inp, out, tta=True)
    assert out.exists()
    assert len(results) == 3  # 2 good + 1 corrupt
    by_name = {r["image_path"]: r for r in results}
    assert by_name["corrupt.jpg"]["pred"] is None
    assert "error" in by_name["corrupt.jpg"]
    assert 0.0 <= by_name["a.jpg"]["pred"] <= 1.0
    assert "uncertainty" in by_name["a.jpg"]

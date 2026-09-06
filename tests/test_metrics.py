from aigc_detect.eval import summarize


def test_summarize_perfect_separation():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    r = summarize(y, s)
    assert r["auroc"] == 1.0
    assert r["acc"] == 1.0

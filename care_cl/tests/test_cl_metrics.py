"""Tests for the derived continual-learning metrics."""
from care_cl.eval.cl_metrics import (
    backward_transfer, final_avg_care, forgetting_per_farm, summarize,
)

SEQ = ["A", "B", "C"]


def _matrix():
    # stage -> {farm: CARE}. Farm A degrades as we learn B then C (forgetting).
    return {
        "A": {"A": 0.70},
        "B": {"A": 0.60, "B": 0.68},
        "C": {"A": 0.50, "B": 0.62, "C": 0.66},
    }


def test_forgetting_positive_when_degrading():
    fpf = forgetting_per_farm(_matrix(), SEQ)
    assert fpf["A"] == 0.70 - 0.50      # 0.20, positive forgetting
    assert fpf["B"] == 0.68 - 0.62
    assert "C" not in fpf               # last farm has no forgetting


def test_backward_transfer_negative_under_forgetting():
    bwt = backward_transfer(_matrix(), SEQ)
    assert bwt < 0


def test_final_avg_care():
    assert abs(final_avg_care(_matrix(), SEQ) - (0.50 + 0.62 + 0.66) / 3) < 1e-9


def test_summarize_keys():
    s = summarize(_matrix(), SEQ)
    for k in ("forgetting_per_farm", "mean_forgetting", "backward_transfer",
              "forward_transfer", "final_avg_care"):
        assert k in s
    assert s["mean_forgetting"] > 0

"""§6 acceptance tests for the CARE score: random~0.5, all-normal~0, monotonicity."""
import numpy as np

from care_cl.eval.care_score import (
    DatasetEval, care_score, evaluate_dataset, f_beta, earliness_weights,
)

NORMAL_IDS = [0, 2]


def _build_farm(predictor, n_anom=10, n_norm=10, n=2000, win=(800, 1600), seed=0):
    """Make a synthetic farm of anomaly + normal datasets scored by `predictor`."""
    rng = np.random.default_rng(seed)
    evals = []
    for k in range(n_anom):
        status = np.zeros(n, dtype=int)  # all normal-operation
        pred = predictor(n, win, rng, is_anom=True)
        evals.append(_tag(evaluate_dataset(
            pred, status, is_normal_only=False, event_window=win,
            normal_status_ids=NORMAL_IDS), f"a{k}"))
    for k in range(n_norm):
        status = np.zeros(n, dtype=int)
        pred = predictor(n, win, rng, is_anom=False)
        evals.append(_tag(evaluate_dataset(
            pred, status, is_normal_only=True, event_window=None,
            normal_status_ids=NORMAL_IDS), f"n{k}"))
    return evals


def _tag(e: DatasetEval, i: str) -> DatasetEval:
    e.id = i
    return e


def test_fbeta_basics():
    assert f_beta(0, 0, 0) == 0.0
    assert f_beta(10, 0, 0) == 1.0
    # precision-weighted: false positives hurt more than false negatives at beta<1
    assert f_beta(10, 5, 0) < f_beta(10, 0, 5)


def test_earliness_weights_shape():
    w = earliness_weights(11)
    assert w[0] == 1.0 and w[5] == 1.0          # first half == 1
    assert w[-1] == 0.0                          # decays to 0
    assert np.all(np.diff(w[5:]) <= 1e-9)        # non-increasing in 2nd half


def test_all_normal_predictions_score_zero():
    """A detector that never fires -> no events detected -> CARE == 0."""
    def never(n, win, rng, is_anom):
        return np.zeros(n, dtype=bool)
    res = care_score(_build_farm(never))
    assert res["CARE"] == 0.0


def test_all_anomaly_predictions_score_near_zero():
    """Flag-everything -> accuracy collapses below 0.5 -> CARE == accuracy ~ 0."""
    def always(n, win, rng, is_anom):
        return np.ones(n, dtype=bool)
    res = care_score(_build_farm(always))
    assert res["CARE"] < 0.1
    assert res["accuracy"] == 0.0


def test_random_predictions_mid_range():
    def rand(n, win, rng, is_anom):
        return rng.random(n) < 0.5
    res = care_score(_build_farm(rand))
    assert 0.2 < res["CARE"] < 0.75   # middling; anchor ~0.5


def test_correct_and_earlier_detection_increases_score():
    """A good detector (fires inside the window, quiet on normals) beats random,
    and detecting EARLIER within the window scores at least as high."""
    def good_late(n, win, rng, is_anom):
        p = np.zeros(n, dtype=bool)
        if is_anom:
            s, e = win
            mid = (s + e) // 2
            p[mid:e + 1] = True       # detect only in the 2nd half
        return p

    def good_early(n, win, rng, is_anom):
        p = np.zeros(n, dtype=bool)
        if is_anom:
            s, e = win
            p[s:e + 1] = True         # detect across whole window (incl. early)
        return p

    def rand(n, win, rng, is_anom):
        return rng.random(n) < 0.5

    care_rand = care_score(_build_farm(rand))["CARE"]
    res_late = care_score(_build_farm(good_late))
    res_early = care_score(_build_farm(good_early))

    assert res_late["CARE"] > care_rand
    # earlier/fuller detection -> higher earliness -> >= later detection
    assert res_early["earliness"] >= res_late["earliness"]
    assert res_early["CARE"] >= res_late["CARE"] - 1e-9
    assert res_early["accuracy"] == 1.0  # never fires on normal datasets


def test_criticality_event_detection():
    """A sustained burst >= 72 points fires the event verdict; a short one doesn't."""
    n = 500
    status = np.zeros(n, dtype=int)
    short = np.zeros(n, dtype=bool); short[100:150] = True   # 50 < 72
    long = np.zeros(n, dtype=bool); long[100:200] = True     # 100 >= 72
    e_short = evaluate_dataset(short, status, is_normal_only=True,
                               event_window=None, normal_status_ids=NORMAL_IDS)
    e_long = evaluate_dataset(long, status, is_normal_only=True,
                              event_window=None, normal_status_ids=NORMAL_IDS)
    assert e_short.event_pred == 0
    assert e_long.event_pred == 1

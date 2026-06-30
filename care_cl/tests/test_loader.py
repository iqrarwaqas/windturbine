"""Loader contract tests. Skip automatically if the dataset is not present."""
import os

import pytest

from care_cl.data.loader import _load_farm, EXPECTED_FEATURE_COUNTS

ROOT = os.environ.get("CARE_ROOT", "D:/Datasets/Care")
HAVE_DATA = os.path.isdir(os.path.join(ROOT, "Wind Farm A"))
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason=f"CARE dataset not at {ROOT}")


def test_farm_a_parses():
    fd = _load_farm(ROOT, "A", sep=";")
    assert fd.farm_id == "A"
    assert fd.n_features == EXPECTED_FEATURE_COUNTS["A"] - 5  # minus 5 meta cols
    assert len(fd.eval) == 22
    # at least one anomaly event parsed with a valid window
    anoms = [s for s in fd.eval if not s.is_normal_only]
    assert anoms, "expected anomaly datasets in farm A"
    s, e = anoms[0].anomaly_events[0]
    assert 0 <= s <= e


def test_farms_differ_in_feature_count():
    a = _load_farm(ROOT, "A", sep=";")
    b = _load_farm(ROOT, "B", sep=";")
    assert a.n_features != b.n_features


def test_train_and_prediction_splits_present():
    fd = _load_farm(ROOT, "A", sep=";")
    sub = fd.eval[0]
    tr = sub.split("train")
    pr = sub.split("prediction")
    assert len(tr) > 0 and len(pr) > 0
    assert sub.status_col == "status_type_id"

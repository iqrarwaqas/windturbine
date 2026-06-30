"""CARE score (§6) — Coverage, Accuracy, Reliability, Earliness.

Implements the metric from Gueck, Roelofs & Faulstich, "CARE to Compare", *Data*
2024, 9(12):138 (arXiv:2404.10320), matching the AEFDI/EnergyFaultDetector
definition:

- Coverage  : per-anomaly-dataset F_beta (beta=0.5) on the *normal-status* points
              of the prediction frame.                                 (Eq. 1)
- Accuracy  : per-normal-dataset tn/(fp+tn) on normal-status points.   (Eq. 2)
- Reliability: event-level F_beta across ALL datasets, where each dataset yields a
              single anomaly/normal verdict via a criticality counter (Algorithm 1,
              threshold t_c = 72 = 12h of consecutive 10-min anomalies).
- Earliness : per-anomaly-dataset weighted detection score; weight 1 over the first
              half of the event window, linearly decreasing to 0 over the second
              half.                                                    (Eq. 3)

Aggregation (Eq. 4-5):
    WA   = (w1*Fbar + w2*WSbar + w3*EFbeta + w4*Accbar) / sum(w),  w=(1,1,1,2)
    CARE = 0      if no anomaly events were detected at all
           Accbar if Accbar < 0.5   (worse than random recognition of normal)
           WA     otherwise

Anchors: random ~ 0.5, all-normal / all-anomaly ~ 0, AE baseline ~ 0.66.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def f_beta(tp: float, fp: float, fn: float, beta: float = 0.5) -> float:
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    if denom <= 0:
        return 0.0
    return float((1 + b2) * tp / denom)


def criticality_max(pred: np.ndarray, is_normal_status: np.ndarray) -> int:
    """Algorithm 1: max of a criticality counter over the prediction frame.

    +1 when status is abnormal; +1 when status is normal and an anomaly is
    predicted; decrement (floored at 0) when status is normal and no anomaly.
    """
    crit = 0
    cmax = 0
    for pi, normal in zip(pred, is_normal_status):
        if not normal:
            crit += 1
        elif pi:
            crit += 1
        else:
            crit = max(crit - 1, 0)
        if crit > cmax:
            cmax = crit
    return cmax


def earliness_weights(window_len: int) -> np.ndarray:
    """Weight 1 over the first half, linearly down to 0 over the second half."""
    if window_len <= 0:
        return np.zeros(0)
    if window_len == 1:
        return np.ones(1)
    tau = np.linspace(0.0, 1.0, window_len)
    w = np.where(tau <= 0.5, 1.0, 2.0 * (1.0 - tau))
    return np.clip(w, 0.0, 1.0)


@dataclass
class DatasetEval:
    id: str
    is_normal_only: bool
    coverage_fbeta: float | None   # anomaly datasets only
    earliness_ws: float | None     # anomaly datasets only
    accuracy: float | None         # normal-only datasets
    event_true: int                # 1 if dataset is an anomaly event
    event_pred: int                # 1 if criticality counter fired
    n_eval_points: int


def evaluate_dataset(
    pred: np.ndarray,
    status_ids: np.ndarray,
    *,
    is_normal_only: bool,
    event_window: tuple[int, int] | None,
    normal_status_ids,
    beta: float = 0.5,
    criticality_threshold: int = 72,
) -> DatasetEval:
    """Score one dataset's prediction frame.

    `pred`       : binary per-row anomaly predictions over the prediction split.
    `status_ids` : status_type_id per row, same length as `pred`.
    `event_window`: (start_pos, end_pos) inclusive in *prediction-frame* row index,
                    or None for normal-only datasets.
    """
    pred = np.asarray(pred).astype(bool)
    status_ids = np.asarray(status_ids)
    normal_mask = np.isin(status_ids, list(normal_status_ids))

    # Event-level verdict via criticality counter (uses all rows).
    cmax = criticality_max(pred, normal_mask)
    event_pred = int(cmax >= criticality_threshold)

    if is_normal_only or event_window is None:
        p = pred[normal_mask]
        n = p.size
        tn = int((~p).sum())
        fp = int(p.sum())
        acc = (tn / (fp + tn)) if (fp + tn) > 0 else 1.0
        return DatasetEval(
            id="", is_normal_only=True, coverage_fbeta=None, earliness_ws=None,
            accuracy=acc, event_true=0, event_pred=event_pred, n_eval_points=n,
        )

    # Anomaly dataset: build per-row ground truth over the prediction frame.
    start, end = event_window
    n_rows = pred.size
    gt = np.zeros(n_rows, dtype=bool)
    s = max(0, start)
    e = min(n_rows - 1, end)
    if e >= s:
        gt[s:e + 1] = True

    # Coverage on normal-status points only.
    g = gt[normal_mask]
    p = pred[normal_mask]
    tp = int((g & p).sum())
    fp = int((~g & p).sum())
    fn = int((g & ~p).sum())
    coverage = f_beta(tp, fp, fn, beta)

    # Earliness over the event window's normal-status points.
    win_idx = np.arange(s, e + 1)
    win_normal = normal_mask[win_idx]
    win_pred = pred[win_idx]
    w_full = earliness_weights(len(win_idx))
    w = w_full[win_normal]
    p_win = win_pred[win_normal].astype(float)
    ws = float((w * p_win).sum() / w.sum()) if w.sum() > 0 else 0.0

    return DatasetEval(
        id="", is_normal_only=False, coverage_fbeta=coverage, earliness_ws=ws,
        accuracy=None, event_true=1, event_pred=event_pred,
        n_eval_points=int(normal_mask.sum()),
    )


def care_score(evals: list[DatasetEval], *, beta: float = 0.5,
               weights: dict | None = None) -> dict:
    """Aggregate per-dataset evals into the CARE score + sub-scores."""
    w = weights or {"coverage": 1.0, "earliness": 1.0, "reliability": 1.0, "accuracy": 2.0}
    w1, w2, w3, w4 = w["coverage"], w["earliness"], w["reliability"], w["accuracy"]

    cov = [e.coverage_fbeta for e in evals if e.coverage_fbeta is not None]
    ear = [e.earliness_ws for e in evals if e.earliness_ws is not None]
    acc = [e.accuracy for e in evals if e.accuracy is not None]

    Fbar = float(np.mean(cov)) if cov else 0.0
    WSbar = float(np.mean(ear)) if ear else 0.0
    Accbar = float(np.mean(acc)) if acc else 1.0  # no normal datasets -> no false alarms to penalize

    # Reliability: event-level F_beta across ALL datasets.
    tp = sum(1 for e in evals if e.event_true == 1 and e.event_pred == 1)
    fp = sum(1 for e in evals if e.event_true == 0 and e.event_pred == 1)
    fn = sum(1 for e in evals if e.event_true == 1 and e.event_pred == 0)
    EFbeta = f_beta(tp, fp, fn, beta)

    total_w = w1 + w2 + w3 + w4
    WA = (w1 * Fbar + w2 * WSbar + w3 * EFbeta + w4 * Accbar) / total_w

    any_detected = any(e.event_pred == 1 for e in evals)
    if not any_detected:
        care = 0.0
    elif Accbar < 0.5:
        care = Accbar
    else:
        care = WA

    return {
        "CARE": float(care),
        "coverage": Fbar,
        "earliness": WSbar,
        "reliability": EFbeta,
        "accuracy": Accbar,
        "WA": float(WA),
        "n_anomaly_datasets": len(cov),
        "n_normal_datasets": len(acc),
        "event_tp": tp, "event_fp": fp, "event_fn": fn,
    }

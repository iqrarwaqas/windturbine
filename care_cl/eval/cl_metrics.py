"""Continual-learning metrics derived from the stage x farm CARE matrix (§6).

The matrix R has R[stage][farm] = CARE on `farm` after training through `stage`,
for farms seen up to that stage (NaN before a farm is first seen).
"""
from __future__ import annotations

import numpy as np


def forgetting_per_farm(matrix: dict[str, dict[str, float]], sequence: list[str]) -> dict[str, float]:
    """forgetting(f) = CARE(f right after training f) - CARE(f at end of sequence).

    Positive = the model got worse on farm f after learning later farms.
    Defined for every farm except the last one in the sequence.
    """
    last = sequence[-1]
    out = {}
    for f in sequence:
        if f == last:
            continue
        after_learn = matrix.get(f, {}).get(f)
        at_end = matrix.get(last, {}).get(f)
        if after_learn is None or at_end is None:
            continue
        out[f] = float(after_learn - at_end)
    return out


def backward_transfer(matrix: dict[str, dict[str, float]], sequence: list[str]) -> float:
    """Mean over past farms of CARE(end) - CARE(right after learning that farm).

    Negative BWT = forgetting; positive = learning later farms helped earlier ones.
    """
    last = sequence[-1]
    deltas = []
    for f in sequence[:-1]:
        a = matrix.get(f, {}).get(f)
        b = matrix.get(last, {}).get(f)
        if a is not None and b is not None:
            deltas.append(b - a)
    return float(np.mean(deltas)) if deltas else 0.0


def forward_transfer(matrix: dict[str, dict[str, float]], sequence: list[str]) -> float:
    """Mean CARE on a farm measured the stage *before* it is trained (optional).

    Uses the diagonal-minus-one entries when available; 0 if none exist.
    """
    vals = []
    for i, f in enumerate(sequence):
        if i == 0:
            continue
        prev_stage = sequence[i - 1]
        v = matrix.get(prev_stage, {}).get(f)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def final_avg_care(matrix: dict[str, dict[str, float]], sequence: list[str]) -> float:
    last = sequence[-1]
    vals = [matrix[last][f] for f in sequence if f in matrix.get(last, {})]
    return float(np.mean(vals)) if vals else 0.0


def summarize(matrix: dict[str, dict[str, float]], sequence: list[str]) -> dict:
    fpf = forgetting_per_farm(matrix, sequence)
    return {
        "forgetting_per_farm": fpf,
        "mean_forgetting": float(np.mean(list(fpf.values()))) if fpf else 0.0,
        "backward_transfer": backward_transfer(matrix, sequence),
        "forward_transfer": forward_transfer(matrix, sequence),
        "final_avg_care": final_avg_care(matrix, sequence),
    }

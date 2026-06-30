"""Contextual-bandit anomaly-threshold tuner (§7) — supporting ablation, tiny.

Arms = candidate thresholds (quantiles of the training reconstruction error).
Context = recent reconstruction-error statistics (mean + quantiles, optional status).
Reward = CARE-aligned detection benefit minus false-alarm penalty.

Two interchangeable algorithms: epsilon-greedy (default) and LinUCB. Both train in
seconds. `bandit="off"` is realized by simply not using this class (fixed quantile).
"""
from __future__ import annotations

import numpy as np


def context_features(errors: np.ndarray, status_id: float | None = None) -> np.ndarray:
    """Compact context vector from a window of reconstruction errors."""
    q = np.quantile(errors, [0.5, 0.9, 0.99]) if errors.size else np.zeros(3)
    feat = [1.0, float(errors.mean() if errors.size else 0.0),
            float(errors.std() if errors.size else 0.0), *q]
    if status_id is not None:
        feat.append(float(status_id))
    return np.asarray(feat, dtype=np.float64)


class EpsilonGreedyBandit:
    def __init__(self, thresholds: np.ndarray, epsilon: float = 0.1, seed: int = 0):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(len(thresholds))
        self.values = np.zeros(len(thresholds))  # running mean reward per arm

    def select(self, context: np.ndarray | None = None) -> int:
        if self.rng.random() < self.epsilon or self.counts.sum() == 0:
            return int(self.rng.integers(len(self.thresholds)))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def best_threshold(self) -> float:
        arm = int(np.argmax(self.values)) if self.counts.sum() > 0 else len(self.thresholds) // 2
        return float(self.thresholds[arm])


class LinUCB:
    """Disjoint LinUCB over a shared context, one linear model per arm."""

    def __init__(self, thresholds: np.ndarray, dim: int, alpha: float = 1.0, seed: int = 0):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.alpha = alpha
        self.A = [np.eye(dim) for _ in thresholds]
        self.b = [np.zeros(dim) for _ in thresholds]
        self._last_ctx = None

    def select(self, context: np.ndarray) -> int:
        self._last_ctx = context
        scores = []
        for a in range(len(self.thresholds)):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            mean = float(theta @ context)
            ucb = self.alpha * float(np.sqrt(context @ Ainv @ context))
            scores.append(mean + ucb)
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float, context: np.ndarray | None = None):
        x = context if context is not None else self._last_ctx
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def best_threshold(self, context: np.ndarray) -> float:
        return float(self.thresholds[self.select(context)])


def candidate_thresholds(train_errors: np.ndarray, quantiles) -> np.ndarray:
    return np.quantile(train_errors, list(quantiles))


def tune_threshold(train_errors: np.ndarray, reward_fn, cfg: dict,
                   status_id: float | None = None) -> float:
    """Run a short bandit loop and return the chosen threshold.

    `reward_fn(threshold) -> float` should encode the CARE-aligned reward (e.g. a
    proxy computed on a held-out slice of *training* error statistics — never eval).
    """
    qs = cfg.get("candidate_quantiles", [0.90, 0.95, 0.97, 0.99, 0.995, 0.999])
    thresholds = candidate_thresholds(train_errors, qs)
    algo = cfg.get("bandit_algo", "epsilon_greedy")
    ctx = context_features(train_errors, status_id)

    if algo == "linucb":
        bandit = LinUCB(thresholds, dim=len(ctx), alpha=1.0)
        for _ in range(200):
            arm = bandit.select(ctx)
            bandit.update(arm, reward_fn(thresholds[arm]), ctx)
        return bandit.best_threshold(ctx)

    bandit = EpsilonGreedyBandit(thresholds, epsilon=cfg.get("epsilon", 0.1))
    for _ in range(200):
        arm = bandit.select(ctx)
        bandit.update(arm, reward_fn(thresholds[arm]))
    return bandit.best_threshold()

"""Continual-learning strategies (§5).

All six strategies share one configurable training loop, differing only by which
penalties/replay are active:

  naive          - sequential fine-tuning (lower bound)
  joint          - pooled training (handled by the protocol; uses naive loop)
  ewc            - Elastic Weight Consolidation on the shared core
  replay         - experience replay from a per-farm ring buffer
  distill        - LwF distillation against a frozen teacher
  replay_distill - DER++ : replay (inputs + stored teacher targets) + distillation

The replay buffer stores samples in the *model input space* so they remain
comparable across farms (the shared latent encoder is the point of §3).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

_FLAGS = {
    "naive":          dict(ewc=False, replay=False, distill=False, der=False),
    "joint":          dict(ewc=False, replay=False, distill=False, der=False),
    "ewc":            dict(ewc=True,  replay=False, distill=False, der=False),
    "replay":         dict(ewc=False, replay=True,  distill=False, der=False),
    "distill":        dict(ewc=False, replay=False, distill=True,  der=False),
    "replay_distill": dict(ewc=False, replay=True,  distill=True,  der=True),
}


class Strategy:
    def __init__(self, name: str, cfg: dict, device):
        if name not in _FLAGS:
            raise ValueError(f"Unknown strategy {name!r}; choose from {list(_FLAGS)}")
        self.name = name
        self.flags = _FLAGS[name]
        self.cfg = cfg
        self.device = device

        self.buffer: list[tuple[str, np.ndarray, np.ndarray | None]] = []  # (farm, x, teacher_recon)
        self.teacher: nn.Module | None = None
        self.fisher: dict[int, torch.Tensor] = {}
        self.opt_params: dict[int, torch.Tensor] = {}

    # -- public interface ---------------------------------------------------
    def train_on(self, farm_id: str, model, optim, X: np.ndarray) -> dict:
        tcfg = self.cfg["train"]
        epochs = tcfg["epochs"]
        bs = tcfg["batch_size"]
        rng = np.random.default_rng(self.cfg.get("seed", 0))
        X = X.astype(np.float32)
        n = len(X)

        lam_ewc = self.cfg["cl"]["lambda_ewc"]
        lam_dis = self.cfg["cl"]["lambda_distill"]
        replay_frac = self.cfg["cl"]["replay_batch_frac"]

        last_loss = 0.0
        for _ in range(epochs):
            order = rng.permutation(n)
            for i in range(0, n, bs):
                idx = order[i:i + bs]
                xb = torch.as_tensor(X[idx], device=self.device)
                optim.zero_grad()

                recon = model(xb, farm_id)
                loss = nn.functional.mse_loss(recon, xb)

                if self.flags["distill"] and self.teacher is not None:
                    with torch.no_grad():
                        t_recon = self.teacher(xb, farm_id)
                    loss = loss + lam_dis * nn.functional.mse_loss(recon, t_recon)

                if self.flags["replay"] and self.buffer:
                    n_replay = max(1, int(bs * replay_frac))
                    loss = loss + self._replay_loss(model, n_replay, rng, lam_dis)

                if self.flags["ewc"] and self.fisher:
                    loss = loss + lam_ewc * self._ewc_penalty(model)

                loss.backward()
                optim.step()
                last_loss = float(loss.detach().cpu())
        return {"final_train_loss": last_loss}

    def end_of_farm(self, farm_id: str, model, X: np.ndarray):
        """Update Fisher / teacher / replay buffer after finishing a farm."""
        if self.flags["ewc"]:
            self._update_fisher(farm_id, model, X)
        if self.flags["distill"] or self.flags["der"]:
            self.teacher = copy.deepcopy(model).eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        if self.flags["replay"]:
            self._update_buffer(farm_id, model, X)

    # -- replay -------------------------------------------------------------
    def _update_buffer(self, farm_id: str, model, X: np.ndarray):
        k = self.cfg["cl"]["buffer_per_farm"]
        rng = np.random.default_rng(self.cfg.get("seed", 0) + 1)
        sel = rng.choice(len(X), size=min(k, len(X)), replace=False)
        xs = X[sel].astype(np.float32)
        targets = None
        if self.flags["der"]:  # store teacher reconstructions (DER++)
            with torch.no_grad():
                xb = torch.as_tensor(xs, device=self.device)
                targets = model(xb, farm_id).cpu().numpy().astype(np.float32)
        for j in range(len(xs)):
            self.buffer.append((farm_id, xs[j], None if targets is None else targets[j]))

    def _replay_loss(self, model, n_replay, rng, lam_dis) -> torch.Tensor:
        idx = rng.choice(len(self.buffer), size=min(n_replay, len(self.buffer)), replace=False)
        loss = torch.zeros((), device=self.device)
        # group by farm so adapter forwards use the right adapter
        by_farm: dict[str, list[int]] = {}
        for j in idx:
            by_farm.setdefault(self.buffer[j][0], []).append(j)
        for fid, js in by_farm.items():
            xb = torch.as_tensor(np.stack([self.buffer[j][1] for j in js]), device=self.device)
            recon = model(xb, fid)
            loss = loss + nn.functional.mse_loss(recon, xb)
            if self.flags["der"] and self.buffer[js[0]][2] is not None:
                tb = torch.as_tensor(np.stack([self.buffer[j][2] for j in js]), device=self.device)
                loss = loss + lam_dis * nn.functional.mse_loss(recon, tb)
        return loss / max(1, len(by_farm))

    # -- EWC ----------------------------------------------------------------
    def _update_fisher(self, farm_id: str, model, X: np.ndarray):
        model.eval()
        params = model.core_parameters()
        fisher = {i: torch.zeros_like(p) for i, p in enumerate(params)}
        rng = np.random.default_rng(self.cfg.get("seed", 0) + 2)
        n_samples = min(2000, len(X))
        sel = rng.choice(len(X), size=n_samples, replace=False)
        for r in sel:
            xb = torch.as_tensor(X[r:r + 1].astype(np.float32), device=self.device)
            model.zero_grad()
            recon = model(xb, farm_id)
            loss = nn.functional.mse_loss(recon, xb)
            loss.backward()
            for i, p in enumerate(params):
                if p.grad is not None:
                    fisher[i] += p.grad.detach() ** 2
        for i in fisher:
            fisher[i] /= n_samples
            # online EWC: accumulate across farms
            self.fisher[i] = fisher[i] + self.fisher.get(i, torch.zeros_like(fisher[i]))
        self.opt_params = {i: p.detach().clone() for i, p in enumerate(params)}
        model.zero_grad()

    def _ewc_penalty(self, model) -> torch.Tensor:
        params = model.core_parameters()
        pen = torch.zeros((), device=self.device)
        for i, p in enumerate(params):
            if i in self.fisher:
                pen = pen + (self.fisher[i] * (p - self.opt_params[i]) ** 2).sum()
        return pen


def build_strategy(name: str, cfg: dict, device) -> Strategy:
    return Strategy(name, cfg, device)

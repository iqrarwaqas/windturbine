"""Autoencoder normal-behaviour model (§4).

Small fully-connected AE. Input = a (flattened) window of standardized signals.
Reconstruction loss = MSE; anomaly score = per-sample MSE (higher = more anomalous).

Architecture: a *shared* encoder/decoder core (what continual learning protects)
plus optional per-farm input/output adapters (align_mode="adapter").
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # no activation on the final layer
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class AENBM(nn.Module):
    """Shared-core autoencoder with optional per-farm adapters.

    - shared_only: in_dim == shared_dim, no adapters. The core sees shared signals.
    - adapter:     each farm registers a linear adapter (full_dim -> shared_dim) and
                   an output head (shared_dim -> full_dim); the core is farm-agnostic.
    """

    def __init__(self, shared_dim: int, latent_dim: int = 16,
                 hidden_dims: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dims = list(hidden_dims or [64, 32])
        self.shared_dim = shared_dim
        self.latent_dim = latent_dim

        enc_dims = [shared_dim] + hidden_dims + [latent_dim]
        dec_dims = [latent_dim] + hidden_dims[::-1] + [shared_dim]
        self.encoder = _mlp(enc_dims, dropout)
        self.decoder = _mlp(dec_dims, dropout)

        # Per-farm adapters (populated lazily in adapter mode).
        self.in_adapters = nn.ModuleDict()
        self.out_heads = nn.ModuleDict()

    # -- adapter management -------------------------------------------------
    def add_farm_adapter(self, farm_id: str, full_dim: int):
        if farm_id not in self.in_adapters:
            self.in_adapters[farm_id] = nn.Linear(full_dim, self.shared_dim)
            self.out_heads[farm_id] = nn.Linear(self.shared_dim, full_dim)

    def core_parameters(self):
        """Encoder + decoder params — the weights CL strategies protect."""
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor, farm_id: str | None = None) -> torch.Tensor:
        if farm_id is not None and farm_id in self.in_adapters:
            h = self.in_adapters[farm_id](x)
            z = self.encoder(h)
            h_rec = self.decoder(z)
            return self.out_heads[farm_id](h_rec)
        z = self.encoder(x)
        return self.decoder(z)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, farm_id: str | None = None) -> torch.Tensor:
        """Per-sample reconstruction MSE (in the model's input space)."""
        recon = self.forward(x, farm_id)
        return ((recon - x) ** 2).mean(dim=1)


def build_model(input_dim: int, cfg: dict):
    m = cfg.get("model", {})
    if m.get("type", "ae") == "transformer":
        from .transformer_nbm import TransformerNBM
        return TransformerNBM(
            shared_dim=input_dim,
            latent_dim=m.get("latent_dim", 16),
            window_len=cfg.get("align", {}).get("window_len", 1),
            dropout=m.get("dropout", 0.0),
        )
    return AENBM(
        shared_dim=input_dim,
        latent_dim=m.get("latent_dim", 16),
        hidden_dims=m.get("hidden_dims", [64, 32]),
        dropout=m.get("dropout", 0.0),
    )


@torch.no_grad()
def reconstruction_errors(model: AENBM, x: np.ndarray, device, farm_id=None,
                          batch_size: int = 4096) -> np.ndarray:
    """Vectorized per-row anomaly scores for a numpy matrix."""
    model.eval()
    out = np.empty(len(x), dtype=np.float64)
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device)
        out[i:i + batch_size] = model.anomaly_score(xb, farm_id).cpu().numpy()
    return out

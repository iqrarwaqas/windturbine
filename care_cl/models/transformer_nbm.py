"""Optional transformer reconstructor (§4, ablation only — not on the core path).

A tiny temporal transformer autoencoder over a window of shared signals. Exposed
behind cfg.model.type == "transformer". Kept minimal; the core results use AENBM.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerNBM(nn.Module):
    def __init__(self, shared_dim: int, latent_dim: int = 16, n_heads: int = 2,
                 n_layers: int = 2, window_len: int = 1, dropout: float = 0.0):
        super().__init__()
        self.shared_dim = shared_dim
        self.window_len = window_len
        d_model = max(16, latent_dim)
        self.embed = nn.Linear(shared_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, shared_dim)
        self.in_adapters = nn.ModuleDict()
        self.out_heads = nn.ModuleDict()

    def add_farm_adapter(self, farm_id: str, full_dim: int):  # parity with AENBM
        if farm_id not in self.in_adapters:
            self.in_adapters[farm_id] = nn.Linear(full_dim, self.shared_dim)
            self.out_heads[farm_id] = nn.Linear(self.shared_dim, full_dim)

    def core_parameters(self):
        return list(self.embed.parameters()) + list(self.encoder.parameters()) \
            + list(self.out.parameters())

    def forward(self, x: torch.Tensor, farm_id: str | None = None) -> torch.Tensor:
        # x: (B, F) single timestep -> treat as length-1 sequence.
        seq = x.unsqueeze(1) if x.dim() == 2 else x
        h = self.embed(seq)
        h = self.encoder(h)
        out = self.out(h)
        return out.squeeze(1) if x.dim() == 2 else out

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, farm_id: str | None = None) -> torch.Tensor:
        recon = self.forward(x, farm_id)
        return ((recon - x) ** 2).mean(dim=1)

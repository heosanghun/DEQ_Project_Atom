#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/g2v4_proj_v2.py - PrefixProjectorV2 (CC, 2026-10-11). Latent -> LM interface with scale control.

Why (DIAG-G2v4 §8, g2v4_path_probe.py): the v3.6 PrefixProjector output had slot norm 1,258-1,945 (real token embeddings: 50.5)
and an item-independent constant dominated it (item-specific energy share 2-4 % after the adapter vs 57.6 % at the input),
so every decoder learned to ignore latent content.

V2:  x = z - center (fixed per-slot mean of the training start latents)  ->  RMS-normalise per slot  ->  w_proj (identity init)
     -> MLPAdapter (zero-init residual)  ->  RMS-normalise per slot  ->  * out_rms (learnable, init = real token-embedding RMS).
"""
import torch
import torch.nn as nn

from g2v3_m3_full_train import MLPAdapter


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class PrefixProjectorV2(nn.Module):
    proj_version = 2

    def __init__(self, d_model: int = 2560, k_latent: int = 64, out_rms: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("center", torch.zeros(k_latent, d_model))
        self.w_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.w_proj.weight.copy_(torch.eye(d_model))
        self.adapter = MLPAdapter(d_model=d_model, hidden_dim=d_model)
        self.out_rms = nn.Parameter(torch.tensor(float(out_rms)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = _rms_norm(z.float() - self.center)
        x = self.adapter(self.w_proj(x))
        return _rms_norm(x) * self.out_rms

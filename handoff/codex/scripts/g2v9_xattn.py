#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/g2v9_xattn.py - X3-min: minimal gated cross-attention latent bridge (CC 2026-10-19).
Plan: shared/proposals/X_INTERFACE_PLAN_20261019.md (X3). Deliberately simple and symmetric:
  * no prefix slots in the prompt (pure C-fmt prompt);
  * at a few decoder layers, hidden += tanh(gate_l) * O_l( Attn( Q_l(h), K_l(z~), V_l(z~) ) ), applied at every position;
  * z~ = RMS-normalised (z - center), center = per-slot mean of the TRAINING start latents of the same condition;
  * gate_l = 0 at start (tanh) -> bit-identical to pure Gemma at step 0; O_l default init so the gate gets gradient (rev2);
  * no matcher, no K-dependent gating, no slot readout: the same mechanism for A (z*'), E (z_src), oracle and shuffled.
`python g2v9_xattn.py --selftest` runs a CPU mock check (identity at init, shapes, grad flow)."""
from __future__ import annotations
import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class XAttnAdapter(nn.Module):
    def __init__(self, d_model: int = 2560, n_heads: int = 8, d_head: int = 64, d_v: int = 64) -> None:
        super().__init__()
        self.n_heads, self.d_head, self.d_v = n_heads, d_head, d_v
        self.q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.o = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))  # tanh(0) = 0 -> exact identity at init (Flamingo-style)
        # rev2 (2026-10-19 17:4x): O keeps its default init. Zero O AND zero gate gave a dead start (d/dgate = O(ctx) = 0,
        # d/dO = tanh(0) = 0): rev1 runs sat at gates [0,0,0,0] with identical loss for O/A/E at step 300.

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D) any dtype; z: (B, K, D) float32 (already normalised)
        B, T, _ = h.shape
        K = z.shape[1]
        hq = h.float()
        q = self.q(hq).view(B, T, self.n_heads, self.d_head).transpose(1, 2)        # (B, H, T, dh)
        k = self.k(z).view(B, K, self.n_heads, self.d_head).transpose(1, 2)         # (B, H, K, dh)
        v = self.v(z).view(B, K, self.n_heads, self.d_v).transpose(1, 2)            # (B, H, K, dv)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.d_head), dim=-1)  # (B, H, T, K)
        ctx = (att @ v).transpose(1, 2).reshape(B, T, self.n_heads * self.d_v)
        return torch.tanh(self.gate) * self.o(ctx)                                   # (B, T, D) float32


class LatentXAttnBridge(nn.Module):
    def __init__(self, layers: List[int], d_model: int = 2560, k_latent: int = 64, n_heads: int = 8, d_head: int = 64, d_v: int = 64) -> None:
        super().__init__()
        self.layers = [int(x) for x in layers]
        self.adapters = nn.ModuleDict({str(l): XAttnAdapter(d_model, n_heads, d_head, d_v) for l in self.layers})
        self.register_buffer("center", torch.zeros(k_latent, d_model))
        self._z: Optional[torch.Tensor] = None
        self._handles = []

    def set_latent(self, z_raw: Optional[torch.Tensor]) -> None:
        """z_raw: (K, D) or (1, K, D) raw latent, or None to disable injection."""
        if z_raw is None:
            self._z = None
            return
        z = z_raw.float().reshape(1, self.center.shape[0], self.center.shape[1]).to(self.center.device)
        self._z = rms_norm(z - self.center.unsqueeze(0))

    def _hook(self, adapter: XAttnAdapter):
        def fn(module, args, output):
            if self._z is None:
                return output
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            delta = adapter(h, self._z.to(h.device)).to(h.dtype)
            h_new = h + delta
            return (h_new,) + tuple(output[1:]) if is_tuple else h_new
        return fn

    def attach(self, lm_model: nn.Module) -> None:
        self.detach()
        for l in self.layers:
            self._handles.append(lm_model.layers[l].register_forward_hook(self._hook(self.adapters[str(l)])))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def n_params(self) -> int:
        return sum(p.numel() for p in self.adapters.parameters())


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        torch.manual_seed(0)
        D, K = 32, 8

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__(); self.lin = nn.Linear(D, D)
            def forward(self, x):
                return (self.lin(x), None)

        class MockLM(nn.Module):
            def __init__(self):
                super().__init__(); self.layers = nn.ModuleList([MockLayer() for _ in range(4)])
            def forward(self, x):
                for l in self.layers:
                    x = l(x)[0]
                return x

        lm = MockLM()
        br = LatentXAttnBridge([1, 3], d_model=D, k_latent=K, n_heads=2, d_head=8, d_v=8)
        x = torch.randn(1, 5, D)
        ref = lm(x).clone()
        br.attach(lm)
        br.set_latent(torch.randn(K, D))
        y0 = lm(x)
        assert torch.equal(ref, y0), "not identity at init"
        y0.sum().backward()
        gg = [float(a.gate.grad.abs()) for a in br.adapters.values()]
        assert all(g > 0 for g in gg), f"gate gradient dead at init: {gg}"
        br.zero_grad(); lm.zero_grad()
        for a in br.adapters.values():
            nn.init.normal_(a.o.weight, std=0.1); a.gate.data.fill_(1.0)
        y1 = lm(x)
        assert not torch.allclose(ref, y1), "no effect after opening gate"
        y1.sum().backward()
        g = sum(float(p.grad.abs().sum()) for p in br.adapters.parameters() if p.grad is not None)
        assert g > 0, "no gradient"
        br.set_latent(None)
        assert torch.equal(ref, lm(x)), "latent None should be identity"
        br.detach()
        print("[SELFTEST-OK] identity-at-init, gate-grad-at-init, effect-after-open, grad-flow, none-latent-identity; params", br.n_params())

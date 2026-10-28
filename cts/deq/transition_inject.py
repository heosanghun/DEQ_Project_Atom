#!/usr/bin/env python3
"""
cts/deq/transition_inject.py - Continuous Input Injection Wrapper for CTS Latent Transition (Version 4)
Directive: EXE-260913-01 (System 2 / Ledger #223)

Version 4 Formulation:
  Fixed constant-scale injection with damping and optional fp32 accumulation:
    z_{k+1} = (1 - alpha) * z_k + alpha * (Phi(cast(z_k)) + g0 * c_hat_0)
  where:
    - Phi(cast(z_k)) allows casting z_k to phi_cast_dtype (e.g. torch.bfloat16) before calling backbone.deq_step
    - The outer update (injection and damping) is strictly accumulated in torch.float32 when accum_fp32=True
    - Retains exact backward compatibility with v3 when phi_cast_dtype=None and accum_fp32=False
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights


def compute_broadcast_context_c(
    context: Any,
    K: int = 64,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Extract prompt token mean c(x) and broadcast across K tokens."""
    if hasattr(context, "prompt_embeds"):
        embeds = context.prompt_embeds  # [1, S0, H]
    elif isinstance(context, torch.Tensor):
        embeds = context
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported context type for injection: {type(context)}")

    # Prompt token mean across sequence length S0
    token_mean = embeds[0].float().mean(dim=0, keepdim=True)  # [1, H]
    c_x = token_mean.expand(K, -1)  # [K, H]

    if device is not None:
        c_x = c_x.to(device=device)
    if dtype is not None:
        c_x = c_x.to(dtype=dtype)
    return c_x


def compute_c_hat_0(
    c_x: torch.Tensor,
    z0: torch.Tensor,
) -> torch.Tensor:
    """Compute constant injection vector c_hat_0 from initial state z0."""
    token_c_norm = torch.norm(c_x[0], p=2).clamp_min(1e-12)
    c_unit = c_x / token_c_norm
    z0_token_norms = torch.norm(z0, p=2, dim=-1)
    z0_mean_token_norm = z0_token_norms.mean()
    c_hat_0 = c_unit * z0_mean_token_norm
    return c_hat_0


class InjectedTransitionStep:
    """Wrapper that calls base Phi and adds constant input injection with damping."""

    def __init__(
        self,
        backbone: Any,
        context: Any,
        w_g: torch.Tensor,
        c_hat_0: torch.Tensor,
        g0: float = 0.25,
        alpha: float = 0.5,
        top_k: int = 3,
        routing_mode: str = "sparse",
        phi_cast_dtype: Optional[torch.dtype] = None,
        accum_fp32: bool = False,
    ):
        self.backbone = backbone
        self.context = context
        self.w_g = w_g
        self.c_hat_0 = c_hat_0
        self.g0 = float(g0)
        self.alpha = float(alpha)
        self.top_k = top_k
        self.routing_mode = routing_mode
        self.phi_cast_dtype = phi_cast_dtype
        self.accum_fp32 = accum_fp32

        # Precompute constant injection term: g0 * c_hat_0
        self.const_inj = self.g0 * self.c_hat_0

        # Diagnostics storage for hooks / inspection
        self.last_r_ids: Optional[List[int]] = None
        self.last_sel_ids: Optional[torch.Tensor] = None
        self.last_f_weights: Optional[torch.Tensor] = None

    def __call__(self, z_cur: torch.Tensor) -> torch.Tensor:
        """Single Picard transition step with constant input injection and damping.

        z_{k+1} = (1 - alpha) * z_cur + alpha * (Phi(cast(z_cur)) + g0 * c_hat_0)
        """
        device = z_cur.device
        dtype = torch.float32 if self.accum_fp32 else z_cur.dtype

        # 1. Compute MoE routing weights
        if self.routing_mode == "dense":
            f_weights = routing_weights(z_cur.float(), self.w_g.to(device), 1.0)
            ex: Dict[str, Any] = {"top_k": self.top_k, "deq_map_mode": self.backbone.deq_map_mode}
        else:
            alpha_r = routing_weights(z_cur.float(), self.w_g.to(device), 1.0)
            f_weights = sparse_module_weights(alpha_r, self.top_k)
            sel_ids = torch.sort(
                torch.topk(f_weights.reshape(-1), self.top_k).indices.to(dtype=torch.long)
            ).values
            r_ids = [int(x) for x in sel_ids.detach().cpu().tolist()]
            ex = {
                "top_k": self.top_k,
                "deq_map_mode": self.backbone.deq_map_mode,
                "routing_module_ids": r_ids,
                "routing_module_ids_tensor": sel_ids,
            }
            self.last_r_ids = r_ids
            self.last_sel_ids = sel_ids
            self.last_f_weights = f_weights

        # 2. Context static inputs hook
        prepare_hook = getattr(self.backbone, "prepare_deq_context", None)
        if callable(prepare_hook) and "r_ids" in locals():
            ex["eq3_static_inputs"] = prepare_hook(z_cur, self.context, r_ids)

        # 3. Base Phi step (with optional precision casting for Phi input)
        z_phi = z_cur.to(dtype=self.phi_cast_dtype) if self.phi_cast_dtype is not None else z_cur
        phi_z = self.backbone.deq_step(z_phi, self.context, f_weights, ex).float()

        # 4. Constant Input Injection and Damping (accumulated in fp32)
        inj = self.const_inj.to(device=device, dtype=torch.float32)
        injected = phi_z + inj

        if self.alpha == 1.0:
            z_next = injected.to(dtype=dtype)
        else:
            z_next = ((1.0 - self.alpha) * z_cur.float() + self.alpha * injected).to(dtype=dtype)

        return z_next


def run_picard_trajectory_injection_v3(
    backbone: Any,
    prompt_text: str,
    g0: float = 0.25,
    alpha: float = 0.5,
    seed: int = 2030,
    nfe_steps: int = 300,
    k_latent: int = 64,
    sigma_noise: float = 0.020,
    device: Optional[Union[torch.device, str]] = None,
    phi_cast_dtype: Optional[torch.dtype] = None,
    accum_fp32: bool = False,
) -> Dict[str, Any]:
    """Execute 300-step Picard iteration with constant input injection and damping."""
    dev = device or next(backbone.parameters()).device
    d = backbone.hidden_size
    w_g = backbone.routing_matrix().to(device=dev, dtype=torch.float32)

    # Encode context
    ctx = backbone.encode_context(prompt_text)
    if hasattr(ctx, "to"):
        ctx = ctx.to(device=dev)

    # Compute broadcast context mean c(x) [64, 2560]
    c_x = compute_broadcast_context_c(ctx, K=k_latent, device=dev, dtype=torch.float32)

    # Prompt-conditioned root warm start
    z_src = backbone.root_fwdpass(ctx, K=k_latent, output_dtype=torch.float32).to(device=dev)

    # Sample initial state z0 with deterministic seed
    torch.manual_seed(seed)
    noise = torch.randn(k_latent, d, device=dev, dtype=torch.float32) * sigma_noise
    z0 = (z_src + noise).clone()

    # Compute constant c_hat_0 from z0
    c_hat_0 = compute_c_hat_0(c_x, z0).to(device=dev, dtype=torch.float32)

    step_fn = InjectedTransitionStep(
        backbone=backbone,
        context=ctx,
        w_g=w_g,
        c_hat_0=c_hat_0,
        g0=g0,
        alpha=alpha,
        top_k=3,
        routing_mode="sparse",
        phi_cast_dtype=phi_cast_dtype,
        accum_fp32=accum_fp32,
    )

    z_traj: List[torch.Tensor] = []
    res_traj: List[float] = []
    z_cur = z0.clone()

    with torch.no_grad():
        for step_i in range(1, nfe_steps + 1):
            if torch.isnan(z_cur).any():
                rem = nfe_steps - (step_i - 1)
                res_traj.extend([float('nan')] * rem)
                z_traj.extend([z_cur.cpu()] * rem)
                break

            z_next = step_fn(z_cur).float()
            diff = torch.norm(z_next - z_cur).item()
            res_traj.append(diff)
            z_traj.append(z_next.cpu())
            z_cur = z_next

    return {
        "z_traj": z_traj,
        "res_traj": res_traj,
        "c_x": c_x.cpu(),
        "c_hat_0": c_hat_0.cpu(),
        "prompt_token_mean": c_x[0].cpu(),
        "z_src": z_src.cpu(),
        "z0": z0.cpu(),
        "final_z": z_traj[-1],
    }

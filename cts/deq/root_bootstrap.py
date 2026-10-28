"""Shared prompt-conditioned root fixed-point bootstrap for train and eval."""

from __future__ import annotations

from typing import Any, Optional

import torch

from cts.deq.broyden_forward import (
    MAX_DENSE_N,
    LimitedMemoryInverseJacobian,
)
from cts.deq.transition import transition
from cts.latent.bottleneck import init_z0
from cts.types import NuVector, RuntimeBudgetState, TransitionResult


def solve_root_z_star(
    prompt: str,
    *,
    backbone,
    context: Any,
    K: int,
    d: int,
    broyden_max_iter: int,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    top_k: int = 3,
    tau_flops_budget: float = 1e14,
    seed: int = 2026,
    device: Optional[torch.device] = None,
    require_converged: bool = True,
    phase_profiler: Optional[Any] = None,
    profile_scope: str = "root.transition",
) -> TransitionResult:
    """Compute the root state before MetaPolicy acts.

    The old train/eval paths fed MetaPolicy a prompt-independent random z0.
    Algorithm 1 instead starts from a prompt-conditioned root fixed point.
    This helper is deliberately shared so Stage-2 and headline evaluation
    cannot drift to different actor-state distributions.
    """
    if device is None:
        device = next(backbone.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    root_fwdpass = getattr(backbone, "root_fwdpass", None)
    if callable(root_fwdpass):
        # Algorithm 1 line 1 is direct: ``z*_0 <- FwdPass(s_0)`` followed by
        # ``B_0 <- 0.1 I`` and ``MAC <- 0``.  Running an additional sparse
        # Eq. (3) Broyden solve here changes the policy state, may reject a
        # perfectly valid prompt state on non-convergence, and charges up to
        # 30 off-paper Phi evaluations before the tree search begins.
        with torch.no_grad():
            z0 = root_fwdpass(context, K=K, output_dtype=torch.float32)
        if z0.shape != (K, d):
            raise RuntimeError(
                "root initialization shape mismatch: "
                f"{tuple(z0.shape)} vs {(K, d)}"
            )

        n = int(z0.numel())
        if n <= MAX_DENSE_N:
            inv_jacobian = 0.1 * torch.eye(
                n, device=z0.device, dtype=torch.float32
            )
        else:
            empty = torch.empty((0, n), device=z0.device, dtype=torch.float32)
            inv_jacobian = LimitedMemoryInverseJacobian(
                scale=0.1,
                u=empty,
                v=empty.clone(),
                validated=True,
            )

        return TransitionResult(
            child_text=None,
            z_star_child=z0.detach().float(),
            solver_stats={
                "root_initialization": "prompt_full_stack",
                "root_bootstrap_mode": "algorithm1_direct_fwdpass",
                "iterations": 0,
                "phi_eval_count": 0,
                "converged": True,
                "residual_norm": float("nan"),
                "absolute_residual_norm": float("nan"),
                "failure_reason": None,
                "failure_fallback": None,
                "inv_jacobian": inv_jacobian,
                # Root FwdPass is deliberately outside the outer Eq. (3) MAC
                # budget in Algorithm 1.  Wall-clock profiling accounts for
                # its real execution separately.
                "logical_mac_transition": 0.0,
                "mac_unit": "MAC",
                "mac_headline_eligible": True,
                "mac_lut_schema_version": 0,
                "mac_lut_sha256": "",
            },
            prune=False,
            budget=RuntimeBudgetState(),
        )
    else:
        # Compatibility path for non-Gemma diagnostic backbones.  Headline
        # Gemma exposes ``root_fwdpass`` and must never route from this noise.
        z0 = init_z0(K, d, device, generator)
        root_initialization = "random_compatibility"
    if z0.shape != (K, d):
        raise RuntimeError(
            "root initialization shape mismatch: "
            f"{tuple(z0.shape)} vs {(K, d)}"
        )
    bootstrap_nu = NuVector(
        nu_expl=1.0,
        nu_tol=5e-3,
        nu_temp=1.0,
        nu_act=1.0,
    )
    with torch.no_grad():
        result = transition(
            prompt,
            0,
            bootstrap_nu,
            RuntimeBudgetState(),
            backbone,
            K=K,
            d=d,
            top_k=top_k,
            broyden_max_iter=broyden_max_iter,
            broyden_tol_min=broyden_tol_min,
            broyden_tol_max=broyden_tol_max,
            tau_flops_budget=tau_flops_budget,
            generator=generator,
            parent_z_star=z0,
            noise_sigma=0.0,
            context_override=context,
            decode_child=False,
            max_decode_tokens=0,
            phase_profiler=phase_profiler,
            profile_scope=profile_scope,
            fallback_to_parent_on_failure=False,
        )
    result.solver_stats["root_initialization"] = root_initialization
    if require_converged and not bool(result.solver_stats.get("converged", False)):
        raise RuntimeError(
            "root fixed-point bootstrap did not converge: "
            f"{result.solver_stats}"
        )
    return result

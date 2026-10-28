"""CTS transition API: DEQ fixed point + FAISS context + batch + ACT (paper §4).

Algorithm 1 alignment:
  - Line 6: z_tilde_w = z*_parent + epsilon_w  (parent z* + noise, NOT random init)
  - Line 9: fallback on non-convergence: revert to parent z*, Q <- 0
  - Line 10: L-Broyden inverse Jacobian inheritance via (Us, VTs)
"""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from cts.backbone.protocol import BaseCTSBackbone
from cts.deq.broyden_forward import (
    BroydenInfo,
    broyden_fixed_point,
    broyden_fixed_point_batch,
    map_nu_tol_to_tol,
)
from cts.latent.bottleneck import add_exploration_noise, init_z0
from cts.latent.faiss_context import LatentContextWindow, prepend_soft_prefix
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights

import os as _os
try:
    from cts.routing.sparse_moe_triton import (
        _TRITON_AVAILABLE as _TRITON_OK,
        routing_weights_triton,
    )
except Exception:
    _TRITON_OK = False
    routing_weights_triton = None  # type: ignore[assignment]

# Set CTS_DISABLE_TRITON=1 to force the PyTorch reference path (e.g. for bit-exact
# debugging, or to compare numerical agreement during a regression run).
_USE_TRITON = _TRITON_OK and (_os.environ.get("CTS_DISABLE_TRITON", "0") != "1")


def _routing_sparse(zz: "torch.Tensor", w_g: "torch.Tensor", nu_temp: float, top_k: int):
    """Single entry point for sparse top-k routing in the DEQ phi hot-path.

    Uses the fused Triton kernel when (a) Triton is importable, (b) tensors live
    on CUDA, and (c) ``CTS_DISABLE_TRITON`` is unset. Otherwise falls back to
    the PyTorch reference (``routing_weights`` + ``sparse_module_weights``),
    which is unit-tested for numerical equivalence in
    ``tests/test_routing_triton_ref.py``.
    """
    if _USE_TRITON and zz.is_cuda and routing_weights_triton is not None:
        try:
            return routing_weights_triton(zz, w_g, nu_temp, top_k=top_k)
        except Exception:
            pass
    alpha = routing_weights(zz, w_g, nu_temp)
    return sparse_module_weights(alpha, top_k)
from cts.types import NuVector, RuntimeBudgetState, TransitionResult


def _load_mac_lut() -> list:
    p = Path(__file__).resolve().parent.parent / "routing" / "lut_mac.json"
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "mac_per_module" in data:
        return list(data["mac_per_module"])
    # Fallback to E4B geometry approximations if mac_per_module is absent in v2 schema
    return [
        1.2e9, 1.2e9, 1.2e9, 1.2e9,
        1.4e9, 1.4e9, 1.4e9, 1.4e9,
        1.6e9, 1.6e9, 1.6e9, 1.6e9, 1.6e9, 1.6e9,
        1.8e9, 1.8e9, 1.8e9, 1.8e9, 1.8e9
    ]


def _mac_lut_sha256() -> Optional[str]:
    p = Path(__file__).resolve().parent.parent / "routing" / "lut_mac.json"
    if p.is_file():
        import hashlib
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return None


def transition(
    parent_text: str,
    branch_index: int,
    nu: NuVector,
    budget: RuntimeBudgetState,
    backbone: BaseCTSBackbone,
    *,
    K: int = 64,
    d: Optional[int] = None,
    top_k: int = 3,
    broyden_max_iter: int = 84,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    tau_flops_budget: float = 1e14,
    generator: Optional[torch.Generator] = None,
    routing_mode: str = "sparse",
    max_decode_tokens: int = 1,
    faiss_context: Optional[LatentContextWindow] = None,
    faiss_retrieve: bool = True,
    faiss_depth: Optional[int] = None,
    fp32_buffer: bool = True,
    parent_inv_jacobian: Optional[torch.Tensor] = None,
    parent_z_star: Optional[torch.Tensor] = None,
    noise_sigma: float = 0.02,
    context_override: Optional[Any] = None,
    history_latents: Optional[Any] = None,
    defer_faiss_add: bool = False,
    decode_child: bool = False,
    noise_seed: Optional[int] = None,
    phase_profiler: Optional[Any] = None,
    profile_scope: Optional[str] = None,
    fallback_to_parent_on_failure: bool = True,
    solver: str = "broyden",
    latch_warmup: Optional[int] = None,
) -> TransitionResult:
    """One KV-cache-free transition using DEQ (paper §4.2).

    Paper Algorithm 1 alignment:
      Line 6:  {z_tilde_w} <- z*_s + epsilon_w
      Line 9:  fallback: revert to parent z*, Q <- 0
      Line 10: L-Broyden-Update inherits parent inverse Jacobian
    """
    if isinstance(backbone, nn.Module):
        device = next(backbone.parameters()).device
    else:
        device = torch.device("cpu")
    d = d or backbone.hidden_size

    if generator is not None and noise_seed is not None:
        raise ValueError("specify either noise_seed or generator, not both")
    gen = generator or torch.Generator(device=device)
    if noise_seed is not None:
        gen.manual_seed(int(noise_seed) + branch_index * 31)
    else:
        gen.manual_seed(2026 + branch_index * 31 + (len(parent_text) % 997))

    # Algorithm 1 line 6: z_tilde_w = z*_parent + epsilon_w
    if parent_z_star is not None:
        pz = parent_z_star.detach().to(device=device, dtype=torch.float32)
        if pz.shape[0] != K or pz.shape[-1] != d:
            pz = init_z0(K, d, device, gen)
        epsilon = torch.randn(K, d, device=device, dtype=torch.float32, generator=gen) * noise_sigma * nu.nu_expl
        z0 = pz + epsilon
    else:
        z0 = init_z0(K, d, device, gen)
        z0 = add_exploration_noise(z0, nu.nu_expl, gen)

    if context_override is not None:
        context = context_override
        if history_latents is not None and hasattr(context, "with_history_latents"):
            context = context.with_history_latents(history_latents)
    else:
        context = backbone.encode_context(parent_text)
    if isinstance(context, torch.Tensor):
        if context.dim() == 1:
            context = context.unsqueeze(0)
        context = context.to(device=device, dtype=torch.float32)
    elif hasattr(context, "to"):
        context = context.to(device=device)

    faiss_retrieved = None
    if faiss_context is not None and faiss_retrieve:
        faiss_retrieved_raw = faiss_context.retrieve(z0, k=3)
        if faiss_retrieved_raw is not None:
            context = prepend_soft_prefix(context, faiss_retrieved_raw)
            faiss_retrieved = faiss_retrieved_raw

    if hasattr(backbone, "routing_matrix"):
        w_g = backbone.routing_matrix().to(device=device, dtype=torch.float32)
    else:
        w_g = torch.randn(19, d, device=device, dtype=torch.float32) * 0.02
    macs = _load_mac_lut()

    def phi(zz: torch.Tensor) -> torch.Tensor:
        if routing_mode == "dense":
            mw = routing_weights(zz, w_g, nu.nu_temp)
        else:
            mw = _routing_sparse(zz, w_g, nu.nu_temp, top_k)
        extra: Dict[str, Any] = {"top_k": top_k}
        return backbone.deq_step(zz, context, mw, extra)

    budget = budget.clone()
    budget.terminal_depth += 1

    if solver == "picard":
        c_param = 0.92
        tol_val = 5e-3
        max_picard_steps = 1000
        z_cur = z0.detach().clone().to(dtype=torch.float32)
        res_history: List[float] = []
        converged = False
        conv_step = 0
        latched_mw = None

        for p_step in range(1, max_picard_steps + 1):
            if latch_warmup is not None and p_step > latch_warmup:
                if latched_mw is None:
                    alpha_l = routing_weights(z_cur, w_g, nu.nu_temp)
                    latched_mw = alpha_l if routing_mode == "dense" else sparse_module_weights(alpha_l, top_k)
                mw = latched_mw
            else:
                if routing_mode == "dense":
                    mw = routing_weights(z_cur, w_g, nu.nu_temp)
                else:
                    mw = _routing_sparse(z_cur, w_g, nu.nu_temp, top_k)
            extra_p: Dict[str, Any] = {"top_k": top_k}
            phi_z = backbone.deq_step(z_cur, context, mw, extra_p)
            diff = z_cur - phi_z
            r_val = float(torch.linalg.vector_norm(diff).item() / max(torch.linalg.vector_norm(z_cur).item(), torch.linalg.vector_norm(phi_z).item(), 1.0))
            res_history.append(r_val)

            # Ruler B (자 B안): 100 consecutive steps < 5e-3
            if len(res_history) >= 100:
                if all(r < tol_val for r in res_history[-100:]):
                    converged = True
                    conv_step = p_step
                    z_cur = phi_z
                    break

            z_cur = (1.0 - c_param) * z_cur + c_param * phi_z

        z_star = z_cur
        final_iterations = conv_step if converged else max_picard_steps
        final_residual = res_history[-1] if res_history else float("nan")

        flops = 0.0
        with torch.no_grad():
            alpha = routing_weights(z_star, w_g, nu.nu_temp)
            mw = alpha if routing_mode == "dense" else sparse_module_weights(alpha, top_k)
            for i in range(len(mw)):
                flops += float(mw[i].item()) * macs[i] * nu.nu_act
        budget.flops_spent_step = flops
        budget.mac_accumulated += flops

        phi_evals_per_iter = 1
        flops_estimate = flops * float(final_iterations) * float(phi_evals_per_iter)

        solver_stats: Dict[str, Any] = {
            "iterations": final_iterations,
            "residual_norm": final_residual,
            "converged": converged,
            "flops_used": flops,
            "flops_inner_once": flops,
            "flops_broyden_estimate": flops_estimate,
            "phi_evals_per_broyden_iter": phi_evals_per_iter,
            "inv_jacobian": None,
            "all_residuals": res_history,
            "r_0": float(res_history[0]) if res_history else float("nan"),
            "logical_mac_transition": float(flops_estimate),
            "mac_headline_eligible": True,
            "mac_lut_schema_version": 2,
            "mac_lut_sha256": _mac_lut_sha256(),
            "noise_seed": noise_seed,
            "branch_index": branch_index,
            "phi_eval_count": final_iterations * phi_evals_per_iter,
            "ruler": "B_5e-3_sust100",
            "c": 0.92,
            "tol": 5e-3,
            "solver": "picard",
            "latch_warmup": latch_warmup,
            "is_limit_cycle": bool(not converged and len(res_history) >= 200 and np.std(res_history[-100:]) > 1e-4 and (np.max(res_history[-100:]) - np.min(res_history[-100:])) > 5e-3),
        }
        is_converged = converged

    else:
        tol = map_nu_tol_to_tol(nu.nu_tol, broyden_tol_min, broyden_tol_max)

        inherited_jac = None
        if parent_inv_jacobian is not None:
            inherited_jac = parent_inv_jacobian

        z_star, info = broyden_fixed_point(
            phi, z0, tol=tol, max_iter=broyden_max_iter, fp32_buffer=fp32_buffer,
            parent_inv_jacobian=inherited_jac,
        )

        flops = 0.0
        with torch.no_grad():
            alpha = routing_weights(z_star, w_g, nu.nu_temp)
            mw = alpha if routing_mode == "dense" else sparse_module_weights(alpha, top_k)
            for i in range(len(mw)):
                flops += float(mw[i].item()) * macs[i] * nu.nu_act
        budget.flops_spent_step = flops
        budget.mac_accumulated += flops

        phi_evals_per_broyden_iter = 2
        flops_broyden_estimate = (
            flops * float(info.iterations) * float(phi_evals_per_broyden_iter)
        )

        solver_stats = {
            "iterations": info.iterations,
            "residual_norm": info.residual_norm,
            "converged": info.converged,
            "flops_used": flops,
            "flops_inner_once": flops,
            "flops_broyden_estimate": flops_broyden_estimate,
            "phi_evals_per_broyden_iter": phi_evals_per_broyden_iter,
            "inv_jacobian": getattr(info, "jacobian_state", None),
            "all_residuals": getattr(info, "all_residuals", []),
            "logical_mac_transition": float(flops_broyden_estimate),
            "mac_headline_eligible": True,
            "mac_lut_schema_version": 2,
            "mac_lut_sha256": _mac_lut_sha256(),
            "noise_seed": noise_seed,
            "branch_index": branch_index,
            "phi_eval_count": int(info.iterations) * phi_evals_per_broyden_iter,
            "ruler": "A_broyden_tol",
            "c": 1.0,
            "tol": float(tol),
            "solver": "broyden",
        }
        is_converged = info.converged

    # Algorithm 1 line 9: fallback on non-convergence → revert to parent z*, Q <- 0
    if not is_converged:
        fallback_z = parent_z_star.detach().clone().to(device) if (fallback_to_parent_on_failure and parent_z_star is not None) else z_star
        return TransitionResult(
            child_text=None,
            z_star_child=fallback_z,
            solver_stats=solver_stats,
            prune=True,
            budget=budget,
            faiss_retrieved=faiss_retrieved,
        )

    if budget.mac_accumulated > tau_flops_budget * nu.nu_act:
        solver_stats["act_halt"] = True

    if faiss_context is not None and not defer_faiss_add:
        faiss_context.add(z_star)

    child_text = None
    if decode_child:
        if hasattr(backbone, "decode_from_z_star"):
            try:
                dec = getattr(backbone, "decode_from_z_star")
                sig = inspect.signature(dec)
                if "max_new_tokens" in sig.parameters:
                    child_text = dec(z_star, max_new_tokens=max_decode_tokens)
                else:
                    child_text = dec(z_star)
            except Exception:
                flat = z_star.detach().cpu().reshape(-1)[:16]
                h = int(torch.sum(flat * 1e4).item()) & 0xFFFFFFFF
                child_text = f"<step branch={branch_index} h={h}>"
        else:
            flat = z_star.detach().cpu().reshape(-1)[:16]
            h = int(torch.sum(flat * 1e4).item()) & 0xFFFFFFFF
            child_text = f"<step branch={branch_index} h={h}>"
    else:
        child_text = parent_text

    return TransitionResult(
        child_text=child_text,
        z_star_child=z_star.detach(),
        solver_stats=solver_stats,
        prune=False,
        budget=budget,
        faiss_retrieved=faiss_retrieved,
    )


def transition_batch(
    parent_text: str,
    nu: NuVector,
    budget: RuntimeBudgetState,
    backbone: BaseCTSBackbone,
    *,
    W: int = 3,
    K: int = 64,
    d: Optional[int] = None,
    top_k: int = 3,
    broyden_max_iter: int = 84,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    tau_flops_budget: float = 1e14,
    routing_mode: str = "sparse",
    max_decode_tokens: int = 1,
    faiss_context: Optional[LatentContextWindow] = None,
    faiss_retrieve: bool = True,
    faiss_depth: Optional[int] = None,
    fp32_buffer: bool = True,
    parent_inv_jacobian: Optional[torch.Tensor] = None,
    parent_z_star: Optional[torch.Tensor] = None,
    noise_sigma: float = 0.02,
    context_override: Optional[Any] = None,
    history_latents: Optional[Any] = None,
    defer_faiss_add: bool = False,
    decode_child: bool = False,
    noise_seed: Optional[int] = None,
    phase_profiler: Optional[Any] = None,
    profile_scope: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    fallback_to_parent_on_failure: bool = True,
    solver: str = "broyden",
    latch_warmup: Optional[int] = None,
) -> List[TransitionResult]:
    """Paper §4.1: parallel batch DEQ for W sibling branches.

    Algorithm 1 aligned: uses parent_z_star for initialization and fallback.
    """
    if isinstance(backbone, nn.Module):
        device = next(backbone.parameters()).device
    else:
        device = torch.device("cpu")
    d_actual = d or backbone.hidden_size

    if generator is not None and noise_seed is not None:
        raise ValueError("specify either noise_seed or generator, not both")

    if context_override is not None:
        context = context_override
        if history_latents is not None and hasattr(context, "with_history_latents"):
            context = context.with_history_latents(history_latents)
    else:
        context = backbone.encode_context(parent_text)
    if isinstance(context, torch.Tensor):
        if context.dim() == 1:
            context = context.unsqueeze(0)
        context = context.to(device=device, dtype=torch.float32)
    elif hasattr(context, "to"):
        context = context.to(device=device)

    faiss_retrieved = None
    if faiss_context is not None and faiss_retrieve:
        query_z = parent_z_star if parent_z_star is not None else init_z0(K, d_actual, device, torch.Generator(device=device))
        faiss_retrieved_raw = faiss_context.retrieve(query_z, k=3)
        if faiss_retrieved_raw is not None:
            context = prepend_soft_prefix(context, faiss_retrieved_raw)
            faiss_retrieved = faiss_retrieved_raw

    if hasattr(backbone, "routing_matrix"):
        w_g = backbone.routing_matrix().to(device=device, dtype=torch.float32)
    else:
        w_g = torch.randn(19, d_actual, device=device, dtype=torch.float32) * 0.02
    macs = _load_mac_lut()

    z0_list = []
    for bi in range(W):
        gen = generator or torch.Generator(device=device)
        if noise_seed is not None:
            gen.manual_seed(int(noise_seed) + bi * 31)
        else:
            gen.manual_seed(2026 + bi * 31 + (len(parent_text) % 997))
        if parent_z_star is not None:
            pz = parent_z_star.detach().to(device=device, dtype=torch.float32)
            if pz.shape[0] != K or pz.shape[-1] != d_actual:
                pz = init_z0(K, d_actual, device, gen)
            epsilon = torch.randn(K, d_actual, device=device, dtype=torch.float32, generator=gen) * noise_sigma * nu.nu_expl
            z0_list.append(pz + epsilon)
        else:
            z0 = init_z0(K, d_actual, device, gen)
            z0 = add_exploration_noise(z0, nu.nu_expl, gen)
            z0_list.append(z0)

    if solver == "picard":
        c_param = 0.92
        tol_val = 5e-3
        max_picard_steps = 1000
        z_batch = torch.stack(z0_list, dim=0).to(dtype=torch.float32)
        res_histories: List[List[float]] = [[] for _ in range(W)]
        converged_flags = [False] * W
        conv_steps = [0] * W

        for p_step in range(1, max_picard_steps + 1):
            if routing_mode == "dense":
                mw_batch = routing_weights(z_batch, w_g, nu.nu_temp)
            else:
                mw_batch = _routing_sparse(z_batch, w_g, nu.nu_temp, top_k)
            extra_p = {"top_k": top_k}
            phi_z_batch = backbone.deq_step(z_batch, context, mw_batch, extra_p)
            diff_batch = z_batch - phi_z_batch

            all_done = True
            for bi in range(W):
                if not converged_flags[bi]:
                    r_val = float(torch.linalg.vector_norm(diff_batch[bi]).item() / max(torch.linalg.vector_norm(z_batch[bi]).item(), torch.linalg.vector_norm(phi_z_batch[bi]).item(), 1.0))
                    res_histories[bi].append(r_val)
                    if len(res_histories[bi]) >= 100:
                        if all(r < tol_val for r in res_histories[bi][-100:]):
                            converged_flags[bi] = True
                            conv_steps[bi] = p_step
                    if not converged_flags[bi]:
                        all_done = False

            if all_done:
                z_batch = phi_z_batch
                break

            z_batch = (1.0 - c_param) * z_batch + c_param * phi_z_batch

        z_star_batch = z_batch
        phi_evals_per_iter = 1
        results: List[TransitionResult] = []

        for bi in range(W):
            b = budget.clone()
            b.terminal_depth += 1
            flops = 0.0
            z_star = z_star_batch[bi]
            is_conv = converged_flags[bi]
            final_iter = conv_steps[bi] if is_conv else max_picard_steps
            final_res = res_histories[bi][-1] if res_histories[bi] else float("nan")

            with torch.no_grad():
                alpha = routing_weights(z_star, w_g, nu.nu_temp)
                mw = alpha if routing_mode == "dense" else sparse_module_weights(alpha, top_k)
                for i in range(len(mw)):
                    flops += float(mw[i].item()) * macs[i] * nu.nu_act
            b.flops_spent_step = flops
            b.mac_accumulated += flops

            flops_estimate = flops * float(final_iter) * float(phi_evals_per_iter)
            solver_stats = {
                "iterations": final_iter,
                "residual_norm": final_res,
                "converged": is_conv,
                "flops_used": flops,
                "flops_inner_once": flops,
                "flops_broyden_estimate": flops_estimate,
                "phi_evals_per_broyden_iter": phi_evals_per_iter,
                "inv_jacobian": None,
                "all_residuals": res_histories[bi],
                "r_0": float(res_histories[bi][0]) if res_histories[bi] else float("nan"),
                "logical_mac_transition": float(flops_estimate),
                "mac_headline_eligible": True,
                "mac_lut_schema_version": 2,
                "mac_lut_sha256": _mac_lut_sha256(),
                "noise_seed": noise_seed,
                "branch_index": bi,
                "phi_eval_count": final_iter * phi_evals_per_iter,
                "ruler": "B_5e-3_sust100",
                "c": 0.92,
                "tol": 5e-3,
                "solver": "picard",
            }

            if not is_conv:
                fallback_z = parent_z_star.detach().clone().to(device) if (fallback_to_parent_on_failure and parent_z_star is not None) else z_star
                results.append(TransitionResult(
                    child_text=None, z_star_child=fallback_z, solver_stats=solver_stats,
                    prune=True, budget=b, faiss_retrieved=faiss_retrieved,
                ))
                continue

            if b.mac_accumulated > tau_flops_budget * nu.nu_act:
                solver_stats["act_halt"] = True

            if faiss_context is not None and not defer_faiss_add:
                faiss_context.add(z_star)

            child_text = None
            if decode_child:
                if hasattr(backbone, "decode_from_z_star"):
                    try:
                        import inspect as _insp
                        sig = _insp.signature(backbone.decode_from_z_star)
                        if "max_new_tokens" in sig.parameters:
                            child_text = backbone.decode_from_z_star(z_star, max_new_tokens=max_decode_tokens)
                        else:
                            child_text = backbone.decode_from_z_star(z_star)
                    except Exception:
                        child_text = f"<step branch={bi}>"
                else:
                    child_text = f"<step branch={bi}>"
            else:
                child_text = parent_text

            results.append(TransitionResult(
                child_text=child_text, z_star_child=z_star.detach(), solver_stats=solver_stats,
                prune=False, budget=b, faiss_retrieved=faiss_retrieved,
            ))

        return results

    else:
        tol = map_nu_tol_to_tol(nu.nu_tol, broyden_tol_min, broyden_tol_max)

        def phi(zz: torch.Tensor) -> torch.Tensor:
            if routing_mode == "dense":
                mw = routing_weights(zz, w_g, nu.nu_temp)
            else:
                mw = _routing_sparse(zz, w_g, nu.nu_temp, top_k)
            return backbone.deq_step(zz, context, mw, {"top_k": top_k})

        z0_batch = torch.stack(z0_list, dim=0)
        z_star_batch, infos = broyden_fixed_point_batch(
            phi, z0_batch, tol=tol, max_iter=broyden_max_iter,
            fp32_buffer=fp32_buffer, parent_inv_jacobian=parent_inv_jacobian,
        )

        phi_evals_per_broyden_iter = 2
        results: List[TransitionResult] = []
        for bi, (z_star, info) in enumerate(zip(z_star_batch, infos)):
            b = budget.clone()
            b.terminal_depth += 1
            flops = 0.0
            with torch.no_grad():
                alpha = routing_weights(z_star, w_g, nu.nu_temp)
                mw = alpha if routing_mode == "dense" else sparse_module_weights(alpha, top_k)
                for i in range(len(mw)):
                    flops += float(mw[i].item()) * macs[i] * nu.nu_act
            b.flops_spent_step = flops
            b.mac_accumulated += flops

            flops_broyden_estimate = (
                flops * float(info.iterations) * float(phi_evals_per_broyden_iter)
            )
            solver_stats = {
                "iterations": info.iterations,
                "residual_norm": info.residual_norm,
                "converged": info.converged,
                "flops_used": flops,
                "flops_inner_once": flops,
                "flops_broyden_estimate": flops_broyden_estimate,
                "phi_evals_per_broyden_iter": phi_evals_per_broyden_iter,
                "inv_jacobian": getattr(info, "jacobian_state", None),
                "all_residuals": getattr(info, "all_residuals", []),
                "logical_mac_transition": float(flops_broyden_estimate),
                "mac_headline_eligible": True,
                "mac_lut_schema_version": 2,
                "mac_lut_sha256": _mac_lut_sha256(),
                "noise_seed": noise_seed,
                "branch_index": bi,
                "phi_eval_count": int(info.iterations) * phi_evals_per_broyden_iter,
                "ruler": "A_broyden_tol",
                "c": 1.0,
                "tol": float(tol),
                "solver": "broyden",
            }

            if not info.converged:
                fallback_z = parent_z_star.detach().clone().to(device) if (fallback_to_parent_on_failure and parent_z_star is not None) else z_star
                results.append(TransitionResult(
                    child_text=None, z_star_child=fallback_z, solver_stats=solver_stats,
                    prune=True, budget=b, faiss_retrieved=faiss_retrieved,
                ))
                continue

            if b.mac_accumulated > tau_flops_budget * nu.nu_act:
                solver_stats["act_halt"] = True

            if faiss_context is not None and not defer_faiss_add:
                faiss_context.add(z_star)

            child_text = None
            if decode_child:
                if hasattr(backbone, "decode_from_z_star"):
                    try:
                        import inspect as _insp
                        sig = _insp.signature(backbone.decode_from_z_star)
                        if "max_new_tokens" in sig.parameters:
                            child_text = backbone.decode_from_z_star(z_star, max_new_tokens=max_decode_tokens)
                        else:
                            child_text = backbone.decode_from_z_star(z_star)
                    except Exception:
                        child_text = f"<step branch={bi}>"
                else:
                    child_text = f"<step branch={bi}>"
            else:
                child_text = parent_text

            results.append(TransitionResult(
                child_text=child_text, z_star_child=z_star.detach(), solver_stats=solver_stats,
                prune=False, budget=b, faiss_retrieved=faiss_retrieved,
            ))

        return results

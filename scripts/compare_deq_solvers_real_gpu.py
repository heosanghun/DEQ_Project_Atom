#!/usr/bin/env python3
"""Diagnostic-only comparison of fixed-point solvers on one real Gemma Phi.

This script never decodes an answer and never writes a checkpoint.  It freezes
one routing support/weight vector and applies every solver to the exact same
prompt-conditioned Eq. (3) operator so convergence and wall time are directly
comparable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.broyden_forward import (
    LimitedMemoryInverseJacobian,
    _anderson_broyden,
    broyden_fixed_point,
)
from cts.model.gemma_loader import load_gemma4_e4b
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights
from cts.train.openmath_text import prompt_text_from_openmath_row


def _relative(state: torch.Tensor, mapped: torch.Tensor) -> float:
    state_f = state.float()
    mapped_f = mapped.float()
    absolute = torch.linalg.vector_norm(mapped_f - state_f)
    scale = torch.maximum(
        torch.linalg.vector_norm(state_f),
        torch.linalg.vector_norm(mapped_f),
    ).clamp_min(1.0)
    return float((absolute / scale).item())


def _picard(phi, z0, *, beta: float, tol: float, max_iter: int) -> dict:
    state = z0.detach().float().clone()
    residuals: list[float] = []
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(int(max_iter)):
        mapped = phi(state).float()
        residual = _relative(state, mapped)
        residuals.append(residual)
        if residual < float(tol):
            break
        state = (1.0 - float(beta)) * state + float(beta) * mapped
    torch.cuda.synchronize()
    return {
        "converged": bool(residuals and residuals[-1] < float(tol)),
        "iterations": len(residuals),
        "final_residual": residuals[-1],
        "min_residual": min(residuals),
        "residuals": residuals,
        "wall_s": perf_counter() - start,
    }


def _broyden_with_scale(
    phi,
    z0: torch.Tensor,
    *,
    scale: float,
    tol: float,
    max_iter: int,
) -> dict:
    """Run the production large-state Broyden path with a diagnostic H0."""
    flat_size = int(z0.numel())
    empty = torch.empty(0, flat_size, device=z0.device, dtype=torch.float32)
    initial = LimitedMemoryInverseJacobian(
        scale=float(scale),
        u=empty,
        v=empty.clone(),
        validated=True,
    )
    torch.cuda.synchronize()
    start = perf_counter()
    _, info = broyden_fixed_point(
        phi,
        z0,
        tol=float(tol),
        max_iter=int(max_iter),
        parent_inv_jacobian=initial,
    )
    torch.cuda.synchronize()
    return {
        "initial_inverse_scale": float(scale),
        "converged": bool(info.converged),
        "iterations": int(info.iterations),
        "final_residual": float(info.residual_norm),
        "min_residual": min(info.all_residuals),
        "residuals": list(info.all_residuals),
        "wall_s": perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--broyden-max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("real solver comparison requires CUDA")
    os.environ["CTS_DEQ_MAP_MODE"] = "parallel"
    os.environ["CTS_EQ3_EXPLICIT_MASKS"] = "1"
    os.environ["CTS_EQ3_SUFFIX_CACHE"] = "1"
    os.environ["CTS_EQ3_SUFFIX_CACHE_TRAIN"] = "1"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    with Path(args.data).open("r", encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    prompt = prompt_text_from_openmath_row(row)

    model, tokenizer = load_gemma4_e4b(
        model_id=args.model_dir,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    backbone = GemmaCTSBackbone(model, tokenizer).eval()
    context = backbone.encode_context(prompt).to(device=torch.device(args.device))

    with torch.inference_mode(), torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, cache_enabled=False
    ):
        z0 = backbone.root_fwdpass(
            context, K=64, output_dtype=torch.float32
        )
        alpha = routing_weights(
            z0.float(), backbone.routing_matrix().float(), 1.0
        )
        module_weights = sparse_module_weights(alpha, 3).detach()
        module_ids = torch.sort(
            torch.topk(module_weights.reshape(-1), 3).indices
        ).values
        ids = [int(x) for x in module_ids.cpu().tolist()]
        extra = {
            "top_k": 3,
            "deq_map_mode": backbone.deq_map_mode,
            "routing_module_ids": ids,
            "routing_module_ids_tensor": module_ids,
            "eq3_static_inputs": backbone.prepare_deq_context(z0, context, ids),
        }

        def phi(state: torch.Tensor) -> torch.Tensor:
            return backbone.deq_step(state, context, module_weights, extra)

        # Warm the shared Phi kernels once without changing any solver state.
        phi(z0)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = perf_counter()
        _, broyden_info = broyden_fixed_point(
            phi,
            z0,
            tol=float(args.tol),
            max_iter=int(args.broyden_max_iter),
        )
        torch.cuda.synchronize()
        broyden_wall = perf_counter() - start

        torch.cuda.synchronize()
        start = perf_counter()
        _, anderson_info = _anderson_broyden(
            phi,
            z0,
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            fp32_buffer=True,
            memory_limit=16,
            is_root=True,
        )
        torch.cuda.synchronize()
        anderson_wall = perf_counter() - start

        report = {
            "diagnostic_only": True,
            "routing_module_ids": ids,
            "z0_norm": float(torch.linalg.vector_norm(z0.float()).item()),
            "tol": float(args.tol),
            "broyden": {
                "converged": bool(broyden_info.converged),
                "iterations": int(broyden_info.iterations),
                "final_residual": float(broyden_info.residual_norm),
                "min_residual": min(broyden_info.all_residuals),
                "residuals": list(broyden_info.all_residuals),
                "wall_s": broyden_wall,
            },
            "broyden_h0_0_5": _broyden_with_scale(
                phi,
                z0,
                scale=0.5,
                tol=args.tol,
                max_iter=args.broyden_max_iter,
            ),
            "broyden_h0_1_0": _broyden_with_scale(
                phi,
                z0,
                scale=1.0,
                tol=args.tol,
                max_iter=args.broyden_max_iter,
            ),
            "broyden_h0_2_0": _broyden_with_scale(
                phi,
                z0,
                scale=2.0,
                tol=args.tol,
                max_iter=args.broyden_max_iter,
            ),
            "anderson": {
                "converged": bool(anderson_info.converged),
                "iterations": int(anderson_info.iterations),
                "final_residual": float(anderson_info.residual_norm),
                "min_residual": min(anderson_info.all_residuals),
                "residuals": list(anderson_info.all_residuals),
                "wall_s": anderson_wall,
            },
            "picard_beta_1_0": _picard(
                phi, z0, beta=1.0, tol=args.tol, max_iter=args.max_iter
            ),
            "picard_beta_0_8": _picard(
                phi, z0, beta=0.8, tol=args.tol, max_iter=args.max_iter
            ),
            "picard_beta_0_5": _picard(
                phi, z0, beta=0.5, tol=args.tol, max_iter=args.max_iter
            ),
        }
    print("CTS_SOLVER_COMPARE_JSON=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sweep inverse-Broyden H0 on initial Stage-1 root-target Eq. (3) maps.

Diagnostic only: no decoding, training, checkpoint writes, or model mutation.
Each prompt freezes one routing support/weight vector, builds one deterministic
Phi, and compares all H0 scales against a high-accuracy Picard reference.

This intentionally loads the raw model snapshot, not a Stage-1/2 checkpoint,
and evaluates prompt/root maps rather than deployment child maps.  Results may
guide initial Stage-1 teacher construction only; they are not evidence for
changing the production child-solver H0.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from time import perf_counter

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.broyden_forward import (
    LimitedMemoryInverseJacobian,
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


def _picard_reference(phi, z0, *, tol: float, max_iter: int):
    state = z0.detach().float().clone()
    residuals: list[float] = []
    absolute_steps: list[float] = []
    previous_state = None
    previous_mapped = None
    stable_chords = []
    for iteration in range(1, int(max_iter) + 1):
        mapped = phi(state).float()
        residual = _relative(state, mapped)
        absolute_steps.append(
            float(torch.linalg.vector_norm(mapped - state).item())
        )
        residuals.append(residual)
        if previous_state is not None and previous_mapped is not None:
            input_chord = state - previous_state
            output_chord = mapped - previous_mapped
            input_norm = float(torch.linalg.vector_norm(input_chord).item())
            if input_norm > 1e-12:
                secant_gain = float(
                    torch.linalg.vector_norm(output_chord).item()
                ) / input_norm
                if iteration >= 5 and residual >= 1e-2:
                    stable_chords.append(
                        (
                            secant_gain,
                            iteration,
                            previous_state.detach().clone(),
                            state.detach().clone(),
                        )
                    )
        if not torch.isfinite(mapped).all() or not torch.isfinite(
            torch.tensor(residual, device=state.device)
        ):
            raise RuntimeError("Picard reference produced a non-finite value")
        if residual < float(tol):
            secant_gains = [
                absolute_steps[index] / max(absolute_steps[index - 1], 1e-12)
                for index in range(1, len(absolute_steps))
            ]
            stable_window = [
                (index + 1, gain)
                for index, gain in enumerate(secant_gains, start=1)
                if index + 1 >= 5 and residuals[index] >= 1e-2
            ]
            if not stable_window:
                stable_window = [
                    (index + 1, gain)
                    for index, gain in enumerate(secant_gains, start=1)
                ]
            worst_iteration, worst_gain = max(
                stable_window, key=lambda item: item[1]
            )
            diagnostics = {
                "stable_window_rule": "iteration>=5_and_relative_residual>=1e-2",
                "worst_secant_iteration": int(worst_iteration),
                "worst_secant_gain": float(worst_gain),
                "median_secant_gain": float(
                    statistics.median(gain for _, gain in stable_window)
                ),
            }
            if stable_chords:
                _, _, anchor, anchor_mapped = max(
                    stable_chords, key=lambda item: item[0]
                )
                endpoint_input = state - anchor
                endpoint_output = mapped - anchor_mapped
                endpoint_input_norm = torch.linalg.vector_norm(endpoint_input)
                endpoint_gain = (
                    torch.linalg.vector_norm(endpoint_output)
                    / endpoint_input_norm.clamp_min(1e-12)
                )
                endpoint_scale = torch.maximum(
                    torch.linalg.vector_norm(state),
                    torch.linalg.vector_norm(anchor),
                ).clamp_min(1.0)
                diagnostics.update(
                    {
                        "endpoint_secant_gain_from_worst_anchor": float(
                            endpoint_gain.item()
                        ),
                        "endpoint_input_chord_relative": float(
                            (endpoint_input_norm / endpoint_scale).item()
                        ),
                    }
                )
            return state, residuals, diagnostics
        previous_state = state
        previous_mapped = mapped
        state = mapped
    raise RuntimeError(
        f"Picard reference failed to reach tol={tol} in {max_iter} evaluations; "
        f"last={residuals[-1]}"
    )


def _empty_inverse(z0: torch.Tensor, scale: float):
    empty = torch.empty(
        0, int(z0.numel()), device=z0.device, dtype=torch.float32
    )
    return LimitedMemoryInverseJacobian(
        scale=float(scale),
        u=empty,
        v=empty.clone(),
        validated=True,
    )


def _state_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.vector_norm(candidate.float() - reference.float())
            / torch.linalg.vector_norm(reference.float()).clamp_min(1.0)
        ).item()
    )


def _parse_csv_numbers(raw: str, cast):
    values = [cast(piece.strip()) for piece in str(raw).split(",") if piece.strip()]
    if not values:
        raise ValueError("CSV argument must contain at least one value")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--row-indices", default="0,1,2,3,4")
    parser.add_argument("--h0-scales", default="0.1,0.5,1,1.5,2,2.5,3,4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--reference-tol", type=float, default=1e-4)
    parser.add_argument("--reference-max-iter", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("real H0 sweep requires CUDA")
    row_indices = _parse_csv_numbers(args.row_indices, int)
    if min(row_indices) < 0 or len(set(row_indices)) != len(row_indices):
        raise ValueError("row indices must be unique non-negative integers")
    scales = _parse_csv_numbers(args.h0_scales, float)
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("all H0 scales must be positive")

    wanted = set(row_indices)
    rows: dict[int, dict] = {}
    with Path(args.data).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in wanted:
                rows[index] = json.loads(line)
            if len(rows) == len(wanted):
                break
    missing = sorted(wanted - rows.keys())
    if missing:
        raise ValueError(f"data file does not contain requested rows: {missing}")

    os.environ["CTS_DEQ_MAP_MODE"] = "parallel"
    os.environ["CTS_EQ3_EXPLICIT_MASKS"] = "1"
    os.environ["CTS_EQ3_SUFFIX_CACHE"] = "1"
    os.environ["CTS_EQ3_SUFFIX_CACHE_TRAIN"] = "1"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model, tokenizer = load_gemma4_e4b(
        model_id=args.model_dir,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    backbone = GemmaCTSBackbone(model, tokenizer).eval()
    device = torch.device(args.device)
    records: list[dict] = []

    with torch.inference_mode(), torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, cache_enabled=False
    ):
        for row_position, row_index in enumerate(row_indices):
            prompt = prompt_text_from_openmath_row(rows[row_index])
            context = backbone.encode_context(prompt).to(device=device)
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
            ids = [int(value) for value in module_ids.cpu().tolist()]
            extra = {
                "top_k": 3,
                "deq_map_mode": backbone.deq_map_mode,
                "routing_module_ids": ids,
                "routing_module_ids_tensor": module_ids,
                "eq3_static_inputs": backbone.prepare_deq_context(
                    z0, context, ids
                ),
            }

            def phi(state: torch.Tensor) -> torch.Tensor:
                return backbone.deq_step(
                    state, context, module_weights, extra
                )

            phi(z0)
            reference, reference_history, reference_diagnostics = _picard_reference(
                phi,
                z0,
                tol=float(args.reference_tol),
                max_iter=int(args.reference_max_iter),
            )
            scale_results: dict[str, dict] = {}
            # Rotate the timed order across prompts so no one scale always
            # benefits from allocator/cache state established by predecessors.
            offset = row_position % len(scales)
            execution_order = scales[offset:] + scales[:offset]
            for scale in execution_order:
                # Untimed per-scale solver warm-up.  H0 changes only scalar
                # arithmetic, but this also equalises factor-buffer allocation.
                broyden_fixed_point(
                    phi,
                    z0,
                    tol=float(args.tol),
                    max_iter=2,
                    parent_inv_jacobian=_empty_inverse(z0, scale),
                )
                torch.cuda.synchronize(device)
                start = perf_counter()
                candidate, info = broyden_fixed_point(
                    phi,
                    z0,
                    tol=float(args.tol),
                    max_iter=int(args.max_iter),
                    parent_inv_jacobian=_empty_inverse(z0, scale),
                )
                torch.cuda.synchronize(device)
                scale_results[str(scale)] = {
                    "converged": bool(info.converged),
                    "iterations": int(info.iterations),
                    "phi_eval_count": int(info.phi_eval_count),
                    "returned_best_residual": float(info.residual_norm),
                    "min_residual": min(info.all_residuals),
                    "trajectory_last_residual": float(info.all_residuals[-1]),
                    "approx_reference_state_error": _state_error(
                        candidate, reference
                    ),
                    "wall_s": perf_counter() - start,
                }
            record = {
                "row_index": row_index,
                "routing_module_ids": ids,
                "reference_iterations": len(reference_history),
                "reference_residual": reference_history[-1],
                "reference_trajectory": reference_diagnostics,
                "execution_order": execution_order,
                "scales": scale_results,
            }
            records.append(record)
            print("CTS_H0_ROW_JSON=" + json.dumps(record, sort_keys=True), flush=True)

    aggregate: dict[str, dict] = {}
    for scale in scales:
        key = str(scale)
        entries = [record["scales"][key] for record in records]
        aggregate[key] = {
            "converged": sum(int(entry["converged"]) for entry in entries),
            "total": len(entries),
            "median_iterations": statistics.median(
                entry["iterations"] for entry in entries
            ),
            "median_residual": statistics.median(
                entry["returned_best_residual"] for entry in entries
            ),
            "max_residual": max(
                entry["returned_best_residual"] for entry in entries
            ),
            "median_approx_reference_state_error": statistics.median(
                entry["approx_reference_state_error"] for entry in entries
            ),
            "max_approx_reference_state_error": max(
                entry["approx_reference_state_error"] for entry in entries
            ),
            "median_wall_s": statistics.median(
                entry["wall_s"] for entry in entries
            ),
        }
    report = {
        "diagnostic_only": True,
        "scope": "raw_snapshot_initial_stage1_root_target_only",
        "checkpoint_loaded": False,
        "data": str(args.data),
        "row_indices": row_indices,
        "tol": float(args.tol),
        "max_iter": int(args.max_iter),
        "reference_tol": float(args.reference_tol),
        "aggregate": aggregate,
    }
    print("CTS_H0_SWEEP_JSON=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

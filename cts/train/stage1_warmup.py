"""Stage 1: DEQ warm-up — IFT residual loss + LM preservation (paper §6).

Paper §6: loss = ||f(z*) - z*||^2 + 0.1 * L_CE

The L_CE term limits perplexity increase to 0.4 nats (vs. 1.8 nats without it;
Appendix P). Without it, DEQ-Only AIME drops from 35.2% to 31.4%.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.broyden_forward import BroydenInfo, broyden_fixed_point
from cts.deq.gemma_latent_forward import raw_ple_for_mixed_public_forward
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights
from cts.types import NuVector


@contextmanager
def _deterministic_deq_dropout(backbone: nn.Module):
    """Disable stochastic dropout only while solving/training the DEQ map.

    Broyden assumes repeated evaluations of one deterministic function.  LoRA
    dropout otherwise changes Phi at every iteration and can manufacture
    non-convergence.  The original training flags are restored before the LM
    preservation/W_proj CE terms, where dropout remains a valid regularizer.
    """
    states = [
        (module, bool(module.training))
        for module in backbone.modules()
        if isinstance(module, nn.Dropout)
    ]
    for module, _ in states:
        module.train(False)
    try:
        yield
    finally:
        for module, training in states:
            module.train(training)


def _picard_fixed_point(
    phi,
    z0: torch.Tensor,
    *,
    tol: float,
    max_iter: int,
    diagnostics_out: Optional[Dict[str, Any]] = None,
) -> tuple[torch.Tensor, BroydenInfo]:
    """Robust diagnostic fixed-point target using plain Picard iteration.

    When requested, ``diagnostics_out`` receives the worst stable secant
    chord observed along the *global* Picard trajectory.  This is distinct
    from an endpoint Jacobian/spectral-radius estimate: it identifies a real
    path segment where the current operator most resists contraction.
    """
    state = z0.detach().float().clone()
    best_state = state.clone()
    best_rel = float("inf")
    best_abs = float("inf")
    residuals: list[float] = []
    previous_state: Optional[torch.Tensor] = None
    previous_mapped: Optional[torch.Tensor] = None
    # Keep only the two winning chords.  Retaining the complete trajectory
    # would pin O(max_iter * K * H) GPU memory for a diagnostic regularizer.
    # Tuple: (gain, anchor_iteration, anchor_residual, anchor, next_state).
    fallback_secant: Optional[
        tuple[float, int, float, torch.Tensor, torch.Tensor]
    ] = None
    stable_secant: Optional[
        tuple[float, int, float, torch.Tensor, torch.Tensor]
    ] = None

    def _record_secant(
        *,
        gain: float,
        anchor_iteration: int,
        anchor_residual: float,
        anchor: torch.Tensor,
        next_state: torch.Tensor,
        stable: bool,
    ) -> None:
        nonlocal fallback_secant, stable_secant
        if diagnostics_out is None:
            return
        current = stable_secant if stable else fallback_secant
        # For an exact tie, prefer the later chord: it is closer to the
        # converged endpoint and gives the more conservative selector.
        replace = current is None or (float(gain), int(anchor_iteration)) >= (
            current[0],
            current[1],
        )
        if replace:
            candidate = (
                float(gain),
                int(anchor_iteration),
                float(anchor_residual),
                anchor.detach().clone(),
                next_state.detach().clone(),
            )
            if stable:
                stable_secant = candidate
            else:
                fallback_secant = candidate

    def _publish_secant_diagnostics() -> None:
        if diagnostics_out is None:
            return
        candidate = stable_secant or fallback_secant
        diagnostics_out.clear()
        diagnostics_out["stable_window_rule"] = (
            "iteration>=5_and_relative_residual>=max(10*tol,1e-2)"
        )
        diagnostics_out["selector"] = "worst_consecutive_picard_secant_v1"
        if candidate is None:
            diagnostics_out["worst_secant_gain"] = None
            return
        gain, anchor_iteration, anchor_residual, probe_a, probe_b = candidate
        diagnostics_out.update(
            {
                "worst_secant_gain": float(gain),
                "worst_secant_iteration": int(anchor_iteration),
                "anchor_teacher_residual": float(anchor_residual),
                "probe_a": probe_a.detach().clone(),
                "probe_b": probe_b.detach().clone(),
            }
        )

    for iteration in range(1, int(max_iter) + 1):
        mapped = phi(state).float()
        delta = mapped - state
        absolute = torch.linalg.vector_norm(delta)
        state_norm = torch.linalg.vector_norm(state)
        mapped_norm = torch.linalg.vector_norm(mapped)
        scale = torch.maximum(state_norm, mapped_norm).clamp_min(1.0)
        relative = float((absolute / scale).item())
        absolute_value = float(absolute.item())
        residuals.append(relative)
        if (
            diagnostics_out is not None
            and previous_state is not None
            and previous_mapped is not None
        ):
            chord = state - previous_state
            mapped_chord = mapped - previous_mapped
            chord_norm = float(torch.linalg.vector_norm(chord).item())
            mapped_chord_norm = float(
                torch.linalg.vector_norm(mapped_chord).item()
            )
            if (
                math.isfinite(chord_norm)
                and math.isfinite(mapped_chord_norm)
                and chord_norm > 1e-12
            ):
                secant_gain = mapped_chord_norm / chord_norm
                anchor_iteration = int(iteration) - 1
                anchor_residual = float(residuals[-2])
                _record_secant(
                    gain=float(secant_gain),
                    anchor_iteration=anchor_iteration,
                    anchor_residual=anchor_residual,
                    anchor=previous_state,
                    next_state=state,
                    stable=(
                        anchor_iteration >= 5
                        and anchor_residual
                        >= max(10.0 * float(tol), 1e-2)
                    ),
                )
        if not all(
            math.isfinite(value)
            for value in (
                relative,
                absolute_value,
                float(state_norm.item()),
                float(mapped_norm.item()),
            )
        ):
            _publish_secant_diagnostics()
            return best_state.to(dtype=z0.dtype), BroydenInfo(
                iterations=iteration,
                residual_norm=best_rel,
                converged=False,
                all_residuals=residuals,
                absolute_residual_norm=best_abs,
                phi_eval_count=iteration,
                failure_reason="non_finite_residual",
            )
        if relative < best_rel:
            best_state = state.clone()
            best_rel = relative
            best_abs = absolute_value
        if relative < float(tol):
            _publish_secant_diagnostics()
            return state.to(dtype=z0.dtype), BroydenInfo(
                iterations=iteration,
                residual_norm=relative,
                converged=True,
                all_residuals=residuals,
                absolute_residual_norm=absolute_value,
                phi_eval_count=iteration,
            )
        if diagnostics_out is not None:
            previous_state = state
            previous_mapped = mapped
        state = mapped
    _publish_secant_diagnostics()
    return best_state.to(dtype=z0.dtype), BroydenInfo(
        iterations=int(max_iter),
        residual_norm=best_rel,
        converged=False,
        all_residuals=residuals,
        absolute_residual_norm=best_abs,
        phi_eval_count=int(max_iter),
        failure_reason="max_iter",
    )


def _sampled_secant_contraction_loss(
    phi,
    anchor: torch.Tensor,
    z_target: torch.Tensor,
    mapped_target: torch.Tensor,
    *,
    target_gain: float,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Endpoint-secant hinge using one extra differentiable Phi evaluation.

    ``mapped_target`` is the already-computed ``Phi(z_target)`` from the fixed
    point loss.  Only ``Phi(anchor)`` is added, while gradients flow through
    both current-parameter endpoints.
    """
    if not math.isfinite(float(target_gain)) or not (
        0.0 < float(target_gain) < 1.0
    ):
        raise ValueError("target_gain must be in (0, 1)")
    if anchor.shape != z_target.shape or mapped_target.shape != z_target.shape:
        raise RuntimeError("sampled contraction endpoint shapes do not match")
    if not bool(torch.isfinite(z_target.detach()).all().item()):
        raise RuntimeError("sampled contraction target is non-finite")
    if not bool(torch.isfinite(anchor.detach()).all().item()):
        raise RuntimeError("sampled contraction anchor is non-finite")
    if not bool(torch.isfinite(mapped_target.detach()).all().item()):
        raise RuntimeError("sampled contraction mapped target is non-finite")
    chord_norm = torch.linalg.vector_norm(
        z_target.detach().float() - anchor.detach().float()
    )
    if not bool(torch.isfinite(chord_norm).item()) or float(chord_norm.item()) <= 1e-12:
        raise RuntimeError("sampled contraction chord is non-finite or degenerate")
    state_scale = torch.maximum(
        torch.linalg.vector_norm(z_target.detach().float()),
        torch.linalg.vector_norm(anchor.detach().float()),
    ).clamp_min(1.0)
    relative_chord = chord_norm / state_scale
    if float(relative_chord.item()) < 0.02:
        raise RuntimeError(
            "sampled contraction chord is below the BF16-safe relative size"
        )
    mapped_anchor = phi(anchor.detach().float()).float()
    if not bool(torch.isfinite(mapped_anchor.detach()).all().item()):
        raise RuntimeError("sampled contraction mapped anchor is non-finite")
    output_chord_norm = torch.linalg.vector_norm(
        mapped_target.float() - mapped_anchor
    )
    gain = output_chord_norm / chord_norm
    if not bool(torch.isfinite(gain.detach()).item()):
        raise RuntimeError("sampled contraction gain is non-finite")
    hinge = torch.relu(gain - float(target_gain))
    # Match the paper FP term's squared-L2 *sum* scale rather than silently
    # averaging over K*H.  The caller's diagnostic lambda controls its weight.
    loss = hinge.square() * int(z_target.numel())
    return loss, gain, {
        "hinge": float(hinge.detach().item()),
        "input_chord_norm": float(chord_norm.detach().item()),
        "input_chord_relative": float(relative_chord.detach().item()),
        "output_chord_norm": float(output_chord_norm.detach().item()),
        "scale_n": int(z_target.numel()),
    }


def _trajectory_secant_contraction_loss(
    phi,
    probe_a: torch.Tensor,
    probe_b: torch.Tensor,
    *,
    target_gain: float,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Penalize the Picard-selected consecutive trajectory chord directly.

    This costs two differentiable Phi calls, but targets the observed global
    expansion instead of the failed anchor-to-endpoint proxy.
    """
    if not math.isfinite(float(target_gain)) or not (
        0.0 < float(target_gain) < 1.0
    ):
        raise ValueError("target_gain must be in (0, 1)")
    if probe_a.shape != probe_b.shape:
        raise RuntimeError("trajectory contraction probe shapes do not match")
    if not bool(torch.isfinite(probe_a.detach()).all().item()) or not bool(
        torch.isfinite(probe_b.detach()).all().item()
    ):
        raise RuntimeError("trajectory contraction probe is non-finite")
    point_a = probe_a.detach().float()
    point_b = probe_b.detach().float()
    chord_norm = torch.linalg.vector_norm(point_b - point_a)
    if not bool(torch.isfinite(chord_norm).item()) or float(chord_norm.item()) <= 1e-12:
        raise RuntimeError("trajectory contraction chord is non-finite or degenerate")
    state_scale = torch.maximum(
        torch.linalg.vector_norm(point_a),
        torch.linalg.vector_norm(point_b),
    ).clamp_min(1.0)
    relative_chord = chord_norm / state_scale
    if float(relative_chord.item()) < 0.02:
        raise RuntimeError(
            "trajectory contraction chord is below the BF16-safe relative size"
        )
    mapped_a = phi(point_a).float()
    mapped_b = phi(point_b).float()
    if not bool(torch.isfinite(mapped_a.detach()).all().item()) or not bool(
        torch.isfinite(mapped_b.detach()).all().item()
    ):
        raise RuntimeError("trajectory contraction mapped probe is non-finite")
    output_chord_norm = torch.linalg.vector_norm(mapped_b - mapped_a)
    gain = output_chord_norm / chord_norm
    if not bool(torch.isfinite(gain.detach()).item()):
        raise RuntimeError("trajectory contraction gain is non-finite")
    hinge = torch.relu(gain - float(target_gain))
    loss = hinge.square() * int(probe_a.numel())
    return loss, gain, {
        "hinge": float(hinge.detach().item()),
        "input_chord_norm": float(chord_norm.detach().item()),
        "input_chord_relative": float(relative_chord.detach().item()),
        "output_chord_norm": float(output_chord_norm.detach().item()),
        "scale_n": int(probe_a.numel()),
    }


def fixed_point_surrogate_loss(
    backbone: nn.Module,
    parent_text: str,
    z: torch.Tensor,
    nu: NuVector,
    *,
    w_g: torch.Tensor,
    top_k: int = 3,
    extra: Dict[str, Any] | None = None,
    lambda_lm: float = 0.1,
    tokenizer: Any = None,
    target_text: Optional[str] = None,
    lm_target_text: Optional[str] = None,
    decode_parent_text: Optional[str] = None,
    lambda_wproj: float = 0.0,
    solve_equilibrium: bool = False,
    equilibrium_tol: float = 1e-3,
    equilibrium_max_iter: int = 30,
    equilibrium_solver: str = "broyden",
    diagnostic_contraction_lambda: float = 0.0,
    diagnostic_contraction_target_gain: float = 0.90,
    diagnostic_contraction_mode: str = "endpoint",
    max_target_tokens: int = 256,
    max_lm_target_tokens: int = 256,
    lambda_c: float = 0.0,
    c_target: float = 0.92,
    metrics_out: Optional[Dict[str, Any]] = None,
    diagnostic_z_target_cache: Optional[Dict[str, Any]] = None,
    diagnostic_routing_weights_cache: Optional[Dict[str, Any]] = None,
    diagnostic_local_gain_directions: int = 0,
    diagnostic_local_gain_epsilon: float = 1e-2,
    diagnostic_local_gain_seed: int = 2026,
    diagnostic_picard_steps: int = 0,
) -> torch.Tensor:
    """Paper §6: ||Phi(z) - z||^2 + lambda_lm * L_CE.

    The IFT residual drives z toward a fixed point.
    The CE term preserves the language model's generative capacity.
    """
    device = z.device
    context = backbone.encode_context(parent_text)
    if isinstance(context, torch.Tensor):
        if context.dim() == 1:
            context = context.unsqueeze(0)
        context = context.to(device=device, dtype=torch.float32)
    elif hasattr(context, "to"):
        context = context.to(device=device)
    ex = dict(extra or {})
    zf = z.float()
    contraction_probe_a: Optional[torch.Tensor] = None
    contraction_probe_b: Optional[torch.Tensor] = None
    teacher_secant_gain: Optional[float] = None
    teacher_anchor_iteration: Optional[int] = None
    teacher_anchor_residual: Optional[float] = None
    contraction_lambda = float(diagnostic_contraction_lambda)
    contraction_mode = str(diagnostic_contraction_mode)
    if contraction_mode not in {"endpoint", "trajectory"}:
        raise ValueError(
            "diagnostic_contraction_mode must be endpoint or trajectory"
        )
    if not math.isfinite(contraction_lambda) or contraction_lambda < 0.0:
        raise ValueError(
            "diagnostic_contraction_lambda must be finite and >= 0"
        )
    if contraction_lambda > 0.0:
        if not solve_equilibrium:
            raise ValueError(
                "diagnostic contraction loss requires solve_equilibrium=True"
            )
        if str(equilibrium_solver) != "picard":
            raise ValueError(
                "diagnostic contraction loss requires a Picard teacher"
            )
        if diagnostic_z_target_cache is None:
            raise ValueError(
                "diagnostic contraction loss requires a fixed target cache"
            )
        if diagnostic_routing_weights_cache is None:
            raise ValueError(
                "diagnostic contraction loss requires frozen routing weights"
            )
        if not math.isfinite(float(diagnostic_contraction_target_gain)) or not (
            0.0 < float(diagnostic_contraction_target_gain) < 1.0
        ):
            raise ValueError(
                "diagnostic_contraction_target_gain must be in (0, 1)"
            )

    def _routing_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = routing_weights(state.float(), w_g.to(state.device), nu.nu_temp)
        support_override = ex.get("routing_support_override")
        if support_override is None:
            return alpha, sparse_module_weights(alpha, top_k)
        support = torch.as_tensor(
            support_override,
            device=alpha.device,
            dtype=torch.long,
        ).reshape(-1)
        if support.numel() != min(int(top_k), int(alpha.shape[-1])):
            raise ValueError("routing_support_override size must equal top_k")
        if int(support.min().item()) < 0 or int(support.max().item()) >= int(
            alpha.shape[-1]
        ):
            raise ValueError("routing_support_override contains an invalid module")
        mask = torch.zeros_like(alpha)
        mask.scatter_(-1, support, 1.0)
        selected = alpha * mask
        selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return alpha, selected

    with _deterministic_deq_dropout(backbone):
        root_fwdpass = getattr(backbone, "root_fwdpass", None)
        if callable(root_fwdpass):
            z_initial = root_fwdpass(
                context,
                K=int(zf.shape[-2]),
                output_dtype=torch.float32,
            ).to(device=device)
            root_initialization = "prompt_full_stack"
        else:
            z_initial = zf
            root_initialization = "random_compatibility"
        # Eq. (3) fixes Top-k routing for the solve from its parent/initial
        # state. Re-selecting modules from every Broyden iterate makes Phi
        # discontinuous and trains a different operator than deployment. Keep
        # the graph to W_g for the phantom-gradient evaluation below.
        routing_alpha, fixed_module_weights = _routing_state(z_initial)
        if (
            diagnostic_routing_weights_cache is not None
            and "module_weights" in diagnostic_routing_weights_cache
        ):
            fixed_module_weights = diagnostic_routing_weights_cache[
                "module_weights"
            ].to(device=device, dtype=routing_alpha.dtype)
            routing_alpha = diagnostic_routing_weights_cache["routing_alpha"].to(
                device=device, dtype=routing_alpha.dtype
            )
        elif diagnostic_routing_weights_cache is not None:
            diagnostic_routing_weights_cache.update(
                {
                    "module_weights": fixed_module_weights.detach().clone(),
                    "routing_alpha": routing_alpha.detach().clone(),
                }
            )
            fixed_module_weights = diagnostic_routing_weights_cache[
                "module_weights"
            ]
            routing_alpha = diagnostic_routing_weights_cache["routing_alpha"]
        support_override = ex.get("routing_support_override")
        if support_override is None:
            selected_module_ids = torch.sort(
                torch.topk(
                    fixed_module_weights.reshape(-1),
                    min(int(top_k), int(fixed_module_weights.numel())),
                ).indices.to(dtype=torch.long)
            ).values
        else:
            selected_module_ids = torch.sort(
                torch.as_tensor(
                    support_override,
                    device=device,
                    dtype=torch.long,
                ).reshape(-1)
            ).values
        routing_top = torch.topk(
            routing_alpha.reshape(-1),
            min(int(top_k) + 1, int(routing_alpha.numel())),
        ).values
        routing_margin = (
            float((routing_top[top_k - 1] - routing_top[top_k]).detach().item())
            if int(routing_top.numel()) > int(top_k)
            else float("nan")
        )
        routing_entropy = float(
            (
                -routing_alpha.float()
                * routing_alpha.float().clamp_min(1e-12).log()
            ).sum().detach().item()
        )
        ex["routing_module_ids"] = [
            int(x)
            for x in selected_module_ids.detach().cpu().tolist()
        ]
        ex["routing_module_ids_tensor"] = selected_module_ids
        prepare_hook = getattr(backbone, "prepare_deq_context", None)
        if callable(prepare_hook):
            ex["eq3_static_inputs"] = prepare_hook(
                z_initial, context, ex["routing_module_ids"]
            )

        # Find the current equilibrium without retaining its 30-step graph,
        # then attach a one-step phantom gradient at z*.
        solve_info = None
        cached_solve_diag: Optional[Dict[str, Any]] = None
        z_target_source = "root_initialization"
        if solve_equilibrium:
            # Do not populate an enclosing autocast context's weight cache
            # from the no-grad Broyden evaluations.  A cached detached cast of
            # an FP32 LoRA weight would be reused by the one-step phantom
            # gradient below and disconnect the adapter from Stage-1.
            solver_autocast = (
                torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    cache_enabled=False,
                )
                if device.type == "cuda"
                else nullcontext()
            )
            if diagnostic_z_target_cache is not None and "z_target" in diagnostic_z_target_cache:
                if diagnostic_z_target_cache.get("parent_text") != parent_text:
                    raise RuntimeError(
                        "diagnostic fixed z-target cannot be reused across prompts"
                    )
                if diagnostic_z_target_cache.get("routing_module_ids") != ex[
                    "routing_module_ids"
                ]:
                    raise RuntimeError(
                        "diagnostic fixed z-target requires fixed routing support"
                    )
                z_target = diagnostic_z_target_cache["z_target"].to(
                    device=device, dtype=torch.float32
                )
                if z_target.shape != z_initial.shape:
                    raise RuntimeError("diagnostic fixed z-target shape changed")
                cached_solve_diag = dict(
                    diagnostic_z_target_cache.get("solve_diag") or {}
                )
                contraction_cache_keys = {
                    "contraction_probe_a",
                    "contraction_probe_b",
                    "teacher_secant_gain",
                    "teacher_anchor_iteration",
                    "teacher_anchor_residual",
                }
                present_contraction_keys = contraction_cache_keys.intersection(
                    diagnostic_z_target_cache
                )
                if present_contraction_keys and (
                    present_contraction_keys != contraction_cache_keys
                ):
                    missing = sorted(
                        contraction_cache_keys - present_contraction_keys
                    )
                    raise RuntimeError(
                        "diagnostic contraction cache is incomplete; missing "
                        + ", ".join(missing)
                    )
                if present_contraction_keys:
                    contraction_probe_a = diagnostic_z_target_cache[
                        "contraction_probe_a"
                    ].to(device=device, dtype=torch.float32)
                    contraction_probe_b = diagnostic_z_target_cache[
                        "contraction_probe_b"
                    ].to(device=device, dtype=torch.float32)
                    teacher_secant_gain = float(
                        diagnostic_z_target_cache["teacher_secant_gain"]
                    )
                    teacher_anchor_iteration = int(
                        diagnostic_z_target_cache["teacher_anchor_iteration"]
                    )
                    teacher_anchor_residual = float(
                        diagnostic_z_target_cache["teacher_anchor_residual"]
                    )
                z_target_source = "diagnostic_first_solve_reuse"
            else:
                # Preserve the lambda=0 production/control path exactly: no
                # trajectory chords, clones, or extra norms are collected.
                picard_diagnostics: Optional[Dict[str, Any]] = (
                    {} if contraction_lambda > 0.0 else None
                )
                with torch.no_grad(), solver_autocast:
                    def _phi_no_grad(state: torch.Tensor) -> torch.Tensor:
                        return backbone.deq_step(
                            state,
                            context,
                            fixed_module_weights,
                            ex,
                        )

                    if str(equilibrium_solver) == "broyden":
                        z_target, solve_info = broyden_fixed_point(
                            _phi_no_grad,
                            z_initial,
                            tol=float(equilibrium_tol),
                            max_iter=int(equilibrium_max_iter),
                        )
                    elif str(equilibrium_solver) == "picard":
                        z_target, solve_info = _picard_fixed_point(
                            _phi_no_grad,
                            z_initial,
                            tol=float(equilibrium_tol),
                            max_iter=int(equilibrium_max_iter),
                            diagnostics_out=picard_diagnostics,
                        )
                    else:
                        raise ValueError(
                            "equilibrium_solver must be 'broyden' or 'picard'"
                        )
                z_target = z_target.detach().float()
                z_target_source = f"{equilibrium_solver}_current_parameters"
                if (
                    picard_diagnostics is not None
                    and picard_diagnostics.get("probe_a") is not None
                ):
                    contraction_probe_a = picard_diagnostics["probe_a"].to(
                        device=device, dtype=torch.float32
                    )
                    contraction_probe_b = picard_diagnostics["probe_b"].to(
                        device=device, dtype=torch.float32
                    )
                    teacher_secant_gain = float(
                        picard_diagnostics["worst_secant_gain"]
                    )
                    teacher_anchor_iteration = int(
                        picard_diagnostics["worst_secant_iteration"]
                    )
                    teacher_anchor_residual = float(
                        picard_diagnostics["anchor_teacher_residual"]
                    )
                if diagnostic_z_target_cache is not None:
                    diagnostic_z_target_cache.update(
                        {
                            "parent_text": parent_text,
                            "routing_module_ids": list(ex["routing_module_ids"]),
                            "z_target": z_target.detach().clone(),
                            "solve_diag": {
                                "converged": bool(
                                    getattr(solve_info, "converged", False)
                                ),
                                "iterations": int(
                                    getattr(solve_info, "iterations", 0)
                                ),
                                "residual_norm": float(
                                    getattr(solve_info, "residual_norm", float("nan"))
                                ),
                                "absolute_residual_norm": float(
                                    getattr(
                                        solve_info,
                                        "absolute_residual_norm",
                                        float("nan"),
                                    )
                                ),
                                "all_residuals": list(
                                    getattr(solve_info, "all_residuals", []) or []
                                ),
                                "failure_reason": getattr(
                                    solve_info, "failure_reason", None
                                ),
                            },
                        }
                    )
                    if contraction_probe_a is not None:
                        diagnostic_z_target_cache.update(
                            {
                                "contraction_probe_a": (
                                    contraction_probe_a.detach().clone()
                                ),
                                "contraction_probe_b": (
                                    contraction_probe_b.detach().clone()
                                ),
                                "teacher_secant_gain": float(
                                    teacher_secant_gain
                                ),
                                "teacher_anchor_iteration": int(
                                    teacher_anchor_iteration
                                ),
                                "teacher_anchor_residual": float(
                                    teacher_anchor_residual
                                ),
                            }
                        )
        else:
            if diagnostic_z_target_cache is not None:
                raise ValueError(
                    "diagnostic fixed z-target requires solve_equilibrium=True"
                )
            z_target = z_initial

        phi_z = backbone.deq_step(
            z_target,
            context,
            fixed_module_weights,
            ex,
        )
        # Paper §6 / App. I specifies the squared L2 norm
        # ``||Phi(z*) - z*||_2^2``.  ``F.mse_loss`` silently divided this
        # signal by K*H (64*2560 = 163,840 for Gemma), leaving the DEQ
        # residual effectively unable to compete with the token CE terms.
        residual = phi_z.float() - z_target
        loss_fp = residual.square().sum()

        # Contraction constraint penalty (EXE-260820-02 Stage-1b)
        loss_contraction_penalty = torch.zeros((), device=device)
        decay_ratio: Optional[torch.Tensor] = None
        if lambda_c > 0.0:
            phi_phi_z = backbone.deq_step(
                phi_z,
                context,
                fixed_module_weights,
                ex,
            )
            r1 = phi_z.float() - z_target.float()
            r2 = phi_phi_z.float() - phi_z.float()
            norm_r1 = torch.linalg.vector_norm(r1.flatten(1), dim=1)
            norm_r2 = torch.linalg.vector_norm(r2.flatten(1), dim=1)
            decay_ratio = norm_r2 / (norm_r1.detach() + 1e-8)
            loss_contraction_penalty = torch.relu(decay_ratio - float(c_target)).square().mean()

        loss_contraction = torch.zeros((), device=device)
        sampled_secant_gain: Optional[torch.Tensor] = None
        contraction_details: Dict[str, float] = {}
        if contraction_lambda > 0.0:
            teacher_converged = (
                bool(getattr(solve_info, "converged", False))
                if solve_info is not None
                else bool((cached_solve_diag or {}).get("converged", False))
            )
            if not teacher_converged:
                raise RuntimeError(
                    "diagnostic contraction loss refuses a non-converged "
                    "Picard teacher"
                )
            if contraction_probe_a is None or contraction_probe_b is None:
                raise RuntimeError(
                    "secant contraction loss requires a converged Picard "
                    "teacher chord"
                )
            if contraction_probe_a.shape != contraction_probe_b.shape:
                raise RuntimeError("diagnostic contraction probe shapes changed")
            if not bool(torch.isfinite(contraction_probe_b).all().item()):
                raise RuntimeError("diagnostic contraction probe is non-finite")
            if (
                teacher_secant_gain is None
                or not math.isfinite(float(teacher_secant_gain))
                or teacher_anchor_iteration is None
                or int(teacher_anchor_iteration) < 1
                or teacher_anchor_residual is None
                or not math.isfinite(float(teacher_anchor_residual))
            ):
                raise RuntimeError(
                    "diagnostic contraction teacher metadata is invalid"
                )

            def _phi_contraction(state: torch.Tensor) -> torch.Tensor:
                return backbone.deq_step(
                    state,
                    context,
                    fixed_module_weights,
                    ex,
                )

            if contraction_mode == "endpoint":
                loss_contraction, sampled_secant_gain, contraction_details = (
                    _sampled_secant_contraction_loss(
                        _phi_contraction,
                        contraction_probe_a,
                        z_target,
                        phi_z,
                        target_gain=float(diagnostic_contraction_target_gain),
                    )
                )
            else:
                loss_contraction, sampled_secant_gain, contraction_details = (
                    _trajectory_secant_contraction_loss(
                        _phi_contraction,
                        contraction_probe_a,
                        contraction_probe_b,
                        target_gain=float(diagnostic_contraction_target_gain),
                    )
                )

        # Diagnostic only: a finite-difference local gain proxy for Phi around
        # the training target.  This is not a spectral radius estimate.  A
        # value above one is direct evidence of local expansion in the sampled
        # direction; values below one do not prove global contraction.
        local_gains: list[float] = []
        if int(diagnostic_local_gain_directions) > 0:
            if float(diagnostic_local_gain_epsilon) <= 0.0:
                raise ValueError("diagnostic_local_gain_epsilon must be > 0")
            gain_generator = torch.Generator(device=device).manual_seed(
                int(diagnostic_local_gain_seed)
            )
            z_norm = torch.linalg.vector_norm(z_target.detach().float()).clamp_min(1.0)
            delta_norm = float(diagnostic_local_gain_epsilon) * z_norm
            gain_autocast = (
                torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, cache_enabled=False
                )
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.no_grad(), gain_autocast:
                phi_base = phi_z.detach().float()
                for direction_idx in range(int(diagnostic_local_gain_directions)):
                    # Probe the fixed-point residual direction first: it is
                    # the direction the solver is actually trying to remove
                    # and is more informative than random average-case probes.
                    # Remaining directions are deterministic Gaussian probes.
                    if direction_idx == 0 and torch.linalg.vector_norm(
                        residual.detach().float()
                    ) > 0:
                        direction = residual.detach().float().clone()
                    else:
                        direction = torch.randn(
                            z_target.shape,
                            generator=gain_generator,
                            device=device,
                            dtype=torch.float32,
                        )
                    direction = direction / torch.linalg.vector_norm(
                        direction
                    ).clamp_min(1e-12)
                    delta = direction * delta_norm
                    phi_perturbed = backbone.deq_step(
                        z_target + delta,
                        context,
                        fixed_module_weights,
                        ex,
                    )
                    gain = torch.linalg.vector_norm(
                        phi_perturbed.float() - phi_base
                    ) / torch.linalg.vector_norm(delta).clamp_min(1e-12)
                    local_gains.append(float(gain.item()))

        picard_residuals: list[float] = []
        picard_secant_gains: list[float] = []
        picard_worst_secant_gain: Optional[float] = None
        picard_worst_secant_iteration: Optional[int] = None
        picard_worst_stable_secant_gain: Optional[float] = None
        picard_worst_stable_secant_iteration: Optional[int] = None
        if int(diagnostic_picard_steps) > 0:
            picard_autocast = (
                torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, cache_enabled=False
                )
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.no_grad(), picard_autocast:
                picard_state = z_initial.detach().float()
                previous_picard_state: Optional[torch.Tensor] = None
                previous_picard_mapped: Optional[torch.Tensor] = None
                previous_picard_residual: Optional[float] = None
                for picard_iteration in range(1, int(diagnostic_picard_steps) + 1):
                    picard_next = backbone.deq_step(
                        picard_state,
                        context,
                        fixed_module_weights,
                        ex,
                    ).float()
                    absolute = torch.linalg.vector_norm(
                        picard_next - picard_state
                    )
                    scale = torch.maximum(
                        torch.linalg.vector_norm(picard_state),
                        torch.linalg.vector_norm(picard_next),
                    ).clamp_min(1.0)
                    current_residual = float((absolute / scale).item())
                    picard_residuals.append(current_residual)
                    if (
                        previous_picard_state is not None
                        and previous_picard_mapped is not None
                        and previous_picard_residual is not None
                    ):
                        input_chord = picard_state - previous_picard_state
                        output_chord = picard_next - previous_picard_mapped
                        input_norm = torch.linalg.vector_norm(input_chord)
                        output_norm = torch.linalg.vector_norm(output_chord)
                        if (
                            bool(torch.isfinite(input_norm).item())
                            and bool(torch.isfinite(output_norm).item())
                            and float(input_norm.item()) > 1e-12
                        ):
                            secant_gain = float((output_norm / input_norm).item())
                            anchor_iteration = int(picard_iteration) - 1
                            picard_secant_gains.append(secant_gain)
                            if (
                                picard_worst_secant_gain is None
                                or (secant_gain, anchor_iteration)
                                >= (
                                    picard_worst_secant_gain,
                                    int(picard_worst_secant_iteration or 0),
                                )
                            ):
                                picard_worst_secant_gain = secant_gain
                                picard_worst_secant_iteration = anchor_iteration
                            if (
                                anchor_iteration >= 5
                                and previous_picard_residual
                                >= max(10.0 * float(equilibrium_tol), 1e-2)
                                and (
                                    picard_worst_stable_secant_gain is None
                                    or (secant_gain, anchor_iteration)
                                    >= (
                                        picard_worst_stable_secant_gain,
                                        int(
                                            picard_worst_stable_secant_iteration
                                            or 0
                                        ),
                                    )
                                )
                            ):
                                picard_worst_stable_secant_gain = secant_gain
                                picard_worst_stable_secant_iteration = (
                                    anchor_iteration
                                )
                    previous_picard_state = picard_state
                    previous_picard_mapped = picard_next
                    previous_picard_residual = current_residual
                    picard_state = picard_next

    loss_wproj = torch.zeros((), device=device)
    if lambda_wproj > 0.0 and target_text and tokenizer is not None:
        loss_wproj = wproj_teacher_forcing_loss(
            backbone,
            phi_z,
            parent_text=decode_parent_text or parent_text,
            target_text=target_text,
            tokenizer=tokenizer,
            max_target_tokens=max_target_tokens,
        )

    loss_ce = torch.zeros((), device=device)
    if lambda_lm > 0.0 and tokenizer is not None and hasattr(backbone, "cg"):
        preservation_target = lm_target_text or target_text
        if not preservation_target:
            raise ValueError(
                "Stage-1 LM preservation requires a non-empty solution target"
            )
        loss_ce = language_model_preservation_loss(
            backbone,
            parent_text=parent_text,
            target_text=preservation_target,
            tokenizer=tokenizer,
            max_target_tokens=max_lm_target_tokens,
        )

    total_loss = (
        loss_fp
        + lambda_lm * loss_ce
        + lambda_wproj * loss_wproj
        + contraction_lambda * loss_contraction
        + lambda_c * loss_contraction_penalty
    )
    if metrics_out is not None:
        with torch.no_grad():
            residual_history = (
                list(getattr(solve_info, "all_residuals", []) or [])
                if solve_info is not None
                else list((cached_solve_diag or {}).get("all_residuals", []) or [])
            )
            solve_converged = (
                bool(getattr(solve_info, "converged", False))
                if solve_info is not None
                else bool((cached_solve_diag or {}).get("converged", False))
            )
            solve_iterations = (
                int(getattr(solve_info, "iterations", 0))
                if solve_info is not None
                else int((cached_solve_diag or {}).get("iterations", 0))
            )
            solve_residual = (
                float(getattr(solve_info, "residual_norm", float("nan")))
                if solve_info is not None
                else float((cached_solve_diag or {}).get("residual_norm", float("nan")))
            )
            solve_failure = (
                getattr(solve_info, "failure_reason", None)
                if solve_info is not None
                else (cached_solve_diag or {}).get("failure_reason")
            )
            metrics_out.update(
                {
                    "loss_total": float(total_loss.detach().float().item()),
                    "loss_fp": float(loss_fp.detach().float().item()),
                    "loss_fp_mean": float(
                        loss_fp.detach().float().item() / z_target.numel()
                    ),
                    "fixed_point_abs_residual_norm": float(
                        torch.linalg.vector_norm(
                            (phi_z.detach().float() - z_target.detach().float())
                        ).item()
                    ),
                    "loss_wproj": float(loss_wproj.detach().float().item()),
                    "loss_lm": float(loss_ce.detach().float().item()),
                    "loss_contraction_penalty": float(
                        loss_contraction_penalty.detach().float().item()
                    ),
                    "decay_ratio_mean": (
                        float(decay_ratio.mean().detach().float().item())
                        if decay_ratio is not None
                        else None
                    ),
                    "decay_ratio_max": (
                        float(decay_ratio.max().detach().float().item())
                        if decay_ratio is not None
                        else None
                    ),
                    "loss_contraction": float(
                        loss_contraction.detach().float().item()
                    ),
                    "loss_contraction_weighted": float(
                        contraction_lambda
                        * loss_contraction.detach().float().item()
                    ),
                    "contraction_lambda": float(
                        contraction_lambda
                    ),
                    "contraction_target_gain": float(
                        diagnostic_contraction_target_gain
                    ),
                    "contraction_mode": contraction_mode,
                    "teacher_selector_secant_gain": teacher_secant_gain,
                    "teacher_anchor_iteration": teacher_anchor_iteration,
                    "teacher_anchor_residual": teacher_anchor_residual,
                    "teacher_secant_selector": (
                        "worst_consecutive_picard_secant_v1"
                        if teacher_secant_gain is not None
                        else None
                    ),
                    # Backward-compatible alias for the first diagnostic logs.
                    "teacher_worst_secant_gain": teacher_secant_gain,
                    "contraction_gain": (
                        float(sampled_secant_gain.detach().float().item())
                        if sampled_secant_gain is not None
                        else None
                    ),
                    "contraction_endpoint_gain": (
                        float(sampled_secant_gain.detach().float().item())
                        if sampled_secant_gain is not None
                        and contraction_mode == "endpoint"
                        else None
                    ),
                    "contraction_trajectory_gain": (
                        float(sampled_secant_gain.detach().float().item())
                        if sampled_secant_gain is not None
                        and contraction_mode == "trajectory"
                        else None
                    ),
                    "sampled_secant_gain": (
                        float(sampled_secant_gain.detach().float().item())
                        if sampled_secant_gain is not None
                        else None
                    ),
                    "contraction_hinge": contraction_details.get("hinge"),
                    "contraction_active": bool(
                        contraction_details.get("hinge", 0.0) > 0.0
                    ),
                    "contraction_input_chord_norm": (
                        contraction_details.get("input_chord_norm")
                    ),
                    "contraction_input_chord_relative": (
                        contraction_details.get("input_chord_relative")
                    ),
                    "contraction_output_chord_norm": (
                        contraction_details.get("output_chord_norm")
                    ),
                    "contraction_scale_n": contraction_details.get("scale_n"),
                    "contraction_extra_phi_evals": (
                        (1 if contraction_mode == "endpoint" else 2)
                        if sampled_secant_gain is not None
                        else 0
                    ),
                    "z_input_norm": float(
                        torch.linalg.vector_norm(
                            z_initial.detach().float()
                        ).item()
                    ),
                    "root_initialization": root_initialization,
                    "equilibrium_solver": str(equilibrium_solver),
                    "z_target_source": z_target_source,
                    "routing_module_ids": [
                        int(x)
                        for x in selected_module_ids.detach().cpu().tolist()
                    ],
                    "routing_support_fixed": bool(support_override is not None),
                    "routing_weights_frozen": bool(
                        diagnostic_routing_weights_cache is not None
                    ),
                    "routing_margin_topk_next": routing_margin,
                    "routing_entropy": routing_entropy,
                    "z_target_norm": float(
                        torch.linalg.vector_norm(z_target.detach().float()).item()
                    ),
                    "phi_norm": float(
                        torch.linalg.vector_norm(phi_z.detach().float()).item()
                    ),
                    "z_target_absmax": float(
                        z_target.detach().float().abs().max().item()
                    ),
                    "phi_absmax": float(
                        phi_z.detach().float().abs().max().item()
                    ),
                    "equilibrium_converged": solve_converged,
                    "equilibrium_iterations": solve_iterations,
                    "equilibrium_residual": solve_residual,
                    "equilibrium_min_residual": (
                        float(min(residual_history))
                        if residual_history
                        else float("nan")
                    ),
                    "equilibrium_residual_history": list(residual_history),
                    "equilibrium_failure": solve_failure,
                    # Backward-compatible names retained for existing log
                    # consumers.  For a diagnostic Picard target, prefer the
                    # generic ``equilibrium_*`` fields above.
                    "broyden_converged": solve_converged,
                    "broyden_iterations": solve_iterations,
                    "broyden_residual": solve_residual,
                    "broyden_min_residual": (
                        float(min(residual_history))
                        if residual_history
                        else float("nan")
                    ),
                    "broyden_failure": solve_failure,
                    "local_gain_proxy_directions": len(local_gains),
                    "local_gain_proxy_epsilon": (
                        float(diagnostic_local_gain_epsilon)
                        if local_gains
                        else None
                    ),
                    "local_gain_proxy_values": list(local_gains),
                    "local_gain_proxy_median": (
                        float(torch.tensor(local_gains).median().item())
                        if local_gains
                        else None
                    ),
                    "local_gain_proxy_max": (
                        max(local_gains) if local_gains else None
                    ),
                    "local_gain_proxy_residual_direction": (
                        local_gains[0] if local_gains else None
                    ),
                    "picard_steps": len(picard_residuals),
                    "picard_residuals": list(picard_residuals),
                    "picard_final_residual": (
                        picard_residuals[-1] if picard_residuals else None
                    ),
                    "picard_min_residual": (
                        min(picard_residuals) if picard_residuals else None
                    ),
                    "picard_secant_gains": list(picard_secant_gains),
                    "picard_worst_secant_gain": picard_worst_secant_gain,
                    "picard_worst_secant_iteration": (
                        picard_worst_secant_iteration
                    ),
                    "picard_worst_stable_secant_gain": (
                        picard_worst_stable_secant_gain
                    ),
                    "picard_worst_stable_secant_iteration": (
                        picard_worst_stable_secant_iteration
                    ),
                }
            )
    return total_loss


def _response_target_ids(
    backbone: nn.Module,
    tokenizer: Any,
    target_text: str,
    *,
    device: torch.device,
    max_target_tokens: int,
) -> torch.Tensor:
    """Tokenize an assistant response and include its canonical turn stop.

    Gemma-4-IT terminates an assistant turn with ``<turn|>`` (id 106 in the
    paper tokenizer).  Training without that token teaches W_proj to emit an
    answer but never teaches the 64-token decoder to stop.  The token is
    resolved from the tokenizer rather than hard-coded so model revisions stay
    auditable.
    """
    budget = int(max_target_tokens)
    if budget < 2:
        raise ValueError("max_target_tokens must be >= 2")

    stop_id: Optional[int] = None
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(converter):
        candidate = converter("<turn|>")
        unk = getattr(tokenizer, "unk_token_id", None)
        if candidate is not None and int(candidate) >= 0 and candidate != unk:
            stop_id = int(candidate)
    if stop_id is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            stop_id = int(eos)

    content_budget = budget - (1 if stop_id is not None else 0)
    target_enc = tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, content_budget),
    )
    target_ids = target_enc["input_ids"].to(device)
    if target_ids.dim() != 2 or target_ids.shape[0] != 1:
        raise ValueError("assistant target must tokenize as one [1,T] sequence")
    if target_ids.numel() == 0:
        raise ValueError("assistant target tokenization produced no tokens")
    if stop_id is not None and int(target_ids[0, -1].item()) != stop_id:
        stop = torch.tensor([[stop_id]], device=device, dtype=target_ids.dtype)
        target_ids = torch.cat([target_ids, stop], dim=1)

    # A chat-template-backed Gemma run must have an explicit response stop.
    if (
        getattr(tokenizer, "chat_template", None) is not None
        and stop_id is None
    ):
        raise RuntimeError("chat tokenizer does not expose a response stop token")
    return target_ids


def _apply_gemma_final_logit_softcap(
    causal_lm: nn.Module,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact Gemma conditional-generation logit contract."""
    text_config = getattr(causal_lm, "config", None)
    if text_config is not None and hasattr(text_config, "get_text_config"):
        text_config = text_config.get_text_config()
    if text_config is None:
        language_model = getattr(getattr(causal_lm, "model", None), "language_model", None)
        text_config = getattr(language_model, "config", None)
    final_logit_softcapping = getattr(
        text_config, "final_logit_softcapping", None
    )
    if final_logit_softcapping is None:
        return logits
    cap = float(final_logit_softcapping)
    return torch.tanh(logits / cap) * cap


def language_model_preservation_loss(
    backbone: nn.Module,
    *,
    parent_text: str,
    target_text: str,
    tokenizer: Any,
    max_target_tokens: int = 256,
) -> torch.Tensor:
    """Canonical prompt-to-solution CE used by Stage-1 LoRA preservation."""
    lm = backbone.cg.model.language_model
    device = lm.embed_tokens.weight.device
    if hasattr(backbone, "prompt_input_ids"):
        prompt_ids, _ = backbone.prompt_input_ids(parent_text)
    else:
        prompt_enc = tokenizer(
            parent_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        prompt_ids = prompt_enc["input_ids"].to(device)
    target_ids = _response_target_ids(
        backbone,
        tokenizer,
        target_text,
        device=device,
        max_target_tokens=max_target_tokens,
    )
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    with torch.set_grad_enabled(backbone.training):
        # Preserve the original Gemma4Model text/mask path, but stop before
        # Gemma4ForConditionalGeneration projects every ignored prompt row
        # through the 262k-way vocabulary head.
        out = backbone.cg.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = getattr(out, "last_hidden_state", None)
        if hidden_states is None:
            raise RuntimeError("Gemma model did not return preservation hidden states")
        target_len = int(target_ids.shape[1])
        prediction_start = int(prompt_ids.shape[1]) - 1
        supervised_hidden = hidden_states[
            :, prediction_start : prediction_start + target_len, :
        ]
        if supervised_hidden.shape[1] != target_len:
            raise RuntimeError(
                "shifted LM-preservation target span is out of bounds: "
                f"wanted {target_len}, got {supervised_hidden.shape[1]}"
            )
        logits = backbone.cg.lm_head(supervised_hidden)
        logits = _apply_gemma_final_logit_softcap(backbone.cg, logits)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1).to(logits.device),
        )


def wproj_teacher_forcing_loss(
    backbone: nn.Module,
    z_latent: torch.Tensor,
    *,
    parent_text: str,
    target_text: str,
    tokenizer: Any,
    max_target_tokens: int = 256,
) -> torch.Tensor:
    """Train W_proj through the same soft-prefix decoder used at evaluation.

    The soft latent prefix and original problem are context-only (label -100);
    the target solution tokens receive causal LM loss.  This is the missing
    objective that left every historical ``w_proj.weight`` checkpoint
    bit-identical to random initialisation.
    """
    if z_latent.ndim != 2:
        raise ValueError(
            "z_latent must have shape [latent_tokens, hidden_size], "
            f"got {tuple(z_latent.shape)}"
        )
    zero = torch.zeros((), device=z_latent.device)
    if not (
        hasattr(backbone, "w_proj")
        and hasattr(backbone, "cg")
        and hasattr(backbone.cg, "model")
    ):
        return zero

    lm = backbone.cg.model.language_model
    device = lm.embed_tokens.weight.device

    # Decoder context must be byte-for-byte the same at train and inference.
    # Evaluation passes the original problem text after the latent prefix, so
    # do not smuggle ``Solution:`` into the context only during Stage-1.  The
    # separator belongs to the supervised continuation instead.
    if hasattr(backbone, "prompt_input_ids"):
        prompt_ids, _ = backbone.prompt_input_ids(parent_text)
    else:
        prompt_enc = tokenizer(
            parent_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        prompt_ids = prompt_enc["input_ids"].to(device)
    target_ids = _response_target_ids(
        backbone,
        tokenizer,
        target_text,
        device=device,
        max_target_tokens=max_target_tokens,
    )
    model_dtype = lm.embed_tokens.weight.dtype
    prefix = backbone.w_proj(
        z_latent.to(
            device=lm.embed_tokens.weight.device,
            dtype=backbone.w_proj.weight.dtype,
        )
    ).to(dtype=model_dtype).unsqueeze(0)
    text_ids = torch.cat([prompt_ids, target_ids], dim=1)
    text_embeds = lm.embed_tokens(text_ids).to(dtype=model_dtype)
    inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
    raw_per_layer_inputs = raw_ple_for_mixed_public_forward(
        lm, prefix, text_ids, text_embeds
    )
    if inputs_embeds.ndim != 3:
        raise RuntimeError(
            "mixed W_proj decoder inputs must have shape [batch, sequence, hidden], "
            f"got {tuple(inputs_embeds.shape)}"
        )
    if raw_per_layer_inputs.ndim != 4:
        raise RuntimeError(
            "Gemma per-layer inputs must have shape "
            "[batch, sequence, layers, per_layer_hidden], "
            f"got {tuple(raw_per_layer_inputs.shape)}"
        )
    if raw_per_layer_inputs.shape[:2] != inputs_embeds.shape[:2]:
        raise RuntimeError(
            "mixed embeddings and per-layer inputs disagree on batch/sequence: "
            f"{tuple(inputs_embeds.shape[:2])} vs "
            f"{tuple(raw_per_layer_inputs.shape[:2])}"
        )
    if raw_per_layer_inputs.device != inputs_embeds.device:
        raise RuntimeError("mixed embeddings and per-layer inputs must share a device")
    if raw_per_layer_inputs.dtype != inputs_embeds.dtype:
        raise RuntimeError("mixed embeddings and per-layer inputs must share a dtype")

    prefix_and_prompt = prefix.shape[1] + prompt_ids.shape[1]
    attention_mask = torch.ones(
        inputs_embeds.shape[:2], dtype=torch.long, device=device
    )
    # Do not call the installed multimodal Gemma4ForConditionalGeneration
    # wrapper with continuous inputs_embeds.  Its multimodal preprocessing
    # recomputes PLE and enters a reverse-embedding lookup over the 262k-token
    # vocabulary, materialising an O(B*S*V*H) comparison tensor.
    # The text decoder owns the public mixed-embedding contract and accepts the
    # precomputed raw PLE directly, so this path is both exact and bounded.
    out = lm(
        inputs_embeds=inputs_embeds,
        per_layer_inputs=raw_per_layer_inputs,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = getattr(out, "last_hidden_state", None)
    if hidden_states is None:
        raise RuntimeError("Gemma text decoder did not return last_hidden_state")

    # Labels before the response are all -100, so projecting those hidden
    # states through the 262k-way vocabulary head cannot affect the loss.  A
    # positionwise lm_head over supervised shifted rows only is mathematically
    # identical and avoids another large [sequence, vocabulary] allocation.
    target_len = int(target_ids.shape[1])
    prediction_start = int(prefix_and_prompt) - 1
    prediction_stop = prediction_start + target_len
    supervised_hidden = hidden_states[
        :, prediction_start:prediction_stop, :
    ]
    if supervised_hidden.shape[1] != target_len:
        raise RuntimeError(
            "shifted W_proj target span is out of bounds: "
            f"wanted {target_len}, got {supervised_hidden.shape[1]}"
        )
    logits = backbone.cg.lm_head(supervised_hidden)
    logits = _apply_gemma_final_logit_softcap(backbone.cg, logits)

    # Match Gemma4ForConditionalGeneration.forward exactly: loss logits are
    # float32 and token t predicts label t+1.  Ignored context rows were
    # removed before the positionwise head rather than after it.
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1).to(logits.device),
    )


def run_stage1_demo_step(
    *,
    lr: float = 1e-2,
    device: torch.device | None = None,
) -> Tuple[float, MockTinyBackbone]:
    """One Adam step on `MockTinyBackbone`; returns (loss_value, backbone)."""
    dev = device or torch.device("cpu")
    bb = MockTinyBackbone(hidden=64, num_layers=42).to(dev)
    opt = torch.optim.Adam(bb.parameters(), lr=lr)
    nu = NuVector(nu_tol=5e-3, nu_temp=1.0, nu_expl=1.0)
    w_g = torch.randn(19, 64, device=dev) * 0.02
    z = torch.randn(8, 64, device=dev)
    opt.zero_grad()
    loss = fixed_point_surrogate_loss(bb, "demo prompt", z, nu, w_g=w_g)
    loss.backward()
    opt.step()
    return float(loss.detach().cpu().item()), bb

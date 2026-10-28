"""
Stage 1: OpenMathInstruct JSONL + GemmaCTSBackbone + IFT residual loss (paper §6.1).

Base Gemma weights are frozen; trains `routing_proj`, `_blend`, and optional LoRA adapters.
LoRA r=8, ~18 MB trainable parameters. 10,000 examples from OpenMathInstruct-2.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional
import numpy as np

import torch
import torch.nn as nn

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.model.gemma_loader import load_gemma4_e4b
from cts.train.jsonl_iter import iter_jsonl
from cts.train.lora_compat import apply_paper_lora
from cts.train.openmath_text import (
    prompt_text_from_openmath_row,
    solution_text_from_openmath_row,
    target_text_from_openmath_row,
)
from cts.prompt_contract import (
    MATH_CONCISE_BOXED_CONTRACT,
    build_math_decode_prompt,
)
from cts.train.stage1_warmup import fixed_point_surrogate_loss
from cts.types import NuVector
from cts.utils.config import load_config
from cts.utils.repro_seed import apply_global_seed


def _set_trainable_params(bb: GemmaCTSBackbone, *, train_lora: bool) -> None:
    """Enable gradients for the Stage 1 trainable parameter set (paper §6.1).

    Paper §6.1: "Frozen Gemma 4; LoRA (r=8, alpha=16) at q/v/o_proj; routing
    projection W_g; and W_proj trained on 10,000 OpenMathInstruct-2 examples."

    Trainable set therefore = ``routing_proj`` (W_g), ``w_proj`` (W_proj),
    and LoRA adapters (when ``train_lora=True``). ``_blend`` belongs only to
    the explicit blend ablation and is frozen on the paper parallel path.
    """
    for n, p in bb.named_parameters():
        if "routing_proj" in n:
            p.requires_grad = True
        elif "_blend" in n and str(getattr(bb, "deq_map_mode", "parallel")) == "blend":
            p.requires_grad = True
        elif "w_proj" in n:
            # Paper-aligned: W_proj is part of the Stage 1 learned set.
            # Previously this parameter was silently frozen (NeurIPS audit
            # Apr 2026 finding P0-2), so the latent->token decoding head
            # was never updated and the headline 50.2 % AIME number was
            # not reproducible from these checkpoints.
            p.requires_grad = True
        elif "gamma_param" in n:
            p.requires_grad = True
        elif "lora_" in n and train_lora:
            p.requires_grad = True
        else:
            p.requires_grad = False


def _promote_stage1_trainables_to_fp32(bb: nn.Module) -> None:
    """Keep trainable Stage-1 weights and AdamW state in FP32.

    The frozen Gemma compute remains BF16.  PEFT-style adapters normally keep
    FP32 master parameters; creating LoRA/W_g/W_proj directly in the base BF16
    dtype also made Adam's first/second moments BF16 and quantized the small
    warm-up updates that must establish DEQ contraction.
    """
    for param in bb.parameters():
        if param.requires_grad and param.is_floating_point():
            param.data = param.data.float()
    
    # Also promote spectral_norm parametrization buffers and original weights to FP32
    for name, module in bb.named_modules():
        if "parametrizations" in name or hasattr(module, "original"):
            for p in module.parameters(recurse=False):
                if p.is_floating_point():
                    p.data = p.data.float()
            for b_name, b_val in list(module.named_buffers(recurse=False)):
                if b_val is not None and b_val.is_floating_point():
                    module.register_buffer(b_name, b_val.float())


def _sample_stage1_z0(
    K: int,
    hidden_size: int,
    device: torch.device,
    *,
    fixed_seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample the Stage-1 latent initial state.

    ``fixed_seed`` is diagnostic-only: a fresh private generator is recreated
    on every call so repeated calls produce exactly the same ``z0`` without
    resetting the global RNG used by the rest of training.
    """
    generator = None
    if fixed_seed is not None:
        seed = int(fixed_seed)
        if not (0 <= seed < 2**63):
            raise ValueError("diagnostic fixed-z0 seed must be in [0, 2**63)")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    return torch.randn(
        int(K),
        int(hidden_size),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) * 0.02


def _require_converged_diagnostic_teacher(
    metrics: Dict[str, Any],
    *,
    solver: str,
    fixed_z_target: bool,
) -> None:
    """Fail before backward when a claimed equilibrium teacher is invalid.

    This gate is deliberately limited to the diagnostic Picard + fixed-target
    arm.  The production Broyden training contract is unchanged.  Without the
    gate, a max-iteration Picard best iterate could be cached and mislabeled as
    an equilibrium teacher for every subsequent diagnostic step.
    """
    if str(solver) != "picard" or not bool(fixed_z_target):
        return
    if not bool(metrics.get("equilibrium_converged", False)):
        raise RuntimeError(
            "diagnostic Picard fixed-point teacher did not converge; "
            "refusing to cache/train on a non-equilibrium target"
        )


def _maybe_apply_lora(
    bb: GemmaCTSBackbone,
    *,
    rank: int,
    target_modules: list[str],
) -> GemmaCTSBackbone:
    """Paper §6.1: LoRA r=8, α=16 on q/v/o_proj of the language model.

    Thin wrapper around :func:`cts.train.lora_compat.apply_paper_lora`
    that keeps Stage 1's existing call site stable while moving the
    actual implementation into :mod:`cts.train.lora_compat` so Stage 2
    PPO and the eval pipelines can reuse the same adapter geometry.
    """
    return apply_paper_lora(
        bb,
        rank=rank,
        target_modules=tuple(target_modules),
        dropout=0.05,
        require_match=True,
    )


def _save_checkpoint(
    bb: GemmaCTSBackbone,
    opt: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    config_name: str,
    openmath_jsonl: str,
    lora: bool,
    losses: list[float],
    path: Path,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    examples_seen: int = 0,
    training_contract: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = dict(training_contract or {})
    contract["accuracy_contract_valid"] = bool(
        int(step) == int(total_steps) == 5_000
        and bool(lora)
        and int(contract.get("contract_version", 0)) == 5
        and int(examples_seen) == 10_000
        and bool(contract.get("solve_equilibrium", False))
        and int(contract.get("equilibrium_max_iter", 0)) == 30
        and abs(float(contract.get("equilibrium_tol", -1.0)) - 1e-3) < 1e-12
        and str(contract.get("deq_map_mode")) in {"parallel", "paper"}
        and int(contract.get("latent_tokens_K", 0)) == 64
        and int(contract.get("max_target_tokens", 0)) == 64
        and int(contract.get("max_lm_target_tokens", 0)) == 256
        and float(contract.get("lambda_wproj", 0.0)) > 0.0
        and int(contract.get("stage1_batch_size", 0)) == 2
        and abs(float(contract.get("stage1_lr", -1.0)) - 1e-4) < 1e-12
        and int(contract.get("stage1_warmup_steps", -1)) == 100
        and str(contract.get("stage1_lr_schedule")) == "cosine"
        and abs(float(contract.get("stage1_lambda_lm", -1.0)) - 0.1) < 1e-12
        and str(contract.get("optimizer")) == "AdamW"
        and contract.get("wproj_target_contract")
        == "boxed_final_answer_plus_turn_stop_v2"
        and contract.get("lm_target_contract")
        == "full_solution_plus_turn_stop_v2"
        and contract.get("wproj_decode_prompt_contract")
        == MATH_CONCISE_BOXED_CONTRACT
        and contract.get("eq3_operator_contract")
        == "explicit_suffix_bf16_v1"
        and bool(contract.get("eq3_explicit_masks", False))
        and bool(contract.get("eq3_suffix_cache", False))
        and bool(contract.get("eq3_suffix_cache_train", False))
        and bool(contract.get("deterministic_deq_dropout", False))
        and contract.get("broyden_failure_return_policy")
        == "best_relative_finite_lm_v1"
        and contract.get("fixed_point_loss_reduction")
        == "squared_l2_sum_v1"
        and abs(float(contract.get("stage1_max_grad_norm", -1.0)) - 1.0)
        < 1e-12
        and contract.get("trainable_parameter_precision")
        == "fp32_master_v1"
        and contract.get("root_initialization_contract")
        == "fullstack_zero_suffix_v1"
        and contract.get("no_grad_autocast_cache_contract")
        == "disabled_v1"
        and bool(contract.get("lora_enabled", False))
        and int(contract.get("lora_rank", 0)) == 8
        and abs(float(contract.get("lora_dropout", -1.0)) - 0.05) < 1e-12
        and sorted(contract.get("lora_targets", []))
        == ["o_proj", "q_proj", "v_proj"]
        and bool(contract.get("model_id"))
        and len(str(contract.get("openmath_sha256", ""))) == 64
        and all(
            ch in "0123456789abcdef"
            for ch in str(contract.get("openmath_sha256", "")).lower()
        )
    )
    torch.save(
        {
            "schema_version": 2,
            "backbone_state_dict": bb.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "step": step,
            "total_steps": total_steps,
            "examples_seen": int(examples_seen),
            "config_name": config_name,
            "openmath_jsonl": openmath_jsonl,
            "lora": lora,
            "losses": losses[-200:],
            "training_contract": contract,
        },
        path,
    )
    print(f"  [checkpoint] saved step {step} → {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stage1_openmath_training(
    *,
    openmath_jsonl: Path | str,
    config_name: str = "default",
    max_steps: Optional[int] = None,
    device: Optional[str] = None,
    lora: bool = False,
    lora_targets: Optional[list[str]] = None,
    log_every: int = 20,
    model_dir: Optional[str] = None,
    resume: bool = False,
    save_every: int = 500,
    smoke_no_save: bool = False,
    diagnostic_no_save: bool = False,
    diagnostic_stop_after: Optional[int] = None,
    diagnostic_components: bool = False,
    diagnostic_fixed_z0_seed: Optional[int] = None,
    diagnostic_fixed_routing_support: bool = False,
    diagnostic_fixed_z_target: bool = False,
    diagnostic_freeze_routing_weights: bool = False,
    diagnostic_local_gain_directions: int = 0,
    diagnostic_local_gain_every: int = 10,
    diagnostic_local_gain_epsilon: float = 1e-2,
    diagnostic_target_solver: str = "broyden",
    diagnostic_target_max_iter: Optional[int] = None,
    diagnostic_contraction_lambda: float = 0.0,
    diagnostic_contraction_target_gain: float = 0.90,
    diagnostic_contraction_mode: str = "endpoint",
    lambda_c: float = 0.0,
    c_target: float = 0.92,
    resume_from: Optional[str] = None,
    save_ckpt_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = load_config(config_name)
    apply_global_seed()
    deq_from_cfg = str(cfg.get("cts_deq_map_mode", "parallel"))
    if deq_from_cfg not in {"parallel", "paper", "full", "blend"}:
        raise ValueError(f"unsupported Stage-1 DEQ map mode: {deq_from_cfg}")
    # Config is the run manifest.  A stale shell variable must not silently
    # turn a paper training run into the cheap blend ablation.
    os.environ["CTS_DEQ_MAP_MODE"] = deq_from_cfg
    os.environ["CTS_EQ3_EXPLICIT_MASKS"] = (
        "1" if bool(cfg.get("eq3_explicit_masks", False)) else "0"
    )
    os.environ["CTS_EQ3_SUFFIX_CACHE"] = (
        "1" if bool(cfg.get("eq3_suffix_cache", False)) else "0"
    )
    os.environ["CTS_EQ3_SUFFIX_CACHE_TRAIN"] = (
        "1" if bool(cfg.get("eq3_suffix_cache_train", False)) else "0"
    )
    steps = int(
        max_steps if max_steps is not None else cfg.get("stage1_max_steps", 5000)
    )
    equilibrium_max_iter = int(
        cfg.get("stage1_equilibrium_max_iter", 512)
    )
    if smoke_no_save:
        if resume:
            raise ValueError("Stage-1 smoke-no-save cannot resume a checkpoint")
        if steps > 2:
            raise ValueError("Stage-1 smoke-no-save is capped at two steps")
        save_every = 0
    if diagnostic_no_save:
        if resume:
            raise ValueError("Stage-1 diagnostic-no-save cannot resume a checkpoint")
        run_until_step = int(
            diagnostic_stop_after
            if diagnostic_stop_after is not None
            else steps
        )
        if not (1 <= run_until_step <= min(steps, 300)):
            raise ValueError(
                "Stage-1 diagnostic stop must be in [1, min(total_steps, 300)]"
            )
        save_every = 0
    else:
        if diagnostic_stop_after is not None:
            raise ValueError(
                "diagnostic_stop_after requires diagnostic_no_save=True"
            )
        run_until_step = steps
    if diagnostic_fixed_z0_seed is not None and not diagnostic_no_save:
        raise ValueError(
            "diagnostic_fixed_z0_seed requires diagnostic_no_save=True"
        )
    if diagnostic_fixed_routing_support and not diagnostic_no_save:
        raise ValueError(
            "diagnostic_fixed_routing_support requires diagnostic_no_save=True"
        )
    if diagnostic_fixed_z_target:
        if not diagnostic_no_save:
            raise ValueError(
                "diagnostic_fixed_z_target requires diagnostic_no_save=True"
            )
        if not diagnostic_fixed_routing_support:
            raise ValueError(
                "diagnostic_fixed_z_target requires fixed routing support"
            )
    if diagnostic_freeze_routing_weights:
        if not diagnostic_no_save:
            raise ValueError(
                "diagnostic_freeze_routing_weights requires diagnostic_no_save=True"
            )
        if not diagnostic_fixed_routing_support:
            raise ValueError(
                "diagnostic_freeze_routing_weights requires fixed routing support"
            )
    if int(diagnostic_local_gain_directions) < 0:
        raise ValueError("diagnostic_local_gain_directions must be >= 0")
    if int(diagnostic_local_gain_directions) > 0:
        if not diagnostic_no_save:
            raise ValueError(
                "diagnostic local-gain probing requires diagnostic_no_save=True"
            )
        if int(diagnostic_local_gain_every) <= 0:
            raise ValueError("diagnostic_local_gain_every must be >= 1")
    if str(diagnostic_target_solver) not in {"broyden", "picard"}:
        raise ValueError("diagnostic_target_solver must be broyden or picard")
    if str(diagnostic_target_solver) != "broyden" and not diagnostic_no_save:
        raise ValueError(
            "non-Broyden target solvers are diagnostic-only and require no-save"
        )
    if diagnostic_target_max_iter is not None and not diagnostic_no_save:
        raise ValueError(
            "diagnostic_target_max_iter requires diagnostic_no_save=True"
        )
    if not math.isfinite(float(diagnostic_contraction_lambda)) or float(
        diagnostic_contraction_lambda
    ) < 0.0:
        raise ValueError(
            "diagnostic_contraction_lambda must be finite and >= 0"
        )
    if str(diagnostic_contraction_mode) not in {"endpoint", "trajectory"}:
        raise ValueError(
            "diagnostic_contraction_mode must be endpoint or trajectory"
        )
    if float(diagnostic_contraction_lambda) > 0.0:
        if not diagnostic_no_save:
            raise ValueError(
                "diagnostic contraction loss requires diagnostic_no_save=True"
            )
        if not diagnostic_fixed_z_target:
            raise ValueError(
                "diagnostic contraction loss requires a fixed z target"
            )
        if not diagnostic_freeze_routing_weights:
            raise ValueError(
                "diagnostic contraction loss requires frozen routing weights"
            )
        if str(diagnostic_target_solver) != "picard":
            raise ValueError(
                "diagnostic contraction loss requires a Picard teacher"
            )
        if not math.isfinite(float(diagnostic_contraction_target_gain)) or not (
            0.0 < float(diagnostic_contraction_target_gain) < 1.0
        ):
            raise ValueError(
                "diagnostic_contraction_target_gain must be in (0, 1)"
            )
    diagnostic_equilibrium_max_iter = int(
        diagnostic_target_max_iter
        if diagnostic_target_max_iter is not None
        else equilibrium_max_iter
    )
    if diagnostic_equilibrium_max_iter < 1:
        raise ValueError("diagnostic_target_max_iter must be >= 1")
    K = int(cfg.get("soft_thought_K", 64))
    nu = NuVector(nu_tol=5e-3, nu_temp=1.0, nu_expl=1.0)

    dev_s = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_s)
    map_gpu = dev_s if dev_s.startswith("cuda") else None
    if map_gpu is None and dev.type == "cuda":
        map_gpu = str(dev)

    mid = (
        model_dir
        or os.environ.get("CTS_GEMMA_MODEL_DIR")
        or str(cfg.get("gemma_model_id", "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"))
    )
    assert "58c4d7b5" in str(mid), f"[BACKBONE LOCK VIOLATION] expected base snapshot (58c4d7b5), got {mid}"
    print(f"[BACKBONE-EVIDENCE] loading from: {mid}", flush=True)
    model, tok = load_gemma4_e4b(
        model_id=mid,
        device_map=map_gpu if map_gpu else "auto",
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
    )
    bb = GemmaCTSBackbone(model, tok)
    bb.train()

    targets = lora_targets or list(
        cfg.get("lora_target", ["q_proj", "v_proj", "o_proj"])
    )
    lora_rank = int(cfg.get("lora_rank", 8))
    if lora:
        bb = _maybe_apply_lora(
            bb, rank=lora_rank, target_modules=targets
        )
    _set_trainable_params(bb, train_lora=lora)
    _promote_stage1_trainables_to_fp32(bb)

    params = [p for p in bb.parameters() if p.requires_grad]
    # Stage 1 lr: prefer the dedicated `stage1_lr` key (paper §6.1: 1e-4)
    # over the legacy `lr` key (which is shared with PPO and defaults to
    # 3e-5). Falling back to `lr` keeps backward compatibility for older
    # configs that did not split the two stages.
    lr = float(cfg.get("stage1_lr", cfg.get("lr", 1e-4)))
    opt = torch.optim.AdamW(params, lr=lr)
    # Stage-1 trainables and AdamW moments are FP32 master values while the
    # frozen Gemma forward remains BF16. Dynamic loss scaling is unnecessary
    # for this BF16-compute/FP32-optimisation contract.
    scaler = None

    # Paper App. I: 100-step linear warm-up + cosine decay over the
    # remaining steps. We compose two SequentialLR-style schedulers.
    warmup_steps = int(cfg.get("stage1_warmup_steps", 100))
    lr_schedule = str(cfg.get("stage1_lr_schedule", "cosine")).lower()
    decay_steps = max(1, steps - warmup_steps)
    if lr_schedule == "cosine":
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=max(1, warmup_steps)
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=decay_steps)
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = (
            torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[max(1, warmup_steps)]
            )
        )
    else:
        scheduler = None

    # Paper §6.1: batch size 2. Implemented as gradient accumulation since
    # OpenMathInstruct-2 is iterated row-by-row (each row is one example).
    grad_accum_steps = max(1, int(cfg.get("stage1_batch_size", 1)))
    solve_equilibrium = bool(cfg.get("stage1_solve_equilibrium", True))
    lambda_wproj = float(cfg.get("stage1_lambda_wproj", 1.0))
    equilibrium_tol = float(cfg.get("stage1_equilibrium_tol", 5e-3))
    max_target_tokens = int(cfg.get("stage1_max_target_tokens", 256))
    max_lm_target_tokens = int(cfg.get("stage1_lm_max_target_tokens", 256))
    stage1_max_grad_norm = float(
        cfg.get("stage1_max_grad_norm", cfg.get("max_grad_norm", 1.0))
    )
    path = Path(openmath_jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"OpenMath JSONL not found: {path}")
    training_contract: Dict[str, Any] = {
        "contract_version": 5,
        "solve_equilibrium": solve_equilibrium,
        "equilibrium_max_iter": equilibrium_max_iter,
        "equilibrium_tol": equilibrium_tol,
        "lambda_wproj": lambda_wproj,
        "stage1_batch_size": grad_accum_steps,
        "stage1_lr": lr,
        "stage1_warmup_steps": warmup_steps,
        "stage1_lr_schedule": lr_schedule,
        "stage1_lambda_lm": float(cfg.get("stage1_lambda_lm", 0.1)),
        "optimizer": "AdamW",
        "max_target_tokens": max_target_tokens,
        "max_lm_target_tokens": max_lm_target_tokens,
        "deq_map_mode": str(bb.deq_map_mode),
        "latent_tokens_K": K,
        "model_id": str(mid),
        "openmath_sha256": _sha256_file(path),
        "wproj_target_contract": "boxed_final_answer_plus_turn_stop_v2",
        "lm_target_contract": "full_solution_plus_turn_stop_v2",
        "wproj_decode_prompt_contract": MATH_CONCISE_BOXED_CONTRACT,
        "eq3_operator_contract": str(
            cfg.get("eq3_operator_contract", "legacy_full_sequence")
        ),
        "eq3_explicit_masks": bool(cfg.get("eq3_explicit_masks", False)),
        "eq3_suffix_cache": bool(cfg.get("eq3_suffix_cache", False)),
        "eq3_suffix_cache_train": bool(
            cfg.get("eq3_suffix_cache_train", False)
        ),
        "deterministic_deq_dropout": True,
        "broyden_failure_return_policy": "best_relative_finite_lm_v1",
        "fixed_point_loss_reduction": "squared_l2_sum_v1",
        "stage1_max_grad_norm": stage1_max_grad_norm,
        "trainable_parameter_precision": "fp32_master_v1",
        "root_initialization_contract": "fullstack_zero_suffix_v1",
        "no_grad_autocast_cache_contract": "disabled_v1",
        "lora_enabled": bool(lora),
        "lora_rank": lora_rank,
        "lora_targets": sorted(str(x) for x in targets),
        "lora_dropout": 0.05,
    }

    losses: list[float] = []
    start_step = 0
    resume_examples_seen = 0
    ckpt_path = Path(save_ckpt_path) if save_ckpt_path else (Path("retrain/ckpt") / "stage1_last.pt")

    if resume_from:
        resume_source = Path(resume_from)
        if resume_source.exists():
            print(f"Loading base checkpoint weights from {resume_source} for Stage-1b ...")
            ckpt = torch.load(resume_source, map_location="cpu", weights_only=True)
            bb.load_state_dict(ckpt["backbone_state_dict"], strict=True)
            start_step = 0
            resume_examples_seen = 0
            print(f"  Loaded weights from {resume_source}, starting Stage-1b run for {steps} steps")
    elif resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if int(ckpt.get("schema_version", 0)) < 2:
            raise RuntimeError(
                "legacy Stage-1 checkpoints cannot be resumed into the "
                "v5 accuracy contract; start a fresh 5,000-step run"
            )
        saved_contract = ckpt.get("training_contract") or {}
        mismatches = {
            key: (saved_contract.get(key), value)
            for key, value in training_contract.items()
            if saved_contract.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Stage-1 resume contract mismatch; refusing provenance "
                f"migration: {mismatches}"
            )
        if int(ckpt.get("total_steps", -1)) != int(steps):
            raise RuntimeError("Stage-1 resume total_steps does not match this run")
        if bool(ckpt.get("lora", False)) != bool(lora):
            raise RuntimeError("Stage-1 resume LoRA setting does not match this run")
        bb.load_state_dict(ckpt["backbone_state_dict"], strict=True)
        if "optimizer_state_dict" not in ckpt:
            raise RuntimeError("Stage-1 v5 resume requires optimizer state")
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None:
            if ckpt.get("scheduler_state_dict") is None:
                raise RuntimeError("Stage-1 v5 resume requires scheduler state")
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        rng_state = ckpt.get("rng_state") or {}
        if rng_state.get("torch") is None:
            raise RuntimeError("Stage-1 v5 resume requires torch RNG state")
        torch.set_rng_state(rng_state["torch"])
        if dev.type == "cuda":
            if rng_state.get("cuda") is None:
                raise RuntimeError("Stage-1 CUDA resume requires CUDA RNG state")
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        start_step = int(ckpt.get("step", 0))
        if not (0 <= start_step <= steps):
            raise RuntimeError("Stage-1 resume step is outside the configured run")
        resume_examples_seen = int(ckpt.get("examples_seen", -1))
        if resume_examples_seen < 0:
            raise RuntimeError("Stage-1 v5 resume requires examples_seen")
        losses = list(ckpt.get("losses", []))
        print(f"  Resumed at step {start_step}, continuing to {steps}")

    row_iter = iter_jsonl(path)
    for _ in range(resume_examples_seen):
        try:
            next(row_iter)
        except StopIteration:
            row_iter = iter_jsonl(path)
            next(row_iter)

    # ``opt_step_idx`` counts only completed optimizer steps (the unit the
    # paper means by "5 000 steps" with batch=2). ``micro_idx`` counts each
    # individual forward/backward, so the loop runs
    # ``steps * grad_accum_steps`` micro-iterations to perform ``steps``
    # actual SGD updates.
    opt.zero_grad(set_to_none=True)
    opt_step_idx = start_step
    micro_idx = 0
    micro_loss_accum = 0.0
    teacher_step_conv = 0
    teacher_step_total = 0
    teacher_step_min_residuals: list[float] = []
    diagnostic_routing_support_ids: Optional[list[int]] = None
    diagnostic_z_target_cache: Optional[Dict[str, Any]] = (
        {} if diagnostic_fixed_z_target else None
    )
    diagnostic_routing_weights_cache: Optional[Dict[str, Any]] = (
        {} if diagnostic_freeze_routing_weights else None
    )
    target_micro = (run_until_step - start_step) * grad_accum_steps
    diagnostic_wall_start: Optional[float] = None
    if diagnostic_no_save:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        diagnostic_wall_start = perf_counter()

    while opt_step_idx < run_until_step and micro_idx < target_micro:
        try:
            row = next(row_iter)
        except StopIteration:
            row_iter = iter_jsonl(path)
            row = next(row_iter)

        text = prompt_text_from_openmath_row(row)
        decode_text = build_math_decode_prompt(text)
        target_text = target_text_from_openmath_row(row)
        lm_target_text = solution_text_from_openmath_row(row)
        d = bb.hidden_size
        z = _sample_stage1_z0(
            K,
            d,
            dev,
            fixed_seed=diagnostic_fixed_z0_seed,
        )
        w_g = bb.routing_matrix().to(device=dev, dtype=torch.float32)

        extra: Dict[str, Any] = {
            "top_k": int(cfg.get("top_k_modules", 3)),
            "deq_map_mode": bb.deq_map_mode,
        }
        if (
            diagnostic_fixed_routing_support
            and diagnostic_routing_support_ids is not None
        ):
            extra["routing_support_override"] = list(
                diagnostic_routing_support_ids
            )

        lambda_lm = float(cfg.get("stage1_lambda_lm", 0.1))
        # Always run the forward inside bf16 autocast on CUDA so the
        # model's bf16 weights match auto-promoted activations from
        # fp32 inputs (z is fp32 to keep GradScaler-free training
        # numerically stable). When scaler is enabled (fp16 path), the
        # backward path scales the loss; with scaler=None (bf16 path)
        # the backward runs at the original loss magnitude.
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if dev.type == "cuda"
            else torch.amp.autocast("cpu", enabled=False)
        )
        component_metrics: Optional[Dict[str, Any]] = {}
        probe_local_gain = bool(
            int(diagnostic_local_gain_directions) > 0
            and micro_idx % int(diagnostic_local_gain_every) == 0
        )
        with autocast_ctx:
            loss = fixed_point_surrogate_loss(
                bb, text, z, nu, w_g=w_g, extra=extra,
                lambda_lm=lambda_lm, tokenizer=tok,
                target_text=target_text,
                lm_target_text=lm_target_text,
                decode_parent_text=decode_text,
                lambda_wproj=lambda_wproj,
                lambda_c=float(lambda_c),
                c_target=float(c_target),
                solve_equilibrium=solve_equilibrium,
                equilibrium_tol=equilibrium_tol,
                equilibrium_max_iter=diagnostic_equilibrium_max_iter,
                equilibrium_solver=str(diagnostic_target_solver),
                diagnostic_contraction_lambda=float(
                    diagnostic_contraction_lambda
                ),
                diagnostic_contraction_target_gain=float(
                    diagnostic_contraction_target_gain
                ),
                diagnostic_contraction_mode=str(diagnostic_contraction_mode),
                max_target_tokens=max_target_tokens,
                max_lm_target_tokens=max_lm_target_tokens,
                metrics_out=component_metrics,
                diagnostic_z_target_cache=diagnostic_z_target_cache,
                diagnostic_routing_weights_cache=diagnostic_routing_weights_cache,
                diagnostic_local_gain_directions=(
                    int(diagnostic_local_gain_directions)
                    if probe_local_gain
                    else 0
                ),
                diagnostic_local_gain_epsilon=float(
                    diagnostic_local_gain_epsilon
                ),
            )
        if not bool(torch.isfinite(loss.detach()).item()):
            raise RuntimeError(
                f"Stage-1 produced a non-finite loss at micro-step {micro_idx + 1}"
            )
        if component_metrics is not None:
            _require_converged_diagnostic_teacher(
                component_metrics,
                solver=str(diagnostic_target_solver),
                fixed_z_target=bool(diagnostic_fixed_z_target),
            )
            if (
                diagnostic_fixed_routing_support
                and diagnostic_routing_support_ids is None
            ):
                diagnostic_routing_support_ids = list(
                    component_metrics["routing_module_ids"]
                )
            if diagnostic_components:
                print(
                    "[stage1-components] "
                    + json.dumps(
                        {"micro_step": micro_idx + 1, **component_metrics},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            is_conv = bool(component_metrics.get("broyden_converged", component_metrics.get("equilibrium_converged", False)))
            min_res = float(component_metrics.get("broyden_min_residual", component_metrics.get("equilibrium_min_residual", float("nan"))))
            teacher_step_total += 1
            if is_conv:
                teacher_step_conv += 1
            if math.isfinite(min_res):
                teacher_step_min_residuals.append(min_res)
        scaled_loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        micro_loss_accum += float(loss.detach().cpu().item())
        micro_idx += 1

        if micro_idx % grad_accum_steps != 0:
            # Still accumulating gradients for this batch.
            continue

        # Fail before the first optimiser update if a declared Stage-1
        # component is disconnected.  The old pipeline silently completed
        # 5k steps while W_proj and routing stayed bitwise at random init.
        if opt_step_idx == start_step:
            grad_sums: Dict[str, float] = {
                "routing_proj": 0.0,
                "w_proj": 0.0,
                "_blend": 0.0,
                "lora": 0.0,
            }
            for name, param in bb.named_parameters():
                if param.grad is None:
                    continue
                value = float(param.grad.detach().float().abs().sum().item())
                if "routing_proj" in name:
                    grad_sums["routing_proj"] += value
                elif "w_proj" in name:
                    grad_sums["w_proj"] += value
                elif "_blend" in name:
                    grad_sums["_blend"] += value
                elif "lora_" in name:
                    grad_sums["lora"] += value
            required = ["w_proj"]
            if not diagnostic_freeze_routing_weights:
                required.insert(0, "routing_proj")
            if str(bb.deq_map_mode) == "blend":
                required.append("_blend")
            if lora:
                required.append("lora")
            disconnected = [name for name in required if grad_sums[name] <= 0.0]
            print(f"[stage1] first-step gradient gate: {grad_sums}", flush=True)
            if disconnected:
                raise RuntimeError(
                    "Stage-1 gradient gate failed for "
                    f"{disconnected}; refusing to start a long run"
                )

        if diagnostic_components:
            grouped_grads: Dict[str, list[torch.Tensor]] = {
                "routing_proj": [],
                "w_proj": [],
                "blend": [],
                "lora": [],
            }
            for name, param in bb.named_parameters():
                if param.grad is None:
                    continue
                if "routing_proj" in name:
                    grouped_grads["routing_proj"].append(param.grad.detach())
                elif "w_proj" in name:
                    grouped_grads["w_proj"].append(param.grad.detach())
                elif "_blend" in name:
                    grouped_grads["blend"].append(param.grad.detach())
                elif "lora_" in name:
                    grouped_grads["lora"].append(param.grad.detach())

            group_norms: Dict[str, float] = {}
            for group_name, grads in grouped_grads.items():
                if not grads:
                    group_norms[group_name] = 0.0
                    continue
                per_param = torch._foreach_norm(grads, 2.0)
                group_norms[group_name] = float(
                    torch.linalg.vector_norm(
                        torch.stack([norm.float() for norm in per_param]), 2.0
                    ).item()
                )
            total_grad_norm = sum(v * v for v in group_norms.values()) ** 0.5
            max_grad_norm = stage1_max_grad_norm
            print(
                "[stage1-gradients] "
                + json.dumps(
                    {
                        "optimizer_step": opt_step_idx + 1,
                        "lr_before_step": float(opt.param_groups[0]["lr"]),
                        "group_l2": group_norms,
                        "total_l2": total_grad_norm,
                        "global_clip_scale": min(
                            1.0, max_grad_norm / max(total_grad_norm, 1e-12)
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if scaler is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                params, stage1_max_grad_norm
            )
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                params, stage1_max_grad_norm
            )
            opt.step()
        opt.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        opt_step_idx += 1

        if opt_step_idx % 250 == 0:
            from cts.train.lora_compat import LoraLinear
            with torch.no_grad():
                norms = []
                # 1. routing_proj (which is RoutingProjModule now)
                r_weight = bb.routing_proj.weight.float()
                _, S_r, _ = torch.linalg.svd(r_weight, full_matrices=False)
                norms.append(("routing_proj", S_r[0].item()))
                # 2. w_proj
                w_weight = bb.w_proj.weight.float()
                _, S_w, _ = torch.linalg.svd(w_weight, full_matrices=False)
                norms.append(("w_proj", S_w[0].item()))
                # 3. LoRA A
                for name, module in bb.named_modules():
                    if isinstance(module, LoraLinear):
                        la_weight = module.lora_A.weight.float()
                        _, S_la, _ = torch.linalg.svd(la_weight, full_matrices=False)
                        norms.append((f"{name}.lora_A", S_la[0].item()))
                
                max_sigma = max(val for _, val in norms)
                print(f"[CONTRACTIVITY-GATE] Step {opt_step_idx}: max_sigma = {max_sigma:.6f}", flush=True)
                for name, val in norms:
                    print(f"  {name} = {val:.6f}", flush=True)

                if max_sigma * 0.92 >= 0.95:
                    raise RuntimeError(
                        f"Contractivity gate violated at step {opt_step_idx}: "
                        f"max effective spectral norm is {max_sigma * 0.92:.6f} >= 0.95 "
                        f"(max raw sigma = {max_sigma:.6f})"
                    )

        lv = micro_loss_accum / grad_accum_steps
        micro_loss_accum = 0.0
        losses.append(lv)
        step = opt_step_idx - 1  # back-compat alias for downstream block
        if log_every and (step + 1) % log_every == 0:
            tail = sum(losses[-log_every:]) / min(len(losses), log_every)
            cur_lr = opt.param_groups[0]["lr"]
            log_msg = (
                f"stage1 step={step + 1}/{steps} loss={lv:.6f} "
                f"avg_last={tail:.6f} lr={cur_lr:.2e}"
            )
            if component_metrics and "decay_ratio_mean" in component_metrics and component_metrics["decay_ratio_mean"] is not None:
                log_msg += (
                    f" [contraction] decay_mean={component_metrics['decay_ratio_mean']:.4f} "
                    f"decay_max={component_metrics['decay_ratio_max']:.4f}"
                )
            print(log_msg, flush=True)
            if teacher_step_total > 0:
                med_res = float(np.median(teacher_step_min_residuals)) if teacher_step_min_residuals else float("nan")
                print(
                    f"[TEACHER] step={step + 1} conv_rate={teacher_step_conv}/{teacher_step_total} ({teacher_step_conv/teacher_step_total*100:.1f}%) min_res_median={med_res:.6f}",
                    flush=True,
                )
                teacher_step_conv = 0
                teacher_step_total = 0
                teacher_step_min_residuals.clear()

        if save_every and (step + 1) % save_every == 0:
            _save_checkpoint(
                bb, opt, step=step + 1, total_steps=steps,
                config_name=config_name, openmath_jsonl=str(path),
                lora=lora, losses=losses, path=ckpt_path,
                scheduler=scheduler,
                examples_seen=(step + 1) * grad_accum_steps,
                training_contract=training_contract,
            )
            step_prefix = ckpt_path.stem.replace("_last", "")
            step_ckpt = ckpt_path.parent / f"{step_prefix}_step{step + 1}.pt"
            _save_checkpoint(
                bb, opt, step=step + 1, total_steps=steps,
                config_name=config_name, openmath_jsonl=str(path),
                lora=lora, losses=losses, path=step_ckpt,
                scheduler=scheduler,
                examples_seen=(step + 1) * grad_accum_steps,
                training_contract=training_contract,
            )

    final_fresh_metrics: Optional[Dict[str, Any]] = None
    if diagnostic_fixed_z_target or diagnostic_freeze_routing_weights:
        final_fresh_metrics = {}
        final_autocast = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if dev.type == "cuda"
            else torch.amp.autocast("cpu", enabled=False)
        )
        with torch.no_grad(), final_autocast:
            fixed_point_surrogate_loss(
                bb,
                text,
                z,
                nu,
                w_g=bb.routing_matrix().to(device=dev, dtype=torch.float32),
                extra=extra,
                lambda_lm=0.0,
                lambda_wproj=0.0,
                solve_equilibrium=solve_equilibrium,
                equilibrium_tol=equilibrium_tol,
                equilibrium_max_iter=equilibrium_max_iter,
                equilibrium_solver="broyden",
                metrics_out=final_fresh_metrics,
                diagnostic_routing_weights_cache=diagnostic_routing_weights_cache,
                diagnostic_local_gain_directions=max(
                    4, int(diagnostic_local_gain_directions)
                ),
                diagnostic_local_gain_epsilon=float(
                    diagnostic_local_gain_epsilon
                ),
                diagnostic_picard_steps=30,
            )
        print(
            "[stage1-diagnostic-final-fresh] "
            + json.dumps(final_fresh_metrics, sort_keys=True),
            flush=True,
        )

    diagnostic_resources: Optional[Dict[str, Any]] = None
    if diagnostic_no_save:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        diagnostic_resources = {
            "wall_s_excluding_model_load": float(
                perf_counter() - float(diagnostic_wall_start)
            ),
            "peak_allocated_gib": (
                float(torch.cuda.max_memory_allocated(dev) / 1024**3)
                if dev.type == "cuda"
                else 0.0
            ),
            "peak_reserved_gib": (
                float(torch.cuda.max_memory_reserved(dev) / 1024**3)
                if dev.type == "cuda"
                else 0.0
            ),
            "device": str(dev),
        }
        print(
            "[stage1-diagnostic-resources] "
            + json.dumps(diagnostic_resources, sort_keys=True),
            flush=True,
        )

    no_save = bool(smoke_no_save or diagnostic_no_save)
    if not no_save:
        _save_checkpoint(
            bb, opt, step=steps, total_steps=steps,
            config_name=config_name, openmath_jsonl=str(path),
            lora=lora, losses=losses, path=ckpt_path,
            scheduler=scheduler,
            examples_seen=steps * grad_accum_steps,
            training_contract=training_contract,
        )
    if diagnostic_no_save and run_until_step < steps:
        print(
            f"Stage 1 diagnostic stop: {run_until_step}/{steps} steps, "
            f"final loss={losses[-1]:.6f}"
        )
    else:
        print(f"Stage 1 complete: {steps} steps, final loss={losses[-1]:.6f}")
    return {
        "checkpoint": None if no_save else str(ckpt_path),
        "final_loss": losses[-1] if losses else 0.0,
        "steps": run_until_step,
        "total_steps": steps,
        "diagnostic_final_fresh": final_fresh_metrics,
        "diagnostic_resources": diagnostic_resources,
    }

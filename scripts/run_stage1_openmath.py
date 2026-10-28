#!/usr/bin/env python3
"""Stage 1: OpenMath JSONL + Gemma + fixed-point surrogate (see `cts/train/stage1_openmath_train.py`)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.train.stage1_openmath_train import run_stage1_openmath_training


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="OpenMath JSONL (default: configs/data_paths.yaml openmath_train_jsonl)",
    )
    p.add_argument("--config", type=str, default="default")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--lora", action="store_true", default=True,
                   help="Apply PEFT LoRA r=8 on language_model (paper §6.1, default: on)")
    p.add_argument("--no-lora", dest="lora", action="store_false",
                   help="Disable LoRA (ablation only)")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--resume", action="store_true", help="Resume from artifacts/stage1_last.pt")
    p.add_argument("--resume-from", type=str, default=None, help="Resume base weights from specific checkpoint")
    p.add_argument("--save-path", type=str, default=None, help="Output checkpoint path")
    p.add_argument("--lambda-c", type=float, default=0.1, help="Stage-1b contraction penalty weight")
    p.add_argument("--c-target", type=float, default=0.92, help="Stage-1b contraction target ratio")
    p.add_argument("--model-dir", type=str, default=None, help="Override CTS_GEMMA_MODEL_DIR")
    p.add_argument(
        "--smoke-no-save",
        action="store_true",
        help="Run at most two optimizer steps without writing a 16-GB checkpoint",
    )
    p.add_argument(
        "--diagnostic-no-save",
        action="store_true",
        help="Run a bounded diagnostic without writing a checkpoint",
    )
    p.add_argument(
        "--diagnostic-stop-after",
        type=int,
        default=None,
        help=(
            "Stop a --diagnostic-no-save run after N optimizer steps while "
            "preserving the --max-steps LR schedule (maximum: 300)"
        ),
    )
    p.add_argument(
        "--diagnostic-components",
        action="store_true",
        help="Log Stage-1 loss components and Broyden scale diagnostics",
    )
    p.add_argument(
        "--diagnostic-fixed-z0-seed",
        type=int,
        default=None,
        help=(
            "Diagnostic only: recreate the same private-generator z0 for "
            "every micro-step without resetting the global training RNG"
        ),
    )
    p.add_argument(
        "--diagnostic-fixed-routing-support",
        action="store_true",
        help=(
            "Diagnostic only: freeze the first micro-step's top-k module IDs "
            "while continuing to train their soft routing weights"
        ),
    )
    p.add_argument(
        "--diagnostic-fixed-z-target",
        action="store_true",
        help=(
            "Diagnostic only: reuse the first Broyden target for every step; "
            "requires --diagnostic-fixed-routing-support and a single prompt"
        ),
    )
    p.add_argument(
        "--diagnostic-freeze-routing-weights",
        action="store_true",
        help=(
            "Diagnostic only: freeze the first step's complete sparse routing "
            "weights so LoRA/W_proj training is not dominated by W_g"
        ),
    )
    p.add_argument(
        "--diagnostic-local-gain-directions",
        type=int,
        default=0,
        help=(
            "Diagnostic only: sampled finite-difference local gain directions "
            "for Phi (not a spectral-radius estimate)"
        ),
    )
    p.add_argument(
        "--diagnostic-local-gain-every",
        type=int,
        default=10,
        help="Run the local-gain probe every N micro-steps",
    )
    p.add_argument(
        "--diagnostic-local-gain-epsilon",
        type=float,
        default=1e-2,
        help="Relative whole-state perturbation used by the local-gain probe",
    )
    p.add_argument(
        "--diagnostic-target-solver",
        choices=("broyden", "picard"),
        default="broyden",
        help="Diagnostic-only solver used to construct the detached Stage-1 target",
    )
    p.add_argument(
        "--diagnostic-target-max-iter",
        type=int,
        default=None,
        help="Diagnostic-only target-solver iteration budget",
    )
    p.add_argument(
        "--diagnostic-contraction-lambda",
        type=float,
        default=0.0,
        help=(
            "Diagnostic-only weight for the selected secant contraction hinge"
        ),
    )
    p.add_argument(
        "--diagnostic-contraction-target-gain",
        type=float,
        default=0.90,
        help="Target upper gain for the diagnostic secant contraction hinge",
    )
    p.add_argument(
        "--diagnostic-contraction-mode",
        choices=("endpoint", "trajectory"),
        default="endpoint",
        help=(
            "Endpoint uses one extra Phi; trajectory directly penalizes the "
            "selected consecutive Picard chord with two extra Phi calls"
        ),
    )
    args = p.parse_args()

    data_path = args.data
    if data_path is None:
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[1]
        dp = yaml.safe_load((root / "configs" / "data_paths.yaml").read_text(encoding="utf-8"))
        data_path = str(root / dp.get("openmath_train_jsonl", "data/openmath_instruct/train_100000.jsonl"))

    if args.model_dir:
        os.environ["CTS_GEMMA_MODEL_DIR"] = args.model_dir

    run_stage1_openmath_training(
        openmath_jsonl=data_path,
        config_name=args.config,
        max_steps=args.max_steps,
        device=args.device,
        lora=args.lora,
        log_every=args.log_every,
        model_dir=args.model_dir,
        resume=args.resume,
        save_every=args.save_every,
        smoke_no_save=args.smoke_no_save,
        diagnostic_no_save=args.diagnostic_no_save,
        diagnostic_stop_after=args.diagnostic_stop_after,
        diagnostic_components=args.diagnostic_components,
        diagnostic_fixed_z0_seed=args.diagnostic_fixed_z0_seed,
        diagnostic_fixed_routing_support=args.diagnostic_fixed_routing_support,
        diagnostic_fixed_z_target=args.diagnostic_fixed_z_target,
        diagnostic_freeze_routing_weights=args.diagnostic_freeze_routing_weights,
        diagnostic_local_gain_directions=args.diagnostic_local_gain_directions,
        diagnostic_local_gain_every=args.diagnostic_local_gain_every,
        diagnostic_local_gain_epsilon=args.diagnostic_local_gain_epsilon,
        diagnostic_target_solver=args.diagnostic_target_solver,
        diagnostic_target_max_iter=args.diagnostic_target_max_iter,
        diagnostic_contraction_lambda=args.diagnostic_contraction_lambda,
        diagnostic_contraction_target_gain=(
            args.diagnostic_contraction_target_gain
        ),
        diagnostic_contraction_mode=args.diagnostic_contraction_mode,
        lambda_c=args.lambda_c,
        c_target=args.c_target,
        resume_from=args.resume_from,
        save_ckpt_path=args.save_path,
    )


if __name__ == "__main__":
    main()

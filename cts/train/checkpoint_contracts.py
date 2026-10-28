"""Fail-closed checkpoint contracts for paper-accuracy CTS runs.

The predicates here are deliberately torch-free so training, evaluation and
post-run verification all use the same source of truth.  A self-asserted
``accuracy_contract_valid`` flag is necessary but never sufficient.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from cts.prompt_contract import MATH_CONCISE_BOXED_CONTRACT


class CheckpointContractError(RuntimeError):
    """Raised when a checkpoint cannot support a headline CTS result."""


def model_family_id(value: object) -> str:
    text = str(value or "").rstrip("/\\")
    return text.replace("\\", "/").split("/")[-1].lower()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text.lower())


def _close(value: object, expected: float, *, atol: float = 1e-12) -> bool:
    try:
        return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _raise(label: str, errors: Sequence[str]) -> None:
    if errors:
        raise CheckpointContractError(label + ": " + "; ".join(errors))


def stage1_accuracy_contract_is_valid(
    checkpoint: Mapping[str, Any],
    *,
    expected_model_id: str | None = None,
    require_declared_valid: bool = True,
) -> Dict[str, Any]:
    contract = checkpoint.get("training_contract") or {}
    state = checkpoint.get("backbone_state_dict")
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(int(checkpoint.get("schema_version", 0)) == 2, "schema_version=2")
    require(int(checkpoint.get("step", -1)) == 5_000, "step=5000")
    require(int(checkpoint.get("total_steps", -1)) == 5_000, "total_steps=5000")
    require(int(checkpoint.get("examples_seen", -1)) == 10_000, "examples_seen=10000")
    require(bool(checkpoint.get("lora", False)), "checkpoint lora=True")
    require(isinstance(state, Mapping), "backbone_state_dict mapping")
    if isinstance(state, Mapping):
        keys = [str(key) for key in state.keys()]
        require(any(k.endswith("lora_A.weight") for k in keys), "LoRA A tensors")
        require(any(k.endswith("lora_B.weight") for k in keys), "LoRA B tensors")
    require(int(contract.get("contract_version", 0)) == 5, "contract_version=5")
    if require_declared_valid:
        require(bool(contract.get("accuracy_contract_valid", False)), "declared Stage-1 accuracy validity")
    require(bool(contract.get("solve_equilibrium", False)), "solve_equilibrium=True")
    require(int(contract.get("equilibrium_max_iter", 0)) == 30, "equilibrium_max_iter=30")
    require(_close(contract.get("equilibrium_tol"), 1e-3), "equilibrium_tol=1e-3")
    require(str(contract.get("deq_map_mode")) in {"parallel", "paper"}, "parallel Eq.3")
    require(int(contract.get("latent_tokens_K", 0)) == 64, "latent_tokens_K=64")
    require(int(contract.get("max_target_tokens", 0)) == 64, "W_proj target tokens=64")
    require(int(contract.get("max_lm_target_tokens", 0)) == 256, "LM target tokens=256")
    require(float(contract.get("lambda_wproj", 0.0)) > 0.0, "lambda_wproj>0")
    require(int(contract.get("stage1_batch_size", 0)) == 2, "Stage-1 batch size=2")
    require(_close(contract.get("stage1_lr"), 1e-4), "Stage-1 lr=1e-4")
    require(int(contract.get("stage1_warmup_steps", -1)) == 100, "Stage-1 warmup=100")
    require(str(contract.get("stage1_lr_schedule")) == "cosine", "Stage-1 cosine schedule")
    require(_close(contract.get("stage1_lambda_lm"), 0.1), "lambda_lm=0.1")
    require(str(contract.get("optimizer")) == "AdamW", "optimizer=AdamW")
    require(contract.get("wproj_target_contract") == "boxed_final_answer_plus_turn_stop_v2", "W_proj target contract v2")
    require(contract.get("lm_target_contract") == "full_solution_plus_turn_stop_v2", "LM target contract v2")
    require(
        contract.get("wproj_decode_prompt_contract")
        == MATH_CONCISE_BOXED_CONTRACT,
        "W_proj decode prompt contract v1",
    )
    require(
        contract.get("eq3_operator_contract") == "explicit_suffix_bf16_v1",
        "explicit suffix Eq.3 operator v1",
    )
    require(bool(contract.get("eq3_explicit_masks", False)), "explicit Eq.3 masks")
    require(bool(contract.get("eq3_suffix_cache", False)), "Eq.3 suffix cache")
    require(
        bool(contract.get("eq3_suffix_cache_train", False)),
        "Stage-1 suffix-cache training",
    )
    require(bool(contract.get("deterministic_deq_dropout", False)), "deterministic DEQ dropout")
    require(
        contract.get("broyden_failure_return_policy")
        == "best_relative_finite_lm_v1",
        "best-finite limited-memory Broyden failure policy v1",
    )
    require(
        contract.get("fixed_point_loss_reduction")
        == "squared_l2_sum_v1",
        "squared-L2 Stage-1 fixed-point loss v1",
    )
    require(
        _close(contract.get("stage1_max_grad_norm"), 1.0),
        "Stage-1 max grad norm=1.0",
    )
    require(
        contract.get("trainable_parameter_precision") == "fp32_master_v1",
        "FP32 Stage-1 trainable master parameters v1",
    )
    require(
        contract.get("root_initialization_contract")
        == "fullstack_zero_suffix_v1",
        "prompt-conditioned full-stack root initialization v1",
    )
    require(
        contract.get("no_grad_autocast_cache_contract") == "disabled_v1",
        "no-grad autocast weight cache disabled v1",
    )
    require(bool(contract.get("lora_enabled", False)), "lora_enabled=True")
    require(int(contract.get("lora_rank", 0)) == 8, "LoRA rank=8")
    require(_close(contract.get("lora_dropout"), 0.05), "LoRA dropout=0.05")
    require(sorted(contract.get("lora_targets", [])) == ["o_proj", "q_proj", "v_proj"], "LoRA q/v/o targets")
    require(_is_sha256(contract.get("openmath_sha256")), "OpenMath SHA-256")
    require(bool(contract.get("model_id")), "model_id")
    if expected_model_id and contract.get("model_id"):
        require(model_family_id(contract.get("model_id")) == model_family_id(expected_model_id), "model family match")
    _raise("Stage-1 checkpoint is not paper-accuracy eligible", errors)
    return dict(contract)


STAGE2_FINAL_EXPECTED: Dict[str, Any] = {
    "total_steps": 10_000,
    "collect_batch": 64,
    "ppo_epochs": 4,
    "actor_lr": 3e-5,
    "critic_lr": 1e-4,
    "lambda_halt": 0.05,
    "transition_contract_version": 5,
    "root_initialization_contract": "algorithm1_direct_fullstack_zero_suffix_v2",
    "root_search_mac_contract": "root_excluded_from_search_budget_v1",
    "rollout_decode_tokens": 64,
    "latent_tokens_K": 64,
    "branching_W": 3,
    "top_k_modules": 3,
    "broyden_max_iter": 30,
    "broyden_tol_min": 1e-4,
    "broyden_tol_max": 1e-2,
    "tau_mac_budget": 1e14,
    "reward_mode": "answer",
    "wproj_decode_prompt_contract": MATH_CONCISE_BOXED_CONTRACT,
    "eq3_operator_contract": "explicit_suffix_bf16_v1",
    "eq3_explicit_masks": True,
    "eq3_suffix_cache": True,
}


def _stage2_common_errors(
    checkpoint: Mapping[str, Any],
    *,
    expected_stage1_sha256: str,
    expected_model_id: str,
) -> tuple[Mapping[str, Any], list[str]]:
    meta = checkpoint.get("training_meta") or {}
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(int(checkpoint.get("schema_version", 0)) == 3, "schema_version=3")
    for key in ("meta_policy_state_dict", "critic_state_dict", "value_head_state_dict"):
        require(isinstance(checkpoint.get(key), Mapping), f"canonical {key}")
    require(str(meta.get("stage1_sha256", "")) == str(expected_stage1_sha256), "exact Stage-1 SHA-256")
    require(bool(meta.get("stage1_accuracy_contract_valid", False)), "validated Stage-1 provenance")
    require(int(meta.get("stage1_contract_version", 0)) == 5, "Stage-1 contract v5")
    require(model_family_id(meta.get("model_id")) == model_family_id(expected_model_id), "model family match")
    require(int(checkpoint.get("W", 0)) == int(meta.get("branching_W", 0)), "top-level W matches metadata")
    require(_is_sha256(meta.get("stage2_dataset_sha256")), "Stage-2 dataset SHA-256")
    require(isinstance(meta.get("rollout_seed"), int), "rollout seed")
    return meta, errors


def validate_stage2_final_contract(
    checkpoint: Mapping[str, Any],
    *,
    expected_stage1_sha256: str,
    expected_model_id: str,
) -> Dict[str, Any]:
    meta, errors = _stage2_common_errors(
        checkpoint,
        expected_stage1_sha256=expected_stage1_sha256,
        expected_model_id=expected_model_id,
    )

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    step = int(meta.get("step", -1))
    optimizer_steps = int(meta.get("optimizer_steps", -2))
    total_steps = int(meta.get("total_steps", -3))
    require(step == optimizer_steps == total_steps == 10_000, "step=optimizer_steps=total_steps=10000")
    require(bool(meta.get("accuracy_contract_valid", False)), "declared Stage-2 accuracy validity")
    require(bool(meta.get("paper_faithful_p0_4", False)), "collect-64/PPO-4 flag")
    require(str(meta.get("deq_map_mode")) in {"parallel", "paper"}, "parallel Eq.3")
    require(meta.get("reward_smoke") is not None, "validated reward smoke")
    require(int(meta.get("rollout_min_depth", 0)) >= 2, "rollout_min_depth>=2")
    require(int(meta.get("rollout_max_depth", 0)) > int(meta.get("rollout_min_depth", 0)), "rollout depth diversity range")
    for key, expected in STAGE2_FINAL_EXPECTED.items():
        actual = meta.get(key)
        if isinstance(expected, float):
            require(_close(actual, expected), f"{key}={expected}")
        else:
            require(actual == expected, f"{key}={expected}")
    _raise("Stage-2 checkpoint is not paper-accuracy eligible", errors)
    return dict(meta)


def validate_stage2_resume_contract(
    checkpoint: Mapping[str, Any],
    *,
    expected_stage1_sha256: str,
    expected_model_id: str,
    expected_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate immutable provenance for an incomplete schema-v3 run."""
    meta, errors = _stage2_common_errors(
        checkpoint,
        expected_stage1_sha256=expected_stage1_sha256,
        expected_model_id=expected_model_id,
    )

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    step = int(meta.get("step", -1))
    optimizer_steps = int(meta.get("optimizer_steps", -2))
    total_steps = int(meta.get("total_steps", -3))
    require(step == optimizer_steps, "step=optimizer_steps")
    require(0 <= step < total_steps, "0<=resume step<total_steps")
    for key, expected in expected_meta.items():
        actual = meta.get(key)
        if isinstance(expected, float):
            require(_close(actual, expected), f"resume {key}={expected}")
        else:
            require(actual == expected, f"resume {key}={expected!r}")
    require(isinstance(checkpoint.get("optimizer_state_dict"), Mapping), "optimizer state")
    rng = checkpoint.get("rng_state") or {}
    require(rng.get("torch") is not None, "torch RNG state")
    _raise("Stage-2 resume provenance mismatch", errors)
    return dict(meta)


__all__ = [
    "CheckpointContractError",
    "STAGE2_FINAL_EXPECTED",
    "model_family_id",
    "stage1_accuracy_contract_is_valid",
    "validate_stage2_final_contract",
    "validate_stage2_resume_contract",
]

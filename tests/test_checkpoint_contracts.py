from __future__ import annotations

import copy

import pytest

from cts.train.checkpoint_contracts import (
    CheckpointContractError,
    stage1_accuracy_contract_is_valid,
    validate_stage2_final_contract,
    validate_stage2_resume_contract,
)


S1_SHA = "1" * 64
MODEL_ID = "google/gemma-4-E4B-it"


def _valid_stage1_checkpoint():
    return {
        "schema_version": 2,
        "step": 5_000,
        "total_steps": 5_000,
        "examples_seen": 10_000,
        "lora": True,
        "backbone_state_dict": {
            "layer.lora_A.weight": object(),
            "layer.lora_B.weight": object(),
        },
        "training_contract": {
            "contract_version": 5,
            "accuracy_contract_valid": True,
            "solve_equilibrium": True,
            "equilibrium_max_iter": 30,
            "equilibrium_tol": 1e-3,
            "deq_map_mode": "parallel",
            "latent_tokens_K": 64,
            "max_target_tokens": 64,
            "max_lm_target_tokens": 256,
            "lambda_wproj": 1.0,
            "stage1_batch_size": 2,
            "stage1_lr": 1e-4,
            "stage1_warmup_steps": 100,
            "stage1_lr_schedule": "cosine",
            "stage1_lambda_lm": 0.1,
            "optimizer": "AdamW",
            "wproj_target_contract": "boxed_final_answer_plus_turn_stop_v2",
            "lm_target_contract": "full_solution_plus_turn_stop_v2",
            "wproj_decode_prompt_contract": "math_concise_boxed_v1",
            "eq3_operator_contract": "explicit_suffix_bf16_v1",
            "eq3_explicit_masks": True,
            "eq3_suffix_cache": True,
            "eq3_suffix_cache_train": True,
            "deterministic_deq_dropout": True,
            "broyden_failure_return_policy": "best_relative_finite_lm_v1",
            "fixed_point_loss_reduction": "squared_l2_sum_v1",
            "stage1_max_grad_norm": 1.0,
            "trainable_parameter_precision": "fp32_master_v1",
            "root_initialization_contract": "fullstack_zero_suffix_v1",
            "no_grad_autocast_cache_contract": "disabled_v1",
            "lora_enabled": True,
            "lora_rank": 8,
            "lora_dropout": 0.05,
            "lora_targets": ["o_proj", "q_proj", "v_proj"],
            "openmath_sha256": "a" * 64,
            "model_id": MODEL_ID,
        },
    }


def test_stage1_contract_pins_best_finite_broyden_failure_policy():
    contract = stage1_accuracy_contract_is_valid(
        _valid_stage1_checkpoint(), expected_model_id=MODEL_ID
    )
    assert contract["broyden_failure_return_policy"] == (
        "best_relative_finite_lm_v1"
    )


def test_stage1_contract_rejects_legacy_last_iterate_failure_policy():
    checkpoint = _valid_stage1_checkpoint()
    checkpoint["training_contract"].pop("broyden_failure_return_policy")
    with pytest.raises(CheckpointContractError, match="Broyden failure policy"):
        stage1_accuracy_contract_is_valid(
            checkpoint, expected_model_id=MODEL_ID
        )


def test_stage1_contract_rejects_mean_reduced_fixed_point_loss():
    checkpoint = _valid_stage1_checkpoint()
    checkpoint["training_contract"]["fixed_point_loss_reduction"] = "mean"
    with pytest.raises(CheckpointContractError, match="squared-L2"):
        stage1_accuracy_contract_is_valid(
            checkpoint, expected_model_id=MODEL_ID
        )


def test_stage1_contract_rejects_detached_autocast_or_random_root_contract():
    checkpoint = _valid_stage1_checkpoint()
    checkpoint["training_contract"]["root_initialization_contract"] = (
        "random_z0"
    )
    checkpoint["training_contract"]["no_grad_autocast_cache_contract"] = (
        "legacy_enabled"
    )
    with pytest.raises(CheckpointContractError, match="root initialization"):
        stage1_accuracy_contract_is_valid(
            checkpoint, expected_model_id=MODEL_ID
        )


def _valid_meta(*, step: int = 10_000):
    return {
        "step": step,
        "optimizer_steps": step,
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
        "rollout_min_depth": 2,
        "rollout_max_depth": 4,
        "latent_tokens_K": 64,
        "branching_W": 3,
        "top_k_modules": 3,
        "broyden_max_iter": 30,
        "broyden_tol_min": 1e-4,
        "broyden_tol_max": 1e-2,
        "tau_mac_budget": 1e14,
        "reward_mode": "answer",
        "wproj_decode_prompt_contract": "math_concise_boxed_v1",
        "eq3_operator_contract": "explicit_suffix_bf16_v1",
        "eq3_explicit_masks": True,
        "eq3_suffix_cache": True,
        "reward_smoke": {"variance": 0.1, "correct": 3},
        "deq_map_mode": "parallel",
        "model_id": MODEL_ID,
        "stage1_sha256": S1_SHA,
        "stage1_contract_version": 5,
        "stage1_accuracy_contract_valid": True,
        "stage2_dataset_sha256": "2" * 64,
        "rollout_seed": 2026,
        "paper_faithful_p0_4": True,
        "accuracy_contract_valid": step == 10_000,
    }


def _valid_checkpoint(*, step: int = 10_000):
    return {
        "schema_version": 3,
        "W": 3,
        "meta_policy_state_dict": {"weight": object()},
        "critic_state_dict": {"weight": object()},
        "value_head_state_dict": {"weight": object()},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "rng_state": {"torch": object(), "cuda": None},
        "training_meta": _valid_meta(step=step),
    }


def test_final_stage2_contract_accepts_complete_checkpoint():
    meta = validate_stage2_final_contract(
        _valid_checkpoint(),
        expected_stage1_sha256=S1_SHA,
        expected_model_id=MODEL_ID,
    )
    assert meta["accuracy_contract_valid"] is True


@pytest.mark.parametrize(
    "key,bad",
    [
        ("optimizer_steps", 9_999),
        ("collect_batch", 32),
        ("ppo_epochs", 2),
        ("rollout_decode_tokens", 1),
        ("latent_tokens_K", 32),
        ("branching_W", 2),
        ("top_k_modules", 2),
        ("broyden_max_iter", 15),
        ("broyden_tol_min", 1e-3),
        ("tau_mac_budget", 1e13),
        ("deq_map_mode", "blend"),
        ("stage2_dataset_sha256", ""),
        ("accuracy_contract_valid", False),
        ("wproj_decode_prompt_contract", "legacy_raw_question"),
        ("eq3_operator_contract", "legacy_full_sequence"),
        ("eq3_suffix_cache", False),
        ("root_initialization_contract", "random_z0"),
        ("root_search_mac_contract", "root_charged_to_search"),
    ],
)
def test_final_stage2_contract_rejects_each_off_paper_field(key, bad):
    ck = _valid_checkpoint()
    ck["training_meta"][key] = bad
    with pytest.raises(CheckpointContractError, match="not paper-accuracy"):
        validate_stage2_final_contract(
            ck,
            expected_stage1_sha256=S1_SHA,
            expected_model_id=MODEL_ID,
        )


def test_final_stage2_contract_rejects_legacy_alias_only_heads():
    ck = _valid_checkpoint()
    ck["meta"] = ck.pop("meta_policy_state_dict")
    with pytest.raises(CheckpointContractError, match="canonical"):
        validate_stage2_final_contract(
            ck,
            expected_stage1_sha256=S1_SHA,
            expected_model_id=MODEL_ID,
        )


def test_resume_contract_checks_immutable_training_environment():
    ck = _valid_checkpoint(step=4_000)
    expected = {
        key: value
        for key, value in ck["training_meta"].items()
        if key not in {
            "step",
            "optimizer_steps",
            "reward_smoke",
            "paper_faithful_p0_4",
            "accuracy_contract_valid",
        }
    }
    validate_stage2_resume_contract(
        ck,
        expected_stage1_sha256=S1_SHA,
        expected_model_id=MODEL_ID,
        expected_meta=expected,
    )
    broken = copy.deepcopy(ck)
    broken["training_meta"]["stage1_sha256"] = "3" * 64
    with pytest.raises(CheckpointContractError, match="resume provenance"):
        validate_stage2_resume_contract(
            broken,
            expected_stage1_sha256=S1_SHA,
            expected_model_id=MODEL_ID,
            expected_meta=expected,
        )

from __future__ import annotations

import pytest

from cts.train.stage2_ppo_train import (
    validate_stage1_accuracy_contract,
    validate_stage2_reward_preflight,
)


def _valid_stage1_checkpoint():
    return {
        "schema_version": 2,
        "step": 5000,
        "total_steps": 5000,
        "examples_seen": 10000,
        "lora": True,
        "backbone_state_dict": {
            "layer.q_proj.lora_A.weight": object(),
            "layer.q_proj.lora_B.weight": object(),
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
            "model_id": "google/gemma-4-E4B-it",
        },
    }


def test_stage1_accuracy_contract_accepts_only_complete_v5_environment():
    contract = validate_stage1_accuracy_contract(
        _valid_stage1_checkpoint(),
        expected_model_id="/models/gemma-4-E4B-it",
    )
    assert contract["accuracy_contract_valid"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ck: ck.update(step=4999),
        lambda ck: ck.update(lora=False),
        lambda ck: ck["training_contract"].update(solve_equilibrium=False),
        lambda ck: ck["training_contract"].update(deq_map_mode="blend"),
        lambda ck: ck["training_contract"].update(max_target_tokens=32),
        lambda ck: ck["training_contract"].update(
            wproj_target_contract="legacy_solution"
        ),
    ],
)
def test_stage1_accuracy_contract_rejects_legacy_or_incomplete_runs(mutation):
    ck = _valid_stage1_checkpoint()
    mutation(ck)
    with pytest.raises(RuntimeError, match="paper-accuracy eligible"):
        validate_stage1_accuracy_contract(
            ck,
            expected_model_id="google/gemma-4-E4B-it",
        )


def test_reward_preflight_rejects_historical_constant_signal():
    with pytest.raises(RuntimeError, match="preflight failed"):
        validate_stage2_reward_preflight(
            [-0.05] * 256,
            correct_count=0,
            min_samples=256,
        )


def test_reward_preflight_rejects_depth_variation_when_all_wrong():
    rewards = [-0.05, -0.10] * 128
    with pytest.raises(RuntimeError, match="correct=0"):
        validate_stage2_reward_preflight(
            rewards,
            correct_count=0,
            min_samples=256,
        )


def test_reward_preflight_passes_mixed_external_outcomes():
    rewards = [-0.05] * 240 + [0.95] * 16
    report = validate_stage2_reward_preflight(
        rewards,
        correct_count=16,
        min_samples=256,
    )
    assert report is not None
    assert report["variance"] > 0.0
    assert report["correct"] == 16.0


def test_reward_preflight_rejects_no_act_depth_signal():
    rewards = [-0.05] * 240 + [0.95] * 16
    with pytest.raises(RuntimeError, match="ACT preflight"):
        validate_stage2_reward_preflight(
            rewards,
            correct_count=16,
            min_samples=256,
            terminal_depths=[4] * 64,
            require_depth_diversity=True,
        )


def test_reward_preflight_accepts_depth_diversity():
    rewards = [-0.05] * 240 + [0.95] * 16
    report = validate_stage2_reward_preflight(
        rewards,
        correct_count=16,
        min_samples=256,
        terminal_depths=[2, 3, 4] * 32,
        require_depth_diversity=True,
    )
    assert report is not None
    assert report["terminal_depth_unique"] == 3.0

#!/usr/bin/env python3
"""Profile one structural real-Gemma CTS-4nu episode without checkpoints.

This runner is diagnostic-only: random controller/head weights preserve the
real tensor shapes and control-flow costs but its answer is never an accuracy
result.  It exists to attribute latency before a newly trained v4 checkpoint
is available.  Final speed acceptance must be repeated with the trained model
and ``CTS_PHASE_PROFILE=0``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.mcts.cts_episode import cts_full_episode
from cts.model.gemma_loader import load_gemma4_e4b
from cts.policy.meta_policy import MetaPolicy
from cts.prompt_contract import build_math_decode_prompt


def _boxed_contract_stats(text: str) -> dict:
    """Report the strict concise-answer box contract without altering text."""
    value = str(text)
    marker = "\\boxed{"
    starts: list[int] = []
    cursor = 0
    while True:
        found = value.find(marker, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + len(marker)

    complete_ends: list[int] = []
    for start in starts:
        depth = 1
        index = start + len(marker)
        while index < len(value):
            char = value[index]
            backslashes = 0
            probe = index - 1
            while probe >= 0 and value[probe] == "\\":
                backslashes += 1
                probe -= 1
            escaped = bool(backslashes % 2)
            if not escaped and char == "{":
                depth += 1
            elif not escaped and char == "}":
                depth -= 1
                if depth == 0:
                    complete_ends.append(index + 1)
                    break
            index += 1

    terminal = bool(complete_ends and not value[complete_ends[-1] :].strip())
    return {
        "marker_count": len(starts),
        "complete_count": len(complete_ends),
        "last_box_terminal": terminal,
        "exactly_one_complete_terminal_box": bool(
            len(starts) == 1 and len(complete_ends) == 1 and terminal
        ),
    }


def _episode(
    prompt: str,
    *,
    decode_prompt: str,
    backbone: GemmaCTSBackbone,
    meta_policy: MetaPolicy,
    critic: NeuroCritic,
    broyden_max_iter: int,
    max_decode_tokens: int,
    tau_budget: float,
) -> dict:
    torch.cuda.synchronize()
    start = perf_counter()
    result = cts_full_episode(
        prompt,
        decode_prompt=decode_prompt,
        backbone=backbone,
        meta_policy=meta_policy,
        critic=critic,
        W=3,
        K=64,
        tau_budget=float(tau_budget),
        broyden_max_iter=int(broyden_max_iter),
        broyden_tol_min=1e-4,
        broyden_tol_max=1e-2,
        top_k=3,
        max_decode_tokens=int(max_decode_tokens),
        routing_mode="sparse",
        device=torch.device("cuda:0"),
        wall_clock_budget_s=None,
        z0_seed=0,
        selection_seed=0,
        require_root_convergence=False,
        batch_siblings=True,
        strict_decode=True,
    )
    torch.cuda.synchronize()
    elapsed_s = perf_counter() - start
    decode_stats = dict(getattr(backbone, "last_decode_stats", {}) or {})
    generated_token_ids = list(decode_stats.pop("generated_token_ids", []))
    return {
        "diagnostic_only_random_heads": True,
        "elapsed_s": elapsed_s,
        "answer": result.answer,
        "generated_token_ids": generated_token_ids,
        "decode_stats": decode_stats,
        "decode_generated_tokens": len(generated_token_ids),
        "decode_eos_terminated": bool(
            decode_stats.get("stop_reason") == "eos"
        ),
        "decode_cap_reached": bool(decode_stats.get("cap_reached", False)),
        "answer_has_boxed": "\\boxed{" in str(result.answer),
        "boxed_stats": _boxed_contract_stats(result.answer),
        "tree_size": result.stats["tree_size"],
        "max_depth": result.stats["max_depth"],
        "sim_count": result.stats["sim_count"],
        "total_iterations": result.total_iterations,
        "total_mac": result.total_mac,
        "root_converged": result.stats["root_converged"],
        "root_iterations": result.stats["root_iterations"],
        "root_residual_norm": result.stats["root_residual_norm"],
        "root_absolute_residual_norm": result.stats[
            "root_absolute_residual_norm"
        ],
        "root_initialization": result.stats["root_initialization"],
        "root_routing_module_ids": result.stats["root_routing_module_ids"],
        "search_logical_mac": result.stats["logical_mac_accumulated"],
        "root_logical_mac": result.stats["root_logical_mac"],
        "total_executed_mac": result.stats["total_executed_mac"],
        "episode_timing_s": result.stats.get("timing_s"),
        "phase_profile": result.stats.get("phase_profile"),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--prompt",
        default=(
            "Solve the following math problem. Put the final answer in "
            "\\boxed{}.\n\nWhat is 2+2?"
        ),
    )
    parser.add_argument(
        "--prompt-jsonl",
        default=None,
        help="Read the first JSONL row and use --prompt-field as the problem",
    )
    parser.add_argument("--prompt-field", default="problem")
    parser.add_argument("--broyden-max-iter", type=int, default=30)
    parser.add_argument("--max-decode-tokens", type=int, default=64)
    parser.add_argument("--tau-budget", type=float, default=1e14)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_jsonl is not None:
        with Path(args.prompt_jsonl).open("r", encoding="utf-8") as handle:
            row = json.loads(handle.readline())
        prompt = str(row[args.prompt_field])
    decode_prompt = build_math_decode_prompt(prompt)

    if not torch.cuda.is_available():
        raise RuntimeError("real CTS profiler requires CUDA")
    # This script is a profiler by definition.  Make direct invocations as
    # complete as the launcher path instead of silently returning no phases.
    os.environ.setdefault("CTS_PHASE_PROFILE", "1")
    os.environ["CTS_DEQ_MAP_MODE"] = "parallel"
    os.environ["CTS_EQ3_EXPLICIT_MASKS"] = "1"
    os.environ["CTS_EQ3_SUFFIX_CACHE"] = "1"
    os.environ["CTS_EQ3_SUFFIX_CACHE_TRAIN"] = "1"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model, tokenizer = load_gemma4_e4b(
        model_id=args.model_dir,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    backbone = GemmaCTSBackbone(model, tokenizer).eval()
    meta_policy = MetaPolicy(
        text_dim=backbone.hidden_size, hidden=256, W=3
    ).to("cuda:0").eval()
    critic = NeuroCritic(z_dim=backbone.hidden_size).to("cuda:0").eval()

    if args.warmup:
        _episode(
            prompt,
            decode_prompt=decode_prompt,
            backbone=backbone,
            meta_policy=meta_policy,
            critic=critic,
            broyden_max_iter=2,
            max_decode_tokens=2,
            tau_budget=1e12,
        )
    torch.cuda.reset_peak_memory_stats()
    report = _episode(
        prompt,
        decode_prompt=decode_prompt,
        backbone=backbone,
        meta_policy=meta_policy,
        critic=critic,
        broyden_max_iter=args.broyden_max_iter,
        max_decode_tokens=args.max_decode_tokens,
        tau_budget=args.tau_budget,
    )
    print("CTS_PROFILE_JSON=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

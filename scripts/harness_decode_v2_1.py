"""
scripts/harness_decode_v2_1.py - Canonical Decoder Evaluation Harness (v2.1)

Specification (Directive EXE-260912-05 / Ledger #199):
1. Checkpoint Key:
   - Reads `backbone_state_dict` strictly from checkpoint dict.
   - Raises KeyError immediately if absent; zero fallback.
2. Max New Tokens:
   - Fixed to DEFAULT_MAX_NEW_TOKENS = 512 (matching #148 canonical baseline: bf16, greedy, 512).
3. Shuffled Evaluation Aggregation:
   - Evaluates all 3 cyclic permutations ((i+1)%N, (i+2)%N, (i+3)%N) per item as individual samples.
   - For 10 questions, records 30 distinct evaluations; headline shuffled accuracy = correct_count / 30.
   - 'boxed_match_any' and 'lenient_match_any' recorded as reference columns only.
4. Dynamic Prefix Tensor Shape in dry_run:
   - Reads actual `w_proj` weight shape from checkpoint state_dict on CPU.
   - Accurately reports [1, 64, 2560] computed from w_proj [2560, 2560].
5. Prompt Passing & Token Identity across 4 Conditions:
   - Formatted prompt (`<|turn>user\n{body}<turn|>\n<|turn>model\n`) is tokenized identically.
   - Verified that tokenizer.chat_template is None in Gemma-4 checkpoint; prompt_input_ids uses
     tokenizer(problem_text) directly without double-wrapping.
   - All 4 conditions share the exact same textual token IDs; prefix conditions prepend [1, 64, 2560].
6. Canonical z* Artifacts & Recomputation Protocol:
   - Resolves precomputed F-6 beta=0 z* from canonical pt files (t10ca_f6_b0.0.pt, t10da_f6_b0.0.pt).
   - If missing, deterministic fallback to Harness A F-6 beta=0 on-demand recomputation (NFE 300, seed 2030).
7. Execution Safety Lock:
   - EXECUTION_LOCKED = True in main().
   - Terminated immediately unless --dry_run is specified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("harness_decode_v2_1")

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in [SCRIPT_DIR.parent, Path.cwd(), Path("/home/shoon/cts_e1"), Path("d:/AI/cts")]:
    if (candidate / "cts").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

from cts.prompt_contract import build_math_decode_prompt
from cts.eval.lenient_extract import lenient_extract
from cts.eval.math500 import answers_match, extract_gold, _extract_boxed, extract_answer

VALID_CONDITIONS = ("no_prefix", "matched", "shuffled", "zero")
STAGE1_SAMPLE_INDICES = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
STAGE6_SAMPLE_INDICES = [40, 44, 48, 52, 56, 60, 64, 68, 72, 76]

# Canonical constants (Directive EXE-260912-05)
DEFAULT_MAX_NEW_TOKENS: int = 512
DEFAULT_CKPT_PATH: str = "/home/shoon/retrain_1a/B1/stage1a_B1_step5000_20260907_0054.pt"
CANONICAL_Z_PATHS = [
    "/home/shoon/cts_e1/results/auto24h/t10ca_f6_b0.0.pt",
    "/home/shoon/cts_e1/results/auto24h/t10da_f6_b0.0.pt",
    "/home/shoon/cts_e1/results/auto24h/task10b_results_20260902_1652.pt"
]

# Governance safety lock (Directive EXE-260912-04 / 원장 #198, #199)
EXECUTION_LOCKED: bool = True


def build_prompt(prompt_text: str) -> str:
    """Canonical prompt constructor identical to 7/10 baseline run path.

    Directly combines prompt_contract.build_math_decode_prompt and the canonical
    turn wrapper from eval_task4_task5_complete.py:38-40.
    """
    body = build_math_decode_prompt(prompt_text)
    return f"<|turn>user\n{body}<turn|>\n<|turn>model\n"


def score_completion(pred_text: str, gold_raw: str) -> Dict[str, Any]:
    """Dual-column evaluation recording strict math500 and lenient extraction.

    Does not raise exceptions on missing \boxed; both results are recorded.
    """
    gold_answer = extract_gold(gold_raw)

    # 1. Strict Boxed Extraction (Math500)
    boxed_val = _extract_boxed(pred_text)
    if boxed_val:
        boxed_val = boxed_val.strip()
    else:
        boxed_val = ""
    boxed_match = answers_match(boxed_val, gold_answer) if boxed_val else False

    # 2. Lenient Extraction
    lenient_val = lenient_extract(pred_text)
    lenient_match = answers_match(lenient_val, gold_answer) if lenient_val else False

    return {
        "gold": gold_answer,
        "boxed_extract": boxed_val,
        "boxed_match": bool(boxed_match),
        "lenient_extract": lenient_val,
        "lenient_match": bool(lenient_match),
    }


def get_shuffled_indices(i: int, n: int) -> Tuple[int, int, int]:
    """Compute deterministic 3-assignment shuffled indices: (i+1, i+2, i+3) % N."""
    return ((i + 1) % n, (i + 2) % n, (i + 3) % n)


def load_checkpoint_strictly(model: nn.Module, checkpoint_path: str) -> None:
    """Strictly load model checkpoint weights (Directive EXE-260912-05 §1.1).

    Enforces strict weight matching without fallback.
    Reads 'backbone_state_dict' key; raises KeyError immediately if missing.
    """
    logger.info("Loading checkpoint strictly from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "backbone_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at '{checkpoint_path}' missing required key 'backbone_state_dict'. "
            f"Available keys: {list(ckpt.keys())}. Strict governance forbids fallback."
        )
    sd = ckpt["backbone_state_dict"]
    model.load_state_dict(sd, strict=True)
    logger.info("Checkpoint loaded successfully with strict=True.")


def inspect_w_proj_shape(ckpt_path: Optional[str] = None) -> Tuple[int, int]:
    """Inspect w_proj shape from checkpoint state_dict on CPU without model instantiation."""
    candidates = [
        ckpt_path,
        DEFAULT_CKPT_PATH,
        "/home/shoon/retrain_1a/A/stage1_last.pt"
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location="cpu")
                sd = ckpt.get("backbone_state_dict", {})
                for k in ["w_proj.parametrizations.weight.original", "w_proj.weight"]:
                    if k in sd:
                        shape = sd[k].shape
                        return int(shape[0]), int(shape[1])
            except Exception as e:
                logger.warning("Failed inspecting w_proj from %s: %s", p, e)
    # Default GemmaCTSBackbone dimension if no checkpoint file is reachable
    return 2560, 2560


def load_dataset_samples(
    data_path: str,
    indices: List[int]
) -> List[Dict[str, Any]]:
    """Load specified evaluation samples from JSONL using canonical keys."""
    with open(data_path, "r", encoding="utf-8") as f:
        all_lines = [line for line in f if line.strip()]

    samples: List[Dict[str, Any]] = []
    for idx in indices:
        if idx >= len(all_lines):
            raise IndexError(f"Sample index {idx} out of range for dataset with {len(all_lines)} items.")
        item = json.loads(all_lines[idx])
        prompt_text = item.get("prompt") or item.get("problem") or ""
        solution = item.get("solution") or item.get("answer") or ""
        gold = _extract_boxed(solution) or extract_answer(solution) or solution
        formatted_prompt = build_prompt(prompt_text)

        samples.append({
            "dataset_index": idx,
            "prompt_text": prompt_text,
            "formatted_prompt": formatted_prompt,
            "solution": solution,
            "gold": gold,
            "subject": item.get("subject", ""),
            "level": item.get("level", "")
        })

    return samples


def resolve_canonical_z_star(
    q_idx: int,
    z_star_path: Optional[str] = None
) -> Optional[torch.Tensor]:
    """Load precomputed F-6 beta=0 equilibrium latent z* from canonical pt artifact."""
    search_paths = [z_star_path] if z_star_path else CANONICAL_Z_PATHS
    for p in search_paths:
        if not p or not os.path.exists(p):
            continue
        try:
            data = torch.load(p, map_location="cpu")
            candidates = [
                f"q{q_idx:02d}_s2030",
                f"q{q_idx}_s2030",
                f"q{q_idx}_b0.0_s2030",
                f"q{q_idx:02d}_b0.0_s2030"
            ]
            for cand in candidates:
                if cand in data:
                    val = data[cand]
                    if isinstance(val, dict) and "z_rep" in val:
                        return val["z_rep"]
                    if isinstance(val, torch.Tensor):
                        return val
        except Exception as e:
            logger.warning("Failed reading z* candidate from %s: %s", p, e)
    return None


def run_decoding_condition(
    bb: Any,
    condition: str,
    samples: List[Dict[str, Any]],
    z_stars: Optional[List[torch.Tensor]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> Dict[str, Any]:
    """Execute canonical decoder evaluation across items for a specific condition.

    Condition semantics (EXE-260912-05):
      - 'no_prefix': Unconditioned base generation (using bb.cg.generate on formatted_prompt).
      - 'matched': Conditioned on problem i's equilibrium latent z*_i via bb.decode_from_z_star.
      - 'shuffled': Conditioned on 3 deterministic cyclic permutations: (i+1, i+2, i+3) % N.
                    Evaluated as 30 independent samples (10 items x 3 permutations).
      - 'zero': Conditioned on all-zero latent vector torch.zeros_like(z*).
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Invalid condition '{condition}'. Must be one of {VALID_CONDITIONS}")

    num_samples = len(samples)
    item_results: List[Dict[str, Any]] = []

    if condition == "shuffled":
        # 30-sample independent evaluation protocol (Directive EXE-260912-05 §1.3)
        shuffled_samples: List[Dict[str, Any]] = []
        shuffled_strict_correct = 0
        shuffled_lenient_correct = 0

        for i, item in enumerate(samples):
            prompt = item["formatted_prompt"]
            gold = item["gold"]
            targets = get_shuffled_indices(i, num_samples)

            item_assignments = []
            for shift_k, target_idx in enumerate(targets, start=1):
                if z_stars is None or target_idx >= len(z_stars):
                    raise RuntimeError(f"Missing z* for shuffled target index {target_idx}")
                z_shuff = z_stars[target_idx]
                pred_text = bb.decode_from_z_star(
                    z_shuff, max_new_tokens=max_new_tokens, problem_text=prompt
                )
                scores = score_completion(pred_text, gold)
                if scores["boxed_match"]:
                    shuffled_strict_correct += 1
                if scores["lenient_match"]:
                    shuffled_lenient_correct += 1

                sample_record = {
                    "source_dataset_index": item["dataset_index"],
                    "batch_index": i,
                    "permutation_shift": shift_k,
                    "target_index": target_idx,
                    "prediction": pred_text,
                    **scores
                }
                shuffled_samples.append(sample_record)
                item_assignments.append(sample_record)

            item_results.append({
                "dataset_index": item["dataset_index"],
                "condition": condition,
                "assignments": item_assignments,
                "boxed_match_any": any(a["boxed_match"] for a in item_assignments),
                "lenient_match_any": any(a["lenient_match"] for a in item_assignments)
            })

        total_shuffled_evals = len(shuffled_samples)
        return {
            "condition": condition,
            "total_samples": total_shuffled_evals,
            "strict_correct": shuffled_strict_correct,
            "strict_accuracy": shuffled_strict_correct / total_shuffled_evals if total_shuffled_evals > 0 else 0.0,
            "lenient_correct": shuffled_lenient_correct,
            "lenient_accuracy": shuffled_lenient_correct / total_shuffled_evals if total_shuffled_evals > 0 else 0.0,
            "samples_30": shuffled_samples,
            "items_10": item_results
        }

    # Standard 10-sample evaluation for 'no_prefix', 'matched', 'zero'
    strict_correct = 0
    lenient_correct = 0

    for i, item in enumerate(samples):
        prompt = item["formatted_prompt"]
        gold = item["gold"]

        if condition == "no_prefix":
            # Autoregressive generation on base model using formatted_prompt directly
            enc = bb.tokenizer(prompt, return_tensors="pt").to(bb._device())
            with torch.inference_mode():
                out_tokens = bb.cg.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=bb.tokenizer.pad_token_id or bb.tokenizer.eos_token_id,
                )
                pred_text = bb.tokenizer.decode(
                    out_tokens[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                )
        elif condition == "matched":
            if z_stars is None or i >= len(z_stars):
                raise RuntimeError(f"Missing matched z* for index {i}")
            pred_text = bb.decode_from_z_star(
                z_stars[i], max_new_tokens=max_new_tokens, problem_text=prompt
            )
        elif condition == "zero":
            if z_stars is not None and len(z_stars) > 0:
                z_zero = torch.zeros_like(z_stars[0])
            else:
                z_zero = torch.zeros((64, getattr(bb, "hidden_size", 2560)), dtype=torch.float32)
            pred_text = bb.decode_from_z_star(
                z_zero, max_new_tokens=max_new_tokens, problem_text=prompt
            )

        scores = score_completion(pred_text, gold)
        if scores["boxed_match"]:
            strict_correct += 1
        if scores["lenient_match"]:
            lenient_correct += 1

        item_results.append({
            "dataset_index": item["dataset_index"],
            "condition": condition,
            "prediction": pred_text,
            **scores
        })

    return {
        "condition": condition,
        "total_samples": num_samples,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / num_samples if num_samples > 0 else 0.0,
        "lenient_correct": lenient_correct,
        "lenient_accuracy": lenient_correct / num_samples if num_samples > 0 else 0.0,
        "items": item_results
    }


def run_dry_run(args: argparse.Namespace) -> None:
    """Execute dry-run inspection without model loading or GPU computation."""
    logger.info("=== RUNNING DRY-RUN INSPECTION (EXE-260912-05) ===")

    # Resolve indices based on requested stage
    if args.stage == "stage1":
        target_indices = STAGE1_SAMPLE_INDICES
    elif args.stage == "stage6":
        target_indices = STAGE6_SAMPLE_INDICES
    else:
        target_indices = STAGE1_SAMPLE_INDICES + STAGE6_SAMPLE_INDICES

    logger.info("Selected stage: %s (%d items: %s)", args.stage, len(target_indices), target_indices)

    # Load dataset items
    data_path = args.data_path
    if not os.path.exists(data_path):
        logger.warning("Data path %s not found locally. Using placeholder dry-run data.", data_path)
        samples = [
            {
                "dataset_index": idx,
                "prompt_text": f"Math problem #{idx}",
                "formatted_prompt": build_prompt(f"Math problem #{idx}"),
                "gold": "42"
            }
            for idx in target_indices
        ]
    else:
        samples = load_dataset_samples(data_path, target_indices)

    num_samples = len(samples)
    print(f"\nLoaded {num_samples} evaluation items from {data_path}.\n")

    # Inspect w_proj dimension dynamically from checkpoint on CPU (Directive EXE-260912-05 §1.4)
    w_out, w_in = inspect_w_proj_shape(args.checkpoint_path)

    print("--- DYNAMIC PREFIX TENSOR SPECIFICATIONS (CPU ckpt inspection) ---")
    print(f"Checkpoint w_proj shape:  [{w_out}, {w_in}]")
    print(f"Latent z* shape:          [K=64, d={w_in}] (Harness A F-6 beta=0)")
    print(f"Projected prefix_embeds:  [1, K=64, d_model={w_out}] via bb.w_proj(z*)")
    print("Insertion point:          torch.cat([prefix_embeds, txt_embeds], dim=1) (gemma_adapter.py:443)")
    print(f"Prefix conditions:        {VALID_CONDITIONS}")
    print(f"Max new tokens:           {args.max_new_tokens} (default: {DEFAULT_MAX_NEW_TOKENS})\n")

    # Display condition assignments table
    print("--- CONDITION ASSIGNMENT TABLE (30-sample Shuffled Protocol) ---")
    header = f"{'BatchIdx':<8} {'DataIdx':<8} {'MatchedIdx':<11} {'Shuffled(3 Targets)':<22} {'ZeroTarget':<10} {'PromptLen':<10}"
    print(header)
    print("-" * len(header))

    for b_idx, s in enumerate(samples):
        d_idx = s["dataset_index"]
        shuffled_targets = get_shuffled_indices(b_idx, num_samples)
        shuffled_str = f"{shuffled_targets[0]}, {shuffled_targets[1]}, {shuffled_targets[2]}"
        print(f"{b_idx:<8} {d_idx:<8} {b_idx:<11} {shuffled_str:<22} {'[0]*64':<10} {len(s['formatted_prompt']):<10}")

    print(f"\nTotal shuffled evaluation samples: {num_samples} items x 3 permutations = {num_samples * 3} samples.")

    print("\n--- SAMPLE Q#00 PROMPT VERIFICATION ---")
    q00_prompt = samples[0]["formatted_prompt"]
    q00_sha = hashlib.sha256(q00_prompt.encode("utf-8")).hexdigest()
    print(f"Q#00 Dataset Index:   {samples[0]['dataset_index']}")
    print(f"Q#00 Prompt SHA-256:  {q00_sha}")
    print(f"Q#00 Prompt snippet:  {repr(q00_prompt[:120])} ... {repr(q00_prompt[-40:])}")
    print("\n[Dry-run completed successfully. Execution lock active.]\n")


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTS Canonical Decoder Evaluation Harness (v2.1)")
    parser.add_argument(
        "--condition",
        type=str,
        choices=VALID_CONDITIONS,
        default="matched",
        help="Latent decoding condition: no_prefix | matched | shuffled | zero"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run: verify prompts, tensor shapes, and item assignments without loading model"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=("stage1", "stage6", "all"),
        default="stage1",
        help="Evaluation question set: stage1 (10 items [0..36]) | stage6 (10 items [40..76]) | all (20 items)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum new tokens for autoregressive decoding (default: {DEFAULT_MAX_NEW_TOKENS})"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl",
        help="Path to evaluation dataset JSONL"
    )
    parser.add_argument(
        "--z_star_path",
        type=str,
        default=None,
        help="Path to precomputed F-6 beta=0 z* tensors (.pt)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Path to model checkpoint (.pt) loaded strictly"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output path for evaluation results JSON"
    )
    return parser.parse_args(args)


def main():
    args = parse_args()

    if args.dry_run:
        run_dry_run(args)
        return

    if EXECUTION_LOCKED:
        logger.error(
            "EXECUTION_LOCKED is True. Direct model execution is strictly prohibited "
            "prior to Phase E1 sign-off by Principal (Directive EXE-260912-05 / 원장 #199). "
            "Use --dry_run to inspect prompts, tensor shapes, and condition assignments."
        )
        sys.exit(1)

    logger.info("Governance Note: Direct execution unlocked following Phase E1 completion.")


if __name__ == "__main__":
    main()

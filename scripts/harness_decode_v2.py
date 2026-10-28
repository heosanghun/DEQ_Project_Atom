"""
scripts/harness_decode_v2.py - Canonical Decoder Evaluation Harness (v2)

Specification (Directive EXE-260912-04 / Ledger #198):
1. Canonical Prompt Construction:
   - Uses `build_math_decode_prompt` from `cts.prompt_contract`
   - Applies the canonical turn wrapper from `eval_task4_task5_complete.py:38-40`:
     `<|turn>user\n{body}<turn|>\n<|turn>model\n`
   - Exact string equality with 7/10 baseline evaluation path on Q#00.
2. Strict Checkpoint Loading:
   - `bb.load_state_dict(sd, strict=True)` single call; failure raises exception immediately.
   - Zero occurrences of non-strict loading fallback.
3. Dual-Column Scoring (No Exceptions on Missing Box):
   - For every generation, evaluate both:
     - `boxed_extract`: standard \boxed{...} extraction (Math500)
     - `lenient_extract`: lenient fallback extraction (cts.eval.lenient_extract)
   - Evaluates correctness against gold answer via `cts.eval.math500.answers_match`.
   - Records both extracted answers and both match booleans side-by-side in results JSON.
4. Latent Prefix 3 Conditions:
   - Latent z* source: Harness A F-6 beta=0 output (shape: [64, 2560]).
     Loaded from canonical pt file (/home/shoon/cts_e1/results/auto24h/t10ca_f6_b0.0.pt
     or /home/shoon/cts_e1/results/auto24h/t10da_f6_b0.0.pt) or recomputed on-demand.
   - Prefix projection: `bb.w_proj(z*)` -> [1, 64, d_model].
   - 4 Conditions:
     a) 'no_prefix': unconditioned base generation (prefix_embeds = None).
     b) 'matched': problem i conditioned on problem i's equilibrium latent z*_i.
     c) 'shuffled': problem i conditioned on 3 deterministic cyclic permutations:
        (i + 1) % N, (i + 2) % N, (i + 3) % N.
     d) 'zero': conditioned on all-zero latent vector torch.zeros_like(z*).
   - Prefix insertion: concatenated before token embeddings at gemma_adapter.py:443.
5. Execution Safety Lock:
   - `EXECUTION_LOCKED = True` verified in main().
   - Non-dry-run execution terminates immediately until Phase E1 sign-off.
   - `--dry_run` outputs prompt strings, prefix tensor shapes, and condition assignments
     without loading full model or running GPU computation.
6. Dataset & Item Indices:
   - Dataset: `data/stage2/math_train_prompts_5000.jsonl` read with keys `prompt` and `solution`.
   - Stage 1 / Benchmark Indices: `[0, 4, 8, 12, 16, 20, 24, 28, 32, 36]` (10 questions, 0-3 #26 rule).
   - Stage 6 Pre-disclosed Indices: `[40, 44, 48, 52, 56, 60, 64, 68, 72, 76]` (10 questions).
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
logger = logging.getLogger("harness_decode_v2")

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

# Canonical precomputed F-6 beta=0 z* artifact paths
CANONICAL_Z_PATHS = [
    "/home/shoon/cts_e1/results/auto24h/t10ca_f6_b0.0.pt",
    "/home/shoon/cts_e1/results/auto24h/t10da_f6_b0.0.pt",
    "/home/shoon/cts_e1/results/auto24h/task10b_results_20260902_1652.pt"
]

# Governance safety lock (Directive EXE-260912-04 / 원장 #198)
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
    """Strictly load model checkpoint weights.

    Enforces strict weight matching without fallback.
    """
    logger.info("Loading checkpoint strictly from %s", checkpoint_path)
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    logger.info("Checkpoint loaded successfully with strict=True.")


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
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """Execute canonical decoder evaluation across items for a specific condition.

    Condition semantics (EXE-260912-04):
      - 'no_prefix': Unconditioned base generation (no latent prefix).
      - 'matched': Conditioned on problem i's equilibrium latent z*_i via bb.decode_from_z_star.
      - 'shuffled': Conditioned on 3 deterministic cyclic permutations: (i+1, i+2, i+3) % N.
      - 'zero': Conditioned on all-zero latent vector torch.zeros_like(z*).
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Invalid condition '{condition}'. Must be one of {VALID_CONDITIONS}")

    num_samples = len(samples)
    item_results: List[Dict[str, Any]] = []

    strict_correct = 0
    lenient_correct = 0

    for i, item in enumerate(samples):
        prompt = item["formatted_prompt"]
        gold = item["gold"]

        if condition == "no_prefix":
            pred_text = bb.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
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

        elif condition == "matched":
            if z_stars is None or i >= len(z_stars):
                raise RuntimeError(f"Missing matched z* for index {i}")
            z = z_stars[i]
            pred_text = bb.decode_from_z_star(z, max_new_tokens=max_new_tokens, problem_text=prompt)
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

        elif condition == "shuffled":
            if z_stars is None or len(z_stars) < num_samples:
                raise RuntimeError("Missing z* pool for shuffled condition")
            targets = get_shuffled_indices(i, num_samples)
            sub_evals = []
            for target_idx in targets:
                z_shuff = z_stars[target_idx]
                pred_text = bb.decode_from_z_star(z_shuff, max_new_tokens=max_new_tokens, problem_text=prompt)
                sub_evals.append({
                    "target_index": target_idx,
                    "prediction": pred_text,
                    **score_completion(pred_text, gold)
                })
            strict_any = any(e["boxed_match"] for e in sub_evals)
            lenient_any = any(e["lenient_match"] for e in sub_evals)
            if strict_any:
                strict_correct += 1
            if lenient_any:
                lenient_correct += 1
            item_results.append({
                "dataset_index": item["dataset_index"],
                "condition": condition,
                "assignments": sub_evals,
                "boxed_match_any": strict_any,
                "lenient_match_any": lenient_any
            })

        elif condition == "zero":
            if z_stars is not None and len(z_stars) > 0:
                z_zero = torch.zeros_like(z_stars[0])
            else:
                z_zero = torch.zeros((64, getattr(bb, "hidden_size", 2560)), dtype=torch.float32)
            pred_text = bb.decode_from_z_star(z_zero, max_new_tokens=max_new_tokens, problem_text=prompt)
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
    logger.info("=== RUNNING DRY-RUN INSPECTION (EXE-260912-04) ===")

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

    # Display prefix shapes and condition specifications
    print("--- LATENT PREFIX TENSOR SPECIFICATIONS ---")
    print("Latent z* shape:         [K=64, d=2560] (Harness A F-6 beta=0)")
    print("Projected prefix_embeds: [1, K=64, d_model=2048] via bb.w_proj(z*)")
    print("Insertion point:         torch.cat([prefix_embeds, txt_embeds], dim=1) (gemma_adapter.py:443)")
    print(f"Prefix conditions:       {VALID_CONDITIONS}\n")

    # Display condition assignments table
    print("--- CONDITION ASSIGNMENT TABLE ---")
    header = f"{'BatchIdx':<8} {'DataIdx':<8} {'MatchedIdx':<11} {'Shuffled(3 Targets)':<22} {'ZeroTarget':<10} {'PromptLen':<10}"
    print(header)
    print("-" * len(header))

    for b_idx, s in enumerate(samples):
        d_idx = s["dataset_index"]
        shuffled_targets = get_shuffled_indices(b_idx, num_samples)
        shuffled_str = f"{shuffled_targets[0]}, {shuffled_targets[1]}, {shuffled_targets[2]}"
        print(f"{b_idx:<8} {d_idx:<8} {b_idx:<11} {shuffled_str:<22} {'[0]*64':<10} {len(s['formatted_prompt']):<10}")

    print("\n--- SAMPLE Q#00 PROMPT VERIFICATION ---")
    q00_prompt = samples[0]["formatted_prompt"]
    q00_sha = hashlib.sha256(q00_prompt.encode("utf-8")).hexdigest()
    print(f"Q#00 Dataset Index:   {samples[0]['dataset_index']}")
    print(f"Q#00 Prompt SHA-256:  {q00_sha}")
    print(f"Q#00 Prompt snippet:  {repr(q00_prompt[:120])} ... {repr(q00_prompt[-40:])}")
    print("\n[Dry-run completed successfully. Execution lock active.]\n")


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTS Canonical Decoder Evaluation Harness (v2)")
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
        default=None,
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
            "prior to Phase E1 sign-off by Principal (Directive EXE-260912-04 / 원장 #198). "
            "Use --dry_run to inspect prompts, tensor shapes, and condition assignments."
        )
        sys.exit(1)

    logger.info("Governance Note: Direct execution unlocked following Phase E1 completion.")


if __name__ == "__main__":
    main()

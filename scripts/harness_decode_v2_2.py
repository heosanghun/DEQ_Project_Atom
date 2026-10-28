"""
scripts/harness_decode_v2_2.py - Canonical Decoder Evaluation Harness (v2.2 Final)

Specification (Directive EXE-260912-06 / Ledger #200):
1. no_prefix Generation Fidelity:
   - Identical generation kwargs and decoding logic to canonical 7/10 baseline run
     (eval_task4_task5_complete.py:69-77):
     `m.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=t.pad_token_id or t.eos_token_id)`
     `resp = t.decode(out_tokens[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)`
2. Token ID Equivalence Proof:
   - Formatted prompt `<|turn>user\n{body}<turn|>\n<|turn>model\n` produces 100% byte-for-byte identical
     token sequence [1, 151] between no_prefix and prefix conditions (verified via token-eq-q00.txt).
   - tokenizer.chat_template is None; prompt_input_ids uses tokenizer(prompt) directly without double-wrapping.
3. Strict Checkpoint Loading & Architecture:
   - Single call bb.load_state_dict(sd, strict=True) reading 'backbone_state_dict'.
   - Zero occurrences of non-strict loading fallback.
4. Shuffled 30-Sample Independent Protocol:
   - Evaluates all 3 cyclic permutations ((i+1)%N, (i+2)%N, (i+3)%N) as individual samples.
   - Headline shuffled accuracy = correct_count / 30.
5. End-to-End Pipeline & Bootstrap:
   - run_full(args) performs: model instantiation -> strict load -> z* collection (raises FileNotFoundError
     if missing; zero in-harness recomputation) -> 4 conditions evaluation -> sample-level JSON
     -> 1,000 bootstrap resamples (item-level, 95% CI of matched - shuffled difference) -> summary JSON.
   - Guarded by EXECUTION_LOCKED = True prior to run_full().
6. Canonical z* Artifact Hierarchy:
   - Resolves from HS-8 reproduction artifact `/home/shoon/cts_e1/results/auto24h/zstar_f6_b0_20q_*.pt` first,
     then canonical F-6 beta=0 files.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("harness_decode_v2_2")

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

# Canonical constants (Directive EXE-260912-06)
DEFAULT_MAX_NEW_TOKENS: int = 512
DEFAULT_BASE_MODEL_PATH: str = "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
DEFAULT_CKPT_PATH: str = "/home/shoon/retrain_1a/B1/stage1a_B1_step5000_20260907_0054.pt"

# Canonical z* artifact search hierarchy (Directive EXE-260912-06 §1.4)
CANONICAL_Z_SEARCH_PATTERNS = [
    "/home/shoon/cts_e1/results/auto24h/zstar_f6_b0_20q_*.pt",
    "/home/shoon/cts_e1/results/auto24h/t10ca_f6_b0.0.pt",
    "/home/shoon/cts_e1/results/auto24h/t10da_f6_b0.0.pt",
    "/home/shoon/cts_e1/results/auto24h/task10b_results_20260902_1652.pt"
]

# Governance safety lock (Directive EXE-260912-06 / 원장 #200)
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
    """Strictly load model checkpoint weights (Directive EXE-260912-05 / EXE-260912-06).

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
    """Load precomputed F-6 beta=0 equilibrium latent z* from canonical pt artifact hierarchy."""
    candidate_files = []
    if z_star_path:
        candidate_files.append(z_star_path)
    for pattern in CANONICAL_Z_SEARCH_PATTERNS:
        matched = sorted(glob.glob(pattern))
        candidate_files.extend(matched)

    candidates_keys = [
        f"q{q_idx:02d}_s2030",
        f"q{q_idx}_s2030",
        f"q{q_idx}_b0.0_s2030",
        f"q{q_idx:02d}_b0.0_s2030"
    ]

    for p in candidate_files:
        if not p or not os.path.exists(p):
            continue
        try:
            data = torch.load(p, map_location="cpu")
            for cand in candidates_keys:
                if cand in data:
                    val = data[cand]
                    if isinstance(val, dict) and "z_rep" in val:
                        return val["z_rep"]
                    if isinstance(val, torch.Tensor):
                        return val
        except Exception as e:
            logger.warning("Failed reading z* candidate from %s: %s", p, e)
    return None


def generate_no_prefix(
    model: Any,
    tokenizer: Any,
    formatted_prompt: str,
    max_new_tokens: int,
    device: torch.device
) -> str:
    """Exact replica of canonical baseline 7/10 generation call (eval_task4_task5_complete.py:69-77)."""
    enc = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out_tokens = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        resp = tokenizer.decode(
            out_tokens[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
    return resp


def run_decoding_condition(
    bb: Any,
    condition: str,
    samples: List[Dict[str, Any]],
    z_stars: Optional[List[torch.Tensor]] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> Dict[str, Any]:
    """Execute canonical decoder evaluation across items for a specific condition."""
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Invalid condition '{condition}'. Must be one of {VALID_CONDITIONS}")

    num_samples = len(samples)
    item_results: List[Dict[str, Any]] = []
    device = bb._device()

    if condition == "shuffled":
        # 30-sample independent evaluation protocol (Directive EXE-260912-05 / EXE-260912-06)
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
            pred_text = generate_no_prefix(
                model=bb.cg,
                tokenizer=bb.tokenizer,
                formatted_prompt=prompt,
                max_new_tokens=max_new_tokens,
                device=device
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


def compute_bootstrap_ci(
    matched_results: Dict[str, Any],
    shuffled_results: Dict[str, Any],
    num_bootstraps: int = 1000,
    seed: int = 2026
) -> Dict[str, Any]:
    """Compute 1,000-iteration item-level bootstrap for (matched - shuffled) difference."""
    rng = np.random.default_rng(seed)

    matched_items = matched_results["items"]
    shuffled_items = shuffled_results["items_10"]
    n_items = len(matched_items)

    diffs_strict = []
    diffs_lenient = []

    for _ in range(num_bootstraps):
        resample_idx = rng.choice(n_items, size=n_items, replace=True)
        m_strict = np.mean([1.0 if matched_items[j]["boxed_match"] else 0.0 for j in resample_idx])
        m_lenient = np.mean([1.0 if matched_items[j]["lenient_match"] else 0.0 for j in resample_idx])

        s_strict_vals = []
        s_lenient_vals = []
        for j in resample_idx:
            for assignment in shuffled_items[j]["assignments"]:
                s_strict_vals.append(1.0 if assignment["boxed_match"] else 0.0)
                s_lenient_vals.append(1.0 if assignment["lenient_match"] else 0.0)

        s_strict = np.mean(s_strict_vals)
        s_lenient = np.mean(s_lenient_vals)

        diffs_strict.append(float(m_strict - s_strict))
        diffs_lenient.append(float(m_lenient - s_lenient))

    ci_strict_low, ci_strict_high = np.percentile(diffs_strict, [2.5, 97.5])
    ci_lenient_low, ci_lenient_high = np.percentile(diffs_lenient, [2.5, 97.5])

    return {
        "num_bootstraps": num_bootstraps,
        "strict": {
            "mean_diff": float(np.mean(diffs_strict)),
            "ci_95": [float(ci_strict_low), float(ci_strict_high)],
            "preregistered_margin_met": bool(np.mean(diffs_strict) >= 0.05)
        },
        "lenient": {
            "mean_diff": float(np.mean(diffs_lenient)),
            "ci_95": [float(ci_lenient_low), float(ci_lenient_high)],
            "preregistered_margin_met": bool(np.mean(diffs_lenient) >= 0.05)
        }
    }


def run_full(args: argparse.Namespace) -> None:
    """Execute end-to-end evaluation pipeline following Phase E1 sign-off."""
    logger.info("=== STARTING FULL CANONICAL EVALUATION PIPELINE (v2.2) ===")

    # 1. Resolve Target Indices
    if args.stage == "stage1":
        target_indices = STAGE1_SAMPLE_INDICES
    elif args.stage == "stage6":
        target_indices = STAGE6_SAMPLE_INDICES
    else:
        target_indices = STAGE1_SAMPLE_INDICES + STAGE6_SAMPLE_INDICES

    # 2. Load Dataset
    samples = load_dataset_samples(args.data_path, target_indices)
    logger.info("Loaded %d items for evaluation.", len(samples))

    # 3. Instantiate Base Model and Adapter
    from cts.model.gemma_loader import load_gemma4_e4b
    from cts.backbone.gemma_adapter import GemmaCTSBackbone

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    logger.info("Loading base Gemma-4 model on %s...", device)
    m, t = load_gemma4_e4b(model_id=args.base_model_path, device_map=str(device), torch_dtype=torch.bfloat16)
    bb = GemmaCTSBackbone(m, t)

    # 4. Strictly Load Checkpoint
    if args.checkpoint_path:
        load_checkpoint_strictly(bb, args.checkpoint_path)
    bb.to(device=device, dtype=torch.bfloat16)
    bb.eval()

    # 5. Collect Canonical z* Vectors (FileNotFoundError if absent; zero in-harness recomputation)
    logger.info("Collecting equilibrium latent z* vectors from canonical artifacts...")
    z_stars: List[torch.Tensor] = []
    for s in samples:
        q_idx = s["dataset_index"]
        z = resolve_canonical_z_star(q_idx, args.z_star_path)
        if z is None:
            raise FileNotFoundError(
                f"Required equilibrium latent z* for question index {q_idx} not found in canonical artifacts "
                f"(Directive EXE-260912-06 §1.4). In-harness recomputation is strictly prohibited; "
                f"z* vectors must be supplied by the HS-8 reproduction artifact."
            )
        z_stars.append(z.to(device=device, dtype=torch.bfloat16))
    logger.info("Successfully loaded %d z* tensors.", len(z_stars))

    # 6. Sequential Evaluation across 4 Conditions
    results_by_condition: Dict[str, Any] = {}
    for cond in ("no_prefix", "matched", "shuffled", "zero"):
        logger.info("Evaluating condition '%s'...", cond)
        cond_res = run_decoding_condition(
            bb=bb,
            condition=cond,
            samples=samples,
            z_stars=z_stars,
            max_new_tokens=args.max_new_tokens
        )
        results_by_condition[cond] = cond_res
        logger.info(
            "Condition '%s' finished: strict=%d/%d (%.1f%%), lenient=%d/%d (%.1f%%)",
            cond,
            cond_res["strict_correct"],
            cond_res["total_samples"],
            cond_res["strict_accuracy"] * 100,
            cond_res["lenient_correct"],
            cond_res["total_samples"],
            cond_res["lenient_accuracy"] * 100
        )

    # 7. Compute Bootstrap CI (1,000 resamples)
    logger.info("Computing 1,000-iteration item-level bootstrap...")
    bootstrap_res = compute_bootstrap_ci(
        matched_results=results_by_condition["matched"],
        shuffled_results=results_by_condition["shuffled"],
        num_bootstraps=1000
    )

    # 8. Assemble and Save Results
    full_output = {
        "metadata": {
            "harness_version": "v2.2",
            "stage": args.stage,
            "target_indices": target_indices,
            "max_new_tokens": args.max_new_tokens,
            "checkpoint_path": args.checkpoint_path,
            "base_model_path": args.base_model_path,
        },
        "results_by_condition": results_by_condition,
        "bootstrap": bootstrap_res
    }

    out_json = args.output_json or f"results_eval_{args.stage}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    logger.info("Full evaluation results saved to %s", out_json)


def run_dry_run(args: argparse.Namespace) -> None:
    """Execute dry-run inspection without model loading or GPU computation."""
    logger.info("=== RUNNING DRY-RUN INSPECTION (EXE-260912-06 / v2.2) ===")

    if args.stage == "stage1":
        target_indices = STAGE1_SAMPLE_INDICES
    elif args.stage == "stage6":
        target_indices = STAGE6_SAMPLE_INDICES
    else:
        target_indices = STAGE1_SAMPLE_INDICES + STAGE6_SAMPLE_INDICES

    logger.info("Selected stage: %s (%d items: %s)", args.stage, len(target_indices), target_indices)

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

    w_out, w_in = inspect_w_proj_shape(args.checkpoint_path)

    print("--- DYNAMIC PREFIX TENSOR SPECIFICATIONS (CPU ckpt inspection) ---")
    print(f"Checkpoint w_proj shape:  [{w_out}, {w_in}]")
    print(f"Latent z* shape:          [K=64, d={w_in}] (Harness A F-6 beta=0)")
    print(f"Projected prefix_embeds:  [1, K=64, d_model={w_out}] via bb.w_proj(z*)")
    print("Insertion point:          torch.cat([prefix_embeds, txt_embeds], dim=1) (gemma_adapter.py:443)")
    print(f"Prefix conditions:        {VALID_CONDITIONS}")
    print(f"Max new tokens:           {args.max_new_tokens} (default: {DEFAULT_MAX_NEW_TOKENS})\n")

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
    parser = argparse.ArgumentParser(description="CTS Canonical Decoder Evaluation Harness (v2.2 Final)")
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
        "--base_model_path",
        type=str,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Path to base Gemma-4 HuggingFace checkpoint directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (e.g. cuda:0 or cpu)"
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

    # Governance safety lock check directly before entering full pipeline
    if EXECUTION_LOCKED:
        logger.error(
            "EXECUTION_LOCKED is True. Direct model execution is strictly prohibited "
            "prior to Phase E1 sign-off by Principal (Directive EXE-260912-06 / 원장 #200). "
            "Use --dry_run to inspect prompts, tensor shapes, and condition assignments."
        )
        sys.exit(1)

    run_full(args)


if __name__ == "__main__":
    main()

"""
scripts/harness_decode_v1.py - Canonical Decoder Evaluation Harness (v1)

Specification (Directive EXE-260912-02 §2 / Ledger #196):
1. Uniform Prompting:
   All 4 decoding conditions ('no_prefix', 'matched', 'shuffled', 'zero')
   MUST use the exact same prompt constructor `build_prompt(problem)`
   (applying MATH_CONCISE_BOXED_PREFIX + Gemma turn wrapper).
2. Strict Evaluation:
   When strict=True, if extraction fails (no \boxed{...}) or evaluation
   cannot strictly verify the answer, it MUST halt with an exception (StrictEvaluationError).
   No silent fallbacks allowed.
3. Execution Policy:
   Code deployment and SHA verification only. Execution is STRICTLY PROHIBITED
   until Phase E1 verification is completed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("harness_decode_v1")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cts.prompt_contract import MATH_CONCISE_BOXED_PREFIX
from cts.eval.lenient_extract import _extract_boxed
from cts.eval.math500 import answers_match, extract_gold

VALID_CONDITIONS = ("no_prefix", "matched", "shuffled", "zero")


class StrictEvaluationError(RuntimeError):
    """Raised when strict=True evaluation fails without fallback."""
    pass


def build_prompt(problem: str, tokenizer: Any = None) -> str:
    """Canonical uniform prompt constructor across all 4 decoding conditions.

    Guarantees that 'no_prefix', 'matched', 'shuffled', and 'zero' share the exact same
    prompt text and turn wrapper without condition-dependent prompt divergence.
    """
    problem_text = str(problem).strip()
    user_content = f"{MATH_CONCISE_BOXED_PREFIX}{problem_text}"

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Canonical Gemma turn wrapper fallback when tokenizer is not supplied
    return f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"


def evaluate_prediction(pred_text: str, gold_raw: str, strict: bool = True) -> bool:
    """Evaluate predicted completion against gold reference answer.

    If strict=True:
      - Requires a valid balanced \boxed{...} answer in pred_text.
      - If no \boxed{...} is present, raises StrictEvaluationError immediately.
      - Never falls back to heuristics or unverified strings.
    """
    gold_answer = extract_gold(gold_raw)

    if strict:
        extracted = _extract_boxed(pred_text)
        if extracted is None or not extracted.strip():
            raise StrictEvaluationError(
                f"strict=True evaluation failure: completion does not contain a valid \\boxed{{...}} answer. "
                f"Completion snippet: {repr(pred_text[-120:])}"
            )
        return bool(answers_match(extracted.strip(), gold_answer))
    else:
        from cts.eval.lenient_extract import lenient_extract
        extracted = lenient_extract(pred_text)
        if not extracted:
            return False
        return bool(answers_match(extracted, gold_answer))


def run_decoding_condition(
    condition: str,
    dataset: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    strict: bool = True,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Run decoding evaluation for a specific latent condition.

    Condition semantics:
      - 'no_prefix': unconditioned base generation (no latent prefix vector).
      - 'matched': conditioned on problem's corresponding equilibrium latent z*.
      - 'shuffled': conditioned on randomly permuted latent vectors from the batch.
      - 'zero': conditioned on all-zero latent vector 0.
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Invalid condition '{condition}'. Must be one of {VALID_CONDITIONS}")

    eval_items = dataset[:max_samples] if max_samples is not None else dataset
    total = len(eval_items)
    correct_count = 0

    logger.info(f"Starting evaluation under condition '{condition}' on {total} samples (strict={strict})...")

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(eval_items):
        problem = item.get("problem") or item.get("question") or ""
        gold = item.get("answer") or ""

        # Uniform prompt construction across all conditions
        prompt = build_prompt(problem, tokenizer=tokenizer)

        # Execution stub (execution prohibited in current phase)
        raise RuntimeError(
            "Execution is locked by Phase E1 governance policy (Directive EXE-260912-02). "
            "harness_decode_v1.py is staged in write-only status."
        )

    return {
        "condition": condition,
        "total": total,
        "correct": correct_count,
        "accuracy": (correct_count / total) if total > 0 else 0.0,
        "results": results
    }


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTS Canonical Decoder Evaluation Harness (v1)")
    parser.add_argument(
        "--condition",
        type=str,
        choices=VALID_CONDITIONS,
        required=True,
        help="Latent decoding condition: no_prefix | matched | shuffled | zero"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enforce strict evaluation (raise exception on missing \boxed, no fallback)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl",
        help="Path to evaluation dataset JSONL"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate"
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    logger.info(f"Harness initialized for condition={args.condition}, strict={args.strict}")
    logger.info("Governance Note: Execution is strictly prohibited prior to E1 sign-off.")


if __name__ == "__main__":
    main()

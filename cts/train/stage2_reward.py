"""Stage-2 PPO rollout reward helpers (paper Eq. 5).

Paper Eq.(5): R_total = 1{correct_answer} - lambda_halt * T.

The shipped JSONL historically carried prompts only (no gold solution).
When ``solution`` / ``answer`` is absent, the trainer falls back to a
**DEQ convergence proxy** (``solver_stats['converged']``) so PPO can
still run without an oracle. When gold is present *and*
``stage2_rollout_decode_tokens`` is large enough for ``child_text`` to
contain an extractable answer, the answer-based path is used instead.

See ``LIMITATIONS.md`` §15 for the honest disclosure.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from cts.rewards.shaping import paper_reward


def gold_from_stage2_row(row: Dict[str, Any]) -> Optional[str]:
    """Extract the *final answer* from a Stage-2 row, if present.

    MATH training rows store a verbose derivation in ``solution``.  Returning
    that whole derivation made the legacy numeric grader compare the first
    number in the proof against the model's answer (only about 3% of audited
    rows happened to match).  Prefer explicit answer fields; otherwise apply
    the same final-answer extractor used for model outputs to the solution.
    """
    if not isinstance(row, dict):
        return None
    for key in ("answer", "expected_answer"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for key in ("solution", "generated_solution"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            from cts.eval.math500 import extract_answer

            extracted = extract_answer(v)
            return extracted.strip() if extracted.strip() else v.strip()
    return None


def stage2_rollout_is_correct(
    row: Dict[str, Any],
    child_text: Optional[str],
    *,
    converged: bool,
    reward_mode: str = "auto",
) -> bool:
    """Decide the ``correct`` bit for ``paper_reward()`` (Eq. 5).

    Modes (``configs/default.yaml`` key ``stage2_reward_mode``):

    - ``converged``: legacy proxy — Broyden fixed-point convergence only.
    - ``answer``: gold required; False when gold or ``child_text`` missing.
    - ``auto`` (default): gold + decodable ``child_text`` → answer match;
      otherwise convergence proxy.
    """
    mode = (reward_mode or "auto").strip().lower()
    if mode == "converged":
        return bool(converged)

    gold = gold_from_stage2_row(row)
    has_oracle = gold is not None and bool((child_text or "").strip())

    if mode == "answer":
        if not has_oracle:
            return False
        return _answer_matches(child_text or "", gold)

    # auto
    if has_oracle:
        return _answer_matches(child_text or "", gold)
    return bool(converged)


def stage2_rollout_reward(
    row: Dict[str, Any],
    *,
    child_text: Optional[str],
    converged: bool,
    terminal_depth: int,
    lambda_halt: float,
    reward_mode: str = "auto",
) -> float:
    """Paper Eq.(5) reward for one Stage-2 PPO rollout step."""
    correct = stage2_rollout_is_correct(
        row,
        child_text,
        converged=converged,
        reward_mode=reward_mode,
    )
    return paper_reward(
        correct=correct,
        terminal_depth=terminal_depth,
        lambda_halt=lambda_halt,
    )


def _answer_matches(pred_raw: str, gold_raw: str) -> bool:
    from cts.eval.math500 import extract_answer, extract_gold, normalize_answer

    pred = extract_answer(pred_raw)
    gold = extract_gold(gold_raw)
    if not pred or not gold:
        return False
    pn = normalize_answer(pred)
    gn = normalize_answer(gold)
    if pn == gn:
        return True

    # The legacy numeric fallback searched the *first* number anywhere in an
    # expression, so ``2/3`` and ``2/7`` both became 2.0 and produced a false
    # positive PPO reward.  Until a separately validated symbolic grader is
    # available, only compare strings that are entirely plain decimals.
    decimal = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
    if decimal.fullmatch(pn) and decimal.fullmatch(gn):
        try:
            return abs(float(pn) - float(gn)) < 1e-6
        except ValueError:
            return False
    return False

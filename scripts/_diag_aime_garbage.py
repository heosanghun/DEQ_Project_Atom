"""Diagnose why local AIME predictions come back as 'Cultura' / '?????' garbage.

Single-problem inspection: load the first AIME problem, run greedy through
the same _build_prompt + GemmaTextPredictor + _extract_pred path that
scripts/run_cts_eval_full.py uses, and dump everything (raw decoded text,
extracted answer, gold answer) so we can see the exact failure mode.

Run:
    python scripts/_diag_aime_garbage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    print("=" * 70)
    print("AIME garbage diagnosis -- single-problem trace")
    print("=" * 70)

    aime_path = ROOT / "data" / "aime" / "test.jsonl"
    with open(aime_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    problem = first["problem"]
    gold = first["answer"]
    print(f"\n[1] Problem (year={first.get('year')}, source={first.get('source')}):")
    print(f"    {problem[:200]}{'...' if len(problem) > 200 else ''}")
    print(f"\n[2] Gold answer: '{gold}'")

    from scripts.run_cts_eval_full import (
        _build_prompt,
        _extract_pred,
        _match_answer,
        _get_predictor,
        _max_tokens_for,
    )

    prompt = _build_prompt(problem, "aime", native_think=False)
    print(f"\n[3] Built prompt ({len(prompt)} chars):")
    print("-" * 70)
    print(prompt)
    print("-" * 70)

    print("\n[4] Loading Gemma-4-E4B predictor (this may take ~20s)...")
    predictor, model, tok = _get_predictor("cuda:0", None)
    pred_max_tok = _max_tokens_for("aime")
    print(f"    Predictor loaded. max_new_tokens budget for AIME: {pred_max_tok}")

    print(f"\n[5] Running greedy decode (this may take ~30s)...")
    raw = predictor(prompt, max_new_tokens=pred_max_tok)
    print(f"\n[6] Raw decoded output ({len(str(raw))} chars):")
    print("=" * 70)
    print(repr(str(raw)))
    print("=" * 70)
    print("\n[6b] Raw decoded output (rendered):")
    print("-" * 70)
    print(str(raw))
    print("-" * 70)

    pred = _extract_pred(str(raw), "aime")
    match = _match_answer(pred, gold, "aime")
    print(f"\n[7] Extracted prediction: '{pred}'")
    print(f"[8] Gold:                  '{gold}'")
    print(f"[9] Match: {match}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS HINTS")
    print("=" * 70)
    raw_str = str(raw)
    has_boxed = r"\boxed{" in raw_str
    has_answer_is = "answer is" in raw_str.lower()
    has_gsm_marker = "####" in raw_str
    has_thinking = "<think>" in raw_str.lower() or "thinking" in raw_str.lower()
    has_chat_token = "<start_of_turn>" in raw_str or "<end_of_turn>" in raw_str
    has_user_replay = ("\nuser\n" in raw_str.lower() or "user:" in raw_str.lower())
    truncated_at_budget = len(raw_str) > 0 and not raw_str.rstrip().endswith((".", "}", ")", "]", "$"))

    import re
    nums = re.findall(r"-?\d+", raw_str)

    print(f"  contains '\\boxed{{}}'                    : {has_boxed}")
    print(f"  contains 'answer is'                     : {has_answer_is}")
    print(f"  contains '####'                          : {has_gsm_marker}")
    print(f"  contains thinking marker                 : {has_thinking}")
    print(f"  contains chat-template control token     : {has_chat_token}")
    print(f"  contains hallucinated user-turn          : {has_user_replay}")
    print(f"  appears truncated at budget              : {truncated_at_budget}")
    print(f"  numeric tokens found in raw              : {len(nums)} (last 5: {nums[-5:] if nums else []})")
    print(f"  if last numeric == gold ({gold}): {nums[-1] == str(int(gold)) if nums else False}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

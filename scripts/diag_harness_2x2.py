#!/usr/bin/env python3
"""Standalone harness-diagnosis runner (does NOT touch the eval path).

Purpose (Track B, Step 0): isolate WHY the greedy baseline is near-zero by
separating the *prompt-format* effect from the *model-weights* effect in a
base/-it x plain/canonical 2x2, plus a REF cell for the current path.

This script imports only pure helpers (cts.eval.math500) + transformers. It
never modifies run_cts_eval_full.py, so the existing diagnostic baseline is
preserved.

Note on Tokenizer / Vocabulary:
Base and -it models have been verified to share the same vocabulary and tokenizer
structures, which makes cross-format testing technically viable.
Note on M3: This is considered a 'paper-faithful candidate' subject to final verification.

BOS policy:
  By default, pre-rendered prompts are encoded with ``add_special_tokens=False``;
  a missing BOS is prepended, multiple BOS tokens are collapsed to one at the
  beginning, and the final one-BOS invariant is asserted.  ``--strict-bos``
  disables that normalization and asserts the rendered-token contract as-is.

Cells (pass exactly one via --cell; each invocation is a FRESH process):
  REF : base weights + current harness prompt (math greedy == plain 'Solution:';
        with --emit-fake-markers it reproduces the native_think <start_of_turn> bug)
        Under EOS mode 'ref' (EOS=1).
  M0  : base weights + plain prompt (controlled EOS)
  M1  : base weights + canonical chat template (controlled EOS)
  M2  : -it  weights + plain prompt (controlled EOS)
  M3  : -it  weights + canonical chat template (controlled EOS) - paper-faithful candidate
  M0_native: base weights + plain prompt (native EOS)
  M1_native: base weights + canonical chat template (native EOS)
  M2_native: -it  weights + plain prompt (native EOS)
  M3_native: -it  weights + canonical chat template (native EOS)

Example:
  python scripts/diag_harness_2x2.py --cell M3 --benchmark math500 --limit 20 \
      --base-model google/gemma-4-E4B --it-model google/gemma-4-E4B-it \
      --data data/math500/test.jsonl --device cuda:0 \
      --out results/diag2x2/M3_math500.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cts.eval.math500 import (  # pure helpers
    answers_match,
    extract_answer,
    extract_answer_method,
    extract_gold,
)

CELLS = {
    # cell -> (weights: 'base'|'it', fmt: 'plain'|'canonical'|'current', eos_mode: 'ref'|'controlled'|'native')
    "REF": ("base", "current", "ref"),
    "M0": ("base", "plain", "controlled"),
    "M1": ("base", "canonical", "controlled"),
    "M2": ("it", "plain", "controlled"),
    "M3": ("it", "canonical", "controlled"),  # paper-faithful candidate
    "M0_native": ("base", "plain", "native"),
    "M1_native": ("base", "canonical", "native"),
    "M2_native": ("it", "plain", "native"),
    "M3_native": ("it", "canonical", "native"),
}

TURN_END_ID = 106
EOS_ID = 1
GRADER_DIAGNOSTIC_NOTE = (
    "Observed equivalence failures are grader-limit hypotheses only; they are "
    "not evidence that the grader caused near-zero accuracy until actual model "
    "predictions are regraded and the accuracy delta is measured."
)


def _get_question(prob: dict, benchmark: str) -> str:
    if benchmark == "gsm8k":
        return prob.get("question") or prob.get("problem") or ""
    return prob.get("problem") or prob.get("question") or ""


def _get_gold(prob: dict, benchmark: str) -> str:
    if benchmark == "gsm8k":
        return str(prob.get("answer", ""))
    return str(prob.get("answer") or prob.get("solution") or "")


def _problem_id(prob: dict, i: int) -> str:
    return str(prob.get("unique_id") or prob.get("id") or prob.get("idx") or i)


_ONE_SHOT_ENABLED = False

# A worked example is a far stronger format cue than an instruction alone.
# Diagnostic finding: with the instruction only, 0/250 completions on MATH-500
# emitted \boxed{} at 512 tokens AND 0% still emitted it at 2048 tokens, so the
# model is not following the format directive at any budget.
ONE_SHOT_EXAMPLE = (
    "Example:\n"
    "Problem: What is $2 + 3 \\times 4$?\n"
    "Solution: Multiplication comes first, so $3 \\times 4 = 12$. "
    "Then $2 + 12 = 14$.\n"
    "The final answer is $\\boxed{14}$.\n"
)


def _instruction(benchmark: str) -> str:
    base = ("Solve the following math problem step by step. "
            "Put your final answer in \\boxed{}.")
    if _ONE_SHOT_ENABLED:
        return f"{base}\n\n{ONE_SHOT_EXAMPLE}"
    return base


def build_plain_prompt(q: str, benchmark: str) -> str:
    return f"{_instruction(benchmark)}\n\n{q}\n\nSolution:"


def build_current_prompt(q: str, benchmark: str, fake_markers: bool) -> str:
    if fake_markers:
        return (f"<start_of_turn>user\n{_instruction(benchmark)}\n{q}"
                f"<end_of_turn>\n<start_of_turn>model\n")
    return build_plain_prompt(q, benchmark)


def render_prompt(tok, cell_fmt: str, q: str, benchmark: str, *, think: bool,
                  fake_markers: bool) -> str:
    if cell_fmt == "plain":
        return build_plain_prompt(q, benchmark)
    if cell_fmt == "current":
        return build_current_prompt(q, benchmark, fake_markers)
    # canonical: apply the model's own chat template
    msgs = [{"role": "user", "content": f"{_instruction(benchmark)}\n\n{q}"}]
    return tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False,
        **({"enable_thinking": True} if think else {}),
    )


def contract_clean(raw: str) -> str:
    import re
    raw = re.sub(r"<\|channel>.*?<channel\|>", " ", raw, flags=re.DOTALL)
    raw = raw.split("<turn|>")[0]
    raw = re.sub(r"<\|?(?:turn|think|channel|eos|bos|pad)\|?>", " ", raw)
    return raw.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return "NA"


def load_problem_ids(path: str) -> List[str]:
    p = Path(path)
    if not p.is_file():
        return []
    content = p.read_text(encoding="utf-8").strip()
    if not content:
        return []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in content.splitlines() if line.strip()]


def run_grader_diagnostics() -> Dict[str, Any]:
    """Test standard equivalence cases for math evaluation."""
    test_cases = [
        ("fraction_decimal_pos1", "\\frac{1}{2}", "0.5", True),
        ("fraction_decimal_pos2", "0.25", "\\frac{1}{4}", True),
        ("fraction_equiv_pos", "\\frac{2}{4}", "\\frac{1}{2}", True),
        ("sqrt_pos1", "\\sqrt{4}", "2", True),
        ("sqrt_pos2", "3\\sqrt{2}", "\\sqrt{18}", True),
        ("tuple_pos1", "(1, 2)", "(1,2)", True),
        ("tuple_pos2", "\\left(1, 2\\right)", "(1, 2)", True),
        # Negative checks (must be false)
        ("diff_val_neg", "0.5", "0.6", False),
        ("fraction_decimal_neg", "\\frac{1}{2}", "0.3", False),
        ("tuple_neg", "(1, 2)", "(2, 1)", False),
    ]
    results = {}
    for name, p, g, expected in test_cases:
        actual = bool(answers_match(p, g))
        results[name] = {
            "pred": p, "gold": g,
            "expected": expected, "actual": actual,
            "passed": (actual == expected)
        }
    return results


def load_model_and_tok(weights: str, base_model: str, it_model: str, device: str):
    from transformers import AutoTokenizer
    from cts.model.gemma_loader import load_gemma4_e4b
    tok = AutoTokenizer.from_pretrained(it_model, trust_remote_code=True)
    model_id = base_model if weights == "base" else it_model
    model, _tok2 = load_gemma4_e4b(model_id=model_id, device_map=device,
                                   torch_dtype=torch.bfloat16)
    model.eval()
    return model, tok, model_id


@torch.inference_mode()
def generate(model, tok, prompt: str, device: str, max_new_tokens: int,
             eos_ids: List[int], *, strict_bos: bool) -> Tuple[str, List[int], List[int]]:
    # Canonical input must be encoded with add_special_tokens=False
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids_list = enc["input_ids"][0].tolist()
    bos_id = tok.bos_token_id

    # Strict mode observes the rendered-token contract without repairing it.
    bos_count = input_ids_list.count(bos_id)
    if strict_bos:
        assert bos_count == 1, f"strict BOS count must be 1, found {bos_count}"
        assert input_ids_list and input_ids_list[0] == bos_id, (
            "strict BOS policy requires BOS to be the first rendered token"
        )
    else:
        if bos_count == 0:
            input_ids_list = [bos_id] + input_ids_list
        elif bos_count > 1:
            input_ids_list = [x for x in input_ids_list if x != bos_id]
            input_ids_list = [bos_id] + input_ids_list

    assert len(input_ids_list) > 0 and input_ids_list[0] == bos_id, f"BOS must be the first token, found {input_ids_list[0]}"
    assert input_ids_list.count(bos_id) == 1, f"BOS count must be exactly 1, found {input_ids_list.count(bos_id)}"

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=0, eos_token_id=eos_ids,
    )
    in_len = input_ids.shape[1]
    new_ids = out[0, in_len:].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text, input_ids_list, new_ids


def oracle_grader_check(gold: str, benchmark: str) -> bool:
    gold_ans = extract_gold(gold)
    synth = f"The final answer is \\boxed{{{gold_ans}}}."
    pred = extract_answer(synth)
    return answers_match(pred, gold_ans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--benchmark", required=True, choices=["math500", "gsm8k"])
    ap.add_argument("--data", required=True, help="benchmark JSONL")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--problem-ids-file", default=None, help="JSON or text file of problem IDs")
    ap.add_argument("--base-model", default=os.environ.get("CTS_BASE_MODEL", "google/gemma-4-E4B"))
    ap.add_argument("--it-model", default=os.environ.get("CTS_IT_MODEL", "google/gemma-4-E4B-it"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--think", action="store_true", help="enable_thinking for canonical cells")
    ap.add_argument("--one-shot", action="store_true",
                    help="prepend a worked example ending in \\boxed{} to elicit the answer format")
    ap.add_argument("--emit-fake-markers", action="store_true",
                    help="REF cell: reproduce the native_think <start_of_turn> bug")
    ap.add_argument(
        "--strict-bos",
        action="store_true",
        help="assert the rendered BOS contract without inserting or deduplicating BOS",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global _ONE_SHOT_ENABLED
    _ONE_SHOT_ENABLED = bool(args.one_shot)

    weights, fmt, eos_mode = CELLS[args.cell]
    
    # Stratified problem IDs
    allowed_ids = []
    if args.problem_ids_file:
        allowed_ids = load_problem_ids(args.problem_ids_file)
        print(f"Loaded {len(allowed_ids)} stratified problem IDs for evaluation filter.", flush=True)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    model, tok, model_id = load_model_and_tok(weights, args.base_model, args.it_model, args.device)

    # Establish EOS config
    if eos_mode == "ref":
        eos_ids = [EOS_ID]
        eos_source = "ref_contract"
    elif eos_mode == "controlled":
        eos_ids = [EOS_ID, TURN_END_ID]
        eos_source = "controlled_contract"
    else:  # native
        generation_config = getattr(model, "generation_config", None)
        native_eos = getattr(generation_config, "eos_token_id", None)
        if native_eos is None:
            native_eos = model.config.eos_token_id
            eos_source = "model_config"
        else:
            eos_source = "generation_config"
        if isinstance(native_eos, int):
            eos_ids = [native_eos]
        elif isinstance(native_eos, list):
            eos_ids = list(native_eos)
        else:
            eos_ids = [EOS_ID]

    # Run grader diagnostics
    grader_diag = run_grader_diagnostics()
    print("\n--- Grader Diagnostics Report ---")
    for name, r in grader_diag.items():
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {name:25} (expected={r['expected']}, actual={r['actual']})")
    grader_integrity_pass = all(result["passed"] for result in grader_diag.values())
    print(f"integrity_pass={grader_integrity_pass}")
    print(f"Interpretation: {GRADER_DIAGNOSTIC_NOTE}")
    print("---------------------------------\n")

    cell_label = args.cell
    if args.cell == "M3":
        cell_label = "M3 (paper-faithful candidate)"

    meta = {
        "cell": cell_label, "weights": weights, "fmt": fmt, "eos_mode": eos_mode, "benchmark": args.benchmark,
        "model_id": model_id, "device": args.device,
        "transformers_version": __import__("transformers").__version__,
        "vocab_size": len(tok.get_vocab()),
        "eos_ids": eos_ids, "eos_source": eos_source,
        "max_new_tokens": args.max_new_tokens,
        "think": bool(args.think), "emit_fake_markers": bool(args.emit_fake_markers),
        "strict_bos": bool(args.strict_bos),
        "bos_policy": "strict_assert_only" if args.strict_bos else "normalize_then_assert",
        "grader_diagnostics": grader_diag,
        "grader_integrity_pass": grader_integrity_pass,
        "grader_interpretation": GRADER_DIAGNOSTIC_NOTE,
    }
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_processed = 0
    
    with open(args.out, "w", encoding="utf-8") as fo:
        fo.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for i, prob in enumerate(rows):
            pid = _problem_id(prob, i)
            if allowed_ids and pid not in allowed_ids:
                continue
            if args.limit is not None and n_processed >= args.limit:
                break
                
            q = _get_question(prob, args.benchmark)
            gold = _get_gold(prob, args.benchmark)
            if not q:
                continue
                
            prompt = render_prompt(tok, fmt, q, args.benchmark,
                                   think=args.think, fake_markers=args.emit_fake_markers)
            t0 = time.time()
            raw, prompt_ids, out_ids = generate(
                model, tok, prompt, args.device, args.max_new_tokens, eos_ids,
                strict_bos=args.strict_bos)
            dt = (time.time() - t0) * 1000.0
            cleaned = contract_clean(raw)
            pred = extract_answer(cleaned)
            gold_ans = extract_gold(gold)
            match = answers_match(pred, gold_ans)
            oracle = oracle_grader_check(gold, args.benchmark)
            
            if match:
                n_ok += 1
            n_processed += 1
            
            rec = {
                "problem_id": pid,
                "gold_raw": gold[:512], "gold_ans": gold_ans[:256],
                "prompt_render": prompt, "prompt_token_ids": prompt_ids,
                "n_prompt_tokens": len(prompt_ids),
                "raw_output_text": raw, "raw_output_token_ids": out_ids,
                "cleaned_output": cleaned, "extracted_pred": pred[:256],
                "graded_match": bool(match), "oracle_grader_check": bool(oracle),
                "extract_method": extract_answer_method(cleaned),
                "wall_ms": round(dt, 1),
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{args.cell}/{args.benchmark}] {n_processed} "
                  f"match={match} oracle={oracle} pred={pred[:30]!r} gold={gold_ans[:30]!r} "
                  f"{dt:.0f}ms", flush=True)
                  
    acc = n_ok / max(1, n_processed)
    print(f"\n[{args.cell}/{args.benchmark}] accuracy={acc:.4f} ({n_ok}/{n_processed}) -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

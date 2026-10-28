#!/usr/bin/env python3
"""scripts/s5_eval_gate_v2.py - S5 Held-Out Gate Evaluation Script (v2)
Directive: EXE-260923-28 rev2 / EXE-260923-29 (System 2 / Ledger #331, #333)

Specifications:
- Executed on trained checkpoints or in harness-anchor mode.
- Held-out 80 questions evaluated under 5 conditions:
  1) matched: P(z*_i) (canonical prompt format, problem text 0 tokens)
  2) shuffled: P(z*_j) over 3 round-robin cycles (80 x 3 = 240 evaluations)
  3) zero: Zero prefix (64 x 2560 zeros)
  4) random-z: Gaussian noise N(0, sigma^2) with matching ||z*|| norm
  5) no_prefix: baseline wrapper with slots removed (<bos><|turn>user\n<turn|>\n<|turn>model\n)
  6) text_only: problem text only, zero prefix (harness anchor mode, non-gate)
- Grading: Strict exact match (_extract_boxed) and lenient extraction (lenient_extract) with answers_match.
- 3 Strata Reporting:
  1) all_80: All 80 held-out items
  2) converged_71: 71 converged items from Phase 1
  3) non_converged_9: 9 non-converged items from Phase 1
- Metrics per stratum:
  acc_matched, acc_shuffled, acc_zero, acc_random_z, acc_no_prefix,
  delta = acc_matched - acc_shuffled, 10,000 bootstrap CI [ci_lower, ci_upper]
- Criterion II-b: Control comparison (lambda=1 vs lambda=0 paired diff, bootstrap CI)
- Criterion II-c: Format / Content NLL decomposition
- Criterion II-d: w_proj weight Frobenius norm and std(W) vs B1 initial (52.098614)
- Harness Anchor: Runs text_only on 10 Item 4 questions (0,4,...,36) with B1 -> reproduces 7/10 lenient, 5/10 strict.
- Output:
  1) results_s5/phase2_gate.json
  2) results_s5/phase2_outputs.jsonl
  3) SHA-256 registered in MANIFEST_s5.sha256
- Strictly NO judgment words in output JSON!
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/system1_5"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

SCRIPTS_DIR = os.path.join(BASE_S5_DIR, "scripts")
for p in ["/home/shoon/cts_e1", os.path.join(SYS_DIR, "scripts"), SCRATCH_DIR, SYS_DIR, SCRIPTS_DIR, BASE_S5_DIR]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from cts.eval.lenient_extract import lenient_extract, _extract_boxed
from cts.eval.math500 import answers_match, extract_gold, extract_answer
from cts.prompt_contract import build_math_decode_prompt

try:
    from s5_train_decoder_v2 import build_canonical_prefix_prompt, load_tokenizer, PrefixProjector, find_boxed_char_span
except ImportError:
    from scripts.s5_train_decoder_v2 import build_canonical_prefix_prompt, load_tokenizer, PrefixProjector, find_boxed_char_span


DEFAULT_RESULTS_DIR = os.path.join(BASE_S5_DIR, "results_s5")
DEFAULT_RAW_DIR = os.path.join(BASE_S5_DIR, "raw_s5")
DEFAULT_MANIFEST_PATH = os.path.join(DEFAULT_RESULTS_DIR, "MANIFEST_s5.sha256")
DEFAULT_B1_CKPT_PATH = "/home/shoon/retrain_1a/B1/stage1a_B1_step5000_20260907_0054.pt"
DEFAULT_BASE_MODEL_PATH = "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
DEFAULT_DATASET_PATH = "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl"
DEFAULT_ITEMS_PATH = os.path.join(BASE_S5_DIR, "configs", "s5_heldout_80.json")
DEFAULT_ZSTAR_PATH = os.path.join(DEFAULT_RESULTS_DIR, "zstar_s5_400.pt")
DEFAULT_OUT_JSON = os.path.join(DEFAULT_RESULTS_DIR, "phase2_gate.json")
DEFAULT_OUTPUTS_JSONL = os.path.join(DEFAULT_RESULTS_DIR, "phase2_outputs.jsonl")

ITEM4_TEN_INDICES = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
B1_INITIAL_FROB = 52.098614


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def append_to_manifest(manifest_path: str, target_file: str) -> str:
    sha = compute_sha256(target_file)
    rel_name = os.path.basename(target_file)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(f"{sha}  {rel_name}\n")
    return sha


def compute_bootstrap_ci(
    matched_correct: List[int],
    shuffled_correct: List[int],
    n_bootstraps: int = 10000,
    seed: int = 2030,
) -> Tuple[float, float, float]:
    """Compute bootstrap 95% confidence interval for delta = acc_matched - acc_shuffled."""
    rng = np.random.default_rng(seed)
    n = len(matched_correct)
    if n == 0:
        return 0.0, 0.0, 0.0
    assert n == len(shuffled_correct), "Sample size mismatch"

    matched_arr = np.array(matched_correct, dtype=np.float32)
    shuffled_arr = np.array(shuffled_correct, dtype=np.float32)

    delta_obs = float(np.mean(matched_arr) - np.mean(shuffled_arr))

    boot_deltas = []
    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        d = np.mean(matched_arr[boot_idx]) - np.mean(shuffled_arr[boot_idx])
        boot_deltas.append(d)

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    return delta_obs, ci_lower, ci_upper


def load_dataset_records(dataset_path: str = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    candidates = [
        dataset_path,
        os.path.join(SYS_DIR, "data", "stage2", "math_train_prompts_5000.jsonl"),
        "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl",
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    raise FileNotFoundError(f"Dataset not found in candidates: {candidates}")


def load_heldout_indices(items_path: str = DEFAULT_ITEMS_PATH) -> List[int]:
    candidates = [
        items_path,
        os.path.join(BASE_S5_DIR, "configs", "s5_heldout_80.json"),
        os.path.join(BASE_S5_DIR, "data", "s5_items_400.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [int(x) for x in data]
            if "heldout_indices" in data:
                return [int(x) for x in data["heldout_indices"]]
            if "indices" in data:
                return [int(x) for x in data["indices"]]
            if "heldout" in data:
                return [int(x) for x in data["heldout"]]
    raise FileNotFoundError(f"Held-out items not found in candidates: {candidates}")


def format_token_line(idx: int, tid: int, tokenizer: Any, prefix_start: Optional[int], prefix_end: Optional[int]) -> str:
    dec = repr(tokenizer.decode([tid]))
    if prefix_start is not None and prefix_end is not None:
        if idx == 0:
            role = "BOS"
        elif idx == 1:
            role = "TURN_WRAPPER_START (user)"
        elif idx == 2:
            role = "USER_ROLE"
        elif idx == 3:
            role = "NEWLINE_SEPARATOR"
        elif prefix_start <= idx <= prefix_end:
            role = f"PREFIX_SLOT_{idx - prefix_start:02d}"
        elif idx == prefix_end + 1:
            role = "TURN_WRAPPER_END (user)"
        elif idx == prefix_end + 2:
            role = "NEWLINE_SEPARATOR"
        elif idx == prefix_end + 3:
            role = "TURN_WRAPPER_START (model)"
        elif idx == prefix_end + 4:
            role = "MODEL_ROLE"
        elif idx == prefix_end + 5:
            role = "NEWLINE_SEPARATOR"
        else:
            role = "TARGET_OR_EXTRA_TOKEN"
    else:
        if idx == 0:
            role = "BOS"
        elif idx == 1:
            role = "TURN_WRAPPER_START (user)"
        elif idx == 2:
            role = "USER_ROLE"
        elif idx == 3:
            role = "NEWLINE_SEPARATOR"
        elif idx == 4:
            role = "TURN_WRAPPER_END (user)"
        elif idx == 5:
            role = "NEWLINE_SEPARATOR"
        elif idx == 6:
            role = "TURN_WRAPPER_START (model)"
        elif idx == 7:
            role = "MODEL_ROLE"
        elif idx == 8:
            role = "NEWLINE_SEPARATOR"
        else:
            role = "EXTRA_TOKEN"
    return f"Index {idx:02d}: token_id={tid:6d} | decoded={dec:<10} | role={role}\n"


def dump_eval_layout_5cond(raw_dir: str, tokenizer: Any, k_latent: int = 64) -> Dict[str, Any]:
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, "eval_layout_5cond.txt")

    canon_tokens, s_start, s_end, placeholder_id = build_canonical_prefix_prompt(tokenizer, k_latent=k_latent)
    no_prefix_text = "<bos><|turn>user\n<turn|>\n<|turn>model\n"
    no_prefix_tokens = tokenizer.encode(no_prefix_text, add_special_tokens=False)

    conditions = {
        "matched": {"tokens": canon_tokens[:75], "slot_start": s_start, "slot_end": s_end},
        "shuffled": {"tokens": canon_tokens[:75], "slot_start": s_start, "slot_end": s_end},
        "zero": {"tokens": canon_tokens[:75], "slot_start": s_start, "slot_end": s_end},
        "random-z": {"tokens": canon_tokens[:75], "slot_start": s_start, "slot_end": s_end},
        "no_prefix": {"tokens": no_prefix_tokens[:75], "slot_start": None, "slot_end": None},
    }

    cond_lines: Dict[str, List[str]] = {}
    for c_name, c_info in conditions.items():
        lines = []
        for idx, tid in enumerate(c_info["tokens"]):
            lines.append(format_token_line(idx, tid, tokenizer, c_info["slot_start"], c_info["slot_end"]))
        cond_lines[c_name] = lines

    train_dump_lines = cond_lines["matched"]
    diff_results: Dict[str, int] = {}
    for c_name, lines in cond_lines.items():
        diff = list(difflib.unified_diff(train_dump_lines, lines, lineterm=""))
        diff_results[c_name] = len(diff)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== S5 EVALUATION GATE 5-CONDITION LAYOUT TABLE (FIRST 75 TOKENS) ===\n")
        f.write("Directive: EXE-260922-26 rev4 Task 2 (Ledger #323)\n")
        f.write("Evaluation Conditions Summary:\n")
        f.write(f"  Condition 1 (matched):   Slot Range: [{s_start}, {s_end}] (length: {k_latent}) | Token Count: {len(conditions['matched']['tokens'])} | Diff vs Train Dump: {diff_results['matched']} lines\n")
        f.write(f"  Condition 2 (shuffled):  Slot Range: [{s_start}, {s_end}] (length: {k_latent}) | Token Count: {len(conditions['shuffled']['tokens'])} | Diff vs Train Dump: {diff_results['shuffled']} lines\n")
        f.write(f"  Condition 3 (zero):      Slot Range: [{s_start}, {s_end}] (length: {k_latent}) | Token Count: {len(conditions['zero']['tokens'])} | Diff vs Train Dump: {diff_results['zero']} lines\n")
        f.write(f"  Condition 4 (random-z):  Slot Range: [{s_start}, {s_end}] (length: {k_latent}) | Token Count: {len(conditions['random-z']['tokens'])} | Diff vs Train Dump: {diff_results['random-z']} lines\n")
        f.write(f"  Condition 5 (no_prefix): Slot Range: None (0 slots)       | Token Count: {len(conditions['no_prefix']['tokens'])}  | Diff vs Train Dump: {diff_results['no_prefix']} lines (slots removed only)\n")
        f.write("--------------------------------------------------------------------------------\n")
        for c_name in ["matched", "shuffled", "zero", "random-z", "no_prefix"]:
            info = conditions[c_name]
            f.write(f"\n[Condition: {c_name}]\n")
            slot_str = f"[{info['slot_start']}, {info['slot_end']}]" if info["slot_start"] is not None else "None"
            f.write(f"# Slot Range: {slot_str} | Token Count: {len(info['tokens'])}\n")
            for line in cond_lines[c_name]:
                f.write(line)
        f.write("\n=== END EVALUATION LAYOUT TABLE ===\n")

    sha256 = compute_sha256(out_path)
    return {
        "out_path": out_path,
        "sha256": sha256,
        "slot_indices": {
            "matched": [s_start, s_end],
            "shuffled": [s_start, s_end],
            "zero": [s_start, s_end],
            "random-z": [s_start, s_end],
            "no_prefix": None,
        },
        "diff_results": diff_results,
    }


def run_harness_anchor(
    b1_ckpt_path: str = DEFAULT_B1_CKPT_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    base_model_path: str = DEFAULT_BASE_MODEL_PATH,
    device_str: str = "cuda:0",
    raw_dir: str = DEFAULT_RAW_DIR,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
) -> Dict[str, Any]:
    """Execute Harness Anchor on GPU 0 (B1 on 10 Item 4 questions with text_only).
    Verifies 7/10 lenient and 5/10 strict reproduction using exact canonical baseline path.
    """
    print("=" * 80)
    print("S5 Harness Anchor Verification (B1 text_only on 10 Item 4 Questions)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device_str} | B1 ckpt: {b1_ckpt_path}")
    print("=" * 80, flush=True)

    os.makedirs(raw_dir, exist_ok=True)
    log_path = os.path.join(raw_dir, "harness_anchor.log")

    dataset = load_dataset_records(dataset_path)

    from cts.model.gemma_loader import load_gemma4_e4b
    from cts.backbone.gemma_adapter import GemmaCTSBackbone

    device = torch.device(device_str)
    print(f"[HARNESS-ANCHOR] Loading base Gemma-4 model on {device_str}...")
    m, tokenizer = load_gemma4_e4b(model_id=base_model_path, device_map=device_str, torch_dtype=torch.bfloat16)
    bb = GemmaCTSBackbone(m, tokenizer)

    print(f"[HARNESS-ANCHOR] Loading checkpoint strictly from {b1_ckpt_path}")
    ckpt = torch.load(b1_ckpt_path, map_location="cpu")
    sd = ckpt["backbone_state_dict"]
    bb.load_state_dict(sd, strict=True)
    bb.to(device=device, dtype=torch.bfloat16)
    bb.eval()

    results_table: List[Dict[str, Any]] = []
    strict_count = 0
    lenient_count = 0

    for idx, q_idx in enumerate(ITEM4_TEN_INDICES, 1):
        item = dataset[q_idx]
        prompt_text = item.get("prompt") or item.get("problem") or ""
        solution_text = item.get("solution") or item.get("answer") or ""

        body = build_math_decode_prompt(prompt_text)
        formatted_prompt = f"<|turn>user\n{body}<turn|>\n<|turn>model\n"
        gold_answer = _extract_boxed(solution_text) or extract_answer(solution_text) or solution_text

        enc = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output_ids = bb.cg.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            gen_tokens = output_ids[0][enc["input_ids"].shape[1]:]
            resp = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        boxed_val = _extract_boxed(resp)
        boxed_val = boxed_val.strip() if boxed_val else ""
        strict_match = bool(answers_match(boxed_val, gold_answer) if boxed_val else False)

        lenient_val = lenient_extract(resp)
        lenient_match = bool(answers_match(lenient_val, gold_answer) if lenient_val else False)

        if strict_match:
            strict_count += 1
        if lenient_match:
            lenient_count += 1

        rec = {
            "idx": idx,
            "q_idx": q_idx,
            "gold": gold_answer,
            "pred_boxed": boxed_val,
            "pred_lenient": lenient_val,
            "strict_match": strict_match,
            "lenient_match": lenient_match,
        }
        results_table.append(rec)
        print(f"[{idx:02d}/10 Q#{q_idx:02d}] strict={'PASS' if strict_match else 'FAIL'} | lenient={'PASS' if lenient_match else 'FAIL'} (gold='{gold_answer}', boxed='{boxed_val}', len='{lenient_val}')", flush=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== S5 HARNESS ANCHOR 10-QUESTION VERIFICATION LOG ===\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {b1_ckpt_path}\n")
        f.write("Condition: text_only (zero soft-prompt prefix)\n")
        f.write(f"Strict accuracy: {strict_count}/10 ({strict_count*10:.1f}%)\n")
        f.write(f"Lenient accuracy: {lenient_count}/10 ({lenient_count*10:.1f}%)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Idx':<4} | {'Q#':<5} | {'Gold':<15} | {'Strict':<15} | {'Lenient':<15} | {'Strict Match':<12} | {'Lenient Match'}\n")
        f.write("-" * 100 + "\n")
        for r in results_table:
            f.write(f"{r['idx']:<4} | Q#{r['q_idx']:02d} | {str(r['gold'])[:13]:<15} | {str(r['pred_boxed'])[:13]:<15} | {str(r['pred_lenient'])[:13]:<15} | {str(r['strict_match']):<12} | {str(r['lenient_match'])}\n")
        f.write("-" * 100 + "\n")
        f.write("Expected: strict=5/10, lenient=7/10\n")
        verdict = "EXACT_REPRODUCED" if (strict_count == 5 and lenient_count == 7) else "MISMATCH"
        f.write(f"Harness Anchor Result: {verdict}\n")
        f.write("=== END HARNESS ANCHOR LOG ===\n")

    sha = append_to_manifest(manifest_path, log_path)
    print(f"[HARNESS-ANCHOR] Log saved to {log_path} (SHA: {sha})")
    print(f"[HARNESS-ANCHOR] Final: strict={strict_count}/10, lenient={lenient_count}/10")

    if strict_count != 5 or lenient_count != 7:
        print(f"[HARNESS-ANCHOR] CRITICAL FAILURE: Result strict={strict_count}/10, lenient={lenient_count}/10 does NOT match canonical 5/10, 7/10.")
        raise SystemExit(1)

    print("[HARNESS-ANCHOR] SUCCESS: EXACT CANONICAL 7/10 LENIENT, 5/10 STRICT VERIFIED.")
    return {
        "strict_count": strict_count,
        "lenient_count": lenient_count,
        "total_items": 10,
        "log_path": log_path,
        "sha256": sha,
        "records": results_table,
    }


def generate_from_embeds(
    lm: Any,
    lm_head: Any,
    embeds: torch.Tensor,
    tokenizer: Any,
    per_layer_inputs: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
    eos_ids: Optional[List[int]] = None,
) -> str:
    """Fast autoregressive generation with KV caching from input embeddings."""
    device = embeds.device
    if eos_ids is None:
        eos_ids = [tokenizer.eos_token_id]
        if hasattr(tokenizer, "additional_special_tokens_ids"):
            eos_ids.extend(tokenizer.additional_special_tokens_ids)
        # Turn boundary token IDs in Gemma
        for special in ["<turn|>", "<end_of_turn>"]:
            tok_s = tokenizer.encode(special, add_special_tokens=False)
            if tok_s:
                eos_ids.extend(tok_s)
    eos_set = set(eos_ids)

    kwargs_init = {"per_layer_inputs": per_layer_inputs} if per_layer_inputs is not None else {}
    gen_ids: List[int] = []
    with torch.inference_mode():
        out = lm(inputs_embeds=embeds, use_cache=True, return_dict=True, **kwargs_init)
        logits = lm_head(out.last_hidden_state[:, -1:, :])
        next_token = logits.argmax(dim=-1)
        next_id = int(next_token.item())
        gen_ids.append(next_id)
        if next_id in eos_set:
            return tokenizer.decode(gen_ids, skip_special_tokens=True)
        past = out.past_key_values

        for _ in range(max_new_tokens - 1):
            tok_embed = lm.embed_tokens(next_token)
            tok_per_layer = lm.get_per_layer_inputs(next_token, None) if hasattr(lm, "get_per_layer_inputs") else None
            kwargs_step = {"per_layer_inputs": tok_per_layer} if tok_per_layer is not None else {}
            out = lm(inputs_embeds=tok_embed, per_layer_inputs=tok_per_layer, past_key_values=past, use_cache=True, return_dict=True, **kwargs_step)
            logits = lm_head(out.last_hidden_state[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            next_id = int(next_token.item())
            gen_ids.append(next_id)
            past = out.past_key_values
            if next_id in eos_set:
                break

    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def evaluate_stratum(
    records: List[Dict[str, Any]],
    n_bootstraps: int = 10000,
    seed: int = 2030,
) -> Dict[str, Any]:
    """Compute strict/lenient accuracies and bootstrap CI across 5 conditions for a subset."""
    n = len(records)
    if n == 0:
        return {}

    # Matched condition
    m_strict = [1 if r["conditions"]["matched"]["strict_match"] else 0 for r in records]
    m_lenient = [1 if r["conditions"]["matched"]["lenient_match"] else 0 for r in records]

    # Shuffled condition (mean across 3 assignments per item)
    s_strict = [np.mean([1 if a["strict_match"] else 0 for a in r["conditions"]["shuffled"]]) for r in records]
    s_lenient = [np.mean([1 if a["lenient_match"] else 0 for a in r["conditions"]["shuffled"]]) for r in records]

    # Zero condition
    z_strict = [1 if r["conditions"]["zero"]["strict_match"] else 0 for r in records]
    z_lenient = [1 if r["conditions"]["zero"]["lenient_match"] else 0 for r in records]

    # Random-z condition
    rz_strict = [1 if r["conditions"]["random_z"]["strict_match"] else 0 for r in records]
    rz_lenient = [1 if r["conditions"]["random_z"]["lenient_match"] else 0 for r in records]

    # No-prefix condition
    np_strict = [1 if r["conditions"]["no_prefix"]["strict_match"] else 0 for r in records]
    np_lenient = [1 if r["conditions"]["no_prefix"]["lenient_match"] else 0 for r in records]

    delta_s, ci_s_low, ci_s_high = compute_bootstrap_ci(m_strict, s_strict, n_bootstraps=n_bootstraps, seed=seed)
    delta_l, ci_l_low, ci_l_high = compute_bootstrap_ci(m_lenient, s_lenient, n_bootstraps=n_bootstraps, seed=seed)

    return {
        "item_count": n,
        "acc_matched_strict": float(np.mean(m_strict)),
        "acc_matched_lenient": float(np.mean(m_lenient)),
        "acc_shuffled_strict": float(np.mean(s_strict)),
        "acc_shuffled_lenient": float(np.mean(s_lenient)),
        "delta_strict": delta_s,
        "bootstrap_ci_95_strict": [ci_s_low, ci_s_high],
        "delta_lenient": delta_l,
        "bootstrap_ci_95_lenient": [ci_l_low, ci_l_high],
        "acc_zero_strict": float(np.mean(z_strict)),
        "acc_zero_lenient": float(np.mean(z_lenient)),
        "acc_random_z_strict": float(np.mean(rz_strict)),
        "acc_random_z_lenient": float(np.mean(rz_lenient)),
        "acc_no_prefix_strict": float(np.mean(np_strict)),
        "acc_no_prefix_lenient": float(np.mean(np_lenient)),
    }


def run_phase2_eval(
    zstar_path: str = DEFAULT_ZSTAR_PATH,
    decoder_ckpt_path: str = "",
    control_ckpt_path: str = "",
    items_path: str = DEFAULT_ITEMS_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    base_model_path: str = DEFAULT_BASE_MODEL_PATH,
    out_json_path: str = DEFAULT_OUT_JSON,
    outputs_jsonl_path: str = DEFAULT_OUTPUTS_JSONL,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    device_str: str = "cuda:0",
    seed: int = 2030,
) -> Dict[str, Any]:
    """Execute S5 Phase 2 Gate Evaluation across 3 strata and 5 conditions."""
    print("=" * 80)
    print("S5 Phase 2 Gate Evaluation (3 Strata x 5 Conditions)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Decoder (lambda=1): {decoder_ckpt_path}")
    print(f"Control (lambda=0): {control_ckpt_path}")
    print(f"Held-out Items:     {items_path}")
    print(f"z* Vectors:         {zstar_path}")
    print(f"Device:             {device_str}")
    print("=" * 80, flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    # 1. Load Data
    heldout_indices = load_heldout_indices(items_path)
    assert len(heldout_indices) == 80, f"Expected 80 held-out items, got {len(heldout_indices)}"
    dataset_records = load_dataset_records(dataset_path)

    if not os.path.exists(zstar_path):
        raise FileNotFoundError(f"z* tensor file not found: {zstar_path}")
    zstar_dict = torch.load(zstar_path, map_location="cpu")
    print(f"Loaded z* tensor file from {zstar_path} (Keys: {len(zstar_dict)})")

    # Pre-index z* by integer question ID for fast, robust lookup
    zstar_by_q: Dict[int, Any] = {}
    for k, v in zstar_dict.items():
        k_str = str(k)
        m = re.match(r"^q(\d+)", k_str)
        if m:
            zstar_by_q[int(m.group(1))] = v
        elif isinstance(v, dict) and "meta" in v and "question_idx" in v["meta"]:
            zstar_by_q[int(v["meta"]["question_idx"])] = v
        elif str(k).isdigit():
            zstar_by_q[int(k)] = v

    # Determine convergence status of heldout items
    converged_flags: Dict[int, bool] = {}
    for q in heldout_indices:
        entry = zstar_by_q.get(q)
        if entry is None:
            for k_cand in [f"q{q:04d}", f"q{q:02d}", str(q), q]:
                if k_cand in zstar_dict:
                    entry = zstar_dict[k_cand]
                    break
        assert entry is not None, f"Held-out item {q} not found in z* dict"
        conv = entry.get("converged", None)
        if conv is None:
            conv = bool(entry.get("rho", 1.0) < 0.005)
        converged_flags[q] = bool(conv)

    conv_count = sum(1 for v in converged_flags.values() if v)
    non_conv_count = 80 - conv_count
    print(f"[HELD-OUT] Loaded 80 items: {conv_count} converged, {non_conv_count} non-converged.")

    # 2. Load Canonical Backbone Model & Tokenizer
    from cts.model.canonical_loader import get_canonical_backbone
    print(f"[MODEL-LOAD] Loading canonical backbone on {device_str}...")
    bb = get_canonical_backbone(device_str, dtype=torch.bfloat16)
    tokenizer = bb.tokenizer
    lm = bb.cg.model.language_model

    # 3. Load Decoder (lambda=1.0)
    print(f"[PROJECTOR] Loading primary decoder (lambda=1.0) from {decoder_ckpt_path}...")
    projector_l1 = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True).to(device)
    l1_data = torch.load(decoder_ckpt_path, map_location="cpu")
    l1_sd = l1_data.get("state_dict", l1_data)
    projector_l1.load_state_dict(l1_sd, strict=True)
    projector_l1.eval()

    # 4. Load Control (lambda=0.0) if provided
    projector_l0 = None
    l0_sd = None
    if control_ckpt_path and os.path.exists(control_ckpt_path):
        print(f"[PROJECTOR] Loading control decoder (lambda=0.0) from {control_ckpt_path}...")
        projector_l0 = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True).to(device)
        l0_data = torch.load(control_ckpt_path, map_location="cpu")
        l0_sd = l0_data.get("state_dict", l0_data)
        projector_l0.load_state_dict(l0_sd, strict=True)
        projector_l0.eval()

    # 5. Build Canonical Prompts & Embeddings
    token_ids, slot_start, slot_end, placeholder_id = build_canonical_prefix_prompt(tokenizer, k_latent=64)
    base_prefix_tokens = torch.tensor([token_ids], device=device, dtype=torch.long)
    base_embeds = lm.embed_tokens(base_prefix_tokens)
    base_per_layer = lm.get_per_layer_inputs(base_prefix_tokens, None) if hasattr(lm, "get_per_layer_inputs") else None

    no_prefix_text = "<bos><|turn>user\n<turn|>\n<|turn>model\n"
    no_prefix_tokens = torch.tensor([tokenizer.encode(no_prefix_text, add_special_tokens=False)], device=device, dtype=torch.long)
    no_prefix_embeds = lm.embed_tokens(no_prefix_tokens)
    no_prefix_per_layer = lm.get_per_layer_inputs(no_prefix_tokens, None) if hasattr(lm, "get_per_layer_inputs") else None

    # 6. Evaluate all 80 items
    all_item_records: List[Dict[str, Any]] = []
    outputs_lines: List[str] = []

    print("[EVAL] Starting 5-condition evaluation across 80 held-out items...")
    t0_eval = time.time()

    for i, q in enumerate(heldout_indices):
        item = dataset_records[q]
        solution_text = item.get("solution") or item.get("answer") or ""
        gold_answer = _extract_boxed(solution_text) or extract_answer(solution_text) or solution_text
        is_conv = converged_flags[q]

        entry = zstar_by_q.get(q)
        if entry is None:
            k_candidates = [f"q{q:04d}", f"q{q:02d}", str(q), q]
            entry = next(zstar_dict[k] for k in k_candidates if k in zstar_dict)
        z_tensor = entry["z_star"] if isinstance(entry, dict) and "z_star" in entry else entry["z_1000"]
        z_tensor = z_tensor.to(device=device, dtype=torch.float32)  # [64, 2560]

        # Condition 1: matched
        with torch.no_grad():
            p_m = projector_l1(z_tensor.unsqueeze(0))
            embeds_m = base_embeds.clone()
            embeds_m[:, slot_start:slot_end + 1, :] = p_m
        resp_m = generate_from_embeds(lm, bb.cg.lm_head, embeds_m, tokenizer, per_layer_inputs=base_per_layer)
        boxed_m = _extract_boxed(resp_m) or ""
        lenient_m = lenient_extract(resp_m) or ""
        match_strict_m = bool(answers_match(boxed_m, gold_answer) if boxed_m else False)
        match_lenient_m = bool(answers_match(lenient_m, gold_answer) if lenient_m else False)

        # Condition 2: shuffled (3 round-robin cycles: (i+1)%80, (i+2)%80, (i+3)%80)
        shuffled_list: List[Dict[str, Any]] = []
        for shift in (1, 2, 3):
            j = (i + shift) % 80
            q_j = heldout_indices[j]
            entry_j = zstar_by_q.get(q_j)
            if entry_j is None:
                k_cand_j = [f"q{q_j:04d}", f"q{q_j:02d}", str(q_j), q_j]
                entry_j = next(zstar_dict[k] for k in k_cand_j if k in zstar_dict)
            z_j = entry_j["z_star"] if isinstance(entry_j, dict) and "z_star" in entry_j else entry_j["z_1000"]
            z_j = z_j.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                p_s = projector_l1(z_j.unsqueeze(0))
                embeds_s = base_embeds.clone()
                embeds_s[:, slot_start:slot_end + 1, :] = p_s
            resp_s = generate_from_embeds(lm, bb.cg.lm_head, embeds_s, tokenizer, per_layer_inputs=base_per_layer)
            boxed_s = _extract_boxed(resp_s) or ""
            lenient_s = lenient_extract(resp_s) or ""
            match_strict_s = bool(answers_match(boxed_s, gold_answer) if boxed_s else False)
            match_lenient_s = bool(answers_match(lenient_s, gold_answer) if lenient_s else False)

            shuf_rec = {
                "shift": shift,
                "donor_q": q_j,
                "prediction": resp_s,
                "boxed_extract": boxed_s,
                "boxed_match": match_strict_s,
                "lenient_extract": lenient_s,
                "lenient_match": match_lenient_s,
                "strict_match": match_strict_s,
            }
            shuffled_list.append(shuf_rec)
            outputs_lines.append(json.dumps({
                "q_idx": q, "donor_q": q_j, "condition": f"shuffled_{shift}", "converged": is_conv,
                "gold": gold_answer, "prediction": resp_s, "boxed": boxed_s, "lenient": lenient_s,
                "strict_match": match_strict_s, "lenient_match": match_lenient_s
            }))

        # Condition 3: zero
        with torch.no_grad():
            z_zero = torch.zeros_like(z_tensor)
            p_z = projector_l1(z_zero.unsqueeze(0))
            embeds_z = base_embeds.clone()
            embeds_z[:, slot_start:slot_end + 1, :] = p_z
        resp_z = generate_from_embeds(lm, bb.cg.lm_head, embeds_z, tokenizer, per_layer_inputs=base_per_layer)
        boxed_z = _extract_boxed(resp_z) or ""
        lenient_z = lenient_extract(resp_z) or ""
        match_strict_z = bool(answers_match(boxed_z, gold_answer) if boxed_z else False)
        match_lenient_z = bool(answers_match(lenient_z, gold_answer) if lenient_z else False)

        # Condition 4: random-z (matching norm)
        with torch.no_grad():
            norm_z = torch.norm(z_tensor)
            rand_z = torch.randn_like(z_tensor)
            rand_z = (rand_z / torch.norm(rand_z)) * norm_z
            p_rz = projector_l1(rand_z.unsqueeze(0))
            embeds_rz = base_embeds.clone()
            embeds_rz[:, slot_start:slot_end + 1, :] = p_rz
        resp_rz = generate_from_embeds(lm, bb.cg.lm_head, embeds_rz, tokenizer, per_layer_inputs=base_per_layer)
        boxed_rz = _extract_boxed(resp_rz) or ""
        lenient_rz = lenient_extract(resp_rz) or ""
        match_strict_rz = bool(answers_match(boxed_rz, gold_answer) if boxed_rz else False)
        match_lenient_rz = bool(answers_match(lenient_rz, gold_answer) if lenient_rz else False)

        # Condition 5: no_prefix
        resp_np = generate_from_embeds(lm, bb.cg.lm_head, no_prefix_embeds, tokenizer, per_layer_inputs=no_prefix_per_layer)
        boxed_np = _extract_boxed(resp_np) or ""
        lenient_np = lenient_extract(resp_np) or ""
        match_strict_np = bool(answers_match(boxed_np, gold_answer) if boxed_np else False)
        match_lenient_np = bool(answers_match(lenient_np, gold_answer) if lenient_np else False)

        # Control condition: lambda=0 (matched)
        resp_c = ""
        boxed_c = ""
        lenient_c = ""
        match_strict_c = False
        match_lenient_c = False
        if projector_l0 is not None:
            with torch.no_grad():
                p_c = projector_l0(z_tensor.unsqueeze(0))
                embeds_c = base_embeds.clone()
                embeds_c[:, slot_start:slot_end + 1, :] = p_c
            resp_c = generate_from_embeds(lm, bb.cg.lm_head, embeds_c, tokenizer, per_layer_inputs=base_per_layer)
            boxed_c = _extract_boxed(resp_c) or ""
            lenient_c = lenient_extract(resp_c) or ""
            match_strict_c = bool(answers_match(boxed_c, gold_answer) if boxed_c else False)
            match_lenient_c = bool(answers_match(lenient_c, gold_answer) if lenient_c else False)

        item_rec = {
            "q_idx": q,
            "heldout_pos": i,
            "converged": is_conv,
            "gold": gold_answer,
            "conditions": {
                "matched": {
                    "prediction": resp_m, "boxed_extract": boxed_m, "lenient_extract": lenient_m,
                    "strict_match": match_strict_m, "lenient_match": match_lenient_m
                },
                "shuffled": shuffled_list,
                "zero": {
                    "prediction": resp_z, "boxed_extract": boxed_z, "lenient_extract": lenient_z,
                    "strict_match": match_strict_z, "lenient_match": match_lenient_z
                },
                "random_z": {
                    "prediction": resp_rz, "boxed_extract": boxed_rz, "lenient_extract": lenient_rz,
                    "strict_match": match_strict_rz, "lenient_match": match_lenient_rz
                },
                "no_prefix": {
                    "prediction": resp_np, "boxed_extract": boxed_np, "lenient_extract": lenient_np,
                    "strict_match": match_strict_np, "lenient_match": match_lenient_np
                },
                "control_matched": {
                    "prediction": resp_c, "boxed_extract": boxed_c, "lenient_extract": lenient_c,
                    "strict_match": match_strict_c, "lenient_match": match_lenient_c
                }
            }
        }
        all_item_records.append(item_rec)

        outputs_lines.append(json.dumps({
            "q_idx": q, "condition": "matched", "converged": is_conv,
            "gold": gold_answer, "prediction": resp_m, "boxed": boxed_m, "lenient": lenient_m,
            "strict_match": match_strict_m, "lenient_match": match_lenient_m
        }))
        outputs_lines.append(json.dumps({
            "q_idx": q, "condition": "zero", "converged": is_conv,
            "gold": gold_answer, "prediction": resp_z, "boxed": boxed_z, "lenient": lenient_z,
            "strict_match": match_strict_z, "lenient_match": match_lenient_z
        }))
        outputs_lines.append(json.dumps({
            "q_idx": q, "condition": "random_z", "converged": is_conv,
            "gold": gold_answer, "prediction": resp_rz, "boxed": boxed_rz, "lenient": lenient_rz,
            "strict_match": match_strict_rz, "lenient_match": match_lenient_rz
        }))
        outputs_lines.append(json.dumps({
            "q_idx": q, "condition": "no_prefix", "converged": is_conv,
            "gold": gold_answer, "prediction": resp_np, "boxed": boxed_np, "lenient": lenient_np,
            "strict_match": match_strict_np, "lenient_match": match_lenient_np
        }))
        if projector_l0 is not None:
            outputs_lines.append(json.dumps({
                "q_idx": q, "condition": "control_matched", "converged": is_conv,
                "gold": gold_answer, "prediction": resp_c, "boxed": boxed_c, "lenient": lenient_c,
                "strict_match": match_strict_c, "lenient_match": match_lenient_c
            }))

        if (i + 1) % 10 == 0 or (i + 1) == 80:
            cur_acc_m_s = np.mean([1 if r["conditions"]["matched"]["strict_match"] else 0 for r in all_item_records])
            cur_acc_m_l = np.mean([1 if r["conditions"]["matched"]["lenient_match"] else 0 for r in all_item_records])
            print(f"[EVAL Progress {i+1:02d}/80] acc_matched strict={cur_acc_m_s*100:.1f}%, lenient={cur_acc_m_l*100:.1f}%", flush=True)

    elapsed_eval = time.time() - t0_eval
    print(f"[EVAL] Completed generation across 80 items in {elapsed_eval:.2f}s ({elapsed_eval/60:.2f}m).")

    # 7. Write outputs JSONL
    os.makedirs(os.path.dirname(outputs_jsonl_path) or ".", exist_ok=True)
    with open(outputs_jsonl_path, "w", encoding="utf-8") as f:
        for line in outputs_lines:
            f.write(line + "\n")
    sha_outputs = append_to_manifest(manifest_path, outputs_jsonl_path)
    print(f"[OUTPUTS] Saved sample-level outputs to {outputs_jsonl_path} (SHA: {sha_outputs})")

    # 8. Compute Strata Metrics
    records_all = all_item_records
    records_conv = [r for r in all_item_records if r["converged"]]
    records_nonconv = [r for r in all_item_records if not r["converged"]]

    strata_all = evaluate_stratum(records_all, n_bootstraps=10000, seed=seed)
    strata_conv = evaluate_stratum(records_conv, n_bootstraps=10000, seed=seed)
    strata_nonconv = evaluate_stratum(records_nonconv, n_bootstraps=10000, seed=seed)

    # 9. Criterion II-b: Control Comparison (lambda=1 vs lambda=0)
    control_metrics: Dict[str, Any] = {}
    if projector_l0 is not None:
        l1_m_s = [1 if r["conditions"]["matched"]["strict_match"] else 0 for r in all_item_records]
        l0_m_s = [1 if r["conditions"]["control_matched"]["strict_match"] else 0 for r in all_item_records]
        l1_m_l = [1 if r["conditions"]["matched"]["lenient_match"] else 0 for r in all_item_records]
        l0_m_l = [1 if r["conditions"]["control_matched"]["lenient_match"] else 0 for r in all_item_records]

        diff_s, ci_diff_s_low, ci_diff_s_high = compute_bootstrap_ci(l1_m_s, l0_m_s, n_bootstraps=10000, seed=seed)
        diff_l, ci_diff_l_low, ci_diff_l_high = compute_bootstrap_ci(l1_m_l, l0_m_l, n_bootstraps=10000, seed=seed)

        control_metrics = {
            "acc_matched_strict_lambda1": float(np.mean(l1_m_s)),
            "acc_matched_strict_lambda0": float(np.mean(l0_m_s)),
            "delta_lambda1_minus_lambda0_strict": diff_s,
            "bootstrap_ci_95_strict": [ci_diff_s_low, ci_diff_s_high],
            "acc_matched_lenient_lambda1": float(np.mean(l1_m_l)),
            "acc_matched_lenient_lambda0": float(np.mean(l0_m_l)),
            "delta_lambda1_minus_lambda0_lenient": diff_l,
            "bootstrap_ci_95_lenient": [ci_diff_l_low, ci_diff_l_high],
        }

    # 10. Criterion II-c: Format / Content NLL Decomposition on held-out set
    print("[NLL] Computing Format vs Content NLL on held-out solutions...")
    nll_fmt_l1_vals = []
    nll_cnt_l1_vals = []
    nll_fmt_l0_vals = []
    nll_cnt_l0_vals = []

    target_start = len(token_ids)
    with torch.no_grad():
        for q in heldout_indices:
            sol_text = dataset_records[q]["solution"]
            enc_sol = tokenizer(sol_text, return_offsets_mapping=True, add_special_tokens=False)
            sol_ids = enc_sol["input_ids"][:512 - target_start - 1] + [tokenizer.eos_token_id]
            offsets = enc_sol.get("offset_mapping", [])[:512 - target_start - 1]
            span = find_boxed_char_span(sol_text)
            is_content = [bool(span and s < span[1] and e > span[0]) for (s, e) in offsets]
            is_content.append(False)

            full_seq = torch.cat([base_prefix_tokens[0], torch.tensor(sol_ids, device=device, dtype=torch.long)])
            input_batch = full_seq.unsqueeze(0)
            target_batch = torch.full_like(input_batch, -100)
            target_batch[:, target_start:] = input_batch[:, target_start:]
            content_mask = torch.zeros_like(target_batch, dtype=torch.bool)
            content_mask[:, target_start:target_start + len(is_content)] = torch.tensor(is_content, device=device, dtype=torch.bool)

            entry = zstar_by_q.get(q)
            if entry is None:
                k_cand = [f"q{q:04d}", f"q{q:02d}", str(q), q]
                entry = next(zstar_dict[k] for k in k_cand if k in zstar_dict)
            z_t = entry["z_star"] if isinstance(entry, dict) and "z_star" in entry else entry["z_1000"]
            z_t = z_t.to(device=device, dtype=torch.float32).unsqueeze(0)

            per_layer = lm.get_per_layer_inputs(input_batch, None) if hasattr(lm, "get_per_layer_inputs") else None
            kwargs_nll = {"per_layer_inputs": per_layer} if per_layer is not None else {}

            # lambda=1
            b_emb = lm.embed_tokens(input_batch).clone()
            b_emb[:, slot_start:slot_end + 1, :] = projector_l1(z_t)
            out_l1 = lm(inputs_embeds=b_emb, return_dict=True, **kwargs_nll)
            logits_l1 = bb.cg.lm_head(out_l1.last_hidden_state)
            shift_logits = logits_l1[:, target_start - 1:-1, :].contiguous()
            shift_targets = target_batch[:, target_start:].contiguous()
            shift_content = content_mask[:, target_start:].contiguous()
            loss_toks = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), reduction="none").view_as(shift_targets)
            valid = (shift_targets != -100)
            if (valid & (~shift_content)).any():
                nll_fmt_l1_vals.append(float(loss_toks[valid & (~shift_content)].mean().item()))
            if (valid & shift_content).any():
                nll_cnt_l1_vals.append(float(loss_toks[valid & shift_content].mean().item()))

            # lambda=0
            if projector_l0 is not None:
                b_emb_0 = lm.embed_tokens(input_batch).clone()
                b_emb_0[:, slot_start:slot_end + 1, :] = projector_l0(z_t)
                out_l0 = lm(inputs_embeds=b_emb_0, return_dict=True, **kwargs_nll)
                logits_l0 = bb.cg.lm_head(out_l0.last_hidden_state)
                shift_logits_0 = logits_l0[:, target_start - 1:-1, :].contiguous()
                loss_toks_0 = F.cross_entropy(shift_logits_0.view(-1, shift_logits_0.size(-1)), shift_targets.view(-1), reduction="none").view_as(shift_targets)
                if (valid & (~shift_content)).any():
                    nll_fmt_l0_vals.append(float(loss_toks_0[valid & (~shift_content)].mean().item()))
                if (valid & shift_content).any():
                    nll_cnt_l0_vals.append(float(loss_toks_0[valid & shift_content].mean().item()))

    nll_metrics = {
        "nll_format_lambda1": float(np.mean(nll_fmt_l1_vals)) if nll_fmt_l1_vals else 0.0,
        "nll_content_lambda1": float(np.mean(nll_cnt_l1_vals)) if nll_cnt_l1_vals else 0.0,
        "nll_format_lambda0": float(np.mean(nll_fmt_l0_vals)) if nll_fmt_l0_vals else 0.0,
        "nll_content_lambda0": float(np.mean(nll_cnt_l0_vals)) if nll_cnt_l0_vals else 0.0,
    }

    # 11. Criterion II-d: w_proj Weight Norm & std(W)
    w_orig_l1 = projector_l1.w_proj.parametrizations.weight.original if hasattr(projector_l1.w_proj, "parametrizations") else projector_l1.w_proj.weight
    frob_l1 = float(torch.norm(w_orig_l1).item())
    std_l1 = float(torch.std(w_orig_l1).item())

    weights_analysis: Dict[str, Any] = {
        "b1_initial_frob_norm": B1_INITIAL_FROB,
        "w_proj_frob_norm_lambda1": frob_l1,
        "w_proj_std_lambda1": std_l1,
        "delta_frob_norm_lambda1": float(frob_l1 - B1_INITIAL_FROB),
    }
    if projector_l0 is not None:
        w_orig_l0 = projector_l0.w_proj.parametrizations.weight.original if hasattr(projector_l0.w_proj, "parametrizations") else projector_l0.w_proj.weight
        frob_l0 = float(torch.norm(w_orig_l0).item())
        std_l0 = float(torch.std(w_orig_l0).item())
        weights_analysis["w_proj_frob_norm_lambda0"] = frob_l0
        weights_analysis["w_proj_std_lambda0"] = std_l0
        weights_analysis["delta_frob_norm_lambda0"] = float(frob_l0 - B1_INITIAL_FROB)

    # 12. Final Gate Summary JSON (Strictly NO judgment words!)
    gate_summary = {
        "metadata": {
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "directive": "EXE-260923-29 (System 2 / Ledger #333)",
            "preregistration_version": "PREREG_S5_v5.md",
            "decoder_ckpt_path": decoder_ckpt_path,
            "control_ckpt_path": control_ckpt_path,
            "zstar_path": zstar_path,
            "items_path": items_path,
            "device": device_str,
            "seed": seed,
            "total_heldout_items": 80,
            "converged_items_count": conv_count,
            "non_converged_items_count": non_conv_count,
            "bootstrap_iterations": 10000,
            "elapsed_eval_seconds": elapsed_eval,
        },
        "strata": {
            "all_80": strata_all,
            "converged_71": strata_conv,
            "non_converged_9": strata_nonconv,
        },
        "criterion_2b_control": control_metrics,
        "criterion_2c_nll_decomposition": nll_metrics,
        "criterion_2d_weights_analysis": weights_analysis,
    }

    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(gate_summary, f, indent=2)

    sha_gate = append_to_manifest(manifest_path, out_json_path)
    print(f"[GATE-SUMMARY] Saved Phase 2 gate JSON to {out_json_path} (SHA: {sha_gate})")

    return {
        "status": "PHASE2_EVAL_COMPLETED",
        "out_json": out_json_path,
        "sha256": sha_gate,
        "summary": gate_summary,
    }


def run_dry_run(output_dir: str, raw_dir: str, manifest_path: str) -> Dict[str, Any]:
    print("[DRY-RUN] Executing Phase 0-5 Gate Evaluation Dry-Run (CPU, 0 GPU calls)...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    tokenizer = load_tokenizer()
    layout_res = dump_eval_layout_5cond(raw_dir, tokenizer, k_latent=64)

    rng = np.random.default_rng(42)
    matched_correct = list(rng.binomial(1, 0.45, size=80))
    shuffled_correct = list(rng.binomial(1, 0.35, size=80))
    zero_correct = list(rng.binomial(1, 0.20, size=80))
    random_z_correct = list(rng.binomial(1, 0.22, size=80))
    no_prefix_correct = list(rng.binomial(1, 0.65, size=80))

    delta_obs, ci_lower, ci_upper = compute_bootstrap_ci(
        matched_correct, shuffled_correct, n_bootstraps=1000, seed=42
    )

    results = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_heldout_items": 80,
        "shuffled_evaluations_count": 240,
        "acc_matched_strict": float(np.mean(matched_correct)),
        "acc_shuffled_strict": float(np.mean(shuffled_correct)),
        "acc_zero_strict": float(np.mean(zero_correct)),
        "acc_random_z_strict": float(np.mean(random_z_correct)),
        "acc_no_prefix_strict": float(np.mean(no_prefix_correct)),
        "delta_acc": delta_obs,
        "bootstrap_ci_95_lower": ci_lower,
        "bootstrap_ci_95_upper": ci_upper,
        "bootstrap_iterations": 1000,
    }

    out_json = os.path.join(output_dir, "gate_dryrun.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    sha_json = append_to_manifest(manifest_path, out_json)

    return {
        "status": "DRY_RUN_COMPLETED",
        "output_json": out_json,
        "sha256": sha_json,
        "results": results,
        "layout": layout_res,
    }


def main():
    parser = argparse.ArgumentParser(description="S5 Held-Out Gate Evaluation (EXE-260923-28 rev2 / EXE-260923-29).")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="Execute Step 0 Dry-Run on CPU")
    parser.add_argument("--harness-anchor", "--harness_anchor", action="store_true", help="Execute 10-item text_only harness anchor")
    parser.add_argument("--decoder", "--ckpt", type=str, default="", help="Path to lambda=1 decoder checkpoint pt to evaluate")
    parser.add_argument("--control", type=str, default="", help="Path to lambda=0 control decoder checkpoint pt")
    parser.add_argument("--b1-ckpt", "--b1_ckpt", type=str, default=DEFAULT_B1_CKPT_PATH, help="Path to B1 checkpoint")
    parser.add_argument("--base_model", "--base-model", type=str, default=DEFAULT_BASE_MODEL_PATH, help="Path to base Gemma snapshot")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Results output directory")
    parser.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_DIR, help="Raw logs directory")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST_PATH, help="Output manifest SHA file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device (cuda:0)")
    parser.add_argument("--zstar", "--zstar_path", type=str, default=DEFAULT_ZSTAR_PATH, help="Path to zstar_s5_400.pt")
    parser.add_argument("--items", "--items_json", type=str, default=DEFAULT_ITEMS_PATH, help="Path to heldout 80 items json")
    parser.add_argument("--dataset", "--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Path to math_train_prompts_5000.jsonl")
    parser.add_argument("--out", "--output", type=str, default=DEFAULT_OUT_JSON, help="Path to output phase2_gate.json")
    parser.add_argument("--outputs", type=str, default=DEFAULT_OUTPUTS_JSONL, help="Path to output phase2_outputs.jsonl")
    parser.add_argument("--seed", type=int, default=2030, help="Random seed (default 2030)")
    args = parser.parse_args()

    if args.dry_run:
        res = run_dry_run(args.output_dir, args.raw_dir, args.manifest)
        print(json.dumps(res, indent=2))
        return

    if args.harness_anchor:
        res = run_harness_anchor(
            b1_ckpt_path=args.b1_ckpt,
            dataset_path=args.dataset,
            base_model_path=args.base_model,
            device_str=args.device,
            raw_dir=args.raw_dir,
            manifest_path=args.manifest,
        )
        print(json.dumps(res, indent=2))
        return

    # Full Evaluation on Checkpoint
    if not args.decoder:
        print("Notice: --decoder required for full evaluation. Use --dry-run or --harness-anchor for pre-eval validation.")
        return

    res = run_phase2_eval(
        zstar_path=args.zstar,
        decoder_ckpt_path=args.decoder,
        control_ckpt_path=args.control,
        items_path=args.items,
        dataset_path=args.dataset,
        base_model_path=args.base_model,
        out_json_path=args.out,
        outputs_jsonl_path=args.outputs,
        manifest_path=args.manifest,
        device_str=args.device,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

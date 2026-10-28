#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p6_greedy_equiv_check_v2.py - CTS S7 P6 Greedy Equivalence Checker v2
Directive: TASK-0018 Section 1 / AUDIT-0017 FIX F2

Purpose:
  Verifies token-level mathematical equivalence of manual greedy KV generation loop
  (v36b `generate_greedy_kv`) against the unassailable baseline:
  **Exact full-sequence non-KV forward loop recomputing `get_per_layer_inputs(input_ids)` at every step**
  (identical to training `scripts/s7_train_decoder_v36.py:869`).

Test Items & Scope:
  - Training items: Q#0014, Q#0029
  - Max new tokens: 32 tokens
  - Condition C (text-only prompt)
  - Condition A (prefix-augmented prompt with test decoder `results_s5/decoder_A_v36_step0500.pt`)
  - Reports match status and first mismatch position with top-2 tokens and logit difference.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple, Set
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"
SCRIPTS_DIR = os.path.join(BASE_S5_DIR, "scripts")
LOCAL_SCRIPTS = os.path.dirname(os.path.abspath(__file__))

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, SCRIPTS_DIR, LOCAL_SCRIPTS]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from s7_eval_gate_p6_v36b import (
        load_tokenizer,
        load_pure_gemma_model,
        load_dataset_records,
        build_canonical_prefix_prompt,
        load_tensor_by_key,
        generate_greedy_kv,
        PrefixProjector,
        compute_sha256,
    )
except ImportError:
    from scripts.s7_eval_gate_p6_v36b import (
        load_tokenizer,
        load_pure_gemma_model,
        load_dataset_records,
        build_canonical_prefix_prompt,
        load_tensor_by_key,
        generate_greedy_kv,
        PrefixProjector,
        compute_sha256,
    )


def generate_greedy_no_kv_baseline(
    lm_model: Any,
    lm_head: Any,
    embed_tokens: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    eos_set: Set[int],
    max_new_tokens: int = 32,
) -> List[int]:
    """Exact baseline greedy autoregressive loop without KV caching.
    
    Re-evaluates the entire concatenated sequence at every step, recomputing
    per-layer inputs via `lm_model.get_per_layer_inputs(curr_ids, None)`
    (strictly following `s7_train_decoder_v36.py:869`).
    Uses `logits.argmax(dim=-1)` identical to standard greedy generation.
    """
    curr_embeds = inputs_embeds.clone()
    curr_ids = input_ids.clone()
    gen_ids: List[int] = []

    with torch.inference_mode():
        for step in range(max_new_tokens):
            per_layer = lm_model.get_per_layer_inputs(curr_ids, None) if hasattr(lm_model, "get_per_layer_inputs") else None
            kwargs = {"per_layer_inputs": per_layer} if per_layer is not None else {}
            out = lm_model(inputs_embeds=curr_embeds, use_cache=False, return_dict=True, **kwargs)
            logits = lm_head(out.last_hidden_state[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            next_id = int(next_token.item())
            gen_ids.append(next_id)

            if next_id in eos_set:
                break

            tok_embed = embed_tokens(next_token)
            curr_embeds = torch.cat([curr_embeds, tok_embed], dim=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

    return gen_ids


def diagnose_step_logits(
    lm_model: Any,
    lm_head: Any,
    embed_tokens: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    prior_tokens: List[int],
    eos_set: Set[int],
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute step-level logits and top-2 candidates for both Non-KV and KV at a specific step."""
    # 1. Non-KV forward at this step
    if prior_tokens:
        tok_tensor = torch.tensor([prior_tokens], dtype=torch.long, device=device)
        p_embed = embed_tokens(tok_tensor)
        seq_embeds = torch.cat([inputs_embeds, p_embed], dim=1)
        seq_ids = torch.cat([input_ids, tok_tensor], dim=1)
    else:
        seq_embeds = inputs_embeds.clone()
        seq_ids = input_ids.clone()

    with torch.inference_mode():
        per_layer = lm_model.get_per_layer_inputs(seq_ids, None) if hasattr(lm_model, "get_per_layer_inputs") else None
        kwargs = {"per_layer_inputs": per_layer} if per_layer is not None else {}
        out_nokv = lm_model(inputs_embeds=seq_embeds, use_cache=False, return_dict=True, **kwargs)
        logits_nokv = lm_head(out_nokv.last_hidden_state[:, -1, :]).float()[0]
        topk_nokv = torch.topk(logits_nokv, k=2)

        nokv_diag = {
            "top1_id": int(topk_nokv.indices[0].item()),
            "top1_val": float(topk_nokv.values[0].item()),
            "top2_id": int(topk_nokv.indices[1].item()),
            "top2_val": float(topk_nokv.values[1].item()),
            "delta": float((topk_nokv.values[0] - topk_nokv.values[1]).item()),
            "argmax_id": int(logits_nokv.argmax().item()),
        }

    # 2. KV forward up to this step
    with torch.inference_mode():
        per_layer_init = lm_model.get_per_layer_inputs(input_ids, None) if hasattr(lm_model, "get_per_layer_inputs") else None
        kwargs_init = {"per_layer_inputs": per_layer_init} if per_layer_init is not None else {}
        out_kv = lm_model(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True, **kwargs_init)
        logits_kv = lm_head(out_kv.last_hidden_state[:, -1, :]).float()[0]
        past = out_kv.past_key_values

        for tid in prior_tokens:
            tok_t = torch.tensor([[tid]], dtype=torch.long, device=device)
            t_emb = embed_tokens(tok_t)
            t_ple = lm_model.get_per_layer_inputs(tok_t, None) if hasattr(lm_model, "get_per_layer_inputs") else None
            kw_step = {"per_layer_inputs": t_ple} if t_ple is not None else {}
            out_kv = lm_model(inputs_embeds=t_emb, past_key_values=past, use_cache=True, return_dict=True, **kw_step)
            logits_kv = lm_head(out_kv.last_hidden_state[:, -1, :]).float()[0]
            past = out_kv.past_key_values

        topk_kv = torch.topk(logits_kv, k=2)
        kv_diag = {
            "top1_id": int(topk_kv.indices[0].item()),
            "top1_val": float(topk_kv.values[0].item()),
            "top2_id": int(topk_kv.indices[1].item()),
            "top2_val": float(topk_kv.values[1].item()),
            "delta": float((topk_kv.values[0] - topk_kv.values[1]).item()),
            "argmax_id": int(logits_kv.argmax().item()),
            "abs_logit_diff": float((logits_nokv[topk_nokv.indices[0]] - logits_kv[topk_nokv.indices[0]]).abs().item()),
        }

    return nokv_diag, kv_diag


def compare_and_diagnose(
    ref_tokens: List[int],
    kv_tokens: List[int],
    lm_model: Any,
    lm_head: Any,
    embed_tokens: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    eos_set: Set[int],
    device: torch.device,
    tokenizer: Any,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Compare two token ID lists. If mismatch, compute step-level diagnostic logits."""
    if ref_tokens == kv_tokens:
        return True, None

    min_len = min(len(ref_tokens), len(kv_tokens))
    first_diff = min_len
    for i in range(min_len):
        if ref_tokens[i] != kv_tokens[i]:
            first_diff = i
            break

    # Prior tokens before the mismatch
    prior_tokens = ref_tokens[:first_diff]
    ref_diag, kv_diag = diagnose_step_logits(
        lm_model=lm_model,
        lm_head=lm_head,
        embed_tokens=embed_tokens,
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        prior_tokens=prior_tokens,
        eos_set=eos_set,
        device=device,
    )

    ref_tok = ref_tokens[first_diff] if first_diff < len(ref_tokens) else None
    kv_tok = kv_tokens[first_diff] if first_diff < len(kv_tokens) else None

    diff_detail = {
        "mismatch_index": first_diff,
        "ref_token_id": ref_tok,
        "kv_token_id": kv_tok,
        "ref_decoded": repr(tokenizer.decode([ref_tok])) if ref_tok is not None else "EOF",
        "kv_decoded": repr(tokenizer.decode([kv_tok])) if kv_tok is not None else "EOF",
        "ref_diag": ref_diag,
        "kv_diag": kv_diag,
        "ref_length": len(ref_tokens),
        "kv_length": len(kv_tokens),
    }
    return False, diff_detail


def main():
    parser = argparse.ArgumentParser(description="P6 Greedy Equivalence Re-verification v2 (TASK-0018 / AUDIT-0017 F2)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:0)")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens to generate (default: 32)")
    parser.add_argument("--decoder_ckpt", type=str, default=None, help="Decoder checkpoint path for Condition A")
    parser.add_argument("--zstar", type=str, default=None, help="z* path for Condition A")
    parser.add_argument("--out_json", type=str, default=None, help="Optional output JSON path for results")
    args = parser.parse_args()

    device_str = args.device
    device = torch.device(device_str)
    max_tokens = args.max_new_tokens

    print("=" * 80)
    print("CTS S7 P6 Greedy Equivalence Check v2 (AUDIT-0017 FIX F2)")
    print(f"Device: {device_str} | Max New Tokens: {max_tokens}")
    print("Reference: Full Non-KV Re-forward with per_layer_inputs = get_per_layer_inputs(curr_ids)")
    print("Target:    v36b generate_greedy_kv (KV-cache autoregressive loop)")
    print("=" * 80, flush=True)

    # 1. Target Items (Q#14, Q#29) per TASK-0018
    items_to_test = [14, 29]
    print(f"Target Items: Q#{items_to_test[0]:04d}, Q#{items_to_test[1]:04d}")

    # 2. Load Tokenizer & EOS
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()
    eos_set = {tokenizer.eos_token_id}
    for special in ["<turn|>", "<end_of_turn>"]:
        t_ids = tokenizer.encode(special, add_special_tokens=False)
        if t_ids:
            eos_set.update(t_ids)

    # 3. Load Pure Gemma Base Model
    print("Loading Pure Gemma Base (58c4d7b5)...")
    pure_gemma, lm_model, lm_head, embed_tokens = load_pure_gemma_model(device_str)

    # 4. Load Dataset Records
    dataset_records = load_dataset_records()

    # 5. Checkpoint & Projector Setup for Condition A
    ckpt_path = args.decoder_ckpt or os.path.join(BASE_S5_DIR, "results_s5", "decoder_A_v36_step0500.pt")
    if not os.path.exists(ckpt_path):
        alt_ckpt = os.path.join(BASE_S5_DIR, "results_s7", "decoder_A_v36.pt")
        if os.path.exists(alt_ckpt):
            ckpt_path = alt_ckpt

    projector = None
    if os.path.exists(ckpt_path):
        print(f"Loading PrefixProjector from {ckpt_path}...")
        projector = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True).to(device)
        ckpt_data = torch.load(ckpt_path, map_location=device_str)
        if "state_dict" in ckpt_data:
            projector.load_state_dict(ckpt_data["state_dict"])
        else:
            projector.load_state_dict(ckpt_data)
        projector.eval()
        for p in projector.parameters():
            p.requires_grad = False
    else:
        raise FileNotFoundError(f"Decoder checkpoint not found at {ckpt_path}")

    # 6. z* Source Setup for Condition A
    z_path = args.zstar or os.path.join(BASE_S5_DIR, "results_s7", "zstar_train1000_P4.pt")
    if not os.path.exists(z_path):
        raise FileNotFoundError(f"z* source not found at {z_path}")
    print(f"Loading z* tensors from {z_path}...")
    z_source_dict = torch.load(z_path, map_location="cpu")

    results: List[Dict[str, Any]] = []

    # =========================================================================
    # Test Condition C (Text-Only): Non-KV Full Forward vs generate_greedy_kv
    # =========================================================================
    print("\n" + "=" * 70)
    print("Testing Condition C (Text-Only, Prompt Q#0014, Q#0029)")
    print("=" * 70)

    for qid in items_to_test:
        prob_text = dataset_records[qid]["prompt"]
        messages = [{"role": "user", "content": prob_text}]
        templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(templated_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        inputs_embeds = embed_tokens(input_tensor)

        # 1. Reference: Full Non-KV Re-forward loop with get_per_layer_inputs(curr_ids)
        t0_ref = time.time()
        ref_tokens = generate_greedy_no_kv_baseline(
            lm_model=lm_model,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            inputs_embeds=inputs_embeds,
            input_ids=input_tensor,
            eos_set=eos_set,
            max_new_tokens=max_tokens,
        )
        t_ref = time.time() - t0_ref

        # 2. KV Generation: v36b generate_greedy_kv
        t0_kv = time.time()
        kv_tokens = generate_greedy_kv(
            lm_model=lm_model,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            inputs_embeds=inputs_embeds,
            input_ids=input_tensor,
            eos_set=eos_set,
            max_new_tokens=max_tokens,
        )
        t_kv = time.time() - t0_kv

        matched, diff_info = compare_and_diagnose(
            ref_tokens=ref_tokens,
            kv_tokens=kv_tokens,
            lm_model=lm_model,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            inputs_embeds=inputs_embeds,
            input_ids=input_tensor,
            eos_set=eos_set,
            device=device,
            tokenizer=tokenizer,
        )

        res_entry = {
            "condition": "C",
            "item_id": qid,
            "ref_method": "Non-KV Full Re-forward Loop",
            "matched": matched,
            "diff_info": diff_info,
            "ref_tokens": ref_tokens,
            "kv_tokens": kv_tokens,
            "t_ref_sec": t_ref,
            "t_kv_sec": t_kv,
        }
        results.append(res_entry)

        status_str = "MATCH" if matched else "MISMATCH"
        print(f"[Condition C | Q#{qid:04d}] {status_str} ({len(kv_tokens)}/{max_tokens} tokens, Ref: {t_ref:.1f}s, KV: {t_kv:.1f}s)")
        if not matched and diff_info:
            d = diff_info
            r_d = d["ref_diag"]
            k_d = d["kv_diag"]
            print(f"  First mismatch at position {d['mismatch_index']}:")
            print(f"    Ref: ID={d['ref_token_id']} ({d['ref_decoded']}), Top1={r_d['top1_val']:.4f} (ID={r_d['top1_id']}), Top2={r_d['top2_val']:.4f} (ID={r_d['top2_id']}), Δ={r_d['delta']:.4f}")
            print(f"    KV : ID={d['kv_token_id']} ({d['kv_decoded']}), Top1={k_d['top1_val']:.4f} (ID={k_d['top1_id']}), Top2={k_d['top2_val']:.4f} (ID={k_d['top2_id']}), Δ={k_d['delta']:.4f}")

    # =========================================================================
    # Test Condition A (Prefix-Augmented): Non-KV Full Forward vs generate_greedy_kv
    # =========================================================================
    print("\n" + "=" * 70)
    print("Testing Condition A (Prefix-Augmented, Prompt Q#0014, Q#0029)")
    print("=" * 70)

    for qid in items_to_test:
        prob_text = dataset_records[qid]["prompt"]
        token_ids, slot_start, slot_end, placeholder_id = build_canonical_prefix_prompt(
            tokenizer, prob_text=prob_text, k_latent=64
        )
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        base_embeds = embed_tokens(input_tensor)

        z_raw = load_tensor_by_key(z_source_dict, qid)
        z_tensor = z_raw.clone().to(device=device, dtype=torch.float32)
        if z_tensor.dim() == 2:
            z_tensor = z_tensor.unsqueeze(0)

        with torch.no_grad():
            p_z = projector(z_tensor).to(dtype=base_embeds.dtype)

        inputs_embeds = base_embeds.clone()
        inputs_embeds[:, slot_start:slot_end + 1, :] = p_z

        # 1. Reference: Full Non-KV Re-forward loop with get_per_layer_inputs(curr_ids)
        t0_ref = time.time()
        ref_tokens = generate_greedy_no_kv_baseline(
            lm_model=lm_model,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            inputs_embeds=inputs_embeds,
            input_ids=input_tensor,
            eos_set=eos_set,
            max_new_tokens=max_tokens,
        )
        t_ref = time.time() - t0_ref

        # 2. KV Generation: v36b generate_greedy_kv
        t0_kv = time.time()
        kv_tokens = generate_greedy_kv(
            lm_model=lm_model,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            inputs_embeds=inputs_embeds,
            input_ids=input_tensor,
            eos_set=eos_set,
            max_new_tokens=max_tokens,
        )
        t_kv = time.time() - t0_kv

        matched, diff_info = compare_and_diagnose(
            ref_tokens=ref_tokens,
            kv_tokens=kv_tokens,
            lm_model=lm_model,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            inputs_embeds=inputs_embeds,
            input_ids=input_tensor,
            eos_set=eos_set,
            device=device,
            tokenizer=tokenizer,
        )

        res_entry = {
            "condition": "A",
            "item_id": qid,
            "ref_method": "Non-KV Full Re-forward Loop",
            "matched": matched,
            "diff_info": diff_info,
            "ref_tokens": ref_tokens,
            "kv_tokens": kv_tokens,
            "t_ref_sec": t_ref,
            "t_kv_sec": t_kv,
        }
        results.append(res_entry)

        status_str = "MATCH" if matched else "MISMATCH"
        print(f"[Condition A | Q#{qid:04d}] {status_str} ({len(kv_tokens)}/{max_tokens} tokens, Ref: {t_ref:.1f}s, KV: {t_kv:.1f}s)")
        if not matched and diff_info:
            d = diff_info
            r_d = d["ref_diag"]
            k_d = d["kv_diag"]
            print(f"  First mismatch at position {d['mismatch_index']}:")
            print(f"    Ref: ID={d['ref_token_id']} ({d['ref_decoded']}), Top1={r_d['top1_val']:.4f} (ID={r_d['top1_id']}), Top2={r_d['top2_val']:.4f} (ID={r_d['top2_id']}), Δ={r_d['delta']:.4f}")
            print(f"    KV : ID={d['kv_token_id']} ({d['kv_decoded']}), Top1={k_d['top1_val']:.4f} (ID={k_d['top1_id']}), Top2={k_d['top2_val']:.4f} (ID={k_d['top2_id']}), Δ={k_d['delta']:.4f}")

    # =========================================================================
    # Final Summary Table
    # =========================================================================
    print("\n" + "=" * 80)
    print("EQUIVALENCE CHECK v2 SUMMARY TABLE")
    print("=" * 80)
    all_matched = True
    for r in results:
        m_str = "MATCH (100% Identical)" if r["matched"] else "MISMATCH"
        if not r["matched"]:
            all_matched = False
        print(f"Condition {r['condition']} | Q#{r['item_id']:04d} | Ref: {r['ref_method']:<30} | {m_str}")

    print("=" * 80)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.out_json}")

    if all_matched:
        print("[CONCLUSION] All tested conditions exhibit 100% token-level equivalence with non-KV full forward baseline.")
        sys.exit(0)
    else:
        print("[CONCLUSION] Equivalence check completed. Discrepancies noted in report.", file=sys.stderr)
        # Per TASK-0018: we report match status and first mismatch position with top-2 tokens and delta
        sys.exit(0)


if __name__ == "__main__":
    main()

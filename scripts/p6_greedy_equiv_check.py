#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p6_greedy_equiv_check.py - CTS S7 P6 Greedy Equivalence Checker
Directive: TASK-0010 Section 1 (F3) / AUDIT-0009 F3

Purpose:
  Verifies token-level mathematical equivalence of the manual greedy KV generation loop
  against standard reference implementations for 2 training items (Q#14, Q#29, 32 tokens):
  1. Condition C (text-only):
     Compares HuggingFace `generate(input_ids, do_sample=False)` vs `generate_greedy_kv`.
  2. Condition A (prefix-augmented):
     Compares reference autoregressive generation (HF `generate(inputs_embeds)` if supported,
     or exact full-sequence non-KV forward loop) vs `generate_greedy_kv`.

Execution Note:
  Per TASK-0010, this script is written and syntax-verified locally.
  Actual server execution will be performed in the subsequent task after SSH_LOCK is released.
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


def generate_greedy_no_kv(
    lm_model: Any,
    lm_head: Any,
    embed_tokens: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    eos_set: Set[int],
    max_new_tokens: int = 32,
) -> List[int]:
    """Exact baseline greedy autoregressive loop without KV caching.
    
    Re-evaluates the entire concatenated sequence at every step to serve
    as the unassailable reference baseline for KV cache equivalence.
    """
    curr_embeds = inputs_embeds.clone()
    curr_ids = input_ids.clone()
    gen_ids: List[int] = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
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


def compare_token_sequences(
    ref_tokens: List[int],
    kv_tokens: List[int],
    tokenizer: Any,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Compare two token ID lists and return match status and first difference."""
    if ref_tokens == kv_tokens:
        return True, None

    min_len = min(len(ref_tokens), len(kv_tokens))
    first_diff = min_len
    for i in range(min_len):
        if ref_tokens[i] != kv_tokens[i]:
            first_diff = i
            break

    ref_tok = ref_tokens[first_diff] if first_diff < len(ref_tokens) else None
    kv_tok = kv_tokens[first_diff] if first_diff < len(kv_tokens) else None

    diff_detail = {
        "mismatch_index": first_diff,
        "ref_token_id": ref_tok,
        "kv_token_id": kv_tok,
        "ref_decoded": repr(tokenizer.decode([ref_tok])) if ref_tok is not None else "EOF",
        "kv_decoded": repr(tokenizer.decode([kv_tok])) if kv_tok is not None else "EOF",
        "ref_length": len(ref_tokens),
        "kv_length": len(kv_tokens),
    }
    return False, diff_detail


def main():
    parser = argparse.ArgumentParser(description="P6 Greedy Generation Equivalence Check (AUDIT-0009 F3)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:0)")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens to generate (default: 32)")
    parser.add_argument("--items", type=str, default=None, help="Items config JSON (default: configs/s7_train_993.json first 2)")
    parser.add_argument("--decoder_ckpt", type=str, default=None, help="Decoder checkpoint path for Condition A")
    parser.add_argument("--zstar", type=str, default=None, help="z* path for Condition A")
    args = parser.parse_args()

    device_str = args.device
    device = torch.device(device_str)
    max_tokens = args.max_new_tokens

    print("=" * 80)
    print(f"CTS S7 P6 Greedy Generation Equivalence Check (AUDIT-0009 F3)")
    print(f"Device: {device_str} | Max New Tokens: {max_tokens}")
    print("=" * 80, flush=True)

    # 1. Target Items (Q#14, Q#29)
    items_to_test = [14, 29]
    if args.items and os.path.exists(args.items):
        with open(args.items, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        raw_list = cfg if isinstance(cfg, list) else cfg.get("indices", [])
        if len(raw_list) >= 2:
            items_to_test = [int(x) for x in raw_list[:2]]
    print(f"Test Items: Q#{items_to_test[0]:04d}, Q#{items_to_test[1]:04d}")

    # 2. Load Tokenizer & EOS
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()
    eos_set = {tokenizer.eos_token_id}
    for special in ["<turn|>", "<end_of_turn>"]:
        t_ids = tokenizer.encode(special, add_special_tokens=False)
        if t_ids:
            eos_set.update(t_ids)

    # 3. Load Pure Gemma Model
    print("Loading Pure Gemma Base (58c4d7b5)...")
    pure_gemma, lm_model, lm_head, embed_tokens = load_pure_gemma_model(device_str)

    # 4. Load Dataset Records
    dataset_records = load_dataset_records()

    # 5. Checkpoint & Projector Setup for Condition A
    ckpt_path = args.decoder_ckpt
    if not ckpt_path:
        cand_ckpts = [
            os.path.join(BASE_S5_DIR, "results_s7", "decoder_A_v36.pt"),
            os.path.join(BASE_S5_DIR, "results_s5", "decoder_A_v36_step0500.pt"),
        ]
        for c in cand_ckpts:
            if os.path.exists(c):
                ckpt_path = c
                break
    
    projector = None
    if ckpt_path and os.path.exists(ckpt_path):
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
        print("[WARNING] Decoder checkpoint not found. Creating initialized PrefixProjector for structural check.")
        projector = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True).to(device)
        projector.eval()

    # 6. z* Source Setup for Condition A
    z_path = args.zstar or os.path.join(BASE_S5_DIR, "results_s7", "zstar_train1000_P4.pt")
    z_source_dict: Dict[str, Any] = {}
    if os.path.exists(z_path):
        print(f"Loading z* tensors from {z_path}...")
        z_source_dict = torch.load(z_path, map_location="cpu")
    else:
        print("[WARNING] z* source not found. Will use synthetic z tensor for structural check if needed.")

    # Results Table
    results: List[Dict[str, Any]] = []

    # =========================================================================
    # Test Condition C (Text-Only): HF generate vs generate_greedy_kv
    # =========================================================================
    print("\n" + "=" * 60)
    print("Testing Condition C: HF generate vs generate_greedy_kv")
    print("=" * 60)

    for qid in items_to_test:
        prob_text = dataset_records[qid]["prompt"]
        messages = [{"role": "user", "content": prob_text}]
        templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(templated_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        inputs_embeds = embed_tokens(input_tensor)

        # 1. Reference: HF generate
        with torch.inference_mode():
            hf_out = pure_gemma.generate(
                input_ids=input_tensor,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=list(eos_set),
            )
        ref_tokens = hf_out[0, input_tensor.shape[1]:].tolist()

        # 2. KV Generation
        kv_tokens = generate_greedy_kv(
            lm_model=lm_model,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            inputs_embeds=inputs_embeds,
            input_ids=input_tensor,
            eos_set=eos_set,
            max_new_tokens=max_tokens,
        )

        matched, diff_info = compare_token_sequences(ref_tokens, kv_tokens, tokenizer)
        res_entry = {
            "condition": "C",
            "item_id": qid,
            "ref_method": "HF generate(input_ids)",
            "matched": matched,
            "diff_info": diff_info,
            "ref_tokens": ref_tokens,
            "kv_tokens": kv_tokens,
        }
        results.append(res_entry)

        status_str = "MATCH" if matched else "MISMATCH"
        print(f"[Condition C | Q#{qid:04d}] {status_str} ({len(kv_tokens)} tokens)")
        if not matched and diff_info:
            print(f"  First mismatch at pos {diff_info['mismatch_index']}:")
            print(f"    HF ref: ID={diff_info['ref_token_id']} ({diff_info['ref_decoded']})")
            print(f"    KV gen: ID={diff_info['kv_token_id']} ({diff_info['kv_decoded']})")

    # =========================================================================
    # Test Condition A (Prefix-Augmented): HF generate / non-KV vs generate_greedy_kv
    # =========================================================================
    print("\n" + "=" * 60)
    print("Testing Condition A: Reference Autoregressive vs generate_greedy_kv")
    print("=" * 60)

    for qid in items_to_test:
        prob_text = dataset_records[qid]["prompt"]
        token_ids, slot_start, slot_end, placeholder_id = build_canonical_prefix_prompt(
            tokenizer, prob_text=prob_text, k_latent=64
        )
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        base_embeds = embed_tokens(input_tensor)

        # Lookup or mock z
        try:
            z_raw = load_tensor_by_key(z_source_dict, qid)
        except Exception:
            z_raw = torch.zeros(64, 2560)
        z_tensor = z_raw.clone().to(device=device, dtype=torch.float32)
        if z_tensor.dim() == 2:
            z_tensor = z_tensor.unsqueeze(0)

        with torch.no_grad():
            p_z = projector(z_tensor).to(dtype=base_embeds.dtype)

        inputs_embeds = base_embeds.clone()
        inputs_embeds[:, slot_start:slot_end + 1, :] = p_z

        # 1. Reference Generation (Try HF generate with inputs_embeds; fallback to exact non-KV loop)
        ref_method_name = "HF generate(inputs_embeds)"
        ref_tokens: List[int] = []
        try:
            per_layer_init = lm_model.get_per_layer_inputs(input_tensor, None) if hasattr(lm_model, "get_per_layer_inputs") else None
            kwargs_gen = {"per_layer_inputs": per_layer_init} if per_layer_init is not None else {}
            with torch.inference_mode():
                hf_out = pure_gemma.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    eos_token_id=list(eos_set),
                    **kwargs_gen,
                )
            if hf_out.shape[1] > inputs_embeds.shape[1]:
                ref_tokens = hf_out[0, inputs_embeds.shape[1]:].tolist()
            else:
                ref_tokens = hf_out[0].tolist()
        except Exception as e:
            ref_method_name = "Exact Non-KV Baseline Loop"
            ref_tokens = generate_greedy_no_kv(
                lm_model=lm_model,
                lm_head=lm_head,
                embed_tokens=embed_tokens,
                inputs_embeds=inputs_embeds,
                input_ids=input_tensor,
                eos_set=eos_set,
                max_new_tokens=max_tokens,
            )

        # 2. KV Generation
        kv_tokens = generate_greedy_kv(
            lm_model=lm_model,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            inputs_embeds=inputs_embeds,
            input_ids=input_tensor,
            eos_set=eos_set,
            max_new_tokens=max_tokens,
        )

        matched, diff_info = compare_token_sequences(ref_tokens, kv_tokens, tokenizer)
        res_entry = {
            "condition": "A",
            "item_id": qid,
            "ref_method": ref_method_name,
            "matched": matched,
            "diff_info": diff_info,
            "ref_tokens": ref_tokens,
            "kv_tokens": kv_tokens,
        }
        results.append(res_entry)

        status_str = "MATCH" if matched else "MISMATCH"
        print(f"[Condition A | Q#{qid:04d}] {status_str} via {ref_method_name} ({len(kv_tokens)} tokens)")
        if not matched and diff_info:
            print(f"  First mismatch at pos {diff_info['mismatch_index']}:")
            print(f"    Ref: ID={diff_info['ref_token_id']} ({diff_info['ref_decoded']})")
            print(f"    KV : ID={diff_info['kv_token_id']} ({diff_info['kv_decoded']})")

    # =========================================================================
    # Final Summary Table
    # =========================================================================
    print("\n" + "=" * 80)
    print("EQUIVALENCE CHECK SUMMARY TABLE")
    print("=" * 80)
    all_matched = True
    for r in results:
        m_str = "PASS (100% Match)" if r["matched"] else "FAIL (Mismatch)"
        if not r["matched"]:
            all_matched = False
        print(f"Condition {r['condition']} | Q#{r['item_id']:04d} | Ref: {r['ref_method']:<30} | {m_str}")

    print("=" * 80)
    if all_matched:
        print("[CONCLUSION] All tested conditions exhibit 100% token-level equivalence with reference autoregression.")
        sys.exit(0)
    else:
        print("[CONCLUSION] Equivalence check revealed token mismatches. Review logs above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

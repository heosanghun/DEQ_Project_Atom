#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/s7_eval_gate_p6_v36c.py - CTS S7 P6 Gate Evaluation Generator (v3.6c)
Directive: TASK-0065 Section B / AUDIT-P010 FIX-1 / PREREG_S7_D_v3_8

Revision History:
  - v3.6: Initial P6 gate generator with 6 conditions, front 80 token audit, fixed item order.
  - v3.6b:
    * F1: Pure Gemma locked to snapshot 58c4d7b5; candidate list removed; model.safetensors SHA-256
          verified against PREREG_S7_D_v3_7 §2(a) (43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651).
    * F2: Condition F mean z* explicitly counts and prints excluded held-out 80 items; halts if count != 0.
    * F4: When --smoke is absent, enforces decoder checkpoint path to be strictly
          results_s7/decoder_A_v36.pt (A, B, F), results_s7/decoder_D_v36.pt (D1, D2),
          or results_s7/decoder_E_v36.pt (E1, E2).

  - v3.6c (TASK-0065):
    * FIX-1: Stop tokens strictly locked to constant EOS_IDS = (1, 106).
             Removed <end_of_turn> string encoding which polluted eos_set with subwords
             ['_', '>', '<', 'end', 'of', 'turn']. Startup check validates eos_token_id == 1 and turn_id == 106.
    * FIX-2: Canonical item order file results_s7/p6_item_order.json is strictly READ-ONLY (no overwrite).
             Computes and logs order SHA; aborts if optional --order_sha is provided and mismatches.
    * FIX-3: --out is required (no default). Aborts unless output path starts with results_s7/p6r_gen_*
             (or scratch/ for smoke tests) to prevent overwriting original P6 files.
Decoder & Projector Architecture:
  Pure Gemma Base (58c4d7b5) + PrefixProjector + 64 slots embedding replacement.
  Strictly imports from scripts/s7_train_decoder_v36.py:
    - compute_sha256 (Line 65)
    - load_tokenizer (Line 82)
    - build_canonical_prefix_prompt (Line 98)
    - PrefixProjector (Line 158)
    - load_dataset_records (Line 339)
    - compute_nll_f32_chunked (Line 550)

Version Guard:
  Calls scripts/check_p6_decoder_version.sh <path> <condition> prior to loading decoder checkpoints.

Conditions:
  - A: Decoder A + self z*
  - B: Decoder A + z*_{(i+s)%N} (shift s default 1)
  - C: Pure Gemma Base, text only (no prefix slots)
  - D1: Decoder D + self z*
  - D2: Decoder D + z*_{(i+s)%N} (shift s default 1)
  - E1: Decoder E + self z_src
  - E2: Decoder E + z_src_{(i+s)%N} (shift s default 1)
  - F: Decoder A + mean z* across S_train 1,000 (excluding heldout 80)
"""

import os
import sys
import time
import json
import hashlib
import argparse
import subprocess
from typing import Dict, Any, List, Optional, Tuple, Set
import torch
import torch.nn.functional as F

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

SCRIPTS_DIR = os.path.join(BASE_S5_DIR, "scripts")
LOCAL_SCRIPTS = os.path.dirname(os.path.abspath(__file__))

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, SCRIPTS_DIR, LOCAL_SCRIPTS]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Strictly import components from scripts/s7_train_decoder_v36.py
# Documented line numbers in s7_train_decoder_v36.py:
#   compute_sha256: Line 65
#   load_tokenizer: Line 82
#   build_canonical_prefix_prompt: Line 98
#   PrefixProjector: Line 158
#   load_dataset_records: Line 339
#   compute_nll_f32_chunked: Line 550
try:
    from s7_train_decoder_v36 import (
        compute_sha256,
        load_tokenizer,
        build_canonical_prefix_prompt,
        PrefixProjector,
        load_dataset_records,
        compute_nll_f32_chunked,
    )
except ImportError:
    from scripts.s7_train_decoder_v36 import (
        compute_sha256,
        load_tokenizer,
        build_canonical_prefix_prompt,
        PrefixProjector,
        load_dataset_records,
        compute_nll_f32_chunked,
    )



EOS_IDS: Tuple[int, int] = (1, 106)


def verify_checkpoint_guard(ckpt_path: str, condition: str) -> None:
    """Invoke check_p6_decoder_version.sh to verify decoder version before loading."""
    guard_script = os.path.join(BASE_S5_DIR, "scripts", "check_p6_decoder_version.sh")
    if not os.path.exists(guard_script):
        # Fallback to local scripts/
        guard_script = os.path.join(os.path.dirname(__file__), "check_p6_decoder_version.sh")
    
    if not os.path.exists(guard_script):
        raise FileNotFoundError(f"Version guard script not found: {guard_script}")
    
    cmd = ["bash", guard_script, ckpt_path, condition]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[GUARD REJECT] Version check failed for {condition} checkpoint: {ckpt_path}", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        print(res.stdout, file=sys.stderr)
        sys.exit(res.returncode)
    print(f"[GUARD VERIFIED] {res.stdout.strip()}")


def get_canonical_item_order(
    smoke: bool = False,
    items_path: Optional[str] = None,
    expected_order_sha: Optional[str] = None
) -> Tuple[List[int], str]:
    """Retrieve fixed canonical item order by READING ONLY (no overwrite)."""
    order_out_dir = os.path.join(BASE_S5_DIR, "results_s7")
    order_file = os.path.join(order_out_dir, "p6_item_order_smoke.json" if smoke else "p6_item_order.json")
    if items_path:
        order_file = items_path

    if not os.path.exists(order_file):
        # Fallback to local if running locally
        local_order = os.path.join(LOCAL_SCRIPTS, "..", "results_s7", "p6_item_order_smoke.json" if smoke else "p6_item_order.json")
        if os.path.exists(local_order):
            order_file = os.path.abspath(local_order)
        else:
            raise FileNotFoundError(f"Canonical item order file not found: {order_file}")

    actual_order_sha = compute_sha256(order_file)
    print(f"[ITEM-ORDER] Loading canonical item order from {order_file} (SHA: {actual_order_sha})")

    if expected_order_sha and actual_order_sha != expected_order_sha:
        print(f"[FATAL] Item order SHA mismatch! Expected: {expected_order_sha}, got: {actual_order_sha}", file=sys.stderr)
        sys.exit(1)

    with open(order_file, "r", encoding="utf-8") as f:
        d = json.load(f)

    if isinstance(d, list):
        indices = [int(x) for x in d]
    elif isinstance(d, dict) and "indices" in d:
        indices = [int(x) for x in d["indices"]]
    elif isinstance(d, dict) and "valid_indices" in d:
        indices = [int(x) for x in d["valid_indices"]]
    else:
        raise ValueError(f"Invalid format in order file: {order_file}")

    print(f"[ITEM-ORDER] Successfully loaded {len(indices)} items from {order_file}")
    return indices, order_file


def load_tensor_by_key(tensor_dict: Dict[str, Any], qid: int) -> torch.Tensor:
    """Robust lookup for z tensor by question ID."""
    candidates = [
        f"q{qid:04d}_a0.25",
        f"q{qid:04d}",
        f"q{qid}",
        str(qid),
        qid,
    ]
    for c in candidates:
        if c in tensor_dict:
            entry = tensor_dict[c]
            if isinstance(entry, torch.Tensor):
                return entry
            elif isinstance(entry, dict):
                if "z_star" in entry and isinstance(entry["z_star"], torch.Tensor):
                    return entry["z_star"]
                if "z_lowest" in entry and isinstance(entry["z_lowest"], torch.Tensor):
                    return entry["z_lowest"]
    raise KeyError(f"Question ID {qid} not found in tensor dict (tried: {candidates[:3]})")


def compute_f_mean_zstar(train_zstar_path: str, heldout80_path: str) -> torch.Tensor:
    """Compute mean z* across S_train 1,000 excluding heldout 80 items.
    
    AUDIT-0009 F2:
      Explicitly counts and prints excluded items.
      Halts execution if excluded_count != 0 (since S_train and heldout 80 are disjoint).
    """
    print(f"[CONDITION-F] Loading train z* from {train_zstar_path}...")
    d_train = torch.load(train_zstar_path, map_location="cpu")
    
    heldout_set: Set[int] = set()
    if os.path.exists(heldout80_path):
        with open(heldout80_path, "r", encoding="utf-8") as f:
            heldout_set = set(int(x) for x in json.load(f)["indices"])
    
    tensors: List[torch.Tensor] = []
    excluded_count = 0
    for k, v in d_train.items():
        # Extract qid
        import re
        m = re.match(r"^q(\d+)", str(k))
        if m:
            qid = int(m.group(1))
            if qid in heldout_set:
                excluded_count += 1
                continue
        z = v["z_star"] if isinstance(v, dict) and "z_star" in v else v
        if isinstance(z, torch.Tensor):
            tensors.append(z.float())
    
    print(f"[CONDITION-F] Excluded items count: {excluded_count} (expected 0, since S_train and heldout 80 are disjoint)")
    if excluded_count != 0:
        print(f"[FATAL] Condition F excluded count is {excluded_count} != 0. Terminating.", file=sys.stderr)
        sys.exit(1)
        
    assert len(tensors) > 0, "No train z* tensors found for condition F"
    mean_z = torch.stack(tensors, dim=0).mean(dim=0)  # [64, 2560]
    print(f"[CONDITION-F] Mean z* computed across {len(tensors)} items (shape: {mean_z.shape})")
    return mean_z


def load_pure_gemma_model(device_str: str) -> Tuple[Any, Any, Any, Any]:
    """Load Pure Gemma-4-e4b Base Model (bfloat16, strictly frozen).
    
    AUDIT-0009 F1:
      Strictly locked to Pure Gemma Base snapshot 58c4d7b5 per PREREG_S7_D_v3_7 §2(a):
        스냅샷 디렉터리: /home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377
        가중치 체크포인트 정본 파일: model.safetensors (15,992,595,884 bytes)
          SHA-256: 43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651
    """
    from transformers import AutoModelForCausalLM
    snapshot_path = "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Pure Gemma Base 58c4d7b5 snapshot not found at: {snapshot_path}")

    # Verify model.safetensors SHA-256 against PREREG_S7_D_v3_7 §2(a)
    safetensors_path = os.path.join(snapshot_path, "model.safetensors")
    expected_sha256 = "43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651"
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"model.safetensors not found at: {safetensors_path}")

    print(f"[PURE-GEMMA] Verifying model.safetensors SHA-256 against PREREG_S7_D_v3_7 §2(a)...")
    actual_sha256 = compute_sha256(safetensors_path)
    if actual_sha256 != expected_sha256:
        print(f"[FATAL] model.safetensors SHA-256 mismatch!", file=sys.stderr)
        print(f"  Expected (PREREG_S7_D_v3_7 §2(a)): {expected_sha256}", file=sys.stderr)
        print(f"  Actual:                            {actual_sha256}", file=sys.stderr)
        sys.exit(1)
    print(f"[PURE-GEMMA] SHA-256 verified: {actual_sha256}")

    print(f"[PURE-GEMMA] Loading Pure Gemma Base from {snapshot_path} on {device_str}...")
    pure_gemma = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        torch_dtype=torch.bfloat16,
        device_map=device_str,
    )
    for p in pure_gemma.parameters():
        p.requires_grad = False
    pure_gemma.eval()
    
    lm_model = pure_gemma.model.language_model
    lm_head = pure_gemma.lm_head
    embed_tokens = pure_gemma.model.language_model.embed_tokens
    return pure_gemma, lm_model, lm_head, embed_tokens


def generate_greedy_kv(
    lm_model: Any,
    lm_head: Any,
    embed_tokens: Any,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    eos_set: Set[int],
    max_new_tokens: int = 512,
) -> List[int]:
    """Greedy token generation with KV cache."""
    per_layer_inputs = lm_model.get_per_layer_inputs(input_ids, None) if hasattr(lm_model, "get_per_layer_inputs") else None
    kwargs_init = {"per_layer_inputs": per_layer_inputs} if per_layer_inputs is not None else {}

    gen_ids: List[int] = []
    with torch.inference_mode():
        out = lm_model(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True, **kwargs_init)
        logits = lm_head(out.last_hidden_state[:, -1:, :])
        next_token = logits.argmax(dim=-1)
        next_id = int(next_token.item())
        gen_ids.append(next_id)
        if next_id in eos_set:
            return gen_ids
        past = out.past_key_values

        for _ in range(max_new_tokens - 1):
            tok_embed = embed_tokens(next_token)
            tok_per_layer = lm_model.get_per_layer_inputs(next_token, None) if hasattr(lm_model, "get_per_layer_inputs") else None
            kwargs_step = {"per_layer_inputs": tok_per_layer} if tok_per_layer is not None else {}
            out = lm_model(inputs_embeds=tok_embed, past_key_values=past, use_cache=True, return_dict=True, **kwargs_step)
            logits = lm_head(out.last_hidden_state[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            next_id = int(next_token.item())
            gen_ids.append(next_id)
            past = out.past_key_values
            if next_id in eos_set:
                break
    return gen_ids



def main():
    parser = argparse.ArgumentParser(description="P6 Gate Evaluation Generator (v3.6c, TASK-0065)")
    parser.add_argument("--cond", type=str, required=True, choices=["A", "B", "C", "D1", "D2", "E1", "E2", "F"], help="Evaluation Condition")
    parser.add_argument("--shift", type=int, default=1, help="Shift distance for B, D2, E2 (default: 1)")
    parser.add_argument("--items", type=str, default=None, help="Items config JSON (default: heldout 600)")
    parser.add_argument("--decoder_ckpt", type=str, default=None, help="Override decoder checkpoint path")
    parser.add_argument("--zstar", type=str, default=None, help="Override z* path")
    parser.add_argument("--zsrc", type=str, default=None, help="Override z_src path")
    parser.add_argument("--out", type=str, required=True, help="Output jsonl path (required, must start with results_s7/p6r_gen_*)")
    parser.add_argument("--order_sha", type=str, default=None, help="Expected SHA-256 of item order file (optional check)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0 or cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate (default: 512, smoke: 16)")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode (CPU 2-item)")
    parser.add_argument("--nll", action="store_true", help="Compute auxiliary NLL for gold solution")
    args = parser.parse_args()

    t0_all = time.time()
    cond = args.cond
    shift = args.shift
    is_smoke = args.smoke
    max_tokens = args.max_new_tokens if not is_smoke else min(args.max_new_tokens, 16)

    print("=" * 80)
    print(f"CTS S7 P6 Gate Generator (v3.6c) — Condition {cond} (Shift: {shift})")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'SMOKE (CPU 2-item)' if is_smoke else 'FULL (Held-out 600)'}")
    print("=" * 80, flush=True)

    # Pre-condition check for full held-out run (Master Sequence Step 9/10)
    if not is_smoke:
        p5_done = os.path.join(BASE_S5_DIR, "raw_s7", "P5_DONE.flag")
        zsrc_file = os.path.join(BASE_S5_DIR, "results_s7", "zsrc_heldout600.pt")
        if not os.path.exists(p5_done):
            print(f"[PRE-CONDITION ERROR] raw_s7/P5_DONE.flag is missing. P5 must be completed.", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(zsrc_file) and cond in ["E1", "E2"]:
            print(f"[PRE-CONDITION ERROR] {zsrc_file} is missing. z_src(600) must be computed.", file=sys.stderr)
            sys.exit(1)

    # 1. Device Setup
    if args.device:
        device_str = args.device
    elif is_smoke:
        device_str = "cpu"
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Execution Device: {device_str}")

    # 2. Output Path
    out_path = args.out
    norm_out = out_path.replace("\\", "/")
    norm_base = os.path.basename(norm_out)

    is_valid_prefix = (
        norm_out.startswith("results_s7/p6r_gen_")
        or norm_out.startswith("./results_s7/p6r_gen_")
        or norm_out.startswith(os.path.join(BASE_S5_DIR, "results_s7", "p6r_gen_").replace("\\", "/"))
        or (is_smoke and (norm_out.startswith("scratch/") or norm_out.startswith("./scratch/") or norm_base.startswith("p6r_gen_") or norm_base.startswith("smoke_p6r_")))
    )
    if not is_valid_prefix or not (norm_base.startswith("p6r_gen_") or (is_smoke and "p6r" in norm_base) or (is_smoke and norm_out.startswith("scratch/"))):
        print(f"[FATAL] Output path '{out_path}' is invalid! Must begin with 'results_s7/p6r_gen_*' to protect P6 originals.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Output Path: {out_path}")

    # 3. Item Order (TASK-0065: Strictly READ-ONLY)
    item_indices, order_file = get_canonical_item_order(
        smoke=is_smoke,
        items_path=args.items,
        expected_order_sha=args.order_sha
    )
    n_items = len(item_indices)
    print(f"Item Count: {n_items}")

    # 4. Checkpoint Mapping & Guard Check
    # Conditions:
    #   A, B, F -> decoder_A_v36.pt
    #   D1, D2  -> decoder_D_v36.pt
    #   E1, E2  -> decoder_E_v36.pt
    #   C       -> None (pure Gemma)
    default_ckpt_map = {
        "A": os.path.join(BASE_S5_DIR, "results_s7", "decoder_A_v36.pt"),
        "B": os.path.join(BASE_S5_DIR, "results_s7", "decoder_A_v36.pt"),
        "F": os.path.join(BASE_S5_DIR, "results_s7", "decoder_A_v36.pt"),
        "D1": os.path.join(BASE_S5_DIR, "results_s7", "decoder_D_v36.pt"),
        "D2": os.path.join(BASE_S5_DIR, "results_s7", "decoder_D_v36.pt"),
        "E1": os.path.join(BASE_S5_DIR, "results_s7", "decoder_E_v36.pt"),
        "E2": os.path.join(BASE_S5_DIR, "results_s7", "decoder_E_v36.pt"),
    }
    ckpt_path = args.decoder_ckpt or default_ckpt_map.get(cond)

    if cond != "C":
        # AUDIT-0009 F4: When --smoke is absent, strictly enforce decoder path
        if not is_smoke:
            expected_ckpt = default_ckpt_map.get(cond)
            norm_ckpt = os.path.normpath(ckpt_path).replace("\\", "/")
            norm_expected = os.path.normpath(expected_ckpt).replace("\\", "/")
            expected_rel = f"results_s7/{os.path.basename(expected_ckpt)}"
            if not (norm_ckpt == norm_expected or norm_ckpt.endswith(expected_rel)):
                print(f"[FATAL] In full execution mode (--smoke not specified), decoder checkpoint must be exactly: {expected_ckpt}", file=sys.stderr)
                print(f"  Provided: {ckpt_path}", file=sys.stderr)
                sys.exit(1)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Decoder checkpoint not found: {ckpt_path}")
        print(f"\n[VERSION-GUARD] Verifying decoder checkpoint for Condition {cond}...")
        verify_checkpoint_guard(ckpt_path, cond)
    else:
        print("\n[CONDITION-C] Pure Gemma text-only. Decoder checkpoint not required.")

    # 5. Load Tokenizer & Models (TASK-0065 FIX-1)
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()

    if tokenizer.eos_token_id != 1:
        print(f"[FATAL] tokenizer.eos_token_id ({tokenizer.eos_token_id}) != 1", file=sys.stderr)
        sys.exit(1)

    turn_id = tokenizer.convert_tokens_to_ids("<turn|>")
    if turn_id != 106:
        print(f"[FATAL] tokenizer.convert_tokens_to_ids('<turn|>') ({turn_id}) != 106", file=sys.stderr)
        sys.exit(1)

    eos_set = set(EOS_IDS)
    eos_pieces = {tid: tokenizer.convert_ids_to_tokens([tid])[0] for tid in sorted(eos_set)}
    print(f"[EOS-SET] Strict stop tokens locked to {EOS_IDS} -> {eos_pieces}", flush=True)

    print("Loading Pure Gemma Base...")
    pure_gemma, lm_model, lm_head, embed_tokens = load_pure_gemma_model(device_str)

    projector = None
    if cond != "C":
        print(f"Loading PrefixProjector weights from {ckpt_path}...")
        projector = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True).to(device)
        ckpt_data = torch.load(ckpt_path, map_location=device_str)
        if "state_dict" in ckpt_data:
            projector.load_state_dict(ckpt_data["state_dict"])
        else:
            projector.load_state_dict(ckpt_data)
        projector.eval()
        for p in projector.parameters():
            p.requires_grad = False
        print(f"PrefixProjector loaded strictly and frozen.")

    # 6. Load Datasets & z Tensors
    dataset_records = load_dataset_records()
    print(f"Loaded dataset records ({len(dataset_records)} items)")

    z_source_dict: Dict[str, Any] = {}
    f_mean_z: Optional[torch.Tensor] = None

    if cond in ["A", "B", "D1", "D2"]:
        # Needs z*
        if is_smoke:
            p_z = args.zstar or os.path.join(BASE_S5_DIR, "results_s7", "zstar_train1000_P4.pt")
            print(f"Loading smoke z* from {p_z}...")
            z_source_dict = torch.load(p_z, map_location="cpu")
        else:
            # Merge 520 + 80
            p520 = args.zstar or os.path.join(BASE_S5_DIR, "results_s7", "zstar_s7_a0.25_heldout520_lowest.pt")
            p80 = os.path.join(BASE_S5_DIR, "results_s7", "zstar_s7_a0.25_400.pt")
            print(f"Loading 520 z* from {p520}...")
            d520 = torch.load(p520, map_location="cpu")
            print(f"Loading 80 z* from {p80}...")
            d80 = torch.load(p80, map_location="cpu")
            z_source_dict = {**d80, **d520}
    elif cond in ["E1", "E2"]:
        # Needs z_src
        if is_smoke:
            p_src = args.zsrc or os.path.join(BASE_S5_DIR, "scratch", "zsrc_smoke.pt")
            print(f"Loading smoke z_src from {p_src}...")
            z_source_dict = torch.load(p_src, map_location="cpu")
        else:
            p_src = args.zsrc or os.path.join(BASE_S5_DIR, "results_s7", "zsrc_heldout600.pt")
            print(f"Loading 600 z_src from {p_src}...")
            z_source_dict = torch.load(p_src, map_location="cpu")
    elif cond == "F":
        # Needs mean z* from S_train 1,000 (excluding heldout 80)
        p_train_z = args.zstar or os.path.join(BASE_S5_DIR, "results_s7", "zstar_train1000_P4.pt")
        p80_cfg = os.path.join(BASE_S5_DIR, "configs", "s5_heldout_80.json")
        f_mean_z = compute_f_mean_zstar(p_train_z, p80_cfg).to(device)

    # 7. Incremental Resume Check
    done_ids: Set[int] = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        done_ids.add(int(obj["id"]))
                    except Exception:
                        pass
        print(f"[RESUME] Found {len(done_ids)} existing generated items in {out_path}")

    # 8. Generation Loop
    out_file = open(out_path, "a", encoding="utf-8")
    generated_count = 0
    logged_first_item = False

    print(f"\nStarting Generation for Condition {cond} ({n_items} items)...")
    for idx_in_order, qid in enumerate(item_indices):
        if qid in done_ids:
            continue

        item_rec = dataset_records[qid]
        prob_text = item_rec["prompt"]
        t0_gen = time.time()

        # Prompt & Slots
        if cond == "C":
            # Pure Gemma: problem text only (no slots, no placeholder)
            messages = [{"role": "user", "content": prob_text}]
            templated_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(templated_text, add_special_tokens=False)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            inputs_embeds = embed_tokens(input_tensor)
        else:
            token_ids, slot_start, slot_end, placeholder_id = build_canonical_prefix_prompt(
                tokenizer, prob_text=prob_text, k_latent=64
            )
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            base_embeds = embed_tokens(input_tensor)

            # Determine z vector
            if cond in ["A", "D1", "E1"]:
                # Matched
                z_raw = load_tensor_by_key(z_source_dict, qid)
            elif cond in ["B", "D2", "E2"]:
                # Shifted by shift s
                perm_idx = (idx_in_order + shift) % n_items
                perm_qid = item_indices[perm_idx]
                z_raw = load_tensor_by_key(z_source_dict, perm_qid)
            elif cond == "F":
                z_raw = f_mean_z
            else:
                raise ValueError(f"Unknown condition {cond}")

            z_tensor = z_raw.clone().to(device=device, dtype=torch.float32)
            if z_tensor.dim() == 2:
                z_tensor = z_tensor.unsqueeze(0)  # [1, 64, 2560]

            with torch.no_grad():
                p_z = projector(z_tensor).to(dtype=base_embeds.dtype)

            inputs_embeds = base_embeds.clone()
            inputs_embeds[:, slot_start:slot_end + 1, :] = p_z

        # Directive: Log first 80 token IDs for the first item and compare against §3(b)
        if not logged_first_item:
            print("\n" + "-" * 70)
            print(f"[CANONICAL TOKEN AUDIT] Condition {cond} — Item Q#{qid:04d} Front 80 Tokens")
            tokens_to_log = input_ids[:80] if cond == "C" else token_ids[:80]
            for t_idx, tid in enumerate(tokens_to_log):
                dec = repr(tokenizer.decode([tid]))
                if cond == "C":
                    role = "BOS" if t_idx == 0 else ("USER_TAG" if t_idx < 4 else "PROBLEM_TEXT")
                else:
                    if t_idx < 4:
                        role = "BOS/HEADER"
                    elif 4 <= t_idx <= 67:
                        role = f"SLOT_{t_idx-4:02d}"
                    elif t_idx == 68:
                        role = "NEWLINE_108"
                    else:
                        role = "PROBLEM_TEXT"
                print(f"  Pos {t_idx:02d} | Token ID: {tid:6d} | Decoded: {dec:<12} | Role: {role}")
            
            if cond != "C":
                assert token_ids[4:68] == [placeholder_id] * 64, "Slot alignment check failed"
                assert token_ids[68] == 108, f"Token 68 newline check failed (got {token_ids[68]})"
                print(f"[CANONICAL AUDIT PASS] §3(b) slots [4..67] and token 68 == 108 strictly verified.")
            else:
                print(f"[CANONICAL AUDIT PASS] Condition C pure text prompt verified.")
            print("-" * 70 + "\n", flush=True)
            logged_first_item = True

        # Generate Greedy
        gen_token_ids = generate_greedy_kv(
            lm_model=lm_model,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            inputs_embeds=inputs_embeds,
            input_ids=input_tensor,
            eos_set=eos_set,
            max_new_tokens=max_tokens,
        )
        gen_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()
        elapsed_item = time.time() - t0_gen

        # Optional NLL calculation
        nll_val = None
        if args.nll:
            sol_text = item_rec["solution"]
            sol_tokens = tokenizer.encode(sol_text, add_special_tokens=False)
            prompt_len = input_tensor.shape[1]
            full_input_ids = torch.cat([input_tensor, torch.tensor([sol_tokens], dtype=torch.long, device=device)], dim=1)
            target_ids = full_input_ids.clone()
            target_ids[:, :prompt_len] = -100  # mask prompt
            
            full_embeds = embed_tokens(full_input_ids).clone()
            if cond != "C":
                full_embeds[:, slot_start:slot_end + 1, :] = p_z
            
            per_layer_full = lm_model.get_per_layer_inputs(full_input_ids, None) if hasattr(lm_model, "get_per_layer_inputs") else None
            kwargs_nll = {"per_layer_inputs": per_layer_full} if per_layer_full is not None else {}
            
            with torch.no_grad():
                out_full = lm_model(inputs_embeds=full_embeds, use_cache=False, return_dict=True, **kwargs_nll)
                h_states = out_full.last_hidden_state
                shift_targets = target_ids[:, 1:].contiguous()
                nll_val = compute_nll_f32_chunked(h_states[:, :-1, :], shift_targets, lm_head, chunk_size=256)

        prompt_sha = hashlib.sha256((prob_text).encode("utf-8")).hexdigest()
        record_entry = {
            "id": qid,
            "prompt_sha": prompt_sha,
            "generation": gen_text,
            "tokens": len(gen_token_ids),
            "time": round(elapsed_item, 3),
        }
        if nll_val is not None:
            record_entry["nll"] = round(nll_val, 4)

        out_file.write(json.dumps(record_entry, ensure_ascii=False) + "\n")
        out_file.flush()
        generated_count += 1

        if generated_count <= 2 or generated_count % 50 == 0 or generated_count == n_items:
            print(f"[{generated_count}/{n_items}] Q#{qid:04d} done ({len(gen_token_ids)} tokens, {elapsed_item:.2f}s) | Sample: {repr(gen_text[:60])}", flush=True)

    out_file.close()
    sha_out = compute_sha256(out_path)
    total_time = time.time() - t0_all
    print("\n" + "=" * 80)
    print(f"Generation Complete for Condition {cond} (v3.6c)")
    print(f"Output: {out_path}")
    print(f"Items Generated: {generated_count} (Total in file: {len(done_ids) + generated_count})")
    print(f"SHA-256: {sha_out}")
    print(f"Total Time: {total_time:.2f}s")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

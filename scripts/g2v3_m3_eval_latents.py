#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/g2v3_m3_eval_latents.py - Compute post-DEQ latents for evaluation items
Directive: TASK-0072 / PREREG_S7_G2v3 / proposals/G2v3_PLAN_20261009.md (Method M3)

Applies the trained Phi_theta mapping to evaluation items (dev 600 or fresh 1,000)
to produce matched latent representations for PrefixProjector decoding.
"""

import os
import sys
import time
import json
import re
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

DEFAULT_PURE_GEMMA_SNAPSHOT = "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
DEFAULT_FROZEN_BACKBONE_CKPT = "/home/shoon/checkpoints/m1_final_frozen.pt"
DEFAULT_DATASET_PATH = "/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl"


def extract_tensor(d: Dict[str, Any], q: int) -> Optional[torch.Tensor]:
    entry = None
    for k_cand in [
        q, str(q),
        f"q{q:04d}_a0.25", f"q{q:04d}", f"q{q:02d}",
        f"q{q:04d}_s2030", f"q{q}_s2030", f"q{q:02d}_s2030"
    ]:
        if k_cand in d:
            entry = d[k_cand]
            break
    if entry is None:
        for k, v in d.items():
            m = re.match(r"^q(\d+)", str(k))
            if m and int(m.group(1)) == q:
                entry = v
                break
    if entry is None:
        return None
    if isinstance(entry, torch.Tensor):
        return entry
    elif isinstance(entry, dict):
        for k in ["z_star", "z_src", "z_1000", "z_lowest"]:
            if k in entry and isinstance(entry[k], torch.Tensor):
                return entry[k]
    return None


def run_eval_latents(args):
    print("=" * 80)
    print(f"CTS G2v3 M3 Latent Transformation - Mode [{args.mode}]")
    print(f"Items File: {args.items}")
    print(f"Trained DEQ: {args.deq_ckpt}")
    print(f"Output File: {args.out}")
    print("=" * 80, flush=True)

    device = torch.device(args.device)

    # 1. Load Items
    with open(args.items, "r", encoding="utf-8") as f:
        items_data = json.load(f)
    if isinstance(items_data, dict):
        item_indices = items_data.get("indices", list(items_data.keys()))
    else:
        item_indices = items_data
    item_indices = [int(x) for x in item_indices]
    print(f"Loaded {len(item_indices)} items from {args.items}")

    # 2. Load cached zstar and zsrc
    zstar_dict = torch.load(args.zstar, map_location="cpu") if args.zstar and os.path.exists(args.zstar) else {}
    zsrc_dict = torch.load(args.zsrc, map_location="cpu") if args.zsrc and os.path.exists(args.zsrc) else {}

    # 3. Load Backbone & Trained DEQ Weights
    sys.path.insert(0, "/home/shoon/cts_e1_s5")
    from cts.backbone.gemma_adapter import GemmaCTSBackbone
    from cts.model.gemma_loader import load_gemma4_e4b
    from cts.train.lora_compat import apply_paper_lora
    from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights

    m_base, tok_base = load_gemma4_e4b(
        model_id=args.base_model,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        offload_vision_audio=True,
    )
    bb = GemmaCTSBackbone(m_base, tok_base)
    apply_paper_lora(bb, rank=8, target_modules=("q_proj", "v_proj", "o_proj"), dropout=0.0, require_match=False, verbose=False)

    base_ckpt = torch.load(args.backbone_ckpt, map_location="cpu")
    sd = base_ckpt.get("backbone_state_dict", base_ckpt)
    new_sd = {("routing_proj.parametrizations.weight.original" if k == "routing_proj" else k): v for k, v in sd.items()}
    bb.load_state_dict(new_sd, strict=True)

    # Overwrite trained DEQ parameters
    if os.path.exists(args.deq_ckpt):
        trained_deq = torch.load(args.deq_ckpt, map_location="cpu")
        deq_sd = trained_deq.get("state_dict", trained_deq)
        bb.load_state_dict(deq_sd, strict=False)
        print(f"[LOAD] Loaded trained DEQ parameters from {args.deq_ckpt}")

    bb.to(device=device, dtype=torch.bfloat16)
    bb.eval()
    for p in bb.parameters():
        p.requires_grad = False

    # 4. Load dataset prompts
    records = {}
    if os.path.exists(args.dataset):
        with open(args.dataset, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                it = json.loads(line)
                q_idx = int(it.get("question_idx", it.get("idx", it.get("q_idx", idx))))
                records[q_idx] = it

    alpha = 0.25
    g0 = 0.25
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)

    out_dict = {}
    t0 = time.time()

    with torch.no_grad():
        for i, q in enumerate(item_indices):
            zr = extract_tensor(zsrc_dict, q)
            if zr is None:
                print(f"[WARN] Item Q#{q} missing z_src! Skipping.")
                continue
            zr = zr.to(device=device, dtype=torch.float32)
            if zr.dim() == 2:
                zr = zr.unsqueeze(0)

            if args.mode == "A":
                zs = extract_tensor(zstar_dict, q)
                if zs is None:
                    print(f"[WARN] Item Q#{q} missing z_star! Skipping.")
                    continue
                zs = zs.to(device=device, dtype=torch.float32)
                if zs.dim() == 2:
                    zs = zs.unsqueeze(0)
                z_cur = zs
            else:
                z_cur = zr

            # Prompt context
            rec = records.get(q, {})
            p_text = rec.get("prompt", rec.get("problem", ""))
            ctx = bb.encode_context(p_text)
            if hasattr(ctx, "to"):
                ctx = ctx.to(device=device)

            f_weights = sparse_module_weights(routing_weights(z_cur.float(), w_g, 1.0), 3)
            sel_ids = torch.sort(torch.topk(f_weights.reshape(-1), 3).indices.to(dtype=torch.long)).values
            r_ids = [int(x) for x in sel_ids.detach().cpu().tolist()]
            ex = {
                "top_k": 3,
                "deq_map_mode": bb.deq_map_mode,
                "routing_module_ids": r_ids,
                "routing_module_ids_tensor": sel_ids,
            }
            prepare_hook = getattr(bb, "prepare_deq_context", None)
            if callable(prepare_hook):
                ex["eq3_static_inputs"] = prepare_hook(z_cur, ctx, r_ids)

            phi_z = bb.deq_step(z_cur, ctx, f_weights, ex).float()
            z_next = (1.0 - alpha) * z_cur + alpha * (phi_z + g0 * zr)

            # Store (64, 2560) tensor
            z_out = z_next.squeeze(0).cpu().to(dtype=torch.float32)
            out_dict[f"q{q:04d}"] = {"z_star": z_out, "z_src": z_out}
            out_dict[q] = {"z_star": z_out, "z_src": z_out}

            if (i + 1) % 100 == 0 or i == len(item_indices) - 1:
                dt = time.time() - t0
                print(f"[{i+1:4d}/{len(item_indices)}] Computed latents ({dt:.1f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out_dict, args.out)
    print(f"[SAVE] Transformed latents saved to {args.out} ({os.path.getsize(args.out):,} bytes, {len(out_dict)//2} items)")


def main():
    parser = argparse.ArgumentParser(description="CTS G2v3 Latent Transformation Script")
    parser.add_argument("--mode", type=str, choices=["A", "E"], required=True, help="Mode A or E")
    parser.add_argument("--items", type=str, required=True, help="Path to evaluation items json")
    parser.add_argument("--deq_ckpt", type=str, required=True, help="Path to trained DEQ checkpoint")
    parser.add_argument("--zstar", type=str, default="results_s7/g2v2/zstar_zsrcinj_dev600.pt", help="Input zstar pt")
    parser.add_argument("--zsrc", type=str, default="results_s7/zsrc_heldout600.pt", help="Input zsrc pt")
    parser.add_argument("--out", type=str, required=True, help="Output pt path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--base_model", type=str, default=DEFAULT_PURE_GEMMA_SNAPSHOT)
    parser.add_argument("--backbone_ckpt", type=str, default=DEFAULT_FROZEN_BACKBONE_CKPT)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH)

    args = parser.parse_args()
    run_eval_latents(args)


if __name__ == "__main__":
    main()

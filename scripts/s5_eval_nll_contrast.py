#!/usr/bin/env python3
"""scripts/s5_eval_nll_contrast.py - S5 NLL Contrast Evaluation Script
Directive: EXE-260929-40 (System 2 / Ledger #398)

Measures token-by-token NLL of gold solutions on held-out items across 6 conditions:
1) matched: P(z*_i)
2) shuffled: P(z*_j) over 3 permutation rounds
3) zero: 64 x 2560 zeros
4) random_z: Gaussian noise matching ||z*|| norm
5) no_prefix: slots removed (<bos><|turn>user\n<turn|>\n<|turn>model\n)

Outputs format vs content NLL decomposition and paired bootstrap 10k CI for
(NLL_shuffled - NLL_matched). Strictly zero judgment words in output JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

from cts.model.gemma_loader import load_gemma4_e4b
from cts.backbone.gemma_adapter import GemmaCTSBackbone

try:
    from s5_train_decoder_v2 import build_canonical_prefix_prompt, load_tokenizer, PrefixProjector, find_boxed_char_span
except ImportError:
    from scripts.s5_train_decoder_v2 import build_canonical_prefix_prompt, load_tokenizer, PrefixProjector, find_boxed_char_span

DEFAULT_BASE_MODEL = "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
DEFAULT_DATASET = "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl"
DEFAULT_ZSTAR = os.path.join(BASE_S5_DIR, "results_s5", "zstar_s5_400.pt")


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_bootstrap_ci(diffs: List[float], n_bootstrap: int = 10000, seed: int = 2026) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    arr = np.array(diffs, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gold solution NLL contrast across prefix conditions.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained decoder checkpoint")
    parser.add_argument("--items", type=str, required=True, help="Path to held-out items JSON")
    parser.add_argument("--out", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--zstar", type=str, default=DEFAULT_ZSTAR, help="Path to zstar dictionary")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Path to base model snapshot")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Path to math prompts jsonl")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device")
    args = parser.parse_args()

    t_start = time.time()
    device = torch.device(args.device)

    # 1. Load dataset records
    print(f"[NLL-CONTRAST] Loading dataset from: {args.dataset}")
    dataset_records: Dict[int, Dict[str, Any]] = {}
    with open(args.dataset, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                dataset_records[idx] = json.loads(line)

    # 2. Load held-out items
    print(f"[NLL-CONTRAST] Loading items from: {args.items}")
    with open(args.items, "r", encoding="utf-8") as f:
        items_data = json.load(f)
    heldout_indices = items_data.get("heldout_indices", items_data) if isinstance(items_data, dict) else items_data

    # 3. Load zstar dictionary
    print(f"[NLL-CONTRAST] Loading zstar from: {args.zstar}")
    zstar_dict = torch.load(args.zstar, map_location="cpu", weights_only=False)
    zstar_by_q: Dict[int, torch.Tensor] = {}
    for k, v in zstar_dict.items():
        q_idx = int(k.replace("q", "")) if isinstance(k, str) and k.startswith("q") else int(k)
        z_t = v["z_star"] if isinstance(v, dict) and "z_star" in v else (v["z_1000"] if isinstance(v, dict) else v)
        zstar_by_q[q_idx] = z_t.to(dtype=torch.float32)

    # 4. Load base model and tokenizer
    print(f"[NLL-CONTRAST] Loading base model from: {args.base_model}")
    m, tok = load_gemma4_e4b(model_id=args.base_model, device_map=args.device, torch_dtype=torch.float32)
    bb = GemmaCTSBackbone(m, tok)
    lm = bb.cg.model.language_model
    lm.eval()

    # 5. Load decoder projector checkpoint
    print(f"[NLL-CONTRAST] Loading decoder projector from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    projector = PrefixProjector(d_in=2560, d_model=2560).to(device)
    if "projector_state_dict" in ckpt:
        projector.load_state_dict(ckpt["projector_state_dict"])
    elif "model_state_dict" in ckpt:
        projector.load_state_dict(ckpt["model_state_dict"])
    else:
        projector.load_state_dict(ckpt)
    projector.eval()

    # 6. Prompt setup
    base_prefix_tokens, slot_start, slot_end = build_canonical_prefix_prompt(tok, n_slots=64)
    base_prefix_tokens = base_prefix_tokens.to(device)
    target_start = base_prefix_tokens.shape[1]

    # No-prefix template: <bos><|turn>user\n<turn|>\n<|turn>model\n
    no_prefix_text = "<bos><start_of_turn>user\n<end_of_turn>\n<start_of_turn>model\n"
    no_prefix_base_ids = tok(no_prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    no_prefix_target_start = no_prefix_base_ids.shape[1]

    # 7. Shuffled cycle index mappings (3 round-robin shifts: +1, +2, +3)
    N_items = len(heldout_indices)
    shuffled_shifts = [1, 2, 3]

    # Item-level accumulator
    item_results: Dict[str, Dict[str, Any]] = {}
    conditions = ["matched", "shuffled_1", "shuffled_2", "shuffled_3", "shuffled_mean", "zero", "random_z", "no_prefix"]
    cond_item_nll: Dict[str, Dict[str, List[float]]] = {c: {"total": [], "content": [], "format": []} for c in conditions}

    print(f"[NLL-CONTRAST] Evaluating {N_items} items across conditions...")

    with torch.no_grad():
        for i_idx, q in enumerate(heldout_indices):
            sol_text = dataset_records[q]["solution"]
            enc_sol = tok(sol_text, return_offsets_mapping=True, add_special_tokens=False)
            sol_ids = enc_sol["input_ids"][:512 - target_start - 1] + [tok.eos_token_id]
            offsets = enc_sol.get("offset_mapping", [])[:512 - target_start - 1]
            span = find_boxed_char_span(sol_text)
            is_content = [bool(span and s < span[1] and e > span[0]) for (s, e) in offsets]
            is_content.append(False)  # eos token

            # Sequence with prefix slots
            full_seq = torch.cat([base_prefix_tokens[0], torch.tensor(sol_ids, device=device, dtype=torch.long)])
            input_batch = full_seq.unsqueeze(0)
            target_batch = torch.full_like(input_batch, -100)
            target_batch[:, target_start:] = input_batch[:, target_start:]
            content_mask = torch.zeros_like(target_batch, dtype=torch.bool)
            content_mask[:, target_start:target_start + len(is_content)] = torch.tensor(is_content, device=device, dtype=torch.bool)

            per_layer = lm.get_per_layer_inputs(input_batch, None) if hasattr(lm, "get_per_layer_inputs") else None
            kwargs_fwd = {"per_layer_inputs": per_layer} if per_layer is not None else {}
            base_embeds = lm.embed_tokens(input_batch).clone()

            # Sequence without prefix slots (no_prefix)
            np_full_seq = torch.cat([no_prefix_base_ids[0], torch.tensor(sol_ids, device=device, dtype=torch.long)])
            np_input_batch = np_full_seq.unsqueeze(0)
            np_target_batch = torch.full_like(np_input_batch, -100)
            np_target_batch[:, no_prefix_target_start:] = np_input_batch[:, no_prefix_target_start:]
            np_content_mask = torch.zeros_like(np_target_batch, dtype=torch.bool)
            np_content_mask[:, no_prefix_target_start:no_prefix_target_start + len(is_content)] = torch.tensor(is_content, device=device, dtype=torch.bool)

            np_per_layer = lm.get_per_layer_inputs(np_input_batch, None) if hasattr(lm, "get_per_layer_inputs") else None
            np_kwargs_fwd = {"per_layer_inputs": np_per_layer} if np_per_layer is not None else {}

            # Helper for NLL computation
            def eval_loss(embeds_in, kwargs_in, targets_in, cmask_in, tstart_in):
                out = lm(inputs_embeds=embeds_in, return_dict=True, **kwargs_in)
                logits = bb.cg.lm_head(out.last_hidden_state)
                shift_logits = logits[:, tstart_in - 1:-1, :].contiguous()
                shift_targets = targets_in[:, tstart_in:].contiguous()
                shift_cmask = cmask_in[:, tstart_in:].contiguous()
                loss_toks = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), reduction="none").view_as(shift_targets)
                valid = (shift_targets != -100)
                nll_tot = float(loss_toks[valid].mean().item()) if valid.any() else 0.0
                nll_cnt = float(loss_toks[valid & shift_cmask].mean().item()) if (valid & shift_cmask).any() else nll_tot
                nll_fmt = float(loss_toks[valid & (~shift_cmask)].mean().item()) if (valid & (~shift_cmask)).any() else nll_tot
                return nll_tot, nll_cnt, nll_fmt

            # z* matched
            z_matched = zstar_by_q[q].to(device).unsqueeze(0)
            b_emb_matched = base_embeds.clone()
            b_emb_matched[:, slot_start:slot_end + 1, :] = projector(z_matched)
            nll_tot_m, nll_cnt_m, nll_fmt_m = eval_loss(b_emb_matched, kwargs_fwd, target_batch, content_mask, target_start)

            # z* shuffled rounds
            shuf_tots, shuf_cnts, shuf_fmts = [], [], []
            for s_round, shift in enumerate(shuffled_shifts):
                shuf_q = heldout_indices[(i_idx + shift) % N_items]
                z_shuf = zstar_by_q[shuf_q].to(device).unsqueeze(0)
                b_emb_shuf = base_embeds.clone()
                b_emb_shuf[:, slot_start:slot_end + 1, :] = projector(z_shuf)
                s_tot, s_cnt, s_fmt = eval_loss(b_emb_shuf, kwargs_fwd, target_batch, content_mask, target_start)
                shuf_tots.append(s_tot)
                shuf_cnts.append(s_cnt)
                shuf_fmts.append(s_fmt)
                cond_item_nll[f"shuffled_{s_round+1}"]["total"].append(s_tot)
                cond_item_nll[f"shuffled_{s_round+1}"]["content"].append(s_cnt)
                cond_item_nll[f"shuffled_{s_round+1}"]["format"].append(s_fmt)

            s_tot_mean = float(np.mean(shuf_tots))
            s_cnt_mean = float(np.mean(shuf_cnts))
            s_fmt_mean = float(np.mean(shuf_fmts))

            # zero prefix
            b_emb_zero = base_embeds.clone()
            b_emb_zero[:, slot_start:slot_end + 1, :] = projector(torch.zeros_like(z_matched))
            nll_tot_z, nll_cnt_z, nll_fmt_z = eval_loss(b_emb_zero, kwargs_fwd, target_batch, content_mask, target_start)

            # random_z prefix (matching norm)
            norm_m = float(torch.norm(z_matched).item())
            z_rand = torch.randn_like(z_matched)
            z_rand = z_rand * (norm_m / (float(torch.norm(z_rand).item()) + 1e-8))
            b_emb_rand = base_embeds.clone()
            b_emb_rand[:, slot_start:slot_end + 1, :] = projector(z_rand)
            nll_tot_r, nll_cnt_r, nll_fmt_r = eval_loss(b_emb_rand, kwargs_fwd, target_batch, content_mask, target_start)

            # no_prefix
            np_embeds = lm.embed_tokens(np_input_batch).clone()
            nll_tot_np, nll_cnt_np, nll_fmt_np = eval_loss(np_embeds, np_kwargs_fwd, np_target_batch, np_content_mask, no_prefix_target_start)

            # Accumulate
            cond_item_nll["matched"]["total"].append(nll_tot_m)
            cond_item_nll["matched"]["content"].append(nll_cnt_m)
            cond_item_nll["matched"]["format"].append(nll_fmt_m)

            cond_item_nll["shuffled_mean"]["total"].append(s_tot_mean)
            cond_item_nll["shuffled_mean"]["content"].append(s_cnt_mean)
            cond_item_nll["shuffled_mean"]["format"].append(s_fmt_mean)

            cond_item_nll["zero"]["total"].append(nll_tot_z)
            cond_item_nll["zero"]["content"].append(nll_cnt_z)
            cond_item_nll["zero"]["format"].append(nll_fmt_z)

            cond_item_nll["random_z"]["total"].append(nll_tot_r)
            cond_item_nll["random_z"]["content"].append(nll_cnt_r)
            cond_item_nll["random_z"]["format"].append(nll_fmt_r)

            cond_item_nll["no_prefix"]["total"].append(nll_tot_np)
            cond_item_nll["no_prefix"]["content"].append(nll_cnt_np)
            cond_item_nll["no_prefix"]["format"].append(nll_fmt_np)

            item_results[f"q{q:04d}"] = {
                "question_idx": q,
                "nll_matched_total": nll_tot_m,
                "nll_matched_content": nll_cnt_m,
                "nll_matched_format": nll_fmt_m,
                "nll_shuffled_mean_total": s_tot_mean,
                "nll_shuffled_mean_content": s_cnt_mean,
                "nll_shuffled_mean_format": s_fmt_mean,
                "delta_total": s_tot_mean - nll_tot_m,
                "delta_content": s_cnt_mean - nll_cnt_m,
                "delta_format": s_fmt_mean - nll_fmt_m,
                "nll_zero_total": nll_tot_z,
                "nll_random_z_total": nll_tot_r,
                "nll_no_prefix_total": nll_tot_np,
            }

            if (i_idx + 1) % 20 == 0 or (i_idx + 1) == N_items:
                print(f"[{i_idx+1:2d}/{N_items}] item q{q:04d} evaluated | matched NLL: {nll_tot_m:.4f} | shuffled NLL: {s_tot_mean:.4f} | diff: {s_tot_mean - nll_tot_m:+.4f}")

    # 8. Compute paired differences and 10k bootstrap CI
    delta_total_list = [cond_item_nll["shuffled_mean"]["total"][i] - cond_item_nll["matched"]["total"][i] for i in range(N_items)]
    delta_content_list = [cond_item_nll["shuffled_mean"]["content"][i] - cond_item_nll["matched"]["content"][i] for i in range(N_items)]
    delta_format_list = [cond_item_nll["shuffled_mean"]["format"][i] - cond_item_nll["matched"]["format"][i] for i in range(N_items)]

    ci_total = compute_bootstrap_ci(delta_total_list, n_bootstrap=10000, seed=2026)
    ci_content = compute_bootstrap_ci(delta_content_list, n_bootstrap=10000, seed=2026)
    ci_format = compute_bootstrap_ci(delta_format_list, n_bootstrap=10000, seed=2026)

    # 9. Aggregate condition summary table
    condition_summary: Dict[str, Dict[str, float]] = {}
    for c in conditions:
        condition_summary[c] = {
            "mean_nll_total": float(np.mean(cond_item_nll[c]["total"])),
            "mean_nll_content": float(np.mean(cond_item_nll[c]["content"])),
            "mean_nll_format": float(np.mean(cond_item_nll[c]["format"])),
        }

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "ckpt_path": args.ckpt,
            "ckpt_sha256": compute_sha256(args.ckpt) if os.path.isfile(args.ckpt) else "NON_FILE",
            "items_path": args.items,
            "items_count": N_items,
            "device": str(device),
            "elapsed_seconds": round(time.time() - t_start, 2),
        },
        "condition_summary": condition_summary,
        "paired_bootstrap_10k_ci": {
            "delta_total": ci_total,
            "delta_content": ci_content,
            "delta_format": ci_format,
        },
        "per_item_results": item_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[NLL-CONTRAST] Output written to {args.out}")
    print(f"  Matched NLL (Total): {condition_summary['matched']['mean_nll_total']:.4f}")
    print(f"  Shuffled NLL (Total): {condition_summary['shuffled_mean']['mean_nll_total']:.4f}")
    print(f"  Delta Total CI [2.5%, 97.5%]: [{ci_total['ci_lower']:+.5f}, {ci_total['ci_upper']:+.5f}]")
    print(f"  Delta Content CI [2.5%, 97.5%]: [{ci_content['ci_lower']:+.5f}, {ci_content['ci_upper']:+.5f}]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diag_d4_traj.py - Trajectory Analysis Across Picard Steps (D4)
Directive: TASK-0070 §D4 (Ledger #430)

Runs canonical Picard trajectory saving states at k in {0, 10, 30, 100, 300, 1000}.
Uses exact canonical parameters:
  alpha=0.25, g0=0.25, seed=2030, sigma_noise=0.020, max_nfe=1000, tol=0.0

Modes:
  1. --anchor_check: Runs Q#20, 24, 32 on GPU 1, asserts max|Δ| = 0 against
     results_s7/anchor_ref_gpu1_check.pt (SHA 5103642d...).
  2. Trajectory run on 40 items from configs/diag_d4_train40.json.
     Saves output to results_s7/diag/d4_traj_train40.pt.
  3. Computes summary metrics:
     - k-step mean cos(z_k, z_src)
     - Cross-retrieval rate (z_k -> z_src) within 40 items
     - Cross-item mean cos across 40 items
     - Median residual r_k
"""

import os
import sys
import json
import time
import argparse
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import torch
import numpy as np

# Add project root to sys.path
BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
for p in [BASE_S5_DIR, SYS_DIR, "."]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cts.deq.zstar_canonical_lowest import DEFAULT_CONFIG_PATH
from cts.deq.zstar_canonical import load_canonical_config
from cts.deq.transition_inject import (
    compute_broadcast_context_c,
    compute_c_hat_0,
    InjectedTransitionStep,
)


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


TARGET_STEPS = [0, 10, 30, 100, 300, 1000]


@torch.no_grad()
def compute_zstar_traj_single(
    x: Any,
    backbone: Any,
    w_g: torch.Tensor,
    device: torch.device,
    cfg_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Computes trajectory saving z_k and r_k at k in {0, 10, 30, 100, 300, 1000}.
    Guaranteed bit-for-bit identical to canonical compute_zstar_lowest.
    """
    config = load_canonical_config(cfg_path if cfg_path is not None else DEFAULT_CONFIG_PATH)
    config.update(kwargs)

    g0 = float(config.get("g0", 0.25))
    alpha = float(config.get("alpha", 0.25))
    seed = int(config.get("seed", 2030))
    max_nfe = int(config.get("max_nfe", 1000))
    k_latent = int(config.get("k_latent", 64))
    sigma_noise = float(config.get("sigma_noise", 0.020))
    top_k = int(config.get("top_k", 3))
    routing_mode = str(config.get("routing_mode", "sparse"))

    orig_mode = backbone.training
    try:
        backbone.eval()
        d = backbone.hidden_size

        if isinstance(x, str):
            ctx = backbone.encode_context(x)
            if hasattr(ctx, "to"):
                ctx = ctx.to(device=device)
        else:
            ctx = x
            if hasattr(ctx, "to"):
                ctx = ctx.to(device=device)

        # 1. z_src via root_fwdpass
        z_src = backbone.root_fwdpass(ctx, K=k_latent, output_dtype=torch.float32).to(device=device)

        # 2. Context broadcast c_x
        c_x = compute_broadcast_context_c(ctx, K=k_latent, device=device, dtype=torch.float32)

        # 3. Canonical initialization z0
        torch.manual_seed(seed)
        z0 = (
            z_src
            + torch.randn(k_latent, d, device=device, dtype=torch.float32) * sigma_noise
        ).clone()

        # 4. Injection vector c_hat_0
        c_hat_0 = compute_c_hat_0(c_x, z0).to(device=device, dtype=torch.float32)

        # 5. Transition step function
        step_fn = InjectedTransitionStep(
            backbone=backbone,
            context=ctx,
            w_g=w_g,
            c_hat_0=c_hat_0,
            g0=g0,
            alpha=alpha,
            top_k=top_k,
            routing_mode=routing_mode,
        )

        saved_zk: Dict[int, torch.Tensor] = {}
        saved_rk: Dict[int, float] = {}

        # Save step 0
        if 0 in TARGET_STEPS:
            saved_zk[0] = z0.detach().cpu().clone()

        z_cur = z0.clone()

        # Picard iteration
        for k in range(1, max_nfe + 1):
            z_next = step_fn(z_cur)

            # float64 residual of z_{k-1}
            diff_64 = z_next.to(torch.float64) - z_cur.to(torch.float64)
            diff_norm_64 = float(torch.norm(diff_64, p=2).item())
            norm_cur_64 = float(torch.norm(z_cur.to(torch.float64), p=2).item())
            r_k_prev = diff_norm_64 / (alpha * max(norm_cur_64, 1e-12))

            step_prev = k - 1
            if step_prev in TARGET_STEPS:
                saved_rk[step_prev] = r_k_prev

            # If current step is a target step, save z_k
            if k in TARGET_STEPS:
                saved_zk[k] = z_next.detach().cpu().clone()

            z_cur = z_next

        # Final state at step 1000
        z_star = z_cur

        # Evaluate residual at step 1000
        z_next_final = step_fn(z_star)
        diff_final_64 = z_next_final.to(torch.float64) - z_star.to(torch.float64)
        diff_final_norm_64 = float(torch.norm(diff_final_64, p=2).item())
        norm_star_64 = float(torch.norm(z_star.to(torch.float64), p=2).item())
        r_1000 = diff_final_norm_64 / (alpha * max(norm_star_64, 1e-12))
        saved_rk[1000] = r_1000

        return {
            "z_src": z_src.detach().cpu().clone(),
            "z_star": z_star.detach().cpu().clone(),
            "saved_zk": saved_zk,
            "saved_rk": saved_rk,
        }
    finally:
        if orig_mode:
            backbone.train()


def load_dataset_prompts(prompts_path: str, qids: List[int]) -> Dict[int, str]:
    """Loads prompt texts for specified question IDs."""
    prompts = {}
    target_set = set(qids)
    with open(prompts_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in target_set:
                data = json.loads(line)
                prompts[idx] = data["prompt"]
                if len(prompts) == len(target_set):
                    break
    return prompts


def compute_d4_metrics(
    qids: List[int],
    data_dict: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes D4 metrics across the 40 items for each k in TARGET_STEPS:
    - mean cos(z_k, z_src)
    - 40-item cross-retrieval rate (z_k -> z_src)
    - cross-item mean cos between z_k vectors
    - median residual r_k
    """
    N = len(qids)
    # Stack z_src [N, 64, 2560]
    zsrc_stack = torch.stack([data_dict[q]["z_src"] for q in qids]).float()  # [N, 64, 2560]
    zsrc_flat = zsrc_stack.reshape(N, -1)
    zsrc_norm = torch.norm(zsrc_flat, p=2, dim=1, keepdim=True)
    zsrc_u = zsrc_flat / torch.clamp(zsrc_norm, min=1e-12)

    metrics_by_k = {}

    for k in TARGET_STEPS:
        zk_stack = torch.stack([data_dict[q]["saved_zk"][k] for q in qids]).float()  # [N, 64, 2560]
        zk_flat = zk_stack.reshape(N, -1)
        zk_norm = torch.norm(zk_flat, p=2, dim=1, keepdim=True)
        zk_u = zk_flat / torch.clamp(zk_norm, min=1e-12)

        # 1. cos(z_k, z_src) item-wise mean
        cos_i = torch.sum(zk_u * zsrc_u, dim=1).numpy()
        mean_cos_zsrc = float(np.mean(cos_i))

        # 2. Cross-retrieval rate (z_k -> z_src)
        S_cross = torch.mm(zk_u, zsrc_u.t())  # [N, N]
        pred_src = torch.argmax(S_cross, dim=1).numpy()
        retrieval_rate = float(np.mean(pred_src == np.arange(N)))

        # 3. Cross-item mean cos between z_k
        S_kk = torch.mm(zk_u, zk_u.t())
        mask_offdiag = ~torch.eye(N, dtype=torch.bool)
        cross_item_cos = float(torch.mean(S_kk[mask_offdiag]).item())

        # 4. Residual r_k median
        rk_vals = [data_dict[q]["saved_rk"][k] for q in qids]
        median_rk = float(np.median(rk_vals))
        mean_rk = float(np.mean(rk_vals))

        metrics_by_k[str(k)] = {
            "mean_cos_zsrc": mean_cos_zsrc,
            "cross_retrieval_rate": retrieval_rate,
            "cross_item_mean_cos": cross_item_cos,
            "median_r_k": median_rk,
            "mean_r_k": mean_rk,
        }

    return metrics_by_k


def run_anchor_check(backbone: Any, w_g: torch.Tensor, device: torch.device, base_dir: str):
    """Verifies anchor 3 items (Q#20, 24, 32) against results_s7/anchor_ref_gpu1_check.pt."""
    print("\n" + "=" * 60)
    print("D4 INSTRUMENT CHECK: Q#20, Q#24, Q#32 on GPU 1")
    print("=" * 60, flush=True)

    ref_path = os.path.join(base_dir, "results_s7", "anchor_ref_gpu1_check.pt")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Anchor reference file missing: {ref_path}")

    ref_sha = compute_sha256(ref_path)
    expected_ref_sha = "5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6"
    print(f"Anchor ref path: {ref_path}")
    print(f"Anchor ref SHA:  {ref_sha}")
    if ref_sha != expected_ref_sha:
        raise ValueError(f"Anchor ref SHA mismatch! Expected {expected_ref_sha}, got {ref_sha}")

    ref_data = torch.load(ref_path, map_location="cpu")
    anchor_qids = [20, 24, 32]

    # Load prompts
    prompts_path = os.path.join(base_dir, "data", "stage2", "math_train_prompts_5000.jsonl")
    anchor_prompts = load_dataset_prompts(prompts_path, anchor_qids)

    max_delta_all = 0.0

    for qid in anchor_qids:
        print(f"Running Anchor Q#{qid:02d} on {device}...")
        prompt = anchor_prompts[qid]
        res = compute_zstar_traj_single(prompt, backbone, w_g, device)
        z_1000 = res["z_star"]  # [64, 2560]

        # Extract reference
        def _extract(data_dict, q_idx):
            candidates = [f"q{q_idx:04d}_a0.25", f"q{q_idx:02d}_s2030", f"q{q_idx:04d}", f"q{q_idx:02d}", f"q{q_idx}", str(q_idx), int(q_idx)]
            for c in candidates:
                if c in data_dict:
                    entry = data_dict[c]
                    if isinstance(entry, torch.Tensor): return entry
                    if isinstance(entry, dict):
                        for k in ["z_star", "z_lowest", "z_canon"]:
                            if k in entry and isinstance(entry[k], torch.Tensor): return entry[k]
            raise KeyError(f"Q#{q_idx} not found in anchor ref")
        z_ref = _extract(ref_data, qid)
        if z_ref.dim() == 3:
            z_ref = z_ref.squeeze(0)

        delta = torch.max(torch.abs(z_1000.float() - z_ref.float())).item()
        print(f"  Anchor Q#{qid:02d}: max|Δ| = {delta:.8e}")
        if delta > max_delta_all:
            max_delta_all = delta

    print(f"\nAnchor Check Result: Overall max|Δ| = {max_delta_all:.8e}")
    if max_delta_all != 0.0:
        print(f"[FATAL] Anchor check non-zero: max|Δ| = {max_delta_all:.8e} != 0", file=sys.stderr)
        sys.exit(1)
    print("[PASS] Anchor instrument check verified with max|Δ| = 0.\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="TASK-0070 D4: Trajectory Step Analysis")
    parser.add_argument("--base_dir", type=str, default="/home/shoon/cts_e1_s5")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--items", type=str, default="configs/diag_d4_train40.json")
    parser.add_argument("--out", type=str, default="results_s7/diag/d4_traj_train40.pt")
    parser.add_argument("--anchor_check", action="store_true", help="Run anchor instrument check")
    parser.add_argument("--skip_anchor", action="store_true", help="Skip anchor instrument check")
    args = parser.parse_args()

    base_dir = args.base_dir
    os.chdir(base_dir)

    print("=" * 80)
    print(f"TASK-0070 D4: Picard Step-wise Trajectory Generator")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device} | Base dir: {base_dir}")
    print("=" * 80, flush=True)

    device = torch.device(args.device)

    # Initialize backbone
    print(f"Initializing backbone model on {device}...")
    from harness_A_v1 import get_model
    backbone = get_model(str(device))
    w_g = backbone.routing_matrix().to(device=device, dtype=torch.float32)

    # Instrument check
    if not args.skip_anchor:
        run_anchor_check(backbone, w_g, device, base_dir)

    if args.anchor_check and args.skip_anchor:
        return

    # Load 40 items
    items_path = os.path.join(base_dir, args.items)
    print(f"Loading 40 items config from {items_path}...")
    items_sha = compute_sha256(items_path)
    print(f"Items config SHA: {items_sha}")
    with open(items_path, "r", encoding="utf-8") as f:
        cfg_obj = json.load(f)
    qids = cfg_obj["indices"] if isinstance(cfg_obj, dict) else cfg_obj
    assert len(qids) == 40, f"Expected 40 items, got {len(qids)}"

    prompts_path = os.path.join(base_dir, "data", "stage2", "math_train_prompts_5000.jsonl")
    prompts = load_dataset_prompts(prompts_path, qids)

    # Run 40 items
    print(f"\nStarting Trajectory Execution on {len(qids)} items...")
    trajectories_data = {}
    t0_all = time.time()

    for idx, qid in enumerate(qids):
        t0_item = time.time()
        prompt = prompts[qid]
        res = compute_zstar_traj_single(prompt, backbone, w_g, device)
        elapsed_item = time.time() - t0_item

        trajectories_data[qid] = {
            "z_src": res["z_src"],
            "saved_zk": res["saved_zk"],
            "saved_rk": res["saved_rk"],
        }
        print(f"[{idx+1}/40] Q#{qid:04d} done ({elapsed_item:.2f}s) | r0={res['saved_rk'][0]:.4f} r1000={res['saved_rk'][1000]:.4f}", flush=True)

    elapsed_all = time.time() - t0_all
    print(f"\nAll 40 items finished in {elapsed_all:.2f}s ({elapsed_all/60:.2f}m).", flush=True)

    # Compute metrics
    print("\nComputing D4 Trajectory Metrics across 40 items...")
    metrics_by_k = compute_d4_metrics(qids, trajectories_data)

    print("\n" + "=" * 60)
    print("D4 Trajectory Metrics Summary Table:")
    print("=" * 60)
    header = f"{'k':<6}{'cos(z_k, z_src)':<20}{'Cross-retrieval':<20}{'Cross-item cos':<20}{'Median r_k':<15}"
    print(header)
    print("-" * len(header))
    for k in TARGET_STEPS:
        m = metrics_by_k[str(k)]
        print(f"{k:<6}{m['mean_cos_zsrc']:<20.4f}{m['cross_retrieval_rate']:<20.2%}{m['cross_item_mean_cos']:<20.4f}{m['median_r_k']:<15.6f}")

    # Save results
    out_path = os.path.join(base_dir, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    save_dict = {
        "metadata": {
            "directive": "TASK-0070 D4",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "target_steps": TARGET_STEPS,
            "qids": qids,
            "items_sha": items_sha,
            "elapsed_seconds": elapsed_all,
        },
        "trajectories": {
            f"q{qid:04d}": {
                "z_src": trajectories_data[qid]["z_src"],
                "saved_zk": trajectories_data[qid]["saved_zk"],
                "saved_rk": trajectories_data[qid]["saved_rk"],
            }
            for qid in qids
        },
        "metrics": metrics_by_k,
    }

    torch.save(save_dict, out_path)
    out_sha = compute_sha256(out_path)
    print(f"\nSaved trajectory pt to {out_path} (SHA: {out_sha})")

    # Save metrics JSON
    json_path = out_path.replace(".pt", "_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": save_dict["metadata"],
            "metrics": metrics_by_k,
            "out_sha": out_sha,
        }, f, indent=2)
    print(f"Saved trajectory metrics JSON to {json_path}")


if __name__ == "__main__":
    main()

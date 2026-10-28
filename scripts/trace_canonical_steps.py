#!/usr/bin/env python3
"""
scripts/trace_canonical_steps.py - Trace canonical compute_zstar step-by-step with routing modules
Directive: 원장 #374 §3
"""

import os
import sys
import torch

torch.set_num_threads(8)

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from harness_A_v1 import get_model, get_dataset
from cts.routing.sparse_moe_ref import routing_weights
from cts.deq.transition_inject import compute_broadcast_context_c, compute_c_hat_0, InjectedTransitionStep

bb = get_model("cpu")
w_g = bb.routing_matrix().to(device="cpu", dtype=torch.float32)
dataset = get_dataset()

def trace_canonical_item(q_idx, start_step=85, end_step=105, total_steps=105):
    p_text = dataset[q_idx]["prompt"]
    with torch.no_grad():
        ctx = bb.encode_context(p_text)
        k_latent = 64
        d = 2560
        z_src = bb.root_fwdpass(ctx, K=k_latent, output_dtype=torch.float32).to(device="cpu")
        c_x = compute_broadcast_context_c(ctx, K=k_latent, device="cpu", dtype=torch.float32)
        
        # Strict canonical seed: 2030
        torch.manual_seed(2030)
        z0 = (z_src + torch.randn(k_latent, d, device="cpu", dtype=torch.float32) * 0.020).clone()
        c_hat_0 = compute_c_hat_0(c_x, z0).to(device="cpu", dtype=torch.float32)

        step_fn = InjectedTransitionStep(
            backbone=bb, context=ctx, w_g=w_g, c_hat_0=c_hat_0,
            g0=0.25, alpha=0.25, top_k=3, routing_mode="sparse"
        )

        z_cur = z0.clone()
        prev_r_ids = None
        records = []

        for k in range(1, total_steps + 1):
            z_next = step_fn(z_cur)
            diff = float(torch.norm(z_next - z_cur).item())
            norm_z = float(torch.norm(z_cur).item())
            rho = diff / max(norm_z, 1e-12)
            r_ids = list(step_fn.last_r_ids)
            r_changed = (prev_r_ids is not None and r_ids != prev_r_ids)

            alpha_gate = routing_weights(z_cur, w_g, 1.0)
            top_vals, top_idx = torch.topk(alpha_gate, 4)
            gap_3_4 = float(top_vals[2] - top_vals[3])

            if k >= start_step and k <= end_step:
                records.append({
                    "step": k,
                    "rho": rho,
                    "r_ids": r_ids,
                    "r_changed": r_changed,
                    "top_weights": [round(float(x), 5) for x in top_vals.tolist()],
                    "gap_3_4": gap_3_4,
                })
            
            prev_r_ids = r_ids
            z_cur = z_next

    return records

print("=" * 80)
print("CANONICAL STEP-BY-STEP TRACE (seed=2030, Steps 85 to 105)")
print("=" * 80)

for q in [2140, 4067, 1790]:
    print(f"\n--- Q#{q:04d} Canonical Step-by-Step Trace (Steps 85 to 105) ---", flush=True)
    recs = trace_canonical_item(q, 85, 105, 105)
    print(f"{'Step':>5} | {'Rho':>12} | {'Prev Rho':>12} | {'Ratio':>7} | {'Active Modules':>16} | {'Gap 3-4':>10} | {'Switch?'}")
    print("-" * 80)
    for i in range(len(recs)):
        r = recs[i]
        k = r["step"]
        rho = r["rho"]
        prev_rho = recs[i-1]["rho"] if i > 0 else rho
        ratio = rho / max(prev_rho, 1e-12)
        sw_mark = "*** SWITCH ***" if r["r_changed"] else ""
        print(f"{k:5d} | {rho:12.6e} | {prev_rho:12.6e} | {ratio:6.2f}x | {str(r['r_ids']):>16} | {r['gap_3_4']:10.4e} | {sw_mark}", flush=True)

print("\nCanonical trace complete.", flush=True)

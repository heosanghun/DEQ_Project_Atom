#!/usr/bin/env python3
"""
scripts/test_routing_hypothesis_v2.py - Trace routing switches and rho on CPU
Directive: 원장 #373 §4
"""

import os
import sys
import time
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

print("Loading model on CPU...", flush=True)
bb = get_model("cpu")
w_g = bb.routing_matrix().to(device="cpu", dtype=torch.float32)
dataset = get_dataset()
print("Model loaded.", flush=True)

# 1. Inspect partial.pt step 1000 routing values
partial_path = os.path.join(BASE_S5_DIR, "results_s7", "zstar_s7_a0.25_400.partial.pt")
d_partial = torch.load(partial_path, map_location="cpu")

print("\n" + "=" * 80, flush=True)
print("PART 1: ROUTING GATE VALUES AT STEP 1000 (from partial.pt)", flush=True)
print("=" * 80, flush=True)

for q in [4067, 1790, 2140]:
    k = f"q{q:04d}_a0.25"
    if k in d_partial:
        entry = d_partial[k]
        z_star = entry["z_star"].float()
        with torch.no_grad():
            alpha = routing_weights(z_star, w_g, 1.0)
            top_vals, top_idx = torch.topk(alpha, 6)
            gap_3_4 = float(top_vals[2] - top_vals[3])
        print(f"Q#{q:04d}: rho@1000={entry['rho_1000']:.6e} | min_rho={entry['min_rho']:.6e}", flush=True)
        print(f"  Top-6 modules: {top_idx.tolist()}", flush=True)
        print(f"  Top-6 weights: {[round(float(x), 6) for x in top_vals.tolist()]}", flush=True)
        print(f"  Gap (3rd - 4th module weight): {gap_3_4:.6e}", flush=True)
        print(flush=True)


# 2. Trace step-by-step routing transitions on CPU
print("=" * 80, flush=True)
print("PART 2: STEP-BY-STEP ROUTING VS RHO TRACE (First 120 steps)", flush=True)
print("=" * 80, flush=True)

def trace_item(q_idx, max_steps=120):
    p_text = dataset[q_idx]["prompt"]
    with torch.no_grad():
        ctx = bb.encode_context(p_text)
        k_latent = 64
        d = 2560
        z_src = bb.root_fwdpass(ctx, K=k_latent, output_dtype=torch.float32).to(device="cpu")
        c_x = compute_broadcast_context_c(ctx, K=k_latent, device="cpu", dtype=torch.float32)
        torch.manual_seed(2026)
        z0 = (z_src + torch.randn(k_latent, d, device="cpu", dtype=torch.float32) * 0.020).clone()
        c_hat_0 = compute_c_hat_0(c_x, z0).to(device="cpu", dtype=torch.float32)

        step_fn = InjectedTransitionStep(
            backbone=bb, context=ctx, w_g=w_g, c_hat_0=c_hat_0,
            g0=0.25, alpha=0.25, top_k=3, routing_mode="sparse"
        )

        z_cur = z0.clone()
        history = []
        prev_r_ids = None

        for k in range(1, max_steps + 1):
            z_next = step_fn(z_cur)
            diff = float(torch.norm(z_next - z_cur).item())
            norm_z = float(torch.norm(z_cur).item())
            rho = diff / max(norm_z, 1e-12)
            r_ids = list(step_fn.last_r_ids)
            r_changed = (prev_r_ids is not None and r_ids != prev_r_ids)

            alpha = routing_weights(z_cur, w_g, 1.0)
            top_vals, top_idx = torch.topk(alpha, 4)
            gap_3_4 = float(top_vals[2] - top_vals[3])

            history.append({
                "step": k,
                "rho": rho,
                "r_ids": r_ids,
                "r_changed": r_changed,
                "gap_3_4": gap_3_4,
            })
            prev_r_ids = r_ids
            z_cur = z_next

    return history


for q in [4067, 1790, 2140]:
    t0 = time.time()
    print(f"\n--- Tracing Q#{q:04d} (Steps 1 to 120) ---", flush=True)
    h = trace_item(q, 120)
    elapsed = time.time() - t0
    print(f"Completed 120 steps in {elapsed:.1f}s ({elapsed/120:.2f}s/step).", flush=True)

    # Find routing switches and large rho jumps
    print(f"\nSummary of Routing Switches and Rho Jumps for Q#{q:04d}:")
    print(f"{'Step':>5} | {'Rho':>12} | {'Prev Rho':>12} | {'Ratio':>8} | {'Active Modules':>16} | {'Gap 3-4':>10} | {'Switch?'}")
    print("-" * 80)
    for i in range(len(h)):
        rec = h[i]
        k = rec["step"]
        rho = rec["rho"]
        prev_rho = h[i-1]["rho"] if i > 0 else rho
        ratio = rho / max(prev_rho, 1e-12)
        r_changed = rec["r_changed"]
        # Print if routing changed, or ratio > 1.3 or ratio < 0.7, or key milestones
        if r_changed or ratio > 1.3 or k in [1, 10, 50, 100, 120]:
            sw_mark = "*** SWITCH ***" if r_changed else ""
            print(f"{k:5d} | {rho:12.6e} | {prev_rho:12.6e} | {ratio:7.2f}x | {str(rec['r_ids']):>16} | {rec['gap_3_4']:10.4e} | {sw_mark}", flush=True)

print("\nAll traces complete.", flush=True)

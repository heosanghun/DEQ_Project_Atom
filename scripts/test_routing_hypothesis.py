#!/usr/bin/env python3
"""scripts/test_routing_hypothesis.py - Test routing boundary hypothesis for Q#4067, 1790, 2140
Directive: 원장 #373 §4
"""

import os
import sys
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from harness_A_v1 import get_model, get_dataset
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights
from cts.deq.transition_inject import compute_broadcast_context_c, compute_c_hat_0, InjectedTransitionStep

print("Loading model on CPU...")
bb = get_model("cpu")
w_g = bb.routing_matrix().to(device="cpu", dtype=torch.float32)
dataset = get_dataset()

# 1. Inspect z_star routing states at step 1000 from partial_pt
partial_path = os.path.join(BASE_S5_DIR, "results_s7", "zstar_s7_a0.25_400.partial.pt")
d_partial = torch.load(partial_path, map_location="cpu")

print("\n" + "=" * 80)
print("PART 1: ROUTING GATE VALUES AT STEP 1000 (from partial.pt)")
print("=" * 80)

for q in [4067, 1790, 2140]:
    k = f"q{q:04d}_a0.25"
    if k in d_partial:
        entry = d_partial[k]
        z_star = entry["z_star"].float()
        alpha = routing_weights(z_star, w_g, 1.0)
        top_vals, top_idx = torch.topk(alpha, 6)
        gap_3_4 = float(top_vals[2] - top_vals[3])
        print(f"Q#{q:04d}: rho@1000={entry['rho_1000']:.6e} | min_rho={entry['min_rho']:.6e}")
        print(f"  Top-6 modules: {top_idx.tolist()}")
        print(f"  Top-6 weights: {[round(float(x), 6) for x in top_vals.tolist()]}")
        print(f"  Gap (3rd - 4th module weight): {gap_3_4:.6e}")
        print()

# 2. Trace step-by-step routing transitions on CPU (first 60 steps)
print("=" * 80)
print("PART 2: STEP-BY-STEP ROUTING VS RHO TRACE (First 50 steps)")
print("=" * 80)


def trace_item(q_idx, steps=50):
    p_text = dataset[q_idx]["prompt"]
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

    for k in range(1, steps + 1):
        z_next = step_fn(z_cur)
        diff = float(torch.norm(z_next - z_cur).item())
        norm_z = float(torch.norm(z_cur).item())
        rho = diff / max(norm_z, 1e-12)
        r_ids = list(step_fn.last_r_ids)
        r_changed = (prev_r_ids is not None and r_ids != prev_r_ids)
        history.append((k, rho, r_ids, r_changed))
        prev_r_ids = r_ids
        z_cur = z_next

    return history


for q in [4067, 1790, 2140]:
    print(f"\n--- Tracing Q#{q:04d} (Steps 1 to 40) ---")
    h = trace_item(q, 40)
    switches = [x for x in h if x[3]]
    print(f"Total routing switches in first 40 steps: {len(switches)}")
    for s in switches:
        # print previous step, switch step, next step
        k_sw = s[0]
        rho_before = h[k_sw - 2][1] if k_sw >= 2 else 0.0
        rho_sw = s[1]
        rho_ratio = rho_sw / max(rho_before, 1e-12)
        print(f"  Step {k_sw:02d}: modules={s[2]} | rho={rho_sw:.6e} (vs prev: {rho_before:.6e}, ratio={rho_ratio:.2f}x)")
    if not switches:
        print("  No routing switch occurred in first 40 steps (routing was static).")

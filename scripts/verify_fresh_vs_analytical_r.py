#!/usr/bin/env python3
"""
scripts/verify_fresh_vs_analytical_r.py - Compare r_analytical = rho / alpha vs fresh forward pass r_fresh
Directive: 원장 #375 §1
"""

import os
import sys
import json
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

s6_meta = json.load(open(f"{BASE_S5_DIR}/configs/s6_items_74.json"))
nonconv_54 = s6_meta["nonconv_54"]

pt_025 = torch.load(f"{BASE_S5_DIR}/results_s6/zstar_s6_a0.25.pt", map_location="cpu")

def eval_fresh_r(q_idx, z_star):
    p_text = dataset[q_idx]["prompt"]
    with torch.no_grad():
        ctx = bb.encode_context(p_text)
        k_latent = 64
        d = 2560
        z_src = bb.root_fwdpass(ctx, K=k_latent, output_dtype=torch.float32).to(device="cpu")
        c_x = compute_broadcast_context_c(ctx, K=k_latent, device="cpu", dtype=torch.float32)
        torch.manual_seed(2030)
        z0 = (z_src + torch.randn(k_latent, d, device="cpu", dtype=torch.float32) * 0.020).clone()
        c_hat_0 = compute_c_hat_0(c_x, z0).to(device="cpu", dtype=torch.float32)

        step_fn_f = InjectedTransitionStep(
            backbone=bb, context=ctx, w_g=w_g, c_hat_0=c_hat_0,
            g0=0.25, alpha=1.0, top_k=3, routing_mode="sparse"
        )
        z_in = z_star.float().to("cpu")
        f_z = step_fn_f(z_in)
        diff = float(torch.norm(f_z - z_in).item())
        norm_z = float(torch.norm(z_in).item())
        return diff / max(norm_z, 1e-12)

print("\nEvaluating fresh r for 54 nonconv items at alpha=0.25...")
t0 = time.time()
r_fresh_list = []
r_analyt_list = []
rho_list = []

for idx, q in enumerate(nonconv_54):
    k = f"q{q:04d}_a0.25"
    e = pt_025[k]
    z_star = e["z_star"]
    rho_1000 = float(e["rho_1000"])
    r_analyt = rho_1000 / 0.25
    r_fresh = eval_fresh_r(q, z_star)
    
    r_fresh_list.append(r_fresh)
    r_analyt_list.append(r_analyt)
    rho_list.append(rho_1000)
    
    if (idx + 1) % 10 == 0 or idx + 1 == len(nonconv_54):
        print(f"  [{idx+1:02d}/54] Q#{q:04d} | rho@1000={rho_1000:.4e} | r_analyt={r_analyt:.4e} | r_fresh={r_fresh:.4e} | diff={abs(r_analyt - r_fresh):.2e}", flush=True)

print(f"\nCompleted in {time.time() - t0:.1f}s.")

n = len(nonconv_54)
print("\n" + "=" * 80)
print("COMPARISON TABLE: rho vs r_analytical vs r_fresh (N=54 nonconv at alpha=0.25)")
print("=" * 80)
print(f"{'Metric':>20} | {'< 1e-3':>10} | {'< 5e-3':>10} | {'< 1e-2':>10} | {'Mean Value':>12}")
print("-" * 80)
print(f"{'rho@1000':>20} | {sum(1 for x in rho_list if x < 1e-3):6d}/{n} | {sum(1 for x in rho_list if x < 5e-3):6d}/{n} | {sum(1 for x in rho_list if x < 1e-2):6d}/{n} | {sum(rho_list)/n:12.6e}")
print(f"{'r_analyt = rho/0.25':>20} | {sum(1 for x in r_analyt_list if x < 1e-3):6d}/{n} | {sum(1 for x in r_analyt_list if x < 5e-3):6d}/{n} | {sum(1 for x in r_analyt_list if x < 1e-2):6d}/{n} | {sum(r_analyt_list)/n:12.6e}")
print(f"{'r_fresh = ||f(z)-z||':>20} | {sum(1 for x in r_fresh_list if x < 1e-3):6d}/{n} | {sum(1 for x in r_fresh_list if x < 5e-3):6d}/{n} | {sum(1 for x in r_fresh_list if x < 1e-2):6d}/{n} | {sum(r_fresh_list)/n:12.6e}")
print("-" * 80)

#!/usr/bin/env python3
"""scripts/trace_q2140_spike.py - Trace routing switches and rho around spike steps for Q#2140
Directive: 원장 #373 §4
"""

import os
import sys
import time
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

q_idx = 2140
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
prev_r_ids = None

print("Starting Q#2140 run up to step 105...")
for k in range(1, 106):
    z_next = step_fn(z_cur)
    diff = float(torch.norm(z_next - z_cur).item())
    norm_z = float(torch.norm(z_cur).item())
    rho = diff / max(norm_z, 1e-12)
    r_ids = list(step_fn.last_r_ids)
    r_changed = (prev_r_ids is not None and r_ids != prev_r_ids)

    alpha = routing_weights(z_cur.float(), w_g, 1.0)
    top_vals, top_idx = torch.topk(alpha, 4)
    gap_3_4 = float(top_vals[2] - top_vals[3])

    if r_changed or k in [1, 10, 50, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]:
        flag = "*** ROUTING SWITCH ***" if r_changed else ""
        print(f"Step {k:03d} | rho={rho:.6e} | modules={r_ids} | gap_3_4={gap_3_4:.4e} | {flag}", flush=True)

    prev_r_ids = r_ids
    z_cur = z_next

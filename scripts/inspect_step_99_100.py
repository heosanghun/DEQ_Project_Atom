#!/usr/bin/env python3
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

bb = get_model("cpu")
w_g = bb.routing_matrix().to(device="cpu", dtype=torch.float32)
dataset = get_dataset()

p_text = dataset[2140]["prompt"]
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
for k in range(1, 102):
    z_next = step_fn(z_cur)
    diff = float(torch.norm(z_next - z_cur).item())
    norm_z = float(torch.norm(z_cur).item())
    rho = diff / max(norm_z, 1e-12)
    if k in [98, 99, 100, 101]:
        print(f"k={k:03d} | diff={diff:.6f} | norm_z={norm_z:.6f} | rho={rho:.6e}")
        alpha = routing_weights(z_cur.float(), w_g, 1.0)
        top4_v, top4_i = torch.topk(alpha, 4)
        print(f"  alpha top-4: idx={top4_i.tolist()}, val={[round(float(x), 6) for x in top4_v.tolist()]}")
    z_cur = z_next

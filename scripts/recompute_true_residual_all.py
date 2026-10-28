#!/usr/bin/env python3
"""
scripts/recompute_true_residual_all.py - Recompute true fixed-point residual r = ||f(z) - z|| / ||z||
Directive: 원장 #375 §1
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F

torch.set_num_threads(8)

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from harness_A_v1 import get_model, get_dataset
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights
from cts.deq.transition_inject import compute_broadcast_context_c, compute_c_hat_0, InjectedTransitionStep

print("Loading model on CPU...", flush=True)
bb = get_model("cpu")
w_g = bb.routing_matrix().to(device="cpu", dtype=torch.float32)
dataset = get_dataset()
print("Model loaded.", flush=True)

# Load S6 item partition
s6_items_path = os.path.join(BASE_S5_DIR, "configs", "s6_items_74.json")
with open(s6_items_path, "r", encoding="utf-8") as f:
    s6_meta = json.load(f)

nonconv_54 = s6_meta["nonconv_54"]
control_20 = s6_meta["control_20"]
all_74 = nonconv_54 + control_20

# Load saved tensors
pt_s6_a025 = torch.load(os.path.join(BASE_S5_DIR, "results_s6", "zstar_s6_a0.25.pt"), map_location="cpu")
pt_s6_a035 = torch.load(os.path.join(BASE_S5_DIR, "results_s6", "zstar_s6_a0.35.pt"), map_location="cpu")
pt_s6_a050_c20 = torch.load(os.path.join(BASE_S5_DIR, "results_s6", "zstar_s6_a0.50_control20.pt"), map_location="cpu")
pt_s5_400 = torch.load(os.path.join(BASE_S5_DIR, "results_s5", "zstar_s5_400.pt"), map_location="cpu")

# Function to compute fresh true residual r = ||f(z*) - z*|| / ||z*||
def eval_fresh_residual(q_idx, z_star):
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

        # InjectedTransitionStep with alpha=1.0 gives f(z) = Phi(z) + g0 * c_hat_0
        step_fn_f = InjectedTransitionStep(
            backbone=bb, context=ctx, w_g=w_g, c_hat_0=c_hat_0,
            g0=0.25, alpha=1.0, top_k=3, routing_mode="sparse"
        )
        z_in = z_star.float().to("cpu")
        f_z = step_fn_f(z_in)
        diff = float(torch.norm(f_z - z_in).item())
        norm_z = float(torch.norm(z_in).item())
        r_fresh = diff / max(norm_z, 1e-12)
    return r_fresh

# 1. Analyze S6 conditions for nonconv 54 and control 20
print("\n" + "=" * 80)
print("PART 1: S6 DAMPING RUNS — TRUE RESIDUAL ANALYSIS")
print("=" * 80, flush=True)

conditions = [
    ("alpha=0.50", 0.50, pt_s5_400, pt_s6_a050_c20),
    ("alpha=0.35", 0.35, pt_s6_a035, pt_s6_a035),
    ("alpha=0.25", 0.25, pt_s6_a025, pt_s6_a025),
]

s6_results = {}

for label, alpha_val, pt_nc, pt_ctrl in conditions:
    t0 = time.time()
    print(f"\nProcessing {label}...", flush=True)
    res_list = []
    for q in all_74:
        is_nc = q in nonconv_54
        pt_dict = pt_nc if is_nc else pt_ctrl
        
        # Find key
        k_candidates = [f"q{q:04d}_a{alpha_val:.2f}", f"q{q:04d}_a{alpha_val}", f"q{q:04d}", q]
        entry = None
        for cand in k_candidates:
            if cand in pt_dict:
                entry = pt_dict[cand]
                break
        if entry is None:
            # fallback for S5 dictionary format
            for k_iter in pt_dict:
                if f"q{q:04d}" in str(k_iter) or str(q) == str(k_iter):
                    entry = pt_dict[k_iter]
                    break
        
        if entry is None:
            print(f"Warning: Q#{q:04d} not found in {label}!")
            continue
        
        z_star = entry["z_star"]
        rho_1000 = float(entry.get("rho_1000", entry.get("rho", 0.0)))
        rho_final = float(entry.get("rho", entry.get("final_rho_fp", 0.0)))
        
        # r = rho / alpha
        r_step_1000 = rho_1000 / alpha_val
        r_step_final = rho_final / alpha_val
        
        # fresh evaluation of ||f(z*) - z*|| / ||z*||
        r_fresh = eval_fresh_residual(q, z_star)
        
        res_list.append({
            "q": q,
            "is_nc": is_nc,
            "rho_1000": rho_1000,
            "rho_final": rho_final,
            "r_step_1000": r_step_1000,
            "r_step_final": r_step_final,
            "r_fresh": r_fresh,
        })
    
    elapsed = time.time() - t0
    s6_results[label] = res_list
    print(f"Completed {label} in {elapsed:.1f}s.")

# Tabulate resolution counts on Nonconv 54
print("\n" + "=" * 80)
print("S6 NONCONV 54 RESOLUTION TABLE: RHO vs TRUE RESIDUAL r")
print("=" * 80)

print(f"{'Condition':>12} | {'Metric':>15} | {'< 1e-3':>10} | {'< 5e-3':>10} | {'< 1e-2':>10} | {'Mean Value':>12}")
print("-" * 80)

for label, alpha_val, _, _ in conditions:
    recs = [r for r in s6_results[label] if r["is_nc"]]
    n_total = len(recs)
    
    # 1. Logged rho@1000
    c_rho_1e3 = sum(1 for r in recs if r["rho_1000"] < 1e-3)
    c_rho_5e3 = sum(1 for r in recs if r["rho_1000"] < 5e-3)
    c_rho_1e2 = sum(1 for r in recs if r["rho_1000"] < 1e-2)
    mean_rho = sum(r["rho_1000"] for r in recs) / n_total
    print(f"{label:>12} | {'rho@1000':>15} | {c_rho_1e3:6d}/{n_total} | {c_rho_5e3:6d}/{n_total} | {c_rho_1e2:6d}/{n_total} | {mean_rho:12.6e}")
    
    # 2. r_step@1000 = rho@1000 / alpha
    c_rstep_1e3 = sum(1 for r in recs if r["r_step_1000"] < 1e-3)
    c_rstep_5e3 = sum(1 for r in recs if r["r_step_1000"] < 5e-3)
    c_rstep_1e2 = sum(1 for r in recs if r["r_step_1000"] < 1e-2)
    mean_rstep = sum(r["r_step_1000"] for r in recs) / n_total
    print(f"{'':>12} | {'r = rho/alpha':>15} | {c_rstep_1e3:6d}/{n_total} | {c_rstep_5e3:6d}/{n_total} | {c_rstep_1e2:6d}/{n_total} | {mean_rstep:12.6e}")
    
    # 3. r_fresh = ||f(z*) - z*|| / ||z*||
    c_rfresh_1e3 = sum(1 for r in recs if r["r_fresh"] < 1e-3)
    c_rfresh_5e3 = sum(1 for r in recs if r["r_fresh"] < 5e-3)
    c_rfresh_1e2 = sum(1 for r in recs if r["r_fresh"] < 1e-2)
    mean_rfresh = sum(r["r_fresh"] for r in recs) / n_total
    print(f"{'':>12} | {'r_fresh(z*)':>15} | {c_rfresh_1e3:6d}/{n_total} | {c_rfresh_5e3:6d}/{n_total} | {c_rfresh_1e2:6d}/{n_total} | {mean_rfresh:12.6e}")
    print("-" * 80)


# Tabulate Control 20
print("\n" + "=" * 80)
print("S6 CONTROL 20 TABLE: RHO vs TRUE RESIDUAL r")
print("=" * 80)
print(f"{'Condition':>12} | {'Metric':>15} | {'< 1e-3':>10} | {'< 5e-3':>10} | {'< 1e-2':>10} | {'Mean Value':>12}")
print("-" * 80)
for label, alpha_val, _, _ in conditions:
    recs = [r for r in s6_results[label] if not r["is_nc"]]
    n_total = len(recs)
    c_rho_5e3 = sum(1 for r in recs if r["rho_1000"] < 5e-3)
    c_rstep_5e3 = sum(1 for r in recs if r["r_step_1000"] < 5e-3)
    c_rfresh_5e3 = sum(1 for r in recs if r["r_fresh"] < 5e-3)
    mean_rfresh = sum(r["r_fresh"] for r in recs) / n_total
    print(f"{label:>12} | {'r_fresh(z*)':>15} | {sum(1 for r in recs if r['r_fresh'] < 1e-3):6d}/{n_total} | {c_rfresh_5e3:6d}/{n_total} | {sum(1 for r in recs if r['r_fresh'] < 1e-2):6d}/{n_total} | {mean_rfresh:12.6e}")


# 2. Analyze S5 400 items
print("\n" + "=" * 80)
print("PART 2: S5 400 ITEMS — TRUE RESIDUAL ANALYSIS (alpha=0.50)")
print("=" * 80, flush=True)

s5_400_results = []
t0 = time.time()
print(f"Evaluating S5 400 items on CPU...", flush=True)

for idx, (k_iter, entry) in enumerate(pt_s5_400.items()):
    q = entry.get("meta", {}).get("question_idx", idx)
    z_star = entry["z_star"]
    rho_fp = float(entry.get("rho", entry.get("final_rho_fp", 0.0)))
    
    # In S5, alpha = 0.50
    r_step = rho_fp / 0.50
    r_fresh = eval_fresh_residual(q, z_star)
    
    s5_400_results.append({
        "q": q,
        "rho": rho_fp,
        "r_step": r_step,
        "r_fresh": r_fresh,
    })
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/400 items ({time.time() - t0:.1f}s)...", flush=True)

print(f"Completed S5 400 items in {time.time() - t0:.1f}s.")

print("\n" + "=" * 80)
print("S5 400 ITEMS SUMMARY TABLE: RHO vs TRUE RESIDUAL r")
print("=" * 80)
print(f"{'Metric':>20} | {'< 1e-3':>12} | {'< 5e-3':>12} | {'< 1e-2':>12} | {'Mean Value':>12}")
print("-" * 80)

n_400 = len(s5_400_results)
c_rho_1e3 = sum(1 for r in s5_400_results if r["rho"] < 1e-3)
c_rho_5e3 = sum(1 for r in s5_400_results if r["rho"] < 5e-3)
c_rho_1e2 = sum(1 for r in s5_400_results if r["rho"] < 1e-2)
mean_rho = sum(r["rho"] for r in s5_400_results) / n_400
print(f"{'rho (logged)':>20} | {c_rho_1e3:8d}/{n_400} | {c_rho_5e3:8d}/{n_400} | {c_rho_1e2:8d}/{n_400} | {mean_rho:12.6e}")

c_rstep_1e3 = sum(1 for r in s5_400_results if r["r_step"] < 1e-3)
c_rstep_5e3 = sum(1 for r in s5_400_results if r["r_step"] < 5e-3)
c_rstep_1e2 = sum(1 for r in s5_400_results if r["r_step"] < 1e-2)
mean_rstep = sum(r["r_step"] for r in s5_400_results) / n_400
print(f"{'r = rho / 0.50':>20} | {c_rstep_1e3:8d}/{n_400} | {c_rstep_5e3:8d}/{n_400} | {c_rstep_1e2:8d}/{n_400} | {mean_rstep:12.6e}")

c_rfresh_1e3 = sum(1 for r in s5_400_results if r["r_fresh"] < 1e-3)
c_rfresh_5e3 = sum(1 for r in s5_400_results if r["r_fresh"] < 5e-3)
c_rfresh_1e2 = sum(1 for r in s5_400_results if r["r_fresh"] < 1e-2)
mean_rfresh = sum(r["r_fresh"] for r in s5_400_results) / n_400
print(f"{'r_fresh(z*)':>20} | {c_rfresh_1e3:8d}/{n_400} | {c_rfresh_5e3:8d}/{n_400} | {c_rfresh_1e2:8d}/{n_400} | {mean_rfresh:12.6e}")
print("-" * 80)

# Save results JSON
out_json_path = os.path.join(BASE_S5_DIR, "results_s6", "s6_s5_true_residual_audit.json")
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "s6": s6_results,
        "s5_400": s5_400_results,
    }, f, indent=2)
print(f"\nSaved detailed results to {out_json_path}")

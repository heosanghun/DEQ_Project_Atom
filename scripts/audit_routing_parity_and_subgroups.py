#!/usr/bin/env python3
"""
scripts/audit_routing_parity_and_subgroups.py
Audit Routing Parity and Converged Subgroups (Ledger #378 Point 2)

Requirements:
(a) Compare inline routing (z.mean -> W_g -> softmax) vs canonical routing_weights on 10 items (max |Delta|).
    Report max |W_g(m1_final_frozen) - bb.routing_matrix()| value.
(b) Divide 346 converged items into:
    - Group 1: r < 5e-3 (N=303)
    - Group 2: 5e-3 <= r < 1e-2 (N=43)
    and compare with
    - Non-converged: r >= 1e-2 (N=54)
    Calculate: median, quartiles (Q1, Q3, IQR), min, max, and boundary proximity (diff34 < 0.005) count & percentage.
"""

import os
import sys
import json
import torch
import numpy as np

torch.set_num_threads(8)

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

CKPT_PATH = "/home/shoon/checkpoints/m1_final_frozen.pt"
S5_ZSTAR_PATH = "/home/shoon/cts_e1_s5/results_s5/zstar_s5_400.pt"

def main():
    print("=" * 80)
    print("AUDIT: ROUTING PARITY & POPULATION SUBGROUPS (LEDGER #378 POINT 2)")
    print("=" * 80)

    # 1. Load W_g from checkpoint directly
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt.get("backbone_state_dict", ckpt)
    W_orig = sd["routing_proj.parametrizations.weight.original"]
    u = sd["routing_proj.parametrizations.weight.0._u"]
    v = sd["routing_proj.parametrizations.weight.0._v"]
    sigma = torch.dot(u, torch.mv(W_orig, v))
    w_g_ckpt = (W_orig / sigma).float()

    # Load backbone routing matrix via harness
    from harness_A_v1 import get_model
    bb = get_model("cpu")
    w_g_bb = bb.routing_matrix().to(device="cpu", dtype=torch.float32)

    max_delta_wg = torch.max(torch.abs(w_g_ckpt - w_g_bb)).item()
    print(f"\n[PART A-1] W_g Checkpoint vs Backbone Routing Matrix Parity:")
    print(f"  Checkpoint W_g shape: {list(w_g_ckpt.shape)}, Frobenius norm: {torch.norm(w_g_ckpt):.6f}")
    print(f"  Backbone   W_g shape: {list(w_g_bb.shape)}, Frobenius norm: {torch.norm(w_g_bb):.6f}")
    print(f"  max |W_g(m1_final_frozen) - bb.routing_matrix()| = {max_delta_wg:.10e}")
    assert max_delta_wg < 1e-6, f"W_g mismatch: {max_delta_wg}"

    # 2. Check 10 items numerical parity: inline vs canonical routing_weights
    from cts.routing.sparse_moe_ref import routing_weights
    s5 = torch.load(S5_ZSTAR_PATH, map_location="cpu")

    sample_keys = list(s5.keys())[:10]
    print(f"\n[PART A-2] Numerical Comparison on 10 Sample Items (Inline vs Canonical):")
    print(f"{'Item Key':<15} | {'max |Delta|':<18} | {'Inline Top-3':<15} | {'Canonical Top-3':<15}")
    print("-" * 75)

    max_delta_all_10 = 0.0
    for k in sample_keys:
        rec = s5[k]
        z = rec["z_star"].float()  # [64, 2560]
        
        # Inline routing
        pooled = z.mean(dim=-2)
        inline_logits = torch.matmul(pooled, w_g_bb.t())
        inline_alpha = torch.softmax(inline_logits, dim=-1)

        # Canonical routing
        canon_alpha = routing_weights(z, w_g_bb, nu_temp=1.0)

        delta = torch.max(torch.abs(inline_alpha - canon_alpha)).item()
        if delta > max_delta_all_10:
            max_delta_all_10 = delta

        inline_top3 = torch.topk(inline_alpha, 3).indices.tolist()
        canon_top3 = torch.topk(canon_alpha, 3).indices.tolist()
        print(f"{str(k):<15} | {delta:<18.10e} | {str(inline_top3):<15} | {str(canon_top3):<15}")

    print("-" * 75)
    print(f"10-Item Max Parity Difference: max|inline_alpha - canonical_alpha| = {max_delta_all_10:.10e}")

    # 3. Population Subgroup Analysis
    # S5 alpha = 0.50, so true residual r = rho / 0.50
    # Categorize all 400 items:
    # Subgroup 1: r < 5e-3 (Strict Converged)
    # Subgroup 2: 5e-3 <= r < 1e-2 (S5-Equivalent Converged)
    # Subgroup 3: r >= 1e-2 (Non-converged)
    g1_diffs = []  # r < 5e-3
    g2_diffs = []  # 5e-3 <= r < 1e-2
    g3_diffs = []  # r >= 1e-2 (nonconv)

    g1_keys = []
    g2_keys = []
    g3_keys = []

    for k, rec in s5.items():
        z = rec["z_star"].float()
        pooled = z.mean(dim=-2)
        logits = torch.matmul(pooled, w_g_bb.t())
        alpha = torch.softmax(logits, dim=-1)
        sorted_alpha, _ = torch.sort(alpha, descending=True)
        w3 = float(sorted_alpha[2].item())
        w4 = float(sorted_alpha[3].item())
        diff34 = w3 - w4

        rho = float(rec.get("rho", rec.get("final_rho_fp", 0.0)))
        r = rho / 0.50

        if r < 5.0e-3:
            g1_diffs.append(diff34)
            g1_keys.append(k)
        elif r < 1.0e-2:
            g2_diffs.append(diff34)
            g2_keys.append(k)
        else:
            g3_diffs.append(diff34)
            g3_keys.append(k)

    print(f"\n[PART B] S5 400 Item Counts by True Residual r (rho / 0.50):")
    print(f"  Subgroup 1 (Strict Converged, r < 5e-3):            N = {len(g1_diffs):3d} (expected 303)")
    print(f"  Subgroup 2 (S5-Equiv Converged, 5e-3 <= r < 1e-2):   N = {len(g2_diffs):3d} (expected 43)")
    print(f"  Converged Total (r < 1e-2):                         N = {len(g1_diffs) + len(g2_diffs):3d} (expected 346)")
    print(f"  Subgroup 3 (Non-converged, r >= 1e-2):              N = {len(g3_diffs):3d} (expected 54)")
    print(f"  Grand Total:                                        N = {len(g1_diffs) + len(g2_diffs) + len(g3_diffs):3d} (expected 400)")

    def get_stats(arr):
        a = np.array(arr)
        med = float(np.median(a))
        q1 = float(np.percentile(a, 25))
        q3 = float(np.percentile(a, 75))
        iqr = q3 - q1
        mean = float(np.mean(a))
        std = float(np.std(a))
        min_v = float(np.min(a))
        max_v = float(np.max(a))
        near_boundary = int(np.sum(a < 0.005))
        near_pct = float(near_boundary / len(a) * 100.0)
        return {
            "N": len(a),
            "median": med,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "near_boundary_count": near_boundary,
            "near_boundary_pct": near_pct
        }

    s1 = get_stats(g1_diffs)
    s2 = get_stats(g2_diffs)
    s_conv = get_stats(g1_diffs + g2_diffs)
    s3 = get_stats(g3_diffs)

    print(f"\n[PART B-2] Population Statistics of 3rd - 4th Routing Weight Gap (diff34):")
    print("-" * 110)
    print(f"{'Group':<35} | {'N':>4} | {'Median':>10} | {'Q1':>10} | {'Q3':>10} | {'IQR':>10} | {'Near (<0.005)':>18}")
    print("-" * 110)
    print(f"{'1. Strict Conv (r < 5e-3)':<35} | {s1['N']:4d} | {s1['median']:10.6f} | {s1['q1']:10.6f} | {s1['q3']:10.6f} | {s1['iqr']:10.6f} | {s1['near_boundary_count']:4d}/{s1['N']} ({s1['near_boundary_pct']:5.2f}%)")
    print(f"{'2. S5-Equiv Conv (5e-3 <= r < 1e-2)':<35} | {s2['N']:4d} | {s2['median']:10.6f} | {s2['q1']:10.6f} | {s2['q3']:10.6f} | {s2['iqr']:10.6f} | {s2['near_boundary_count']:4d}/{s2['N']} ({s2['near_boundary_pct']:5.2f}%)")
    print(f"{'   -> All Conv Total (r < 1e-2)':<35} | {s_conv['N']:4d} | {s_conv['median']:10.6f} | {s_conv['q1']:10.6f} | {s_conv['q3']:10.6f} | {s_conv['iqr']:10.6f} | {s_conv['near_boundary_count']:4d}/{s_conv['N']} ({s_conv['near_boundary_pct']:5.2f}%)")
    print(f"{'3. Non-converged (r >= 1e-2)':<35} | {s3['N']:4d} | {s3['median']:10.6f} | {s3['q1']:10.6f} | {s3['q3']:10.6f} | {s3['iqr']:10.6f} | {s3['near_boundary_count']:4d}/{s3['N']} ({s3['near_boundary_pct']:5.2f}%)")
    print("-" * 110)

    ratio_s3_vs_s1 = s3['near_boundary_pct'] / s1['near_boundary_pct']
    ratio_s3_vs_s2 = s3['near_boundary_pct'] / s2['near_boundary_pct']
    ratio_s3_vs_conv = s3['near_boundary_pct'] / s_conv['near_boundary_pct']
    print(f"\nBoundary Proximity (< 0.005) Odds / Ratio:")
    print(f"  Non-converged vs Strict Converged:       {s3['near_boundary_pct']:.2f}% vs {s1['near_boundary_pct']:.2f}% ({ratio_s3_vs_s1:.2f}x higher)")
    print(f"  Non-converged vs S5-Equiv Converged:     {s3['near_boundary_pct']:.2f}% vs {s2['near_boundary_pct']:.2f}% ({ratio_s3_vs_s2:.2f}x higher)")
    print(f"  Non-converged vs All Converged (346):    {s3['near_boundary_pct']:.2f}% vs {s_conv['near_boundary_pct']:.2f}% ({ratio_s3_vs_conv:.2f}x higher)")

    # Save summary json
    summary = {
        "max_delta_wg": max_delta_wg,
        "max_delta_alpha_10_items": max_delta_all_10,
        "strict_conv_303": s1,
        "s5_equiv_conv_43": s2,
        "all_conv_346": s_conv,
        "nonconv_54": s3,
        "proximity_ratio_vs_strict": ratio_s3_vs_s1,
        "proximity_ratio_vs_equiv": ratio_s3_vs_s2,
        "proximity_ratio_vs_all_conv": ratio_s3_vs_conv,
    }
    with open("/home/shoon/cts_e1_s5/results_s7/routing_subgroup_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary to /home/shoon/cts_e1_s5/results_s7/routing_subgroup_analysis.json")

if __name__ == "__main__":
    main()

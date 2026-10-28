#!/usr/bin/env python3
"""
scripts/analyze_spectral_scatter.py
Analyze Spectral Radius (74 items) & Routing Proximity Scatter Relationship (Ledger #378 Point 3)

Calculates:
1. Dominant eigenvalue sign and magnitude distribution for nonconv 54 and control 20.
2. Exact matching of each item's spectral radius with its 3rd-4th routing weight gap (diff34).
3. Pearson and Spearman rank correlation between spectral radius and diff34.
4. Quadrant breakdown: contractive (<1.0) vs boundary proximity (<0.005).
"""

import os
import sys
import json
import torch
import numpy as np
from scipy import stats

torch.set_num_threads(8)

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

SPECTRAL_JSON = "/home/shoon/cts_e1_s5/results_s7/spectral_radius_74.json"
CKPT_PATH = "/home/shoon/checkpoints/m1_final_frozen.pt"
S5_ZSTAR_PATH = "/home/shoon/cts_e1_s5/results_s5/zstar_s5_400.pt"

def main():
    print("=" * 80)
    print("SPECTRAL RADIUS (74 ITEMS) & ROUTING PROXIMITY SCATTER ANALYSIS (LEDGER #378 POINT 3)")
    print("=" * 80)

    # 1. Load spectral radius results
    with open(SPECTRAL_JSON, "r") as f:
        spec_data = json.load(f)
    print(f"Loaded {len(spec_data)} items from {SPECTRAL_JSON}")

    # 2. Load W_g to compute diff34 for each item
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt.get("backbone_state_dict", ckpt)
    W_orig = sd["routing_proj.parametrizations.weight.original"]
    u = sd["routing_proj.parametrizations.weight.0._u"]
    v = sd["routing_proj.parametrizations.weight.0._v"]
    sigma = torch.dot(u, torch.mv(W_orig, v))
    w_g = (W_orig / sigma).float()

    s5 = torch.load(S5_ZSTAR_PATH, map_location="cpu")

    # Match spectral items with S5 z_star
    matched_records = []
    for item in spec_data:
        qid = item["qid"]
        group = item["group"]
        cand_keys = [
            f"q{qid}_s2030",
            f"q{qid:04d}_s2030",
            f"q{qid}",
            f"q{qid:04d}",
            f"q{qid}_diag",
            f"q{qid:04d}_diag"
        ]
        found_rec = None
        for ck in cand_keys:
            if ck in s5:
                found_rec = s5[ck]
                break
        if found_rec is None:
            for k in s5:
                # Exact numeric match in key
                k_clean = str(k).split("_")[0].replace("q", "")
                if k_clean.isdigit() and int(k_clean) == int(qid):
                    found_rec = s5[k]
                    break
        
        assert found_rec is not None, f"Could not find Q#{qid} in S5 z_star"

        z = found_rec["z_star"].float()
        pooled = z.mean(dim=-2)
        logits = torch.matmul(pooled, w_g.t())
        alpha = torch.softmax(logits, dim=-1)
        sorted_alpha, _ = torch.sort(alpha, descending=True)
        w3 = float(sorted_alpha[2].item())
        w4 = float(sorted_alpha[3].item())
        diff34 = w3 - w4

        item_rec = {
            "qid": qid,
            "group": group,
            "true_r": item["true_r"],
            "dominant_lambda": item["dominant_lambda"],
            "spectral_radius": item["spectral_radius"],
            "sign": item["sign"],
            "w3": w3,
            "w4": w4,
            "diff34": diff34,
            "near_boundary": diff34 < 0.005,
            "contractive": item["spectral_radius"] < 1.0
        }
        matched_records.append(item_rec)

    # Separate nonconv 54 and control 20
    nonconv = [r for r in matched_records if r["group"] == "nonconv54"]
    control = [r for r in matched_records if r["group"] == "control20"]
    all_74 = matched_records

    print(f"\n[PART 1] Dominant Eigenvalue Sign & Magnitude Distribution:")
    print("-" * 95)
    print(f"{'Group':<15} | {'N':>4} | {'Sign (POS)':>12} | {'Contractive (<1.0)':>20} | {'Median |lambda|':>15} | {'Range [Min, Max]':>20}")
    print("-" * 95)

    def print_dist(name, group_recs):
        n = len(group_recs)
        pos_cnt = sum(1 for r in group_recs if r["sign"] == "POS")
        pos_pct = pos_cnt / n * 100
        contr_cnt = sum(1 for r in group_recs if r["contractive"])
        contr_pct = contr_cnt / n * 100
        l_vals = [r["spectral_radius"] for r in group_recs]
        med = float(np.median(l_vals))
        min_v = float(np.min(l_vals))
        max_v = float(np.max(l_vals))
        print(f"{name:<15} | {n:4d} | {pos_cnt:4d}/{n} ({pos_pct:5.1f}%) | {contr_cnt:4d}/{n} ({contr_pct:5.1f}%) | {med:15.6f} | [{min_v:.4f}, {max_v:.4f}]")

    print_dist("Nonconv 54", nonconv)
    print_dist("Control 20", control)
    print_dist("Total 74", all_74)
    print("-" * 95)

    # Detailed percentiles for nonconv 54
    nc_lambdas = [r["spectral_radius"] for r in nonconv]
    print(f"\nNonconv 54 Spectral Radius Percentiles:")
    print(f"  Min:    {np.min(nc_lambdas):.6f}")
    print(f"  P25:    {np.percentile(nc_lambdas, 25):.6f}")
    print(f"  Median: {np.median(nc_lambdas):.6f}")
    print(f"  P75:    {np.percentile(nc_lambdas, 75):.6f}")
    print(f"  Max:    {np.max(nc_lambdas):.6f}")
    print(f"  IQR:    {np.percentile(nc_lambdas, 75) - np.percentile(nc_lambdas, 25):.6f}")

    # Outlier check: items with rho(J) >= 1.0
    outliers = [r for r in nonconv if not r["contractive"]]
    print(f"\nNon-contractive Outliers (spectral_radius >= 1.0): N = {len(outliers)}")
    for o in outliers:
        print(f"  Q#{o['qid']}: spectral_radius = {o['spectral_radius']:.6f}, diff34 = {o['diff34']:.6f}, true_r = {o['true_r']:.6f}")

    # 3. Correlation between Spectral Radius and diff34
    print(f"\n[PART 2] Scatter & Correlation: Spectral Radius (|lambda|) vs Routing Gap (diff34):")
    print("-" * 80)

    def calc_corr(name, group_recs):
        x = [r["diff34"] for r in group_recs]
        y = [r["spectral_radius"] for r in group_recs]
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        print(f"{name} (N={len(group_recs)}):")
        print(f"  Pearson  r = {pearson_r:+.4f} (p = {pearson_p:.4e})")
        print(f"  Spearman rho = {spearman_rho:+.4f} (p = {spearman_p:.4e})")

    calc_corr("Nonconv 54", nonconv)
    calc_corr("Control 20", control)
    calc_corr("Combined 74", all_74)

    # 4. Quadrant Breakdown (Contingency Table)
    print(f"\n[PART 3] Quadrant Breakdown: Contraction vs Boundary Proximity:")
    print("-" * 80)

    def print_quadrants(name, group_recs):
        n = len(group_recs)
        q1 = sum(1 for r in group_recs if r["contractive"] and not r["near_boundary"])
        q2 = sum(1 for r in group_recs if r["contractive"] and r["near_boundary"])
        q3 = sum(1 for r in group_recs if not r["contractive"] and not r["near_boundary"])
        q4 = sum(1 for r in group_recs if not r["contractive"] and r["near_boundary"])
        print(f"{name} (N={n}):")
        print(f"  Quadrant A [Contractive (<1.0) & Far   (>=0.005)]: {q1:3d} ({q1/n*100:5.1f}%)")
        print(f"  Quadrant B [Contractive (<1.0) & Near  (<0.005) ]: {q2:3d} ({q2/n*100:5.1f}%)  <-- Trap Zone (Chattering)")
        print(f"  Quadrant C [Expansive   (>=1.0) & Far  (>=0.005)]: {q3:3d} ({q3/n*100:5.1f}%)")
        print(f"  Quadrant D [Expansive   (>=1.0) & Near (<0.005) ]: {q4:3d} ({q4/n*100:5.1f}%)")

    print_quadrants("Nonconv 54", nonconv)
    print_quadrants("Control 20", control)
    print_quadrants("Combined 74", all_74)

    # Save detailed records to JSON
    out_path = "/home/shoon/cts_e1_s5/results_s7/spectral_routing_scatter_74.json"
    with open(out_path, "w") as f:
        json.dump({
            "records": matched_records,
            "nonconv_count": len(nonconv),
            "control_count": len(control)
        }, f, indent=2)
    print(f"\nSaved matched scatter records to {out_path}")

if __name__ == "__main__":
    main()

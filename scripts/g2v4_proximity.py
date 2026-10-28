#!/usr/bin/env python3
"""scripts/g2v4_proximity.py - PREREG_S7_G2v4 §4-7 report item (not a gate; CC 2026-10-11).
Per confirm item: A: ||T_A(z*'0) - z*'0|| / ||z*'0||  (fresh_A_latent vs fresh_zsrcinj_1758 z_star)
                  E: ||T_E(z_src) - z_src|| / ||z_src||  (fresh_E_latent vs fresh_zsrc_1758 z_star field = z_src)
Prints n, median, p10, p90, mean for each. Reads only latent tensors (no text, no generations)."""
import json, statistics as st, sys, torch
sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_eval_latents_v2 import extract_tensor
G = "results_s7/g2v4"
q = json.load(open("configs/g2v4b_fresh.json"))["indices"]
zs = torch.load(f"{G}/fresh_zsrcinj_1758.pt", map_location="cpu"); zr = torch.load(f"{G}/fresh_zsrc_1758.pt", map_location="cpu")
la = torch.load(f"{G}/fresh_A_latent.pt", map_location="cpu"); le = torch.load(f"{G}/fresh_E_latent.pt", map_location="cpu")
def rel(a, b): a, b = a.double().reshape(-1), b.double().reshape(-1); return float((a - b).norm() / b.norm())
A = [rel(extract_tensor(la, i), zs[f"q{i:04d}_a0.25"]["z_star"]) for i in q]
E = [rel(extract_tensor(le, i), zr[f"q{i:04d}_a0.25"]["z_star"]) for i in q]
for name, x in (("A ||T_A(z*'0)-z*'0||/||z*'0||", A), ("E ||T_E(z_src)-z_src||/||z_src||", E)):
    s = sorted(x); n = len(s)
    print(f"[PROX] {name}: n {n} median {st.median(s):.4f} p10 {s[n // 10]:.4f} p90 {s[9 * n // 10]:.4f} mean {st.mean(s):.4f}")

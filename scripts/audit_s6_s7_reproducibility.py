#!/usr/bin/env python3
"""
scripts/audit_s6_s7_reproducibility.py - Compare S6 and S7 results for 8 former nonconv items
Directive: 원장 #374 §1
"""

import os
import sys
import torch
import torch.nn.functional as F

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
s6_path = os.path.join(BASE_S5_DIR, "results_s6", "zstar_s6_a0.25.pt")
s7_path = os.path.join(BASE_S5_DIR, "results_s7", "zstar_s7_a0.25_400.partial.pt")

d6 = torch.load(s6_path, map_location="cpu")
d7 = torch.load(s7_path, map_location="cpu")

target_items = [424, 1177, 1790, 1844, 2076, 2215, 2573, 2844]

print(f"Loaded S6 items: {len(d6)}")
print(f"Loaded S7 items: {len(d7)}")
print()

header = f"{'Q#':>6} | {'S6 rho@1000':>13} | {'S7 rho@1000':>13} | {'Diff rho':>11} | {'S6 NFE':>7} | {'S7 NFE':>7} | {'cos(z_S6, z_S7)':>16}"
print(header)
print("-" * len(header))

for q in target_items:
    k6 = f"q{q:04d}_a0.25"
    k7 = f"q{q:04d}_a0.25"
    
    in_s6 = k6 in d6
    in_s7 = k7 in d7
    
    if not in_s6 or not in_s7:
        print(f"Q#{q:04d} | in_s6={in_s6} | in_s7={in_s7} | NOT IN BOTH")
        continue
    
    e6 = d6[k6]
    e7 = d7[k7]
    
    r6 = float(e6["rho_1000"])
    r7 = float(e7["rho_1000"])
    diff_r = abs(r6 - r7)
    
    nfe6 = e6["meta"]["nfe"]
    nfe7 = e7["meta"]["nfe"]
    
    z6 = e6["z_star"].float().flatten()
    z7 = e7["z_star"].float().flatten()
    
    cos_sim = float(F.cosine_similarity(z6.unsqueeze(0), z7.unsqueeze(0)).item())
    
    print(f"Q#{q:04d} | {r6:13.6e} | {r7:13.6e} | {diff_r:11.2e} | {nfe6:7d} | {nfe7:7d} | {cos_sim:16.8f}")

print()
print("Sample meta from Q#0424:")
print("S6 meta:", d6["q0424_a0.25"]["meta"])
print("S7 meta:", d7["q0424_a0.25"]["meta"])

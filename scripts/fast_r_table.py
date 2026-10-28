#!/usr/bin/env python3
"""
scripts/fast_r_table.py - Complete table of r = rho / alpha on S6 and S5 saved tensors
Directive: 원장 #375 §1
"""

import json
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
s6_meta = json.load(open(f"{BASE_S5_DIR}/configs/s6_items_74.json"))
nonconv_54 = s6_meta["nonconv_54"]
control_20 = s6_meta["control_20"]

pt_025 = torch.load(f"{BASE_S5_DIR}/results_s6/zstar_s6_a0.25.pt", map_location="cpu")
pt_035 = torch.load(f"{BASE_S5_DIR}/results_s6/zstar_s6_a0.35.pt", map_location="cpu")
pt_diag = torch.load(f"{BASE_S5_DIR}/results_s5/diag_nonconv.pt", map_location="cpu")
pt_s5 = torch.load(f"{BASE_S5_DIR}/results_s5/zstar_s5_400.pt", map_location="cpu")

def get_entry_54(pt, q, alpha):
    cands = [
        f"q{q:04d}_a{alpha:.2f}",
        f"q{q:04d}_a{alpha}",
        f"q{q:04d}_diag",
        f"q{q:04d}_s2030",
        f"q{q:04d}",
    ]
    for c in cands:
        if c in pt:
            return pt[c]
    for k in pt:
        if f"q{q:04d}" in str(k):
            return pt[k]
    return None

print("=" * 80)
print("S6 NONCONV 54: RHO vs r = RHO / ALPHA TABLE")
print("=" * 80)
print(f"{'Condition':>12} | {'Metric':>15} | {'< 1e-3':>10} | {'< 5e-3':>10} | {'< 1e-2':>10} | {'Mean Value':>12}")
print("-" * 80)

conditions = [
    ("alpha=0.50", 0.50, pt_diag),
    ("alpha=0.35", 0.35, pt_035),
    ("alpha=0.25", 0.25, pt_025),
]

for label, alpha, pt in conditions:
    rhos_1000 = []
    rs_1000 = []
    for q in nonconv_54:
        e = get_entry_54(pt, q, alpha)
        if e is None:
            print(f"Missing Q#{q} in {label}")
            continue
        r1000 = float(e.get("rho_1000", e.get("rho", e.get("final_rho_fp", 0.0))))
        rhos_1000.append(r1000)
        rs_1000.append(r1000 / alpha)
    
    n = len(rhos_1000)
    # rho
    c_rho_1e3 = sum(1 for x in rhos_1000 if x < 1e-3)
    c_rho_5e3 = sum(1 for x in rhos_1000 if x < 5e-3)
    c_rho_1e2 = sum(1 for x in rhos_1000 if x < 1e-2)
    m_rho = sum(rhos_1000) / n
    print(f"{label:>12} | {'rho@1000':>15} | {c_rho_1e3:6d}/{n} | {c_rho_5e3:6d}/{n} | {c_rho_1e2:6d}/{n} | {m_rho:12.6e}")
    
    # r = rho / alpha
    c_r_1e3 = sum(1 for x in rs_1000 if x < 1e-3)
    c_r_5e3 = sum(1 for x in rs_1000 if x < 5e-3)
    c_r_1e2 = sum(1 for x in rs_1000 if x < 1e-2)
    m_r = sum(rs_1000) / n
    print(f"{'':>12} | {'r = rho/alpha':>15} | {c_r_1e3:6d}/{n} | {c_r_5e3:6d}/{n} | {c_r_1e2:6d}/{n} | {m_r:12.6e}")
    print("-" * 80)

print("\n" * 1)
print("=" * 80)
print("S5 400 ITEMS: RHO vs r = RHO / 0.50 TABLE")
print("=" * 80)
print(f"{'Metric':>20} | {'< 1e-3':>12} | {'< 5e-3':>12} | {'< 1e-2':>12} | {'Mean Value':>12}")
print("-" * 80)
s5_rhos = [float(e.get("rho", e.get("final_rho_fp", 0.0))) for e in pt_s5.values() if isinstance(e, dict)]
s5_rs = [x / 0.50 for x in s5_rhos]
n400 = len(s5_rhos)
print(f"{'rho (logged)':>20} | {sum(1 for x in s5_rhos if x < 1e-3):8d}/{n400} | {sum(1 for x in s5_rhos if x < 5e-3):8d}/{n400} | {sum(1 for x in s5_rhos if x < 1e-2):8d}/{n400} | {sum(s5_rhos)/n400:12.6e}")
print(f"{'r = rho/0.50':>20} | {sum(1 for x in s5_rs if x < 1e-3):8d}/{n400} | {sum(1 for x in s5_rs if x < 5e-3):8d}/{n400} | {sum(1 for x in s5_rs if x < 1e-2):8d}/{n400} | {sum(s5_rs)/n400:12.6e}")
print("-" * 80)

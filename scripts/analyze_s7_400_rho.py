#!/usr/bin/env python3
"""scripts/analyze_s7_400_rho.py - Analyze rho distribution across 3 tiers for S7 400 items
Directive: 원장 #371 §2
Computes:
1. Overall 400 items rho 3-tier distribution:
   - Tier 1: rho < 1e-3
   - Tier 2: 1e-3 <= rho < 5e-3
   - Tier 3: rho >= 5e-3
2. Specifically among former 54 nonconv items:
   - Count in each tier
   - Count above threshold (rho >= 5e-3) in separate row
"""

import json
import os
import re
import sys
import torch

BASE_DIR = "/home/shoon/cts_e1_s5"
PT_PATH = os.path.join(BASE_DIR, "results_s7", "zstar_s7_a0.25_400.pt")
PARTIAL_PATH = os.path.join(BASE_DIR, "results_s7", "zstar_s7_a0.25_400.partial.pt")
ITEMS_74_PATH = os.path.join(BASE_DIR, "configs", "s6_items_74.json")


def load_target_data():
    if os.path.exists(PT_PATH):
        return torch.load(PT_PATH, map_location="cpu"), PT_PATH
    elif os.path.exists(PARTIAL_PATH):
        return torch.load(PARTIAL_PATH, map_location="cpu"), PARTIAL_PATH
    else:
        raise FileNotFoundError("Neither final nor partial 400 pt file found.")


def main():
    data, path = load_target_data()
    print(f"Loaded {len(data)} items from {path}")

    with open(ITEMS_74_PATH, "r", encoding="utf-8") as f:
        cfg_74 = json.load(f)
    nonconv_54_set = set(cfg_74["nonconv_54"])

    all_rhos = {}
    for k, v in data.items():
        m = re.match(r"^q(\d+)", str(k))
        q_idx = int(m.group(1)) if m else int(v["meta"]["question_idx"])
        rho = float(v.get("rho_1000", v.get("rho", v.get("final_rho_fp"))))
        all_rhos[q_idx] = rho

    total_n = len(all_rhos)

    # 3-tier classification
    t1_all = [q for q, r in all_rhos.items() if r < 1e-3]
    t2_all = [q for q, r in all_rhos.items() if 1e-3 <= r < 5e-3]
    t3_all = [q for q, r in all_rhos.items() if r >= 5e-3]

    print("\n" + "=" * 60)
    print(f"S7 400 ITEMS RHO 3-TIER DISTRIBUTION (Current N={total_n})")
    print("=" * 60)
    print(f"Tier 1 (rho < 1e-3):         {len(t1_all):3d} / {total_n} ({len(t1_all)/total_n*100:.2f}%)")
    print(f"Tier 2 (1e-3 <= rho < 5e-3): {len(t2_all):3d} / {total_n} ({len(t2_all)/total_n*100:.2f}%)")
    print(f"Tier 3 (rho >= 5e-3):        {len(t3_all):3d} / {total_n} ({len(t3_all)/total_n*100:.2f}%)")
    print(f"Total Converged (< 5e-3):    {len(t1_all)+len(t2_all):3d} / {total_n} ({(len(t1_all)+len(t2_all))/total_n*100:.2f}%)")

    # Nonconv 54 breakdown
    present_nonconv = [q for q in nonconv_54_set if q in all_rhos]
    n_nc = len(present_nonconv)
    if n_nc > 0:
        nc_t1 = [q for q in present_nonconv if all_rhos[q] < 1e-3]
        nc_t2 = [q for q in present_nonconv if 1e-3 <= all_rhos[q] < 5e-3]
        nc_t3 = [q for q in present_nonconv if all_rhos[q] >= 5e-3]

        print("-" * 60)
        print(f"FORMER NONCONV 54 BREAKDOWN (Present N={n_nc}/54)")
        print("-" * 60)
        print(f"Tier 1 (rho < 1e-3):         {len(nc_t1):2d} / {n_nc} ({len(nc_t1)/n_nc*100:.1f}%)")
        print(f"Tier 2 (1e-3 <= rho < 5e-3): {len(nc_t2):2d} / {n_nc} ({len(nc_t2)/n_nc*100:.1f}%)")
        print(f"Tier 3 (rho >= 5e-3):        {len(nc_t3):2d} / {n_nc} ({len(nc_t3)/n_nc*100:.1f}%) [기준선 위]")
        print(f"Total Resolved (< 5e-3):     {len(nc_t1)+len(nc_t2):2d} / {n_nc} ({(len(nc_t1)+len(nc_t2))/n_nc*100:.1f}%)")
        print(f"기준선 위 문항 목록: {sorted(nc_t3)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

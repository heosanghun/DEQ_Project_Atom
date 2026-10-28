#!/usr/bin/env python3
"""
scripts/check_nonconv_sources.py
"""
import json
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
s6_meta = json.load(open(f"{BASE_S5_DIR}/configs/s6_items_74.json"))
nonconv_54 = s6_meta["nonconv_54"]

pt_diag = torch.load(f"{BASE_S5_DIR}/results_s5/diag_nonconv.pt", map_location="cpu")
print("diag_nonconv.pt length:", len(pt_diag))
print("Sample keys in diag_nonconv.pt:", list(pt_diag.keys())[:5])

found_in_diag = sum(1 for q in nonconv_54 if any(f"q{q:04d}" in str(k) or str(q) in str(k) for k in pt_diag))
print(f"Nonconv 54 in diag_nonconv.pt: {found_in_diag}/54")

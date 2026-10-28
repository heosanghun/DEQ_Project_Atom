#!/usr/bin/env python3
"""scripts/g2v4_fresh_prep.py - PREREG_S7_G2v4 §4-2 merge + convert (CC, 2026-10-11; copy of g2v3_fresh_prep.py 59889cd3 with g2v4 paths, 1,758 items).
Merges results_s7/g2v4/fresh_zsrcinj_shard{0,1,2}.pt (g2v2_zstar_run.py --inject zsrc outputs) into
results_s7/g2v4/fresh_zsrcinj_1758.pt (keys q{idx:04d}_a0.25, all 1,758 of configs/g2v4b_fresh.json, no extras), and writes
results_s7/g2v4/fresh_zsrc_1758.pt = {key: {"z_star": entry["z_src"]}} (so extract_tensor never reads z*' as z_src).
Checks after writing: key sets equal the fresh list; converted z_star == source z_src exactly (max|d| = 0) for every key."""
import json
import torch

q = json.load(open("configs/g2v4b_fresh.json"))["indices"]
keys = [f"q{i:04d}_a0.25" for i in q]
out = {}
for k in range(3):
    out.update(torch.load(f"results_s7/g2v4/fresh_zsrcinj_shard{k}.pt", map_location="cpu"))
assert set(out.keys()) == set(keys), (len(out), len(set(out.keys()) ^ set(keys)))
z = {k: out[k] for k in keys}
src = {k: {"z_star": out[k]["z_src"].clone()} for k in keys}
torch.save(z, "results_s7/g2v4/fresh_zsrcinj_1758.pt")
torch.save(src, "results_s7/g2v4/fresh_zsrc_1758.pt")
z2 = torch.load("results_s7/g2v4/fresh_zsrcinj_1758.pt", map_location="cpu")
s2 = torch.load("results_s7/g2v4/fresh_zsrc_1758.pt", map_location="cpu")
assert set(z2) == set(keys) and set(s2) == set(keys)
dmax = max(float((s2[k]["z_star"] - z2[k]["z_src"]).abs().max()) for k in keys)
r1 = [z2[k]["r_1000"] for k in keys]
print(f"[PREP] keys {len(keys)} | converted z_src max|d| {dmax:.3e} | r_1000<1e-2 {sum(x < 1e-2 for x in r1)}/{len(keys)}")
assert dmax == 0.0

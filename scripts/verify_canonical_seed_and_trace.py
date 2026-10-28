#!/usr/bin/env python3
"""
scripts/verify_canonical_seed_and_trace.py - Run canonical compute_zstar on CPU and compare with S7 raw log
Directive: 원장 #374 §3
"""

import os
import sys
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
from cts.deq.zstar_canonical import compute_zstar, load_canonical_config
from cts.routing.sparse_moe_ref import routing_weights

bb = get_model("cpu")
w_g = bb.routing_matrix().to(device="cpu", dtype=torch.float32)
dataset = get_dataset()
cfg_path = os.path.join(BASE_S5_DIR, "configs", "zstar_canonical.yaml")

# Raw S7 log values to verify against
# Q#2140: Item 012
# Q#4067: Item 043
# Q#1790: Item 065
s7_log_targets = {
    2140: {
        10: 5.20967804e-02,
        20: 2.92584855e-02,
        30: 1.89353799e-02,
        40: 1.33378039e-02,
        50: 1.20431563e-02,
        60: 1.08105634e-02,
        70: 8.29301987e-03,
        80: 6.56551681e-03,
        90: 5.45400038e-03,
        100: 1.54491363e-02,
    },
    4067: {
        10: 5.78589655e-02,
        20: 4.18200002e-02,
        30: 2.62795448e-02,
        40: 1.69029777e-02,
        50: 1.25379538e-02,
        60: 1.31546846e-02,
        70: 9.64682636e-03,
        80: 7.45816890e-03,
        90: 6.05495485e-03,
        100: 5.10159448e-03,
    },
    1790: {
        10: 5.78038379e-02,
        20: 3.35271033e-02,
        30: 2.08962393e-02,
        40: 1.37494126e-02,
        50: 1.44786119e-02,
        60: 1.08313306e-02,
        70: 8.50424753e-03,
        80: 1.41250445e-02,
        90: 9.66393990e-03,
        100: 7.23688032e-03,
    }
}

print("=" * 80)
print("CANONICAL compute_zstar REPRODUCTION TEST (seed=2030, config=zstar_canonical.yaml)")
print("=" * 80)

for q in [2140, 4067, 1790]:
    p_text = dataset[q]["prompt"]
    print(f"\nRunning Q#{q:04d} on CPU with compute_zstar (max_nfe=120, alpha=0.25)...", flush=True)
    with torch.no_grad():
        z_star, rho_fp, nfe, log = compute_zstar(
            x=p_text,
            cfg=cfg_path,
            backbone=bb,
            w_g=w_g,
            device="cpu",
            tol=0.0,
            max_nfe=120,
            alpha=0.25,
        )
    
    rho_traj = log["rho_traj"]
    print(f"Q#{q:04d} completed 120 steps. Comparing rho values with raw_s7 log:")
    print(f"{'Step':>6} | {'S7 Raw Log':>14} | {'CPU compute_zstar':>18} | {'Diff':>10} | {'Rel Diff':>10}")
    print("-" * 65)
    for s in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        log_val = s7_log_targets[q][s]
        cpu_val = rho_traj[s - 1]
        diff = abs(log_val - cpu_val)
        rel_diff = diff / max(log_val, 1e-12)
        print(f"{s:6d} | {log_val:14.8e} | {cpu_val:18.8e} | {diff:10.2e} | {rel_diff:9.4%}")

print("\nDone.")

#!/usr/bin/env python3
"""scripts/check_alpha_effect.py - Verify Damping Factor (alpha) Empirical Scaling
Directive: EXE-260926-36 / 통지 원장 #349 Check 1

Verification:
1 control item, same z0, same seed (canonical DEQ path)
Run 1 Picard step with alpha=0.50 and alpha=0.35
Output: ||z1 - z0||(alpha=0.35) / ||z1 - z0||(alpha=0.50)
Expected: 0.7000 (+/- 1e-4)
"""

import os
import sys
import time
import argparse
from datetime import datetime
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from harness_A_v1 import get_model, get_dataset
from cts.deq.zstar_canonical import compute_zstar, load_canonical_config

DEFAULT_CONFIG_PATH = os.path.join(BASE_S5_DIR, "configs", "zstar_canonical.yaml")

def main():
    parser = argparse.ArgumentParser(description="Verify Alpha Effect on First Picard Step (원장 #349)")
    parser.add_argument("--q_idx", type=int, default=249, help="Question index for verification (default: 249 from control 20)")
    parser.add_argument("--id", type=str, default=None, help="Execution ID")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config yaml")
    args = parser.parse_args()

    exec_id = args.id or datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cpu")
    q_idx = args.q_idx

    print("=" * 80)
    print(f"EXE-260926-36 / 원장 #349 Check 1: Alpha Effect Empirical Verification")
    print(f"Execution ID: {exec_id} | Device: cpu | Q-ID: {q_idx}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80, flush=True)

    t0_load = time.time()
    bb = get_model("cpu")
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
    dataset = get_dataset()
    p_text = dataset[q_idx]["prompt"]
    print(f"Loaded model and dataset in {time.time() - t0_load:.2f}s.", flush=True)

    # 1. Step with alpha = 0.50
    print("\n--- Running Picard Step 1 with alpha=0.50 ---", flush=True)
    t0_05 = time.time()
    z_star_05, rho_05, nfe_05, log_05 = compute_zstar(
        x=p_text,
        cfg=args.config,
        backbone=bb,
        w_g=w_g,
        device=device,
        tol=0.0,
        max_nfe=1,
        sigma_noise=0.020,
        alpha=0.50,
    )
    t_05 = time.time() - t0_05
    z0_05 = log_05["z0"]
    z1_05 = z_star_05
    diff_05 = float(torch.norm(z1_05.to(torch.float64) - z0_05.to(torch.float64)).item())
    norm_z0_05 = float(torch.norm(z0_05.to(torch.float64)).item())
    norm_z1_05 = float(torch.norm(z1_05.to(torch.float64)).item())

    # 2. Step with alpha = 0.35
    print("--- Running Picard Step 1 with alpha=0.35 ---", flush=True)
    t0_035 = time.time()
    z_star_035, rho_035, nfe_035, log_035 = compute_zstar(
        x=p_text,
        cfg=args.config,
        backbone=bb,
        w_g=w_g,
        device=device,
        tol=0.0,
        max_nfe=1,
        sigma_noise=0.020,
        alpha=0.35,
    )
    t_035 = time.time() - t0_035
    z0_035 = log_035["z0"]
    z1_035 = z_star_035
    diff_035 = float(torch.norm(z1_035.to(torch.float64) - z0_035.to(torch.float64)).item())
    norm_z0_035 = float(torch.norm(z0_035.to(torch.float64)).item())
    norm_z1_035 = float(torch.norm(z1_035.to(torch.float64)).item())

    # Bit-level parity check for initial z0
    delta_z0 = float(torch.max(torch.abs(z0_05.to(torch.float64) - z0_035.to(torch.float64))).item())

    # Compute ratio
    ratio = diff_035 / diff_05 if diff_05 > 0 else 0.0
    expected_ratio = 0.35 / 0.50
    abs_err = abs(ratio - expected_ratio)
    pass_check = bool(abs_err <= 1e-4)

    print("=" * 80)
    print("ALPHA EFFECT EMPIRICAL VERIFICATION REPORT (원장 #349)")
    print(f"Execution ID: {exec_id}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: cpu | Dtype: float32 | Target Item: Q#{q_idx}")
    print("-" * 80)
    print(f"z0 max|delta| (alpha=0.50 vs alpha=0.35): {delta_z0:.10e}")
    print(f"alpha=0.50: ||z0||_F = {norm_z0_05:.8f} | ||z1||_F = {norm_z1_05:.8f} | ||z1 - z0||_F = {diff_05:.8f} | elapsed = {t_05:.2f}s")
    print(f"alpha=0.35: ||z0||_F = {norm_z0_035:.8f} | ||z1||_F = {norm_z1_035:.8f} | ||z1 - z0||_F = {diff_035:.8f} | elapsed = {t_035:.2f}s")
    print("-" * 80)
    print(f"Observed ratio ||z1 - z0||(a=0.35) / ||z1 - z0||(a=0.50): {ratio:.8f}")
    print(f"Expected ratio (0.35 / 0.50): {expected_ratio:.8f}")
    print(f"Absolute error |ratio - 0.7000|: {abs_err:.8e}")
    print(f"Check Status: {'PASS' if pass_check else 'FAIL'} (criterion: |ratio - 0.7000| <= 1e-4)")
    print("=" * 80, flush=True)

    if not pass_check:
        print("\n[CRITICAL FAILURE] Alpha parameter does NOT affect Picard step! Aborting chain.", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

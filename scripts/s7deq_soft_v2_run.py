#!/usr/bin/env python3
"""
scripts/s7deq_soft_v2_run.py - Runner for Smooth Top-3 Switch Fixed-Point Computation
Directive: CTS-428 Task 3 / PREREG_S7_DEQ_v2 §2·§5·§6

Saves dict to .pt with keys q{idx:04d}_a{alpha:.2f}:
- z_star (= z_1000)
- r_1000
- z_lowest (= argmin_{k in [0, 1000]} r_k, tie-break: min k)
- k_star
- r_lowest
- rho_traj (1000 entries)
- top3_final
- top3_lowest
- eps
- d_traj (1000 entries of d = a_i3 - a_i4)
- w_traj (1000 entries of weight w)
- boundary_step_count
- boundary_step_ratio
- last_100_boundary_ratio
- r_hard_lowest (hard canonical transition residual evaluated at z_lowest)
- d_lowest
- in_boundary_lowest
- r_hard_final (hard canonical transition residual evaluated at z_star)
- d_final
- in_boundary_final
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"
EXE_SCRATCH_DIR = os.path.join(BASE_S5_DIR, "exe_scratch", "s7", "soft_v2")

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, EXE_SCRATCH_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cts
import cts.deq

s5_deq = os.path.join(BASE_S5_DIR, "cts", "deq")
if s5_deq not in cts.deq.__path__:
    cts.deq.__path__.insert(0, s5_deq)

from harness_A_v1 import get_model, get_dataset
from exe_scratch.s7.soft_v2.zstar_soft_lowest import compute_zstar_soft_lowest

DEFAULT_CONFIG_PATH = os.path.join(BASE_S5_DIR, "configs", "zstar_canonical.yaml")
if not os.path.exists(DEFAULT_CONFIG_PATH):
    DEFAULT_CONFIG_PATH = os.path.join(SYS_DIR, "configs", "zstar_canonical.yaml")


def load_items_from_source(items_source: str) -> List[int]:
    """Parse item indices from JSON path or comma-separated list."""
    if os.path.exists(items_source):
        with open(items_source, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [int(x) for x in data]
        elif isinstance(data, dict):
            if "indices" in data:
                return [int(x) for x in data["indices"]]
            elif "items" in data:
                return [int(x) for x in data["items"]]
            first_key = next(iter(data.keys()))
            if isinstance(data[first_key], list):
                return [int(x) for x in data[first_key]]
    else:
        return [int(x.strip()) for x in items_source.split(",") if x.strip()]
    raise ValueError(f"Unable to parse item list from {items_source}")


def main():
    parser = argparse.ArgumentParser(description="Run S7 DEQ Smooth Top-3 Switch Computation")
    parser.add_argument("--items", type=str, required=True, help="Path to items JSON or comma-separated list")
    parser.add_argument("--out", type=str, required=True, help="Path to output .pt file")
    parser.add_argument("--eps", type=float, default=0.01, help="Soft boundary half-width eps (default: 0.01)")
    parser.add_argument("--alpha", type=float, default=0.25, help="Damping factor alpha (default: 0.25)")
    parser.add_argument("--g0", type=float, default=0.25, help="Context injection scale g0 (default: 0.25)")
    parser.add_argument("--seed", type=int, default=2030, help="Initial noise seed (default: 2030)")
    parser.add_argument("--sigma", type=float, default=0.020, help="Noise std sigma (default: 0.020)")
    parser.add_argument("--max_nfe", type=int, default=1000, help="Max iterations (default: 1000)")
    parser.add_argument("--tol", type=float, default=0.0, help="Convergence tolerance (default: 0.0)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to canonical YAML")
    args = parser.parse_args()

    items = load_items_from_source(args.items)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 80)
    print(f"S7 DEQ Smooth Top-3 Switch Run: eps={args.eps}, alpha={args.alpha}, max_nfe={args.max_nfe}, items={len(items)}")
    print(f"Device: {args.device} | Seed: {args.seed} | Sigma: {args.sigma} | Tol: {args.tol}")
    print(f"Output: {args.out}")
    print("=" * 80, flush=True)

    device = torch.device(args.device)
    bb = get_model(str(device))
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
    dataset = get_dataset()

    out_dict: Dict[str, Any] = {}
    if os.path.exists(args.out):
        out_dict = torch.load(args.out, map_location="cpu")
        print(f"Loaded existing output dictionary with {len(out_dict)} items.", flush=True)

    for idx, q_idx in enumerate(items, 1):
        key = f"q{q_idx:04d}_a{args.alpha:.2f}"
        if key in out_dict:
            print(f"[{idx:04d}/{len(items):04d}] Skipping {key} (already present)", flush=True)
            continue

        p_text = dataset[q_idx]["prompt"]
        t0 = time.time()

        z_star, final_rho, nfe, log = compute_zstar_soft_lowest(
            x=p_text,
            cfg=args.config,
            backbone=bb,
            w_g=w_g,
            device=device,
            tol=args.tol,
            max_nfe=args.max_nfe,
            sigma_noise=args.sigma,
            alpha=args.alpha,
            g0=args.g0,
            seed=args.seed,
            eps=args.eps,
        )
        elapsed = time.time() - t0

        out_dict[key] = {
            "z_star": z_star.detach().cpu(),
            "r_1000": float(log["r_1000"]),
            "z_lowest": log["z_lowest"].detach().cpu(),
            "k_star": int(log["k_star"]),
            "r_lowest": float(log["r_lowest"]),
            "rho_traj": [float(x) for x in log["rho_traj"]],
            "top3_final": [int(x) for x in log["top3_final"]],
            "top3_lowest": [int(x) for x in log["top3_lowest"]],
            "eps": float(args.eps),
            "d_traj": [float(x) for x in log["d_traj"]],
            "w_traj": [float(x) for x in log["w_traj"]],
            "boundary_step_count": int(log["boundary_step_count"]),
            "boundary_step_ratio": float(log["boundary_step_ratio"]),
            "last_100_boundary_ratio": float(log["last_100_boundary_ratio"]),
            "r_hard_lowest": float(log["r_hard_lowest"]),
            "d_lowest": float(log["d_lowest"]),
            "in_boundary_lowest": bool(log["in_boundary_lowest"]),
            "r_hard_final": float(log["r_hard_final"]),
            "d_final": float(log["d_final"]),
            "in_boundary_final": bool(log["in_boundary_final"]),
        }

        print(
            f"[{idx:04d}/{len(items):04d}] {key}: r_1000={log['r_1000']:.6e}, "
            f"r_lowest={log['r_lowest']:.6e} at k*={log['k_star']}, "
            f"r_hard_lowest={log['r_hard_lowest']:.6e}, b_steps={log['boundary_step_count']}/1000 ({log['boundary_step_ratio']*100:.1f}%), "
            f"in_b={log['in_boundary_lowest']} ({elapsed:.2f}s)",
            flush=True,
        )

        # Periodic checkpoint every 5 items
        if idx % 5 == 0:
            torch.save(out_dict, args.out)

    torch.save(out_dict, args.out)
    print("=" * 80)
    print(f"Finished. Total items saved: {len(out_dict)} to {args.out}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

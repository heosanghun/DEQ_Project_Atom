#!/usr/bin/env python3
"""
scripts/g2v2_zstar_run.py - G2 개선판(M2′) 고정점 실행기 (CC, 2026-10-08)
정본 s7_zstar_lowest_run.py 복사. 바꾼 것: 계산 함수 compute_zstar_lowest_v2(cts/deq/zstar_inject_v2.py),
인자 --inject {mean,zsrc}, --save_steps, 출력에 z_src·z_k·inject 필드 추가. --inject mean이면 정본과 같은 계산.

Saves dict to .pt with keys q{idx:04d}_a{alpha:.2f}:
- z_star (= z_1000)
- r_1000
- z_lowest (= argmin_{k in [0, 1000]} r_k, tie-break: min k)
- k_star
- r_lowest
- rho_traj (1000 entries)
- top3_final
- top3_lowest
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

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cts
import cts.deq

s5_deq = os.path.join(BASE_S5_DIR, "cts", "deq")
if s5_deq not in cts.deq.__path__:
    cts.deq.__path__.insert(0, s5_deq)

from harness_A_v1 import get_model, get_dataset
from cts.deq.zstar_inject_v2 import compute_zstar_lowest_v2

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
    parser = argparse.ArgumentParser(description="Run S7 Lowest-Residual DEQ computation")
    parser.add_argument("--items", type=str, required=True, help="Path to items JSON or comma-separated list")
    parser.add_argument("--out", type=str, required=True, help="Path to output .pt file")
    parser.add_argument("--alpha", type=float, default=0.25, help="Damping factor alpha (default: 0.25)")
    parser.add_argument("--g0", type=float, default=0.25, help="Context injection scale g0 (default: 0.25)")
    parser.add_argument("--seed", type=int, default=2030, help="Initial noise seed (default: 2030)")
    parser.add_argument("--sigma", type=float, default=0.020, help="Noise std sigma (default: 0.020)")
    parser.add_argument("--max_nfe", type=int, default=1000, help="Max iterations (default: 1000)")
    parser.add_argument("--tol", type=float, default=0.0, help="Convergence tolerance (default: 0.0)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (default: cuda:0)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to canonical YAML")
    parser.add_argument("--inject", type=str, choices=["mean", "zsrc"], required=True, help="Injection vector: mean (canonical) or zsrc")
    parser.add_argument("--save_steps", type=str, default="", help="Comma list of k to save z_k (k=0 is z0)")
    args = parser.parse_args()

    items = load_items_from_source(args.items)
    save_steps = tuple(int(x) for x in args.save_steps.split(",") if x.strip())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 80)
    print(f"S7 Lowest-Residual DEQ Run: alpha={args.alpha}, max_nfe={args.max_nfe}, items={len(items)}")
    print(f"Device: {args.device} | Seed: {args.seed} | Sigma: {args.sigma} | Tol: {args.tol}")
    print(f"Output: {args.out} | inject={args.inject} | save_steps={save_steps}")
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

        z_star, final_rho, nfe, log = compute_zstar_lowest_v2(
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
            inject_mode=args.inject,
            save_steps=save_steps,
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
            "z_src": log["z_src"].detach().cpu(),
            "inject": args.inject,
            "z_k": {int(k): v for k, v in log["z_k"].items()},
        }

        print(
            f"[{idx:04d}/{len(items):04d}] {key}: r_1000={log['r_1000']:.6e}, "
            f"r_lowest={log['r_lowest']:.6e} at k*={log['k_star']}, "
            f"top3_final={log['top3_final']}, top3_lowest={log['top3_lowest']} ({elapsed:.2f}s)",
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

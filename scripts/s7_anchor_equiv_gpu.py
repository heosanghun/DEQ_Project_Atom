#!/usr/bin/env python3
"""
scripts/s7_anchor_equiv_gpu.py - Anchor 20 Device Determinism / Equivalence Checker
Directive: PREREG_S7_DEQ_v1.3 §10(f) / Principal Directive 2026-10-05

Evaluates 20 anchor questions (data/anchor20.json) on specified GPU (cuda:1 or cuda:2)
using seed 2030 and canonical parameters (alpha=0.25, g0=0.25, sigma=0.020, max_nfe=1000, tol=0.0).
Computes canonical fixed-point z_canon via compute_zstar (cts.deq.zstar_canonical) and
lowest-residual z_new via compute_zstar_lowest (cts.deq.zstar_canonical_lowest).
Reports:
  1) Bitwise match count (max|Delta| == 0.0)
  2) cos >= 0.9999 only match count
  3) Sub-threshold count (cos < 0.9999)
Requirement: All 20 items must satisfy max|Delta| == 0.0 or cos >= 0.9999.
Saves summary JSON and output tensor .pt for cross-GPU validation.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List
import torch
import torch.nn.functional as F

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(BASE_S5_DIR, "scratch")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cts
import cts.deq

s5_deq = os.path.join(BASE_S5_DIR, "cts", "deq")
if s5_deq not in cts.deq.__path__:
    cts.deq.__path__.insert(0, s5_deq)

from harness_A_v1 import get_model, get_dataset
from cts.deq.zstar_canonical import compute_zstar
from cts.deq.zstar_canonical_lowest import compute_zstar_lowest

DEFAULT_ITEMS = os.path.join(BASE_S5_DIR, "data", "anchor20.json")
DEFAULT_CONFIG = os.path.join(BASE_S5_DIR, "configs", "zstar_canonical.yaml")
if not os.path.exists(DEFAULT_CONFIG):
    DEFAULT_CONFIG = os.path.join(SYS_DIR, "configs", "zstar_canonical.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Anchor 20 Equivalence Checker on GPU")
    parser.add_argument("--device", type=str, required=True, help="Target device (must be cuda:1 or cuda:2)")
    parser.add_argument("--items", type=str, default=DEFAULT_ITEMS, help="Path to anchor 20 JSON")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to canonical YAML")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON results path")
    parser.add_argument("--out_pt", type=str, default=None, help="Output tensor .pt path")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--g0", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--sigma", type=float, default=0.020)
    parser.add_argument("--max_nfe", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=0.0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device_str = args.device.strip().lower()

    # Strict Device Enforcement (Rule #203 and Principal directive)
    if device_str in ["cuda:0", "0"]:
        print("[CRITICAL ERROR] GPU 0 is reserved for P4' decoder training! Access rejected.", file=sys.stderr)
        sys.exit(1)
    if "4" in device_str:
        print("[CRITICAL ERROR] GPU 4 is strictly sealed (Rule #203)! Access rejected.", file=sys.stderr)
        sys.exit(1)
    if device_str not in ["cuda:1", "cuda:2"]:
        print(f"[CRITICAL ERROR] Device must be cuda:1 or cuda:2 (got: {device_str})! Access rejected.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(device_str)
    print("=" * 80)
    print(f"ANCHOR 20 EQUIVALENCE EVALUATION ON {device_str.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters: alpha={args.alpha}, g0={args.g0}, seed={args.seed}, sigma={args.sigma}, max_nfe={args.max_nfe}, tol={args.tol}")
    print(f"Items: {args.items}")
    print("Criteria per PREREG_S7_DEQ_v1.3 §10(f): max|Delta| == 0.0 or cos >= 0.9999 for all 20 items")
    print("=" * 80, flush=True)

    # Load anchor question indices
    with open(args.items, "r", encoding="utf-8") as f:
        anchor_items = json.load(f)
    print(f"Loaded {len(anchor_items)} anchor item indices: {anchor_items}\n", flush=True)

    # Initialize model on target device
    print(f"Initializing backbone model on {device} ...", flush=True)
    bb = get_model(str(device))
    bb.eval()
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
    dataset = get_dataset()
    print("Model initialized successfully.\n", flush=True)

    t0_start = time.time()
    results = []
    tensors_out = {}
    bitwise_count = 0
    cos_09999_count = 0
    sub_threshold_count = 0

    for idx, q_idx in enumerate(anchor_items, 1):
        t0_q = time.time()
        prompt = dataset[q_idx]["prompt"]

        # 1. Compute canonical z_star via compute_zstar (canonical Picard solver)
        z_star_canon, rho_canon, nfe_canon, log_canon = compute_zstar(
            x=prompt,
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
        )

        # 2. Compute lowest-residual z_star via compute_zstar_lowest (lowest-residual tracker)
        z_star_new, rho_new, nfe_new, log_new = compute_zstar_lowest(
            x=prompt,
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
        )

        # Compute differences in float64 for absolute precision
        z_new_f64 = z_star_new.cpu().double()
        z_canon_f64 = z_star_canon.cpu().double()

        abs_diff = torch.abs(z_new_f64 - z_canon_f64)
        max_delta = float(torch.max(abs_diff).item())

        v_new = z_new_f64.view(1, -1)
        v_canon = z_canon_f64.view(1, -1)
        cos_sim = float(F.cosine_similarity(v_new, v_canon).item())
        elapsed_q = time.time() - t0_q

        # Classification
        if max_delta == 0.0:
            status = "BITWISE_EXACT"
            bitwise_count += 1
        elif cos_sim >= 0.9999:
            status = "COS_GE_0.9999"
            cos_09999_count += 1
        else:
            status = "SUB_THRESHOLD_FAIL"
            sub_threshold_count += 1

        print(
            f"[{idx:02d}/20] Q#{q_idx:02d}: max|Δ|={max_delta:.6e}, cos={cos_sim:.8f} "
            f"[{status}] (r_1000={log_new['r_1000']:.4e}, k*={log_new['k_star']}, {elapsed_q:.1f}s)",
            flush=True,
        )

        results.append({
            "idx": idx,
            "q_idx": q_idx,
            "max_delta": max_delta,
            "cos_sim": cos_sim,
            "status": status,
            "r_1000": log_new["r_1000"],
            "k_star": log_new["k_star"],
            "r_lowest": log_new["r_lowest"],
            "top3_final": log_new.get("top3_final", []),
            "top3_lowest": log_new.get("top3_lowest", []),
            "elapsed_seconds": elapsed_q,
        })

        tensors_out[f"q{q_idx:02d}_s{args.seed}"] = {
            "z_star": z_star_new.cpu().half(),
            "z_canon": z_star_canon.cpu().half(),
            "r_1000": float(log_new["r_1000"]),
            "k_star": int(log_new["k_star"]),
            "r_lowest": float(log_new["r_lowest"]),
        }

    total_time = time.time() - t0_start
    total_qualified = bitwise_count + cos_09999_count
    all_passed = (total_qualified == len(anchor_items))

    summary = {
        "device": device_str,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_items": len(anchor_items),
        "bitwise_match_count": bitwise_count,
        "cos_ge_09999_only_count": cos_09999_count,
        "sub_threshold_count": sub_threshold_count,
        "total_qualified": total_qualified,
        "criteria_met_20_of_20": all_passed,
        "total_elapsed_seconds": total_time,
        "items_detail": results,
    }

    print("\n" + "=" * 80)
    print(f"ANCHOR 20 DETERMINISM SUMMARY ({device_str.upper()})")
    print(f"Total Evaluated:              {len(anchor_items)}")
    print(f"Bitwise Match (max|Δ| == 0):  {bitwise_count} / {len(anchor_items)}")
    print(f"cos >= 0.9999 Only Match:     {cos_09999_count} / {len(anchor_items)}")
    print(f"Sub-threshold (cos < 0.9999): {sub_threshold_count} / {len(anchor_items)}")
    print(f"Total Qualified:              {total_qualified} / {len(anchor_items)}")
    print(f"Criteria Met (20/20):         {all_passed}")
    print(f"Total Elapsed Time:           {total_time:.1f}s ({total_time / 60:.2f} min)")
    print("=" * 80, flush=True)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to: {args.out_json}", flush=True)

    if args.out_pt:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_pt)), exist_ok=True)
        torch.save(tensors_out, args.out_pt)
        print(f"Tensors saved to: {args.out_pt}", flush=True)

    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

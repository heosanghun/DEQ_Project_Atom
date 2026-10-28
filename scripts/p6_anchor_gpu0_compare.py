#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p6_anchor_gpu0_compare.py - Anchor 20 GPU 0 vs GPU 1 & GPU 2 CPU Equivalence Checker
Directive: TASK-0018 Section 3 / AUDIT-0017 FIX F1 / PREREG_S7_DEQ_v1.3 §10(f)

Compares anchor20_gpu0.pt against anchor20_gpu1.pt and anchor20_gpu2.pt on CPU in float64.
Computes item-level max|Δ| and cosine similarity.
Saves results to results_s7/anchor20_gpu0_vs_12.json and writes raw_s7/ANCHOR_GPU0_DONE.flag.
Per directive, chain continues regardless of sub-threshold count; sub-threshold count is recorded in the flag.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F

ANCHOR_ITEMS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]


def extract_tensor(data_dict: Dict[str, Any], qid: int) -> torch.Tensor:
    """Extract z_star tensor for a given question id from output dict."""
    candidates = [
        f"q{qid:04d}_a0.25",
        f"q{qid:02d}_s2030",
        f"q{qid:04d}",
        f"q{qid:02d}",
        f"q{qid}",
        str(qid),
        int(qid),
    ]
    for c in candidates:
        if c in data_dict:
            entry = data_dict[c]
            if isinstance(entry, torch.Tensor):
                return entry
            if isinstance(entry, dict):
                if "z_star" in entry and isinstance(entry["z_star"], torch.Tensor):
                    return entry["z_star"]
                if "z_lowest" in entry and isinstance(entry["z_lowest"], torch.Tensor):
                    return entry["z_lowest"]
                if "z_canon" in entry and isinstance(entry["z_canon"], torch.Tensor):
                    return entry["z_canon"]
    raise KeyError(f"Question ID {qid} not found in tensor dict (tried: {candidates[:3]})")


def main():
    parser = argparse.ArgumentParser(description="Anchor 20 GPU 0 vs GPU 1 & 2 Equivalence Checker")
    parser.add_argument("--gpu0", type=str, required=True, help="Path to anchor20_gpu0.pt")
    parser.add_argument("--gpu1", type=str, required=True, help="Path to anchor20_gpu1.pt")
    parser.add_argument("--gpu2", type=str, required=True, help="Path to anchor20_gpu2.pt")
    parser.add_argument("--out_json", type=str, default="results_s7/anchor20_gpu0_vs_12.json", help="Path to output JSON")
    parser.add_argument("--flag", type=str, default="raw_s7/ANCHOR_GPU0_DONE.flag", help="Path to completion flag")
    args = parser.parse_args()

    print("=" * 80)
    print("CTS S7 Anchor 20 GPU 0 vs GPU 1 & GPU 2 CPU Equivalence Comparison")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU 0: {args.gpu0}")
    print(f"GPU 1: {args.gpu1}")
    print(f"GPU 2: {args.gpu2}")
    print("=" * 80, flush=True)

    if not os.path.exists(args.gpu0):
        raise FileNotFoundError(f"GPU 0 anchor file not found: {args.gpu0}")
    if not os.path.exists(args.gpu1):
        raise FileNotFoundError(f"GPU 1 anchor file not found: {args.gpu1}")
    if not os.path.exists(args.gpu2):
        raise FileNotFoundError(f"GPU 2 anchor file not found: {args.gpu2}")

    t0_dict = torch.load(args.gpu0, map_location="cpu")
    t1_dict = torch.load(args.gpu1, map_location="cpu")
    t2_dict = torch.load(args.gpu2, map_location="cpu")

    items_detail = []
    bitwise_01 = 0
    cos_09999_01 = 0
    sub_01 = 0

    bitwise_02 = 0
    cos_09999_02 = 0
    sub_02 = 0

    bitwise_12 = 0
    cos_09999_12 = 0
    sub_12 = 0

    print(f"{'Q#':<5} | {'GPU0 vs GPU1 max|Δ|':<20} | {'cos_01':<12} | {'GPU0 vs GPU2 max|Δ|':<20} | {'cos_02':<12} | {'Status 0vs1/2'}")
    print("-" * 90)

    for qid in ANCHOR_ITEMS:
        z0 = extract_tensor(t0_dict, qid).double()
        z1 = extract_tensor(t1_dict, qid).double()
        z2 = extract_tensor(t2_dict, qid).double()

        # GPU 0 vs GPU 1
        d01 = float((z0 - z1).abs().max().item())
        c01 = float(F.cosine_similarity(z0.view(1, -1), z1.view(1, -1)).item())
        if d01 == 0.0:
            s01 = "BITWISE_EXACT"
            bitwise_01 += 1
        elif c01 >= 0.9999:
            s01 = "COS_GE_0.9999"
            cos_09999_01 += 1
        else:
            s01 = "SUB_THRESHOLD"
            sub_01 += 1

        # GPU 0 vs GPU 2
        d02 = float((z0 - z2).abs().max().item())
        c02 = float(F.cosine_similarity(z0.view(1, -1), z2.view(1, -1)).item())
        if d02 == 0.0:
            s02 = "BITWISE_EXACT"
            bitwise_02 += 1
        elif c02 >= 0.9999:
            s02 = "COS_GE_0.9999"
            cos_09999_02 += 1
        else:
            s02 = "SUB_THRESHOLD"
            sub_02 += 1

        # GPU 1 vs GPU 2 (control)
        d12 = float((z1 - z2).abs().max().item())
        c12 = float(F.cosine_similarity(z1.view(1, -1), z2.view(1, -1)).item())
        if d12 == 0.0:
            bitwise_12 += 1
        elif c12 >= 0.9999:
            cos_09999_12 += 1
        else:
            sub_12 += 1

        print(f"Q#{qid:02d}  | {d01:<20.6e} | {c01:<12.8f} | {d02:<20.6e} | {c02:<12.8f} | {s01}/{s02}")

        items_detail.append({
            "qid": qid,
            "gpu0_vs_gpu1": {"max_delta": d01, "cos_sim": c01, "status": s01},
            "gpu0_vs_gpu2": {"max_delta": d02, "cos_sim": c02, "status": s02},
            "gpu1_vs_gpu2": {"max_delta": d12, "cos_sim": c12},
        })

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_items": len(ANCHOR_ITEMS),
        "gpu0_vs_gpu1": {
            "bitwise_match_count": bitwise_01,
            "cos_ge_09999_only_count": cos_09999_01,
            "sub_threshold_count": sub_01,
            "total_qualified": bitwise_01 + cos_09999_01,
        },
        "gpu0_vs_gpu2": {
            "bitwise_match_count": bitwise_02,
            "cos_ge_09999_only_count": cos_09999_02,
            "sub_threshold_count": sub_02,
            "total_qualified": bitwise_02 + cos_09999_02,
        },
        "gpu1_vs_gpu2_control": {
            "bitwise_match_count": bitwise_12,
            "cos_ge_09999_only_count": cos_09999_12,
            "sub_threshold_count": sub_12,
        },
        "items_detail": items_detail,
    }

    print("\n" + "=" * 80)
    print("ANCHOR 20 CROSS-GPU SUMMARY")
    print(f"GPU 0 vs GPU 1: Bitwise={bitwise_01}/20, cos>=0.9999={cos_09999_01}/20, Sub-threshold={sub_01}/20")
    print(f"GPU 0 vs GPU 2: Bitwise={bitwise_02}/20, cos>=0.9999={cos_09999_02}/20, Sub-threshold={sub_02}/20")
    print(f"GPU 1 vs GPU 2: Bitwise={bitwise_12}/20, cos>=0.9999={cos_09999_12}/20, Sub-threshold={sub_12}/20")
    print("=" * 80, flush=True)

    # Save JSON output
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved comparison JSON to: {args.out_json}")

    # Write completion flag with sub-threshold counts
    if args.flag:
        os.makedirs(os.path.dirname(os.path.abspath(args.flag)), exist_ok=True)
        with open(args.flag, "w", encoding="utf-8") as f:
            f.write("ANCHOR_GPU0_DONE\n")
            f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"total_items: {len(ANCHOR_ITEMS)}\n")
            f.write(f"gpu0_vs_gpu1_sub_threshold_count: {sub_01}\n")
            f.write(f"gpu0_vs_gpu2_sub_threshold_count: {sub_02}\n")
            f.write(f"gpu0_vs_gpu1_bitwise_count: {bitwise_01}\n")
            f.write(f"gpu0_vs_gpu2_bitwise_count: {bitwise_02}\n")
            f.write(f"gpu0_vs_gpu1_cos_ge_09999_count: {cos_09999_01}\n")
            f.write(f"gpu0_vs_gpu2_cos_ge_09999_count: {cos_09999_02}\n")
        print(f"Created completion flag: {args.flag}")

    print("[ANCHOR-GPU0-COMPARE] Execution completed. Returning exit code 0.")
    sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/s7deq_soft_v2_analyze.py - Metric Analyzer for Smooth Top-3 Switch (PREREG_S7_DEQ_v2 §4·§7)
Directive: CTS-428 Task 3 / PREREG_S7_DEQ_v2

Computes Metrics (1) through (5) for S7 DEQ smooth top-3 switch results,
formats comparison table alongside P3 baseline, and records input file SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ITEMS_54_NONCONV = [
    31, 127, 138, 279, 424, 441, 481, 619, 684, 754, 865, 946, 972, 1177, 1351,
    1567, 1623, 1676, 1790, 1844, 1863, 1958, 2076, 2159, 2164, 2215, 2226, 2240,
    2249, 2288, 2392, 2535, 2536, 2573, 2669, 2692, 2722, 2844, 2874, 2877, 3088,
    3098, 3893, 3913, 3998, 4005, 4068, 4171, 4189, 4298, 4568, 4583, 4585, 4609,
]

ITEMS_20_CONTROL = [
    249, 395, 622, 822, 942, 954, 1430, 2060, 2884, 2933, 3184, 3634, 3656, 3802,
    3953, 3976, 4208, 4404, 4722, 4773,
]


def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_items_config(path: str) -> Tuple[List[int], List[int], List[int]]:
    """Load items config, returning (items_74, items_54, items_20)."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "nonconv_54" in data and "control_20" in data:
                return data.get("items", data["nonconv_54"] + data["control_20"]), data["nonconv_54"], data["control_20"]
            elif "items" in data:
                items = data["items"]
                nonconv = [x for x in items if x in ITEMS_54_NONCONV]
                control = [x for x in items if x in ITEMS_20_CONTROL]
                return items, nonconv, control
        elif isinstance(data, list):
            nonconv = [x for x in data if x in ITEMS_54_NONCONV]
            control = [x for x in data if x in ITEMS_20_CONTROL]
            return data, nonconv, control
    return ITEMS_54_NONCONV + ITEMS_20_CONTROL, ITEMS_54_NONCONV, ITEMS_20_CONTROL


def compute_metrics(
    target_data: Dict[str, Any],
    ref_data: Optional[Dict[str, Any]],
    items_54: List[int],
    items_20: List[int],
    eps_val: float,
    alpha: float = 0.25,
) -> Dict[str, Any]:
    """Compute metrics (1) to (5) per PREREG_S7_DEQ_v2 §4."""
    # (1) Non-converged 54 fixed-point arrival
    r_lowest_54 = []
    for q in items_54:
        key = f"q{q:04d}_a{alpha:.2f}"
        if key in target_data:
            r_lowest_54.append(float(target_data[key]["r_lowest"]))

    count_54_lt_1e3 = sum(1 for r in r_lowest_54 if r < 1e-3)
    count_54_lt_5e3 = sum(1 for r in r_lowest_54 if r < 5e-3)
    ratio_54_lt_5e3 = float(count_54_lt_5e3 / len(r_lowest_54)) if r_lowest_54 else 0.0

    # (2) Control 20 position stability vs ref
    cos_20 = []
    if ref_data is not None:
        for q in items_20:
            key = f"q{q:04d}_a{alpha:.2f}"
            if key in target_data and key in ref_data:
                z_target = target_data[key]["z_lowest"].reshape(-1).to(torch.float64)
                z_ref = ref_data[key]
                if isinstance(z_ref, dict):
                    z_ref = z_ref.get("z_lowest", z_ref.get("z_star"))
                z_ref = z_ref.reshape(-1).to(torch.float64)
                cos_val = float(torch.nn.functional.cosine_similarity(z_target.unsqueeze(0), z_ref.unsqueeze(0)).item())
                cos_20.append(cos_val)

    min_cos_20 = float(min(cos_20)) if cos_20 else None
    count_cos_ge_09990 = sum(1 for c in cos_20 if c >= 0.9990) if cos_20 else 0
    count_cos_lt_09950 = sum(1 for c in cos_20 if c < 0.9950) if cos_20 else 0

    # (3) Final smooth mapping residual r_eps(z_1000)
    r_1000_all = []
    for k, v in target_data.items():
        if isinstance(v, dict) and "r_1000" in v:
            r_1000_all.append(float(v["r_1000"]))

    med_r1000 = float(np.median(r_1000_all)) if r_1000_all else 0.0
    max_r1000 = float(np.max(r_1000_all)) if r_1000_all else 0.0
    count_r1000_lt_1e3 = sum(1 for r in r_1000_all if r < 1e-3)

    # (4) Boundary step ratio (steps with d <= eps)
    b_ratios_54 = []
    b_ratios_20 = []
    b_last100_54 = []
    b_last100_20 = []

    for q in items_54:
        key = f"q{q:04d}_a{alpha:.2f}"
        if key in target_data:
            entry = target_data[key]
            b_ratios_54.append(float(entry.get("boundary_step_ratio", 0.0)))
            b_last100_54.append(float(entry.get("last_100_boundary_ratio", 0.0)))

    for q in items_20:
        key = f"q{q:04d}_a{alpha:.2f}"
        if key in target_data:
            entry = target_data[key]
            b_ratios_20.append(float(entry.get("boundary_step_ratio", 0.0)))
            b_last100_20.append(float(entry.get("last_100_boundary_ratio", 0.0)))

    b_summary = {
        "group_54": {
            "mean": float(np.mean(b_ratios_54)) if b_ratios_54 else 0.0,
            "median": float(np.median(b_ratios_54)) if b_ratios_54 else 0.0,
            "max": float(np.max(b_ratios_54)) if b_ratios_54 else 0.0,
            "last100_mean": float(np.mean(b_last100_54)) if b_last100_54 else 0.0,
            "last100_median": float(np.median(b_last100_54)) if b_last100_54 else 0.0,
            "last100_max": float(np.max(b_last100_54)) if b_last100_54 else 0.0,
        },
        "group_20": {
            "mean": float(np.mean(b_ratios_20)) if b_ratios_20 else 0.0,
            "median": float(np.median(b_ratios_20)) if b_ratios_20 else 0.0,
            "max": float(np.max(b_ratios_20)) if b_ratios_20 else 0.0,
            "last100_mean": float(np.mean(b_last100_20)) if b_last100_20 else 0.0,
            "last100_median": float(np.median(b_last100_20)) if b_last100_20 else 0.0,
            "last100_max": float(np.max(b_last100_20)) if b_last100_20 else 0.0,
        },
    }

    # (5) Hard machine residual evaluation r(z*_eps)
    r_hard_54 = []
    in_b_lowest_54 = 0
    for q in items_54:
        key = f"q{q:04d}_a{alpha:.2f}"
        if key in target_data:
            entry = target_data[key]
            r_hard_54.append(float(entry.get("r_hard_lowest", 0.0)))
            if entry.get("in_boundary_lowest", False):
                in_b_lowest_54 += 1

    count_hard_54_lt_1e3 = sum(1 for r in r_hard_54 if r < 1e-3)
    count_hard_54_lt_5e3 = sum(1 for r in r_hard_54 if r < 5e-3)

    # Candidate Assessment (§7)
    cond1_met = bool(ratio_54_lt_5e3 >= 0.95 and count_54_lt_5e3 >= 52)
    cond2_met = bool(min_cos_20 is not None and min_cos_20 >= 0.9990)
    shift_flag = bool(min_cos_20 is not None and min_cos_20 < 0.9950)
    is_candidate = bool(cond1_met and cond2_met)

    return {
        "eps": eps_val,
        "metric_1": {
            "evaluated_count": len(r_lowest_54),
            "denominator": 54,
            "count_lt_1e3": count_54_lt_1e3,
            "count_lt_5e3": count_54_lt_5e3,
            "ratio_lt_5e3": ratio_54_lt_5e3,
            "criterion_threshold_count": 52,
            "criterion_met": cond1_met,
        },
        "metric_2": {
            "evaluated_count": len(cos_20),
            "denominator": 20,
            "min_cos": min_cos_20,
            "count_ge_09990": count_cos_ge_09990,
            "count_lt_09950": count_cos_lt_09950,
            "criterion_met": cond2_met,
            "shift_flag": shift_flag,
        },
        "metric_3": {
            "evaluated_count": len(r_1000_all),
            "denominator": len(target_data),
            "median": med_r1000,
            "max": max_r1000,
            "count_lt_1e3": count_r1000_lt_1e3,
        },
        "metric_4": b_summary,
        "metric_5": {
            "evaluated_count": len(r_hard_54),
            "denominator": 54,
            "count_lt_1e3": count_hard_54_lt_1e3,
            "count_lt_5e3": count_hard_54_lt_5e3,
            "in_boundary_count": in_b_lowest_54,
        },
        "candidate_assessment": {
            "cond1_ratio_ge_95pct": cond1_met,
            "cond2_min_cos_ge_09990": cond2_met,
            "is_candidate": is_candidate,
            "shift_flag": shift_flag,
        },
        "p3_baseline_reference": {
            "metric_1_lt_1e3": "53/54 (0.981)",
            "metric_1_lt_5e3": "52/54 (0.963)",
            "metric_2_min_cos": 0.9326308762296536,
            "metric_2_ge_09990": 10,
            "metric_2_lt_09950": 8,
            "note": "P3 metric 1 used mask-freeze mapping; v2 metric 1 uses smooth-switch mapping.",
        },
    }


def format_markdown_table(results: Dict[str, Any]) -> str:
    """Format comparison table alongside P3 baseline."""
    eps = results["eps"]
    m1 = results["metric_1"]
    m2 = results["metric_2"]
    m3 = results["metric_3"]
    m4 = results["metric_4"]
    m5 = results["metric_5"]
    p3 = results["p3_baseline_reference"]

    m2_min_str = f"{m2['min_cos']:.6f}" if m2["min_cos"] is not None else "N/A"

    lines = [
        f"# S7 DEQ Smooth Top-3 Switch Analysis (eps={eps})",
        "",
        "## Comparison Table (Smooth Switch vs P3 Baseline)",
        "| Metric | Definition | Smooth Switch (eps=" + str(eps) + ") | P3 Baseline (Mask Freeze) |",
        "|---|---|:---:|:---:|",
        f"| (1) Fixed-point arrival (54) | r* < 5e-3 count / 54 | {m1['count_lt_5e3']}/{m1['denominator']} ({m1['ratio_lt_5e3']*100:.1f}%) | {p3['metric_1_lt_5e3']} |",
        f"| (1) Strict arrival (54) | r* < 1e-3 count / 54 | {m1['count_lt_1e3']}/{m1['denominator']} | {p3['metric_1_lt_1e3']} |",
        f"| (2) Control stability min cos | min cos(z*_eps, z_ref) | {m2_min_str} | {p3['metric_2_min_cos']:.6f} |",
        f"| (2) Control cos >= 0.9990 | count / 20 | {m2['count_ge_09990']}/{m2['denominator']} | {p3['metric_2_ge_09990']}/20 |",
        f"| (2) Control cos < 0.9950 | count / 20 | {m2['count_lt_09950']}/{m2['denominator']} | {p3['metric_2_lt_09950']}/20 |",
        f"| (3) Final residual r_1000 | median (max) | {m3['median']:.6e} ({m3['max']:.6e}) | - |",
        f"| (4) Boundary step ratio (54) | mean (max) | {m4['group_54']['mean']*100:.1f}% ({m4['group_54']['max']*100:.1f}%) | - |",
        f"| (4) Boundary step ratio (20) | mean (max) | {m4['group_20']['mean']*100:.1f}% ({m4['group_20']['max']*100:.1f}%) | - |",
        f"| (5) Original machine residual (54) | r_hard < 5e-3 count | {m5['count_lt_5e3']}/{m5['denominator']} | - |",
        f"| (5) Boundary fixed-point (54) | d <= eps count | {m5['in_boundary_count']}/{m5['denominator']} | - |",
        "",
        "> [!NOTE]",
        "> P3 metric (1) evaluated residual under mask-freeze mapping; v2 metric (1) evaluates residual under smooth-switch mapping.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze S7 DEQ Smooth Top-3 Switch Results")
    parser.add_argument("--input_pt", type=str, required=True, help="Path to s7deq_soft_v2_eps{eps}.pt")
    parser.add_argument("--items_json", type=str, default="configs/s6_items_74.json", help="Path to items JSON")
    parser.add_argument("--ref_pt", type=str, default=None, help="Path to reference zstar_s6_a0.25.pt")
    parser.add_argument("--eps", type=float, default=0.01, help="Eps value (default: 0.01)")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha damping (default: 0.25)")
    parser.add_argument("--out_json", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--out_md", type=str, required=True, help="Path to output Markdown")
    args = parser.parse_args()

    input_sha = sha256_file(args.input_pt)
    items_sha = sha256_file(args.items_json)
    ref_sha = sha256_file(args.ref_pt) if args.ref_pt else "NONE"
    script_sha = sha256_file(__file__)

    items_74, items_54, items_20 = load_items_config(args.items_json)
    target_data = torch.load(args.input_pt, map_location="cpu")
    ref_data = torch.load(args.ref_pt, map_location="cpu") if args.ref_pt and os.path.exists(args.ref_pt) else None

    results = compute_metrics(
        target_data=target_data,
        ref_data=ref_data,
        items_54=items_54,
        items_20=items_20,
        eps_val=args.eps,
        alpha=args.alpha,
    )

    results["metadata"] = {
        "script_path": __file__,
        "script_sha256": script_sha,
        "input_pt_path": args.input_pt,
        "input_pt_sha256": input_sha,
        "items_json_path": args.items_json,
        "items_json_sha256": items_sha,
        "ref_pt_path": args.ref_pt,
        "ref_pt_sha256": ref_sha,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_content = format_markdown_table(results)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")

    print(f"Analysis saved to {args.out_json} and {args.out_md}")
    print(f"Input SHA: {input_sha}")
    print(f"Metric 1 (<5e-3): {results['metric_1']['count_lt_5e3']}/54 ({results['metric_1']['ratio_lt_5e3']*100:.1f}%)")
    if results["metric_2"]["min_cos"] is not None:
        print(f"Metric 2 (min cos): {results['metric_2']['min_cos']:.6f}")


if __name__ == "__main__":
    main()

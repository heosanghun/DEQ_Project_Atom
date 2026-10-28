#!/usr/bin/env python3
"""scripts/s5_analyze_diag.py - Refined Trajectory Analysis & Report Generator for S5 Task 3
Directive: EXE-260924-30 보충 2 (원장 #335)

Specifications:
- Parses results_s5/diag_nonconv.pt (and raw_s5/diag_nonconv_trace_*.log)
- Computes for each item:
  * min_rho across all 2000 steps and its corresponding step (min_rho_step)
  * rho@1500 and last 500-step delta: Delta_rho = rho@2000 - rho@1500
  * Refined 4-category classification:
    1) monotonic_decrease (단조감소): rel_change > 0.20 and arr[-1] < arr[0] and not oscillation
    2) increase (상승): rel_change < -0.20 and arr[-1] > arr[0] and not oscillation
    3) oscillation (진동): oscillation_rate > 0.35 and cv > 0.05
    4) plateau (정체): |rel_change| <= 0.20 and not oscillation
- Outputs markdown table and JSON with ZERO judgmental sentences.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
DEFAULT_PT_PATH = os.path.join(BASE_S5_DIR, "results_s5", "diag_nonconv.pt")
DEFAULT_RAW_DIR = os.path.join(BASE_S5_DIR, "raw_s5")
DEFAULT_OUT_JSON = os.path.join(BASE_S5_DIR, "results_s5", "diag_nonconv_refined.json")
DEFAULT_OUT_MD = os.path.join(BASE_S5_DIR, "results_s5", "report_EXE_260924_30_c.md")


def classify_refined(arr: np.ndarray) -> str:
    """Refined 4-category classification on post-1000 step trajectory."""
    if len(arr) < 5:
        return "plateau"

    diffs = np.diff(arr)
    sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
    oscillation_rate = sign_changes / max(len(diffs) - 1, 1)

    start_val = float(arr[0])
    end_val = float(arr[-1])
    rel_change = (start_val - end_val) / max(start_val, 1e-12)

    half_idx = len(arr) // 2
    second_half_std = np.std(arr[half_idx:])
    second_half_mean = np.mean(arr[half_idx:])
    cv_second_half = second_half_std / max(second_half_mean, 1e-12)

    if oscillation_rate > 0.35 and cv_second_half > 0.05:
        return "oscillation"
    elif rel_change > 0.20 and end_val < start_val:
        return "monotonic_decrease"
    elif rel_change < -0.20 and end_val > start_val:
        return "increase"
    else:
        return "plateau"


def analyze_diag(
    pt_path: str = DEFAULT_PT_PATH,
    raw_dir: str = DEFAULT_RAW_DIR,
    out_json: str = DEFAULT_OUT_JSON,
    out_md: str = DEFAULT_OUT_MD,
) -> Dict[str, Any]:
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Diagnostic tensor not found: {pt_path}")

    data = torch.load(pt_path, map_location="cpu")
    print(f"Loaded {len(data)} items from {pt_path}")

    # Parse trace log if available to get exact step numbers
    trace_files = sorted(glob.glob(os.path.join(raw_dir, "diag_nonconv_trace_*.log")))
    trace_steps_dict: Dict[int, List[Tuple[int, float]]] = {}
    if trace_files:
        latest_trace = trace_files[-1]
        print(f"Reading trace log: {latest_trace}")
        cur_q = None
        with open(latest_trace, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("--- Item"):
                    # e.g. --- Item 01/60 | Q#0031 | ...
                    parts = line.split("|")
                    if len(parts) >= 2:
                        q_str = parts[1].strip()
                        if "Q#" in q_str:
                            cur_q = int(q_str.replace("Q#", ""))
                            trace_steps_dict[cur_q] = []
                elif cur_q is not None and "\t" in line:
                    parts = line.split("\t")
                    if len(parts) == 2:
                        try:
                            s = int(parts[0])
                            r = float(parts[1])
                            trace_steps_dict[cur_q].append((s, r))
                        except ValueError:
                            pass

    records: List[Dict[str, Any]] = []
    for k, v in data.items():
        q_idx = int(v["question_idx"])
        is_control = bool(v.get("is_control", False))
        rho_traj_step10 = v.get("rho_traj_step10", [])

        # Step 10 list: index i corresponds to step (i+1)*10
        # If trace_steps_dict has exact steps:
        if q_idx in trace_steps_dict and len(trace_steps_dict[q_idx]) > 0:
            step_pairs = trace_steps_dict[q_idx]
        else:
            step_pairs = [((i + 1) * 10, val) for i, val in enumerate(rho_traj_step10)]

        # Find min_rho and min_rho_step
        min_rho = float("inf")
        min_rho_step = -1
        for s, r in step_pairs:
            if r < min_rho:
                min_rho = r
                min_rho_step = s

        # rho@1000, rho@1500, rho@2000
        rho_1000 = float(v.get("rho_1000", -1.0))
        rho_2000 = float(v.get("rho_2000", -1.0))
        rho_1500 = None
        for s, r in step_pairs:
            if s == 1500:
                rho_1500 = r
                break
        if rho_1500 is None:
            # fallback to index 149 in step10
            if len(rho_traj_step10) >= 150:
                rho_1500 = float(rho_traj_step10[149])
            else:
                rho_1500 = rho_1000

        delta_last500 = float(rho_2000 - rho_1500)

        # Refined classification on steps 1010 to 2000
        post_1000_vals = [r for (s, r) in step_pairs if s > 1000]
        if len(post_1000_vals) == 0:
            post_1000_vals = rho_traj_step10[100:]
        shape_refined = classify_refined(np.array(post_1000_vals))
        shape_orig = str(v.get("shape", "unknown"))

        rec = {
            "question_idx": q_idx,
            "is_control": is_control,
            "type": "CTRL" if is_control else "NON-CONV",
            "rho_1000": rho_1000,
            "rho_1500": rho_1500,
            "rho_2000": rho_2000,
            "reached_5e_3_at_2000": bool(rho_2000 < 5e-3),
            "min_rho": min_rho,
            "min_rho_step": min_rho_step,
            "delta_last500": delta_last500,
            "shape_orig": shape_orig,
            "shape_refined": shape_refined,
        }
        records.append(rec)

    records.sort(key=lambda x: x["question_idx"])

    # Classification counts
    counts_orig = {"monotonic_decrease": 0, "plateau": 0, "oscillation": 0, "increase": 0}
    counts_refined = {"monotonic_decrease": 0, "plateau": 0, "oscillation": 0, "increase": 0}
    for r in records:
        counts_orig[r["shape_orig"]] = counts_orig.get(r["shape_orig"], 0) + 1
        counts_refined[r["shape_refined"]] = counts_refined.get(r["shape_refined"], 0) + 1

    summary = {
        "metadata": {
            "directive": "EXE-260924-30 보충 2 (원장 #335)",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_items": len(records),
            "non_converged_count": sum(1 for r in records if not r["is_control"]),
            "control_count": sum(1 for r in records if r["is_control"]),
        },
        "counts_original_3cat": counts_orig,
        "counts_refined_4cat": counts_refined,
        "records": records,
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved refined summary to {out_json}")

    # Generate Markdown Table
    md_lines = []
    md_lines.append("# S5 Phase 1 미수렴 54문항 + 대조군 6문항 NFE 2000 진단 상세 보고서")
    md_lines.append("")
    md_lines.append("- **지시문**: EXE-260924-30 Task 3 및 보충 2 (원장 #334, #335)")
    md_lines.append(f"- **생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"- **대상 문항**: 총 {len(records)}문항 (미수렴 54문항 + 수렴 대조군 6문항)")
    md_lines.append("")
    md_lines.append("## 1. 궤적 분류 기준 (수치 정의)")
    md_lines.append("")
    md_lines.append("| 분류 | 수치 기준 (Step 1010~2000 구간) | 설명 |")
    md_lines.append("|:---|:---|:---|")
    md_lines.append("| **단조감소 (monotonic_decrease)** | `rel_change > 0.20` and `arr[-1] < arr[0]` and not oscillation | 1000스텝 대비 2000스텝 상대 감소율 20% 초과 |")
    md_lines.append("| **상승 (increase)** | `rel_change < -0.20` and `arr[-1] > arr[0]` and not oscillation | 1000스텝 대비 2000스텝 상대 증가율 20% 초과 |")
    md_lines.append("| **진동 (oscillation)** | `oscillation_rate > 0.35` and `cv_second_half > 0.05` | 차분 부호 변경률 35% 초과 및 후반 변동계수 5% 초과 |")
    md_lines.append("| **정체 (plateau)** | `|rel_change| <= 0.20` and not oscillation | 1000스텝 대비 2000스텝 변동폭 ±20% 이내 |")
    md_lines.append("")
    md_lines.append("## 2. 분류 결과 비교 집계")
    md_lines.append("")
    md_lines.append(f"- 기존 3분류: 단조감소 {counts_orig.get('monotonic_decrease', 0)} / 정체 {counts_orig.get('plateau', 0)} / 진동 {counts_orig.get('oscillation', 0)}")
    md_lines.append(f"- 정밀 4분류: 단조감소 {counts_refined.get('monotonic_decrease', 0)} / 상승 {counts_refined.get('increase', 0)} / 정체 {counts_refined.get('plateau', 0)} / 진동 {counts_refined.get('oscillation', 0)}")
    md_lines.append("")
    md_lines.append("## 3. 문항별 전수 실측 데이터 표")
    md_lines.append("")
    md_lines.append("| 번호 | 문항 | 구분 | rho@1000 | rho@2000 | <5e-3? | min_rho | min_rho_step | rho@1500 | delta_last500 | 기존분류 | 정밀분류 |")
    md_lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for i, r in enumerate(records, 1):
        reach_str = "YES" if r["reached_5e_3_at_2000"] else "NO"
        md_lines.append(
            f"| {i:02d} | Q#{r['question_idx']:04d} | {r['type']} | "
            f"{r['rho_1000']:.6e} | {r['rho_2000']:.6e} | {reach_str} | "
            f"{r['min_rho']:.6e} | {r['min_rho_step']:04d} | {r['rho_1500']:.6e} | "
            f"{r['delta_last500']:+.6e} | {r['shape_orig']} | {r['shape_refined']} |"
        )

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved markdown report to {out_md}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze S5 Task 3 diagnostic results.")
    parser.add_argument("--pt", type=str, default=DEFAULT_PT_PATH)
    parser.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_md", type=str, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    analyze_diag(
        pt_path=args.pt,
        raw_dir=args.raw_dir,
        out_json=args.out_json,
        out_md=args.out_md,
    )


if __name__ == "__main__":
    main()

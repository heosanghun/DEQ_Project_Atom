#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_prep_inputs_v2.py - Prepares canonical intermediate inputs v2 with k* source determination.
Directives: TASK-0030 §1 & §2 / AUDIT-P006 재감사 3 / PREREG_S7_DEQ_v1.3 §10(c) 172행

Outputs:
  - results_s7/p5_inputs_train1000_v2.json (Train 1,000 items with k_star, k_star_src)
  - results_s7/p5_inputs_heldout80_v2.json (Held-Out 80 items)
Preserves existing v1 intermediate files without modification (R3-4).

Per-item schema (Train 1,000):
  {
    "q_idx": int,
    "r_1000": float,
    "min_rho": float,
    "r_min": float,
    "src_r1000": str,
    "src_min_rho": str,
    "k_star": int | null,
    "k_star_src": str ("final" | "log_10step" | "undetermined" | "pt_traj")
  }
"""

from __future__ import annotations

import os
import sys
import glob
import re
import json
import math
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Set, Tuple, Optional
import torch

EXPECTED_TRAIN1000_CONFIG_SHA256 = "adb09419c69bda0ca9e6e3975d2eae6fb8d6a7c6dea934cb0aab2a897722c94f"
EXPECTED_HELDOUT80_CONFIG_SHA256 = "27899edbea218b46bc0a5caabada4eb1c1c9dcc17772c001de43374cb208b8c5"
EXPECTED_P1_SHA256 = "97b712ed555cba78113860faf8b3553868ea4ec8ebbb0ad815f46dba041be41a"
EXPECTED_TRAIN1000_PT_SHA256 = "37ea6807d112e44f55ab401f48b6c8e7051f3ca2800aa7cd4caa04a546fb4696"
EXPECTED_HELDOUT80_PT_SHA256 = "bbe7ad3bd2758b4e3ef622ef7c5270307afa5831e4ec6748ca24a126078e79ca"


def compute_sha256(path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_self_sha256() -> str:
    """Compute SHA-256 of this script."""
    return compute_sha256(os.path.abspath(__file__))


def load_config_ids(config_path: str, expected_sha: str, expected_count: int, key_name: str = "indices") -> List[int]:
    """Load and strictly validate a canonical question ID configuration."""
    if not os.path.exists(config_path):
        sys.stderr.write(f"[ERROR] Config file missing: {config_path}\n")
        sys.exit(1)

    actual_sha = compute_sha256(config_path)
    if actual_sha != expected_sha:
        sys.stderr.write(
            f"[ERROR] Config SHA mismatch for {config_path}!\n"
            f"  Expected: {expected_sha}\n"
            f"  Actual:   {actual_sha}\n"
        )
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        ids = data.get(key_name, [])
    elif isinstance(data, list):
        ids = data
    else:
        sys.stderr.write(f"[ERROR] Invalid config format in {config_path}\n")
        sys.exit(1)

    if len(ids) != expected_count:
        sys.stderr.write(f"[ERROR] Expected {expected_count} IDs in {config_path}, found {len(ids)}\n")
        sys.exit(1)

    if len(set(ids)) != expected_count:
        sys.stderr.write(f"[ERROR] Duplicate IDs detected in {config_path}\n")
        sys.exit(1)

    return [int(x) for x in ids]


def parse_log_step_trajectories(log_paths: List[str]) -> Dict[int, List[Tuple[int, float]]]:
    """Parse Picard step trajectories from S7 text logs."""
    q_trajs: Dict[int, List[Tuple[int, float]]] = {}
    for lp in log_paths:
        if not os.path.exists(lp):
            continue
        print(f"Scanning log file: {lp} ({os.path.getsize(lp):,} bytes)")
        current_qid: Optional[int] = None
        in_steps = False
        with open(lp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "--- Item" in line and "Q#" in line:
                    m = re.search(r"Q#(\d+)", line)
                    if m:
                        current_qid = int(m.group(1))
                        in_steps = False
                    continue
                if current_qid is not None and "Step\tRho" in line:
                    in_steps = True
                    if current_qid not in q_trajs:
                        q_trajs[current_qid] = []
                    continue
                if in_steps and current_qid is not None:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        try:
                            step = int(parts[0])
                            rho = float(parts[1])
                            q_trajs[current_qid].append((step, rho))
                        except ValueError:
                            in_steps = False
                    else:
                        if line.startswith("---") or line.strip() == "":
                            in_steps = False
    return q_trajs


def build_train1000_v2(
    train_ids: List[int],
    p1_path: str,
    pt_path: str,
    log_trajs: Dict[int, List[Tuple[int, float]]],
    output_path: str,
    config_path: str,
    script_sha: str
) -> Dict[str, Any]:
    """Build standardized intermediate JSON v2 for Train 1,000 items."""
    print(f"\nBuilding Train 1,000 intermediate v2 -> {output_path}")

    # Check P1
    if not os.path.exists(p1_path):
        sys.stderr.write(f"[ERROR] P1 file missing: {p1_path}\n")
        sys.exit(1)
    p1_sha = compute_sha256(p1_path)
    if p1_sha != EXPECTED_P1_SHA256:
        sys.stderr.write(f"[ERROR] P1 SHA mismatch! Expected {EXPECTED_P1_SHA256}, got {p1_sha}\n")
        sys.exit(1)

    with open(p1_path, "r", encoding="utf-8") as f:
        p1_data = json.load(f)
    if not isinstance(p1_data, list):
        sys.stderr.write(f"[ERROR] P1 JSON is not a list\n")
        sys.exit(1)

    # Check P1 duplicate q_idx
    p1_qids = [x["q_idx"] for x in p1_data if isinstance(x, dict) and "q_idx" in x]
    if len(p1_qids) != len(set(p1_qids)):
        sys.stderr.write(f"[ERROR] Duplicate q_idx detected in P1 JSON ({len(p1_qids)} total, {len(set(p1_qids))} unique)\n")
        sys.exit(1)

    p1_map: Dict[int, float] = {}
    for entry in p1_data:
        if "q_idx" not in entry or "r_1000" not in entry:
            sys.stderr.write(f"[ERROR] P1 entry missing required field: {entry}\n")
            sys.exit(1)
        qid = int(entry["q_idx"])
        r1000 = float(entry["r_1000"])
        if math.isnan(r1000) or math.isinf(r1000):
            sys.stderr.write(f"[ERROR] P1 entry q_idx={qid} has non-finite r_1000={r1000}\n")
            sys.exit(1)
        p1_map[qid] = r1000

    # Check PT
    if not os.path.exists(pt_path):
        sys.stderr.write(f"[ERROR] PT file missing: {pt_path}\n")
        sys.exit(1)
    pt_sha = compute_sha256(pt_path)
    if pt_sha != EXPECTED_TRAIN1000_PT_SHA256:
        sys.stderr.write(f"[ERROR] Train PT SHA mismatch! Expected {EXPECTED_TRAIN1000_PT_SHA256}, got {pt_sha}\n")
        sys.exit(1)

    pt_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Check pt max_nfe and nfe
    for k, v in pt_dict.items():
        if isinstance(v, dict):
            max_nfe = v.get("meta", {}).get("max_nfe")
            nfe = v.get("nfe")
            if max_nfe != 1000 or nfe != 1000:
                sys.stderr.write(f"[ERROR] PT entry {k} has max_nfe={max_nfe}, nfe={nfe} (expected 1000)\n")
                sys.exit(1)

    # Check for rho_traj inside pt
    pt_has_rho_traj = any("rho_traj" in v for v in pt_dict.values() if isinstance(v, dict))
    print(f"Train PT has internal rho_traj: {pt_has_rho_traj}")

    train_set = set(train_ids)
    items: List[Dict[str, Any]] = []

    kstar_stats = {
        "final_count": 0,
        "log_10step_count": 0,
        "undetermined_count": 0,
        "pt_traj_count": 0,
        "kstar_lt_50_count": 0
    }

    step_min_positions: List[int] = []

    for qid in train_ids:
        # Check P1
        if qid not in p1_map:
            sys.stderr.write(f"[ERROR] Train item q_idx={qid} missing from P1 {p1_path}\n")
            sys.exit(1)
        r_1000 = p1_map[qid]

        # Check PT
        pt_key = f"q{qid:04d}_a0.25"
        if pt_key not in pt_dict:
            sys.stderr.write(f"[ERROR] Train item {pt_key} missing from PT {pt_path}\n")
            sys.exit(1)
        pt_entry = pt_dict[pt_key]
        if not isinstance(pt_entry, dict) or "min_rho" not in pt_entry:
            sys.stderr.write(f"[ERROR] PT entry {pt_key} does not contain 'min_rho'\n")
            sys.exit(1)

        min_rho_val = pt_entry["min_rho"]
        min_rho = float(min_rho_val)
        if math.isnan(min_rho) or math.isinf(min_rho):
            sys.stderr.write(f"[ERROR] PT entry {pt_key} has non-finite min_rho={min_rho}\n")
            sys.exit(1)

        # Compute r_min strictly in float64
        four_min_rho = 4.0 * min_rho
        r_min = float(min(four_min_rho, r_1000))

        # Determine k_star and k_star_src
        k_star: Optional[int] = None
        k_star_src: str = "undetermined"

        if pt_has_rho_traj and "rho_traj" in pt_entry:
            # Case (a): rho_traj in pt
            traj = pt_entry["rho_traj"]
            r_traj = [float(rho) / 0.25 for rho in traj]  # alpha = 0.25
            r_all = r_traj[:1000] + [r_1000]
            min_val = min(r_all)
            k_st = min(i for i, val in enumerate(r_all) if val == min_val)
            k_star = k_st
            k_star_src = "pt_traj"
            kstar_stats["pt_traj_count"] += 1
        elif r_1000 <= four_min_rho + 1e-12:
            # r_min = r_1000 (Case b: 626 items) -> k* = 1000
            k_star = 1000
            k_star_src = "final"
            kstar_stats["final_count"] += 1
        else:
            # four_min_rho < r_1000 (Case b: 374 items) -> inspect log
            if qid in log_trajs and len(log_trajs[qid]) > 0:
                steps = log_trajs[qid]
                min_rho_log = min(s[1] for s in steps)
                # Smallest step with min value
                min_step = min(s[0] for s in steps if s[1] == min_rho_log)
                step_min_positions.append(min_step)

                # Check: min_step >= 60 AND all log values in 0..59 are strictly greater
                early_steps = [s for s in steps if s[0] < 60]
                all_early_greater = all(s[1] > min_rho_log for s in early_steps)

                if min_step >= 60 and all_early_greater:
                    k_star = min_step
                    k_star_src = "log_10step"
                    kstar_stats["log_10step_count"] += 1
                else:
                    k_star = None
                    k_star_src = "undetermined"
                    kstar_stats["undetermined_count"] += 1
            else:
                k_star = None
                k_star_src = "undetermined"
                kstar_stats["undetermined_count"] += 1

        if k_star is not None and k_star < 50:
            kstar_stats["kstar_lt_50_count"] += 1

        items.append({
            "q_idx": qid,
            "r_1000": r_1000,
            "min_rho": min_rho,
            "r_min": r_min,
            "src_r1000": f"{p1_path}:q_idx={qid}",
            "src_min_rho": f"{pt_path}:{pt_key}.min_rho",
            "k_star": k_star,
            "k_star_src": k_star_src
        })

    # Integrity assertions
    if len(items) != 1000:
        sys.stderr.write(f"[ERROR] Train items count is {len(items)} (expected 1000)\n")
        sys.exit(1)

    item_ids = [it["q_idx"] for it in items]
    if set(item_ids) != train_set:
        sys.stderr.write(f"[ERROR] Train item IDs set mismatch with canonical config\n")
        sys.exit(1)

    kst_now = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")
    header = {
        "script_name": "scripts/p5_prep_inputs_v2.py",
        "script_sha256": script_sha,
        "created_at": kst_now,
        "target_dataset": "train_1000",
        "expected_count": 1000,
        "note": "min_rho = pt 전 걸음 최솟값(로그 10걸음 아님)·float32 노름",
        "input_file_shas": {
            config_path: EXPECTED_TRAIN1000_CONFIG_SHA256,
            p1_path: p1_sha,
            pt_path: pt_sha
        },
        "kstar_stats": kstar_stats
    }

    out_obj = {
        "header": header,
        "items": items
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    out_sha = compute_sha256(output_path)
    print(f"Generated {output_path}:")
    print(f"  Item count: {len(items)}")
    print(f"  SHA-256:    {out_sha}")
    print(f"  k* stats:   {kstar_stats}")
    if step_min_positions:
        print(f"  Min step positions in log (count {len(step_min_positions)}):")
        print(f"    min: {min(step_min_positions)}, max: {max(step_min_positions)}")
    return out_obj


def build_heldout80_v2(
    heldout_ids: List[int],
    p1_path: str,
    pt_path: str,
    output_path: str,
    config_path: str,
    script_sha: str
) -> Dict[str, Any]:
    """Build standardized intermediate JSON v2 for Held-Out 80 items."""
    print(f"\nBuilding Held-Out 80 intermediate v2 -> {output_path}")

    # Check P1
    if not os.path.exists(p1_path):
        sys.stderr.write(f"[ERROR] P1 file missing: {p1_path}\n")
        sys.exit(1)
    p1_sha = compute_sha256(p1_path)
    if p1_sha != EXPECTED_P1_SHA256:
        sys.stderr.write(f"[ERROR] P1 SHA mismatch! Expected {EXPECTED_P1_SHA256}, got {p1_sha}\n")
        sys.exit(1)

    with open(p1_path, "r", encoding="utf-8") as f:
        p1_data = json.load(f)

    p1_map: Dict[int, float] = {}
    for entry in p1_data:
        qid = int(entry["q_idx"])
        r1000 = float(entry["r_1000"])
        p1_map[qid] = r1000

    # Check PT
    if not os.path.exists(pt_path):
        sys.stderr.write(f"[ERROR] PT file missing: {pt_path}\n")
        sys.exit(1)
    pt_sha = compute_sha256(pt_path)
    if pt_sha != EXPECTED_HELDOUT80_PT_SHA256:
        sys.stderr.write(f"[ERROR] Held-Out 80 PT SHA mismatch! Expected {EXPECTED_HELDOUT80_PT_SHA256}, got {pt_sha}\n")
        sys.exit(1)

    pt_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Check pt max_nfe and nfe
    for k, v in pt_dict.items():
        if isinstance(v, dict):
            max_nfe = v.get("meta", {}).get("max_nfe")
            nfe = v.get("nfe")
            if max_nfe != 1000 or nfe != 1000:
                sys.stderr.write(f"[ERROR] PT entry {k} has max_nfe={max_nfe}, nfe={nfe} (expected 1000)\n")
                sys.exit(1)

    heldout_set = set(heldout_ids)
    items: List[Dict[str, Any]] = []

    for qid in heldout_ids:
        if qid not in p1_map:
            sys.stderr.write(f"[ERROR] Heldout item q_idx={qid} missing from P1 {p1_path}\n")
            sys.exit(1)
        r_1000 = p1_map[qid]

        pt_key = f"q{qid:04d}_a0.25"
        if pt_key not in pt_dict:
            sys.stderr.write(f"[ERROR] Heldout item {pt_key} missing from PT {pt_path}\n")
            sys.exit(1)
        pt_entry = pt_dict[pt_key]
        min_rho = float(pt_entry["min_rho"])

        r_min = float(min(4.0 * min_rho, r_1000))

        # Existing 80 follows PREREG v1.3 §10(c) 173행 (no k* penalty rule)
        items.append({
            "q_idx": qid,
            "r_1000": r_1000,
            "min_rho": min_rho,
            "r_min": r_min,
            "src_r1000": f"{p1_path}:q_idx={qid}",
            "src_min_rho": f"{pt_path}:{pt_key}.min_rho",
            "k_star": None,
            "k_star_src": "not_applicable"
        })

    if len(items) != 80:
        sys.stderr.write(f"[ERROR] Heldout items count is {len(items)} (expected 80)\n")
        sys.exit(1)

    kst_now = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")
    header = {
        "script_name": "scripts/p5_prep_inputs_v2.py",
        "script_sha256": script_sha,
        "created_at": kst_now,
        "target_dataset": "heldout_80",
        "expected_count": 80,
        "note": "min_rho = pt 전 걸음 최솟값(로그 10걸음 아님)·float32 노름",
        "input_file_shas": {
            config_path: EXPECTED_HELDOUT80_CONFIG_SHA256,
            p1_path: p1_sha,
            pt_path: pt_sha
        }
    }

    out_obj = {
        "header": header,
        "items": items
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    out_sha = compute_sha256(output_path)
    print(f"Generated {output_path}:")
    print(f"  Item count: {len(items)}")
    print(f"  SHA-256:    {out_sha}")
    return out_obj


def main():
    script_sha = compute_self_sha256()
    print("=" * 60)
    print(f"p5_prep_inputs_v2.py - Canonical Intermediate Input Generator v2")
    print(f"Self SHA-256: {script_sha}")
    print("=" * 60)

    # 1. Load canonical configs
    train_config_path = "configs/s7_train_1000.json"
    heldout_config_path = "configs/s5_heldout_80.json"
    train_ids = load_config_ids(train_config_path, EXPECTED_TRAIN1000_CONFIG_SHA256, 1000)
    heldout_ids = load_config_ids(heldout_config_path, EXPECTED_HELDOUT80_CONFIG_SHA256, 80)

    # 2. Parse Picard step logs
    log_candidates = [
        "raw_s7/s6_a0.25_20260927_220928.log",
        "raw_s7/s6_a0.25_20260928_060121.log",
        "raw_s7/s7_zstar_pipeline.log"
    ]
    log_trajs = parse_log_step_trajectories(log_candidates)
    print(f"Total question step trajectories parsed: {len(log_trajs)}")

    # 3. Build Train 1,000 intermediate v2
    p1_path = "results_s7/r1000_reeval_P1.json"
    train_pt_path = "results_s7/zstar_train1000_P4.pt"
    train_out_path = "results_s7/p5_inputs_train1000_v2.json"
    build_train1000_v2(
        train_ids=train_ids,
        p1_path=p1_path,
        pt_path=train_pt_path,
        log_trajs=log_trajs,
        output_path=train_out_path,
        config_path=train_config_path,
        script_sha=script_sha
    )

    # 4. Build Held-Out 80 intermediate v2
    heldout_pt_path = "results_s7/zstar_s7_a0.25_400.pt"
    heldout_out_path = "results_s7/p5_inputs_heldout80_v2.json"
    build_heldout80_v2(
        heldout_ids=heldout_ids,
        p1_path=p1_path,
        pt_path=heldout_pt_path,
        output_path=heldout_out_path,
        config_path=heldout_config_path,
        script_sha=script_sha
    )

    print("\n[FINISH] Both canonical intermediate files generated without overwriting existing files.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_survey_kstar.py - Server-side read-only survey of k* sources and logs for TASK-0030 §1.
Zero modification to existing files. Zero GPU usage.
"""

import os
import sys
import json
import glob
import re
import hashlib
import math
from typing import Dict, Any, List, Tuple
import torch

EXPECTED_TRAIN1000_CONFIG_SHA256 = "adb09419c69bda0ca9e6e3975d2eae6fb8d6a7c6dea934cb0aab2a897722c94f"
EXPECTED_P1_SHA256 = "97b712ed555cba78113860faf8b3553868ea4ec8ebbb0ad815f46dba041be41a"


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=== TASK-0030 §1 k* Source Survey ===")
    
    # 1. Survey PT files
    pt_files = [
        "results_s7/zstar_train1000_P4.pt",
        "results_s7/zstar_s7_a0.25_400.pt",
        "results_s7/zstar_s7_a0.25_train680.pt"
    ]
    
    for pt_path in pt_files:
        if not os.path.exists(pt_path):
            print(f"[PT] {pt_path}: NOT FOUND")
            continue
        sha = compute_sha256(pt_path)
        size = os.path.getsize(pt_path)
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        print(f"\n[PT] File: {pt_path}")
        print(f"  Size: {size:,} bytes")
        print(f"  SHA-256: {sha}")
        print(f"  Type: {type(data)}, Key count: {len(data)}")
        
        # Check first entry
        first_k = next(iter(data.keys()))
        first_v = data[first_k]
        if isinstance(first_v, dict):
            fields = list(first_v.keys())
            print(f"  First key: {first_k}")
            print(f"  Fields in entry: {fields}")
            has_rho_traj = any("rho_traj" in str(f) or "traj" in str(f) for f in fields)
            has_kstar = any("k_star" in str(f) or "kstar" in str(f) for f in fields)
            print(f"  Has rho_traj field: {has_rho_traj}")
            print(f"  Has k_star field: {has_kstar}")
            
            # Check meta
            meta = first_v.get("meta", {})
            print(f"  meta.max_nfe: {meta.get('max_nfe')}")
            print(f"  meta.nfe: {meta.get('nfe')}")
            print(f"  nfe: {first_v.get('nfe')}")
            
            # Verify max_nfe and nfe across all entries
            all_max_nfe_1000 = all(
                isinstance(v, dict) and v.get("meta", {}).get("max_nfe") == 1000 
                for v in data.values() if isinstance(v, dict)
            )
            all_nfe_1000 = all(
                isinstance(v, dict) and v.get("nfe") == 1000 
                for v in data.values() if isinstance(v, dict)
            )
            print(f"  All meta.max_nfe == 1000: {all_max_nfe_1000}")
            print(f"  All nfe == 1000: {all_nfe_1000}")
        else:
            print(f"  Values are not dicts: {type(first_v)}")

    # 2. Check P1 duplicate q_idx
    p1_path = "results_s7/r1000_reeval_P1.json"
    if os.path.exists(p1_path):
        with open(p1_path, "r", encoding="utf-8") as f:
            p1_data = json.load(f)
        p1_qids = [x["q_idx"] for x in p1_data]
        p1_dup = len(p1_qids) - len(set(p1_qids))
        print(f"\n[P1] Total items: {len(p1_data)}, Unique q_idx: {len(set(p1_qids))}, Duplicates: {p1_dup}")

    # 3. Survey logs in raw_s7/
    print("\n=== Surveying raw_s7/ Logs ===")
    log_candidates = sorted(glob.glob("raw_s7/*.log"))
    for lc in log_candidates:
        print(f"  Log: {lc} ({os.path.getsize(lc):,} bytes)")

    # 4. Parse logs for 1,000 train items
    config_path = "configs/s7_train_1000.json"
    with open(config_path, "r", encoding="utf-8") as f:
        train_ids = json.load(f)["indices"]
    print(f"\n[Config] Loaded {len(train_ids)} train question IDs from {config_path}")

    # Load intermediate v1 to get B's 374 items where 4*min_rho < r_1000
    v1_intermediate = "results_s7/p5_inputs_train1000.json"
    if not os.path.exists(v1_intermediate):
        print(f"[ERROR] {v1_intermediate} not found")
        sys.exit(1)
    with open(v1_intermediate, "r", encoding="utf-8") as f:
        v1_data = json.load(f)
    v1_items = v1_data["items"]
    
    rmin_lt_r1000_items = []
    rmin_eq_r1000_items = []
    for it in v1_items:
        qid = it["q_idx"]
        r1000 = it["r_1000"]
        min_rho = it["min_rho"]
        if 4.0 * min_rho < r1000:
            rmin_lt_r1000_items.append(it)
        else:
            rmin_eq_r1000_items.append(it)
    print(f"Items with 4*min_rho < r_1000: {len(rmin_lt_r1000_items)} (Expected: 374)")
    print(f"Items with r_min == r_1000: {len(rmin_eq_r1000_items)} (Expected: 626)")

    # Parse logs
    # Find all Q#(\d+) sections in all logs
    q_log_steps: Dict[int, List[Tuple[int, float]]] = {}
    
    for lc in log_candidates:
        print(f"Scanning log: {lc}...")
        current_qid = None
        in_steps = False
        with open(lc, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "--- Item" in line and "Q#" in line:
                    m = re.search(r"Q#(\d+)", line)
                    if m:
                        current_qid = int(m.group(1))
                        in_steps = False
                    continue
                if current_qid is not None and "Step\tRho" in line:
                    in_steps = True
                    if current_qid not in q_log_steps:
                        q_log_steps[current_qid] = []
                    continue
                if in_steps and current_qid is not None:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        try:
                            step = int(parts[0])
                            rho = float(parts[1])
                            q_log_steps[current_qid].append((step, rho))
                        except ValueError:
                            in_steps = False
                    else:
                        if line.startswith("---") or line.strip() == "":
                            in_steps = False

    print(f"Total questions found with step logs: {len(q_log_steps)}")
    
    # Analyze the 374 items
    missing_in_log = 0
    kstar_ge_50_count = 0
    kstar_undetermined_count = 0
    step_positions = []
    
    for it in rmin_lt_r1000_items:
        qid = it["q_idx"]
        if qid not in q_log_steps or len(q_log_steps[qid]) == 0:
            missing_in_log += 1
            kstar_undetermined_count += 1
            continue
        
        steps = q_log_steps[qid]
        # Find min rho in log
        min_rho_log = min(s[1] for s in steps)
        # argmin step (smallest step in case of ties)
        min_step = min(s[0] for s in steps if s[1] == min_rho_log)
        step_positions.append(min_step)
        
        # Check condition: min_step >= 60 and all values in 0..59 are strictly greater
        # Steps in 0..59 are typically 10, 20, 30, 40, 50
        early_steps = [s for s in steps if s[0] < 60]
        all_early_greater = all(s[1] > min_rho_log for s in early_steps)
        
        if min_step >= 60 and all_early_greater:
            kstar_ge_50_count += 1
        else:
            kstar_undetermined_count += 1
            print(f"  [Undetermined] Q#{qid:04d}: min_step={min_step}, min_rho_log={min_rho_log:.4e}, early_steps={[(s[0], f'{s[1]:.4e}') for s in early_steps]}")

    print("\n=== Survey Results for 374 Items ===")
    print(f"  Missing in log: {missing_in_log}")
    print(f"  k* >= 50 확정 count: {kstar_ge_50_count}")
    print(f"  k* 판별 불가 count: {kstar_undetermined_count}")
    if step_positions:
        print(f"  Min step observed: {min(step_positions)}")
        print(f"  Max step observed: {max(step_positions)}")
        # Bin step positions
        bins = {"<50": 0, "50-59": 0, "60-99": 0, "100-299": 0, "300-699": 0, "700-1000": 0}
        for sp in step_positions:
            if sp < 50:
                bins["<50"] += 1
            elif sp < 60:
                bins["50-59"] += 1
            elif sp < 100:
                bins["60-99"] += 1
            elif sp < 300:
                bins["100-299"] += 1
            elif sp < 700:
                bins["300-699"] += 1
            else:
                bins["700-1000"] += 1
        print(f"  Step distribution: {bins}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_prep_inputs.py - Prepares canonical intermediate inputs for S7 G1 gate (①-a & ①-b).
Directive: TASK-0029 §2 / AUDIT-P006 재감사 2 / PREREG_S7_DEQ_v1.3 §10(c) / PROMPT-5 6행

Outputs:
  - results_s7/p5_inputs_train1000.json (Train 1,000 items)
  - results_s7/p5_inputs_heldout80.json (Held-Out 80 items)

Per-item schema:
  {
    "q_idx": int,
    "r_1000": float,
    "min_rho": float,
    "r_min": float,
    "src_r1000": str,
    "src_min_rho": str
  }

r_min is strictly computed as min(4.0 * min_rho, r_1000) using float64.
Strict validation:
  - Exact question ID set match against configs/s7_train_1000.json and configs/s5_heldout_80.json.
  - Zero duplicates, zero missing, zero out-of-set IDs.
  - Zero NaN, zero Inf, zero missing in source. Any defect triggers immediate error exit.
"""

from __future__ import annotations

import os
import sys
import json
import math
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Set
import torch

EXPECTED_TRAIN1000_CONFIG_SHA256 = "adb09419c69bda0ca9e6e3975d2eae6fb8d6a7c6dea934cb0aab2a897722c94f"
EXPECTED_HELDOUT80_CONFIG_SHA256 = "27899edbea218b46bc0a5caabada4eb1c1c9dcc17772c001de43374cb208b8c5"
EXPECTED_P1_SHA256 = "97b712ed555cba78113860faf8b3553868ea4ec8ebbb0ad815f46dba041be41a"


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


def build_train1000_intermediate(
    train_ids: List[int],
    p1_path: str,
    pt_path: str,
    output_path: str,
    config_path: str,
    script_sha: str
) -> Dict[str, Any]:
    """Build standardized intermediate JSON for train 1,000 items."""
    print(f"Building train 1,000 intermediate file -> {output_path}")

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
    pt_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

    train_set = set(train_ids)
    items: List[Dict[str, Any]] = []

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
        r_min = float(min(4.0 * min_rho, r_1000))

        items.append({
            "q_idx": qid,
            "r_1000": r_1000,
            "min_rho": min_rho,
            "r_min": r_min,
            "src_r1000": f"{p1_path}:q_idx={qid}",
            "src_min_rho": f"{pt_path}:{pt_key}.min_rho"
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
        "script_name": "scripts/p5_prep_inputs.py",
        "script_sha256": script_sha,
        "created_at": kst_now,
        "target_dataset": "train_1000",
        "expected_count": 1000,
        "input_file_shas": {
            config_path: EXPECTED_TRAIN1000_CONFIG_SHA256,
            p1_path: p1_sha,
            pt_path: pt_sha
        }
    }

    result = {"header": header, "items": items}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    out_sha = compute_sha256(output_path)
    print(f"Generated {output_path} (count: {len(items)}, SHA-256: {out_sha})")
    return {"path": output_path, "sha256": out_sha, "count": len(items)}


def build_heldout80_intermediate(
    heldout_ids: List[int],
    p1_path: str,
    pt_path: str,
    output_path: str,
    config_path: str,
    script_sha: str
) -> Dict[str, Any]:
    """Build standardized intermediate JSON for held-out 80 items."""
    print(f"Building held-out 80 intermediate file -> {output_path}")

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
    pt_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

    heldout_set = set(heldout_ids)
    items: List[Dict[str, Any]] = []

    for qid in heldout_ids:
        # Check P1
        if qid not in p1_map:
            sys.stderr.write(f"[ERROR] Held-out item q_idx={qid} missing from P1 {p1_path}\n")
            sys.exit(1)
        r_1000 = p1_map[qid]

        # Check PT
        pt_key = f"q{qid:04d}_a0.25"
        if pt_key not in pt_dict:
            sys.stderr.write(f"[ERROR] Held-out item {pt_key} missing from PT {pt_path}\n")
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
        r_min = float(min(4.0 * min_rho, r_1000))

        items.append({
            "q_idx": qid,
            "r_1000": r_1000,
            "min_rho": min_rho,
            "r_min": r_min,
            "src_r1000": f"{p1_path}:q_idx={qid}",
            "src_min_rho": f"{pt_path}:{pt_key}.min_rho"
        })

    # Integrity assertions
    if len(items) != 80:
        sys.stderr.write(f"[ERROR] Held-out items count is {len(items)} (expected 80)\n")
        sys.exit(1)

    item_ids = [it["q_idx"] for it in items]
    if set(item_ids) != heldout_set:
        sys.stderr.write(f"[ERROR] Held-out item IDs set mismatch with canonical config\n")
        sys.exit(1)

    kst_now = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")
    header = {
        "script_name": "scripts/p5_prep_inputs.py",
        "script_sha256": script_sha,
        "created_at": kst_now,
        "target_dataset": "heldout_80",
        "expected_count": 80,
        "input_file_shas": {
            config_path: EXPECTED_HELDOUT80_CONFIG_SHA256,
            p1_path: p1_sha,
            pt_path: pt_sha
        }
    }

    result = {"header": header, "items": items}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    out_sha = compute_sha256(output_path)
    print(f"Generated {output_path} (count: {len(items)}, SHA-256: {out_sha})")
    return {"path": output_path, "sha256": out_sha, "count": len(items)}


def main():
    script_sha = compute_self_sha256()
    print(f"Starting p5_prep_inputs.py (Script SHA: {script_sha})")

    train_config = "configs/s7_train_1000.json"
    heldout_config = "configs/s5_heldout_80.json"
    p1_path = "results_s7/r1000_reeval_P1.json"
    train_pt_path = "results_s7/zstar_train1000_P4.pt"
    heldout_pt_path = "results_s7/zstar_s7_a0.25_400.pt"

    out_train = "results_s7/p5_inputs_train1000.json"
    out_heldout = "results_s7/p5_inputs_heldout80.json"

    # Step 1: Load configs
    train_ids = load_config_ids(train_config, EXPECTED_TRAIN1000_CONFIG_SHA256, 1000, "indices")
    heldout_ids = load_config_ids(heldout_config, EXPECTED_HELDOUT80_CONFIG_SHA256, 80, "indices")

    # Step 2: Build train 1000 intermediate file
    res_train = build_train1000_intermediate(
        train_ids=train_ids,
        p1_path=p1_path,
        pt_path=train_pt_path,
        output_path=out_train,
        config_path=train_config,
        script_sha=script_sha
    )

    # Step 3: Build held-out 80 intermediate file
    res_heldout = build_heldout80_intermediate(
        heldout_ids=heldout_ids,
        p1_path=p1_path,
        pt_path=heldout_pt_path,
        output_path=out_heldout,
        config_path=heldout_config,
        script_sha=script_sha
    )

    print("\n=== GENERATION SUMMARY ===")
    print(f"Train 1000 intermediate: {res_train['path']} | SHA: {res_train['sha256']} | Count: {res_train['count']}")
    print(f"Heldout 80 intermediate: {res_heldout['path']} | SHA: {res_heldout['sha256']} | Count: {res_heldout['count']}")


if __name__ == "__main__":
    main()

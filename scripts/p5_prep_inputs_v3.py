#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_prep_inputs_v3.py - Prepares canonical intermediate inputs v3 with refined k* source determination.
Directives: TASK-0031 §1 / AUDIT-P006 재감사 4 / PREREG_S7_DEQ_v1.3 §10(c) 172행

Outputs:
  - results_s7/p5_inputs_train1000_v3.json (Train 1,000 items with k_star, k_star_src)
  - results_s7/p5_inputs_heldout80_v3.json (Held-Out 80 items)
Preserves existing v1 and v2 intermediate files without modification (R3-4).

Refined k* determination rule (TASK-0031 §1):
  - When r_1000 <= 4 * min_rho (final point reach):
      k_star = 1000, k_star_src = "final"
  - When 4 * min_rho < r_1000:
      Compare log minimum rho_log_min against pt min_rho (relative diff <= 1e-7):
      - If equal: k_star = (min_step_L) - 1 (in 0..999), k_star_src = "log_10step_exact"
      - If rho_log_min > pt min_rho: k_star = null, k_star_src = "undetermined"
      - If rho_log_min < pt min_rho (rel_diff > 1e-7): abort (data correspondence error)
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


def parse_and_validate_logs(log_configs: List[Tuple[str, str]]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, str]]:
    """
    Parse Picard step trajectories from S7 text logs with strict validation (TASK-0031 §1).

    log_configs: List of (log_path, run_source_label), e.g. [("raw_s7/...220928.log", "400"), ...]
    Returns:
      qid_to_log: Dict mapping q_idx to {
        "run_source": str,
        "log_path": str,
        "log_basename": str,
        "steps": List[Tuple[int, float]],  # exactly 100 points (L=10..1000)
        "step_1000_rho": float,
        "min_rho": float,
        "min_step_L": int
      }
      log_shas: Dict mapping log_path to its SHA-256
    """
    qid_to_log: Dict[int, Dict[str, Any]] = {}
    log_shas: Dict[str, str] = {}

    expected_steps = [10 * i for i in range(1, 101)]  # 10, 20, ..., 1000

    for lp, run_src in log_configs:
        if not os.path.exists(lp):
            sys.stderr.write(f"[ERROR] Log file missing: {lp}\n")
            sys.exit(1)

        l_sha = compute_sha256(lp)
        log_shas[lp] = l_sha
        l_basename = os.path.basename(lp)
        print(f"Scanning and validating log: {lp} ({os.path.getsize(lp):,} B, SHA: {l_sha[:16]}...)")

        seen_in_file: Set[int] = set()

        # State machine for parsing
        current_qid: Optional[int] = None
        current_steps: List[Tuple[int, float]] = []
        in_steps = False

        def commit_item(qid: int, steps: List[Tuple[int, float]]):
            if len(steps) != 100:
                sys.stderr.write(f"[ERROR] Item Q#{qid} in {lp} has {len(steps)} points (expected exactly 100)\n")
                sys.exit(1)
            step_nums = [s[0] for s in steps]
            if step_nums != expected_steps:
                sys.stderr.write(f"[ERROR] Item Q#{qid} in {lp} has invalid step sequence\n")
                sys.exit(1)

            if qid in seen_in_file:
                sys.stderr.write(f"[ERROR] Duplicate item Q#{qid} block detected in {lp}\n")
                sys.exit(1)
            seen_in_file.add(qid)

            if qid in qid_to_log:
                sys.stderr.write(
                    f"[ERROR] Item Q#{qid} appears in multiple log files!\n"
                    f"  First seen in: {qid_to_log[qid]['log_path']}\n"
                    f"  Also seen in:  {lp}\n"
                )
                sys.exit(1)

            step_1000 = steps[-1][1]
            min_r = min(s[1] for s in steps)
            min_L = min(s[0] for s in steps if s[1] == min_r)

            qid_to_log[qid] = {
                "run_source": run_src,
                "log_path": lp,
                "log_basename": l_basename,
                "steps": steps,
                "step_1000_rho": step_1000,
                "min_rho": min_r,
                "min_step_L": min_L
            }

        with open(lp, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                if "--- Item" in line and "Q#" in line:
                    if current_qid is not None and in_steps:
                        commit_item(current_qid, current_steps)
                        current_qid = None
                        current_steps = []
                        in_steps = False

                    m = re.search(r"Q#(\d+)", line)
                    if m:
                        current_qid = int(m.group(1))
                        current_steps = []
                        in_steps = False
                    continue

                if current_qid is not None and "Step\tRho" in line:
                    in_steps = True
                    current_steps = []
                    continue

                if in_steps and current_qid is not None:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        try:
                            s_num = int(parts[0])
                            r_val = float(parts[1])
                            current_steps.append((s_num, r_val))
                        except ValueError:
                            in_steps = False
                    else:
                        if line.startswith("---") or line.strip() == "":
                            commit_item(current_qid, current_steps)
                            current_qid = None
                            current_steps = []
                            in_steps = False

        if current_qid is not None and in_steps:
            commit_item(current_qid, current_steps)

        print(f"  -> Extracted {len(seen_in_file)} valid 100-point question items from {l_basename}")

    print(f"Total unique validated log items: {len(qid_to_log)}")
    return qid_to_log, log_shas


def build_train1000_v3(
    train_ids: List[int],
    p1_path: str,
    pt_path: str,
    log_info: Dict[int, Dict[str, Any]],
    log_shas: Dict[str, str],
    output_path: str,
    config_path: str,
    script_sha: str
) -> Dict[str, Any]:
    """Build standardized intermediate JSON v3 for Train 1,000 items (TASK-0031 §1)."""
    print(f"\nBuilding Train 1,000 intermediate v3 -> {output_path}")

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

    items: List[Dict[str, Any]] = []

    kstar_stats = {
        "final_count": 0,
        "log_10step_exact_count": 0,
        "undetermined_count": 0,
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

        # Check PT rho_1000
        if "rho_1000" not in pt_entry:
            sys.stderr.write(f"[ERROR] PT entry {pt_key} does not contain 'rho_1000'\n")
            sys.exit(1)
        pt_rho_1000 = float(pt_entry["rho_1000"])

        # Check Log Entry
        if qid not in log_info:
            sys.stderr.write(f"[ERROR] Train item q_idx={qid} missing from text logs!\n")
            sys.exit(1)
        linfo = log_info[qid]

        # Verify step 1000 rho match between log and pt (TASK-0031 §1)
        log_rho_1000 = linfo["step_1000_rho"]
        diff_1000 = abs(log_rho_1000 - pt_rho_1000)
        rel_diff_1000 = diff_1000 / max(abs(log_rho_1000), abs(pt_rho_1000), 1e-12)
        if rel_diff_1000 > 1e-7:
            sys.stderr.write(
                f"[ERROR] Q#{qid}: log step 1000 rho ({log_rho_1000:.8e}) != pt rho_1000 ({pt_rho_1000:.8e})\n"
                f"  rel_diff={rel_diff_1000:.4e} > 1e-7\n"
            )
            sys.exit(1)

        # Compute r_min strictly in float64
        four_min_rho = 4.0 * min_rho
        r_min = float(min(four_min_rho, r_1000))

        # Determine k_star and k_star_src (TASK-0031 §1)
        k_star: Optional[int] = None
        k_star_src: str = "undetermined"

        if r_1000 <= four_min_rho + 1e-12:
            # Case (1): r_1000 <= 4 * min_rho -> k* = 1000 ("final")
            k_star = 1000
            k_star_src = "final"
            kstar_stats["final_count"] += 1
        else:
            # Case (2): 4 * min_rho < r_1000 -> compare log min against pt min_rho
            rho_log_min = linfo["min_rho"]
            min_L = linfo["min_step_L"]

            diff_min = abs(rho_log_min - min_rho)
            rel_diff_min = diff_min / max(abs(rho_log_min), abs(min_rho), 1e-12)

            if rel_diff_min <= 1e-7:
                # Log minimum matches pt min_rho -> k* confirmed at L - 1
                k_star = min_L - 1
                k_star_src = "log_10step_exact"
                kstar_stats["log_10step_exact_count"] += 1
                step_min_positions.append(k_star)
            elif rho_log_min > min_rho:
                # Log minimum is strictly greater than all-step minimum -> undetermined
                k_star = None
                k_star_src = "undetermined"
                kstar_stats["undetermined_count"] += 1
            else:
                # rho_log_min < min_rho by > 1e-7: impossible under subset sampling, data error
                sys.stderr.write(
                    f"[ERROR] Q#{qid}: log min rho ({rho_log_min:.8e}) < pt min_rho ({min_rho:.8e})\n"
                    f"  rel_diff={rel_diff_min:.4e}\n"
                )
                sys.exit(1)

        if k_star is not None and k_star < 50:
            kstar_stats["kstar_lt_50_count"] += 1

        items.append({
            "q_idx": qid,
            "r_1000": r_1000,
            "min_rho": min_rho,
            "r_min": r_min,
            "src_r1000": f"{p1_path}:q_idx={qid}",
            "src_min_rho": f"{pt_path}:{pt_key}.min_rho",
            "run_source": linfo["run_source"],
            "k_star": k_star,
            "k_star_src": k_star_src
        })

    # Summary
    now_utc = datetime.now(timezone.utc).isoformat()
    config_sha = compute_sha256(config_path)

    input_file_shas = {
        "config_s7_train_1000": config_sha,
        "r1000_reeval_P1": p1_sha,
        "zstar_train1000_P4": pt_sha,
    }
    for lp, lsha in sorted(log_shas.items()):
        input_file_shas[f"log_{os.path.basename(lp)}"] = lsha

    doc = {
        "header": {
            "script_name": "scripts/p5_prep_inputs_v3.py",
            "script_sha256": script_sha,
            "generated_at_utc": now_utc,
            "target_dataset": "train_1000",
            "total_items": len(items),
            "governing_prereg_clause": "PREREG_S7_DEQ_v1.3 §10(c) 172행 (①-a k* < 50 미도달 계산 적용)",
            "norm_convention": "min_rho = pt all-step minimum (float32 norm); log_rho = 10-step sample (float32 norm)",
            "input_file_shas": input_file_shas,
            "kstar_stats": kstar_stats
        },
        "items": items
    }

    # Write out
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    out_sha = compute_sha256(output_path)
    print(f"Generated {output_path}: {len(items)} items, SHA-256: {out_sha}")
    print(f"  k* stats: {kstar_stats}")
    if step_min_positions:
        print(f"  log_10step_exact k* range: min={min(step_min_positions)}, max={max(step_min_positions)}")
    return doc


def build_heldout80_v3(
    heldout_ids: List[int],
    p1_path: str,
    pt_path: str,
    output_path: str,
    config_path: str,
    script_sha: str
) -> Dict[str, Any]:
    """Build standardized intermediate JSON v3 for Held-Out 80 items."""
    print(f"\nBuilding Held-Out 80 intermediate v3 -> {output_path}")

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
    p1_map = {int(x["q_idx"]): float(x["r_1000"]) for x in p1_data if isinstance(x, dict) and "q_idx" in x and "r_1000" in x}

    # Check PT
    if not os.path.exists(pt_path):
        sys.stderr.write(f"[ERROR] PT file missing: {pt_path}\n")
        sys.exit(1)
    pt_sha = compute_sha256(pt_path)
    if pt_sha != EXPECTED_HELDOUT80_PT_SHA256:
        sys.stderr.write(f"[ERROR] Held-Out PT SHA mismatch! Expected {EXPECTED_HELDOUT80_PT_SHA256}, got {pt_sha}\n")
        sys.exit(1)

    pt_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

    items: List[Dict[str, Any]] = []
    for qid in heldout_ids:
        if qid not in p1_map:
            sys.stderr.write(f"[ERROR] Held-out item q_idx={qid} missing from P1\n")
            sys.exit(1)
        r_1000 = p1_map[qid]

        pt_key = f"q{qid:04d}_a0.25"
        if pt_key not in pt_dict:
            sys.stderr.write(f"[ERROR] Held-out item {pt_key} missing from PT\n")
            sys.exit(1)
        pt_entry = pt_dict[pt_key]
        min_rho = float(pt_entry["min_rho"])

        four_min_rho = 4.0 * min_rho
        r_min = float(min(four_min_rho, r_1000))

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

    now_utc = datetime.now(timezone.utc).isoformat()
    config_sha = compute_sha256(config_path)

    doc = {
        "header": {
            "script_name": "scripts/p5_prep_inputs_v3.py",
            "script_sha256": script_sha,
            "generated_at_utc": now_utc,
            "target_dataset": "heldout_80",
            "total_items": len(items),
            "governing_prereg_clause": "PREREG_S7_DEQ_v1.3 §10(c) 173행 (기존 80 k* 규칙 미적용 규약)",
            "norm_convention": "min_rho = pt all-step minimum (float32 norm)",
            "input_file_shas": {
                "config_s5_heldout_80": config_sha,
                "r1000_reeval_P1": p1_sha,
                "zstar_s7_a0.25_400": pt_sha
            }
        },
        "items": items
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    out_sha = compute_sha256(output_path)
    print(f"Generated {output_path}: {len(items)} items, SHA-256: {out_sha}")
    return doc


def main():
    print("=== scripts/p5_prep_inputs_v3.py Starting ===")
    script_sha = compute_self_sha256()
    print(f"Script SHA-256: {script_sha}")

    # Paths
    config_train1000 = "configs/s7_train_1000.json"
    config_heldout80 = "configs/s5_heldout_80.json"
    p1_path = "results_s7/r1000_reeval_P1.json"
    train_pt_path = "results_s7/zstar_train1000_P4.pt"
    heldout_pt_path = "results_s7/zstar_s7_a0.25_400.pt"

    log_configs = [
        ("raw_s7/s6_a0.25_20260927_220928.log", "400"),
        ("raw_s7/s6_a0.25_20260928_060121.log", "680")
    ]

    out_train1000 = "results_s7/p5_inputs_train1000_v3.json"
    out_heldout80 = "results_s7/p5_inputs_heldout80_v3.json"

    # Step 1: Load configs
    train_ids = load_config_ids(config_train1000, EXPECTED_TRAIN1000_CONFIG_SHA256, 1000)
    heldout_ids = load_config_ids(config_heldout80, EXPECTED_HELDOUT80_CONFIG_SHA256, 80)
    print(f"Loaded {len(train_ids)} train IDs and {len(heldout_ids)} heldout IDs.")

    # Step 2: Parse and strictly validate logs
    log_info, log_shas = parse_and_validate_logs(log_configs)

    # Step 3: Build Train 1,000 v3
    train_doc = build_train1000_v3(
        train_ids=train_ids,
        p1_path=p1_path,
        pt_path=train_pt_path,
        log_info=log_info,
        log_shas=log_shas,
        output_path=out_train1000,
        config_path=config_train1000,
        script_sha=script_sha
    )

    # Step 4: Build Held-Out 80 v3
    heldout_doc = build_heldout80_v3(
        heldout_ids=heldout_ids,
        p1_path=p1_path,
        pt_path=heldout_pt_path,
        output_path=out_heldout80,
        config_path=config_heldout80,
        script_sha=script_sha
    )

    print("\n=== scripts/p5_prep_inputs_v3.py Completed Cleanly ===")


if __name__ == "__main__":
    main()

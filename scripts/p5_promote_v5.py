#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_promote_v5.py - CTS S7 P5 Result Analyzer & Promotion Evaluator v5
Directives: TASK-0030 §3 / AUDIT-P006 재감사 3 / PREREG_S7_DEQ_v1.3 §10(b)(c)(d) / GOAL.md G1

Document Definitions Quoted:
  - PROMPT-5 6행: "①-a S_train min(4·min ρ, r_1000)"
  - PREREG_S7_DEQ_v1.3 §10(c) 172행: "①-a … r_min = min(4·min_{k<=999} rho_k, r_1000) < 10^-2 … (k* < 50 미도달 계산은 로그 궤적으로 동일 적용)"
  - PREREG_S7_DEQ_v1.3 §10(c) 173행: "①-b … 기존 80 (로그 r_min)"
  - Governing Note:
      For ①-a (Train 1,000), the k* < 50 non-convergence rule applies via Picard log trajectory.
      Items with undetermined k* (null) are counted under reason 'kstar_undetermined' without defaults.
      Both conservative (undetermined counted as non-converged) and excluded-undetermined ratios are reported;
      the formal judgment value is the conservative ratio.
      For Existing 80, following PREREG line 173 text, no k* rule applies.

Operational Standards & Audit Compliance:
  1. Config integrity: Enforces canonical 520 SHA-256 (48abce65...), canonical 80 SHA-256 (27899edb...),
     and canonical train 1,000 SHA-256 (adb09419...).
  2. Intermediate input protocol v2:
     - Accepts standardized intermediate files with valid 'header' and 'items'. Bare lists rejected.
     - Header must include script_sha256, target_dataset, and input_file_shas matching expected config and P1 SHAs.
     - Per-item schema strictly checked:
         train_1000: {q_idx, r_1000, min_rho, r_min, src_r1000, src_min_rho, k_star, k_star_src}
         heldout_80: {q_idx, r_1000, min_rho, r_min, src_r1000, src_min_rho}
     - Unconditional question ID set matching against canonical configs. Zero missing, zero duplicates, zero mismatch.
     - Strict re-verification: |r_min - min(4 * min_rho, r_1000)| <= 1e-12.
  3. Strict key queries: Looks up exact keys f"q{idx:04d}_a0.25" for canonical 520 IDs.
  4. Fixed denominators: All rates computed against fixed denominators (520, 600, 1000). Missing/invalid items counted as non-converged.
  5. Missing k* handling: If k_star is absent or None in new 520 records, no default value is used; counted as non-converged with reason missing_field.
  6. Trajectory classification: Implements exact 5-branch taxonomy (unclassified, monotonic_decrease, increase, plateau, oscillation)
     on steps 500..1000 with 1:1 text-to-code alignment including zero guards.
  7. Rescue items: Strictly PREREG §10(d)3 (r_1000 >= 1e-2 and r_lowest < 1e-2); k* < 50 items tracked and reported separately.
  8. Output metadata: Includes script runtime SHA-256, config SHAs, input file SHAs, and cos_nan_count.
  9. Neutral quantitative reporting: Strictly zero forbidden qualitative judgment words.
"""

from __future__ import annotations

import os
import sys
import json
import math
import hashlib
import argparse
from typing import Dict, Any, List, Optional, Tuple, Set
import torch

EXPECTED_CONFIG_SHA256 = "48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988"
EXPECTED_HELDOUT80_SHA256 = "27899edbea218b46bc0a5caabada4eb1c1c9dcc17772c001de43374cb208b8c5"
EXPECTED_TRAIN1000_SHA256 = "adb09419c69bda0ca9e6e3975d2eae6fb8d6a7c6dea934cb0aab2a897722c94f"
EXPECTED_P1_SHA256 = "97b712ed555cba78113860faf8b3553868ea4ec8ebbb0ad815f46dba041be41a"

TRAJECTORY_RULE_SPEC = (
    "Rules for trajectory shape classification on the last 500 steps (steps 500..1000) of r_k = rho_k / alpha (alpha=0.25):\n"
    "0. unclassified: rho_traj length != 1000 or sub_r length != 501 (input defect, not a valid shape).\n"
    "1. monotonic_decrease: fraction of steps where r_k > r_{k-1} is < 0.05 (< 5%).\n"
    "2. increase: r_500 > 0 and r_1000 > 2.0 * r_500.\n"
    "3. plateau: min(sub_r) > 0 and (max(sub_r) - min(sub_r)) / min(sub_r) < 0.10 (< 10%).\n"
    "4. oscillation: trajectory does not meet criteria 1, 2, or 3."
)


def compute_sha256(path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_self_sha256() -> str:
    """Compute runtime SHA-256 of this script."""
    return compute_sha256(os.path.abspath(__file__))


def load_canonical_config(config_path: str = "configs/s7_new_heldout_520.json") -> Tuple[List[int], str]:
    """Load and verify canonical 520 heldout config."""
    if not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "s7_new_heldout_520.json")
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            sys.stderr.write(f"[ERROR] Canonical 520 config missing: {config_path}\n")
            sys.exit(1)

    actual_sha = compute_sha256(config_path)
    if actual_sha != EXPECTED_CONFIG_SHA256:
        sys.stderr.write(
            f"[ERROR] Canonical 520 config SHA mismatch for {config_path}!\n"
            f"  Expected: {EXPECTED_CONFIG_SHA256}\n"
            f"  Actual:   {actual_sha}\n"
        )
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    indices = data.get("indices", [])
    if len(indices) != 520:
        sys.stderr.write(f"[ERROR] Config does not contain exactly 520 items (got {len(indices)})\n")
        sys.exit(1)

    if len(set(indices)) != 520:
        sys.stderr.write(f"[ERROR] Config contains duplicate item IDs!\n")
        sys.exit(1)

    return indices, actual_sha


def load_canonical_heldout80_config(config_path: str = "configs/s5_heldout_80.json") -> Tuple[List[int], str]:
    """Load and verify canonical 80 heldout config."""
    if not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "s5_heldout_80.json")
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            sys.stderr.write(f"[ERROR] Canonical heldout 80 config missing: {config_path}\n")
            sys.exit(1)

    actual_sha = compute_sha256(config_path)
    if actual_sha != EXPECTED_HELDOUT80_SHA256:
        sys.stderr.write(
            f"[ERROR] Canonical heldout 80 config SHA mismatch for {config_path}!\n"
            f"  Expected: {EXPECTED_HELDOUT80_SHA256}\n"
            f"  Actual:   {actual_sha}\n"
        )
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    indices = data.get("indices", data.get("heldout_indices", []))
    if len(indices) != 80:
        sys.stderr.write(f"[ERROR] Heldout 80 config does not contain exactly 80 items (got {len(indices)})\n")
        sys.exit(1)

    if len(set(indices)) != 80:
        sys.stderr.write(f"[ERROR] Heldout 80 config contains duplicate item IDs!\n")
        sys.exit(1)

    return [int(x) for x in indices], actual_sha


def load_canonical_train1000_config(config_path: str = "configs/s7_train_1000.json") -> Tuple[List[int], str]:
    """Load and verify canonical train 1,000 config."""
    if not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "s7_train_1000.json")
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            sys.stderr.write(f"[ERROR] Canonical train 1000 config missing: {config_path}\n")
            sys.exit(1)

    actual_sha = compute_sha256(config_path)
    if actual_sha != EXPECTED_TRAIN1000_SHA256:
        sys.stderr.write(
            f"[ERROR] Canonical train 1000 config SHA mismatch for {config_path}!\n"
            f"  Expected: {EXPECTED_TRAIN1000_SHA256}\n"
            f"  Actual:   {actual_sha}\n"
        )
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    indices = data.get("indices", [])
    if len(indices) != 1000:
        sys.stderr.write(f"[ERROR] Train 1000 config does not contain exactly 1000 items (got {len(indices)})\n")
        sys.exit(1)

    if len(set(indices)) != 1000:
        sys.stderr.write(f"[ERROR] Train 1000 config contains duplicate item IDs!\n")
        sys.exit(1)

    return [int(x) for x in indices], actual_sha


def load_intermediate_input(
    input_path: Optional[str],
    expected_count: int,
    canonical_ids: List[int],
    dataset_name: str
) -> Tuple[Dict[str, Any], str]:
    """Load and strictly validate a standardized intermediate JSON input file (TASK-0030 §3)."""
    if input_path is None or not os.path.exists(input_path):
        sys.stderr.write(f"[ERROR] Required intermediate input file missing or not provided: {input_path}\n")
        sys.exit(1)

    actual_sha = compute_sha256(input_path)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to parse JSON from {input_path}: {e}\n")
        sys.exit(1)

    if not isinstance(raw_data, dict):
        sys.stderr.write(f"[ERROR] Intermediate input {input_path} root is not a dict (bare lists rejected)\n")
        sys.exit(1)

    # Validate header
    header = raw_data.get("header")
    if not isinstance(header, dict):
        sys.stderr.write(f"[ERROR] Intermediate input {input_path} missing valid 'header' dictionary\n")
        sys.exit(1)

    script_sha = header.get("script_sha256")
    if not script_sha or not isinstance(script_sha, str):
        sys.stderr.write(f"[ERROR] Intermediate input {input_path} header missing 'script_sha256'\n")
        sys.exit(1)

    target_dataset = header.get("target_dataset")
    if target_dataset != dataset_name:
        sys.stderr.write(f"[ERROR] Intermediate input {input_path} target_dataset '{target_dataset}' != expected '{dataset_name}'\n")
        sys.exit(1)

    input_file_shas = header.get("input_file_shas")
    if not isinstance(input_file_shas, dict):
        sys.stderr.write(f"[ERROR] Intermediate input {input_path} header missing 'input_file_shas' dict\n")
        sys.exit(1)

    sha_values = list(input_file_shas.values())
    if EXPECTED_P1_SHA256 not in sha_values:
        sys.stderr.write(f"[ERROR] Intermediate input {input_path} header input_file_shas missing expected P1 SHA\n")
        sys.exit(1)

    if dataset_name == "train_1000" and EXPECTED_TRAIN1000_SHA256 not in sha_values:
        sys.stderr.write(f"[ERROR] Intermediate input {input_path} header input_file_shas missing expected train 1000 config SHA\n")
        sys.exit(1)

    if dataset_name == "heldout_80" and EXPECTED_HELDOUT80_SHA256 not in sha_values:
        sys.stderr.write(f"[ERROR] Intermediate input {input_path} header input_file_shas missing expected heldout 80 config SHA\n")
        sys.exit(1)

    # Validate items
    if "items" not in raw_data or not isinstance(raw_data["items"], list):
        sys.stderr.write(f"[ERROR] Intermediate JSON {input_path} missing top-level 'items' array\n")
        sys.exit(1)

    items_list = raw_data["items"]
    if len(items_list) != expected_count:
        sys.stderr.write(
            f"[ERROR] {dataset_name} intermediate file item count mismatch: "
            f"expected {expected_count}, got {len(items_list)}\n"
        )
        sys.exit(1)

    req_fields = ["q_idx", "r_1000", "min_rho", "r_min", "src_r1000", "src_min_rho"]
    if dataset_name == "train_1000":
        req_fields.extend(["k_star", "k_star_src"])

    parsed_qids: List[int] = []
    r1000_vals: List[float] = []
    rmin_vals: List[float] = []

    kstar_lt_50_count = 0
    kstar_undetermined_count = 0
    kstar_ge_50_count = 0
    conservative_count = 0

    for idx, item in enumerate(items_list):
        if not isinstance(item, dict):
            sys.stderr.write(f"[ERROR] Item {idx} in {dataset_name} is not a dict\n")
            sys.exit(1)

        # Check all required fields
        for field in req_fields:
            if field not in item:
                sys.stderr.write(f"[ERROR] Item {idx} in {dataset_name} missing required field '{field}'\n")
                sys.exit(1)

        # Validate q_idx
        qid_raw = item["q_idx"]
        if not isinstance(qid_raw, int) or isinstance(qid_raw, bool):
            sys.stderr.write(f"[ERROR] Item {idx} in {dataset_name} has non-integer q_idx: {qid_raw}\n")
            sys.exit(1)
        parsed_qids.append(qid_raw)

        # Validate numeric types and finiteness
        try:
            r1000 = float(item["r_1000"])
            min_rho = float(item["min_rho"])
            r_min = float(item["r_min"])
        except (ValueError, TypeError):
            sys.stderr.write(f"[ERROR] Item {idx} in {dataset_name} has non-numeric float values\n")
            sys.exit(1)

        if math.isnan(r1000) or math.isinf(r1000):
            sys.stderr.write(f"[ERROR] Item {idx} (q_idx={qid_raw}) has non-finite r_1000: {r1000}\n")
            sys.exit(1)
        if math.isnan(min_rho) or math.isinf(min_rho):
            sys.stderr.write(f"[ERROR] Item {idx} (q_idx={qid_raw}) has non-finite min_rho: {min_rho}\n")
            sys.exit(1)
        if math.isnan(r_min) or math.isinf(r_min):
            sys.stderr.write(f"[ERROR] Item {idx} (q_idx={qid_raw}) has non-finite r_min: {r_min}\n")
            sys.exit(1)

        # Re-verify r_min == min(4 * min_rho, r_1000) within 1e-12 tolerance
        expected_rmin = min(4.0 * min_rho, r1000)
        if abs(r_min - expected_rmin) > 1e-12:
            sys.stderr.write(
                f"[ERROR] Item {idx} (q_idx={qid_raw}) r_min verification failed!\n"
                f"  Recorded r_min: {r_min}\n"
                f"  Recomputed:     {expected_rmin}\n"
                f"  Diff:           {abs(r_min - expected_rmin)}\n"
            )
            sys.exit(1)

        # Validate provenance string fields
        if not isinstance(item["src_r1000"], str) or not isinstance(item["src_min_rho"], str):
            sys.stderr.write(f"[ERROR] Item {idx} in {dataset_name} has non-string provenance fields\n")
            sys.exit(1)

        # Train 1,000 specific k* validations
        if dataset_name == "train_1000":
            k_st_val = item["k_star"]
            if k_st_val is None:
                kstar_undetermined_count += 1
            else:
                try:
                    k_st_f = float(k_st_val)
                    if math.isnan(k_st_f) or math.isinf(k_st_f) or k_st_f != math.floor(k_st_f) or not (0 <= k_st_f <= 1000):
                        sys.stderr.write(f"[ERROR] Item {idx} in train_1000 has invalid k_star value: {k_st_val}\n")
                        sys.exit(1)
                    k_st_int = int(k_st_f)
                    if k_st_int < 50:
                        kstar_lt_50_count += 1
                    else:
                        kstar_ge_50_count += 1
                        if r_min < 1e-2:
                            conservative_count += 1
                except (ValueError, TypeError):
                    sys.stderr.write(f"[ERROR] Item {idx} in train_1000 has non-numeric k_star: {k_st_val}\n")
                    sys.exit(1)

        r1000_vals.append(r1000)
        rmin_vals.append(r_min)

    # UNCONDITIONAL question ID set comparison
    if len(set(parsed_qids)) != expected_count:
        sys.stderr.write(f"[ERROR] {dataset_name} contains duplicate question IDs!\n")
        sys.exit(1)

    canonical_set = set(canonical_ids)
    parsed_set = set(parsed_qids)
    if parsed_set != canonical_set:
        missing_from_file = canonical_set - parsed_set
        extra_in_file = parsed_set - canonical_set
        sys.stderr.write(
            f"[ERROR] {dataset_name} question ID set mismatch with canonical config!\n"
            f"  Missing from file ({len(missing_from_file)}): {sorted(list(missing_from_file))[:10]}\n"
            f"  Extra in file ({len(extra_in_file)}): {sorted(list(extra_in_file))[:10]}\n"
        )
        sys.exit(1)

    # Compute metric counts
    r1000_lt_1e2 = sum(1 for v in r1000_vals if v < 1e-2)
    r1000_lt_5e3 = sum(1 for v in r1000_vals if v < 5e-3)
    rmin_lt_1e2 = sum(1 for v in rmin_vals if v < 1e-2)
    rmin_lt_5e3 = sum(1 for v in rmin_vals if v < 5e-3)

    stats = {
        "n": expected_count,
        "source": input_path,
        "source_sha256": actual_sha,
        "r1000_values": r1000_vals,
        "rmin_values": rmin_vals,
        "r1000_lt_1e2_count": r1000_lt_1e2,
        "r1000_lt_1e2_ratio": round(r1000_lt_1e2 / float(expected_count), 6),
        "r1000_lt_5e3_count": r1000_lt_5e3,
        "r1000_lt_5e3_ratio": round(r1000_lt_5e3 / float(expected_count), 6),
        "rmin_lt_1e2_count": rmin_lt_1e2,
        "rmin_lt_1e2_ratio": round(rmin_lt_1e2 / float(expected_count), 6),
        "rmin_lt_5e3_count": rmin_lt_5e3,
        "rmin_lt_5e3_ratio": round(rmin_lt_5e3 / float(expected_count), 6),
    }

    if dataset_name == "train_1000":
        stats["kstar_lt_50_count"] = kstar_lt_50_count
        stats["kstar_undetermined_count"] = kstar_undetermined_count
        stats["kstar_ge_50_count"] = kstar_ge_50_count
        stats["conservative_count"] = conservative_count
        stats["conservative_ratio"] = round(conservative_count / float(expected_count), 6)
        excl_denom = expected_count - kstar_undetermined_count
        stats["excl_undetermined_denom"] = excl_denom
        stats["excl_undetermined_count"] = conservative_count
        stats["excl_undetermined_ratio"] = round(conservative_count / float(excl_denom), 6) if excl_denom > 0 else 0.0

    return stats, actual_sha


def classify_trajectory_shape_v4(rho_traj: List[float], r_1000: float, alpha: float = 0.25) -> str:
    """Classify Picard trajectory on the last 500 steps (steps 500..1000)."""
    if len(rho_traj) != 1000:
        return "unclassified"

    r_traj = [float(rho) / alpha for rho in rho_traj]
    sub_r = r_traj[500:] + [float(r_1000)]
    if len(sub_r) != 501:
        return "unclassified"

    inc_count = sum(1 for i in range(1, len(sub_r)) if sub_r[i] > sub_r[i - 1])
    fraction_inc = inc_count / 500.0

    r_500 = sub_r[0]
    min_val = min(sub_r)
    max_val = max(sub_r)

    # 1. Monotonic decrease
    if fraction_inc < 0.05:
        return "monotonic_decrease"

    # 2. Increase
    if r_500 > 0 and float(r_1000) > 2.0 * r_500:
        return "increase"

    # 3. Plateau
    if min_val > 0 and (max_val - min_val) / min_val < 0.10:
        return "plateau"

    # 4. Oscillation
    return "oscillation"


def compute_rescue_window_max(rho_traj: List[float], r_1000: float, k_star: int, alpha: float = 0.25) -> float:
    """Compute maximum r in the window [max(0, k*-10), min(1000, k*+10)]."""
    r_traj = [float(rho) / alpha for rho in rho_traj]
    k_min = max(0, k_star - 10)
    k_max = min(1000, k_star + 10)
    vals = [r_traj[k] for k in range(k_min, min(k_max + 1, 1000))]
    if k_max >= 1000:
        vals.append(float(r_1000))
    return float(max(vals))


def analyze_p5_promote_v5(
    heldout520_path: str,
    heldout80_intermediate_path: str,
    train1000_intermediate_path: str,
    config_path: str = "configs/s7_new_heldout_520.json",
    heldout80_config_path: str = "configs/s5_heldout_80.json",
    train1000_config_path: str = "configs/s7_train_1000.json",
) -> Dict[str, Any]:
    """Execute complete analysis and promotion evaluation v5."""
    # Step 1: Load and verify canonical configs
    canonical_520_ids, config_sha = load_canonical_config(config_path)
    canonical_80_ids, h80_config_sha = load_canonical_heldout80_config(heldout80_config_path)
    canonical_1000_ids, t1000_config_sha = load_canonical_train1000_config(train1000_config_path)

    # Step 2: Load intermediate files for Existing 80 and Train 1,000 (Mandatory & Strict)
    rec_80, h80_intermediate_sha = load_intermediate_input(
        input_path=heldout80_intermediate_path,
        expected_count=80,
        canonical_ids=canonical_80_ids,
        dataset_name="heldout_80"
    )

    rec_train1000, train1000_intermediate_sha = load_intermediate_input(
        input_path=train1000_intermediate_path,
        expected_count=1000,
        canonical_ids=canonical_1000_ids,
        dataset_name="train_1000"
    )

    # Step 3: Load and inspect heldout 520 results pt
    if not os.path.exists(heldout520_path):
        sys.stderr.write(f"[ERROR] Heldout 520 results file missing: {heldout520_path}\n")
        sys.exit(1)

    heldout_sha = compute_sha256(heldout520_path)
    try:
        heldout_data = torch.load(heldout520_path, map_location="cpu", weights_only=False)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to load pt file {heldout520_path}: {e}\n")
        sys.exit(1)

    if not isinstance(heldout_data, dict):
        sys.stderr.write(f"[ERROR] Root structure in {heldout520_path} is not a dictionary\n")
        sys.exit(1)

    # Scan and count non-canonical keys
    canonical_key_set = {f"q{idx:04d}_a0.25" for idx in canonical_520_ids}
    extra_keys_count = 0
    alpha_collision_count = 0

    for k in heldout_data.keys():
        if k not in canonical_key_set:
            extra_keys_count += 1
            if "_a" in str(k) and not str(k).endswith("_a0.25"):
                alpha_collision_count += 1

    # Query exactly canonical 520 items
    parsed_items: Dict[int, Dict[str, Any]] = {}
    invalid_reasons: Dict[str, int] = {
        "missing_item": 0,
        "missing_field": 0,
        "nan_inf_r": 0,
        "invalid_kstar": 0,
    }

    req_keys = [
        "r_1000",
        "r_lowest",
        "k_star",
        "rho_traj",
        "top3_final",
        "top3_lowest",
        "z_star",
        "z_lowest",
    ]

    for qid in canonical_520_ids:
        key = f"q{qid:04d}_a0.25"
        if key not in heldout_data:
            invalid_reasons["missing_item"] += 1
            continue

        item_raw = heldout_data[key]
        if not isinstance(item_raw, dict):
            invalid_reasons["missing_item"] += 1
            continue

        missing_fields = [k for k in req_keys if k not in item_raw or item_raw[k] is None]
        if missing_fields:
            invalid_reasons["missing_field"] += 1
            continue

        try:
            r1000_f = float(item_raw["r_1000"])
            rlowest_f = float(item_raw["r_lowest"])
            if math.isnan(r1000_f) or math.isinf(r1000_f) or math.isnan(rlowest_f) or math.isinf(rlowest_f):
                invalid_reasons["nan_inf_r"] += 1
                continue
        except (ValueError, TypeError):
            invalid_reasons["nan_inf_r"] += 1
            continue

        try:
            k_st_val = float(item_raw["k_star"])
            if math.isnan(k_st_val) or math.isinf(k_st_val):
                invalid_reasons["invalid_kstar"] += 1
                continue
            if not (0 <= k_st_val <= 1000) or (k_st_val != math.floor(k_st_val)):
                invalid_reasons["invalid_kstar"] += 1
                continue
            k_st = int(k_st_val)
        except (ValueError, TypeError):
            invalid_reasons["invalid_kstar"] += 1
            continue

        parsed_items[qid] = {
            "qid": qid,
            "r_1000": r1000_f,
            "r_lowest": rlowest_f,
            "k_star": k_st,
            "rho_traj": item_raw["rho_traj"],
            "top3_final": item_raw["top3_final"],
            "top3_lowest": item_raw["top3_lowest"],
            "z_star": item_raw["z_star"],
            "z_lowest": item_raw["z_lowest"],
        }

    # Step 4: Metric evaluations
    N_520 = 520
    N_80 = 80
    N_600 = 600
    N_1000 = 1000

    # 1. Primary convergence (r_1000 < 1e-2 and < 5e-3)
    heldout520_r1000_lt_1e2 = sum(1 for it in parsed_items.values() if it["r_1000"] < 1e-2)
    heldout520_r1000_lt_5e3 = sum(1 for it in parsed_items.values() if it["r_1000"] < 5e-3)
    heldout520_rlow_lt_1e2 = sum(1 for it in parsed_items.values() if it["r_lowest"] < 1e-2)
    heldout520_rlow_lt_5e3 = sum(1 for it in parsed_items.values() if it["r_lowest"] < 5e-3)

    combined600_r1000_lt_1e2 = heldout520_r1000_lt_1e2 + rec_80["r1000_lt_1e2_count"]
    combined600_r1000_lt_5e3 = heldout520_r1000_lt_5e3 + rec_80["r1000_lt_5e3_count"]

    primary_convergence = {
        "heldout520": {
            "total": N_520,
            "valid_parsed": len(parsed_items),
            "r1000_lt_1e2_count": heldout520_r1000_lt_1e2,
            "r1000_lt_1e2_ratio": round(heldout520_r1000_lt_1e2 / float(N_520), 6),
            "r1000_lt_5e3_count": heldout520_r1000_lt_5e3,
            "r1000_lt_5e3_ratio": round(heldout520_r1000_lt_5e3 / float(N_520), 6),
            "r_lowest_lt_1e2_count": heldout520_rlow_lt_1e2,
            "r_lowest_lt_1e2_ratio": round(heldout520_rlow_lt_1e2 / float(N_520), 6),
            "r_lowest_lt_5e3_count": heldout520_rlow_lt_5e3,
            "r_lowest_lt_5e3_ratio": round(heldout520_rlow_lt_5e3 / float(N_520), 6),
        },
        "existing80": {
            "total": N_80,
            "r1000_lt_1e2_count": rec_80["r1000_lt_1e2_count"],
            "r1000_lt_1e2_ratio": rec_80["r1000_lt_1e2_ratio"],
            "r1000_lt_5e3_count": rec_80["r1000_lt_5e3_count"],
            "r1000_lt_5e3_ratio": rec_80["r1000_lt_5e3_ratio"],
            "rmin_lt_1e2_count": rec_80["rmin_lt_1e2_count"],
            "rmin_lt_1e2_ratio": rec_80["rmin_lt_1e2_ratio"],
            "rmin_lt_5e3_count": rec_80["rmin_lt_5e3_count"],
            "rmin_lt_5e3_ratio": rec_80["rmin_lt_5e3_ratio"],
        },
        "combined600": {
            "total": N_600,
            "r1000_lt_1e2_count": combined600_r1000_lt_1e2,
            "r1000_lt_1e2_ratio": round(combined600_r1000_lt_1e2 / float(N_600), 6),
            "r1000_lt_5e3_count": combined600_r1000_lt_5e3,
            "r1000_lt_5e3_ratio": round(combined600_r1000_lt_5e3 / float(N_600), 6),
        },
        "train1000": {
            "total": N_1000,
            "r1000_lt_1e2_count": rec_train1000["r1000_lt_1e2_count"],
            "r1000_lt_1e2_ratio": rec_train1000["r1000_lt_1e2_ratio"],
            "r1000_lt_5e3_count": rec_train1000["r1000_lt_5e3_count"],
            "r1000_lt_5e3_ratio": rec_train1000["r1000_lt_5e3_ratio"],
            "rmin_lt_1e2_count": rec_train1000["rmin_lt_1e2_count"],
            "rmin_lt_1e2_ratio": rec_train1000["rmin_lt_1e2_ratio"],
            "rmin_lt_5e3_count": rec_train1000["rmin_lt_5e3_count"],
            "rmin_lt_5e3_ratio": rec_train1000["rmin_lt_5e3_ratio"],
        }
    }

    # 2. Secondary promotion metric (520: r_lowest < 1e-2 and 50 <= k* <= 1000)
    promo_count = sum(1 for it in parsed_items.values() if it["r_lowest"] < 1e-2 and 50 <= it["k_star"] <= 1000)
    promo_ratio = round(promo_count / float(N_520), 6)

    promotion_evaluation = {
        "qualified_count": promo_count,
        "total": N_520,
        "ratio": promo_ratio,
        "threshold_target_count": 494,
        "threshold_target_ratio": round(494.0 / 520.0, 6),
        "delta_from_494": promo_count - 494,
    }

    # 3. G1 definitions (①-a & ①-b)
    combined_1b_count = promo_count + rec_80["rmin_lt_1e2_count"]
    ratio_1b_600 = round(combined_1b_count / float(N_600), 6)
    ratio_1b_520 = round(promo_count / float(N_520), 6)

    promotion_g1_definitions = {
        "definition_1b_combined600": {
            "description": "①-b: heldout 600 = new 520 r_k* (k*>=50) + existing 80 log r_min < 1e-2",
            "combined600_total": N_600,
            "combined600_count": combined_1b_count,
            "combined600_ratio": ratio_1b_600,
            "heldout520_subset_count": promo_count,
            "heldout520_subset_total": N_520,
            "heldout520_subset_ratio": ratio_1b_520,
            "heldout520_target_count": 468,
            "heldout520_target_ratio": 0.900000,
        },
        "definition_1a_train1000": {
            "description": "①-a: train 1,000 r_min = min(4*min rho, r_1000) < 1e-2 (PREREG v1.3 §10(c) 172행 문언, k* < 50 미도달 계산은 로그 궤적으로 동일 적용)",
            "train1000_total": N_1000,
            "train1000_rmin_lt_1e2_count": rec_train1000["rmin_lt_1e2_count"],
            "train1000_rmin_lt_1e2_ratio": rec_train1000["rmin_lt_1e2_ratio"],
            "kstar_lt_50_count": rec_train1000["kstar_lt_50_count"],
            "kstar_undetermined_count": rec_train1000["kstar_undetermined_count"],
            "kstar_ge_50_count": rec_train1000["kstar_ge_50_count"],
            "conservative_count": rec_train1000["conservative_count"],
            "conservative_ratio": rec_train1000["conservative_ratio"],
            "excl_undetermined_denom": rec_train1000["excl_undetermined_denom"],
            "excl_undetermined_count": rec_train1000["excl_undetermined_count"],
            "excl_undetermined_ratio": rec_train1000["excl_undetermined_ratio"],
            "governing_judgment_ratio": rec_train1000["conservative_ratio"],
            "train1000_source": rec_train1000["source"],
            "train1000_source_sha256": rec_train1000["source_sha256"],
        }
    }

    # 4. (d)1: k_star 구간 4개 ([0,50), [50,300), [300,700), [700,1000])
    bin_0_50 = sum(1 for it in parsed_items.values() if it["k_star"] < 50)
    bin_50_300 = sum(1 for it in parsed_items.values() if 50 <= it["k_star"] < 300)
    bin_300_700 = sum(1 for it in parsed_items.values() if 300 <= it["k_star"] < 700)
    bin_700_1000 = sum(1 for it in parsed_items.values() if 700 <= it["k_star"] <= 1000)
    total_invalid = sum(invalid_reasons.values())

    kstar_distribution = {
        "bin_0_50": {"count": bin_0_50, "ratio": round(bin_0_50 / float(N_520), 6)},
        "bin_50_300": {"count": bin_50_300, "ratio": round(bin_50_300 / float(N_520), 6)},
        "bin_300_700": {"count": bin_300_700, "ratio": round(bin_300_700 / float(N_520), 6)},
        "bin_700_1000": {"count": bin_700_1000, "ratio": round(bin_700_1000 / float(N_520), 6)},
        "invalid_not_reached": {"count": total_invalid, "ratio": round(total_invalid / float(N_520), 6)},
        "sum_check_520": bin_0_50 + bin_50_300 + bin_300_700 + bin_700_1000 + total_invalid,
    }

    # 5. (d)2: Top-3 set equality
    top3_matches = sum(1 for it in parsed_items.values() if set(it["top3_final"]) == set(it["top3_lowest"]))
    top3_evaluation = {
        "match_count": top3_matches,
        "total": N_520,
        "ratio": round(top3_matches / float(N_520), 6),
    }

    # 6. (d)3: Rescue items and trajectory shapes
    rescue_items = [it for it in parsed_items.values() if it["r_1000"] >= 1e-2 and it["r_lowest"] < 1e-2]
    rescue_kstar_lt_50 = sum(1 for it in rescue_items if it["k_star"] < 50)

    shape_crosstab: Dict[str, int] = {
        "unclassified": 0,
        "monotonic_decrease": 0,
        "increase": 0,
        "plateau": 0,
        "oscillation": 0,
    }

    rescue_window_records: List[Dict[str, Any]] = []
    for it in rescue_items:
        shape = classify_trajectory_shape_v4(it["rho_traj"], it["r_1000"])
        shape_crosstab[shape] = shape_crosstab.get(shape, 0) + 1
        w_max = compute_rescue_window_max(it["rho_traj"], it["r_1000"], it["k_star"])
        rescue_window_records.append({
            "qid": it["qid"],
            "k_star": it["k_star"],
            "r_1000": it["r_1000"],
            "r_lowest": it["r_lowest"],
            "shape": shape,
            "window_max_r": w_max,
        })

    rescue_evaluation = {
        "count": len(rescue_items),
        "total": N_520,
        "ratio": round(len(rescue_items) / float(N_520), 6),
        "kstar_lt_50_count": rescue_kstar_lt_50,
        "trajectory_shape_crosstab": shape_crosstab,
        "records": rescue_window_records,
    }

    # 7. (d)4: cos(z_lowest, z_final) distribution
    cos_vals: List[float] = []
    cos_nan_count = 0
    for it in parsed_items.values():
        zl = it["z_lowest"]
        zf = it["z_star"]
        if isinstance(zl, torch.Tensor) and isinstance(zf, torch.Tensor):
            zl_f = zl.detach().cpu().double().flatten()
            zf_f = zf.detach().cpu().double().flatten()
            norm_l = torch.norm(zl_f)
            norm_f = torch.norm(zf_f)
            if norm_l > 0 and norm_f > 0:
                c = float(torch.dot(zl_f, zf_f) / (norm_l * norm_f))
                if math.isnan(c) or math.isinf(c):
                    cos_nan_count += 1
                else:
                    cos_vals.append(c)
            else:
                cos_nan_count += 1
        else:
            cos_nan_count += 1

    cos_distribution: Optional[Dict[str, Any]] = None
    if cos_vals:
        sorted_cos = sorted(cos_vals)
        n_c = len(sorted_cos)

        def pct(p: float) -> float:
            idx = int(round(p * (n_c - 1)))
            return float(sorted_cos[idx])

        mean_c = sum(sorted_cos) / float(n_c)
        var_c = sum((x - mean_c) ** 2 for x in sorted_cos) / float(n_c)
        std_c = math.sqrt(var_c)

        cos_distribution = {
            "n_valid": n_c,
            "cos_nan_count": cos_nan_count,
            "min": float(sorted_cos[0]),
            "q25": pct(0.25),
            "median": pct(0.50),
            "q75": pct(0.75),
            "max": float(sorted_cos[-1]),
            "mean": float(mean_c),
            "std": float(std_c),
        }

    # 8. Discrimination among items (mean pairwise cosine)
    discrimination: Dict[str, Any] = {
        "nearest_neighbor_note": "최근접 일치율 및 다중 평형 판별은 §10(g)에 따라 p5_analyze_v5 시드 쌍 모드(GPU 1 시드 2030 vs GPU 2 시드 2031)에서 단독 산출함"
    }

    z_vectors = []
    for qid in sorted(parsed_items.keys()):
        z_t = parsed_items[qid]["z_star"]
        if isinstance(z_t, torch.Tensor):
            z_flat = z_t.detach().cpu().double().flatten()
            norm_z = torch.norm(z_flat)
            if norm_z > 0 and not (torch.isnan(norm_z) or torch.isinf(norm_z)):
                z_vectors.append(z_flat / norm_z)

    if len(z_vectors) >= 2:
        z_stack = torch.stack(z_vectors)
        cos_matrix = torch.matmul(z_stack, z_stack.T)
        n_v = len(z_vectors)
        triu_indices = torch.triu_indices(n_v, n_v, offset=1)
        pairwise_vals = cos_matrix[triu_indices[0], triu_indices[1]].tolist()

        if pairwise_vals:
            mean_pw = float(sum(pairwise_vals) / len(pairwise_vals))
            count_gt_095 = sum(1 for x in pairwise_vals if x > 0.95)
            discrimination["n_valid_items"] = n_v
            discrimination["total_pairs"] = len(pairwise_vals)
            discrimination["mean_pairwise_cos"] = round(mean_pw, 6)
            discrimination["pairs_gt_095_count"] = count_gt_095
            discrimination["pairs_gt_095_ratio"] = round(count_gt_095 / float(len(pairwise_vals)), 6)

    # Step 5: Summary object compilation
    summary = {
        "metadata": {
            "analyzer_name": "scripts/p5_promote_v5.py",
            "analyzer_sha256": compute_self_sha256(),
            "config_path": config_path,
            "config_sha256": config_sha,
            "heldout80_config_path": heldout80_config_path,
            "heldout80_config_sha256": h80_config_sha,
            "train1000_config_path": train1000_config_path,
            "train1000_config_sha256": t1000_config_sha,
            "heldout520_input_path": heldout520_path,
            "heldout520_input_sha256": heldout_sha,
            "heldout80_intermediate_path": heldout80_intermediate_path,
            "heldout80_intermediate_sha256": h80_intermediate_sha,
            "train1000_intermediate_path": train1000_intermediate_path,
            "train1000_intermediate_sha256": train1000_intermediate_sha,
            "extra_keys_in_520_pt": extra_keys_count,
            "alpha_collision_count": alpha_collision_count,
            "invalid_520_reasons": invalid_reasons,
            "cos_nan_count": cos_nan_count,
            "trajectory_rule_specification": TRAJECTORY_RULE_SPEC,
        },
        "primary_convergence": primary_convergence,
        "promotion_evaluation": promotion_evaluation,
        "promotion_g1_definitions": promotion_g1_definitions,
        "kstar_distribution_520": kstar_distribution,
        "top3_evaluation": top3_evaluation,
        "rescue_evaluation": rescue_evaluation,
        "cos_distribution": cos_distribution,
        "discrimination": discrimination,
    }

    return summary


def render_markdown(summary: Dict[str, Any]) -> str:
    """Render comprehensive markdown report strictly adhering to neutral quantitative tone."""
    meta = summary["metadata"]
    pc = summary["primary_convergence"]
    pr = summary["promotion_evaluation"]
    g1 = summary["promotion_g1_definitions"]["definition_1b_combined600"]
    g1a = summary["promotion_g1_definitions"]["definition_1a_train1000"]
    kd = summary["kstar_distribution_520"]
    t3 = summary["top3_evaluation"]
    res = summary["rescue_evaluation"]
    cos_d = summary["cos_distribution"]
    disc = summary["discrimination"]

    lines = [
        "# CTS S7 P5 결과 분석 및 승격 평가 보고서 v5 (TASK-0030 §3)",
        "",
        "## 1. 평가 개요 및 메타데이터",
        f"- 분석기: `{meta['analyzer_name']}` (SHA-256 `{meta['analyzer_sha256']}`)",
        f"- 520 정본 config: `{meta['config_path']}` (SHA-256 `{meta['config_sha256']}`)",
        f"- 80 정본 config: `{meta['heldout80_config_path']}` (SHA-256 `{meta['heldout80_config_sha256']}`)",
        f"- train 1,000 정본 config: `{meta['train1000_config_path']}` (SHA-256 `{meta['train1000_config_sha256']}`)",
        f"- 520 입력 파일: `{meta['heldout520_input_path']}` (SHA-256 `{meta['heldout520_input_sha256']}`)",
        f"- 80 중간 입력 파일: `{meta['heldout80_intermediate_path']}` (SHA-256 `{meta['heldout80_intermediate_sha256']}`)",
        f"- train 1,000 중간 입력 파일: `{meta['train1000_intermediate_path']}` (SHA-256 `{meta['train1000_intermediate_sha256']}`)",
        f"- 외래 키 건수: {meta['extra_keys_in_520_pt']} (타 alpha 충돌: {meta['alpha_collision_count']})",
        f"- 무효/결측 사유 집계 (520): {meta['invalid_520_reasons']}",
        "",
        "## 2. 1차 수렴 지표 집계",
        "| 집합 | 분모 | r_1000 < 1e-2 건수 (비율) | r_1000 < 5e-3 건수 (비율) |",
        "|---|---|---|---|",
        f"| 신규 520 | {pc['heldout520']['total']} | {pc['heldout520']['r1000_lt_1e2_count']} ({pc['heldout520']['r1000_lt_1e2_ratio']:.6f}) | {pc['heldout520']['r1000_lt_5e3_count']} ({pc['heldout520']['r1000_lt_5e3_ratio']:.6f}) |",
        f"| 기존 80 | {pc['existing80']['total']} | {pc['existing80']['r1000_lt_1e2_count']} ({pc['existing80']['r1000_lt_1e2_ratio']:.6f}) | {pc['existing80']['r1000_lt_5e3_count']} ({pc['existing80']['r1000_lt_5e3_ratio']:.6f}) |",
        f"| 통합 600 | {pc['combined600']['total']} | {pc['combined600']['r1000_lt_1e2_count']} ({pc['combined600']['r1000_lt_1e2_ratio']:.6f}) | {pc['combined600']['r1000_lt_5e3_count']} ({pc['combined600']['r1000_lt_5e3_ratio']:.6f}) |",
        f"| train 1000 | {pc['train1000']['total']} | {pc['train1000']['r1000_lt_1e2_count']} ({pc['train1000']['r1000_lt_1e2_ratio']:.6f}) | {pc['train1000']['r1000_lt_5e3_count']} ({pc['train1000']['r1000_lt_5e3_ratio']:.6f}) |",
        "",
        "## 3. 2차 승격 지표 집계 (신규 520)",
        "- 조건: r_lowest < 1e-2 ∧ 50 ≤ k* ≤ 1000",
        f"- 적격 건수: {pr['qualified_count']} / {pr['total']}",
        f"- 산출 비율: {pr['ratio']:.6f} (기준치: {pr['threshold_target_ratio']:.6f}, 494건)",
        f"- 494건 대비 편차: {pr['delta_from_494']:+d}건",
        "",
        "## 4. 승격 G1 정의 집계",
        "### ①-a: train 1,000 집계",
        f"- 기준 문언: PREREG_S7_DEQ_v1.3 §10(c) 172행 (k* < 50 미도달 계산은 로그 궤적으로 동일 적용)",
        f"- r_min < 1e-2 건수 (비율): {g1a['train1000_rmin_lt_1e2_count']} / {g1a['train1000_total']} ({g1a['train1000_rmin_lt_1e2_ratio']:.6f})",
        f"- k* 분포: k* < 50: {g1a['kstar_lt_50_count']}건, 판별 불가(null): {g1a['kstar_undetermined_count']}건, k* >= 50: {g1a['kstar_ge_50_count']}건",
        f"- 보수적 판정 (판별 불가 미도달 산입): {g1a['conservative_count']} / {g1a['train1000_total']} ({g1a['conservative_ratio']:.6f})",
        f"- 판별 불가 제외 산출: {g1a['excl_undetermined_count']} / {g1a['excl_undetermined_denom']} ({g1a['excl_undetermined_ratio']:.6f})",
        f"- 공식 판정값 (보수적 기준): {g1a['governing_judgment_ratio']:.6f}",
        f"- 출처: `{g1a['train1000_source']}` (SHA-256 `{g1a['train1000_source_sha256']}`)",
        "",
        "### ①-b: 통합 600 및 신규 520 부분집합 집계",
        f"- 통합 600: {g1['combined600_count']} / {g1['combined600_total']} ({g1['combined600_ratio']:.6f})",
        f"- 신규 520 부분집합: {g1['heldout520_subset_count']} / {g1['heldout520_subset_total']} ({g1['heldout520_subset_ratio']:.6f}) (기준치: {g1['heldout520_target_count']}건 / 0.900000)",
        f"- 참고: v1.3 §10(c) 173행 문언에 따라 기존 80에는 k* 규칙 없음",
        "",
        "## 5. k* 분포 집계 (분모 520 고정)",
        "| 구간 | 건수 | 비율 |",
        "|---|---|---|",
        f"| [0, 50) | {kd['bin_0_50']['count']} | {kd['bin_0_50']['ratio']:.6f} |",
        f"| [50, 300) | {kd['bin_50_300']['count']} | {kd['bin_50_300']['ratio']:.6f} |",
        f"| [300, 700) | {kd['bin_300_700']['count']} | {kd['bin_300_700']['ratio']:.6f} |",
        f"| [700, 1000] | {kd['bin_700_1000']['count']} | {kd['bin_700_1000']['ratio']:.6f} |",
        f"| 미도달/무효 | {kd['invalid_not_reached']['count']} | {kd['invalid_not_reached']['ratio']:.6f} |",
        f"| 합계 | {kd['sum_check_520']} | 1.000000 |",
        "",
        "## 6. Top-3 집합 일치 통계",
        f"- 일치 건수: {t3['match_count']} / {t3['total']} ({t3['ratio']:.6f})",
        "",
        "## 7. 구제 문항 및 궤적 모양 분류",
        f"- 구제 대상 건수: {res['count']} / {res['total']} ({res['ratio']:.6f})",
        f"- 구제 문항 중 k* < 50 건수: {res['kstar_lt_50_count']}건",
        "- 궤적 분류 기준: 마지막 500걸음(500..1000) 대상 unclassified / monotonic_decrease / increase / plateau / oscillation",
        "| 궤적 모양 | 건수 |",
        "|---|---|",
        f"| monotonic_decrease | {res['trajectory_shape_crosstab'].get('monotonic_decrease', 0)} |",
        f"| increase | {res['trajectory_shape_crosstab'].get('increase', 0)} |",
        f"| plateau | {res['trajectory_shape_crosstab'].get('plateau', 0)} |",
        f"| oscillation | {res['trajectory_shape_crosstab'].get('oscillation', 0)} |",
        f"| unclassified | {res['trajectory_shape_crosstab'].get('unclassified', 0)} |",
        "",
        "## 8. z 벡터 분포 및 변별 통계",
    ]

    if cos_d:
        lines.extend([
            "### cos(z_lowest, z_final) 분포 분위수",
            f"- min: {cos_d['min']:.6f}, q25: {cos_d['q25']:.6f}, median: {cos_d['median']:.6f}, q75: {cos_d['q75']:.6f}, max: {cos_d['max']:.6f}",
            f"- mean: {cos_d['mean']:.6f}, std: {cos_d['std']:.6f}",
            f"- cos NaN 건수: {cos_d.get('cos_nan_count', 0)}",
            "",
        ])
    else:
        lines.append("- cos(z_lowest, z_final) 분포: 산출 대상 없음 (null)\n")

    if "mean_pairwise_cos" in disc:
        lines.extend([
            "### 문항 간 변별 통계",
            f"- 유효 문항 수: {disc['n_valid_items']}",
            f"- 문항 간 평균 cosine: {disc['mean_pairwise_cos']:.6f}",
            f"- cos > 0.95 쌍 건수 (비율): {disc['pairs_gt_095_count']} / {disc['total_pairs']} ({disc['pairs_gt_095_ratio']:.6f})",
            f"- 안내문: {disc['nearest_neighbor_note']}",
        ])

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="P5 Result Analyzer & Promotion Evaluator v5 (TASK-0030 §3)")
    parser.add_argument("--heldout520", type=str, required=True, help="Path to heldout 520 z* pt file")
    parser.add_argument("--heldout80", "--r1000_p1", dest="heldout80", type=str, required=True, help="Path to heldout 80 intermediate JSON v2")
    parser.add_argument("--train1000", type=str, required=True, help="Path to train 1000 intermediate JSON v2")
    parser.add_argument("--config", type=str, default="configs/s7_new_heldout_520.json", help="Path to canonical 520 config")
    parser.add_argument("--heldout80_config", type=str, default="configs/s5_heldout_80.json", help="Path to canonical 80 heldout config")
    parser.add_argument("--train1000_config", type=str, default="configs/s7_train_1000.json", help="Path to canonical train 1000 config")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--out_md", type=str, default=None, help="Output Markdown path")
    args = parser.parse_args()

    summary = analyze_p5_promote_v5(
        heldout520_path=args.heldout520,
        heldout80_intermediate_path=args.heldout80,
        train1000_intermediate_path=args.train1000,
        config_path=args.config,
        heldout80_config_path=args.heldout80_config,
        train1000_config_path=args.train1000_config,
    )

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[OUTPUT] JSON saved to {args.out_json}")

    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        md_text = render_markdown(summary)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"[OUTPUT] Markdown saved to {args.out_md}")

    # Output short stdout line
    pc = summary["primary_convergence"]
    pr = summary["promotion_evaluation"]
    g1a = summary["promotion_g1_definitions"]["definition_1a_train1000"]
    print(
        f"[SUMMARY] 520 r1000<1e-2: {pc['heldout520']['r1000_lt_1e2_count']}/{pc['heldout520']['total']} "
        f"| Promo: {pr['qualified_count']}/{pr['total']} (delta={pr['delta_from_494']:+d}) "
        f"| 600: {pc['combined600']['r1000_lt_1e2_count']}/{pc['combined600']['total']} "
        f"| train1000 conservative: {g1a['conservative_count']}/{g1a['train1000_total']} ({g1a['conservative_ratio']:.6f})"
    )


if __name__ == "__main__":
    main()

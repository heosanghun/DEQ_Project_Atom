#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_promote_v2.py - CTS S7 P5 Result Analyzer & Promotion Evaluator v2
Directive: TASK-0023 / AUDIT-P006 (FIX) / PREREG_S7_DEQ_v1.3 §10(b)(c)(d) / GOAL.md G1

Operational Standards & Audit Compliance:
  1. Config integrity: Enforces SHA-256 == 48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988 (520 unique IDs).
  2. Strict key queries: Looks up exact keys f"q{idx:04d}_a0.25" for the 520 canonical IDs only.
  3. Fixed denominators: All rates are computed against fixed denominators (520, 600, 1000). Missing/invalid items are counted as non-converged.
  4. Top-3 set equality: Evaluated via set(top3_lowest) == set(top3_final).
  5. Rescue window: Evaluates max r in k* +- 10; includes r_1000 when k* + 10 >= 1000.
  6. Trajectory classification: Implements 4-shape taxonomy (monotonic_decrease, increase, plateau, oscillation) on steps 500..1000.
  7. Discrimination: Evaluates float64 pairwise cosine and ratio of pairs > 0.95. Nearest neighbor is deferred to §10(g) p5_analyze_v5.
  8. Neutral quantitative reporting: Strictly zero forbidden qualitative judgment words.
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

TRAJECTORY_RULE_SPEC = (
    "Rules for trajectory shape classification on the last 500 steps (steps 500..1000) of r_k = rho_k / alpha (alpha=0.25): "
    "1. monotonic_decrease: fraction of steps where r_k > r_{k-1} is < 0.05 (< 5%). "
    "2. increase: r_1000 > 2.0 * r_500. "
    "3. plateau: otherwise, max(r_{500..1000}) / min(r_{500..1000}) < 1.5. "
    "4. oscillation: remainder."
)


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_quantiles(vals: List[float]) -> Optional[Dict[str, float]]:
    """Compute distribution statistics for a list of float values (float64)."""
    valid = [float(v) for v in vals if not (math.isnan(v) or math.isinf(v))]
    if not valid:
        return None
    valid.sort()
    n = len(valid)

    def get_pct(p: float) -> float:
        idx = p * (n - 1)
        i = int(math.floor(idx))
        rem = idx - i
        if i + 1 < n:
            return valid[i] * (1.0 - rem) + valid[i + 1] * rem
        return valid[i]

    mean_val = float(sum(valid) / n)
    var_val = float(sum((x - mean_val) ** 2 for x in valid) / n) if n > 0 else 0.0
    std_val = math.sqrt(var_val)

    return {
        "min": round(float(valid[0]), 6),
        "q25": round(get_pct(0.25), 6),
        "median": round(get_pct(0.50), 6),
        "q75": round(get_pct(0.75), 6),
        "max": round(float(valid[-1]), 6),
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
    }


def classify_trajectory_shape(rho_traj: List[float], r_1000: float, alpha: float = 0.25) -> str:
    """Classify the trajectory shape of the last 500 steps (steps 500..1000)."""
    if not rho_traj:
        return "oscillation"

    # Convert rho_traj to r_k = rho_k / alpha
    r_traj = [float(rho) / alpha for rho in rho_traj]

    # If trajectory has at least 501 elements (k=0..500..999)
    if len(r_traj) >= 501:
        # Steps 500 to end
        sub_r = r_traj[500:] + [float(r_1000)]
        r_500 = r_traj[500]
    elif len(r_traj) >= 2:
        sub_r = r_traj + [float(r_1000)]
        r_500 = r_traj[0]
    else:
        return "oscillation"

    # 1. monotonic_decrease: fraction of increases < 5%
    n_steps = len(sub_r) - 1
    if n_steps > 0:
        inc_count = sum(1 for i in range(1, len(sub_r)) if sub_r[i] > sub_r[i - 1])
        if (inc_count / n_steps) < 0.05 and float(r_1000) <= r_500:
            return "monotonic_decrease"

    # 2. increase: r_1000 > 2.0 * r_500
    if r_500 > 0 and float(r_1000) > 2.0 * r_500:
        return "increase"

    # 3. plateau: max / min < 1.5
    min_val = min(sub_r)
    max_val = max(sub_r)
    if min_val > 0 and (max_val / min_val) < 1.5:
        return "plateau"

    # 4. oscillation: remainder
    return "oscillation"


def load_canonical_config(config_path: str) -> Tuple[List[int], str]:
    """Load and verify the canonical 520 held-out items configuration."""
    if not os.path.exists(config_path):
        sys.stderr.write(f"[ERROR] Config file missing: {config_path}\n")
        sys.exit(1)

    actual_sha = compute_sha256(config_path)
    if actual_sha != EXPECTED_CONFIG_SHA256:
        sys.stderr.write(
            f"[ERROR] Canonical config SHA mismatch for {config_path}!\n"
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


def load_existing80_data(p1_json_path: str) -> Tuple[Dict[str, Any], str]:
    """Load existing 80 heldout items from P1 re-evaluation records.
    Strictly exits with error if the file is missing, ill-formed, or count != 80.
    """
    if not os.path.exists(p1_json_path):
        sys.stderr.write(f"[ERROR] Existing 80 P1 re-eval file missing: {p1_json_path}\n")
        sys.exit(1)

    actual_sha = compute_sha256(p1_json_path)

    try:
        with open(p1_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to parse P1 JSON {p1_json_path}: {e}\n")
        sys.exit(1)

    if not isinstance(data, dict) or "heldout_80" not in data:
        sys.stderr.write(f"[ERROR] P1 JSON format mismatch: key 'heldout_80' not found in {p1_json_path}\n")
        sys.exit(1)

    h80_list = data["heldout_80"]
    if not isinstance(h80_list, list) or len(h80_list) != 80:
        sys.stderr.write(f"[ERROR] P1 heldout_80 must contain exactly 80 items (got {len(h80_list) if isinstance(h80_list, list) else type(h80_list)})\n")
        sys.exit(1)

    r1000_vals: List[float] = []
    rmin_vals: List[float] = []
    for idx, entry in enumerate(h80_list):
        if not isinstance(entry, dict) or "r_1000" not in entry:
            sys.stderr.write(f"[ERROR] Entry {idx} in heldout_80 missing 'r_1000'\n")
            sys.exit(1)
        r_val = float(entry["r_1000"])
        r1000_vals.append(r_val)

        # r_min: either from r_min field or fall back to r_1000
        if "r_min" in entry:
            rmin_vals.append(float(entry["r_min"]))
        else:
            rmin_vals.append(r_val)

    lt_1e2 = sum(1 for r in r1000_vals if r < 1e-2)
    lt_5e3 = sum(1 for r in r1000_vals if r < 5e-3)
    rmin_lt_1e2 = sum(1 for r in rmin_vals if r < 1e-2)

    stats = {
        "n": 80,
        "source": p1_json_path,
        "source_sha256": actual_sha,
        "r1000_values": r1000_vals,
        "rmin_values": rmin_vals,
        "r_lt_1e2_count": lt_1e2,
        "r_lt_1e2_ratio": round(lt_1e2 / 80.0, 6),
        "r_lt_5e3_count": lt_5e3,
        "r_lt_5e3_ratio": round(lt_5e3 / 80.0, 6),
        "rmin_lt_1e2_count": rmin_lt_1e2,
        "rmin_lt_1e2_ratio": round(rmin_lt_1e2 / 80.0, 6),
    }
    return stats, actual_sha


def analyze_p5_promote_v2(
    heldout520_path: str,
    r1000_p1_path: str,
    config_path: str,
    train1000_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete statistical analysis of P5 results conforming to AUDIT-P006 FIX."""
    if not os.path.exists(heldout520_path):
        sys.stderr.write(f"[ERROR] Held-out 520 file missing: {heldout520_path}\n")
        sys.exit(1)

    heldout_sha = compute_sha256(heldout520_path)
    canonical_ids, config_sha = load_canonical_config(config_path)
    rec_80, p1_sha = load_existing80_data(r1000_p1_path)

    # Load raw data dictionary
    raw_pt = torch.load(heldout520_path, map_location="cpu")
    if not isinstance(raw_pt, dict):
        sys.stderr.write(f"[ERROR] Held-out 520 file must contain a dict (got {type(raw_pt)})\n")
        sys.exit(1)

    # Detect duplicate / extra keys
    raw_keys = list(raw_pt.keys())
    canonical_id_set = set(canonical_ids)
    seen_canonical_keys: Set[str] = set()
    extra_keys_count = 0
    alpha_collision_count = 0

    import re
    alpha_pattern = re.compile(r"^q(\d{4})_a0\.25$")

    for k in raw_keys:
        k_str = str(k)
        m = alpha_pattern.match(k_str)
        if m:
            qid_val = int(m.group(1))
            if qid_val in canonical_id_set:
                if k_str in seen_canonical_keys:
                    sys.stderr.write(f"[ERROR] Duplicate key detected in tensor dictionary: {k_str}\n")
                    sys.exit(1)
                seen_canonical_keys.add(k_str)
            else:
                extra_keys_count += 1
        else:
            extra_keys_count += 1
            if "_a0." in k_str and not k_str.endswith("_a0.25"):
                alpha_collision_count += 1

    if extra_keys_count > 0:
        sys.stderr.write(
            f"[INFO] Ignored {extra_keys_count} keys not matching canonical pattern ^q\\d{{4}}_a0\\.25 "
            f"(non-a0.25 count: {alpha_collision_count})\n"
        )

    # Parse and validate exactly the 520 canonical items
    parsed_items: Dict[int, Dict[str, Any]] = {}
    invalid_reasons: Dict[str, int] = {
        "missing_item": 0,
        "missing_field": 0,
        "nan_inf_r": 0,
        "invalid_kstar": 0,
    }

    required_fields = ["r_1000", "r_lowest", "k_star", "rho_traj", "top3_final", "top3_lowest", "z_star", "z_lowest"]

    for qid in canonical_ids:
        key = f"q{qid:04d}_a0.25"
        if key not in raw_pt:
            invalid_reasons["missing_item"] += 1
            continue

        item_raw = raw_pt[key]
        if not isinstance(item_raw, dict):
            invalid_reasons["missing_field"] += 1
            continue

        # Check required fields
        has_all_fields = True
        for rf in required_fields:
            if rf not in item_raw:
                has_all_fields = False
                break
        if not has_all_fields:
            invalid_reasons["missing_field"] += 1
            continue

        # Validate r_1000 and r_lowest
        try:
            r1000_f = float(item_raw["r_1000"])
            rlowest_f = float(item_raw["r_lowest"])
            if math.isnan(r1000_f) or math.isinf(r1000_f) or math.isnan(rlowest_f) or math.isinf(rlowest_f):
                invalid_reasons["nan_inf_r"] += 1
                continue
        except (ValueError, TypeError):
            invalid_reasons["nan_inf_r"] += 1
            continue

        # Validate k_star
        try:
            k_st_val = float(item_raw["k_star"])
            if math.isnan(k_st_val) or math.isinf(k_st_val):
                invalid_reasons["invalid_kstar"] += 1
                continue
            k_st = int(k_st_val)
            if k_st < 0 or k_st > 1000:
                invalid_reasons["invalid_kstar"] += 1
                continue
        except (ValueError, TypeError):
            invalid_reasons["invalid_kstar"] += 1
            continue

        # Valid item parsed
        parsed_items[qid] = {
            "qid": qid,
            "r_1000": r1000_f,
            "r_lowest": rlowest_f,
            "k_star": k_st,
            "rho_traj": [float(x) for x in item_raw.get("rho_traj", [])],
            "top3_final": [int(x) for x in item_raw.get("top3_final", [])],
            "top3_lowest": [int(x) for x in item_raw.get("top3_lowest", [])],
            "z_star": item_raw["z_star"].double() if isinstance(item_raw["z_star"], torch.Tensor) else None,
            "z_lowest": item_raw["z_lowest"].double() if isinstance(item_raw["z_lowest"], torch.Tensor) else None,
        }

    # Denominators are strictly fixed
    N_520 = 520
    N_80 = 80
    N_600 = 600

    # 1. 1차 (최종점 r_1000): 520 및 600
    r1000_lt_1e2_520 = sum(1 for it in parsed_items.values() if it["r_1000"] < 1e-2)
    r1000_lt_5e3_520 = sum(1 for it in parsed_items.values() if it["r_1000"] < 5e-3)

    r1000_lt_1e2_80 = rec_80["r_lt_1e2_count"]
    r1000_lt_5e3_80 = rec_80["r_lt_5e3_count"]

    r1000_lt_1e2_600 = r1000_lt_1e2_520 + r1000_lt_1e2_80
    r1000_lt_5e3_600 = r1000_lt_5e3_520 + r1000_lt_5e3_80

    primary_convergence = {
        "heldout520": {
            "total": N_520,
            "valid_count": len(parsed_items),
            "r1000_lt_1e2_count": r1000_lt_1e2_520,
            "r1000_lt_1e2_ratio": round(r1000_lt_1e2_520 / float(N_520), 6),
            "r1000_lt_5e3_count": r1000_lt_5e3_520,
            "r1000_lt_5e3_ratio": round(r1000_lt_5e3_520 / float(N_520), 6),
        },
        "existing80": {
            "total": N_80,
            "source": rec_80["source"],
            "source_sha256": rec_80["source_sha256"],
            "r1000_lt_1e2_count": r1000_lt_1e2_80,
            "r1000_lt_1e2_ratio": round(r1000_lt_1e2_80 / float(N_80), 6),
            "r1000_lt_5e3_count": r1000_lt_5e3_80,
            "r1000_lt_5e3_ratio": round(r1000_lt_5e3_80 / float(N_80), 6),
        },
        "combined600": {
            "total": N_600,
            "r1000_lt_1e2_count": r1000_lt_1e2_600,
            "r1000_lt_1e2_ratio": round(r1000_lt_1e2_600 / float(N_600), 6),
            "r1000_lt_5e3_count": r1000_lt_5e3_600,
            "r1000_lt_5e3_ratio": round(r1000_lt_5e3_600 / float(N_600), 6),
        }
    }

    # 2. 2차 승격 지표: 520 중 (r_lowest < 1e-2 and 50 <= k_star <= 1000)
    promo_count = 0
    for it in parsed_items.values():
        if it["r_lowest"] < 1e-2 and 50 <= it["k_star"] <= 1000:
            promo_count += 1

    promo_ratio = round(promo_count / float(N_520), 6)
    promo_threshold_count = 494  # ceil(0.95 * 520) = 494
    promo_diff = promo_count - promo_threshold_count

    promotion_stats = {
        "qualified_count": promo_count,
        "total": N_520,
        "ratio": promo_ratio,
        "threshold_target_count": promo_threshold_count,
        "threshold_target_ratio": 0.950000,
        "delta_from_494": promo_diff,
    }

    # 3. 승격 시 G1 정의 (①-a 및 ①-b)
    # ①-b: 600 = 신규 520 r_k* (k* < 50 미도달) + 기존 80 log r_min (< 1e-2)
    # 신규 520의 r_k* < 1e-2 (k* >= 50) is promo_count
    # 기존 80의 r_min < 1e-2 count is rec_80["rmin_lt_1e2_count"]
    count_1b_600 = promo_count + rec_80["rmin_lt_1e2_count"]
    ratio_1b_600 = round(count_1b_600 / float(N_600), 6)
    ratio_1b_520 = round(promo_count / float(N_520), 6)

    g1_definitions = {
        "definition_1b_600": {
            "description": "①-b: 600 = 신규 520 r_k*(k*<50 미도달) + 기존 80 로그 r_min의 <1e-2 집계",
            "combined600_count": count_1b_600,
            "combined600_total": N_600,
            "combined600_ratio": ratio_1b_600,
            "heldout520_subset_count": promo_count,
            "heldout520_subset_total": N_520,
            "heldout520_subset_ratio": ratio_1b_520,
            "heldout520_target_count": 468,  # 90% of 520
            "heldout520_target_ratio": 0.900000,
        },
        "definition_1a_train1000": {
            "description": "①-a: train 1,000 r_min = min(4*min rho, r_1000) < 1e-2 집계",
            "status": "deferred_or_external",
            "train1000_file": train1000_path if train1000_path else "None",
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

    # 5. (d)2: top-3 집합 일치 비율 (set equality)
    top3_set_match_count = 0
    for it in parsed_items.values():
        if set(it["top3_lowest"]) == set(it["top3_final"]):
            top3_set_match_count += 1

    top3_match_stats = {
        "match_count": top3_set_match_count,
        "total": N_520,
        "ratio": round(top3_set_match_count / float(N_520), 6),
    }

    # 6. (d)3: 구제 문항 (r_1000 >= 1e-2 and r_lowest < 1e-2 and 50 <= k_star <= 1000)
    alpha = 0.25
    rescue_items_list: List[Dict[str, Any]] = []
    shape_crosstab: Dict[str, int] = {
        "monotonic_decrease": 0,
        "increase": 0,
        "plateau": 0,
        "oscillation": 0,
    }

    for it in parsed_items.values():
        r_f = it["r_1000"]
        r_l = it["r_lowest"]
        k_st = it["k_star"]
        if r_f >= 1e-2 and r_l < 1e-2 and 50 <= k_st <= 1000:
            rho_tr = it["rho_traj"]
            r_tr = [rho / alpha for rho in rho_tr]

            # Window k* +- 10
            w_start = max(0, k_st - 10)
            w_end = min(len(r_tr), k_st + 11)
            window_slice = list(r_tr[w_start:w_end]) if r_tr else []

            # CRUCIAL FIX (AUDIT-P006 #12): If k* + 10 >= 1000, include r_1000!
            if k_st + 10 >= 1000:
                window_slice.append(float(r_f))

            max_r_in_window = float(max(window_slice)) if window_slice else float(r_l)

            # Classify trajectory shape
            shape = classify_trajectory_shape(rho_tr, r_f, alpha)
            shape_crosstab[shape] += 1

            rescue_items_list.append({
                "qid": it["qid"],
                "k_star": k_st,
                "r_lowest": round(r_l, 6),
                "r_1000": round(r_f, 6),
                "kstar_pm10_max_r": round(max_r_in_window, 6),
                "trajectory_shape": shape,
            })

    rescue_stats = {
        "count": len(rescue_items_list),
        "total": N_520,
        "ratio": round(len(rescue_items_list) / float(N_520), 6),
        "trajectory_shape_crosstab": shape_crosstab,
        "items": rescue_items_list,
    }

    # 7. (d)4: cos(z_lowest, z_final) 분포 (float64)
    cos_low_fin_list: List[float] = []
    cos_nan_count = 0
    for it in parsed_items.values():
        zl = it["z_lowest"]
        zf = it["z_star"]
        if zl is not None and zf is not None:
            fl_zl = zl.flatten()
            fl_zf = zf.flatten()
            if torch.isnan(fl_zl).any() or torch.isnan(fl_zf).any():
                cos_nan_count += 1
                continue
            norm_zl = torch.norm(fl_zl, p=2)
            norm_zf = torch.norm(fl_zf, p=2)
            if norm_zl > 1e-12 and norm_zf > 1e-12:
                cos_val = float(torch.dot(fl_zl, fl_zf) / (norm_zl * norm_zf))
                if not (math.isnan(cos_val) or math.isinf(cos_val)):
                    cos_low_fin_list.append(cos_val)
                else:
                    cos_nan_count += 1
            else:
                cos_nan_count += 1

    cos_lowest_final_distribution = compute_quantiles(cos_low_fin_list)

    # 8. 변별 (보고만): 520 z_star 문항 간 평균 cos, cos > 0.95 쌍 비율 (float64)
    valid_z_tensors: List[torch.Tensor] = []
    z_nan_count = 0
    for it in parsed_items.values():
        z_t = it["z_star"]
        if z_t is not None:
            fl_z = z_t.flatten()
            if not torch.isnan(fl_z).any() and not torch.isinf(fl_z).any():
                valid_z_tensors.append(fl_z)
            else:
                z_nan_count += 1

    discrimination_stats: Dict[str, Any] = {
        "n_valid_items": len(valid_z_tensors),
        "z_nan_or_missing_count": N_520 - len(valid_z_tensors),
    }

    if len(valid_z_tensors) > 1:
        Z = torch.stack(valid_z_tensors, dim=0)  # [N, D] float64
        norms = torch.norm(Z, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        Z_norm = Z / norms
        cos_matrix = (Z_norm @ Z_norm.T)

        N_valid = len(valid_z_tensors)
        triu_idx = torch.triu_indices(N_valid, N_valid, offset=1)
        pairwise_cos = cos_matrix[triu_idx[0], triu_idx[1]]

        mean_pw_cos = float(pairwise_cos.mean())
        pairs_gt_095 = int((pairwise_cos > 0.95).sum())
        tot_pairs = len(pairwise_cos)
        ratio_gt_095 = float(pairs_gt_095 / tot_pairs) if tot_pairs else 0.0

        discrimination_stats.update({
            "mean_pairwise_cos": round(mean_pw_cos, 6),
            "total_pairs": tot_pairs,
            "pairs_gt_095_count": pairs_gt_095,
            "pairs_gt_095_ratio": round(ratio_gt_095, 6),
            "nearest_neighbor_note": "최근접 일치율·다중 평형 = §10(g), p5_analyze_v5 시드 쌍 담당 (단일 스크립트 산출 0)",
        })

    # Assemble summary
    summary: Dict[str, Any] = {
        "metadata": {
            "script_name": "scripts/p5_promote_v2.py",
            "heldout520_file": heldout520_path,
            "heldout520_sha256": heldout_sha,
            "existing80_file": r1000_p1_path,
            "existing80_sha256": p1_sha,
            "canonical_config_file": config_path,
            "canonical_config_sha256": config_sha,
            "n_canonical_items": N_520,
            "n_existing80_items": N_80,
            "n_combined_items": N_600,
            "ignored_extra_keys_count": extra_keys_count,
            "alpha_collision_keys_count": alpha_collision_count,
            "invalid_items_count": total_invalid,
            "invalid_breakdown": invalid_reasons,
            "trajectory_rule": TRAJECTORY_RULE_SPEC,
        },
        "primary_convergence": primary_convergence,
        "promotion_evaluation": promotion_stats,
        "promotion_g1_definitions": g1_definitions,
        "kstar_distribution": kstar_distribution,
        "top3_match": top3_match_stats,
        "rescue_items": rescue_stats,
        "cos_lowest_final_distribution": cos_lowest_final_distribution,
        "discrimination": discrimination_stats,
    }
    return summary


def render_markdown(summary: Dict[str, Any]) -> str:
    """Render neutral quantitative markdown report with zero forbidden judgment words."""
    m = summary["metadata"]
    pc = summary["primary_convergence"]
    pr = summary["promotion_evaluation"]
    g1 = summary["promotion_g1_definitions"]["definition_1b_600"]
    kd = summary["kstar_distribution"]
    t3 = summary["top3_match"]
    res = summary["rescue_items"]
    cos_d = summary["cos_lowest_final_distribution"]
    disc = summary["discrimination"]

    lines: List[str] = [
        "# CTS S7 P5 수렴 및 승격 통계 집계표 (v2)",
        "",
        "## 1. 메타데이터 및 입력 출처",
        f"- 대상 520 파일: `{m['heldout520_file']}` (SHA-256 `{m['heldout520_sha256']}`)",
        f"- 기존 80 파일: `{m['existing80_file']}` (SHA-256 `{m['existing80_sha256']}`)",
        f"- 정본 config: `{m['canonical_config_file']}` (SHA-256 `{m['canonical_config_sha256']}`, 520개)",
        f"- 분석 스크립트: `{m['script_name']}`",
        f"- 제외된 외래 키 개수: {m['ignored_extra_keys_count']}건 (타 알파: {m['alpha_collision_keys_count']}건)",
        f"- 미도달/결측 집계: 총 {m['invalid_items_count']}건 (세부: {m['invalid_breakdown']})",
        "",
        "## 2. 1차 수렴 지표 (최종점 r_1000)",
        "| 집합 | 분모 | r_1000 < 1e-2 건수 (비율) | r_1000 < 5e-3 건수 (비율) |",
        "|---|---|---|---|",
        f"| 신규 520 | {pc['heldout520']['total']} | {pc['heldout520']['r1000_lt_1e2_count']} ({pc['heldout520']['r1000_lt_1e2_ratio']:.6f}) | {pc['heldout520']['r1000_lt_5e3_count']} ({pc['heldout520']['r1000_lt_5e3_ratio']:.6f}) |",
        f"| 기존 80 | {pc['existing80']['total']} | {pc['existing80']['r1000_lt_1e2_count']} ({pc['existing80']['r1000_lt_1e2_ratio']:.6f}) | {pc['existing80']['r1000_lt_5e3_count']} ({pc['existing80']['r1000_lt_5e3_ratio']:.6f}) |",
        f"| 통합 600 | {pc['combined600']['total']} | {pc['combined600']['r1000_lt_1e2_count']} ({pc['combined600']['r1000_lt_1e2_ratio']:.6f}) | {pc['combined600']['r1000_lt_5e3_count']} ({pc['combined600']['r1000_lt_5e3_ratio']:.6f}) |",
        "",
        "## 3. 2차 승격 지표 집계",
        "- 조건: r_lowest < 1e-2 ∧ 50 ≤ k* ≤ 1000",
        f"- 적격 건수: {pr['qualified_count']} / {pr['total']}",
        f"- 산출 비율: {pr['ratio']:.6f} (기준치: {pr['threshold_target_ratio']:.6f}, 494건)",
        f"- 494건 대비 편차: {pr['delta_from_494']:+d}건",
        "",
        "## 4. 승격 G1 정의 집계",
        "### ①-b: 통합 600 및 신규 520 부분집합 집계",
        f"- 통합 600: {g1['combined600_count']} / {g1['combined600_total']} ({g1['combined600_ratio']:.6f})",
        f"- 신규 520 부분집합: {g1['heldout520_subset_count']} / {g1['heldout520_subset_total']} ({g1['heldout520_subset_ratio']:.6f}) (기준치: {g1['heldout520_target_count']}건 / 0.900000)",
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
        "- 궤적 분류 기준: 마지막 500걸음(500..1000) 대상 monotonic_decrease / increase / plateau / oscillation",
        "| 궤적 모양 | 건수 |",
        "|---|---|",
        f"| monotonic_decrease | {res['trajectory_shape_crosstab']['monotonic_decrease']} |",
        f"| increase | {res['trajectory_shape_crosstab']['increase']} |",
        f"| plateau | {res['trajectory_shape_crosstab']['plateau']} |",
        f"| oscillation | {res['trajectory_shape_crosstab']['oscillation']} |",
        "",
        "## 8. z 벡터 분포 및 변별 통계",
    ]

    if cos_d:
        lines.extend([
            "### cos(z_lowest, z_final) 분포 분위수",
            f"- min: {cos_d['min']:.6f}, q25: {cos_d['q25']:.6f}, median: {cos_d['median']:.6f}, q75: {cos_d['q75']:.6f}, max: {cos_d['max']:.6f}",
            f"- mean: {cos_d['mean']:.6f}, std: {cos_d['std']:.6f}",
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
    parser = argparse.ArgumentParser(description="P5 Result Analyzer & Promotion Evaluator v2 (AUDIT-P006 FIX)")
    parser.add_argument("--heldout520", type=str, required=True, help="Path to heldout 520 z* pt file")
    parser.add_argument("--r1000_p1", type=str, required=True, help="Path to existing 80 P1 JSON file")
    parser.add_argument("--config", type=str, default="scratch/server_s7_new_heldout_520.json", help="Path to canonical 520 config")
    parser.add_argument("--train1000", type=str, default=None, help="Optional train 1000 z* file")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--out_md", type=str, default=None, help="Output Markdown path")
    args = parser.parse_args()

    summary = analyze_p5_promote_v2(
        heldout520_path=args.heldout520,
        r1000_p1_path=args.r1000_p1,
        config_path=args.config,
        train1000_path=args.train1000,
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
    print(
        f"[SUMMARY] 520 r1000<1e-2: {pc['heldout520']['r1000_lt_1e2_count']}/{pc['heldout520']['total']} "
        f"| Promo: {pr['qualified_count']}/{pr['total']} (delta={pr['delta_from_494']:+d}) "
        f"| 600: {pc['combined600']['r1000_lt_1e2_count']}/{pc['combined600']['total']}"
    )


if __name__ == "__main__":
    main()

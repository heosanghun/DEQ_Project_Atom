#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_analyze_v2.py - CTS S7 P5 Result Analyzer (v2: Nearest-Neighbor Modes [A] and [B])
Directive: TASK-0012 Section 1 / AUDIT-0011 / ESC-0004 / PROMPT-5 / GOAL.md Line 6

Operational Definition & Modes:
  - Input: results_s7/zstar_s7_a0.25_heldout520_lowest.pt (keys: q{idx:04d}_a0.25, 8 fields)
  - Existing 80 items: results_s7/r1000_reeval_P1.json (REPORT-CTS-P1 L26-27: 74/80, 65/80)
  - Nearest Neighbor Modes (--nn_mode):
    * none: Default single-run analysis. Trivial self-retrieval (1.0) is relegated to metadata.
    * seedpair ([A]): Dual-seed replication with --seed2_file.
      Computes cross-seed 1-NN retrieval accuracy: argmax_j cos(z*(2)_i, z*(1)_j) == i.
      Computes multiple equilibria ratio: cos(z*(1)_i, z*(2)_i) < 0.99.
    * zsrc ([B]): Static representation alignment with --zsrc_file (600 items).
      Computes 1-NN retrieval against z_src across heldout 520 subset and full 600 set.
  - Representations:
    Both flattened 163,840-d (64x2560) and mean-pooled 2,560-d representations are computed and labeled.
  - Zero qualitative judgment words in JSON and Markdown outputs.
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, Any, List, Optional, Tuple, Set
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
DEFAULT_HELDOUT520 = os.path.join(BASE_S5_DIR, "results_s7", "zstar_s7_a0.25_heldout520_lowest.pt")
DEFAULT_R1000_P1 = os.path.join(BASE_S5_DIR, "results_s7", "r1000_reeval_P1.json")
DEFAULT_OUT_JSON = os.path.join(BASE_S5_DIR, "results_s7", "p5_summary.json")
DEFAULT_OUT_MD = os.path.join(BASE_S5_DIR, "results_s7", "p5_summary.md")


def load_heldout520_records(pt_path: str) -> Dict[int, Dict[str, Any]]:
    """Load and parse 520 items from zstar_s7_a0.25_heldout520_lowest.pt."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Heldout 520 file not found: {pt_path}")

    raw_data = torch.load(pt_path, map_location="cpu")
    parsed: Dict[int, Dict[str, Any]] = {}

    import re
    for k, v in raw_data.items():
        m = re.match(r"^q(\d+)", str(k))
        if m:
            qid = int(m.group(1))
        else:
            try:
                qid = int(k)
            except ValueError:
                continue

        parsed[qid] = {
            "r_1000": float(v["r_1000"]),
            "r_lowest": float(v["r_lowest"]),
            "k_star": int(v["k_star"]),
            "rho_traj": [float(x) for x in v.get("rho_traj", [])],
            "top3_final": [int(x) for x in v.get("top3_final", [])],
            "top3_lowest": [int(x) for x in v.get("top3_lowest", [])],
            "z_star": v["z_star"].float() if isinstance(v["z_star"], torch.Tensor) else None,
            "z_lowest": v["z_lowest"].float() if isinstance(v["z_lowest"], torch.Tensor) else None,
        }
    return parsed


def load_raw_z_dict(pt_path: str) -> Dict[int, torch.Tensor]:
    """Load tensor dictionary mapping question ID to z tensor."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Tensor file not found: {pt_path}")

    raw_data = torch.load(pt_path, map_location="cpu")
    out: Dict[int, torch.Tensor] = {}

    import re
    for k, v in raw_data.items():
        m = re.match(r"^q(\d+)", str(k))
        if m:
            qid = int(m.group(1))
        else:
            try:
                qid = int(k)
            except ValueError:
                continue

        if isinstance(v, torch.Tensor):
            out[qid] = v.float()
        elif isinstance(v, dict):
            if "z_star" in v and isinstance(v["z_star"], torch.Tensor):
                out[qid] = v["z_star"].float()
            elif "z_lowest" in v and isinstance(v["z_lowest"], torch.Tensor):
                out[qid] = v["z_lowest"].float()
    return out


def load_existing80_r1000(p1_json_path: str) -> Dict[str, Any]:
    """Retrieve existing 80 heldout items r_1000 from P1 re-evaluation records."""
    if os.path.exists(p1_json_path):
        with open(p1_json_path, "r", encoding="utf-8") as f:
            p1_data = json.load(f)
        r_list = []
        if isinstance(p1_data, dict) and "heldout_80" in p1_data:
            r_list = [float(x["r_1000"]) for x in p1_data["heldout_80"]]
        elif isinstance(p1_data, list):
            cfg80 = os.path.join(BASE_S5_DIR, "configs", "s5_heldout_80.json")
            if os.path.exists(cfg80):
                with open(cfg80, "r", encoding="utf-8") as f80:
                    set80 = set(int(x) for x in json.load(f80)["indices"])
                for entry in p1_data:
                    if int(entry.get("q_idx", -1)) in set80:
                        r_list.append(float(entry["r_1000"]))
        if len(r_list) == 80:
            lt_1e2 = sum(1 for r in r_list if r < 1e-2)
            lt_5e3 = sum(1 for r in r_list if r < 5e-3)
            return {
                "source": p1_json_path,
                "n": 80,
                "r_lt_1e2_count": lt_1e2,
                "r_lt_1e2_ratio": lt_1e2 / 80.0,
                "r_lt_5e3_count": lt_5e3,
                "r_lt_5e3_ratio": lt_5e3 / 80.0,
                "raw_values": r_list,
            }

    # Reference values from REPORT-CTS-P1 Lines 26-27 (74/80 and 65/80)
    return {
        "source": "REPORT-CTS-P1 (Lines 26-27, raw_s7/r1000_reeval_P1.log:32-33)",
        "n": 80,
        "r_lt_1e2_count": 74,
        "r_lt_1e2_ratio": 74 / 80.0,
        "r_lt_5e3_count": 65,
        "r_lt_5e3_ratio": 65 / 80.0,
        "raw_values": None,
    }


def compute_quantiles(tensor_or_list: Any) -> Dict[str, float]:
    """Compute distribution percentiles and descriptive statistics."""
    if isinstance(tensor_or_list, torch.Tensor):
        vals = tensor_or_list.flatten().cpu().tolist()
    else:
        vals = list(tensor_or_list)

    if not vals:
        return {"min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    vals.sort()
    n = len(vals)

    def get_pct(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(vals[int(k)])
        d0 = vals[int(f)] * (c - k)
        d1 = vals[int(c)] * (k - f)
        return float(d0 + d1)

    mean_val = float(sum(vals) / n)
    var_val = float(sum((x - mean_val) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
    std_val = float(math.sqrt(var_val))

    return {
        "min": round(float(vals[0]), 6),
        "q25": round(get_pct(0.25), 6),
        "median": round(get_pct(0.50), 6),
        "q75": round(get_pct(0.75), 6),
        "max": round(float(vals[-1]), 6),
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
    }


def compute_cos_matrix(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix between row tensors T1 [N, D] and T2 [M, D]."""
    norms1 = torch.norm(T1, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    norms2 = torch.norm(T2, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    return (T1 / norms1) @ (T2 / norms2).T


def evaluate_seedpair_option_a(
    qids_520: List[int],
    z_dict_seed1: Dict[int, torch.Tensor],
    seed2_pt_path: str,
) -> Dict[str, Any]:
    """Option [A]: Dual-seed replication analysis against seed2 (.pt)."""
    z_dict_seed2 = load_raw_z_dict(seed2_pt_path)

    # Ensure aligned lists
    valid_qids = [q for q in qids_520 if q in z_dict_seed1 and q in z_dict_seed2]
    n_valid = len(valid_qids)
    if n_valid == 0:
        return {"error": "Zero matching question IDs found between seed 1 and seed 2"}

    # Stack representations
    z1_raw = torch.stack([z_dict_seed1[q] for q in valid_qids], dim=0)  # [N, 64, 2560]
    z2_raw = torch.stack([z_dict_seed2[q] for q in valid_qids], dim=0)  # [N, 64, 2560]

    out_reps: Dict[str, Any] = {}
    for rep_name, z1, z2 in [
        ("flattened_163840d", z1_raw.view(n_valid, -1), z2_raw.view(n_valid, -1)),
        ("mean_pooled_2560d", z1_raw.mean(dim=1), z2_raw.mean(dim=1)),
    ]:
        cos_mat = compute_cos_matrix(z2, z1)  # [N_seed2, N_seed1]
        
        # 1-NN retrieval: For each item i in seed2, does argmax over seed1 equal i?
        best_match_indices = cos_mat.argmax(dim=1)
        correct_matches = int((best_match_indices == torch.arange(n_valid)).sum())
        nn_match_rate = float(correct_matches / n_valid)

        # Diagonal: cos between seed 1 and seed 2 for identical items
        diag_cos = cos_mat.diag()
        multiple_equilibria_count = int((diag_cos < 0.99).sum())
        multiple_equilibria_ratio = float(multiple_equilibria_count / n_valid)

        out_reps[rep_name] = {
            "n_items": n_valid,
            "nearest_neighbor_match_count": correct_matches,
            "nearest_neighbor_match_rate": round(nn_match_rate, 6),
            "multiple_equilibria_count_cos_lt_099": multiple_equilibria_count,
            "multiple_equilibria_ratio": round(multiple_equilibria_ratio, 6),
            "cross_seed_self_cos_distribution": compute_quantiles(diag_cos),
        }

    return {
        "option": "[A] Dual-Seed Replication (Seed 2030 vs Seed 2031)",
        "seed2_file": seed2_pt_path,
        "n_evaluated": n_valid,
        "representations": out_reps,
    }


def evaluate_zsrc_option_b(
    qids_520: List[int],
    z_dict_star: Dict[int, torch.Tensor],
    zsrc_pt_path: str,
) -> Dict[str, Any]:
    """Option [B]: Alignment against static z_src (600 items)."""
    zsrc_dict = load_raw_z_dict(zsrc_pt_path)

    # 1. 520 subset alignment
    qids_in_both_520 = [q for q in qids_520 if q in z_dict_star and q in zsrc_dict]
    n_520 = len(qids_in_both_520)

    # 2. Full 600 alignment
    qids_600 = sorted(zsrc_dict.keys())
    qid_to_600_idx = {q: idx for idx, q in enumerate(qids_600)}

    if n_520 == 0:
        return {"error": "Zero matching question IDs between z* and z_src"}

    zstar_raw = torch.stack([z_dict_star[q] for q in qids_in_both_520], dim=0)  # [520, 64, 2560]
    zsrc_520_raw = torch.stack([zsrc_dict[q] for q in qids_in_both_520], dim=0)  # [520, 64, 2560]
    zsrc_600_raw = torch.stack([zsrc_dict[q] for q in qids_600], dim=0)  # [600, 64, 2560]

    out_reps: Dict[str, Any] = {}
    for rep_name, zstar, zsrc520, zsrc600 in [
        (
            "flattened_163840d",
            zstar_raw.view(n_520, -1),
            zsrc_520_raw.view(n_520, -1),
            zsrc_600_raw.view(len(qids_600), -1),
        ),
        (
            "mean_pooled_2560d",
            zstar_raw.mean(dim=1),
            zsrc_520_raw.mean(dim=1),
            zsrc_600_raw.mean(dim=1),
        ),
    ]:
        # Subset 520 retrieval: argmax over 520 z_src
        cos_mat_520 = compute_cos_matrix(zstar, zsrc520)  # [520, 520]
        best_520 = cos_mat_520.argmax(dim=1)
        match_520_count = int((best_520 == torch.arange(n_520)).sum())
        match_520_ratio = float(match_520_count / n_520)

        # Full 600 retrieval: argmax over 600 z_src
        cos_mat_600 = compute_cos_matrix(zstar, zsrc600)  # [520, 600]
        best_600 = cos_mat_600.argmax(dim=1)
        expected_600_indices = torch.tensor([qid_to_600_idx[q] for q in qids_in_both_520])
        match_600_count = int((best_600 == expected_600_indices).sum())
        match_600_ratio = float(match_600_count / n_520)

        # Diagonal cos(z*, z_src) distribution
        diag_cos = cos_mat_520.diag()

        out_reps[rep_name] = {
            "n_items": n_520,
            "subset520_nearest_neighbor_match_count": match_520_count,
            "subset520_nearest_neighbor_match_ratio": round(match_520_ratio, 6),
            "full600_nearest_neighbor_match_count": match_600_count,
            "full600_nearest_neighbor_match_ratio": round(match_600_ratio, 6),
            "zstar_vs_zsrc_self_cos_distribution": compute_quantiles(diag_cos),
        }

    return {
        "option": "[B] Static z_src Representation Alignment",
        "zsrc_file": zsrc_pt_path,
        "n_evaluated": n_520,
        "representations": out_reps,
    }


def analyze_p5_v2(
    heldout520_path: str,
    r1000_p1_path: str,
    nn_mode: str = "none",
    seed2_file: Optional[str] = None,
    zsrc_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete statistical analysis of P5 results with configurable NN modes."""
    records_520 = load_heldout520_records(heldout520_path)
    n_520 = len(records_520)
    qids = sorted(records_520.keys())

    # 1. 1차 (최종점 r_1000): 520 및 600(520 + 기존 80)
    r1000_520 = [records_520[q]["r_1000"] for q in qids]
    r1000_lt_1e2_520 = sum(1 for r in r1000_520 if r < 1e-2)
    r1000_lt_5e3_520 = sum(1 for r in r1000_520 if r < 5e-3)

    rec_80 = load_existing80_r1000(r1000_p1_path)
    n_80 = rec_80["n"]
    r1000_lt_1e2_80 = rec_80["r_lt_1e2_count"]
    r1000_lt_5e3_80 = rec_80["r_lt_5e3_count"]

    n_600 = n_520 + n_80
    r1000_lt_1e2_600 = r1000_lt_1e2_520 + r1000_lt_1e2_80
    r1000_lt_5e3_600 = r1000_lt_5e3_520 + r1000_lt_5e3_80

    primary_convergence = {
        "heldout520": {
            "total": n_520,
            "r1000_lt_1e2_count": r1000_lt_1e2_520,
            "r1000_lt_1e2_ratio": round(r1000_lt_1e2_520 / n_520, 6) if n_520 else 0.0,
            "r1000_lt_5e3_count": r1000_lt_5e3_520,
            "r1000_lt_5e3_ratio": round(r1000_lt_5e3_520 / n_520, 6) if n_520 else 0.0,
        },
        "existing80": {
            "total": n_80,
            "source": rec_80["source"],
            "r1000_lt_1e2_count": r1000_lt_1e2_80,
            "r1000_lt_1e2_ratio": round(rec_80["r_lt_1e2_ratio"], 6),
            "r1000_lt_5e3_count": r1000_lt_5e3_80,
            "r1000_lt_5e3_ratio": round(rec_80["r_lt_5e3_ratio"], 6),
        },
        "combined600": {
            "total": n_600,
            "r1000_lt_1e2_count": r1000_lt_1e2_600,
            "r1000_lt_1e2_ratio": round(r1000_lt_1e2_600 / n_600, 6) if n_600 else 0.0,
            "r1000_lt_5e3_count": r1000_lt_5e3_600,
            "r1000_lt_5e3_ratio": round(r1000_lt_5e3_600 / n_600, 6) if n_600 else 0.0,
        }
    }

    # 2. 2차 승격 판정용: 520 중 (r_lowest < 1e-2 and k_star >= 50)
    promotion_qualified_count = 0
    for q in qids:
        r_low = records_520[q]["r_lowest"]
        k_st = records_520[q]["k_star"]
        if r_low < 1e-2 and k_st >= 50:
            promotion_qualified_count += 1

    promotion_ratio = round(promotion_qualified_count / n_520, 6) if n_520 else 0.0
    promotion_threshold_count = 494  # 95.0% of 520
    promotion_diff_from_494 = promotion_qualified_count - promotion_threshold_count

    promotion_stats = {
        "qualified_count": promotion_qualified_count,
        "total": n_520,
        "ratio": promotion_ratio,
        "threshold_target_count": promotion_threshold_count,
        "threshold_target_ratio": 0.950000,
        "delta_from_494": promotion_diff_from_494,
    }

    # 3. PROMPT-5: k_star 분포
    bin_0_50 = sum(1 for q in qids if records_520[q]["k_star"] < 50)
    bin_50_300 = sum(1 for q in qids if 50 <= records_520[q]["k_star"] < 300)
    bin_300_700 = sum(1 for q in qids if 300 <= records_520[q]["k_star"] < 700)
    bin_700_1000 = sum(1 for q in qids if 700 <= records_520[q]["k_star"] <= 1000)

    kstar_distribution = {
        "bin_0_50": {"count": bin_0_50, "ratio": round(bin_0_50 / n_520, 6) if n_520 else 0.0},
        "bin_50_300": {"count": bin_50_300, "ratio": round(bin_50_300 / n_520, 6) if n_520 else 0.0},
        "bin_300_700": {"count": bin_300_700, "ratio": round(bin_300_700 / n_520, 6) if n_520 else 0.0},
        "bin_700_1000": {"count": bin_700_1000, "ratio": round(bin_700_1000 / n_520, 6) if n_520 else 0.0},
    }

    # 4. PROMPT-5: top3 라우팅 일치
    top3_equal_count = sum(
        1 for q in qids if records_520[q]["top3_lowest"] == records_520[q]["top3_final"]
    )
    top3_match_stats = {
        "match_count": top3_equal_count,
        "total": n_520,
        "ratio": round(top3_equal_count / n_520, 6) if n_520 else 0.0,
    }

    # 5. PROMPT-5: 구제 문항 (r_1000 >= 1e-2 and r_lowest < 1e-2)
    alpha = 0.25
    rescue_items_list: List[Dict[str, Any]] = []
    for q in qids:
        r_f = records_520[q]["r_1000"]
        r_l = records_520[q]["r_lowest"]
        if r_f >= 1e-2 and r_l < 1e-2:
            k_st = records_520[q]["k_star"]
            rho_tr = records_520[q]["rho_traj"]
            r_tr = [rho / alpha for rho in rho_tr]
            w_start = max(0, k_st - 10)
            w_end = min(len(r_tr), k_st + 11)
            window_slice = r_tr[w_start:w_end] if r_tr else []
            max_r_in_window = float(max(window_slice)) if window_slice else float(r_l)

            rescue_items_list.append({
                "qid": q,
                "k_star": k_st,
                "r_lowest": round(r_l, 6),
                "r_1000": round(r_f, 6),
                "kstar_pm10_max_r": round(max_r_in_window, 6),
            })

    rescue_stats = {
        "count": len(rescue_items_list),
        "ratio": round(len(rescue_items_list) / n_520, 6) if n_520 else 0.0,
        "items": rescue_items_list,
    }

    # 6. PROMPT-5: cos(z_lowest, z_final) 분포
    cos_low_fin_list: List[float] = []
    for q in qids:
        z_low = records_520[q]["z_lowest"]
        z_fin = records_520[q]["z_star"]
        if z_low is not None and z_fin is not None:
            c = float(torch.cosine_similarity(z_low.flatten(), z_fin.flatten(), dim=0))
            cos_low_fin_list.append(c)

    cos_lowest_final_distribution = compute_quantiles(cos_low_fin_list)

    # 7. PROMPT-5 & GOAL G1: 520 z_star 문항 간 pairwise cos 및 오프다이애고널 최근접
    z_tensors_list = [records_520[q]["z_star"].flatten() for q in qids if records_520[q]["z_star"] is not None]
    z_dict_star = {q: records_520[q]["z_star"] for q in qids if records_520[q]["z_star"] is not None}

    discrimination_stats: Dict[str, Any] = {
        "goal_reference_definition": "GOAL.md Line 6: 변별: 문항 간 평균 cos <= 0.75, 최근접 일치율 >= 99 %, cos > 0.95 쌍 비율·다중 평형 비율 보고",
        "n_items": len(z_tensors_list),
        "nearest_neighbor_self_retrieval_trivial": 1.0,
    }

    if len(z_tensors_list) > 1:
        Z = torch.stack(z_tensors_list, dim=0)
        norms = torch.norm(Z, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        Z_norm = Z / norms
        cos_matrix = (Z_norm @ Z_norm.T)

        N = len(z_tensors_list)
        triu_idx = torch.triu_indices(N, N, offset=1)
        pairwise_cos = cos_matrix[triu_idx[0], triu_idx[1]]

        mean_pairwise_cos = float(pairwise_cos.mean())
        pairs_gt_095_count = int((pairwise_cos > 0.95).sum())
        total_pairs = len(pairwise_cos)
        ratio_gt_095 = float(pairs_gt_095_count / total_pairs) if total_pairs else 0.0

        diag_mask = torch.eye(N, dtype=torch.bool)
        cos_offdiag = cos_matrix.masked_fill(diag_mask, -1e9)
        max_other_cos, _ = cos_offdiag.max(dim=1)

        discrimination_stats.update({
            "mean_pairwise_cos": round(mean_pairwise_cos, 6),
            "total_pairs": total_pairs,
            "pairs_gt_095_count": pairs_gt_095_count,
            "pairs_gt_095_ratio": round(ratio_gt_095, 6),
            "offdiagonal_nearest_max_cos_distribution": compute_quantiles(max_other_cos),
        })

    # 8. Nearest Neighbor Operational Modes
    nn_evaluation: Dict[str, Any] = {"mode": nn_mode}
    if nn_mode == "seedpair":
        if not seed2_file:
            raise ValueError("--seed2_file must be specified when --nn_mode is 'seedpair'")
        nn_evaluation["seedpair"] = evaluate_seedpair_option_a(qids, z_dict_star, seed2_file)
    elif nn_mode == "zsrc":
        if not zsrc_file:
            raise ValueError("--zsrc_file must be specified when --nn_mode is 'zsrc'")
        nn_evaluation["zsrc"] = evaluate_zsrc_option_b(qids, z_dict_star, zsrc_file)

    full_summary = {
        "metadata": {
            "heldout520_file": heldout520_path,
            "existing80_file": r1000_p1_path,
            "nn_mode": nn_mode,
            "n_items_heldout520": n_520,
            "n_items_existing80": n_80,
            "n_items_total": n_600,
        },
        "primary_convergence": primary_convergence,
        "promotion_evaluation": promotion_stats,
        "kstar_distribution": kstar_distribution,
        "top3_match": top3_match_stats,
        "rescue_items": rescue_stats,
        "cos_lowest_final_distribution": cos_lowest_final_distribution,
        "discrimination": discrimination_stats,
        "nearest_neighbor_evaluation": nn_evaluation,
    }
    return full_summary


def format_markdown_table_v2(summary: Dict[str, Any]) -> str:
    """Format numerical analysis as structured Markdown tables without qualitative judgment words."""
    m = summary["metadata"]
    p = summary["primary_convergence"]
    pr = summary["promotion_evaluation"]
    kd = summary["kstar_distribution"]
    tm = summary["top3_match"]
    rc = summary["rescue_items"]
    clf = summary["cos_lowest_final_distribution"]
    disc = summary["discrimination"]
    nn_eval = summary.get("nearest_neighbor_evaluation", {})

    lines = [
        "# P5 Analysis Summary Table (v2)",
        "",
        f"- Target File: `{m['heldout520_file']}`",
        f"- 80-Item Source: `{m['existing80_file']}`",
        f"- Nearest-Neighbor Mode: `{m['nn_mode']}`",
        f"- Counts: Held-out 520 = {m['n_items_heldout520']}, Existing 80 = {m['n_items_existing80']}, Total = {m['n_items_total']}",
        "",
        "## 1. Primary Residual Convergence (r_1000)",
        "",
        "| Set | N | r_1000 < 1e-2 Count | r_1000 < 1e-2 Ratio | r_1000 < 5e-3 Count | r_1000 < 5e-3 Ratio |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
        f"| Held-out 520 | {p['heldout520']['total']} | {p['heldout520']['r1000_lt_1e2_count']} | {p['heldout520']['r1000_lt_1e2_ratio']:.6f} | {p['heldout520']['r1000_lt_5e3_count']} | {p['heldout520']['r1000_lt_5e3_ratio']:.6f} |",
        f"| Existing 80 | {p['existing80']['total']} | {p['existing80']['r1000_lt_1e2_count']} | {p['existing80']['r1000_lt_1e2_ratio']:.6f} | {p['existing80']['r1000_lt_5e3_count']} | {p['existing80']['r1000_lt_5e3_ratio']:.6f} |",
        f"| Combined 600 | {p['combined600']['total']} | {p['combined600']['r1000_lt_1e2_count']} | {p['combined600']['r1000_lt_1e2_ratio']:.6f} | {p['combined600']['r1000_lt_5e3_count']} | {p['combined600']['r1000_lt_5e3_ratio']:.6f} |",
        "",
        "## 2. Promotion Evaluation (2차 승격 판정)",
        "",
        "| Metric | Value | Reference / Threshold | Delta |",
        "|:---|:---:|:---:|:---:|",
        f"| Qualified (r_lowest < 1e-2 and k* >= 50) | {pr['qualified_count']} / {pr['total']} ({pr['ratio']:.6f}) | 494 / 520 ({pr['threshold_target_ratio']:.6f}) | {pr['delta_from_494']} |",
        "",
        "## 3. PROMPT-5 Required Distributions",
        "",
        "### 3.1 k_star Step Distribution",
        "",
        "| Bin | Step Range | Count | Ratio |",
        "|:---|:---|:---:|:---:|",
        f"| Bin 1 | [0, 50) | {kd['bin_0_50']['count']} | {kd['bin_0_50']['ratio']:.6f} |",
        f"| Bin 2 | [50, 300) | {kd['bin_50_300']['count']} | {kd['bin_50_300']['ratio']:.6f} |",
        f"| Bin 3 | [300, 700) | {kd['bin_300_700']['count']} | {kd['bin_300_700']['ratio']:.6f} |",
        f"| Bin 4 | [700, 1000] | {kd['bin_700_1000']['count']} | {kd['bin_700_1000']['ratio']:.6f} |",
        "",
        "### 3.2 top3 Routing Match (top3_lowest == top3_final)",
        "",
        f"- Match Count: {tm['match_count']} / {tm['total']} ({tm['ratio']:.6f})",
        "",
        "### 3.3 Rescue Items (r_1000 >= 1e-2 and r_lowest < 1e-2)",
        "",
        f"- Rescue Count: {rc['count']} / {m['n_items_heldout520']} ({rc['ratio']:.6f})",
    ]

    if rc["items"]:
        lines.extend([
            "",
            "| Item QID | k* | r_lowest | r_1000 | k*+-10 Max r |",
            "|:---:|:---:|:---:|:---:|:---:|",
        ])
        for it in rc["items"][:20]:
            lines.append(f"| Q#{it['qid']:04d} | {it['k_star']} | {it['r_lowest']:.6e} | {it['r_1000']:.6e} | {it['kstar_pm10_max_r']:.6e} |")
        if len(rc["items"]) > 20:
            lines.append(f"| ... ({len(rc['items']) - 20} more rescue items) | | | | |")

    lines.extend([
        "",
        "### 3.4 cos(z_lowest, z_final) Distribution",
        "",
        "| Min | Q25 | Median | Q75 | Max | Mean | Std |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| {clf['min']:.6f} | {clf['q25']:.6f} | {clf['median']:.6f} | {clf['q75']:.6f} | {clf['max']:.6f} | {clf['mean']:.6f} | {clf['std']:.6f} |",
        "",
        "## 4. Discrimination (변별 — GOAL G1 Line 6)",
        "",
        f"> {disc['goal_reference_definition']}",
        "",
        "| Metric | Value | Reference Target |",
        "|:---|:---:|:---:|",
        f"| Inter-item Mean cos | {disc.get('mean_pairwise_cos', 0.0):.6f} | <= 0.75 |",
        f"| Pairs with cos > 0.95 | {disc.get('pairs_gt_095_count', 0)} / {disc.get('total_pairs', 0)} ({disc.get('pairs_gt_095_ratio', 0.0):.6f}) | Report ratio |",
    ])

    # Section 5: Nearest Neighbor Operational Modes
    if m["nn_mode"] == "seedpair" and "seedpair" in nn_eval:
        sp = nn_eval["seedpair"]
        lines.extend([
            "",
            "## 5. Nearest-Neighbor Replication Option [A] (Seed 2030 vs Seed 2031)",
            "",
            f"- 2nd Seed File: `{sp['seed2_file']}`",
            f"- Evaluated Pairs: {sp['n_evaluated']}",
            "",
            "| Representation | Dimension | 1-NN Match Count | 1-NN Match Rate | Multiple Eq (cos < 0.99) Count | Multiple Eq Ratio |",
            "|:---|:---:|:---:|:---:|:---:|:---:|",
        ])
        for rep_k, rep_v in sp["representations"].items():
            dim_str = "163,840-d (flattened 64x2560)" if "flattened" in rep_k else "2,560-d (mean-pooled 64)"
            lines.append(
                f"| {rep_k} | {dim_str} | {rep_v['nearest_neighbor_match_count']} / {rep_v['n_items']} | {rep_v['nearest_neighbor_match_rate']:.6f} | {rep_v['multiple_equilibria_count_cos_lt_099']} | {rep_v['multiple_equilibria_ratio']:.6f} |"
            )
        lines.extend([
            "",
            "### Cross-Seed Pairwise Self-Cosine Quantiles",
            "",
            "| Representation | Min | Q25 | Median | Q75 | Max | Mean | Std |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for rep_k, rep_v in sp["representations"].items():
            q_stats = rep_v["cross_seed_self_cos_distribution"]
            lines.append(
                f"| {rep_k} | {q_stats['min']:.6f} | {q_stats['q25']:.6f} | {q_stats['median']:.6f} | {q_stats['q75']:.6f} | {q_stats['max']:.6f} | {q_stats['mean']:.6f} | {q_stats['std']:.6f} |"
            )

    elif m["nn_mode"] == "zsrc" and "zsrc" in nn_eval:
        zs = nn_eval["zsrc"]
        lines.extend([
            "",
            "## 5. Nearest-Neighbor Alignment Option [B] (z* vs z_src)",
            "",
            f"- z_src File: `{zs['zsrc_file']}`",
            f"- Evaluated Items: {zs['n_evaluated']}",
            "",
            "| Representation | Dimension | Subset 520 Match Rate | Full 600 Match Rate |",
            "|:---|:---:|:---:|:---:|",
        ])
        for rep_k, rep_v in zs["representations"].items():
            dim_str = "163,840-d (flattened 64x2560)" if "flattened" in rep_k else "2,560-d (mean-pooled 64)"
            lines.append(
                f"| {rep_k} | {dim_str} | {rep_v['subset520_nearest_neighbor_match_count']} / {rep_v['n_items']} ({rep_v['subset520_nearest_neighbor_match_ratio']:.6f}) | {rep_v['full600_nearest_neighbor_match_count']} / {rep_v['n_items']} ({rep_v['full600_nearest_neighbor_match_ratio']:.6f}) |"
            )
        lines.extend([
            "",
            "### z* vs z_src Pairwise Cosine Quantiles",
            "",
            "| Representation | Min | Q25 | Median | Q75 | Max | Mean | Std |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for rep_k, rep_v in zs["representations"].items():
            q_stats = rep_v["zstar_vs_zsrc_self_cos_distribution"]
            lines.append(
                f"| {rep_k} | {q_stats['min']:.6f} | {q_stats['q25']:.6f} | {q_stats['median']:.6f} | {q_stats['q75']:.6f} | {q_stats['max']:.6f} | {q_stats['mean']:.6f} | {q_stats['std']:.6f} |"
            )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="P5 Result Analyzer v2 (NN Modes)")
    parser.add_argument("--heldout520", type=str, default=DEFAULT_HELDOUT520, help="Path to heldout 520 .pt file")
    parser.add_argument("--r1000_p1", type=str, default=DEFAULT_R1000_P1, help="Path to P1 r_1000 json file")
    parser.add_argument("--nn_mode", type=str, default="none", choices=["none", "seedpair", "zsrc"], help="Nearest neighbor mode")
    parser.add_argument("--seed2_file", type=str, default=None, help="Path to seed 2031 heldout 520 .pt file (Option [A])")
    parser.add_argument("--zsrc_file", type=str, default=None, help="Path to z_src 600 .pt file (Option [B])")
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON, help="Path to output json file")
    parser.add_argument("--out_md", type=str, default=DEFAULT_OUT_MD, help="Path to output markdown file")
    args = parser.parse_args()

    print("=" * 80)
    print("CTS S7 P5 Result Analyzer (v2)")
    print(f"Target:    {args.heldout520}")
    print(f"NN Mode:   {args.nn_mode}")
    if args.nn_mode == "seedpair":
        print(f"Seed2 File: {args.seed2_file}")
    elif args.nn_mode == "zsrc":
        print(f"z_src File: {args.zsrc_file}")
    print(f"Output JSON: {args.out_json}")
    print(f"Output MD:   {args.out_md}")
    print("=" * 80, flush=True)

    summary = analyze_p5_v2(
        heldout520_path=args.heldout520,
        r1000_p1_path=args.r1000_p1,
        nn_mode=args.nn_mode,
        seed2_file=args.seed2_file,
        zsrc_file=args.zsrc_file,
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_text = format_markdown_table_v2(summary)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"\n[DONE] Analysis written to:\n  JSON: {args.out_json}\n  MD:   {args.out_md}")


if __name__ == "__main__":
    main()

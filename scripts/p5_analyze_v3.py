#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_analyze_v3.py - CTS S7 P5 Result Analyzer v3 (PREREG_S7_DEQ_v1.3 §10(g))
Directive: TASK-0019 / PREREG_S7_DEQ_v1.3 §10(g) / AUDIT-P004 / AUDIT-0018

Operational Definition (PREREG_S7_DEQ_v1.3 §10(g)):
  - Reference item list: configs/s7_new_heldout_520.json (520 items).
  - Denominator is strictly 520 (n_denominator = 520). Never reduced to intersection.
  - Strictly uses 'z_star' field only. If absent, counted as mismatch; no fallback to z_lowest.
  - Representations:
    * Primary: Flattened 163,840-d (64x2560), cosine computed in float64.
    * Secondary (reporting only): Mean-pooled 2,560-d, cosine computed in float64.
  - 1-NN Match Criteria: For query i, target j (existing among 520), match iff:
    1) argmax_j cos(q_i, t_j) == i
    2) cos_i - max_{j != i} cos_j >= 1e-12 (ties count as mismatch).
  - Mismatches categorized into:
    * argmax_mismatch (argmax 다름)
    * margin_tie (동률 / Delta < 1e-12)
    * query_missing (질의 결측)
    * target_missing (비교 대상 결측)
    * z_star_missing (z_star 없음)
  - Multiple equilibria ratio: count of cos(z^(2030)_i, z^(2031)_i) < 0.99 / 520.
  - Zero qualitative judgment words in JSON and text outputs.
"""

import os
import sys
import json
import math
import argparse
import re
from typing import Dict, Any, List, Optional, Tuple, Set
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
DEFAULT_CONFIG_520 = os.path.join(BASE_S5_DIR, "configs", "s7_new_heldout_520.json")
if not os.path.exists(DEFAULT_CONFIG_520):
    DEFAULT_CONFIG_520 = "configs/s7_new_heldout_520.json"

DEFAULT_TARGET_FILE = os.path.join(BASE_S5_DIR, "results_s7", "zstar_s7_a0.25_heldout520_lowest.pt")
DEFAULT_QUERY_FILE = os.path.join(BASE_S5_DIR, "results_s7", "zstar_s7_a0.25_heldout520_lowest_seed2031.pt")
DEFAULT_OUT_JSON = os.path.join(BASE_S5_DIR, "results_s7", "p5_summary_v3.json")
DEFAULT_OUT_MD = os.path.join(BASE_S5_DIR, "results_s7", "p5_summary_v3.md")


def load_config_520_indices(config_path: str) -> List[int]:
    """Load canonical 520 item indices from config JSON."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config 520 file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        indices = [int(x) for x in data]
    elif isinstance(data, dict):
        if "indices" in data:
            indices = [int(x) for x in data["indices"]]
        elif "items" in data:
            indices = [int(x) for x in data["items"]]
        else:
            raise KeyError(f"Unable to find item indices in {config_path}")
    else:
        raise ValueError(f"Invalid format in {config_path}")
    assert len(indices) == 520, f"Expected 520 items, got {len(indices)}"
    return indices


def load_raw_zstar_only(pt_path: str) -> Tuple[Dict[int, torch.Tensor], Set[int]]:
    """Load dictionary mapping question ID to z_star tensor only.
    
    Strictly adheres to PREREG_S7_DEQ_v1.3 §10(g):
    - Uses ONLY the 'z_star' field.
    - Does NOT fallback to 'z_lowest' (v2 lines 91-94 removed).
    - Returns:
      z_out: mapping of qid -> z_star
      missing_zstar: set of qids present in file but missing 'z_star' key.
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Tensor file not found: {pt_path}")

    raw_data = torch.load(pt_path, map_location="cpu")
    z_out: Dict[int, torch.Tensor] = {}
    missing_zstar: Set[int] = set()

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
            z_out[qid] = v
        elif isinstance(v, dict):
            if "z_star" in v and isinstance(v["z_star"], torch.Tensor):
                z_out[qid] = v["z_star"]
            else:
                missing_zstar.add(qid)
    return z_out, missing_zstar


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


def evaluate_representation_nn(
    qids_520: List[int],
    z_target: Dict[int, torch.Tensor],
    missing_target: Set[int],
    z_query: Dict[int, torch.Tensor],
    missing_query: Set[int],
    mode: str = "flattened_163840d",
) -> Dict[str, Any]:
    """Evaluate 1-NN retrieval under PREREG_S7_DEQ_v1.3 §10(g) for a specific representation."""
    available_target_qids = [q for q in qids_520 if q in z_target]
    m_targets = len(available_target_qids)

    # Prepare normalized target tensor matrix in float64
    if m_targets > 0:
        target_tensors = []
        for q in available_target_qids:
            t = z_target[q]
            if mode == "mean_pooled_2560d":
                t_vec = t.float().mean(dim=0).double() if t.dim() == 2 else t.float().double()
            else:
                t_vec = t.flatten().double()
            target_tensors.append(t_vec)
        T = torch.stack(target_tensors, dim=0)  # [M, D] in float64
        T_norm = T / T.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        target_qid_to_idx = {q: idx for idx, q in enumerate(available_target_qids)}
    else:
        T_norm = None
        target_qid_to_idx = {}

    match_count = 0
    mismatch_counts = {
        "argmax_mismatch": 0,
        "margin_tie": 0,
        "query_missing": 0,
        "target_missing": 0,
        "z_star_missing": 0,
        "total_mismatch": 0,
    }

    per_item_detail = []
    diag_cos_list = []
    missing_items_for_multieq = 0

    for qid in qids_520:
        # Check query presence
        if qid not in z_query:
            if qid in missing_query:
                reason = "z_star_missing"
            else:
                reason = "query_missing"
            mismatch_counts[reason] += 1
            mismatch_counts["total_mismatch"] += 1
            missing_items_for_multieq += 1
            per_item_detail.append({"qid": qid, "status": "mismatch", "reason": reason})
            continue

        # Check target presence for item i
        if qid not in z_target:
            if qid in missing_target:
                reason = "z_star_missing"
            else:
                reason = "target_missing"
            mismatch_counts[reason] += 1
            mismatch_counts["total_mismatch"] += 1
            missing_items_for_multieq += 1
            per_item_detail.append({"qid": qid, "status": "mismatch", "reason": reason})
            continue

        # Both query and target exist for item i
        q_raw = z_query[qid]
        if mode == "mean_pooled_2560d":
            q_vec = q_raw.float().mean(dim=0).double() if q_raw.dim() == 2 else q_raw.float().double()
        else:
            q_vec = q_raw.flatten().double()
        q_norm = q_vec / q_vec.norm(p=2).clamp(min=1e-12)

        # Compute cosine with all available targets in float64
        cos_row = (T_norm @ q_norm).view(-1)  # [M]

        target_idx = target_qid_to_idx[qid]
        cos_i = float(cos_row[target_idx].item())
        diag_cos_list.append(cos_i)

        # Exclude index i to find max_{j != i} cos_j
        if m_targets > 1:
            mask = torch.ones(m_targets, dtype=torch.bool)
            mask[target_idx] = False
            max_other_cos = float(cos_row[mask].max().item())
        else:
            max_other_cos = -1.0

        margin = cos_i - max_other_cos
        argmax_idx = int(cos_row.argmax().item())
        argmax_qid = available_target_qids[argmax_idx]

        # 1-NN Match Criteria (§10(g)):
        # 1) argmax is i
        # 2) cos_i - max_{j != i} cos_j >= 1e-12
        if argmax_qid != qid:
            reason = "argmax_mismatch"
            mismatch_counts[reason] += 1
            mismatch_counts["total_mismatch"] += 1
            per_item_detail.append({
                "qid": qid,
                "status": "mismatch",
                "reason": reason,
                "cos_i": cos_i,
                "argmax_qid": argmax_qid,
                "max_other_cos": max_other_cos,
                "margin": margin,
            })
        elif margin < 1e-12:
            reason = "margin_tie"
            mismatch_counts[reason] += 1
            mismatch_counts["total_mismatch"] += 1
            per_item_detail.append({
                "qid": qid,
                "status": "mismatch",
                "reason": reason,
                "cos_i": cos_i,
                "argmax_qid": argmax_qid,
                "max_other_cos": max_other_cos,
                "margin": margin,
            })
        else:
            match_count += 1
            per_item_detail.append({
                "qid": qid,
                "status": "match",
                "cos_i": cos_i,
                "max_other_cos": max_other_cos,
                "margin": margin,
            })

    # Multiple equilibria: cos(z^(2030)_i, z^(2031)_i) < 0.99
    multieq_count = sum(1 for c in diag_cos_list if c < 0.99)
    multieq_ratio = round(multieq_count / 520.0, 6)

    return {
        "representation": mode,
        "n_denominator": 520,
        "match_count": match_count,
        "match_rate": round(match_count / 520.0, 6),
        "diff_from_threshold_515": match_count - 515,
        "mismatch_breakdown": mismatch_counts,
        "multiple_equilibria": {
            "count_cos_lt_099": multieq_count,
            "ratio": multieq_ratio,
            "missing_count": missing_items_for_multieq,
            "cos_distribution": compute_quantiles(diag_cos_list),
        },
        "per_item_detail": per_item_detail,
    }


def analyze_p5_v3(
    target_file: str,
    query_file: str,
    config_520_path: str = DEFAULT_CONFIG_520,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete PREREG_S7_DEQ_v1.3 §10(g) Nearest-Neighbor analysis."""
    print("=" * 80)
    print("CTS S7 P5 Result Analyzer v3 (PREREG_S7_DEQ_v1.3 §10(g))")
    print(f"Target File (Seed 2030): {target_file}")
    print(f"Query File  (Seed 2031): {query_file}")
    print(f"Config 520 Path:         {config_520_path}")
    print("=" * 80, flush=True)

    # 1. Load canonical 520 item indices
    qids_520 = load_config_520_indices(config_520_path)

    # 2. Check identical file warning
    same_file_warning = None
    if os.path.exists(target_file) and os.path.exists(query_file):
        if os.path.abspath(target_file) == os.path.abspath(query_file):
            same_file_warning = "[경고] 동일 파일 지정: 대상 파일과 질의 파일이 동일합니다. 다중 평형 수는 정의상 0입니다."
            print(same_file_warning, file=sys.stderr)

    # 3. Load target and query tensors (strictly z_star only)
    z_target, missing_target = load_raw_zstar_only(target_file)
    z_query, missing_query = load_raw_zstar_only(query_file)

    # 4. Primary Evaluation: Flattened 163,840-d representation in float64
    eval_flat = evaluate_representation_nn(
        qids_520=qids_520,
        z_target=z_target,
        missing_target=missing_target,
        z_query=z_query,
        missing_query=missing_query,
        mode="flattened_163840d",
    )

    # 5. Secondary Evaluation (Reporting only): Mean-pooled 2,560-d in float64
    eval_pooled = evaluate_representation_nn(
        qids_520=qids_520,
        z_target=z_target,
        missing_target=missing_target,
        z_query=z_query,
        missing_query=missing_query,
        mode="mean_pooled_2560d",
    )
    eval_pooled["reporting_only"] = True

    # Build summary dict
    summary: Dict[str, Any] = {
        "definition_source": "PREREG_S7_DEQ_v1.3 §10(g)",
        "target_file": target_file,
        "query_file": query_file,
        "config_520": config_520_path,
        "n_denominator": 520,
        "same_file_warning": same_file_warning,
        "primary_flattened_163840d": {
            "match_count": eval_flat["match_count"],
            "match_rate": eval_flat["match_rate"],
            "diff_from_threshold_515": eval_flat["diff_from_threshold_515"],
            "mismatch_breakdown": eval_flat["mismatch_breakdown"],
            "multiple_equilibria": eval_flat["multiple_equilibria"],
        },
        "mean_pooled_2560d_reporting_only": {
            "reporting_only": True,
            "match_count": eval_pooled["match_count"],
            "match_rate": eval_pooled["match_rate"],
            "diff_from_threshold_515": eval_pooled["diff_from_threshold_515"],
            "mismatch_breakdown": eval_pooled["mismatch_breakdown"],
            "multiple_equilibria": eval_pooled["multiple_equilibria"],
        },
    }

    # Print summary table
    print("\n" + "=" * 80)
    print("NEAREST-NEIGHBOR RETRIEVAL SUMMARY (§10(g))")
    print(f"Denominator:             520")
    print(f"Primary (163,840-d):     Match={eval_flat['match_count']}/520 ({eval_flat['match_rate']:.6f}) | Delta from 515: {eval_flat['diff_from_threshold_515']:+d}")
    print(f"  Mismatches Breakdown:  argmax={eval_flat['mismatch_breakdown']['argmax_mismatch']}, tie={eval_flat['mismatch_breakdown']['margin_tie']}, query_miss={eval_flat['mismatch_breakdown']['query_missing']}, target_miss={eval_flat['mismatch_breakdown']['target_missing']}, zstar_miss={eval_flat['mismatch_breakdown']['z_star_missing']}")
    print(f"  Multi-Equilibria (<.99): count={eval_flat['multiple_equilibria']['count_cos_lt_099']}/520 (ratio: {eval_flat['multiple_equilibria']['ratio']:.6f}, missing: {eval_flat['multiple_equilibria']['missing_count']})")
    print(f"Secondary (2,560-d pool): Match={eval_pooled['match_count']}/520 ({eval_pooled['match_rate']:.6f}) [Reporting Only]")
    print("=" * 80, flush=True)

    # Save JSON output
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Results JSON saved to: {out_json}")

    # Save Markdown output
    if out_md:
        os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
        md_text = generate_markdown_report(summary)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"Results Markdown saved to: {out_md}")

    return summary


def generate_markdown_report(s: Dict[str, Any]) -> str:
    """Generate markdown table summary without forbidden judgment words."""
    flat = s["primary_flattened_163840d"]
    pool = s["mean_pooled_2560d_reporting_only"]
    warn_line = f"\n> **경고**: {s['same_file_warning']}\n" if s.get("same_file_warning") else ""

    lines = [
        f"# P5 Nearest-Neighbor Analysis Report ({s['definition_source']})",
        warn_line,
        "## 1. 1-NN Retrieval Evaluation Summary",
        "| Representation | Evaluated / Denom | Match Count | Match Rate | Delta from 515 | Mult-Eq (<0.99) | Note |",
        "|---|:---:|:---:|:---:|:---:|:---:|---|",
        f"| Flattened (163,840-d) | 520 / 520 | {flat['match_count']} | {flat['match_rate']:.6f} | {flat['diff_from_threshold_515']:+d} | {flat['multiple_equilibria']['count_cos_lt_099']}/520 | Primary Criteria (§10(g)) |",
        f"| Mean-Pooled (2,560-d) | 520 / 520 | {pool['match_count']} | {pool['match_rate']:.6f} | {pool['diff_from_threshold_515']:+d} | {pool['multiple_equilibria']['count_cos_lt_099']}/520 | Reporting Only |",
        "",
        "## 2. Mismatch Breakdown (Flattened 163,840-d)",
        "| Mismatch Category | Count | Proportion (/ 520) |",
        "|---|:---:|:---:|",
        f"| argmax 다름 (`argmax_mismatch`) | {flat['mismatch_breakdown']['argmax_mismatch']} | {flat['mismatch_breakdown']['argmax_mismatch'] / 520.0:.6f} |",
        f"| 동률 (`margin_tie`, $\\Delta < 10^{{-12}}$) | {flat['mismatch_breakdown']['margin_tie']} | {flat['mismatch_breakdown']['margin_tie'] / 520.0:.6f} |",
        f"| 질의 결측 (`query_missing`) | {flat['mismatch_breakdown']['query_missing']} | {flat['mismatch_breakdown']['query_missing'] / 520.0:.6f} |",
        f"| 비교 대상 결측 (`target_missing`) | {flat['mismatch_breakdown']['target_missing']} | {flat['mismatch_breakdown']['target_missing'] / 520.0:.6f} |",
        f"| z_star 필드 없음 (`z_star_missing`) | {flat['mismatch_breakdown']['z_star_missing']} | {flat['mismatch_breakdown']['z_star_missing'] / 520.0:.6f} |",
        f"| **총 불일치 수 (`total_mismatch`)** | **{flat['mismatch_breakdown']['total_mismatch']}** | **{flat['mismatch_breakdown']['total_mismatch'] / 520.0:.6f}** |",
        "",
        "## 3. Cosine Distribution (Diagonal $\\cos(z^{{(2030)}}_i, z^{{(2031)}}_i)$)",
        "| Metric | Min | Q25 | Median | Q75 | Max | Mean | Std |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| Flattened (163,840-d) | {flat['multiple_equilibria']['cos_distribution']['min']} | {flat['multiple_equilibria']['cos_distribution']['q25']} | {flat['multiple_equilibria']['cos_distribution']['median']} | {flat['multiple_equilibria']['cos_distribution']['q75']} | {flat['multiple_equilibria']['cos_distribution']['max']} | {flat['multiple_equilibria']['cos_distribution']['mean']} | {flat['multiple_equilibria']['cos_distribution']['std']} |",
        f"| Mean-Pooled (2,560-d) | {pool['multiple_equilibria']['cos_distribution']['min']} | {pool['multiple_equilibria']['cos_distribution']['q25']} | {pool['multiple_equilibria']['cos_distribution']['median']} | {pool['multiple_equilibria']['cos_distribution']['q75']} | {pool['multiple_equilibria']['cos_distribution']['max']} | {pool['multiple_equilibria']['cos_distribution']['mean']} | {pool['multiple_equilibria']['cos_distribution']['std']} |",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="P5 Result Analyzer v3 (PREREG_S7_DEQ_v1.3 §10(g))")
    parser.add_argument("--target_file", "--heldout520", "--seed1_file", dest="target_file", type=str, default=DEFAULT_TARGET_FILE, help="Path to target .pt (seed 2030)")
    parser.add_argument("--query_file", "--seed2_file", dest="query_file", type=str, default=DEFAULT_QUERY_FILE, help="Path to query .pt (seed 2031)")
    parser.add_argument("--config_520", type=str, default=DEFAULT_CONFIG_520, help="Path to s7_new_heldout_520.json")
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON, help="Output JSON path")
    parser.add_argument("--out_md", type=str, default=DEFAULT_OUT_MD, help="Output Markdown path")
    args = parser.parse_args()

    analyze_p5_v3(
        target_file=args.target_file,
        query_file=args.query_file,
        config_520_path=args.config_520,
        out_json=args.out_json,
        out_md=args.out_md,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_analyze_v4.py - CTS S7 P5 Result Analyzer v4 (PREREG_S7_DEQ_v1.3 §10(g))
Directive: TASK-0020 / AUDIT-P005 (FIX 1~5) / PREREG_S7_DEQ_v1.3 §10(g)

Operational Definition & Fixes (AUDIT-P005):
  - Reference config: configs/s7_new_heldout_520.json (520 unique items).
    Strict config SHA-256 validation against canonical hash:
    48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988.
  - Denominator is strictly 520 (n_denominator = 520). Never reduced to intersection.
  - Key Matching (FIX 3): Strictly matches ^q(\\d{4})_a0\\.25$.
    Other alpha keys (e.g. _a0.50) are ignored and logged. Duplicate qids raise error.
  - Dictionary requirement (FIX 2): Entry MUST be a dict containing 'z_star' Tensor.
    Bare tensors without dict are treated as 'z_star_missing'.
  - Tensor validation (FIX 1 & 3): 'z_star' MUST have shape [64, 2560], be finite
    (no NaN/Inf), and have norm > 0. Otherwise categorized as 'invalid_tensor'.
  - Target candidate isolation (FIX 1): Invalid targets are excluded from the candidate
    gallery matrix to prevent NaN contamination of argmax across other queries.
  - 1-NN Match Criteria (§10(g)):
    1) argmax_j cos(q_i, t_j) == i (over available valid targets in 520)
    2) cos_i - max_{j != i} cos_j >= 1e-12 (ties count as mismatch).
  - Mismatch breakdown categories:
    * argmax_mismatch (argmax 다름)
    * margin_tie (동률 / Delta < 1e-12)
    * query_missing (질의 파일에 문항 결측)
    * target_missing (비교 대상 파일에 문항 결측)
    * z_star_missing (z_star 필드 없음 / dict 아님)
    * invalid_tensor (NaN / Inf / 0-norm / 형상 불일치)
  - CLI (FIX 4): Accepts --nn_mode seedpair.
    Defaults aligned with §10(g):
    Target = results_s7/zstar_s7_a0.25_heldout520_lowest.pt (Seed 2030)
    Query  = results_s7/zstar_s7_a0.25_heldout520_seed2031.pt (Seed 2031)
  - Identical file warning: Emitted to stderr if target and query paths are identical.
  - Metadata: Records script SHA, config SHA, target SHA, query SHA.
  - Zero qualitative judgment words in JSON and text outputs.
"""

import os
import sys
import json
import math
import argparse
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple, Set
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
EXPECTED_CONFIG_SHA256 = "48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988"

DEFAULT_CONFIG_520 = os.path.join(BASE_S5_DIR, "configs", "s7_new_heldout_520.json")
if not os.path.exists(DEFAULT_CONFIG_520):
    DEFAULT_CONFIG_520 = "configs/s7_new_heldout_520.json"

DEFAULT_TARGET_FILE = os.path.join(BASE_S5_DIR, "results_s7", "zstar_s7_a0.25_heldout520_lowest.pt")
DEFAULT_QUERY_FILE = os.path.join(BASE_S5_DIR, "results_s7", "zstar_s7_a0.25_heldout520_seed2031.pt")
DEFAULT_OUT_JSON = os.path.join(BASE_S5_DIR, "results_s7", "p5_summary_v4.json")
DEFAULT_OUT_MD = os.path.join(BASE_S5_DIR, "results_s7", "p5_summary_v4.md")


def compute_file_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_config_520_indices(config_path: str, skip_hash_check: bool = False) -> List[int]:
    """Load canonical 520 item indices from config JSON and enforce SHA-256 and uniqueness."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config 520 file not found: {config_path}")

    actual_sha = compute_file_sha256(config_path)
    if not skip_hash_check:
        if actual_sha != EXPECTED_CONFIG_SHA256:
            raise ValueError(
                f"Config SHA-256 mismatch for {config_path}!\n"
                f"  Expected: {EXPECTED_CONFIG_SHA256}\n"
                f"  Actual:   {actual_sha}"
            )

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

    if len(indices) != 520:
        raise ValueError(f"Expected exactly 520 items in config, got {len(indices)}")
    if len(set(indices)) != 520:
        raise ValueError(f"Duplicate indices found in {config_path}: {len(indices)} total, {len(set(indices))} unique")

    return indices


def load_raw_zstar_only(
    pt_path: str,
) -> Tuple[Dict[int, torch.Tensor], Set[int], Set[int], int]:
    """Load dictionary mapping question ID to z_star tensor only.

    Strictly adheres to PREREG_S7_DEQ_v1.3 §10(g) & AUDIT-P005 FIX 1, 2, 3:
    - Keys MUST match r"^q(\\d{4})_a0\\.25$".
    - Keys with other alpha suffixes (e.g. _a0.50) are ignored and counted.
    - Duplicate qid for _a0.25 raises ValueError.
    - Value MUST be a dict containing 'z_star' key as a torch.Tensor.
      Bare tensors without dict are treated as missing_zstar (FIX 2).
    - 'z_star' tensor MUST have shape [64, 2560], be finite (no NaN/Inf), and have norm > 0.
      Otherwise categorized as invalid_tensor (FIX 1 & 3).

    Returns:
      z_out: mapping of qid -> z_star (valid tensors only)
      missing_zstar: set of qids whose entry is missing 'z_star' key or not dict
      invalid_tensors: set of qids whose 'z_star' has NaN/Inf, norm==0, or wrong shape
      ignored_other_alpha_count: count of keys ignored due to other alpha suffixes
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Tensor file not found: {pt_path}")

    raw_data = torch.load(pt_path, map_location="cpu")
    if not isinstance(raw_data, dict):
        raise ValueError(f"Root object in {pt_path} must be a dict, got {type(raw_data)}")

    z_out: Dict[int, torch.Tensor] = {}
    missing_zstar: Set[int] = set()
    invalid_tensors: Set[int] = set()
    seen_qids: Set[int] = set()
    ignored_other_alpha_count = 0

    for k, v in raw_data.items():
        k_str = str(k).strip()

        # Check for other alpha suffixes like ^q(\d{4})_a(\d+\.\d+)$
        m_alpha = re.match(r"^q(\d{4})_a(\d+\.\d+)$", k_str)
        if m_alpha:
            alpha_val = m_alpha.group(2)
            if alpha_val != "0.25":
                ignored_other_alpha_count += 1
                continue

        # Exact match required for a0.25
        m = re.match(r"^q(\d{4})_a0\.25$", k_str)
        if not m:
            continue

        qid = int(m.group(1))
        if qid in seen_qids:
            raise ValueError(f"Duplicate index q{qid:04d}_a0.25 encountered in {pt_path}")
        seen_qids.add(qid)

        # FIX 2: Must be dict containing 'z_star' torch.Tensor
        if not isinstance(v, dict):
            missing_zstar.add(qid)
            continue
        if "z_star" not in v or not isinstance(v["z_star"], torch.Tensor):
            missing_zstar.add(qid)
            continue

        z = v["z_star"]

        # FIX 1 & 3: Check shape [64, 2560], finiteness, and norm > 0
        if list(z.shape) != [64, 2560]:
            invalid_tensors.add(qid)
            continue
        if not torch.isfinite(z).all():
            invalid_tensors.add(qid)
            continue
        if float(z.double().norm(p=2).item()) == 0.0:
            invalid_tensors.add(qid)
            continue

        z_out[qid] = z

    if ignored_other_alpha_count > 0:
        print(f"[INFO] Ignored {ignored_other_alpha_count} keys with alpha != 0.25 in {pt_path}", file=sys.stderr)

    return z_out, missing_zstar, invalid_tensors, ignored_other_alpha_count


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
    invalid_target: Set[int],
    z_query: Dict[int, torch.Tensor],
    missing_query: Set[int],
    invalid_query: Set[int],
    mode: str = "flattened_163840d",
) -> Dict[str, Any]:
    """Evaluate 1-NN retrieval under PREREG_S7_DEQ_v1.3 §10(g) & AUDIT-P005 FIX 1~5."""
    # Available targets for candidate pool: strictly valid targets in 520
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
        "invalid_tensor": 0,
        "total_mismatch": 0,
    }

    per_item_detail = []

    for qid in qids_520:
        # Check query presence and validity
        if qid in invalid_query:
            reason = "invalid_tensor"
            mismatch_counts[reason] += 1
            mismatch_counts["total_mismatch"] += 1
            per_item_detail.append({"qid": qid, "status": "mismatch", "reason": reason})
            continue
        if qid in missing_query:
            reason = "z_star_missing"
            mismatch_counts[reason] += 1
            mismatch_counts["total_mismatch"] += 1
            per_item_detail.append({"qid": qid, "status": "mismatch", "reason": reason})
            continue
        if qid not in z_query:
            reason = "query_missing"
            mismatch_counts[reason] += 1
            mismatch_counts["total_mismatch"] += 1
            per_item_detail.append({"qid": qid, "status": "mismatch", "reason": reason})
            continue

        # Check target presence and validity for item i
        if qid in invalid_target:
            reason = "invalid_tensor"
            mismatch_counts[reason] += 1
            mismatch_counts["total_mismatch"] += 1
            per_item_detail.append({"qid": qid, "status": "mismatch", "reason": reason})
            continue
        if qid in missing_target:
            reason = "z_star_missing"
            mismatch_counts[reason] += 1
            mismatch_counts["total_mismatch"] += 1
            per_item_detail.append({"qid": qid, "status": "mismatch", "reason": reason})
            continue
        if qid not in z_target:
            reason = "target_missing"
            mismatch_counts[reason] += 1
            mismatch_counts["total_mismatch"] += 1
            per_item_detail.append({"qid": qid, "status": "mismatch", "reason": reason})
            continue

        # Both query and target exist and are valid for item i
        q_raw = z_query[qid]
        if mode == "mean_pooled_2560d":
            q_vec = q_raw.float().mean(dim=0).double() if q_raw.dim() == 2 else q_raw.float().double()
        else:
            q_vec = q_raw.flatten().double()
        q_norm = q_vec / q_vec.norm(p=2).clamp(min=1e-12)

        # Compute cosine with all available valid targets in float64
        cos_row = (T_norm @ q_norm).view(-1)  # [M]

        target_idx = target_qid_to_idx[qid]
        cos_i = float(cos_row[target_idx].item())

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

    # Multiple equilibria calculation: count of cos(z^(2030)_i, z^(2031)_i) < 0.99 / 520
    diag_cos_list = []
    multieq_count = 0
    missing_for_multieq = 0
    invalid_for_multieq = 0

    for qid in qids_520:
        if qid in invalid_query or qid in invalid_target:
            invalid_for_multieq += 1
            continue
        if qid not in z_query or qid not in z_target:
            missing_for_multieq += 1
            continue

        q_t = z_query[qid]
        t_t = z_target[qid]
        if mode == "mean_pooled_2560d":
            q_v = q_t.float().mean(dim=0).double() if q_t.dim() == 2 else q_t.float().double()
            t_v = t_t.float().mean(dim=0).double() if t_t.dim() == 2 else t_t.float().double()
        else:
            q_v = q_t.flatten().double()
            t_v = t_t.flatten().double()

        q_n = q_v / q_v.norm(p=2).clamp(min=1e-12)
        t_n = t_v / t_v.norm(p=2).clamp(min=1e-12)
        diag_c = float((q_n @ t_n).item())

        if math.isnan(diag_c) or math.isinf(diag_c):
            invalid_for_multieq += 1
            continue

        diag_cos_list.append(diag_c)
        if diag_c < 0.99:
            multieq_count += 1

    return {
        "mode": mode,
        "n_denominator": 520,
        "match_count": match_count,
        "match_ratio": round(match_count / 520.0, 6),
        "diff_from_threshold_515": match_count - 515,
        "mismatch_breakdown": mismatch_counts,
        "multiple_equilibria": {
            "threshold": 0.99,
            "count_cos_lt_099": multieq_count,
            "ratio_cos_lt_099": round(multieq_count / 520.0, 6),
            "evaluated_items": len(diag_cos_list),
            "missing_count": missing_for_multieq,
            "invalid_count": invalid_for_multieq,
            "cos_distribution": compute_quantiles(diag_cos_list),
        },
        "per_item_detail": per_item_detail,
    }


def analyze_p5_v4(
    target_file: str,
    query_file: str,
    config_520_path: str = DEFAULT_CONFIG_520,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
    nn_mode: str = "seedpair",
    skip_config_hash_check: bool = False,
) -> Dict[str, Any]:
    """Run full P5 analysis pipeline v4 conforming to PREREG_S7_DEQ_v1.3 §10(g) & AUDIT-P005."""
    if nn_mode != "seedpair":
        raise ValueError(f"Unsupported nn_mode: {nn_mode}. PREREG_S7_DEQ_v1.3 §10(g) specifies 'seedpair'.")

    # Identical file check: emit warning to stderr
    is_same_file = False
    try:
        if os.path.abspath(target_file) == os.path.abspath(query_file):
            is_same_file = True
            print("[경고] 동일 파일 지정: 비교 대상과 질의 파일이 동일합니다.", file=sys.stderr)
    except Exception:
        pass

    qids_520 = load_config_520_indices(config_520_path, skip_hash_check=skip_config_hash_check)

    # Load target (seed 2030)
    z_target, missing_target, invalid_target, target_ignored_alpha = load_raw_zstar_only(target_file)

    # Load query (seed 2031)
    z_query, missing_query, invalid_query, query_ignored_alpha = load_raw_zstar_only(query_file)

    # Primary representation: Flattened 163,840-d
    res_flat = evaluate_representation_nn(
        qids_520=qids_520,
        z_target=z_target,
        missing_target=missing_target,
        invalid_target=invalid_target,
        z_query=z_query,
        missing_query=missing_query,
        invalid_query=invalid_query,
        mode="flattened_163840d",
    )

    # Secondary representation: Mean-pooled 2,560-d (reporting only)
    res_pooled = evaluate_representation_nn(
        qids_520=qids_520,
        z_target=z_target,
        missing_target=missing_target,
        invalid_target=invalid_target,
        z_query=z_query,
        missing_query=missing_query,
        invalid_query=invalid_query,
        mode="mean_pooled_2560d",
    )
    res_pooled["reporting_only"] = True

    # Compute hashes for metadata
    script_sha = compute_file_sha256(__file__)
    config_sha = compute_file_sha256(config_520_path)
    target_sha = compute_file_sha256(target_file)
    query_sha = compute_file_sha256(query_file)

    summary: Dict[str, Any] = {
        "metadata": {
            "analyzer_version": "v4",
            "analyzer_script_sha256": script_sha,
            "definition_source": "PREREG_S7_DEQ_v1.3 §10(g)",
            "nn_mode": nn_mode,
            "n_denominator": 520,
            "identical_file_warning": is_same_file,
            "config_520_path": config_520_path,
            "config_520_sha256": config_sha,
            "target_file_path": target_file,
            "target_file_sha256": target_sha,
            "target_ignored_other_alpha_count": target_ignored_alpha,
            "query_file_path": query_file,
            "query_file_sha256": query_sha,
            "query_ignored_other_alpha_count": query_ignored_alpha,
        },
        "primary_flattened_163840d": res_flat,
        "mean_pooled_2560d_reporting_only": res_pooled,
    }

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if out_md:
        os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
        md_text = format_markdown_report_v4(summary)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_text)

    return summary


def format_markdown_report_v4(summary: Dict[str, Any]) -> str:
    """Format neutral markdown report without qualitative judgment words."""
    meta = summary["metadata"]
    flat = summary["primary_flattened_163840d"]
    pool = summary["mean_pooled_2560d_reporting_only"]

    lines = [
        "# P5 1-NN Retrieval Analysis Summary (v4)",
        "",
        "- **Definition Source**: PREREG_S7_DEQ_v1.3 §10(g)",
        f"- **Analyzer Version**: {meta['analyzer_version']} (SHA-256: `{meta['analyzer_script_sha256']}`)",
        f"- **NN Mode**: `{meta['nn_mode']}`",
        f"- **Canonical Denominator**: {meta['n_denominator']}",
        f"- **Config 520 Path**: `{meta['config_520_path']}` (SHA-256: `{meta['config_520_sha256']}`)",
        f"- **Target File (Seed 2030)**: `{meta['target_file_path']}` (SHA-256: `{meta['target_file_sha256']}`)",
        f"- **Query File (Seed 2031)**: `{meta['query_file_path']}` (SHA-256: `{meta['query_file_sha256']}`)",
        f"- **Identical File Specified**: {meta['identical_file_warning']}",
        "",
        "## 1. 1-NN Retrieval Accuracy & Multiple Equilibria",
        "| Representation | Dimension | Match Count (/ 520) | Match Ratio | Diff from 515 | Mult-Eq Count (< 0.99) | Mult-Eq Ratio | Mult-Eq Evaluated | Missing | Invalid | Status |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| Flattened (Primary) | 163,840 | {flat['match_count']} | {flat['match_ratio']:.6f} | {flat['diff_from_threshold_515']} | {flat['multiple_equilibria']['count_cos_lt_099']} | {flat['multiple_equilibria']['ratio_cos_lt_099']:.6f} | {flat['multiple_equilibria']['evaluated_items']} | {flat['multiple_equilibria']['missing_count']} | {flat['multiple_equilibria']['invalid_count']} | Primary |",
        f"| Mean-Pooled | 2,560 | {pool['match_count']} | {pool['match_ratio']:.6f} | {pool['diff_from_threshold_515']} | {pool['multiple_equilibria']['count_cos_lt_099']} | {pool['multiple_equilibria']['ratio_cos_lt_099']:.6f} | {pool['multiple_equilibria']['evaluated_items']} | {pool['multiple_equilibria']['missing_count']} | {pool['multiple_equilibria']['invalid_count']} | Reporting Only |",
        "",
        "## 2. Mismatch Breakdown (Flattened 163,840-d)",
        "| Mismatch Category | Count | Proportion (/ 520) |",
        "|---|:---:|:---:|",
        f"| argmax 다름 (`argmax_mismatch`) | {flat['mismatch_breakdown']['argmax_mismatch']} | {flat['mismatch_breakdown']['argmax_mismatch'] / 520.0:.6f} |",
        f"| 동률 (`margin_tie`, $\\Delta < 10^{{-12}}$) | {flat['mismatch_breakdown']['margin_tie']} | {flat['mismatch_breakdown']['margin_tie'] / 520.0:.6f} |",
        f"| 질의 결측 (`query_missing`) | {flat['mismatch_breakdown']['query_missing']} | {flat['mismatch_breakdown']['query_missing'] / 520.0:.6f} |",
        f"| 비교 대상 결측 (`target_missing`) | {flat['mismatch_breakdown']['target_missing']} | {flat['mismatch_breakdown']['target_missing'] / 520.0:.6f} |",
        f"| z_star 필드 없음 (`z_star_missing`) | {flat['mismatch_breakdown']['z_star_missing']} | {flat['mismatch_breakdown']['z_star_missing'] / 520.0:.6f} |",
        f"| 무효 텐서 (`invalid_tensor`) | {flat['mismatch_breakdown']['invalid_tensor']} | {flat['mismatch_breakdown']['invalid_tensor'] / 520.0:.6f} |",
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
    parser = argparse.ArgumentParser(description="P5 Result Analyzer v4 (PREREG_S7_DEQ_v1.3 §10(g))")
    parser.add_argument("--target_file", "--heldout520", "--seed1_file", "--seed2030_file", dest="target_file", type=str, default=DEFAULT_TARGET_FILE, help="Path to target .pt (seed 2030)")
    parser.add_argument("--query_file", "--seed2_file", "--seed2031_file", dest="query_file", type=str, default=DEFAULT_QUERY_FILE, help="Path to query .pt (seed 2031)")
    parser.add_argument("--config_520", type=str, default=DEFAULT_CONFIG_520, help="Path to s7_new_heldout_520.json")
    parser.add_argument("--nn_mode", type=str, choices=["seedpair"], default="seedpair", help="Nearest-neighbor analysis mode (PREREG_S7_DEQ_v1.3 §10(g))")
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON, help="Output JSON path")
    parser.add_argument("--out_md", type=str, default=DEFAULT_OUT_MD, help="Output Markdown path")
    parser.add_argument("--skip_config_hash_check", action="store_true", help="Skip strict SHA-256 validation of config_520")
    args = parser.parse_args()

    analyze_p5_v4(
        target_file=args.target_file,
        query_file=args.query_file,
        config_520_path=args.config_520,
        out_json=args.out_json,
        out_md=args.out_md,
        nn_mode=args.nn_mode,
        skip_config_hash_check=args.skip_config_hash_check,
    )


if __name__ == "__main__":
    main()

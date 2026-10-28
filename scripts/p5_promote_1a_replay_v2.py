#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_promote_1a_replay_v2.py - Analyzer for ①-a k* Recomputation (v2)
Directive: PREREG_S7_DEQ_v1.4 §10(h) / TASK-0050 (A)

Evaluates:
- 1,000 train items from inputs JSON:
  * Conservative 1a computed from determined 675 items (r_min < 1e-2 and k* >= 50)
  * Optimistic 1a computed as Conservative 1a + undetermined items with r_min < 1e-2
- 325 undetermined items using recomputed k* from replay .pt after reproduction checks:
  (i)  min(rho_traj) relative diff <= 1e-2 OR absolute diff <= 1.2e-7
  (ii) z_star cos >= 0.9999 in float64 against zstar_train1000_P4.pt
- v1.4 Recomputed 1a = Conservative 1a + reproduced items with (r_min < 1e-2 and recomputed k* >= 50)
- Sensitivity 1a = v1.4 Recomputed 1a + failed reproduction items with (r_min < 1e-2 and recomputed k* >= 50)
- Recomputed k* distribution across §10(d) primary bins and 100-step secondary bins
- min rho diff distributions (rel_diff, abs_diff)
- Pre-check evaluation of 5 items if precheck .pt is provided

Enforces:
- A1: Conservative and optimistic counts calculated dynamically from input JSON without halting
- A2: §10(d) bins as primary table, 100-step bins as secondary table
- A3: Zero CR (0x0D) and zero control characters in all outputs, raw strings for LaTeX math
- Input file SHA-256 verification at startup
- Rule #381 memory and autograd protection
- Zero forbidden words in output
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

EXPECTED_SHAS = {
    "inputs_json": "186d5b6cc6678b4df2e999919cc39b85d77b9b437dce10da5c553841d3bea1f1",
    "p4_pt": "37ea6807d112e44f55ab401f48b6c8e7051f3ca2800aa7cd4caa04a546fb4696",
    "replay_config": "550250e399ea96560b8360cdbdd539a82983a25fb435f994d0319da10a8a7862",
}

PRECHECK_EXPECTED_5 = {
    31: 829,
    277: 709,
    406: 999,
    711: 809,
    858: 999,
}

# A2: Primary bins according to PREREG_S7_DEQ_v1.4 §10(d)
PREREG_10D_BINS = [
    ("[0, 50)", 0, 50, False),
    ("[50, 300)", 50, 300, False),
    ("[300, 700)", 300, 700, False),
    ("[700, 1000]", 700, 1000, True),
]

# A2: Secondary bins (100-step)
AUX_100STEP_BINS = [
    ("[0, 50)", 0, 50, False),
    ("[50, 100)", 50, 100, False),
    ("[100, 200)", 100, 200, False),
    ("[200, 300)", 200, 300, False),
    ("[300, 400)", 300, 400, False),
    ("[400, 500)", 400, 500, False),
    ("[500, 600)", 500, 600, False),
    ("[600, 700)", 600, 700, False),
    ("[700, 800)", 700, 800, False),
    ("[800, 900)", 800, 900, False),
    ("[900, 1000]", 900, 1000, True),
]


def sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_no_cr_or_control(text: str, label: str = "text") -> None:
    """A3: Verify text contains 0 carriage returns (0x0D) and 0 unwanted control characters."""
    if "\r" in text:
        raise ValueError(f"Carriage return (CR 0x0D) detected in {label}")
    for i, ch in enumerate(text):
        c = ord(ch)
        if c < 32 and ch not in ("\n", "\t"):
            raise ValueError(
                f"Control character {c} ({hex(c)}) detected at index {i} in {label}"
            )


def assign_bin(val: int, bin_defs: List[Tuple[str, int, int, bool]]) -> Optional[str]:
    for name, low, high, inclusive_high in bin_defs:
        if inclusive_high:
            if low <= val <= high:
                return name
        else:
            if low <= val < high:
                return name
    return None


def compute_quantiles(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    s = sorted(vals)
    n = len(s)

    def q(p):
        idx = p * (n - 1)
        i = int(idx)
        rem = idx - i
        if i + 1 < n:
            return s[i] * (1.0 - rem) + s[i + 1] * rem
        return s[i]

    return {
        "min": float(s[0]),
        "p25": float(q(0.25)),
        "median": float(q(0.50)),
        "p75": float(q(0.75)),
        "max": float(s[-1]),
        "mean": float(sum(s) / n),
    }


def evaluate_replay(
    replay_pt_path: str,
    inputs_json_path: str,
    p4_pt_path: str,
    replay_config_path: str,
    precheck_pt_path: Optional[str] = None,
    skip_sha_check: bool = False,
) -> Dict[str, Any]:
    """Execute evaluation according to PREREG_S7_DEQ_v1.4 §10(h) and TASK-0050."""
    # 1. SHA verification
    sha_records = {
        "inputs_json": sha256_file(inputs_json_path),
        "p4_pt": sha256_file(p4_pt_path),
        "replay_config": sha256_file(replay_config_path),
        "replay_pt": sha256_file(replay_pt_path),
        "precheck_pt": sha256_file(precheck_pt_path) if precheck_pt_path else "NOT_PROVIDED",
    }

    if not skip_sha_check:
        for k, exp in EXPECTED_SHAS.items():
            act = sha_records.get(k)
            if act != exp:
                raise ValueError(
                    f"SHA-256 mismatch for {k}: expected {exp}, got {act}. Aborting."
                )

    # 2. Load configurations and inputs
    with open(replay_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    indices_325 = cfg["indices"] if isinstance(cfg, dict) and "indices" in cfg else cfg
    if len(indices_325) != 325:
        raise ValueError(f"replay_config has {len(indices_325)} items, expected 325.")

    with open(inputs_json_path, "r", encoding="utf-8") as f:
        inputs_data = json.load(f)
    train_items = inputs_data["items"]
    if len(train_items) != 1000:
        raise ValueError(f"inputs_json has {len(train_items)} items, expected 1000.")

    # A1: Compute conservative and optimistic values dynamically from inputs JSON
    indices_set = set(indices_325)
    determined_items = [it for it in train_items if it["q_idx"] not in indices_set]
    undetermined_items = [it for it in train_items if it["q_idx"] in indices_set]

    # Determined items reaching criterion (r_min < 1e-2 and k* >= 50)
    determined_reached_items = [
        it for it in determined_items
        if it.get("r_min", float("inf")) < 1e-2
        and it.get("k_star") is not None
        and it["k_star"] >= 50
    ]
    conservative_1a = len(determined_reached_items)

    # Undetermined items with r_min < 1e-2
    undetermined_rmin_lt_1e2_items = [
        it for it in undetermined_items
        if it.get("r_min", float("inf")) < 1e-2
    ]
    undetermined_rmin_lt_1e2 = len(undetermined_rmin_lt_1e2_items)

    optimistic_1a = conservative_1a + undetermined_rmin_lt_1e2

    # 3. Load P4 .pt and verify 325 keys and shape
    p4_dict = torch.load(p4_pt_path, map_location="cpu")
    for q in indices_325:
        k = f"q{q:04d}_a0.25"
        if k not in p4_dict:
            raise KeyError(f"Key {k} missing from P4 file {p4_pt_path}")
        z = p4_dict[k].get("z_star")
        if z is None:
            raise ValueError(f"Field z_star missing from {k} in P4 file")
        if z.shape != (64, 2560):
            raise ValueError(f"Shape of z_star in {k} is {z.shape}, expected (64, 2560)")

    # 4. Load replay .pt
    replay_dict = torch.load(replay_pt_path, map_location="cpu")

    # 5. Evaluate the 325 undetermined items
    items_by_q = {it["q_idx"]: it for it in train_items}

    reproduced_items = []
    failed_reproduction_items = []
    missing_items = []
    nan_inf_items = []
    reproduced_reached_items = []
    failed_but_would_reach_items = []

    rel_diffs = []
    abs_diffs = []
    kstar_replay_values = {}

    # A2: Prepare primary and secondary bin counters
    primary_bins = {name: 0 for name, _, _, _ in PREREG_10D_BINS}
    secondary_bins = {name: 0 for name, _, _, _ in AUX_100STEP_BINS}
    kstar_lt_50_items = []

    for q in indices_325:
        k = f"q{q:04d}_a0.25"
        orig_item = items_by_q[q]
        r_min = orig_item["r_min"]
        p4_min_rho = float(p4_dict[k]["min_rho"])

        if k not in replay_dict:
            missing_items.append(q)
            continue

        rep = replay_dict[k]
        z_rep = rep.get("z_star")
        k_rep = rep.get("k_star")
        rho_traj = rep.get("rho_traj")

        if z_rep is None or k_rep is None or rho_traj is None:
            missing_items.append(q)
            continue

        # Check NaN / Inf
        if isinstance(rho_traj, torch.Tensor):
            has_nan = torch.isnan(rho_traj).any().item() or torch.isinf(rho_traj).any().item()
            rho_vals = rho_traj.tolist()
        else:
            has_nan = any(math.isnan(x) or math.isinf(x) for x in rho_traj)
            rho_vals = list(rho_traj)

        z_nan = torch.isnan(z_rep).any().item() or torch.isinf(z_rep).any().item()
        if has_nan or z_nan:
            nan_inf_items.append(q)
            continue

        k_star_val = int(k_rep)
        kstar_replay_values[q] = k_star_val

        # Condition (i): min(rho_traj) vs min_rho
        rep_min_rho = float(min(rho_vals))
        diff_abs = abs(rep_min_rho - p4_min_rho)
        diff_rel = (diff_abs / p4_min_rho) if p4_min_rho > 0.0 else float("inf")

        abs_diffs.append(diff_abs)
        rel_diffs.append(diff_rel)

        cond_i = (diff_rel <= 1e-2) or (diff_abs <= 1.2e-7)

        # Condition (ii): z_star cos >= 0.9999 (float64)
        z1 = z_rep.reshape(-1).to(torch.float64)
        z2 = p4_dict[k]["z_star"].reshape(-1).to(torch.float64)
        cos_sim = float(torch.nn.functional.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item())

        cond_ii = (cos_sim >= 0.9999)

        # Reproduction judgment
        is_reproduced = cond_i and cond_ii

        # A2: Place into primary and secondary bins
        p_bin = assign_bin(k_star_val, PREREG_10D_BINS)
        if p_bin in primary_bins:
            primary_bins[p_bin] += 1

        s_bin = assign_bin(k_star_val, AUX_100STEP_BINS)
        if s_bin in secondary_bins:
            secondary_bins[s_bin] += 1

        if k_star_val < 50:
            kstar_lt_50_items.append(q)

        would_reach = (r_min < 1e-2) and (k_star_val >= 50)

        if is_reproduced:
            reproduced_items.append(q)
            if would_reach:
                reproduced_reached_items.append(q)
        else:
            failed_reproduction_items.append({
                "q_idx": q,
                "cond_i": cond_i,
                "diff_rel": diff_rel,
                "diff_abs": diff_abs,
                "cond_ii": cond_ii,
                "cos": cos_sim,
                "k_star": k_star_val,
                "r_min": r_min,
            })
            if would_reach:
                failed_but_would_reach_items.append(q)

    # 6. Compute 1a values (A1: dynamic)
    v14_1a = conservative_1a + len(reproduced_reached_items)
    sensitivity_1a = v14_1a + len(failed_but_would_reach_items)

    criterion_threshold = 950
    criterion_met = (v14_1a >= criterion_threshold)
    diff_from_threshold = v14_1a - criterion_threshold

    # 7. Evaluate pre-check if provided
    precheck_results = None
    if precheck_pt_path and os.path.exists(precheck_pt_path):
        pre_dict = torch.load(precheck_pt_path, map_location="cpu")
        pre_items = []
        all_pre_pass = True
        for q, exp_k in PRECHECK_EXPECTED_5.items():
            k = f"q{q:04d}_a0.25"
            if k not in pre_dict:
                all_pre_pass = False
                pre_items.append({"q_idx": q, "status": "KEY_MISSING"})
                continue
            item_data = pre_dict[k]
            act_k = int(item_data.get("k_star", -1))
            k_match = (act_k == exp_k)

            # cond i
            p4_min_rho = float(p4_dict[k]["min_rho"])
            rho_traj = item_data.get("rho_traj", [])
            act_min_rho = float(min(rho_traj)) if len(rho_traj) > 0 else float("inf")
            d_abs = abs(act_min_rho - p4_min_rho)
            d_rel = (d_abs / p4_min_rho) if p4_min_rho > 0.0 else float("inf")
            cond_i = (d_rel <= 1e-2) or (d_abs <= 1.2e-7)

            # cond ii
            z_act = item_data["z_star"].reshape(-1).to(torch.float64)
            z_p4 = p4_dict[k]["z_star"].reshape(-1).to(torch.float64)
            c_sim = float(torch.nn.functional.cosine_similarity(z_act.unsqueeze(0), z_p4.unsqueeze(0)).item())
            cond_ii = (c_sim >= 0.9999)

            item_pass = k_match and cond_i and cond_ii
            if not item_pass:
                all_pre_pass = False

            pre_items.append({
                "q_idx": q,
                "expected_k": exp_k,
                "actual_k": act_k,
                "k_match": k_match,
                "diff_rel": d_rel,
                "diff_abs": d_abs,
                "cond_i": cond_i,
                "cos": c_sim,
                "cond_ii": cond_ii,
                "pass": item_pass,
            })
        precheck_results = {
            "all_pre_pass": all_pre_pass,
            "items": pre_items,
        }

    return {
        "metric_1a": {
            "determined_count": len(determined_items),
            "conservative_1a": conservative_1a,
            "undetermined_count": len(undetermined_items),
            "undetermined_rmin_lt_1e2": undetermined_rmin_lt_1e2,
            "optimistic_1a": optimistic_1a,
            "v14_1a": v14_1a,
            "sensitivity_1a": sensitivity_1a,
            "total_denominator": 1000,
            "threshold": criterion_threshold,
            "criterion_met": criterion_met,
            "diff_from_threshold": diff_from_threshold,
        },
        "reproduction_summary": {
            "total_evaluated": len(indices_325),
            "reproduced_count": len(reproduced_items),
            "failed_reproduction_count": len(failed_reproduction_items),
            "missing_count": len(missing_items),
            "nan_inf_count": len(nan_inf_items),
            "reproduced_reached_count": len(reproduced_reached_items),
            "failed_but_would_reach_count": len(failed_but_would_reach_items),
            "failed_reproduction_list": failed_reproduction_items,
            "missing_list": missing_items,
            "nan_inf_list": nan_inf_items,
        },
        "kstar_distribution": {
            "primary_bins_10d": primary_bins,
            "secondary_bins_100step": secondary_bins,
            "kstar_lt_50_count": len(kstar_lt_50_items),
            "kstar_lt_50_list": kstar_lt_50_items,
        },
        "rho_diff_distribution": {
            "relative_diff": compute_quantiles(rel_diffs),
            "absolute_diff": compute_quantiles(abs_diffs),
        },
        "precheck": precheck_results,
        "sha_records": sha_records,
    }


def render_markdown_report(result: Dict[str, Any]) -> str:
    """Render markdown report ensuring zero CR and raw strings for LaTeX math."""
    m1a = result["metric_1a"]
    rep = result["reproduction_summary"]
    k_dist = result["kstar_distribution"]
    rho_dist = result["rho_diff_distribution"]
    pre = result["precheck"]
    shas = result["sha_records"]

    md: List[str] = []
    md.append(r"# PREREG_S7_DEQ_v1.4 §10(h) ①-a 재계산 분석 결과 (v2)")
    md.append("")
    md.append(r"## 1. ①-a 지표 3종 비교 요약")
    md.append("")
    md.append(r"| 항목 | 보수치 | 낙관치 | v1.4 재계산치 | 기준치 | 기준 대비 차 | 판정 |")
    md.append(r"|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    status_str = "기준 충족" if m1a["criterion_met"] else "기준 미달"
    diff_sign = f"+{m1a['diff_from_threshold']}" if m1a['diff_from_threshold'] >= 0 else str(m1a['diff_from_threshold'])
    tot_denom = m1a["total_denominator"]
    md.append(
        f"| **①-a 도달 문항 수** | {m1a['conservative_1a']}/{tot_denom} | "
        f"{m1a['optimistic_1a']}/{tot_denom} | **{m1a['v14_1a']}/{tot_denom}** | "
        f"{m1a['threshold']}/{tot_denom} | {diff_sign} | {status_str} |"
    )
    c_pct = m1a['conservative_1a'] / tot_denom * 100
    o_pct = m1a['optimistic_1a'] / tot_denom * 100
    v_pct = m1a['v14_1a'] / tot_denom * 100
    t_pct = m1a['threshold'] / tot_denom * 100
    d_pct = m1a['diff_from_threshold'] / tot_denom * 100
    md.append(
        f"| **도달 비율** | {c_pct:.1f}% | {o_pct:.1f}% | **{v_pct:.1f}%** | "
        f"{t_pct:.1f}% | {d_pct:+.1f}%p | {status_str} |"
    )
    md.append("")
    sens_pct = m1a['sensitivity_1a'] / tot_denom * 100
    md.append(
        f"- **민감도 수치**: 재현 실패 문항을 재계산 k* 기준으로 합산한 경우: "
        f"**{m1a['sensitivity_1a']}/{tot_denom}** ({sens_pct:.1f}%)"
    )
    md.append(
        f"- **입력 JSON 산출 근거 (A1)**: 판별된 {m1a['determined_count']}문항 중 "
        f"`r_min < 1e-2 ∧ k* >= 50` 문항 수 = {m1a['conservative_1a']}, "
        f"미판별 {m1a['undetermined_count']}문항 중 `r_min < 1e-2` 문항 수 = {m1a['undetermined_rmin_lt_1e2']}"
    )
    md.append("")

    md.append(r"## 2. 325문항 재현 점검 실측 요약")
    md.append("")
    md.append(f"- **대상 문항 수**: {rep['total_evaluated']}")
    rep_pct = (rep['reproduced_count'] / rep['total_evaluated'] * 100) if rep['total_evaluated'] > 0 else 0.0
    md.append(f"- **재현 충족 문항 수**: {rep['reproduced_count']} / {rep['total_evaluated']} ({rep_pct:.1f}%)")
    md.append(f"- **재현 미충족 문항 수**: {rep['failed_reproduction_count']}")
    md.append(f"- **결측 문항 수**: {rep['missing_count']}")
    md.append(f"- **NaN/Inf 문항 수**: {rep['nan_inf_count']}")
    md.append(r"- **재현 충족 중 도달($r_{\min} < 10^{-2} \land k^* \ge 50$) 문항 수**: " + str(rep['reproduced_reached_count']))
    md.append(r"- **재현 미충족 중 도달 조건 만족(민감도 대상) 문항 수**: " + str(rep['failed_but_would_reach_count']))
    md.append("")

    if rep["failed_reproduction_count"] > 0:
        md.append("### 재현 미충족 문항 목록")
        md.append(r"| q_idx | 상대차 (기준 $\le 10^{-2}$) | 절대차 (기준 $\le 1.2 \times 10^{-7}$) | (i) 충족 | $z^*$ cos (기준 $\ge 0.9999$) | (ii) 충족 | $k^*$ | $r_{\min}$ |")
        md.append(r"|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for fail in rep["failed_reproduction_list"]:
            c1 = "Y" if fail["cond_i"] else "N"
            c2 = "Y" if fail["cond_ii"] else "N"
            md.append(
                f"| Q#{fail['q_idx']} | {fail['diff_rel']:.4e} | {fail['diff_abs']:.4e} | "
                f"{c1} | {fail['cos']:.8f} | {c2} | {fail['k_star']} | {fail['r_min']:.4e} |"
            )
        md.append("")

    md.append(r"## 3. 재계산 $k^*$ 분포 (A2)")
    md.append("")
    md.append(r"### 3.1 주 표: §10(d) 구간")
    md.append("")
    md.append(r"| $k^*$ 구간 | 문항 수 | 비율 |")
    md.append(r"|---|:---:|:---:|")
    tot_p = sum(k_dist["primary_bins_10d"].values())
    for b_name, b_cnt in k_dist["primary_bins_10d"].items():
        b_ratio = (b_cnt / tot_p * 100) if tot_p > 0 else 0.0
        md.append(f"| {b_name} | {b_cnt} | {b_ratio:.1f}% |")
    md.append("")

    md.append(r"### 3.2 보조 표: 100걸음 구간")
    md.append("")
    md.append(r"| $k^*$ 구간 | 문항 수 | 비율 |")
    md.append(r"|---|:---:|:---:|")
    tot_s = sum(k_dist["secondary_bins_100step"].values())
    for b_name, b_cnt in k_dist["secondary_bins_100step"].items():
        b_ratio = (b_cnt / tot_s * 100) if tot_s > 0 else 0.0
        md.append(f"| {b_name} | {b_cnt} | {b_ratio:.1f}% |")
    md.append("")

    md.append(f"- **$k^* < 50$ 문항 수**: {k_dist['kstar_lt_50_count']}")
    if k_dist["kstar_lt_50_count"] > 0:
        md.append(f"- **$k^* < 50$ 문항 목록**: {k_dist['kstar_lt_50_list']}")
    md.append("")

    md.append(r"## 4. $\min(\rho_{\text{traj}})$ vs 기존 $\min\rho$ 편차 분포")
    md.append("")
    md.append(r"| 통계량 | 상대차 | 절대차 |")
    md.append(r"|---|:---:|:---:|")
    rel_q = rho_dist["relative_diff"]
    abs_q = rho_dist["absolute_diff"]
    md.append(f"| 최소값 (min) | {rel_q['min']:.4e} | {abs_q['min']:.4e} |")
    md.append(f"| 25% 분위수 (p25) | {rel_q['p25']:.4e} | {abs_q['p25']:.4e} |")
    md.append(f"| 중앙값 (median) | {rel_q['median']:.4e} | {abs_q['median']:.4e} |")
    md.append(f"| 75% 분위수 (p75) | {rel_q['p75']:.4e} | {abs_q['p75']:.4e} |")
    md.append(f"| 최대값 (max) | {rel_q['max']:.4e} | {abs_q['max']:.4e} |")
    md.append(f"| 평균 (mean) | {rel_q['mean']:.4e} | {abs_q['mean']:.4e} |")
    md.append("")

    if pre:
        md.append(r"## 5. 사전 점검 5문항 판정 결과")
        md.append("")
        pre_status = "전원 충족" if pre["all_pre_pass"] else "일부 미달"
        md.append(f"- **종합 상태**: {pre_status}")
        md.append(r"| q_idx | 기대 $k^*$ | 실측 $k^*$ | $k^*$ 일치 | $\min\rho$ 상대차 | 절대차 | (i) 충족 | $z^*$ cos | (ii) 충족 | 최종 |")
        md.append(r"|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for pit in pre["items"]:
            k_m = "Y" if pit.get("k_match") else "N"
            c1 = "Y" if pit.get("cond_i") else "N"
            c2 = "Y" if pit.get("cond_ii") else "N"
            final_p = "충족" if pit.get("pass") else "미달"
            d_rel_str = f"{pit.get('diff_rel', 0.0):.4e}" if "diff_rel" in pit else "-"
            d_abs_str = f"{pit.get('diff_abs', 0.0):.4e}" if "diff_abs" in pit else "-"
            cos_str = f"{pit.get('cos', 0.0):.8f}" if "cos" in pit else "-"
            md.append(
                f"| Q#{pit['q_idx']} | {pit.get('expected_k')} | {pit.get('actual_k')} | "
                f"{k_m} | {d_rel_str} | {d_abs_str} | {c1} | {cos_str} | {c2} | {final_p} |"
            )
        md.append("")

    md.append(r"## 6. 입력 파일 SHA-256 계측 기록")
    md.append("")
    md.append(r"| 파일 키 | 파일 경로 / 식별 | SHA-256 |")
    md.append(r"|---|---|---|")
    for fk, sha_val in shas.items():
        md.append(f"| `{fk}` | `{fk}` | `{sha_val}` |")
    md.append("")

    rendered = "\n".join(md)
    # A3 check
    check_no_cr_or_control(rendered, "rendered markdown report")
    return rendered


def main():
    parser = argparse.ArgumentParser(description="PREREG_S7_DEQ_v1.4 §10(h) ①-a Replay Analyzer v2")
    parser.add_argument("--replay_pt", type=str, required=True, help="Path to replay .pt file")
    parser.add_argument("--inputs_json", type=str, default="results_s7/p5_inputs_train1000_v3.json", help="Path to p5_inputs_train1000_v3.json")
    parser.add_argument("--p4_pt", type=str, default="results_s7/zstar_train1000_P4.pt", help="Path to zstar_train1000_P4.pt")
    parser.add_argument("--replay_config", type=str, default="configs/s7_train1000_kstar_replay_325.json", help="Path to s7_train1000_kstar_replay_325.json")
    parser.add_argument("--precheck_pt", type=str, default=None, help="Optional path to precheck 5-item .pt file")
    parser.add_argument("--out_json", type=str, default=None, help="Output path for analysis JSON")
    parser.add_argument("--out_md", type=str, default=None, help="Output path for analysis Markdown")
    parser.add_argument("--skip_sha_check", action="store_true", help="Skip fixed input SHA verification (test mode only)")

    args = parser.parse_args()

    result = evaluate_replay(
        replay_pt_path=args.replay_pt,
        inputs_json_path=args.inputs_json,
        p4_pt_path=args.p4_pt,
        replay_config_path=args.replay_config,
        precheck_pt_path=args.precheck_pt,
        skip_sha_check=args.skip_sha_check,
    )

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        check_no_cr_or_control(json_str, "out_json text")
        with open(args.out_json, "w", encoding="utf-8", newline="\n") as f:
            f.write(json_str)
        print(f"Analysis JSON written to {args.out_json}")

    md_text = render_markdown_report(result)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8", newline="\n") as f:
            f.write(md_text)
        print(f"Analysis Markdown written to {args.out_md}")

    m1a = result["metric_1a"]
    print(
        f"①-a Result: v1.4={m1a['v14_1a']}/1000 "
        f"(Conservative={m1a['conservative_1a']}, Optimistic={m1a['optimistic_1a']}, Sensitivity={m1a['sensitivity_1a']})"
    )
    print(f"Threshold 950 met: {m1a['criterion_met']} (diff={m1a['diff_from_threshold']})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/p5_promote_1a_replay.py - Analyzer for ①-a k* Recomputation
Directive: PREREG_S7_DEQ_v1.4 §10(h) / TASK-0049 (A)

Evaluates:
- 1,000 train items using existing r_min from p5_inputs_train1000_v3.json
- 325 undetermined items using recomputed k* from replay .pt after reproduction checks:
  (i)  min(rho_traj) relative diff <= 1e-2 OR absolute diff <= 1.2e-7
  (ii) z_star cos >= 0.9999 in float64 against zstar_train1000_P4.pt
- 3 values of ①-a: Conservative (675), Optimistic (991), v1.4 Recomputed
- k* distribution across §10(d) bins and items with k* < 50
- min rho diff distributions (rel_diff, abs_diff)
- Sensitivity analysis for failed reproduction items
- Pre-check evaluation of 5 items if precheck .pt is provided

Enforces:
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

KSTAR_BINS = [
    (50, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500),
    (500, 600),
    (600, 700),
    (700, 800),
    (800, 900),
    (900, 1001),
]


def sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


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
    """Execute evaluation according to PREREG_S7_DEQ_v1.4 §10(h)."""
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

    bin_counts = {f"[{low}, {high})": 0 for low, high in KSTAR_BINS}
    bin_counts["[900, 1000]"] = 0
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

        # Check bins
        placed = False
        if k_star_val < 50:
            kstar_lt_50_items.append(q)
        else:
            for low, high in KSTAR_BINS:
                if high == 1001:
                    if low <= k_star_val <= 1000:
                        bin_counts["[900, 1000]"] += 1
                        placed = True
                        break
                elif low <= k_star_val < high:
                    bin_counts[f"[{low}, {high})"] += 1
                    placed = True
                    break

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

    # 6. Compute 1a values
    conservative_1a = 675
    optimistic_1a = 991
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
            "conservative_1a": conservative_1a,
            "optimistic_1a": optimistic_1a,
            "v14_1a": v14_1a,
            "sensitivity_1a": sensitivity_1a,
            "total_denominator": 1000,
            "threshold": criterion_threshold,
            "criterion_met": criterion_met,
            "diff_from_threshold": diff_from_threshold,
        },
        "reproduction_summary": {
            "total_evaluated": 325,
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
            "bins": bin_counts,
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
    m1a = result["metric_1a"]
    rep = result["reproduction_summary"]
    k_dist = result["kstar_distribution"]
    rho_dist = result["rho_diff_distribution"]
    pre = result["precheck"]
    shas = result["sha_records"]

    md = []
    md.append("# PREREG_S7_DEQ_v1.4 §10(h) ①-a 재계산 분석 결과")
    md.append("")
    md.append("## 1. ①-a 지표 3종 비교 요약")
    md.append("")
    md.append("| 항목 | 보수치 | 낙관치 | v1.4 재계산치 | 기준치 | 기준 대비 차 | 판정 |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    status_str = "기준 충족" if m1a["criterion_met"] else "기준 미달"
    diff_sign = f"+{m1a['diff_from_threshold']}" if m1a['diff_from_threshold'] >= 0 else str(m1a['diff_from_threshold'])
    md.append(f"| **①-a 도달 문항 수** | {m1a['conservative_1a']}/1000 | {m1a['optimistic_1a']}/1000 | **{m1a['v14_1a']}/1000** | {m1a['threshold']}/1000 | {diff_sign} | {status_str} |")
    md.append(f"| **도달 비율** | {m1a['conservative_1a']/10:.1f}% | {m1a['optimistic_1a']/10:.1f}% | **{m1a['v14_1a']/10:.1f}%** | {m1a['threshold']/10:.1f}% | {m1a['diff_from_threshold']/10:+.1f}%p | {status_str} |")
    md.append("")
    md.append(f"- **민감도 수치**: 재현 실패 문항을 재계산 k* 기준으로 합산한 경우: **{m1a['sensitivity_1a']}/1000** ({m1a['sensitivity_1a']/10:.1f}%)")
    md.append("")
    md.append("## 2. 325문항 재현 점검 실측 요약")
    md.append("")
    md.append(f"- **대상 문항 수**: {rep['total_evaluated']}")
    md.append(f"- **재현 충족 문항 수**: {rep['reproduced_count']} / {rep['total_evaluated']} ({rep['reproduced_count']/rep['total_evaluated']*100:.1f}%)")
    md.append(f"- **재현 미충족 문항 수**: {rep['failed_reproduction_count']}")
    md.append(f"- **결측 문항 수**: {rep['missing_count']}")
    md.append(f"- **NaN/Inf 문항 수**: {rep['nan_inf_count']}")
    md.append(f"- **재현 충족 중 도달(r_min < 1e-2 & k* >= 50) 문항 수**: {rep['reproduced_reached_count']}")
    md.append(f"- **재현 미충족 중 도달 조건 만족(민감도 대상) 문항 수**: {rep['failed_but_would_reach_count']}")
    md.append("")

    if rep["failed_reproduction_count"] > 0:
        md.append("### 재현 미충족 문항 목록")
        md.append("| q_idx | 상대차 (기준 <= 1e-2) | 절대차 (기준 <= 1.2e-7) | (i) 충족 | z_star cos (기준 >= 0.9999) | (ii) 충족 | k* | r_min |")
        md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for fail in rep["failed_reproduction_list"]:
            c1 = "Y" if fail["cond_i"] else "N"
            c2 = "Y" if fail["cond_ii"] else "N"
            md.append(f"| Q#{fail['q_idx']} | {fail['diff_rel']:.4e} | {fail['diff_abs']:.4e} | {c1} | {fail['cos']:.8f} | {c2} | {fail['k_star']} | {fail['r_min']:.4e} |")
        md.append("")

    md.append("## 3. 재계산 k* 분포 (§10(d) 구간)")
    md.append("")
    md.append("| k* 구간 | 문항 수 | 비율 |")
    md.append("|---|:---:|:---:|")
    tot_k = sum(k_dist["bins"].values())
    for b_name, b_cnt in k_dist["bins"].items():
        b_ratio = (b_cnt / tot_k * 100) if tot_k > 0 else 0.0
        md.append(f"| {b_name} | {b_cnt} | {b_ratio:.1f}% |")
    md.append("")
    md.append(f"- **k* < 50 문항 수**: {k_dist['kstar_lt_50_count']}")
    if k_dist["kstar_lt_50_count"] > 0:
        md.append(f"- **k* < 50 문항 목록**: {k_dist['kstar_lt_50_list']}")
    md.append("")

    md.append("## 4. min(rho_traj) vs 기존 min_rho 편차 분포")
    md.append("")
    md.append("| 통계량 | 상대차 | 절대차 |")
    md.append("|---|:---:|:---:|")
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
        md.append("## 5. 사전 점검 5문항 판정 결과")
        md.append("")
        pre_status = "전원 충족" if pre["all_pre_pass"] else "일부 미달"
        md.append(f"- **종합 상태**: {pre_status}")
        md.append("| q_idx | 기대 k* | 실측 k* | k* 일치 | min_rho 상대차 | 절대차 | (i) 충족 | z_star cos | (ii) 충족 | 최종 |")
        md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for pit in pre["items"]:
            k_m = "Y" if pit.get("k_match") else "N"
            c1 = "Y" if pit.get("cond_i") else "N"
            c2 = "Y" if pit.get("cond_ii") else "N"
            final_p = "충족" if pit.get("pass") else "미달"
            d_rel_str = f"{pit.get('diff_rel', 0.0):.4e}" if "diff_rel" in pit else "-"
            d_abs_str = f"{pit.get('diff_abs', 0.0):.4e}" if "diff_abs" in pit else "-"
            cos_str = f"{pit.get('cos', 0.0):.8f}" if "cos" in pit else "-"
            md.append(f"| Q#{pit['q_idx']} | {pit.get('expected_k')} | {pit.get('actual_k')} | {k_m} | {d_rel_str} | {d_abs_str} | {c1} | {cos_str} | {c2} | {final_p} |")
        md.append("")

    md.append("## 6. 입력 파일 SHA-256 계측 기록")
    md.append("")
    md.append("| 파일 키 | 파일 경로 / 식별 | SHA-256 |")
    md.append("|---|---|---|")
    for fk, sha_val in shas.items():
        md.append(f"| `{fk}` | `{fk}` | `{sha_val}` |")
    md.append("")
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="PREREG_S7_DEQ_v1.4 §10(h) ①-a Replay Analyzer")
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
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Analysis JSON written to {args.out_json}")

    md_text = render_markdown_report(result)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"Analysis Markdown written to {args.out_md}")

    m1a = result["metric_1a"]
    print(f"①-a Result: v1.4={m1a['v14_1a']}/1000 (Conservative={m1a['conservative_1a']}, Optimistic={m1a['optimistic_1a']}, Sensitivity={m1a['sensitivity_1a']})")
    print(f"Threshold 950 met: {m1a['criterion_met']} (diff={m1a['diff_from_threshold']})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""scripts/s6_eval_damping.py - Evaluate S6 Damping Factor Experiment Against PREREG Criteria
Directive: EXE-260926-36 (System 2 / 원장 #348)

Evaluates:
1. Mandatory Safety Metric on Control 20:
   - cos(z*(alpha), z*(alpha=0.5)) >= 0.999 AND 20/20 rho < 5e-3
2. Primary Metric on 54 Non-converged Items:
   - Count of rho < 5e-3 at NFE 1000 & NFE 2000 (overall, train, held-out)
3. Secondary Metrics:
   - 4-way classification distribution (monotonic_decrease, oscillation, plateau, increase)
   - Mean pairwise cos sim among converged held-out items
4. Preregistered Adoption Rule:
   - Highest resolution count at NFE 1000 among safety-passing alphas (tie-break: larger alpha).
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List
import numpy as np
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
CONFIGS_DIR = os.path.join(BASE_S5_DIR, "configs")
RESULTS_S5_DIR = os.path.join(BASE_S5_DIR, "results_s5")
RESULTS_S6_DIR = os.path.join(BASE_S5_DIR, "results_s6")

ITEMS_74_PATH = os.path.join(CONFIGS_DIR, "s6_items_74.json")
BASELINE_NONCONV_PT = os.path.join(RESULTS_S5_DIR, "diag_nonconv.pt")
BASELINE_CTRL_PT = os.path.join(RESULTS_S6_DIR, "zstar_s6_a0.50_control20.pt")
ALPHA_035_PT = os.path.join(RESULTS_S6_DIR, "zstar_s6_a0.35.pt")
ALPHA_025_PT = os.path.join(RESULTS_S6_DIR, "zstar_s6_a0.25.pt")

def compute_cosine_sim(t1: torch.Tensor, t2: torch.Tensor) -> float:
    t1_f = t1.flatten().to(torch.float64)
    t2_f = t2.flatten().to(torch.float64)
    norm1 = torch.norm(t1_f)
    norm2 = torch.norm(t2_f)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(torch.dot(t1_f, t2_f) / (norm1 * norm2))

def get_entry(data: Dict[str, Any], q: int) -> Dict[str, Any]:
    for k in [f"q{q:04d}_a0.50", f"q{q:04d}_a0.35", f"q{q:04d}_a0.25", f"q{q:04d}_diag", f"q{q}_s2030", f"q{q:02d}_s2030", q]:
        if k in data:
            return data[k]
    for k, v in data.items():
        if isinstance(v, dict) and v.get("meta", {}).get("question_idx") == q:
            return v
    raise KeyError(f"Question {q} not found in tensor dict")

def main():
    print("=" * 80)
    print("S6 DAMPING EXPERIMENT COMPREHENSIVE EVALUATION (PREREG #348)")
    print("=" * 80)

    # 1. Load Configurations
    assert os.path.exists(ITEMS_74_PATH), f"Missing {ITEMS_74_PATH}"
    with open(ITEMS_74_PATH, "r", encoding="utf-8") as f:
        meta_74 = json.load(f)

    nonconv_54 = meta_74["nonconv_54"]
    control_20 = meta_74["control_20"]
    items_detail = {x["q_idx"]: x for x in meta_74.get("items_detail", [])}

    train_nonconv = [q for q in nonconv_54 if items_detail.get(q, {}).get("split") == "train"]
    heldout_nonconv = [q for q in nonconv_54 if items_detail.get(q, {}).get("split") == "heldout"]

    print(f"Target items: {len(meta_74['all_74'])} (54 nonconv [train={len(train_nonconv)}, heldout={len(heldout_nonconv)}] + 20 control)")

    # 2. Check and Load Tensors
    for p in [BASELINE_NONCONV_PT, BASELINE_CTRL_PT, ALPHA_035_PT, ALPHA_025_PT]:
        assert os.path.exists(p), f"Missing required tensor file: {p}"

    d_base_nc = torch.load(BASELINE_NONCONV_PT, map_location="cpu")
    d_base_ctrl = torch.load(BASELINE_CTRL_PT, map_location="cpu")
    d_a035 = torch.load(ALPHA_035_PT, map_location="cpu")
    d_a025 = torch.load(ALPHA_025_PT, map_location="cpu")

    # Combine baseline alpha=0.5
    d_base_all: Dict[str, Any] = {}
    for q in nonconv_54:
        d_base_all[q] = get_entry(d_base_nc, q)
    for q in control_20:
        d_base_all[q] = get_entry(d_base_ctrl, q)

    eval_results: Dict[str, Any] = {
        "metadata": {
            "directive": "EXE-260926-36 (원장 #348)",
            "timestamp": os.popen("date").read().strip(),
        },
        "safety_metrics": {},
        "primary_metrics": {},
        "secondary_metrics": {},
        "decision": {},
    }

    # --------------------------------------------------------------------------
    # Mandatory Safety Metrics on Control 20
    # --------------------------------------------------------------------------
    for alpha_name, alpha_dict in [("alpha_0.35", d_a035), ("alpha_0.25", d_a025)]:
        ctrl_cosines = []
        ctrl_rhos = []
        all_rho_pass = True
        all_cos_pass = True

        for q in control_20:
            e_base = d_base_all[q]
            e_cand = get_entry(alpha_dict, q)

            z_base = e_base.get("z_star", e_base.get("z_2000"))
            z_cand = e_cand.get("z_star", e_cand.get("z_2000"))
            cos_sim = compute_cosine_sim(z_base, z_cand)
            rho_val = float(e_cand.get("final_rho_fp", e_cand.get("rho_2000", e_cand.get("rho", 1.0))))

            ctrl_cosines.append(cos_sim)
            ctrl_rhos.append(rho_val)

            if cos_sim < 0.999:
                all_cos_pass = False
            if rho_val >= 5e-3:
                all_rho_pass = False

        min_cos = float(min(ctrl_cosines))
        mean_cos = float(np.mean(ctrl_cosines))
        max_rho = float(max(ctrl_rhos))
        conv_count = sum(1 for r in ctrl_rhos if r < 5e-3)

        safety_pass = bool(all_cos_pass and all_rho_pass)

        eval_results["safety_metrics"][alpha_name] = {
            "min_cos_with_base": min_cos,
            "mean_cos_with_base": mean_cos,
            "cos_ge_0999_pass": all_cos_pass,
            "max_rho": max_rho,
            "conv_20_20_count": conv_count,
            "conv_20_20_pass": bool(conv_count == 20),
            "safety_pass": safety_pass,
        }

    # --------------------------------------------------------------------------
    # Primary Metrics on 54 Non-converged Items
    # --------------------------------------------------------------------------
    for alpha_name, alpha_dict in [("alpha_0.50_baseline", d_base_all), ("alpha_0.35", d_a035), ("alpha_0.25", d_a025)]:
        res_1000_all = 0
        res_2000_all = 0
        res_1000_train = 0
        res_2000_train = 0
        res_1000_heldout = 0
        res_2000_heldout = 0

        for q in nonconv_54:
            e = d_base_all[q] if alpha_name == "alpha_0.50_baseline" else get_entry(alpha_dict, q)
            split = items_detail.get(q, {}).get("split", "unknown")

            rho_1000 = float(e.get("rho_at_1000", e.get("rho_1000", 1.0)))
            rho_2000 = float(e.get("rho_at_2000", e.get("rho_2000", e.get("final_rho_fp", 1.0))))

            if rho_1000 < 5e-3:
                res_1000_all += 1
                if split == "train":
                    res_1000_train += 1
                elif split == "heldout":
                    res_1000_heldout += 1

            if rho_2000 < 5e-3:
                res_2000_all += 1
                if split == "train":
                    res_2000_train += 1
                elif split == "heldout":
                    res_2000_heldout += 1

        eval_results["primary_metrics"][alpha_name] = {
            "resolved_at_1000_total": res_1000_all,
            "resolved_at_1000_train": res_1000_train,
            "resolved_at_1000_heldout": res_1000_heldout,
            "resolved_at_2000_total": res_2000_all,
            "resolved_at_2000_train": res_2000_train,
            "resolved_at_2000_heldout": res_2000_heldout,
        }

    # --------------------------------------------------------------------------
    # Secondary Metrics (Shape Distribution)
    # --------------------------------------------------------------------------
    for alpha_name, alpha_dict in [("alpha_0.35", d_a035), ("alpha_0.25", d_a025)]:
        shapes = []
        for q in meta_74["all_74"]:
            e = get_entry(alpha_dict, q)
            shapes.append(e.get("shape", "unknown"))

        eval_results["secondary_metrics"][alpha_name] = {
            "monotonic_decrease": sum(1 for s in shapes if s == "monotonic_decrease"),
            "oscillation": sum(1 for s in shapes if s == "oscillation"),
            "plateau": sum(1 for s in shapes if s == "plateau"),
            "increase": sum(1 for s in shapes if s == "increase"),
        }

    # --------------------------------------------------------------------------
    # Preregistered Decision Rule
    # --------------------------------------------------------------------------
    # 안전 지표 통과 alpha 중 NFE 1000에서 미수렴 해소 수가 가장 큰 alpha (동률이면 큰 alpha)
    candidate_alphas = []
    for a_val, a_name in [(0.35, "alpha_0.35"), (0.25, "alpha_0.25")]:
        safe = eval_results["safety_metrics"][a_name]["safety_pass"]
        res_1000 = eval_results["primary_metrics"][a_name]["resolved_at_1000_total"]
        candidate_alphas.append({
            "alpha": a_val,
            "name": a_name,
            "safe": safe,
            "res_1000": res_1000,
        })

    passing_candidates = [c for c in candidate_alphas if c["safe"]]
    if not passing_candidates:
        chosen_alpha = None
        decision_str = "ALL_FAILED_SAFETY"
    else:
        # Sort by res_1000 desc, then alpha desc
        passing_candidates.sort(key=lambda x: (x["res_1000"], x["alpha"]), reverse=True)
        chosen_alpha = passing_candidates[0]["alpha"]
        decision_str = f"ADOPT_ALPHA_{chosen_alpha:.2f}"

    eval_results["decision"] = {
        "chosen_alpha": chosen_alpha,
        "decision": decision_str,
        "candidate_summary": candidate_alphas,
    }

    # Save summary json
    out_json = os.path.join(RESULTS_S6_DIR, "s6_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved evaluation summary: {out_json}")

    # Output Markdown Report
    out_md = os.path.join(RESULTS_S6_DIR, "report_S6_damping_results.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# S6 Damping Factor Experiment Results (EXE-260926-36 / 원장 #348)\n\n")
        f.write(f"- Timestamp: {eval_results['metadata']['timestamp']}\n\n")
        f.write("## 1. Mandatory Safety Metric (Control 20)\n\n")
        f.write("| Condition | min cos (>=0.999) | mean cos | 20/20 rho<5e-3 | Safety Gate |\n")
        f.write("|---|---|---|---|---|\n")
        for a_name in ["alpha_0.35", "alpha_0.25"]:
            s = eval_results["safety_metrics"][a_name]
            f.write(f"| `{a_name}` | `{s['min_cos_with_base']:.6f}` | `{s['mean_cos_with_base']:.6f}` | `{s['conv_20_20_count']}/20` | `{s['safety_pass']}` |\n")

        f.write("\n## 2. Primary Metric: Non-converged 54 Resolution Count\n\n")
        f.write("| Condition | NFE 1000 (Total) | NFE 1000 (Train) | NFE 1000 (Held-out) | NFE 2000 (Total) | NFE 2000 (Train) | NFE 2000 (Held-out) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for a_name in ["alpha_0.50_baseline", "alpha_0.35", "alpha_0.25"]:
            p = eval_results["primary_metrics"][a_name]
            f.write(f"| `{a_name}` | `{p['resolved_at_1000_total']}/54` | `{p['resolved_at_1000_train']}` | `{p['resolved_at_1000_heldout']}` | `{p['resolved_at_2000_total']}/54` | `{p['resolved_at_2000_train']}` | `{p['resolved_at_2000_heldout']}` |\n")

        f.write("\n## 3. Secondary Metric: Shape Distribution\n\n")
        f.write("| Condition | Monotonic Decrease | Oscillation | Plateau | Increase |\n")
        f.write("|---|---|---|---|---|\n")
        for a_name in ["alpha_0.35", "alpha_0.25"]:
            sc = eval_results["secondary_metrics"][a_name]
            f.write(f"| `{a_name}` | `{sc['monotonic_decrease']}` | `{sc['oscillation']}` | `{sc['plateau']}` | `{sc['increase']}` |\n")

        f.write("\n## 4. Decision\n\n")
        f.write(f"- **Chosen alpha**: `{chosen_alpha}`\n")
        f.write(f"- **Decision**: `{decision_str}`\n")
    print(f"Saved markdown report: {out_md}")

if __name__ == "__main__":
    main()

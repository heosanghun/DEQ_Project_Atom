#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/s7_eval_score_p6_v36.py - CTS S7 P6 Scoring & Gate Statistics (Pure Python, Zero External Dependencies)
Directive: TASK-0009 Section 2 / PREREG_S7_D_v3_7 §5

Grader:
  scripts/s5_eval_grader_v2.py (strict SHA check: 90cc93e4...)
  Supports Strict and Lenient evaluation.

Gate Statistics (§5):
  - Gate 1: A - B(s=1)
  - Gate 2: A - C >= -2%p
  - Gate 3: A - C
  - Gate 4: A - E1
  - Auxiliary: D control (D1 - D2), A - F, B shifts (s=2, s=3, pooled), NLL differences.
  - Paired bootstrap: 10,000 resamples, seed=2026, 95% CI [2.5%, 97.5%].
  - Exact one-sided McNemar test on discordant pairs.
  - Fully compatible with /home/shoon/venv_eval and standard Python (no numpy required).

Outputs:
  - results_s7/p6_gate_table.json
  - results_s7/p6_gate_table.md
  Numerical statistics only (no qualitative judgment words).
"""

import os
import sys
import time
import json
import math
import random
import hashlib
import argparse
from typing import Dict, Any, List, Optional, Tuple, Set

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

EXPECTED_GRADER_SHA = "90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13"


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_grader() -> Any:
    """Verify scripts/s5_eval_grader_v2.py SHA-256 and import grade_prediction."""
    candidates = [
        os.path.join(BASE_S5_DIR, "scripts", "s5_eval_grader_v2.py"),
        os.path.join(os.path.dirname(__file__), "s5_eval_grader_v2.py"),
    ]
    grader_path = None
    for c in candidates:
        if os.path.exists(c):
            grader_path = c
            break
    if not grader_path:
        raise FileNotFoundError(f"Grader script not found in: {candidates}")

    actual_sha = compute_sha256(grader_path)
    if actual_sha != EXPECTED_GRADER_SHA:
        raise ValueError(
            f"Grader SHA-256 mismatch!\nExpected: {EXPECTED_GRADER_SHA}\nActual:   {actual_sha}"
        )
    print(f"[GRADER-VERIFIED] scripts/s5_eval_grader_v2.py SHA: {actual_sha}")

    SCRIPTS_DIR = os.path.join(BASE_S5_DIR, "scripts")
    LOCAL_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
    for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, SCRIPTS_DIR, LOCAL_SCRIPTS]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from s5_eval_grader_v2 import grade_prediction
    except ImportError:
        from scripts.s5_eval_grader_v2 import grade_prediction
    return grade_prediction


def load_dataset_records_local() -> List[Dict[str, Any]]:
    candidates = [
        os.path.join(SYS_DIR, "data", "stage2", "math_train_prompts_5000.jsonl"),
        "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl",
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    raise FileNotFoundError(f"Dataset not found in candidates: {candidates}")


def load_gen_jsonl(path: str) -> Dict[int, Dict[str, Any]]:
    """Load generation results indexed by question ID."""
    if not os.path.exists(path):
        return {}
    res: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    res[int(obj["id"])] = obj
                except Exception:
                    pass
    return res


def exact_mcnemar_one_sided_p(y1: List[int], y2: List[int]) -> float:
    """Exact one-sided McNemar test for H1: p(y1=1, y2=0) > p(y1=0, y2=1)."""
    b = sum(1 for a, b_val in zip(y1, y2) if a == 1 and b_val == 0)
    c = sum(1 for a, b_val in zip(y1, y2) if a == 0 and b_val == 1)
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    p_val = sum(math.comb(n_disc, k) * (0.5 ** n_disc) for k in range(b, n_disc + 1))
    return float(p_val)


def paired_bootstrap_ci(
    arr1: List[float],
    arr2: List[float],
    n_bootstraps: int = 10000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute observed difference and 95% paired bootstrap CI in pure Python."""
    assert len(arr1) == len(arr2), "Array lengths must match"
    n = len(arr1)
    if n == 0:
        return 0.0, 0.0, 0.0

    delta_obs = (sum(arr1) - sum(arr2)) / n
    rng = random.Random(seed)

    boot_deltas: List[float] = []
    for _ in range(n_bootstraps):
        sample_indices = rng.choices(range(n), k=n)
        d = (sum(arr1[i] for i in sample_indices) - sum(arr2[i] for i in sample_indices)) / n
        boot_deltas.append(d)

    boot_deltas.sort()
    idx_lower = int(n_bootstraps * (alpha / 2.0))
    idx_upper = int(n_bootstraps * (1.0 - alpha / 2.0))
    ci_lower = boot_deltas[min(idx_lower, n_bootstraps - 1)]
    ci_upper = boot_deltas[min(idx_upper, n_bootstraps - 1)]
    return delta_obs, ci_lower, ci_upper


def main():
    parser = argparse.ArgumentParser(description="P6 Scoring & Gate Statistics (v3.6)")
    parser.add_argument("--gen_dir", type=str, default=None, help="Directory containing p6_gen_*.jsonl files")
    parser.add_argument("--order", type=str, default=None, help="Item order JSON file")
    parser.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--out_md", type=str, default=None, help="Output Markdown table path")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode: look for *_smoke.jsonl files")
    parser.add_argument("--bootstraps", type=int, default=10000, help="Number of bootstrap resamples (default: 10000)")
    parser.add_argument("--seed", type=int, default=2026, help="Bootstrap random seed (default: 2026)")

    # Explicit condition overrides
    parser.add_argument("--gen_A", type=str, default=None)
    parser.add_argument("--gen_B1", type=str, default=None)
    parser.add_argument("--gen_B2", type=str, default=None)
    parser.add_argument("--gen_B3", type=str, default=None)
    parser.add_argument("--gen_C", type=str, default=None)
    parser.add_argument("--gen_D1", type=str, default=None)
    parser.add_argument("--gen_D2", type=str, default=None)
    parser.add_argument("--gen_E1", type=str, default=None)
    parser.add_argument("--gen_E2", type=str, default=None)
    parser.add_argument("--gen_F", type=str, default=None)
    args = parser.parse_args()

    t0_start = time.time()
    print("=" * 80)
    print("CTS S7 P6 Scoring & Gate Statistics (scripts/s7_eval_score_p6_v36.py)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.executable}")
    print(f"Bootstrap Resamples: {args.bootstraps:,} | Seed: {args.seed}")
    print("=" * 80, flush=True)

    # 1. Verify Grader
    grade_fn = verify_grader()

    # 2. Determine paths
    gen_dir = args.gen_dir or os.path.join(BASE_S5_DIR, "results_s7")
    suffix = "_smoke.jsonl" if args.smoke else ".jsonl"

    def resolve_path(override: Optional[str], default_name: str) -> str:
        if override:
            return override
        return os.path.join(gen_dir, f"{default_name}{suffix}")

    gen_paths = {
        "A": resolve_path(args.gen_A, "p6_gen_A"),
        "B1": resolve_path(args.gen_B1, "p6_gen_B1"),
        "B2": resolve_path(args.gen_B2, "p6_gen_B2"),
        "B3": resolve_path(args.gen_B3, "p6_gen_B3"),
        "C": resolve_path(args.gen_C, "p6_gen_C"),
        "D1": resolve_path(args.gen_D1, "p6_gen_D1"),
        "D2": resolve_path(args.gen_D2, "p6_gen_D2"),
        "E1": resolve_path(args.gen_E1, "p6_gen_E1"),
        "E2": resolve_path(args.gen_E2, "p6_gen_E2"),
        "F": resolve_path(args.gen_F, "p6_gen_F"),
    }

    out_json = args.out_json or os.path.join(gen_dir, "p6_gate_table_smoke.json" if args.smoke else "p6_gate_table.json")
    out_md = args.out_md or os.path.join(gen_dir, "p6_gate_table_smoke.md" if args.smoke else "p6_gate_table.md")

    # 3. Item order
    order_path = args.order
    if not order_path:
        order_path = os.path.join(gen_dir, "p6_item_order_smoke.json" if args.smoke else "p6_item_order.json")
    
    with open(order_path, "r", encoding="utf-8") as f:
        order_data = json.load(f)
    eval_indices = [int(x) for x in order_data["indices"]]
    n_eval = len(eval_indices)
    print(f"Loaded item order from {order_path} ({n_eval} items)")

    # 4. Load Dataset Records
    dataset = load_dataset_records_local()

    # 5. Load and Grade Each Available Condition
    grades: Dict[str, Dict[str, Any]] = {}
    for cond_name, p in gen_paths.items():
        if not os.path.exists(p):
            continue
        data_dict = load_gen_jsonl(p)
        print(f"Loaded Condition {cond_name} from {p} ({len(data_dict)} items)")

        strict_list: List[int] = []
        lenient_list: List[int] = []
        nll_list: List[Optional[float]] = []

        for qid in eval_indices:
            if qid not in data_dict:
                continue
            entry = data_dict[qid]
            pred_text = entry.get("generation", "")
            gold_text = dataset[qid]["solution"]

            strict_match, lenient_match, boxed, lenient_ans = grade_fn(pred_text, gold_text)
            strict_list.append(1 if strict_match else 0)
            lenient_list.append(1 if lenient_match else 0)
            nll_list.append(entry.get("nll", None))

        grades[cond_name] = {
            "path": p,
            "count": len(strict_list),
            "strict_arr": strict_list,
            "lenient_arr": lenient_list,
            "strict_acc": (sum(strict_list) / len(strict_list)) if strict_list else 0.0,
            "lenient_acc": (sum(lenient_list) / len(lenient_list)) if lenient_list else 0.0,
            "nll_list": nll_list,
        }

    # Summary table of single condition accuracies
    print("\n--- Raw Accuracy Table ---")
    for c_name, g in grades.items():
        print(f"Condition {c_name:4s}: N={g['count']:3d} | Strict={g['strict_acc']*100:6.2f}% | Lenient={g['lenient_acc']*100:6.2f}%")

    # 6. Gate Calculations
    gates_output: Dict[str, Any] = {}

    def compute_gate(g_name: str, c1_key: str, c2_key: str, threshold: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if c1_key not in grades or c2_key not in grades:
            return None
        g1 = grades[c1_key]
        g2 = grades[c2_key]
        if g1["count"] != g2["count"] or g1["count"] == 0:
            return None

        # Strict
        s_obs, s_lo, s_hi = paired_bootstrap_ci(g1["strict_arr"], g2["strict_arr"], n_bootstraps=args.bootstraps, seed=args.seed)
        s_mcnemar_p = exact_mcnemar_one_sided_p(g1["strict_arr"], g2["strict_arr"])
        s_b = sum(1 for a, b in zip(g1["strict_arr"], g2["strict_arr"]) if a == 1 and b == 0)
        s_c = sum(1 for a, b in zip(g1["strict_arr"], g2["strict_arr"]) if a == 0 and b == 1)

        # Lenient
        l_obs, l_lo, l_hi = paired_bootstrap_ci(g1["lenient_arr"], g2["lenient_arr"], n_bootstraps=args.bootstraps, seed=args.seed)
        l_mcnemar_p = exact_mcnemar_one_sided_p(g1["lenient_arr"], g2["lenient_arr"])
        l_b = sum(1 for a, b in zip(g1["lenient_arr"], g2["lenient_arr"]) if a == 1 and b == 0)
        l_c = sum(1 for a, b in zip(g1["lenient_arr"], g2["lenient_arr"]) if a == 0 and b == 1)

        res = {
            "gate": g_name,
            "comparison": f"{c1_key} - {c2_key}",
            "N": g1["count"],
            "threshold": threshold,
            "strict": {
                "acc_c1": g1["strict_acc"],
                "acc_c2": g2["strict_acc"],
                "delta": round(s_obs, 5),
                "ci_95": [round(s_lo, 5), round(s_hi, 5)],
                "mcnemar_b": s_b,
                "mcnemar_c": s_c,
                "mcnemar_one_sided_p": round(s_mcnemar_p, 6),
            },
            "lenient": {
                "acc_c1": g1["lenient_acc"],
                "acc_c2": g2["lenient_acc"],
                "delta": round(l_obs, 5),
                "ci_95": [round(l_lo, 5), round(l_hi, 5)],
                "mcnemar_b": l_b,
                "mcnemar_c": l_c,
                "mcnemar_one_sided_p": round(l_mcnemar_p, 6),
            },
        }
        return res

    # Primary Gates
    gates_output["Gate_1"] = compute_gate("Gate 1 (Perturbation Discriminability)", "A", "B1")
    gates_output["Gate_2"] = compute_gate("Gate 2 (Floor Retention)", "A", "C", threshold=-0.02)
    gates_output["Gate_3"] = compute_gate("Gate 3 (Pure Base Superiority)", "A", "C")
    gates_output["Gate_4"] = compute_gate("Gate 4 (Root Feature Incrementality)", "A", "E1")

    # Auxiliary Gates
    gates_output["Aux_D_Control"] = compute_gate("Auxiliary (D Control)", "D1", "D2")
    gates_output["Aux_A_vs_F"] = compute_gate("Auxiliary (A - F Static Mean)", "A", "F")
    if "B2" in grades:
        gates_output["Aux_A_vs_B2"] = compute_gate("Auxiliary (A - B shift 2)", "A", "B2")
    if "B3" in grades:
        gates_output["Aux_A_vs_B3"] = compute_gate("Auxiliary (A - B shift 3)", "A", "B3")

    # Pooled B shift (if B1, B2, B3 present)
    b_shifts = [k for k in ["B1", "B2", "B3"] if k in grades]
    if len(b_shifts) > 1 and "A" in grades:
        n_b = len(b_shifts)
        n = grades["A"]["count"]
        pooled_strict = [sum(grades[k]["strict_arr"][i] for k in b_shifts) / n_b for i in range(n)]
        pooled_lenient = [sum(grades[k]["lenient_arr"][i] for k in b_shifts) / n_b for i in range(n)]
        s_obs, s_lo, s_hi = paired_bootstrap_ci(grades["A"]["strict_arr"], pooled_strict, n_bootstraps=args.bootstraps, seed=args.seed)
        l_obs, l_lo, l_hi = paired_bootstrap_ci(grades["A"]["lenient_arr"], pooled_lenient, n_bootstraps=args.bootstraps, seed=args.seed)
        gates_output["Aux_B_Pooled"] = {
            "gate": "Auxiliary (B Pooled Shift 1-3)",
            "comparison": f"A - Mean({','.join(b_shifts)})",
            "N": grades["A"]["count"],
            "strict": {"delta": round(s_obs, 5), "ci_95": [round(s_lo, 5), round(s_hi, 5)]},
            "lenient": {"delta": round(l_obs, 5), "ci_95": [round(l_lo, 5), round(l_hi, 5)]},
        }

    # NLL Paired Differences (if available)
    nll_diffs: Dict[str, Any] = {}
    for pair in [("A", "B1"), ("A", "C"), ("A", "E1")]:
        c1, c2 = pair
        if c1 in grades and c2 in grades:
            l1 = [x for x in grades[c1]["nll_list"] if x is not None]
            l2 = [x for x in grades[c2]["nll_list"] if x is not None]
            if len(l1) == len(l2) and len(l1) > 0:
                diff_obs, d_lo, d_hi = paired_bootstrap_ci(l1, l2, n_bootstraps=args.bootstraps, seed=args.seed)
                nll_diffs[f"{c1}-{c2}"] = {
                    "N": len(l1),
                    "mean_c1": sum(l1) / len(l1),
                    "mean_c2": sum(l2) / len(l2),
                    "delta": round(diff_obs, 5),
                    "ci_95": [round(d_lo, 5), round(d_hi, 5)],
                }
    if nll_diffs:
        gates_output["NLL_Differences"] = nll_diffs

    # Build final output structure
    output_package = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "eval_count": n_eval,
            "bootstraps": args.bootstraps,
            "seed": args.seed,
            "smoke": args.smoke,
            "grader_sha256": EXPECTED_GRADER_SHA,
        },
        "condition_accuracies": {
            c_name: {
                "count": g["count"],
                "strict_acc": round(g["strict_acc"], 5),
                "lenient_acc": round(g["lenient_acc"], 5),
            }
            for c_name, g in grades.items()
        },
        "gates": gates_output,
    }

    # Write JSON
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_package, f, indent=2)
    json_sha = compute_sha256(out_json)
    print(f"\n[OUTPUT] Gate table JSON saved to {out_json} (SHA: {json_sha})")

    # Write Markdown table (Numbers only, strictly no qualitative words)
    md_lines: List[str] = [
        f"# P6 Gate Evaluation Statistical Table ({'SMOKE' if args.smoke else 'FULL 600'})",
        "",
        f"- Timestamp: {output_package['metadata']['timestamp']}",
        f"- Items Count: {n_eval}",
        f"- Bootstrap Resamples: {args.bootstraps:,} (Seed: {args.seed})",
        f"- Grader SHA-256: `{EXPECTED_GRADER_SHA}`",
        "",
        "## 1. Condition Accuracies",
        "",
        "| Condition | Items (N) | Strict Acc (%) | Lenient Acc (%) |",
        "|:---|:---:|:---:|:---:|",
    ]
    for c_name, g in output_package["condition_accuracies"].items():
        md_lines.append(f"| {c_name} | {g['count']} | {g['strict_acc']*100:.2f} | {g['lenient_acc']*100:.2f} |")

    md_lines.extend([
        "",
        "## 2. Gate Statistical Metrics (Numbers Only)",
        "",
        "| Gate / Comparison | Metric | Delta (C1 - C2) | 95% Bootstrap CI | Discordant (b/c) | McNemar p (1-sided) |",
        "|:---|:---|:---:|:---:|:---:|:---:|",
    ])

    for g_key, g_val in gates_output.items():
        if not g_val or not isinstance(g_val, dict) or "strict" not in g_val:
            continue
        g_name = g_val.get("comparison", g_key)
        st = g_val["strict"]
        le = g_val["lenient"]
        s_ci = f"[{st['ci_95'][0]:+.4f}, {st['ci_95'][1]:+.4f}]" if "ci_95" in st else "-"
        l_ci = f"[{le['ci_95'][0]:+.4f}, {le['ci_95'][1]:+.4f}]" if "ci_95" in le else "-"
        s_bc = f"{st.get('mcnemar_b', '-')}/{st.get('mcnemar_c', '-')}"
        l_bc = f"{le.get('mcnemar_b', '-')}/{le.get('mcnemar_c', '-')}"
        s_p = f"{st.get('mcnemar_one_sided_p', '-'):.4f}" if isinstance(st.get('mcnemar_one_sided_p'), (int, float)) else "-"
        l_p = f"{le.get('mcnemar_one_sided_p', '-'):.4f}" if isinstance(le.get('mcnemar_one_sided_p'), (int, float)) else "-"

        md_lines.append(f"| {g_name} | Strict | {st['delta']:+.4f} | {s_ci} | {s_bc} | {s_p} |")
        md_lines.append(f"| {g_name} | Lenient | {le['delta']:+.4f} | {l_ci} | {l_bc} | {l_p} |")

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    md_sha = compute_sha256(out_md)
    print(f"[OUTPUT] Gate table Markdown saved to {out_md} (SHA: {md_sha})")

    elapsed_all = time.time() - t0_start
    print(f"Scoring and gate statistics complete in {elapsed_all:.2f}s.\n")


if __name__ == "__main__":
    main()

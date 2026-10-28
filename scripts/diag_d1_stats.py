#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diag_d1_stats.py - D1 Solution NLL Statistical Analysis
Directive: TASK-0070 §D1 (Ledger #430)

Computes:
- Per-condition mean, median NLL across 600 items (A, B1, C, E1, D1, F)
- Item-paired differences:
    * A - C
    * A - E1
    * E1 - C
- Bootstrap 95% CI (10,000 resamples, seed 2026, item-paired)
- Count and percentage of items where NLL(A) < NLL(E1)
"""

import os
import sys
import json
import numpy as np

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
if not os.path.exists(BASE_S5_DIR):
    BASE_S5_DIR = "."

P6R_DIR = os.path.join(BASE_S5_DIR, "results_s7", "p6r")
OUT_JSON = os.path.join(BASE_S5_DIR, "results_s7", "diag", "d1_nll_stats.json")


def load_nll_by_qid(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing NLL file: {filepath}")
    nll_dict = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                qid = int(rec["id"])
                nll = float(rec["nll"])
                nll_dict[qid] = nll
    return nll_dict


def bootstrap_ci(diff_array: np.ndarray, n_boot: int = 10000, seed: int = 2026) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(diff_array)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(diff_array, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return ci_lower, ci_upper


def main():
    conds = ["A", "B1", "C", "E1", "D1", "F"]
    nll_data = {}

    for c in conds:
        fp = os.path.join(P6R_DIR, f"diag_nll_{c}.jsonl")
        nll_data[c] = load_nll_by_qid(fp)
        print(f"Loaded {len(nll_data[c])} items for Condition {c} from {fp}")
        assert len(nll_data[c]) == 600, f"Expected 600 items for {c}, got {len(nll_data[c])}"

    # Load canonical order to ensure identical pairing
    order_file = os.path.join(BASE_S5_DIR, "results_s7", "p6_item_order.json")
    with open(order_file, "r", encoding="utf-8") as f:
        order_obj = json.load(f)
    qids = order_obj["indices"] if isinstance(order_obj, dict) else order_obj
    assert len(qids) == 600

    stats = {}

    # 1. Per-condition metrics
    for c in conds:
        vals = np.array([nll_data[c][q] for q in qids], dtype=np.float64)
        stats[c] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # 2. Pairwise paired differences
    # Pairs: A - C, A - E1, E1 - C
    pairs = [
        ("A", "C"),
        ("A", "E1"),
        ("E1", "C"),
    ]
    paired_stats = {}

    for c1, c2 in pairs:
        pair_key = f"{c1}_minus_{c2}"
        vals1 = np.array([nll_data[c1][q] for q in qids], dtype=np.float64)
        vals2 = np.array([nll_data[c2][q] for q in qids], dtype=np.float64)
        diff = vals1 - vals2  # paired diff
        diff_mean = float(np.mean(diff))
        diff_median = float(np.median(diff))
        ci_low, ci_high = bootstrap_ci(diff, n_boot=10000, seed=2026)

        paired_stats[pair_key] = {
            "mean_delta": diff_mean,
            "median_delta": diff_median,
            "ci_95": [ci_low, ci_high],
            "ci_95_str": f"[{ci_low:+.4f}, {ci_high:+.4f}]",
        }

    # 3. Count of A < E1
    vals_A = np.array([nll_data["A"][q] for q in qids], dtype=np.float64)
    vals_E1 = np.array([nll_data["E1"][q] for q in qids], dtype=np.float64)
    a_lt_e1_count = int(np.sum(vals_A < vals_E1))
    a_eq_e1_count = int(np.sum(vals_A == vals_E1))
    a_gt_e1_count = int(np.sum(vals_A > vals_E1))
    a_lt_e1_pct = float(a_lt_e1_count / 600.0) * 100.0

    out_obj = {
        "directive": "TASK-0070 D1",
        "conditions": stats,
        "pairwise_differences": paired_stats,
        "comparison_A_vs_E1": {
            "A_lt_E1_count": a_lt_e1_count,
            "A_lt_E1_pct": a_lt_e1_pct,
            "A_eq_E1_count": a_eq_e1_count,
            "A_gt_E1_count": a_gt_e1_count,
            "total_items": 600,
        }
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print("\n" + "=" * 60)
    print("D1 NLL Statistics Summary:")
    print("=" * 60)
    for c in conds:
        print(f"Condition {c:<3}: Mean = {stats[c]['mean']:.4f} | Median = {stats[c]['median']:.4f}")

    print("\nPairwise Differences:")
    for k, v in paired_stats.items():
        print(f"  {k:<12}: Mean Δ = {v['mean_delta']:+.4f} | 95% CI: {v['ci_95_str']}")

    print(f"\nA < E1 Items: {a_lt_e1_count} / 600 ({a_lt_e1_pct:.2f}%)")
    print(f"\nSaved results to {OUT_JSON}")


if __name__ == "__main__":
    main()

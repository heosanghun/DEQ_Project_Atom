#!/usr/bin/env python3
"""
scripts/rescore_phase2_v2.py - Phase 2 Re-scoring & Grader Unit Test Runner (TASK-0002 Part b).

Executes:
1. 20 Unit Tests covering Defect 9.
2. Re-scoring of results_s5/phase2_outputs.jsonl (640 items) using s5_eval_grader_v2.
3. Generation of condition-wise variation table (strict & lenient).
4. Output JSON: results_s5/phase2_rescore_v2.json.
5. All executions under CPU only, ulimit -v 62914560, nice -n 19.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Path setup
BASE_DIR = Path("/home/shoon/cts_e1_s5")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "scripts"))
if str(BASE_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "tests"))

from s5_eval_grader_v2 import grade_prediction
from test_grader_v2 import TEST_CASES, run_all_tests


def run_rescoring():
    print("=" * 80)
    print("TASK-0002 Part (b): Grader Unit Tests & Phase 2 Re-scoring")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Step 1: Run 20 Unit Tests
    print("\n--- STEP 1: Running 20 Unit Tests (Defect 9 Coverage) ---")
    unit_ok = run_all_tests()
    if not unit_ok:
        print("ERROR: Unit tests failed!", file=sys.stderr)
        sys.exit(1)

    # Step 2: Re-score phase2_outputs.jsonl
    print("\n--- STEP 2: Re-scoring Phase 2 Saved Outputs ---")
    input_path = BASE_DIR / "results_s5" / "phase2_outputs.jsonl"
    out_json_path = BASE_DIR / "results_s5" / "phase2_rescore_v2.json"

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {input_path}")

    # Track metrics per condition
    cond_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "old_strict_correct": 0,
        "new_strict_correct": 0,
        "old_lenient_correct": 0,
        "new_lenient_correct": 0,
        "strict_changes": [],
        "lenient_changes": [],
    })

    rescored_records = []

    for i, r in enumerate(records):
        cond = r.get("condition", "unknown")
        q_idx = r.get("q_idx")
        gold = r.get("gold", "")
        pred = r.get("prediction", "")

        old_strict = bool(r.get("strict_match", False))
        old_lenient = bool(r.get("lenient_match", False))

        new_strict, new_lenient, b_ext, l_ext = grade_prediction(pred, gold)

        stats = cond_stats[cond]
        stats["total"] += 1
        if old_strict:
            stats["old_strict_correct"] += 1
        if new_strict:
            stats["new_strict_correct"] += 1
        if old_lenient:
            stats["old_lenient_correct"] += 1
        if new_lenient:
            stats["new_lenient_correct"] += 1

        if old_strict != new_strict:
            stats["strict_changes"].append({
                "record_idx": i,
                "q_idx": q_idx,
                "old": old_strict,
                "new": new_strict,
                "gold": gold,
                "boxed_extract": b_ext,
            })

        if old_lenient != new_lenient:
            stats["lenient_changes"].append({
                "record_idx": i,
                "q_idx": q_idx,
                "old": old_lenient,
                "new": new_lenient,
                "gold": gold,
                "lenient_extract": l_ext,
            })

        r_copy = dict(r)
        r_copy["new_strict_match"] = new_strict
        r_copy["new_lenient_match"] = new_lenient
        r_copy["new_boxed_extract"] = b_ext
        r_copy["new_lenient_extract"] = l_ext
        rescored_records.append(r_copy)

    # Print variation table
    print("\n--- STEP 3: Condition-Wise Variation Table ---")
    print(f"{'Condition':<18} | {'Total':>5} | {'Old Strict':>10} | {'New Strict':>10} | {'Strict Δ':>8} | {'Old Lenient':>11} | {'New Lenient':>11} | {'Lenient Δ':>9}")
    print("-" * 100)

    summary_table = {}
    tot_records = len(records)
    tot_old_strict = 0
    tot_new_strict = 0
    tot_old_lenient = 0
    tot_new_lenient = 0

    for cond in sorted(cond_stats.keys()):
        s = cond_stats[cond]
        n = s["total"]
        o_st = s["old_strict_correct"]
        n_st = s["new_strict_correct"]
        d_st = n_st - o_st
        o_le = s["old_lenient_correct"]
        n_le = s["new_lenient_correct"]
        d_le = n_le - o_le

        tot_old_strict += o_st
        tot_new_strict += n_st
        tot_old_lenient += o_le
        tot_new_lenient += n_le

        print(f"{cond:<18} | {n:>5} | {o_st:>5} ({o_st/n*100:>4.1f}%) | {n_st:>5} ({n_st/n*100:>4.1f}%) | {d_st:>+8d} | {o_le:>6} ({o_le/n*100:>4.1f}%) | {n_le:>6} ({n_le/n*100:>4.1f}%) | {d_le:>+9d}")

        summary_table[cond] = {
            "total": n,
            "old_strict_count": o_st,
            "old_strict_acc": o_st / n,
            "new_strict_count": n_st,
            "new_strict_acc": n_st / n,
            "strict_delta": d_st,
            "old_lenient_count": o_le,
            "old_lenient_acc": o_le / n,
            "new_lenient_count": n_le,
            "new_lenient_acc": n_le / n,
            "lenient_delta": d_le,
            "strict_changes_count": len(s["strict_changes"]),
            "lenient_changes_count": len(s["lenient_changes"]),
            "strict_changes": s["strict_changes"],
            "lenient_changes": s["lenient_changes"],
        }

    print("-" * 100)
    d_tot_st = tot_new_strict - tot_old_strict
    d_tot_le = tot_new_lenient - tot_old_lenient
    print(f"{'TOTAL':<18} | {tot_records:>5} | {tot_old_strict:>5} ({tot_old_strict/tot_records*100:>4.1f}%) | {tot_new_strict:>5} ({tot_new_strict/tot_records*100:>4.1f}%) | {d_tot_st:>+8d} | {tot_old_lenient:>6} ({tot_old_lenient/tot_records*100:>4.1f}%) | {tot_new_lenient:>6} ({tot_new_lenient/tot_records*100:>4.1f}%) | {d_tot_le:>+9d}")
    print("=" * 100)

    # Save to JSON
    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": str(input_path),
            "total_records": tot_records,
            "unit_tests_passed": 20,
            "unit_tests_total": 20,
        },
        "summary": {
            "total_records": tot_records,
            "old_strict_total": tot_old_strict,
            "new_strict_total": tot_new_strict,
            "strict_delta_total": d_tot_st,
            "old_lenient_total": tot_old_lenient,
            "new_lenient_total": tot_new_lenient,
            "lenient_delta_total": d_tot_le,
        },
        "conditions": summary_table,
    }

    os.makedirs(out_json_path.parent, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved re-scoring delta table to: {out_json_path}")
    print(f"File size: {out_json_path.stat().st_size} bytes")


if __name__ == "__main__":
    run_rescoring()

#!/usr/bin/env python3
"""scripts/diagnose_tier3.py - Diagnose Tier 3 (rho >= 5e-3) items in current S7 partial
Directive: 원장 #372
"""

import json
import os
import re
import sys
import torch

BASE_DIR = "/home/shoon/cts_e1_s5"
S5_PATH = os.path.join(BASE_DIR, "results_s5", "zstar_s5_400.pt")
S7_PARTIAL = os.path.join(BASE_DIR, "results_s7", "zstar_s7_a0.25_400.partial.pt")
ITEMS_74_PATH = os.path.join(BASE_DIR, "configs", "s6_items_74.json")
ITEMS_400_PATH = os.path.join(BASE_DIR, "configs", "s7_items_400.json")


def main():
    d_s5 = torch.load(S5_PATH, map_location="cpu")
    d_s7 = torch.load(S7_PARTIAL, map_location="cpu")

    with open(ITEMS_74_PATH, "r", encoding="utf-8") as f:
        cfg_74 = json.load(f)
    nonconv_54 = set(cfg_74["nonconv_54"])

    # Index S5 by int q
    s5_by_q = {}
    for k, v in d_s5.items():
        m = re.match(r"^q(\d+)", str(k))
        q = int(m.group(1)) if m else int(v["meta"]["question_idx"])
        s5_by_q[q] = v

    # Parse current S7 items
    s7_by_q = {}
    for k, v in d_s7.items():
        m = re.match(r"^q(\d+)", str(k))
        q = int(m.group(1)) if m else int(v["meta"]["question_idx"])
        s7_by_q[q] = v

    total_s7 = len(s7_by_q)
    print(f"Total S7 partial items loaded: {total_s7}")

    processed_qs = list(s7_by_q.keys())
    in_nonconv_54 = [q for q in processed_qs if q in nonconv_54]
    print(f"Processed items in former nonconv_54: {len(in_nonconv_54)} / {total_s7} ({len(in_nonconv_54)/total_s7*100:.1f}%)")

    # Find all items with rho >= 5e-3 in S7
    tier3_items = []
    for q, v in s7_by_q.items():
        rho_s7 = float(v.get("rho_1000", v.get("rho", v.get("final_rho_fp"))))
        if rho_s7 >= 5e-3:
            tier3_items.append((q, rho_s7, v))

    print(f"\nTotal items with rho >= 5e-3 in current S7: {len(tier3_items)} / {total_s7} ({len(tier3_items)/total_s7*100:.2f}%)")

    print("\n" + "=" * 96)
    print(f"{'Q#':<6} | {'In 54?':<7} | {'S5 Conv?':<9} | {'S5 rho':<12} | {'S7 rho@1000':<13} | {'S7 Shape':<18} | {'S7 Min rho':<12}")
    print("-" * 96)

    newly_failed_total = []
    newly_failed_monotonic = []

    for q, rho_s7, v in sorted(tier3_items, key=lambda x: x[0]):
        is_in_54 = q in nonconv_54
        v_s5 = s5_by_q.get(q)
        if v_s5 is not None:
            s5_conv = bool(v_s5.get("converged", float(v_s5.get("final_rho_fp", 1.0)) < 5e-3))
            s5_rho = float(v_s5.get("final_rho_fp", v_s5.get("rho", -1.0)))
        else:
            s5_conv = "UNKNOWN"
            s5_rho = -1.0

        shape = v.get("shape", "unknown")
        min_rho = float(v.get("min_rho", rho_s7))

        if not is_in_54:
            newly_failed_total.append((q, s5_rho, rho_s7, shape, min_rho))
            if shape == "monotonic_decrease":
                newly_failed_monotonic.append((q, s5_rho, rho_s7, shape, min_rho))

        print(f"Q#{q:04d} | {str(is_in_54):<7} | {str(s5_conv):<9} | {s5_rho:<12.4e} | {rho_s7:<13.4e} | {shape:<18} | {min_rho:<12.4e}")

    print("=" * 96)

    print("\n[SUMMARY & DIAGNOSIS]")
    print(f"1. Total Tier 3 (rho >= 5e-3): {len(tier3_items)}")
    print(f"2. Former nonconv_54 members in Tier 3: {len([q for q, _, _ in tier3_items if q in nonconv_54])} / {len(tier3_items)}")
    print(f"3. Newly failed items (formerly converged in S5): {len(newly_failed_total)}")
    for nf in newly_failed_total:
        print(f"   - Q#{nf[0]:04d}: S5 rho={nf[1]:.4e}, S7 rho={nf[2]:.4e}, Shape={nf[3]}, Min rho={nf[4]:.4e}")
    print(f"4. Newly failed items with monotonic_decrease (still descending): {len(newly_failed_monotonic)}")

    # Decision logic
    if len(newly_failed_total) < 3:
        print("\n=> DECISION: (가) 순서 쏠림 (Order skew). 8개 중 대부분이 예전 미수렴이거나 새로 탈락한 수렴 문항이 3개 미만임. 그대로 진행.")
    elif len(newly_failed_monotonic) >= 3:
        print("\n=> DECISION: (나) 걸음 부족 (Insufficient steps). 예전 수렴 문항 3개 이상 신규 탈락 + 단조 감소 지속. NFE 연장 검토 필요.")
    else:
        print(f"\n=> DECISION: 새로 탈락한 수렴 문항 {len(newly_failed_total)}개 중 단조 감소는 {len(newly_failed_monotonic)}개임.")


if __name__ == "__main__":
    main()

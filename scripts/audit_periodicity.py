#!/usr/bin/env python3
"""
scripts/audit_periodicity.py - Test whether rho jumps (rho_k / rho_{k-10} > 2) cluster at k = 0 (mod 100)
Directive: 원장 #374 §2
"""

import os
import re
from collections import defaultdict

LOG_PATH = "/home/shoon/cts_e1_s5/raw_s7/s6_a0.25_20260927_220928.log"

items = []
current_item = None
current_steps = {}

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m_item = re.search(r"--- Item (\d+)/\d+ \| Q#(\d+)", line)
        if m_item:
            if current_item is not None:
                items.append((current_item, current_steps))
            current_item = int(m_item.group(2))
            current_steps = {}
            continue
        
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                step = int(parts[0])
                rho = float(parts[1])
                current_steps[step] = rho
            except ValueError:
                pass

if current_item is not None:
    items.append((current_item, current_steps))

print(f"Total parsed items from log: {len(items)}")

# Aggregate jumps rho_k / rho_{k-10} > 2.0
jump_mod_counts = defaultdict(int)
total_jumps = 0
jumps_list = []

for q, steps in items:
    sorted_steps = sorted(steps.keys())
    for i in range(1, len(sorted_steps)):
        k_prev = sorted_steps[i - 1]
        k_cur = sorted_steps[i]
        if k_cur - k_prev == 10:
            r_prev = steps[k_prev]
            r_cur = steps[k_cur]
            ratio = r_cur / max(r_prev, 1e-12)
            if ratio > 2.0:
                mod_val = k_cur % 100
                jump_mod_counts[mod_val] += 1
                total_jumps += 1
                jumps_list.append((q, k_prev, k_cur, r_prev, r_cur, ratio, mod_val))

print(f"\nTotal jumps (rho_k / rho_{{k-10}} > 2.0): {total_jumps}")
print("\nDistribution across k mod 100:")
print(f"{'k mod 100':>10} | {'Count':>6} | {'Percentage':>10} | {'Expected':>10}")
print("-" * 45)
for m in sorted(range(0, 100, 10)):
    cnt = jump_mod_counts[m]
    pct = (cnt / total_jumps * 100) if total_jumps > 0 else 0
    print(f"{m:10d} | {cnt:6d} | {pct:9.2f}% | {'10.00%':>10}")

print("\nSample top 15 jumps:")
print(f"{'Q#':>6} | {'k_prev':>6} | {'k_cur':>6} | {'rho_prev':>12} | {'rho_cur':>12} | {'Ratio':>8} | {'k_cur mod 100'}")
print("-" * 75)
for j in sorted(jumps_list, key=lambda x: x[5], reverse=True)[:15]:
    print(f"Q#{j[0]:04d} | {j[1]:6d} | {j[2]:6d} | {j[3]:12.6e} | {j[4]:12.6e} | {j[5]:7.2f}x | {j[6]:13d}")

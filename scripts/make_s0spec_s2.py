#!/usr/bin/env python3
"""scripts/make_s0spec_s2.py - S2 grading spec (CC 2026-10-14): oracle matched / shuffled per seed + C with format instruction (from S1)."""
import json, sys
O = "results_s7/p6r/g2v7s2/"; D = "results_s7/p6_item_order.json"
spec = {"C_fmt": ["results_s7/p6r/g2v6s1/gen_C.jsonl", D]}
for t in ["S2O47", "S2O49", "S2O50"]:
    spec[t] = [O + f"gen_{t}.jsonl", D]
    spec["shuf_" + t] = [O + f"genB1_{t}.jsonl", D]
json.dump(spec, open(sys.argv[1], "w"))
print("[SPEC]", len(spec))

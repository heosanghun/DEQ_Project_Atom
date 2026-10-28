#!/usr/bin/env python3
"""scripts/make_s0spec_ag.py - grading spec for AG's g2v8 stage-2 generations (CC 2026-10-19): A / B1 / E per seed + C_fmt."""
import json, sys
out = sys.argv[1]; G = "/tmp/cts_g2v8_v2_s2/"; S1 = "/tmp/cts_g2v8_v2_s1/"; D = "results_s7/p6_item_order.json"
spec = {"C_fmt": ["results_s7/p6r/g2v6s1/gen_C.jsonl", D]}
for s in ["2047", "2049", "2050"]:
    for c in ["A", "B1", "E"]:
        spec[f"ag{c}_{s}"] = [G + f"gen_{c}_{s}.jsonl", D]
    for c in ["matched", "shuffled"]:
        spec[f"agS1{c}_{s}"] = [S1 + f"gen_{c}_{s}.jsonl", D]
json.dump(spec, open(out, "w"))
print("[SPEC]", len(spec))

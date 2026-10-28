#!/usr/bin/env python3
"""scripts/make_s0spec_x.py - grading spec for X experiments (CC 2026-10-19).
Usage: make_s0spec_x.py <out_json> <gen_dir> <tag1,tag2,...>   -> {C_fmt, <tag>, shuf_<tag>}"""
import json, sys
out, gen_dir, tags = sys.argv[1], sys.argv[2].rstrip("/") + "/", sys.argv[3].split(",")
D = "results_s7/p6_item_order.json"
spec = {"C_fmt": ["results_s7/p6r/g2v6s1/gen_C.jsonl", D]}
for t in tags:
    spec[t] = [gen_dir + f"gen_{t}.jsonl", D]
    spec["shuf_" + t] = [gen_dir + f"genB1_{t}.jsonl", D]
json.dump(spec, open(out, "w"))
print("[SPEC]", len(spec))

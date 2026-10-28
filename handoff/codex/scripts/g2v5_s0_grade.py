#!/usr/bin/env python3
"""scripts/g2v5_s0_grade.py - S0 (CC 2026-10-13): per-item strict / lenient / boxed flags for EXISTING generation files (no generation, no training).
Uses the locked grader via s7_eval_score_p6_v36.verify_grader (SHA 90cc93e4 check). Output: one JSON {label: {qid: [strict, lenient, has_boxed]}}."""
import json, os, sys
sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from s7_eval_score_p6_v36 import verify_grader, load_dataset_records_local, load_gen_jsonl
grade = verify_grader()
ds = load_dataset_records_local()
spec = json.load(open(sys.argv[1]))  # {label: [path, order_json]}
out = {}
for label, (path, order) in spec.items():
    o = json.load(open(order)); ids = [int(x) for x in (o["indices"] if isinstance(o, dict) else o)]
    g = load_gen_jsonl(path); res = {}
    for q in ids:
        if q not in g: continue
        s, l, b, _ = grade(g[q].get("generation", ""), ds[q]["solution"])
        res[q] = [int(bool(s)), int(bool(l)), int(bool(b))]
    out[label] = res
    print(f"[S0] {label}: n {len(res)} strict {sum(v[0] for v in res.values())} lenient {sum(v[1] for v in res.values())} boxed {sum(v[2] for v in res.values())}", flush=True)
json.dump(out, open(sys.argv[2], "w"))
print("[S0-DONE]", sys.argv[2])

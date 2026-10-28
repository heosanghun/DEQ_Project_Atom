#!/usr/bin/env python3
"""scripts/cc_server_json_union.py - conservative exclusion source for G2v2 (CC, 2026-10-08, AUDIT-P012 re-FIX 4).
Walks /home/shoon/cts_e1 and /home/shoon/cts_e1_s5 json files (< 20 MB), collects every integer in [0, 5000) found in
integer lists (len <= 6000) or as q-key digits (keys "qNNNN..."). Writes configs/g2v2_server_json_union.json.
Some collected integers are not item ids; they are excluded anyway (conservative)."""
import json, os
roots = ["/home/shoon/cts_e1", "/home/shoon/cts_e1_s5"]
skip = {".git", "node_modules", ".venv", "__pycache__", ".cache", "site-packages"}
own = ("g2v2_fresh1000", "g2v2_exclude", "g2v2_server_json_union")
def collect(o, out, depth=0):
    if depth > 4: return
    if isinstance(o, list) and o and all(isinstance(x, int) for x in o[:50]) and len(o) <= 6000:
        out.update(x for x in o if isinstance(x, int))
    elif isinstance(o, dict):
        for k, v in o.items():
            collect(v, out, depth + 1)
            if isinstance(k, str) and k[:1] == "q" and k[1:5].isdigit():
                out.add(int(k[1:5]))
union, srcs = set(), {}
for root in roots:
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip]
        for fn in fns:
            if not fn.endswith(".json") or any(t in fn for t in own): continue
            p = os.path.join(dp, fn)
            try:
                if os.path.getsize(p) > 20_000_000: continue
                d = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            s = set(); collect(d, s); s = {x for x in s if 0 <= x < 5000}
            if s: srcs[p] = len(s); union |= s
json.dump({"indices": sorted(union), "n_files": len(srcs), "sources": srcs}, open("configs/g2v2_server_json_union.json", "w"))
print("files", len(srcs), "union", len(union))

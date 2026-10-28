#!/usr/bin/env python3
"""scripts/g2v4_select_fresh.py - G2 4th attempt fresh items (CC, 2026-10-11, PREREG_S7_G2v4 §2). Run on the server from /home/shoon/cts_e1_s5.
Exclusion = configs/g2v2_exclude.json (2,242: G2v2 rule incl. server json union of 10/08)
          + configs/g2v2_fresh1000.json (USED by G2v3)
          + a NEW conservative server json scan (same rule as scripts/cc_server_json_union.py: every int in [0,5000) in int lists (len <= 6000)
            or q-key digits, all .json < 20 MB under /home/shoon/cts_e1 and /home/shoon/cts_e1_s5), skipping only this script's own outputs
            and the two G2v2 selection outputs (g2v2_exclude, g2v2_server_json_union) whose content is already included above.
Selection: ALL remaining q_idx (no sampling), sorted -> configs/g2v4_fresh.json; shards k = sorted[k::3] -> configs/g2v4_fresh_shard{k}.json.
Reads no prompt/solution text."""
import hashlib, json, os
roots = ["/home/shoon/cts_e1", "/home/shoon/cts_e1_s5"]
skip_dirs = {".git", "node_modules", ".venv", "__pycache__", ".cache", "site-packages"}
own = ("g2v4_fresh", "g2v4_exclude", "g2v4_server_json_union", "g2v2_exclude", "g2v2_server_json_union")
def collect(o, out, depth=0):
    if depth > 4: return
    if isinstance(o, list) and o and all(isinstance(x, int) for x in o[:50]) and len(o) <= 6000:
        out.update(x for x in o if isinstance(x, int))
    elif isinstance(o, dict):
        for k, v in o.items():
            collect(v, out, depth + 1)
            if isinstance(k, str) and k[:1] == "q" and k[1:5].isdigit():
                out.add(int(k[1:5]))
base = set(json.load(open("configs/g2v2_exclude.json"))["indices"])
used = set(json.load(open("configs/g2v2_fresh1000.json"))["indices"])
cand0 = sorted(set(range(5000)) - base - used)
union, srcs, hits = set(), {}, {}
for root in roots:
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip_dirs]
        for fn in fns:
            if not fn.endswith(".json") or any(t in fn for t in own): continue
            p = os.path.join(dp, fn)
            try:
                if os.path.getsize(p) > 20_000_000: continue
                d = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            s = set(); collect(d, s); s = {x for x in s if 0 <= x < 5000}
            if s:
                srcs[p] = len(s); union |= s
                h = len(s & set(cand0))
                if h: hits[p] = h
excl = base | used | union
chosen = sorted(set(range(5000)) - excl)
json.dump({"indices": sorted(union), "n_files": len(srcs), "sources": srcs}, open("configs/g2v4_server_json_union.json", "w"))
json.dump({"indices": sorted(excl), "rule": "g2v2_exclude + g2v2_fresh1000 (used) + g2v4 server json union", "n_base": len(base), "n_used": len(used), "n_union": len(union)}, open("configs/g2v4_exclude.json", "w"))
json.dump({"indices": chosen}, open("configs/g2v4_fresh.json", "w"))
for k in range(3):
    json.dump({"indices": chosen[k::3]}, open(f"configs/g2v4_fresh_shard{k}.json", "w"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print(f"[SELECT] candidates before scan {len(cand0)} | scan files {len(srcs)} union {len(union)} | scan hits on candidates {len(set(cand0) & union)} | chosen {len(chosen)}")
for p, h in sorted(hits.items(), key=lambda x: -x[1])[:15]:
    print(f"[HIT] {h:5d} {p}")
for p in ["configs/g2v4_fresh.json"] + [f"configs/g2v4_fresh_shard{k}.json" for k in range(3)] + ["configs/g2v4_exclude.json", "configs/g2v4_server_json_union.json"]:
    print(sha(p), p)

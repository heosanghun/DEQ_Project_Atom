#!/usr/bin/env python3
"""scripts/g2v4_idscan_v2.py (v2: optional list path argv[1] for a positive control)
scripts/g2v4_idscan.py - AUDIT-P014 FIX-1(c) (CC, 2026-10-11). Item-id-only scan: no prompt/solution text is read or printed.
Collects ids from every .jsonl ("id" / "q_idx" / "question_idx" fields) and every .pt dict key matching q\d{4} (or int keys)
under results_s7/, raw_s7/, scratch/, results/ of /home/shoon/cts_e1_s5 and /home/shoon/cts_e1, and intersects with configs/g2v4b_fresh.json (1,758).
Prints per-file intersection counts (only files with hits), split into the 524 (run1-only) part and the 1,234 (v1) part."""
import json, os, re, sys, torch
F = set(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/g2v4b_fresh.json"))["indices"]); V1 = set(json.load(open("configs/g2v4_fresh.json"))["indices"]); D = F - V1
roots = [os.path.join(r, s) for r in ("/home/shoon/cts_e1_s5", "/home/shoon/cts_e1") for s in ("results_s7", "raw_s7", "scratch", "results")]
pat = re.compile(r"^q(\d{4})")
n_jsonl = n_pt = n_skip = 0; hits = []
def ids_from_obj(o):
    out = set()
    if isinstance(o, dict):
        for k in o.keys():
            if isinstance(k, int): out.add(k)
            elif isinstance(k, str):
                m = pat.match(k)
                if m: out.add(int(m.group(1)))
    return out
for root in roots:
    if not os.path.isdir(root): continue
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in {".git", "__pycache__", ".venv"}]
        for fn in fns:
            p = os.path.join(dp, fn); s = set()
            try:
                if fn.endswith(".jsonl"):
                    if os.path.getsize(p) > 2_000_000_000: n_skip += 1; continue
                    n_jsonl += 1
                    for line in open(p, encoding="utf-8", errors="ignore"):
                        line = line.strip()
                        if not line.startswith("{"): continue
                        try: r = json.loads(line)
                        except Exception: continue
                        for k in ("id", "q_idx", "question_idx", "qid"):
                            v = r.get(k)
                            if isinstance(v, int): s.add(v)
                elif fn.endswith(".pt"):
                    if os.path.getsize(p) > 20_000_000_000: n_skip += 1; continue
                    n_pt += 1
                    try: o = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
                    except Exception: o = torch.load(p, map_location="cpu", weights_only=False)
                    s = ids_from_obj(o); del o
                else:
                    continue
            except Exception as e:
                print(f"[ERR] {p}: {type(e).__name__}", flush=True); continue
            h = s & F
            if h: hits.append((p, len(h), len(h & D), len(h & V1)))
print(f"[SCAN] jsonl {n_jsonl} | pt {n_pt} | skipped(size) {n_skip} | files with hits on 1,758: {len(hits)}")
for p, a, b, c in sorted(hits, key=lambda x: -x[1]):
    print(f"[HIT] total {a} | in 524 {b} | in 1,234 {c} | {p}")
print("[RESULT]", "PASS (0 hits)" if not hits else "HITS FOUND")

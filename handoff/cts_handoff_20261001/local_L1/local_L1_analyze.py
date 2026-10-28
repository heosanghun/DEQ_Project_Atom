#!/usr/bin/env python3
"""local_L1_analyze.py — pre-specified (PREREG_LOCAL_L1) summary of results_local/L1_trace.pt. Report only."""
import json
import os
import statistics as st
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results_local", "L1_trace.pt")
d = torch.load(path, map_location="cpu", weights_only=False)
meta = d.pop("_meta", None)
recs = {k: v for k, v in d.items() if k.startswith("q")}
n = len(recs)
T = 1e-2


def pct(a, b):
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else "0/0"


final_ok = [k for k, r in recs.items() if r["r_final_fresh"] < T]
low_ok = [k for k, r in recs.items() if r["r_lowest"] < T and r["k_star"] >= 50]
low_ok_any = [k for k, r in recs.items() if r["r_lowest"] < T]
strict_final = sum(r["r_final_fresh"] < 5e-3 for r in recs.values())
strict_low = sum(r["r_lowest"] < 5e-3 and r["k_star"] >= 50 for r in recs.values())
nonconv = [k for k in recs if k not in final_ok]
sw_non = [recs[k]["switches_301_1000"] for k in nonconv]
sw_conv = [recs[k]["switches_301_1000"] for k in final_ok]
bins = {"[0,50)": 0, "[50,300)": 0, "[300,700)": 0, "[700,1000]": 0}
for r in recs.values():
    k = r["k_star"]
    bins["[0,50)" if k < 50 else "[50,300)" if k < 300 else "[300,700)" if k < 700 else "[700,1000]"] += 1
top3_eq = sum(r["top3_final"] == r["top3_lowest"] for r in recs.values())
rescued = [k for k in recs if k in low_ok and k not in final_ok]
diff = [abs(r["r_last_log"] - r["r_final_fresh"]) / max(r["r_final_fresh"], 1e-30) for r in recs.values()]

out = {
    "n": n,
    "1_final_r_lt_1e-2": pct(len(final_ok), n),
    "1_lowest_r_lt_1e-2_kstar_ge_50": pct(len(low_ok), n),
    "1_lowest_r_lt_1e-2_any_k": pct(len(low_ok_any), n),
    "1_gap_pp": round(100.0 * (len(low_ok) - len(final_ok)) / n, 1) if n else None,
    "1_strict_5e-3_final_vs_lowest": [pct(strict_final, n), pct(strict_low, n)],
    "2_switches_301_1000_median_nonconv_vs_conv": [st.median(sw_non) if sw_non else None, st.median(sw_conv) if sw_conv else None],
    "2_switches_nonconv_gt0": pct(sum(s > 0 for s in sw_non), len(sw_non)),
    "2_switches_conv_gt0": pct(sum(s > 0 for s in sw_conv), len(sw_conv)),
    "3_kstar_bins": bins,
    "3_top3_lowest_eq_final": pct(top3_eq, n),
    "3_rescued_items": rescued,
    "4_rel_diff_rlog999_vs_r1000_median": st.median(diff) if diff else None,
    "wall_s_mean": st.mean(r["wall_s"] for r in recs.values()) if n else None,
}
print(json.dumps(out, ensure_ascii=False, indent=1))
with open(os.path.join(ROOT, "results_local", "L1_summary.json"), "w", encoding="utf-8") as f:
    json.dump({"summary": out, "meta": {k: v for k, v in (meta or {}).items() if k != "cfg"}}, f, ensure_ascii=False, indent=1)

"""X6 analysis (CC 2026-10-20, before results). Pooled 3-seed Stage O (X5O47 + X6R49 + X6R50) and X6S47 alone.
Usage: python scratch/x6_analyze.py shared/reports/x5/x5_items.json shared/reports/x6/x6_items.json"""
import json, random, statistics as st, sys
A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2])); D = dict(A); D.update({k: v for k, v in B.items() if k != "C_fmt"})
items = sorted(D["C_fmt"].keys(), key=int); n = len(items)
def acc(t, i): return sum(D[t][k][i] for k in items) / n * 100
def boot(pairs, idx, B=10000, seed=2026):
    rng = random.Random(seed); X = [[D[a][k][idx] for k in items] for a, _ in pairs]; Y = [[D[b][k][idx] for k in items] for _, b in pairs]
    obs = st.mean((sum(x) - sum(y)) / n for x, y in zip(X, Y)) * 100; bs = []
    for _ in range(B):
        ix = [rng.randrange(n) for _ in range(n)]; bs.append(st.mean(sum(x[i] - y[i] for i in ix) / n for x, y in zip(X, Y)) * 100)
    bs.sort(); return obs, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
def report(name, pairs):
    print("==", name)
    for idx, lab in ((1, "lenient"), (0, "strict")):
        o, lo, hi = boot(pairs, idx); per = [round(acc(a, idx) - acc(b, idx), 2) for a, b in pairs]
        print(f"  {lab:8s} mean {o:+.2f}%p  95% CI [{lo:+.2f}, {hi:+.2f}]  per-run {per}")
print("== per condition")
for t in ["C_fmt", "X5O47", "shuf_X5O47", "X6R49", "shuf_X6R49", "X6R50", "shuf_X6R50", "X6S47", "shuf_X6S47"]:
    if t in D: print(f"  {t:11s} boxed {acc(t,2):5.1f}%  lenient {acc(t,1):5.2f}%  strict {acc(t,0):5.2f}%")
R = [t for t in ["X5O47", "X6R49", "X6R50"] if t in D]
report("X6-R pooled 3 seeds: matched - shuffled (O-A: CI lower > 0)", [(t, "shuf_" + t) for t in R])
report("X6-R pooled 3 seeds: matched - C_fmt (O-B: >= +5%p)", [(t, "C_fmt") for t in R])
report("X6-R pooled: shuffled - C_fmt (harm report)", [("shuf_" + t, "C_fmt") for t in R])
if "X6S47" in D:
    report("X6-S (aw20, 3000 steps): matched - shuffled (O-A)", [("X6S47", "shuf_X6S47")])
    report("X6-S: matched - C_fmt (O-B)", [("X6S47", "C_fmt")])
    report("X6-S: shuffled - C_fmt (harm report)", [("shuf_X6S47", "C_fmt")])

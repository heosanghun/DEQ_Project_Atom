"""X8-0 analysis (CC 2026-10-21, before results). Pooled 3-seed problem-latent control.
Usage: python scratch/x8_0_analyze.py shared/reports/x8_0/x8_0_items.json"""
import json, random, statistics as st, sys
D = json.load(open(sys.argv[1]))
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
for t in ["C_fmt", "X80P47", "shuf_X80P47", "X80P49", "shuf_X80P49", "X80P50", "shuf_X80P50"]:
    if t in D: print(f"  {t:11s} boxed {acc(t,2):5.1f}%  lenient {acc(t,1):5.2f}%  strict {acc(t,0):5.2f}%")
R = [t for t in ["X80P47", "X80P49", "X80P50"] if t in D]
report("X8-0 pooled 3 seeds: matched - shuffled (O-A: CI lower > 0)", [(t, "shuf_" + t) for t in R])
report("X8-0 pooled 3 seeds: matched - C_fmt (O-B: >= +5%p)", [(t, "C_fmt") for t in R])
report("X8-0 pooled: shuffled - C_fmt (harm report)", [("shuf_" + t, "C_fmt") for t in R])

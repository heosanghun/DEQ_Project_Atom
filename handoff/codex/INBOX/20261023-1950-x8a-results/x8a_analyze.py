"""X8-A analysis (CC 2026-10-23, before results). A = X8A{47,49,50} (DEQ+bridge joint), E = X7E{47,49,50} (z_src, same recipe),
C = C_fmt. Rules (X8 plan §2/§7): (1) A-E lenient same sign all seeds AND pooled CI lower > 0; (2) A matched-shuffled CI lower > 0.
Usage: python scratch/x8a_analyze.py shared/reports/x8a/x8a_items.json shared/reports/x7/x7_items.json"""
import json, random, statistics as st, sys
A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2])); D = dict(A); D.update({k: v for k, v in B.items() if k.startswith("X7E") or k.startswith("shuf_X7E")})
items = sorted(D["C_fmt"].keys(), key=int); n = len(items)
def acc(t, i, sub=None):
    ks = sub if sub is not None else items; return sum(D[t][k][i] for k in ks) / max(1, len(ks)) * 100
def boot(pairs, idx, B=10000, seed=2026, sub=None):
    ks = sub if sub is not None else items; m = len(ks); rng = random.Random(seed)
    X = [[D[a][k][idx] for k in ks] for a, _ in pairs]; Y = [[D[b][k][idx] for k in ks] for _, b in pairs]
    obs = st.mean((sum(x) - sum(y)) / m for x, y in zip(X, Y)) * 100; bs = []
    for _ in range(B):
        ix = [rng.randrange(m) for _ in range(m)]; bs.append(st.mean(sum(x[i] - y[i] for i in ix) / m for x, y in zip(X, Y)) * 100)
    bs.sort(); return obs, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
def report(name, pairs, sub=None):
    print("==", name, "" if sub is None else f"(subset n={len(sub)})")
    for idx, lab in ((1, "lenient"), (0, "strict")):
        o, lo, hi = boot(pairs, idx, sub=sub); per = [round(acc(a, idx, sub) - acc(b, idx, sub), 2) for a, b in pairs]
        print(f"  {lab:8s} mean {o:+.2f}%p  95% CI [{lo:+.2f}, {hi:+.2f}]  per-seed {per}")
AA = ["X8A47", "X8A49", "X8A50"]; EE = ["X7E47", "X7E49", "X7E50"]
print("== per condition (n=600): boxed / lenient / strict")
for t in ["C_fmt"] + AA + ["shuf_" + t for t in AA] + EE + ["shuf_" + t for t in EE]:
    if t in D: print(f"  {t:11s} boxed {acc(t,2):5.1f}%  lenient {acc(t,1):5.2f}%  strict {acc(t,0):5.2f}%")
report("Rule (1): A - E (same sign all seeds AND pooled CI lower > 0)", list(zip(AA, EE)))
report("Rule (2): A matched - shuffled (CI lower > 0)", [(t, "shuf_" + t) for t in AA])
report("(ref) A - C_fmt (mechanism cost offset?)", [(t, "C_fmt") for t in AA])
report("(ref) A shuffled - C_fmt (harm)", [("shuf_" + t, "C_fmt") for t in AA])
report("(ref) E matched - shuffled", [(t, "shuf_" + t) for t in EE])
both = [k for k in items if all(D[t][k][2] for t in AA + EE)]
report("Rule (1) both-boxed subset: A - E", list(zip(AA, EE)), sub=both)

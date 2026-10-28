"""X5 analysis (from x3_analyze) (CC 2026-10-19, written before Stage 1 results): CTO metrics on per-item grades from g2v5_s0_grade.
Usage: python scratch/x3_analyze.py shared/reports/x3b/x3_items_1.json   (Stage 1: O47, A47, E47)
       python scratch/x3_analyze.py shared/reports/x3b/x3_items.json     (final: + A49, E49, A50, E50)
Rules: X_INTERFACE_PLAN_20261019.md X3-min. Stage O: O matched-shuffled lenient CI lower>0 AND O matched-C_fmt >= +5%p.
Stage AE: (1) A-E lenient same sign all seeds AND pooled cluster-bootstrap CI lower>0; (2) A matched-shuffled CI lower>0."""
import json, random, statistics as st, sys
D = json.load(open(sys.argv[1]))
tags = [t for t in D if not t.startswith("shuf_") and t != "C_fmt"]
items = sorted(D[tags[0]].keys(), key=int); n = len(items)
def acc(t, i, sub=None):
    ks = sub if sub is not None else items
    return sum(D[t][k][i] for k in ks) / max(1, len(ks)) * 100
print(f"== 0) per condition (n={n}): boxed / lenient / strict / strict|boxed")
for t in ["C_fmt"] + tags + ["shuf_" + t for t in tags]:
    if t not in D: continue
    v = [D[t][k] for k in items]; nb = sum(x[2] for x in v)
    print(f"  {t:11s} boxed {acc(t,2):5.1f}%  lenient {acc(t,1):5.2f}%  strict {acc(t,0):5.2f}%  strict|boxed {sum(x[0] for x in v)/max(1,nb)*100:5.1f}% (nb {nb})")
def boot(pairs, idx, B=10000, seed=2026, sub=None):
    ks = sub if sub is not None else items; m = len(ks)
    rng = random.Random(seed)
    X = [[D[a][k][idx] for k in ks] for a, _ in pairs]; Y = [[D[b][k][idx] for k in ks] for _, b in pairs]
    obs = st.mean((sum(x) - sum(y)) / m for x, y in zip(X, Y)) * 100
    bs = []
    for _ in range(B):
        ix = [rng.randrange(m) for _ in range(m)]
        bs.append(st.mean(sum(x[i] - y[i] for i in ix) / m for x, y in zip(X, Y)) * 100)
    bs.sort(); return obs, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
def report(name, pairs, sub=None):
    print("==", name, "" if sub is None else f"(subset n={len(sub)})")
    for idx, lab in ((1, "lenient (primary)"), (0, "strict (secondary)")):
        o, lo, hi = boot(pairs, idx, sub=sub)
        per = [round(acc(a, idx, sub) - acc(b, idx, sub), 2) for a, b in pairs]
        print(f"  {lab:20s} seed-mean {o:+.2f}%p  95% CI [{lo:+.2f}, {hi:+.2f}]  per-seed {per}")
O = [t for t in tags if t.startswith("X5O")]; A = sorted(t for t in tags if t.startswith("X5A")); E = sorted(t for t in tags if t.startswith("X5E"))
if O:
    report("Stage O: oracle matched - shuffled (rule: lenient CI lower > 0)", [(t, "shuf_" + t) for t in O])
    report("Stage O: oracle matched - C_fmt (rule: >= +5%p)", [(t, "C_fmt") for t in O])
    report("Stage O harm check: oracle shuffled - C_fmt (report rule: CI includes 0)", [("shuf_" + t, "C_fmt") for t in O])
if A and E and len(A) == len(E):
    report("Stage AE (1): A - E (rule: same sign all seeds AND pooled CI lower > 0)", list(zip(A, E)))
    report("Stage AE (2): A matched - shuffled (rule: CI lower > 0)", [(t, "shuf_" + t) for t in A])
    report("Stage AE (ref): E matched - shuffled", [(t, "shuf_" + t) for t in E])
    report("Stage AE (ref): A - C_fmt", [(t, "C_fmt") for t in A]); report("Stage AE (ref): E - C_fmt", [(t, "C_fmt") for t in E])
    both = [k for k in items if all(D[t][k][2] for t in A + E)]
    report("Stage AE (1) both-boxed subset: A - E", list(zip(A, E)), sub=both)

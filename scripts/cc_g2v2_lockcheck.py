#!/usr/bin/env python3
"""scripts/cc_g2v2_lockcheck.py - PREREG_S7_G2v2 lock-time checks (CC, 2026-10-08; AUDIT-P012 re-audit 3 lock conditions).
(1) z_src consistency on dev 600: runner z_src (results_s7/g2v2/zstar_zsrcinj_dev600.pt, field z_src) vs
    results_s7/zsrc_heldout600.pt (field z_star holds z_src; key priority qNNNN_a0.25 over qNNNN). Pass: all 600 cos >= 0.99999.
(2) SHA of merged z*' files.
(3) first-modification times of decoder / dev outputs (must not exist yet, or be later than lock time).
Read-only."""
import glob, hashlib, json, os, time
import torch

os.chdir("/home/shoon/cts_e1_s5")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
print("now", time.strftime("%F %T"))
dev = torch.load("results_s7/g2v2/zstar_zsrcinj_dev600.pt", map_location="cpu")
ref = torch.load("results_s7/zsrc_heldout600.pt", map_location="cpu")
qs = json.load(open("configs/g2v2_dev600.json"))["indices"]
cos_all, maxd_all = [], []
for q in qs:
    a = dev[f"q{q:04d}_a0.25"]["z_src"].double().flatten()
    e = ref.get(f"q{q:04d}_a0.25", ref.get(f"q{q:04d}"))
    b = (e["z_star"] if isinstance(e, dict) else e).double().flatten()
    cos_all.append(float(a @ b / (a.norm() * b.norm())))
    maxd_all.append(float((a - b).abs().max()))
n_ok = sum(c >= 0.99999 for c in cos_all)
print(f"[ZSRC] n={len(qs)} cos>=0.99999: {n_ok} | min cos {min(cos_all):.9f} | max|d| max {max(maxd_all):.6e} | exact-equal items {sum(m == 0.0 for m in maxd_all)}")
print("[ZSRC] RESULT", "PASS" if n_ok == len(qs) else "FAIL")
for p in ("results_s7/g2v2/zstar_zsrcinj_train1000.pt", "results_s7/g2v2/zstar_zsrcinj_dev600.pt"):
    print("[SHA]", sha(p), p)
tr = torch.load("results_s7/g2v2/zstar_zsrcinj_train1000.pt", map_location="cpu")
print(f"[KEYS] train1000 {len(tr)} dev600 {len(dev)}")
for name, d in (("train1000", tr), ("dev600", dev)):
    r1 = [v["r_1000"] for v in d.values()]
    print(f"[CONV] {name}: r_1000<1e-2 {sum(x < 1e-2 for x in r1)}/{len(r1)}")
print("[TIMES] decoder/dev outputs (first-modification):")
for p in sorted(glob.glob("results_s7/g2v2/decoder_*") + glob.glob("results_s7/p6r/g2v2dev_*")):
    print("   ", time.strftime("%F %T", time.localtime(os.path.getmtime(p))), p)
print("[TIMES] end")

#!/usr/bin/env bash
# scripts/cc_g2v2_pilot.sh - G2v2 M2′ pilot (CC, 2026-10-08)
# (1) Equivalence: --inject mean on anchors 20,24,32 (GPU 1) must equal results_s7/anchor_ref_gpu1_check.pt z_star (max|Δ|=0).
# (2) Pilot: --inject zsrc on 20 dev items (old held-out 520 subset; dev only) split GPU 0 / GPU 3, save z_k at 0,10,30,100,300,1000.
set -uo pipefail
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
GPU0="GPU-71be98b0-6b81-9257-c622-97dc97d46480"
GPU1="GPU-9410691f-6857-1642-ec7c-7596970e2f48"
GPU3="GPU-a9fed959-7113-e747-dac4-a515ac50c160"
mkdir -p results_s7/g2v2 raw_s7
LOG=raw_s7/g2v2_pilot_master.log
exec > >(tee -a "$LOG") 2>&1
echo "=== g2v2 pilot start $(date '+%F %T') ==="
sha256sum cts/deq/zstar_canonical_lowest.py cts/deq/zstar_inject_v2.py scripts/s7_zstar_lowest_run.py scripts/g2v2_zstar_run.py results_s7/anchor_ref_gpu1_check.pt
for u in $GPU0 $GPU1 $GPU3; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$u" | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$u" | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
CUDA_VISIBLE_DEVICES=$GPU1 $PY scripts/g2v2_zstar_run.py --items 20,24,32 --inject mean --out results_s7/g2v2/anchor_mean_gpu1.pt > raw_s7/g2v2_anchor_mean.log 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=$GPU0 $PY scripts/g2v2_zstar_run.py --items 6,35,38,43,45,67,71,88,93,97 --inject zsrc --save_steps 0,10,30,100,300,1000 --out results_s7/g2v2/pilot_zsrc_a.pt > raw_s7/g2v2_pilot_a.log 2>&1 &
PB=$!
CUDA_VISIBLE_DEVICES=$GPU3 $PY scripts/g2v2_zstar_run.py --items 102,111,116,121,147,173,178,185,187,193 --inject zsrc --save_steps 0,10,30,100,300,1000 --out results_s7/g2v2/pilot_zsrc_b.pt > raw_s7/g2v2_pilot_b.log 2>&1 &
PC=$!
wait $PA; echo "[anchor run rc=$?]"
$PY - <<'PYEOF'
import torch
ref = torch.load("results_s7/anchor_ref_gpu1_check.pt", map_location="cpu")
new = torch.load("results_s7/g2v2/anchor_mean_gpu1.pt", map_location="cpu")
def find(d, q):
    for k in (f"q{q:04d}_a0.25", f"q{q:04d}", q, str(q)):
        if k in d:
            e = d[k]
            return e["z_star"] if isinstance(e, dict) else e
    raise KeyError(q)
print("ref keys sample:", list(ref.keys())[:4])
worst = 0.0
for q in (20, 24, 32):
    a = find(ref, q).float(); b = find(new, q).float()
    dlt = (a - b).abs().max().item(); worst = max(worst, dlt)
    print(f"[EQUIV] Q#{q}: max|d| = {dlt:.6e}")
print(f"[EQUIV] overall max|d| = {worst:.6e} -> {'MATCH' if worst == 0.0 else 'MISMATCH'}")
PYEOF
wait $PB; echo "[pilot a rc=$?]"; wait $PC; echo "[pilot b rc=$?]"
$PY - <<'PYEOF'
import torch, statistics as st
d = {}
for f in ("results_s7/g2v2/pilot_zsrc_a.pt", "results_s7/g2v2/pilot_zsrc_b.pt"):
    d.update(torch.load(f, map_location="cpu"))
def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm()))
rl = [v["r_lowest"] for v in d.values()]; r1 = [v["r_1000"] for v in d.values()]; ks = [v["k_star"] for v in d.values()]
print(f"[PILOT] n={len(d)} r_lowest<1e-2: {sum(x < 1e-2 for x in rl)} | r_1000<1e-2: {sum(x < 1e-2 for x in r1)} | k*>=50: {sum(k >= 50 for k in ks)}")
print(f"[PILOT] r_lowest median {st.median(rl):.3e}, r_1000 median {st.median(r1):.3e}, k* median {st.median(ks)}")
for k in (0, 10, 30, 100, 300, 1000):
    cs = [cos(v["z_k"][k], v["z_src"]) for v in d.values() if k in v["z_k"]]
    print(f"[PILOT] cos(z_{k}, z_src) mean {st.mean(cs):.4f} min {min(cs):.4f}")
cs_star = [cos(v["z_star"], v["z_src"]) for v in d.values()]
print(f"[PILOT] cos(z*, z_src) mean {st.mean(cs_star):.4f}")
keys = list(d.keys()); zs = [d[k]["z_star"] for k in keys]
off = [cos(zs[i], zs[j]) for i in range(len(zs)) for j in range(len(zs)) if i < j]
print(f"[PILOT] cross-item mean cos(z*) {st.mean(off):.4f}")
hit = 0
for i, k in enumerate(keys):
    sims = [cos(zs[i], d[kk]["z_src"]) for kk in keys]
    hit += int(max(range(len(keys)), key=lambda j: sims[j]) == i)
print(f"[PILOT] cross retrieval z* -> z_src: {hit}/{len(keys)}")
PYEOF
echo "=== g2v2 pilot end $(date '+%F %T') ==="

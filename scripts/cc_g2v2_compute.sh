#!/usr/bin/env bash
# scripts/cc_g2v2_compute.sh - G2v2 M2′ fixed points for train1000 + dev600 (CC, 2026-10-08)
# 6 shards on GPU 0,1,3,5,6,7 (UUID). GPU 2 (hardware ERR) and GPU 4 (sealed) are never used.
# Runner: scripts/g2v2_zstar_run.py --inject zsrc (alpha 0.25, g0 0.25, seed 2030, sigma 0.020, max_nfe 1000, tol 0).
set -uo pipefail
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
UUIDS=(GPU-71be98b0-6b81-9257-c622-97dc97d46480 GPU-9410691f-6857-1642-ec7c-7596970e2f48 GPU-a9fed959-7113-e747-dac4-a515ac50c160 GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b)
mkdir -p results_s7/g2v2 raw_s7
LOG=raw_s7/g2v2_compute_master.log
exec > >(tee -a "$LOG") 2>&1
echo "=== g2v2 compute start $(date '+%F %T') ==="
[ "$(sha256sum scripts/g2v2_zstar_run.py | cut -d' ' -f1)" = "687d4480a622c845727a260ebcf49e052142269f10310b016fa82875323b4f2d" ] || { echo "[FATAL] runner SHA"; exit 1; }
[ "$(sha256sum cts/deq/zstar_inject_v2.py | cut -d' ' -f1)" = "7024f1f2d9ff534c3ec43251ac9c8177fa0e384717e8752b330481f6aed91650" ] || { echo "[FATAL] lib SHA"; exit 1; }
for u in "${UUIDS[@]}"; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$u" | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$u" | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
PIDS=()
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=${UUIDS[$i]} $PY scripts/g2v2_zstar_run.py --items configs/g2v2_compute_shard$i.json --inject zsrc \
    --out results_s7/g2v2/zstar_zsrcinj_shard$i.pt > raw_s7/g2v2_compute_shard$i.log 2>&1 &
  PIDS+=($!); echo "[LAUNCH] shard $i on ${UUIDS[$i]} pid ${PIDS[-1]}"
done
rc_all=0
for i in 0 1 2 3 4 5; do wait ${PIDS[$i]}; rc=$?; echo "[DONE] shard $i rc=$rc $(date '+%T')"; [ $rc -eq 0 ] || rc_all=1; done
if [ $rc_all -eq 0 ]; then
  $PY - <<'PYEOF'
import torch, json
out = {}
for i in range(6):
    out.update(torch.load(f"results_s7/g2v2/zstar_zsrcinj_shard{i}.pt", map_location="cpu"))
tr = json.load(open("configs/s7_train_1000.json"))["indices"]; dv = json.load(open("configs/g2v2_dev600.json"))["indices"]
def sub(qs): return {f"q{q:04d}_a0.25": out[f"q{q:04d}_a0.25"] for q in qs}
t, d = sub(tr), sub(dv)
torch.save(t, "results_s7/g2v2/zstar_zsrcinj_train1000.pt"); torch.save(d, "results_s7/g2v2/zstar_zsrcinj_dev600.pt")
for name, dd in (("train1000", t), ("dev600", d)):
    rl = [v["r_lowest"] for v in dd.values()]; r1 = [v["r_1000"] for v in dd.values()]
    print(f"[MERGE] {name}: n={len(dd)} r_1000<1e-2={sum(x<1e-2 for x in r1)} r_lowest<1e-2={sum(x<1e-2 for x in rl)}")
PYEOF
  sha256sum results_s7/g2v2/zstar_zsrcinj_train1000.pt results_s7/g2v2/zstar_zsrcinj_dev600.pt
  touch raw_s7/G2V2_COMPUTE_DONE.flag
fi
echo "=== g2v2 compute end $(date '+%F %T') rc_all=$rc_all ==="

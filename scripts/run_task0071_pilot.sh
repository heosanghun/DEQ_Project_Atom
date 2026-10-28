#!/usr/bin/env bash
# scripts/run_task0071_pilot.sh - G2v3 M3 Joint Pilot Training Pipeline (AG, 2026-10-10)
# Directive: TASK-0071 / GRAPH Node N8 / proposals/G2v3_PLAN_20261009.md
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python

GPU5="GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
GPU6="GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c"

mkdir -p results_s7/g2v3 raw_s7
LOG="raw_s7/task0071_pilot_master.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== TASK-0071 G2v3 M3 Pilot Execution Started at $(date '+%F %T') ==="

# 1. SHA verification of key scripts
echo "[SHA-256 Check]"
sha256sum scripts/g2v3_m3_pilot_train.py results_s7/g2v2/zstar_zsrcinj_train1000.pt results_s7/zsrc_train1000_P4.pt

# 2. Check GPU states
for u in $GPU5 $GPU6; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$u" | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$u" | grep -c '^[0-9]' || true)
  echo "[CHECK] $u memory=${m}MiB apps=${a}"
  if [ "$m" -ge 1024 ] || [ "$a" -ne 0 ]; then
    echo "[FATAL] GPU $u is busy"
    exit 1
  fi
done

# 3. Fast server smoke test
echo "[SMOKE] Running in-memory smoke test on server..."
$PY scripts/g2v3_m3_pilot_train.py --smoke --device cpu --max_steps 5 --batch_size 2
echo "[SMOKE] Server smoke test PASSED."

# 4. Launch Pilot Mode A on GPU 5
echo "[LAUNCH] Mode A (M3 Phantom Gradient) on GPU 5..."
CUDA_VISIBLE_DEVICES=$GPU5 $PY scripts/g2v3_m3_pilot_train.py \
  --mode A \
  --device cuda:0 \
  --max_items 32 \
  --max_steps 50 \
  --batch_size 4 \
  --out results_s7/g2v3/pilot_model_A.pt \
  --manifest results_s7/g2v3/pilot_manifest_A.json > raw_s7/g2v3_pilot_A.log 2>&1 &
PID_A=$!

# 5. Launch Pilot Mode E on GPU 6
echo "[LAUNCH] Mode E (E''' Contrast Baseline) on GPU 6..."
CUDA_VISIBLE_DEVICES=$GPU6 $PY scripts/g2v3_m3_pilot_train.py \
  --mode E \
  --device cuda:0 \
  --max_items 32 \
  --max_steps 50 \
  --batch_size 4 \
  --out results_s7/g2v3/pilot_model_E.pt \
  --manifest results_s7/g2v3/pilot_manifest_E.json > raw_s7/g2v3_pilot_E.log 2>&1 &
PID_E=$!

echo "[WAIT] Waiting for Pilot A (PID $PID_A) and Pilot E (PID $PID_E)..."
wait $PID_A
RC_A=$?
echo "[DONE] Mode A finished with exit code $RC_A at $(date '+%T')"

wait $PID_E
RC_E=$?
echo "[DONE] Mode E finished with exit code $RC_E at $(date '+%T')"

if [ "$RC_A" -ne 0 ] || [ "$RC_E" -ne 0 ]; then
  echo "[FATAL] Pilot execution encountered non-zero return code (A: $RC_A, E: $RC_E)"
  echo "--- Mode A log tail ---"
  tail -n 25 raw_s7/g2v3_pilot_A.log
  echo "--- Mode E log tail ---"
  tail -n 25 raw_s7/g2v3_pilot_E.log
  exit 1
fi

echo "=== Output Artifact Verification ==="
ls -lh results_s7/g2v3/pilot_model_*.pt results_s7/g2v3/pilot_manifest_*.json
sha256sum results_s7/g2v3/pilot_model_*.pt results_s7/g2v3/pilot_manifest_*.json

echo "=== Mode A Manifest Summary ==="
cat results_s7/g2v3/pilot_manifest_A.json
echo ""

echo "=== Mode E Manifest Summary ==="
cat results_s7/g2v3/pilot_manifest_E.json
echo ""

echo "=== TASK-0071 G2v3 M3 Pilot Execution Completed at $(date '+%F %T') ==="

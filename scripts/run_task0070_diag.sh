#!/usr/bin/env bash
# scripts/run_task0070_diag.sh - Master Diagnostic Pipeline for TASK-0070 (D1~D5)
# Directive: TASK-0070 / CTS-430
set -eo pipefail

cd /home/shoon/cts_e1_s5
PYTHON="/home/shoon/system1_5/.venv/bin/python"

mkdir -p raw_s7 results_s7/p6r results_s7/diag

echo "================================================================================"
echo "CTS TASK-0070 Master Diagnostic Pipeline"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo "================================================================================"

# Verify pre-conditions & GPUs
echo "[CHECK] Verifying GPU status (memory < 1 GiB, 0 compute apps)..."
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
COMP_APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v '^\\[N/A\\]' | grep -v '^$' | wc -l)
if [ "$COMP_APPS" -gt 0 ]; then
  echo "[FATAL] Compute apps running ($COMP_APPS apps detected). Aborting."
  exit 1
fi

ORDER_SHA="6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c"

# ==============================================================================
# PHASE 1: D1 Condition A Smoke Test
# ==============================================================================
echo ""
echo "=== [Phase 1] D1 Condition A Smoke Test on GPU 0 ==="

if [ -f results_s7/p6r/diag_nll_smoke_A.jsonl ] && [ "$(wc -l < results_s7/p6r/diag_nll_smoke_A.jsonl)" -eq 600 ]; then
  echo "D1 Condition A Smoke already completed (600 lines, 249s elapsed in run 1)."
else
  t0_smoke=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/s7_eval_gate_p6_v36d.py \
    --cond A \
    --max_new_tokens 1 \
    --nll \
    --order_sha $ORDER_SHA \
    --out results_s7/p6r/diag_nll_smoke_A.jsonl > raw_s7/diag_nll_smoke_A.log 2>&1
  t1_smoke=$(date +%s)
  elapsed_smoke=$((t1_smoke - t0_smoke))
  echo "D1 Condition A Smoke finished in ${elapsed_smoke}s."
fi

SMOKE_LINES=$(wc -l < results_s7/p6r/diag_nll_smoke_A.jsonl)
echo "Smoke test lines: $SMOKE_LINES (expected 600)"
if [ "$SMOKE_LINES" -ne 600 ]; then
  echo "[FATAL] Smoke test line count mismatch: $SMOKE_LINES != 600."
  exit 1
fi

# Use smoke output directly for Condition A
cp results_s7/p6r/diag_nll_smoke_A.jsonl results_s7/p6r/diag_nll_A.jsonl
echo "[Phase 1] Condition A established."

# ==============================================================================
# PHASE 2: Parallel Launch of D2 (CPU), D4 (GPU 1), and D1 Conditions
# ==============================================================================
echo ""
echo "=== [Phase 2] Launching D2 (CPU), D4 (GPU 1), and D1 (GPU 0 & GPU 2) ==="

# 2a. D2: CPU Latent State Geometry
echo "Launching D2 (CPU, nice -n 19, ulimit -v 62914560)..."
nice -n 19 bash -c "ulimit -v 62914560 && $PYTHON scripts/diag_d2_geometry.py --out results_s7/diag/d2_geometry.json" > raw_s7/diag_d2_geometry.log 2>&1 &
PID_D2=$!

# 2b. D4: GPU 1 Picard Trajectory on 40 items (includes anchor check)
echo "Launching D4 on GPU 1 (CUDA_VISIBLE_DEVICES=1)..."
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/diag_d4_traj.py \
  --device cuda:0 \
  --items configs/diag_d4_train40.json \
  --out results_s7/diag/d4_traj_train40.pt > raw_s7/diag_d4_traj.log 2>&1 &
PID_D4=$!

# 2c. D1 Conditions on GPU 3: B1, then D1, then F
echo "Launching D1 Conditions (B1, D1, F) on GPU 3..."
(
  set -eo pipefail
  echo "Starting Condition B1 on GPU 3..."
  CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/s7_eval_gate_p6_v36d.py \
    --cond B --shift 1 --max_new_tokens 1 --nll \
    --order_sha $ORDER_SHA \
    --out results_s7/p6r/diag_nll_B1.jsonl > raw_s7/diag_nll_B1.log 2>&1

  echo "Starting Condition D1 on GPU 3..."
  CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/s7_eval_gate_p6_v36d.py \
    --cond D1 --max_new_tokens 1 --nll \
    --order_sha $ORDER_SHA \
    --out results_s7/p6r/diag_nll_D1.jsonl > raw_s7/diag_nll_D1.log 2>&1

  echo "Starting Condition F on GPU 3..."
  CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/s7_eval_gate_p6_v36d.py \
    --cond F --max_new_tokens 1 --nll \
    --order_sha $ORDER_SHA \
    --out results_s7/p6r/diag_nll_F.jsonl > raw_s7/diag_nll_F.log 2>&1
  echo "GPU 3 conditions finished."
) &
PID_GPU3=$!

# 2d. D1 Conditions on GPU 0: C, then E1
echo "Running Condition C on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/s7_eval_gate_p6_v36d.py \
  --cond C --max_new_tokens 1 --nll \
  --order_sha $ORDER_SHA \
  --out results_s7/p6r/diag_nll_C.jsonl > raw_s7/diag_nll_C.log 2>&1

echo "Running Condition E1 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/s7_eval_gate_p6_v36d.py \
  --cond E1 --max_new_tokens 1 --nll \
  --order_sha $ORDER_SHA \
  --out results_s7/p6r/diag_nll_E1.jsonl > raw_s7/diag_nll_E1.log 2>&1
echo "GPU 0 conditions (A, C, E1) finished."

# ==============================================================================
# PHASE 3: Wait for All Async Tasks
# ==============================================================================
echo ""
echo "=== [Phase 3] Waiting for Async Tasks (D2, D4, GPU 3) ==="
wait $PID_D2
echo "D2 finished."
wait $PID_D4
echo "D4 finished."
wait $PID_GPU3
echo "GPU 3 conditions finished."

# ==============================================================================
# PHASE 4: Post-Processing & Statistical Compilation
# ==============================================================================
echo ""
echo "=== [Phase 4] Computing D1 Statistical Metrics ==="
$PYTHON scripts/diag_d1_stats.py | tee raw_s7/diag_d1_stats.log

echo ""
echo "=== [Phase 5] Verification of All Output Files ==="
ls -lh results_s7/p6r/diag_nll_*.jsonl
ls -lh results_s7/diag/

touch raw_s7/TASK0070_DONE.flag
echo ""
echo "================================================================================"
echo "CTS TASK-0070 Master Diagnostic Pipeline COMPLETED at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

#!/bin/bash
set -euo pipefail

# scripts/run_s7_zstar_pipeline.sh
# Directive: 원장 #367 Point 2
# Execute S7 z* calculation with alpha=0.25 on GPU 0:
# 1. Existing 400 (results_s7/zstar_s7_a0.25_400.pt)
# 2. New 1,200: Train 680 (results_s7/zstar_s7_a0.25_train680.pt) -> Heldout 520 (results_s7/zstar_s7_a0.25_heldout520.pt)
# Zero overwriting, per-item incremental saving and automatic resume.

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

PYTHON="/home/shoon/system1_5/.venv/bin/python"
ALPHA=0.25
DEVICE="cuda:0"
MAX_NFE=1000
RAW_DIR="${BASE_DIR}/raw_s7"
RESULTS_DIR="${BASE_DIR}/results_s7"

mkdir -p "${RESULTS_DIR}" "${RAW_DIR}"

echo "================================================================================"
echo "S7 z* PIPELINE LAUNCH (Alpha: ${ALPHA}, Device: ${DEVICE})"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# Step 1: Existing 400 items
echo "[STEP 1/3] Computing existing 400 items..."
"${PYTHON}" scripts/s6_damping_run.py \
  --alpha ${ALPHA} \
  --items "${BASE_DIR}/configs/s7_items_400.json" \
  --device "${DEVICE}" \
  --out "${RESULTS_DIR}/zstar_s7_a0.25_400.pt" \
  --raw_dir "${RAW_DIR}" \
  --max_nfe ${MAX_NFE}

echo "[STEP 1/3] Step 1 (400 items) completed at $(date '+%Y-%m-%d %H:%M:%S')"

# Step 2: New Train 680 items
echo "[STEP 2/3] Computing new train 680 items..."
"${PYTHON}" scripts/s6_damping_run.py \
  --alpha ${ALPHA} \
  --items "${BASE_DIR}/configs/s7_new_train_680.json" \
  --device "${DEVICE}" \
  --out "${RESULTS_DIR}/zstar_s7_a0.25_train680.pt" \
  --raw_dir "${RAW_DIR}" \
  --max_nfe ${MAX_NFE}

echo "[STEP 2/3] Step 2 (train 680 items) completed at $(date '+%Y-%m-%d %H:%M:%S')"

# Step 3: New Held-out 520 items
echo "[STEP 3/3] Computing new heldout 520 items..."
"${PYTHON}" scripts/s6_damping_run.py \
  --alpha ${ALPHA} \
  --items "${BASE_DIR}/configs/s7_new_heldout_520.json" \
  --device "${DEVICE}" \
  --out "${RESULTS_DIR}/zstar_s7_a0.25_heldout520.pt" \
  --raw_dir "${RAW_DIR}" \
  --max_nfe ${MAX_NFE}

echo "[STEP 3/3] Step 3 (heldout 520 items) completed at $(date '+%Y-%m-%d %H:%M:%S')"

echo "================================================================================"
echo "S7 z* PIPELINE COMPLETED SUCCESSFULLY at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

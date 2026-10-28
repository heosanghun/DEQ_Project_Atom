#!/bin/bash
set -euo pipefail

# scripts/run_s7_zstar_400.sh
# Directive: 원장 #365 Point 2
# Execute S7 400 items z* recalculation with alpha=0.25 on GPU 0
# Saves to new file results_s7/zstar_s7_a0.25_400.pt (Zero overwriting)

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

ALPHA=0.25
ITEMS="${BASE_DIR}/configs/s7_items_400.json"
OUT="${BASE_DIR}/results_s7/zstar_s7_a0.25_400.pt"
RAW="${BASE_DIR}/raw_s7"
DEVICE="cuda:0"
MAX_NFE=1000

mkdir -p "${BASE_DIR}/results_s7" "${BASE_DIR}/raw_s7"

echo "================================================================================"
echo "S7 400 Items z* Recalculation (Alpha: ${ALPHA})"
echo "Device: ${DEVICE}, Items: ${ITEMS}, Max NFE: ${MAX_NFE}"
echo "Target Output: ${OUT}"
echo "================================================================================"

/home/shoon/system1_5/.venv/bin/python scripts/s6_damping_run.py \
  --alpha ${ALPHA} \
  --items "${ITEMS}" \
  --device "${DEVICE}" \
  --out "${OUT}" \
  --raw_dir "${RAW}" \
  --max_nfe ${MAX_NFE}

echo "================================================================================"
echo "S7 400 Items z* Recalculation Completed Successfully"
echo "================================================================================"

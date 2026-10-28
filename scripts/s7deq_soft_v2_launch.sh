#!/usr/bin/env bash
# ==============================================================================
# scripts/s7deq_soft_v2_launch.sh - S7 DEQ Smooth Top-3 Switch Launcher
# Directive: CTS-428 Task 3 / PREREG_S7_DEQ_v2 §2·§5·§6
# Pinned strictly to physical GPU 1 UUID: GPU-9410691f-6857-1642-ec7c-7596970e2f48
# ==============================================================================

set -euo pipefail

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

LOG_FILE="raw_s7/s7deq_soft_v2.log"
mkdir -p raw_s7 results_s7

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "=== S7 DEQ SMOOTH TOP-3 SWITCH LAUNCH ===" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# 1. Verification of Items Config SHA-256
CONFIG_74="configs/s6_items_74.json"
EXPECTED_74_SHA="42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3"

if [ ! -f "${CONFIG_74}" ]; then
  echo "[ERROR] Config file ${CONFIG_74} not found. Aborting." | tee -a "${LOG_FILE}"
  exit 1
fi

ACTUAL_74_SHA=$(sha256sum "${CONFIG_74}" | awk '{print $1}')
echo "Config 74 SHA: ${ACTUAL_74_SHA}" | tee -a "${LOG_FILE}"
if [ "${ACTUAL_74_SHA}" != "${EXPECTED_74_SHA}" ]; then
  echo "[ERROR] Config 74 SHA mismatch. Expected: ${EXPECTED_74_SHA}, Got: ${ACTUAL_74_SHA}. Aborting." | tee -a "${LOG_FILE}"
  exit 1
fi

# 2. Hardware Device Binding
GPU1_UUID="GPU-9410691f-6857-1642-ec7c-7596970e2f48"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "[MODE] SMOKE=1 CPU execution mode enabled." | tee -a "${LOG_FILE}"
  export CUDA_VISIBLE_DEVICES=""
  DEVICE_ARG="cpu"
  MAX_NFE=5
  TEST_ITEMS="20,24"
else
  export CUDA_VISIBLE_DEVICES="${GPU1_UUID}"
  DEVICE_ARG="cuda:0"
  MAX_NFE=1000
  TEST_ITEMS="${CONFIG_74}"

  echo "=== GPU STATUS PRE-LAUNCH ===" | tee -a "${LOG_FILE}"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv | tee -a "${LOG_FILE}"
  nvidia-smi -i 1 --query-gpu=uuid,utilization.gpu,memory.used,temperature.gpu --format=csv | tee -a "${LOG_FILE}"

  # 3. Section 5 Pre-Launch Anchor Check (Code Equivalence on GPU 1)
  echo "=== §5 PRE-LAUNCH ANCHOR CHECK (GPU 1) ===" | tee -a "${LOG_FILE}"
  ANCHOR_ITEMS="20,24,32"
  ANCHOR_REF_PT="results_s7/anchor_ref_gpu1_check.pt"
  ANCHOR_SOFT_PT="results_s7/anchor_soft_eps0.02_check.pt"

  echo "Computing canonical anchor reference on GPU 1 (Q#20, 24, 32)..." | tee -a "${LOG_FILE}"
  /home/shoon/system1_5/.venv/bin/python scripts/s7_zstar_lowest_run.py \
    --items "${ANCHOR_ITEMS}" \
    --out "${ANCHOR_REF_PT}" \
    --device "${DEVICE_ARG}" \
    --max_nfe 1000 >> "${LOG_FILE}" 2>&1

  echo "Computing smooth switch runner with eps=0.02 on GPU 1 (Q#20, 24, 32)..." | tee -a "${LOG_FILE}"
  /home/shoon/system1_5/.venv/bin/python scripts/s7deq_soft_v2_run.py \
    --items "${ANCHOR_ITEMS}" \
    --out "${ANCHOR_SOFT_PT}" \
    --eps 0.02 \
    --device "${DEVICE_ARG}" \
    --max_nfe 1000 >> "${LOG_FILE}" 2>&1

  # Perform verification via python check
  /home/shoon/system1_5/.venv/bin/python - << 'EOF' | tee -a "${LOG_FILE}"
import sys
import torch

ref = torch.load("results_s7/anchor_ref_gpu1_check.pt", map_location="cpu")
soft = torch.load("results_s7/anchor_soft_eps0.02_check.pt", map_location="cpu")

zero_boundary_count = 0
for k in ref.keys():
    b_steps = soft[k].get("boundary_step_count", 0)
    print(f"Item {k}: boundary steps = {b_steps}")
    if b_steps == 0:
        zero_boundary_count += 1
        d_star = torch.max(torch.abs(ref[k]["z_star"] - soft[k]["z_star"])).item()
        d_low = torch.max(torch.abs(ref[k]["z_lowest"] - soft[k]["z_lowest"])).item()
        print(f"Item {k} (b_steps=0): max|Delta_star|={d_star}, max|Delta_lowest|={d_low}")
        if d_star != 0.0 or d_low != 0.0:
            print("[FAIL] Non-zero difference detected on zero boundary item!")
            sys.exit(1)

print(f"Zero boundary count: {zero_boundary_count}")
EOF

fi

# 4. Main Sequence: Run eps in order {0.005, 0.01, 0.02}
EPS_VALUES=(0.005 0.01 0.02)

for EPS in "${EPS_VALUES[@]}"; do
  OUT_PT="results_s7/s7deq_soft_v2_eps${EPS}.pt"
  OUT_JSON="results_s7/s7deq_soft_v2_eps${EPS}.json"
  OUT_MD="results_s7/s7deq_soft_v2_eps${EPS}.md"

  echo "================================================================================" | tee -a "${LOG_FILE}"
  echo "=== RUNNING EPS=${EPS} AT $(date '+%Y-%m-%d %H:%M:%S KST') ===" | tee -a "${LOG_FILE}"
  echo "================================================================================" | tee -a "${LOG_FILE}"

  /home/shoon/system1_5/.venv/bin/python scripts/s7deq_soft_v2_run.py \
    --items "${TEST_ITEMS}" \
    --out "${OUT_PT}" \
    --eps "${EPS}" \
    --alpha 0.25 \
    --g0 0.25 \
    --seed 2030 \
    --sigma 0.020 \
    --max_nfe "${MAX_NFE}" \
    --device "${DEVICE_ARG}" >> "${LOG_FILE}" 2>&1

  echo "Analyzing results for eps=${EPS}..." | tee -a "${LOG_FILE}"
  /home/shoon/system1_5/.venv/bin/python scripts/s7deq_soft_v2_analyze.py \
    --input_pt "${OUT_PT}" \
    --items_json "${CONFIG_74}" \
    --ref_pt "results_s6/zstar_s6_a0.25.pt" \
    --eps "${EPS}" \
    --alpha 0.25 \
    --out_json "${OUT_JSON}" \
    --out_md "${OUT_MD}" >> "${LOG_FILE}" 2>&1

  echo "Completed eps=${EPS} at $(date '+%Y-%m-%d %H:%M:%S KST')" | tee -a "${LOG_FILE}"
done

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "=== ALL S7 DEQ SOFT V2 RUNS FINISHED ===" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

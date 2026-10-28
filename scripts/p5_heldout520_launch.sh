#!/usr/bin/env bash
# scripts/p5_heldout520_launch.sh - P5 Held-Out 520 z* Runner (PREREG_S7_DEQ_v1.2 §10(e)(f))
# Directive: TASK-0006 A.3
# PREPARATION ONLY - EXECUTION STRICTLY PROHIBITED IN TASK-0006 (실행 0)

set -euo pipefail

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

PYTHON="/home/shoon/system1_5/.venv/bin/python"
RAW_DIR="${BASE_DIR}/raw_s7"
RESULTS_DIR="${BASE_DIR}/results_s7"
CONFIGS_DIR="${BASE_DIR}/configs"

mkdir -p "${RAW_DIR}" "${RESULTS_DIR}"

LOG_FILE="${RAW_DIR}/p5_heldout520.log"
OUT_PT="${RESULTS_DIR}/zstar_heldout520_P5.pt"
ITEMS_JSON="${CONFIGS_DIR}/s7_new_heldout_520.json"

echo "================================================================================"
echo "P5 HELDOUT-520 z* COMPUTATION LAUNCHER"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo "Target Output: ${OUT_PT}"
echo "================================================================================"

# ------------------------------------------------------------------------------
# 1. SHA-256 Verification of 6 Critical Prerequisites (PREREG v1.2 §10(e))
# ------------------------------------------------------------------------------
echo "--- [1] Checking SHA-256 for 6 prerequisite files ---"

declare -A EXPECTED_SHAS=(
    ["scripts/s7_zstar_lowest_run.py"]="779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba"
    ["cts/deq/zstar_canonical.py"]="b7170aeb68833448c8bb0977eb7ac035ef06614c9a5edf2f228448c2bb96d45d"
    ["cts/deq/transition_inject.py"]="127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c"
    ["exe_scratch/s7/ref_solvers/anderson.py"]="06c93011058b9dfa83e1846636fa05ae65a117b49d9b35045ba8a59e2a49d7e3"
    ["/home/shoon/checkpoints/m1_final_frozen.pt"]="f2119bbe5c0ad914ee00c34e9109674d15de07c5e028e288ab05b88dd448d1f9"
    ["configs/s7_new_heldout_520.json"]="48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988"
)

for file in "${!EXPECTED_SHAS[@]}"; do
    if [[ ! -f "${file}" ]]; then
        echo "[ERROR] Prerequisite file missing: ${file}" >&2
        exit 1
    fi
    actual_sha=$(sha256sum "${file}" | awk '{print $1}')
    expected_sha="${EXPECTED_SHAS[$file]}"
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
        echo "[ERROR] SHA mismatch on ${file}:" >&2
        echo "  Expected: ${expected_sha}" >&2
        echo "  Actual:   ${actual_sha}" >&2
        exit 1
    fi
    echo "  [PASS] ${file}: ${actual_sha}"
done

# ------------------------------------------------------------------------------
# 2. GPU 0 Availability & Conflict Check
# ------------------------------------------------------------------------------
echo "--- [2] Checking GPU 0 availability and active processes ---"

if pgrep -f "s7_train_decoder" >/dev/null 2>&1; then
    echo "[ERROR] Training process s7_train_decoder is actively running on GPU 0!" >&2
    ps -eo pid,ppid,%cpu,%mem,cmd | grep s7_train_decoder | grep -v grep >&2
    exit 1
fi

if pgrep -f "s7_train_p4_prime" >/dev/null 2>&1; then
    echo "[ERROR] Session s7_train_p4_prime is actively running!" >&2
    exit 1
fi

echo "  [PASS] GPU 0 is free of competing training processes."

# ------------------------------------------------------------------------------
# 3. Canonical Parameters & Execution Command (PREREG v1.2 §10(e))
# ------------------------------------------------------------------------------
ALPHA=0.25
G0=0.25
SIGMA=0.020
SEED=2030
MAX_NFE=1000
DEVICE="cuda:0"

echo "--- [3] Canonical parameters (PREREG_S7_DEQ_v1.2 §10(e)) ---"
echo "  Alpha:     ${ALPHA}"
echo "  g0:        ${G0}"
echo "  Sigma:     ${SIGMA}"
echo "  Seed:      ${SEED}"
echo "  Max NFE:   ${MAX_NFE} (Picard stabilizing)"
echo "  Device:    ${DEVICE}"
echo "  Items:     ${ITEMS_JSON}"
echo "  Output:    ${OUT_PT}"
echo "  Log:       ${LOG_FILE}"

echo "Starting P5 Held-Out 520 z* calculation..."
exec "${PYTHON}" scripts/s7_zstar_lowest_run.py \
    --alpha "${ALPHA}" \
    --g0 "${G0}" \
    --sigma "${SIGMA}" \
    --seed "${SEED}" \
    --max_nfe "${MAX_NFE}" \
    --device "${DEVICE}" \
    --items "${ITEMS_JSON}" \
    --checkpoint "/home/shoon/checkpoints/m1_final_frozen.pt" \
    --out "${OUT_PT}" \
    --raw_dir "${RAW_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

#!/usr/bin/env bash
# scripts/p5_heldout520_launch_v3.sh - P5 Held-Out 520 z* Runner v3 (PREREG_S7_DEQ_v1.2 §10(e)(f))
# Directive: TASK-0007 (AUDIT-0006 F1~F3 & CPU Memory Limit Enforcement)

set -euo pipefail

# ------------------------------------------------------------------------------
# Memory arena and thread controls to respect ulimit -v 62914560 (Rule #381)
# Prevents glibc multi-arena virtual memory explosion during model loading
# ------------------------------------------------------------------------------
export MALLOC_ARENA_MAX=1
export MALLOC_TRIM_THRESHOLD_=131072
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export RAYON_NUM_THREADS=4

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

PYTHON="/home/shoon/system1_5/.venv/bin/python"
RAW_DIR="${BASE_DIR}/raw_s7"
RESULTS_DIR="${BASE_DIR}/results_s7"
CONFIGS_DIR="${BASE_DIR}/configs"
SCRATCH_DIR="${BASE_DIR}/scratch"

mkdir -p "${RAW_DIR}" "${RESULTS_DIR}" "${CONFIGS_DIR}" "${SCRATCH_DIR}"

CONFIG_YAML="${CONFIGS_DIR}/zstar_canonical.yaml"
if [[ ! -f "${CONFIG_YAML}" ]]; then
    CONFIG_YAML="/home/shoon/cts_e1/configs/zstar_canonical.yaml"
fi
ALPHA=0.25
G0=0.25
SIGMA=0.020
SEED=2030
TOL=0.0

if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "================================================================================"
    echo "P5 LAUNCHER v3: SMOKE TEST MODE (CPU, max_nfe=5, 1 train item)"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    ITEMS_JSON="${CONFIGS_DIR}/smoke_train1_p5.json"
    OUT_PT="${SCRATCH_DIR}/p5_smoke.pt"
    LOG_FILE="${SCRATCH_DIR}/p5_smoke.log"
    DEVICE="cpu"
    MAX_NFE=5

    # Absolute safeguard against accidental held-out calculation
    if [[ "${ITEMS_JSON}" =~ heldout ]]; then
        echo "[CRITICAL ERROR] SMOKE=1 must never touch held-out data!" >&2
        exit 1
    fi
else
    echo "================================================================================"
    echo "P5 HELDOUT-520 z* COMPUTATION LAUNCHER v3"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    ITEMS_JSON="${CONFIGS_DIR}/s7_new_heldout_520.json"
    OUT_PT="${RESULTS_DIR}/zstar_s7_a0.25_heldout520_lowest.pt"
    LOG_FILE="${RAW_DIR}/heldout520_P5.log"
    DEVICE="cuda:0"
    MAX_NFE=1000

    # GPU 0 Availability & Conflict Check (Normal Mode only)
    if pgrep -f "s7_train_decoder" >/dev/null 2>&1; then
        echo "[ERROR] Training process s7_train_decoder is actively running on GPU 0!" >&2
        ps -eo pid,ppid,%cpu,%mem,cmd | grep s7_train_decoder | grep -v grep >&2
        exit 1
    fi

    if pgrep -f "s7_train_p4_prime" >/dev/null 2>&1; then
        echo "[ERROR] Session s7_train_p4_prime is actively running!" >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 1. SHA-256 Verification of 7 Prerequisite Files
# ------------------------------------------------------------------------------
echo "--- [1] Checking SHA-256 for 7 prerequisite files ---"

declare -A EXPECTED_SHAS=(
    ["scripts/s7_zstar_lowest_run.py"]="779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba"
    ["cts/deq/zstar_canonical.py"]="b7170aeb68833448c8bb0977eb7ac035ef06614c9a5edf2f228448c2bb96d45d"
    ["cts/deq/transition_inject.py"]="127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c"
    ["exe_scratch/s7/ref_solvers/anderson.py"]="06c93011058b9dfa83e1846636fa05ae65a117b49d9b35045ba8a59e2a49d7e3"
    ["/home/shoon/checkpoints/m1_final_frozen.pt"]="f2119bbe5c0ad914ee00c34e9109674d15de07c5e028e288ab05b88dd448d1f9"
    ["configs/s7_new_heldout_520.json"]="48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988"
    ["PREREG_S7_D_v3_7.md"]="45dbe5cdf62cda06601fb277805f0d1857d508e41d8cf44a15f5a23405fc4be4"
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
# 2. Canonical Parameters & Execution Command (Exact Argparse Conformance)
# ------------------------------------------------------------------------------
echo "--- [2] Execution parameters ---"
echo "  Alpha:     ${ALPHA}"
echo "  g0:        ${G0}"
echo "  Sigma:     ${SIGMA}"
echo "  Seed:      ${SEED}"
echo "  Max NFE:   ${MAX_NFE}"
echo "  Tol:       ${TOL}"
echo "  Device:    ${DEVICE}"
echo "  Items:     ${ITEMS_JSON}"
echo "  Config:    ${CONFIG_YAML}"
echo "  Output:    ${OUT_PT}"
echo "  Log:       ${LOG_FILE}"

echo "Starting s7_zstar_lowest_run.py..."
"${PYTHON}" scripts/s7_zstar_lowest_run.py \
    --items "${ITEMS_JSON}" \
    --out "${OUT_PT}" \
    --alpha "${ALPHA}" \
    --g0 "${G0}" \
    --seed "${SEED}" \
    --sigma "${SIGMA}" \
    --max_nfe "${MAX_NFE}" \
    --tol "${TOL}" \
    --device "${DEVICE}" \
    --config "${CONFIG_YAML}" \
    2>&1 | tee -a "${LOG_FILE}"

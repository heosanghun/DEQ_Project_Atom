#!/usr/bin/env bash
# scripts/p5b_seed2031_launch.sh - P5b Held-Out 520 z* Runner (Seed 2031, PREREG_S7_DEQ_v1.2 §10(e)(f))
# Directive: TASK-0012 Section 2 (Seed 2031, Option [A] Replication, GPU 4 Sealed)

set -euo pipefail

# ------------------------------------------------------------------------------
# Memory arena and thread controls to respect ulimit -v 62914560 (Rule #381)
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

HARNESS_PY="/home/shoon/cts_e1/scratch/harness_A_v1.py"
if [[ ! -f "${HARNESS_PY}" ]]; then
    HARNESS_PY="${BASE_DIR}/scratch/harness_A_v1.py"
fi

ALPHA=0.25
G0=0.25
SIGMA=0.020
SEED=2031
TOL=0.0

# ------------------------------------------------------------------------------
# Parse command-line device argument
# ------------------------------------------------------------------------------
CLI_DEVICE="cuda:0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            CLI_DEVICE="$2"
            shift 2
            ;;
        --device=*)
            CLI_DEVICE="${1#*=}"
            shift 1
            ;;
        *)
            shift 1
            ;;
    esac
done

# Absolute ban on GPU 4 (Rule 09-02, Ledger #203)
if [[ "${CLI_DEVICE}" == *"4"* ]]; then
    echo "[CRITICAL ERROR] GPU 4 is strictly sealed (Rule 09-02, Ledger #203)! Access rejected: ${CLI_DEVICE}" >&2
    exit 1
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "================================================================================"
    echo "P5b LAUNCHER: SMOKE TEST MODE (CPU, seed=2031, max_nfe=5, 1 train item)"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    ITEMS_JSON="${CONFIGS_DIR}/smoke_train1_p5.json"
    OUT_PT="${SCRATCH_DIR}/p5b_smoke.pt"
    LOG_FILE="${SCRATCH_DIR}/p5b_smoke.log"
    DEVICE="cpu"
    MAX_NFE=5
    STARTED_FLAG="${SCRATCH_DIR}/P5B_STARTED.flag"
    DONE_FLAG="${SCRATCH_DIR}/P5B_DONE.flag"

    # Absolute safeguard against accidental held-out calculation in smoke mode
    if [[ "${ITEMS_JSON}" =~ heldout ]]; then
        echo "[CRITICAL ERROR] SMOKE=1 must never touch held-out data!" >&2
        exit 1
    fi
else
    echo "================================================================================"
    echo "P5b HELDOUT-520 z* COMPUTATION LAUNCHER (SEED 2031)"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Device: ${CLI_DEVICE}"
    echo "================================================================================"
    ITEMS_JSON="${CONFIGS_DIR}/s7_new_heldout_520.json"
    OUT_PT="${RESULTS_DIR}/zstar_s7_a0.25_heldout520_seed2031.pt"
    LOG_FILE="${RAW_DIR}/heldout520_P5b_seed2031.log"
    DEVICE="${CLI_DEVICE}"
    MAX_NFE=1000
    STARTED_FLAG="${RAW_DIR}/P5B_STARTED.flag"
    DONE_FLAG="${RAW_DIR}/P5B_DONE.flag"

    # Active training conflicts check
    if pgrep -f "s7_train_decoder" >/dev/null 2>&1; then
        # If running on cuda:0, prevent conflict with training
        if [[ "${DEVICE}" == "cuda:0" ]]; then
            echo "[ERROR] Training process s7_train_decoder is actively running on GPU 0!" >&2
            ps -eo pid,ppid,%cpu,%mem,cmd | grep s7_train_decoder | grep -v grep >&2
            exit 1
        fi
    fi
fi

# ------------------------------------------------------------------------------
# 1. SHA-256 Verification of 9 Prerequisite Files (Identical to v4)
# ------------------------------------------------------------------------------
echo "--- [1] Checking SHA-256 for 9 prerequisite files ---"

declare -A EXPECTED_SHAS=(
    ["scripts/s7_zstar_lowest_run.py"]="779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba"
    ["cts/deq/zstar_canonical.py"]="b7170aeb68833448c8bb0977eb7ac035ef06614c9a5edf2f228448c2bb96d45d"
    ["cts/deq/transition_inject.py"]="127068435ec561c36a5e8a5e2e208763b80e53a70853ab50528ede7f84f2997c"
    ["exe_scratch/s7/ref_solvers/anderson.py"]="06c93011058b9dfa83e1846636fa05ae65a117b49d9b35045ba8a59e2a49d7e3"
    ["/home/shoon/checkpoints/m1_final_frozen.pt"]="f2119bbe5c0ad914ee00c34e9109674d15de07c5e028e288ab05b88dd448d1f9"
    ["configs/s7_new_heldout_520.json"]="48abce650791c781da254a490b0f25e60dfc2a777bc77053eaf20e23b1176988"
    ["PREREG_S7_D_v3_7.md"]="45dbe5cdf62cda06601fb277805f0d1857d508e41d8cf44a15f5a23405fc4be4"
    ["${CONFIG_YAML}"]="bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157"
    ["${HARNESS_PY}"]="5fb768a8f56add15c7cdca069f5e1a7f80d16fd231f8ecae1fca4fcf6920bf52"
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
# 2. Start Flag Governance (Double-Run Prevention / Resume Check)
# ------------------------------------------------------------------------------
echo "--- [2] Checking Start Flag Governance ---"
if [[ -f "${STARTED_FLAG}" ]]; then
    if [[ "${P5B_RESUME:-0}" == "1" ]] && [[ -f "${OUT_PT}" ]]; then
        echo "  [RESUME] P5B_RESUME=1 set and output file exists (${OUT_PT}). Resuming..."
    else
        echo "[ERROR] ${STARTED_FLAG} already exists! Double-run prevented." >&2
        echo "  Existing start record: $(cat "${STARTED_FLAG}")" >&2
        echo "  To resume existing calculation on ${OUT_PT}, explicitly pass P5B_RESUME=1" >&2
        exit 1
    fi
fi

LAUNCHER_SHA=$(sha256sum "$0" | awk '{print $1}')
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "START_TIME=\"${START_TIMESTAMP}\" HOST=\"$(hostname)\" LAUNCHER_SHA=\"${LAUNCHER_SHA}\" TARGET=\"${OUT_PT}\" SEED=\"${SEED}\" DEVICE=\"${DEVICE}\"" > "${STARTED_FLAG}"
echo "  [WRITTEN] ${STARTED_FLAG}: $(cat "${STARTED_FLAG}")"

# ------------------------------------------------------------------------------
# 3. Execution Parameters & Command
# ------------------------------------------------------------------------------
echo "--- [3] Execution parameters ---"
echo "  Alpha:     ${ALPHA}"
echo "  g0:        ${G0}"
echo "  Sigma:     ${SIGMA}"
echo "  Seed:      ${SEED}"
echo "  Max NFE:   ${MAX_NFE}"
echo "  Tol:       ${TOL}"
echo "  Device:    ${DEVICE}"
echo "  Items:     ${ITEMS_JSON}"
echo "  Config:    ${CONFIG_YAML}"
echo "  Harness:   ${HARNESS_PY}"
echo "  Output:    ${OUT_PT}"
echo "  Log:       ${LOG_FILE}"

echo "Starting s7_zstar_lowest_run.py (Seed ${SEED})..."
set +e
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
PY_RC=${PIPESTATUS[0]}
set -e

if [[ ${PY_RC} -eq 0 ]]; then
    DONE_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "DONE_TIME=\"${DONE_TIMESTAMP}\" HOST=\"$(hostname)\" TARGET=\"${OUT_PT}\" LAUNCHER_SHA=\"${LAUNCHER_SHA}\" SEED=\"${SEED}\"" > "${DONE_FLAG}"
    echo "  [WRITTEN] ${DONE_FLAG}: $(cat "${DONE_FLAG}")"
    echo "P5b (seed 2031) computation finished with code 0."
else
    echo "[ERROR] s7_zstar_lowest_run.py exited with non-zero code ${PY_RC}" >&2
    exit ${PY_RC}
fi

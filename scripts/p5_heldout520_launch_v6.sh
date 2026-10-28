#!/usr/bin/env bash
# scripts/p5_heldout520_launch_v6.sh - P5 Held-Out 520 z* Runner v6 (SEED 2030, RESUME Support, Multi-Device Ready)
# Directive: TASK-0025 / PREREG_S7_DEQ_v1.3 §10(f) / ESC-0005
# Allowed devices: cuda:1, cuda:2, or cuda:0 depending on ESC-0005 resolution. GPU 4 strictly sealed.

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
SEED=2030
TOL=0.0

# ------------------------------------------------------------------------------
# Parse command-line device argument (cuda:1, cuda:2, or cuda:0 allowed)
# ------------------------------------------------------------------------------
CLI_DEVICE="cuda:1"
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
    echo "P5 LAUNCHER v6: SMOKE TEST MODE (CPU, seed=2030, max_nfe=5, 1 train item)"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    ITEMS_JSON="${CONFIGS_DIR}/smoke_train1_p5.json"
    OUT_PT="${SCRATCH_DIR}/p5_smoke.pt"
    LOG_FILE="${SCRATCH_DIR}/p5_smoke.log"
    DEVICE="cpu"
    MAX_NFE=5
    STARTED_FLAG="${SCRATCH_DIR}/P5_STARTED.flag"
    RESUMED_FLAG="${SCRATCH_DIR}/P5_RESUMED.flag"
    DONE_FLAG="${SCRATCH_DIR}/P5_DONE.flag"

    # Absolute safeguard against accidental held-out calculation in smoke mode
    if [[ "${ITEMS_JSON}" =~ heldout ]]; then
        echo "[CRITICAL ERROR] SMOKE=1 must never touch held-out data!" >&2
        exit 1
    fi
    rm -f "${SCRATCH_DIR}/p5_smoke.log" "${SCRATCH_DIR}/P5_DONE.flag"
    if [[ "${P5_RESUME:-0}" != "1" ]]; then
        rm -f "${SCRATCH_DIR}/P5_STARTED.flag" "${SCRATCH_DIR}/P5_RESUMED.flag" "${SCRATCH_DIR}/p5_smoke.pt"
    fi
else
    # Multi-device allowance for ESC-0005: cuda:1, cuda:2, or cuda:0
    if [[ "${CLI_DEVICE}" != "cuda:1" && "${CLI_DEVICE}" != "cuda:2" && "${CLI_DEVICE}" != "cuda:0" ]]; then
        echo "[CRITICAL ERROR] p5_heldout520_launch_v6 requires --device cuda:1, cuda:2, or cuda:0 (got: ${CLI_DEVICE})!" >&2
        exit 1
    fi

    echo "================================================================================"
    echo "P5 HELDOUT-520 z* COMPUTATION LAUNCHER v6 (SEED 2030, RESUME SUPPORT)"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Device: ${CLI_DEVICE}"
    echo "================================================================================"
    ITEMS_JSON="${CONFIGS_DIR}/s7_new_heldout_520.json"
    OUT_PT="${RESULTS_DIR}/zstar_s7_a0.25_heldout520_lowest.pt"
    LOG_FILE="${RAW_DIR}/heldout520_P5.log"
    DEVICE="${CLI_DEVICE}"
    MAX_NFE=1000
    STARTED_FLAG="${RAW_DIR}/P5_STARTED.flag"
    RESUMED_FLAG="${RAW_DIR}/P5_RESUMED.flag"
    DONE_FLAG="${RAW_DIR}/P5_DONE.flag"

    # Target GPU Availability Check
    TARGET_GPU_INDEX="${CLI_DEVICE#cuda:}"
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${TARGET_GPU_INDEX}" 2>/dev/null | tr -d ' ' || echo "0")
    if [[ -n "${GPU_MEM}" ]] && [[ "${GPU_MEM}" -gt 1024 ]]; then
        echo "[ERROR] GPU ${TARGET_GPU_INDEX} is not idle! Memory used: ${GPU_MEM} MiB" >&2
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 1. SHA-256 Verification of 9 Prerequisite Files
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
    ACTUAL_SHA=$(sha256sum "${file}" | awk '{print $1}')
    EXPECTED_SHA="${EXPECTED_SHAS[${file}]}"
    if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
        echo "[ERROR] SHA mismatch for ${file}!" >&2
        echo "  Expected: ${EXPECTED_SHA}" >&2
        echo "  Actual:   ${ACTUAL_SHA}" >&2
        exit 1
    fi
    echo "  [OK] ${file}"
done

# ------------------------------------------------------------------------------
# 2. Flag Governance & Resume Enforcement (TASK-0025 §2)
# ------------------------------------------------------------------------------
echo "--- [2] Checking Execution Flags & Resume Governance ---"
if [[ -f "${DONE_FLAG}" ]]; then
    echo "[INFO] DONE flag already exists: ${DONE_FLAG}"
    echo "P5 computation is already completed. Halting to prevent accidental overwrite."
    exit 0
fi

if [[ "${P5_RESUME:-0}" == "1" ]]; then
    echo "[RESUME] P5_RESUME=1 specified."
    if [[ ! -f "${STARTED_FLAG}" ]]; then
        echo "[ERROR] P5_RESUME=1 requires STARTED flag to exist: ${STARTED_FLAG}" >&2
        exit 1
    fi
    if [[ ! -f "${OUT_PT}" ]]; then
        echo "[ERROR] P5_RESUME=1 requires output file to exist: ${OUT_PT}" >&2
        exit 1
    fi
    if pgrep -f "s7_zstar_lowest_run.py.*${OUT_PT}" >/dev/null 2>&1; then
        echo "[ERROR] Another instance of s7_zstar_lowest_run.py is currently running for ${OUT_PT}!" >&2
        exit 1
    fi

    OUT_SHA=$(sha256sum "${OUT_PT}" | awk '{print $1}')
    KEY_COUNT=$("${PYTHON}" -c "import torch; print(len(torch.load('${OUT_PT}', map_location='cpu')))")
    RESUME_TS=$(date '+%Y-%m-%d %H:%M:%S')

    cat << EOF > "${RESUMED_FLAG}"
resumed_at: ${RESUME_TS}
output_file: ${OUT_PT}
output_sha256: ${OUT_SHA}
key_count: ${KEY_COUNT}
device: ${DEVICE}
EOF
    echo "Created RESUMED flag: ${RESUMED_FLAG}"
    echo "  Resumed at:     ${RESUME_TS}"
    echo "  Output file:    ${OUT_PT}"
    echo "  Output SHA-256: ${OUT_SHA}"
    echo "  Existing keys:  ${KEY_COUNT}"
    echo "  Device:         ${DEVICE}"
else
    if [[ -f "${STARTED_FLAG}" ]]; then
        echo "[ERROR] STARTED flag exists (${STARTED_FLAG}) but P5_RESUME is not 1! Refusing to overwrite." >&2
        exit 1
    else
        echo "Creating STARTED flag: ${STARTED_FLAG}"
        date '+%Y-%m-%d %H:%M:%S' > "${STARTED_FLAG}"
    fi
fi

# ------------------------------------------------------------------------------
# 3. Execution
# ------------------------------------------------------------------------------
echo "--- [3] Launching s7_zstar_lowest_run.py (Seed 2030) ---"
echo "Command: ${PYTHON} scripts/s7_zstar_lowest_run.py --items ${ITEMS_JSON} --out ${OUT_PT} --alpha ${ALPHA} --g0 ${G0} --seed ${SEED} --sigma ${SIGMA} --max_nfe ${MAX_NFE} --tol ${TOL} --device ${DEVICE} --config ${CONFIG_YAML}"

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
    echo "--- [4] Execution finished successfully (RC=0) ---"
    date '+%Y-%m-%d %H:%M:%S' > "${DONE_FLAG}"
    echo "Created DONE flag: ${DONE_FLAG}"
    if [[ -f "${OUT_PT}" ]]; then
        OUT_SHA=$(sha256sum "${OUT_PT}" | awk '{print $1}')
        OUT_SIZE=$(stat -c%s "${OUT_PT}")
        echo "Output Tensor: ${OUT_PT}"
        echo "  Size: ${OUT_SIZE} bytes"
        echo "  SHA:  ${OUT_SHA}"
    fi
    exit 0
else
    echo "[CRITICAL ERROR] s7_zstar_lowest_run.py failed with return code ${PY_RC}!" >&2
    exit ${PY_RC}
fi

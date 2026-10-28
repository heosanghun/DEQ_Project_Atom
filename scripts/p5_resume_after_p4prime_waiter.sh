#!/usr/bin/env bash
# scripts/p5_resume_after_p4prime_waiter.sh - P5 Auto-Resume Waiter after P4' on GPU 0
# Directive: TASK-0033 / PREREG_S7_DEQ_v1.3 §10(f) / AUDIT-0032
# Monitors P4' training termination and launches P5 held-out 520 resume on GPU 0.

set -euo pipefail

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

RAW_DIR="${BASE_DIR}/raw_s7"
LOG_FILE="${RAW_DIR}/p5_resume_waiter.log"
CANCEL_FLAG="${RAW_DIR}/P5_RESUME_CANCEL.flag"
ABNORMAL_FLAG="${RAW_DIR}/P4PRIME_ABNORMAL.flag"
P5_DONE_FLAG="${RAW_DIR}/P5_DONE.flag"

LOG_D="${RAW_DIR}/train_decoder_D_v36.log"
LOG_E="${RAW_DIR}/train_decoder_E_v36.log"

TARGET_GPU_UUID="GPU-71be98b0-6b81-9257-c622-97dc97d46480"

mkdir -p "${RAW_DIR}"

echo "[WAITER] Starting p5_resume_after_p4prime_waiter.sh (PID: $$, Started: $(date '+%Y-%m-%d %H:%M:%S'))" | tee -a "${LOG_FILE}"

run_single_check() {
    # 0. Check cancellation flag
    if [[ -f "${CANCEL_FLAG}" ]]; then
        echo "[CANCEL] $(date '+%Y-%m-%d %H:%M:%S') P5_RESUME_CANCEL.flag detected. Halting waiter." | tee -a "${LOG_FILE}"
        exit 0
    fi

    # (a) Check if s7_train_p4_prime* runners and s7_train_decoder processes are 0
    local p4_procs
    p4_procs=$(pgrep -f "s7_train_p4_prime.*\.sh|s7_train_decoder" 2>/dev/null | wc -l || echo "0")

    # (b) Check final checkpoint save lines in D and E logs
    local d_done=0
    local e_done=0
    if [[ -f "${LOG_D}" ]] && grep -q "Saved final model to results_s7/decoder_D_v36.pt" "${LOG_D}" 2>/dev/null; then
        d_done=1
    fi
    if [[ -f "${LOG_E}" ]] && grep -q "Saved final model to results_s7/decoder_E_v36.pt" "${LOG_E}" 2>/dev/null; then
        e_done=1
    fi
    local de_done=0
    if [[ ${d_done} -eq 1 ]] && [[ ${e_done} -eq 1 ]]; then
        de_done=1
    fi

    # (c) Check GPU 0 used memory < 1024 MiB
    local gpu0_mem
    gpu0_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ' || echo "99999")
    local gpu0_idle=0
    if [[ -n "${gpu0_mem}" ]] && [[ "${gpu0_mem}" -lt 1024 ]]; then
        gpu0_idle=1
    fi

    # (d) Check raw_s7/P5_DONE.flag does not exist
    local p5_not_done=0
    if [[ ! -f "${P5_DONE_FLAG}" ]]; then
        p5_not_done=1
    fi

    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')

    # Status summary line logged every check
    echo "[CHECK ${ts}] procs=${p4_procs} | de_done=${de_done} (D=${d_done}, E=${e_done}) | gpu0_mem=${gpu0_mem}MiB (idle=${gpu0_idle}) | p5_not_done=${p5_not_done} | state=WAITING" | tee -a "${LOG_FILE}"

    # Check abnormal stop condition: (a) is true (procs == 0) but (b) is false (de_done == 0)
    if [[ ${p4_procs} -eq 0 ]] && [[ ${de_done} -eq 0 ]]; then
        echo "[ABNORMAL] $(date '+%Y-%m-%d %H:%M:%S') P4' processes 0 but checkpoints incomplete (D=${d_done}, E=${e_done}). Writing P4PRIME_ABNORMAL.flag." | tee -a "${LOG_FILE}"
        echo "ABNORMAL_STOP at $(date '+%Y-%m-%d %H:%M:%S') | D_done=${d_done} | E_done=${e_done} | gpu0_mem=${gpu0_mem}" > "${ABNORMAL_FLAG}"
        exit 1
    fi

    # Trigger condition: (a), (b), (c), (d) all true
    if [[ ${p4_procs} -eq 0 ]] && [[ ${de_done} -eq 1 ]] && [[ ${gpu0_idle} -eq 1 ]] && [[ ${p5_not_done} -eq 1 ]]; then
        echo "[TRIGGER] $(date '+%Y-%m-%d %H:%M:%S') All 4 conditions satisfied. Launching P5 resume on GPU 0 UUID..." | tee -a "${LOG_FILE}"
        set +e
        P5_RESUME=1 bash scripts/p5_heldout520_launch_v8.sh --device "cuda:0" 2>&1 | tee -a "${LOG_FILE}"
        local p5_rc=${PIPESTATUS[0]}
        set -e
        echo "[TRIGGER] $(date '+%Y-%m-%d %H:%M:%S') P5 resume execution exited with code ${p5_rc}." | tee -a "${LOG_FILE}"
        exit ${p5_rc}
    fi
}

# Run immediately once, then loop every 300 seconds
while true; do
    run_single_check
    sleep 300
done

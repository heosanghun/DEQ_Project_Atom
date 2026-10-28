#!/usr/bin/env bash
# scripts/p5_after_p4prime_waiter.sh - Daemon waiting for P4' completion to trigger P5
# Directive: TASK-0008 (Section 2)

set -uo pipefail

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

LOG_FILE="${BASE_DIR}/raw_s7/p5_waiter.log"
ABNORMAL_FLAG="${BASE_DIR}/raw_s7/P4PRIME_ABNORMAL.flag"
mkdir -p "${BASE_DIR}/raw_s7"

echo "=== P5 Waiter Daemon started at $(date '+%Y-%m-%d %H:%M:%S') ===" >> "${LOG_FILE}"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # (a) Process count check
    count_all=$(pgrep -f "s7_train_p4_prime_all" | wc -l)
    count_dec=$(pgrep -f "s7_train_decoder" | wc -l)
    if [[ "${count_all}" -eq 0 && "${count_dec}" -eq 0 ]]; then
        is_a=true
    else
        is_a=false
    fi

    # (b) Log completion string & .pt existence check
    has_log_a=false
    has_log_d=false
    has_log_e=false
    has_pt_a=false
    has_pt_d=false
    has_pt_e=false

    if [[ -f "raw_s7/train_decoder_A_v36.log" ]] && grep -q "\[OUT-CHECKPOINT\] Saved final model to results_s7/decoder_A_v36.pt" "raw_s7/train_decoder_A_v36.log"; then
        has_log_a=true
    fi
    if [[ -f "raw_s7/train_decoder_D_v36.log" ]] && grep -q "\[OUT-CHECKPOINT\] Saved final model to results_s7/decoder_D_v36.pt" "raw_s7/train_decoder_D_v36.log"; then
        has_log_d=true
    fi
    if [[ -f "raw_s7/train_decoder_E_v36.log" ]] && grep -q "\[OUT-CHECKPOINT\] Saved final model to results_s7/decoder_E_v36.pt" "raw_s7/train_decoder_E_v36.log"; then
        has_log_e=true
    fi

    if [[ -f "results_s7/decoder_A_v36.pt" ]]; then has_pt_a=true; fi
    if [[ -f "results_s7/decoder_D_v36.pt" ]]; then has_pt_d=true; fi
    if [[ -f "results_s7/decoder_E_v36.pt" ]]; then has_pt_e=true; fi

    if [[ "${has_log_a}" == "true" && "${has_log_d}" == "true" && "${has_log_e}" == "true" && \
          "${has_pt_a}" == "true" && "${has_pt_d}" == "true" && "${has_pt_e}" == "true" ]]; then
        is_b=true
    else
        is_b=false
    fi

    # (c) GPU 0 used memory < 1 GiB (1024 MiB)
    gpu0_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | awk '{print $1}')
    if [[ -n "${gpu0_mem}" && "${gpu0_mem}" -lt 1024 ]]; then
        is_c=true
    else
        is_c=false
    fi

    STATUS_LINE="[CHECK ${TIMESTAMP}] a=${is_a} (all=${count_all}, dec=${count_dec}) | b=${is_b} (logs=[A:${has_log_a}, D:${has_log_d}, E:${has_log_e}], pts=[A:${has_pt_a}, D:${has_pt_d}, E:${has_pt_e}]) | c=${is_c} (gpu0_mem=${gpu0_mem}MiB)"
    echo "${STATUS_LINE}" | tee -a "${LOG_FILE}"

    # Decision tree
    if [[ "${is_a}" == "true" ]]; then
        if [[ "${is_b}" == "true" ]]; then
            if [[ "${is_c}" == "true" ]]; then
                echo "[TRIGGER ${TIMESTAMP}] Conditions (a)(b)(c) all satisfied. Launching scripts/p5_heldout520_launch_v4.sh..." | tee -a "${LOG_FILE}"
                bash scripts/p5_heldout520_launch_v4.sh >> "${LOG_FILE}" 2>&1
                LAUNCH_RC=$?
                echo "[EXIT ${TIMESTAMP}] P5 launcher exited with code ${LAUNCH_RC}. Waiter daemon terminating." | tee -a "${LOG_FILE}"
                exit ${LAUNCH_RC}
            else
                echo "[WAIT ${TIMESTAMP}] (a) and (b) true, but GPU 0 memory (${gpu0_mem}MiB >= 1024MiB). Waiting 30s..." | tee -a "${LOG_FILE}"
                sleep 30
                continue
            fi
        else
            echo "[ABNORMAL ${TIMESTAMP}] P4' processes finished (a=true), but outputs incomplete (b=false). Writing ${ABNORMAL_FLAG} and aborting." | tee -a "${LOG_FILE}"
            echo "ABNORMAL_FINISH TIME=\"${TIMESTAMP}\" HOST=\"$(hostname)\" DETAILS=\"logs=[A:${has_log_a}, D:${has_log_d}, E:${has_log_e}], pts=[A:${has_pt_a}, D:${has_pt_d}, E:${has_pt_e}]\"" > "${ABNORMAL_FLAG}"
            exit 1
        fi
    fi

    sleep 300
done

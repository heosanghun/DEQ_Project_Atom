#!/usr/bin/env bash
# scripts/p6_chain_after_p5.sh - Automated P5 -> z_src 600 -> P6 Gate Evaluation Pipeline Chain
# Directive: TASK-0011 Section 2 / PREREG_S7_DEQ_v1.2 §10(c)(d) / PREREG_S7_D_v3_7 §5·§8(b)
#
# Execution Architecture:
#   Runs inside tmux session 'p6_chain' on remote host.
#   Monitors for raw_s7/P5_DONE.flag and GPU 0 idle (< 1024 MiB).
#
# Execution Chain:
#   1. z_src(600) Computation:
#      bash scripts/zsrc_heldout600_launch.sh -> raw_s7/ZSRC600_DONE.flag
#   2. P6 Generation (Sequential Order per PREREG_S7_D_v3_7 §8(b)):
#      C -> A -> B(s=1) -> E1 -> F -> D1 -> D2(s=1) -> E2(s=1) -> B(s=2) -> B(s=3) -> D2(s=2) -> D2(s=3)
#      Each step leaves raw_s7/P6_GEN_{cond}{shift}_DONE.flag
#   3. Scoring:
#      - Main Gates (C, A, B1, E1): raw_s7/P6_SCORE_MAIN_DONE.flag -> results_s7/p6_gate_table_main.json
#      - Full Gates (all conditions): raw_s7/P6_SCORE_FULL_DONE.flag -> results_s7/p6_gate_table_full.json
#   4. Auxiliary NLL Mode:
#      A, B1, C, E1 with --nll flag -> raw_s7/P6_NLL_{cond}_DONE.flag
#
# Error Governance:
#   On any non-zero exit code, immediately logs stage, rc, and tail 20 lines to raw_s7/P6_CHAIN_ERROR.flag and halts.
#   Zero retries on error.

set -euo pipefail

BASE_DIR="/home/shoon/cts_e1_s5"
RAW_DIR="${BASE_DIR}/raw_s7"
RESULTS_DIR="${BASE_DIR}/results_s7"
SCRIPTS_DIR="${BASE_DIR}/scripts"
LOG_DIR="${RAW_DIR}"

PYTHON="/home/shoon/system1_5/.venv/bin/python"
PYTHON_EVAL="/home/shoon/venv_eval/bin/python"

mkdir -p "${RAW_DIR}" "${RESULTS_DIR}"

log_error_and_halt() {
    local stage="$1"
    local rc="$2"
    local log_file="$3"
    echo "========================================================" > "${RAW_DIR}/P6_CHAIN_ERROR.flag"
    echo "[P6-CHAIN-ERROR] Stage ${stage} failed with return code ${rc}" >> "${RAW_DIR}/P6_CHAIN_ERROR.flag"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> "${RAW_DIR}/P6_CHAIN_ERROR.flag"
    echo "Log file: ${log_file}" >> "${RAW_DIR}/P6_CHAIN_ERROR.flag"
    echo "Last 20 log lines:" >> "${RAW_DIR}/P6_CHAIN_ERROR.flag"
    if [ -f "${log_file}" ]; then
        tail -n 20 "${log_file}" >> "${RAW_DIR}/P6_CHAIN_ERROR.flag"
    else
        echo "(Log file not found: ${log_file})" >> "${RAW_DIR}/P6_CHAIN_ERROR.flag"
    fi
    echo "========================================================" >> "${RAW_DIR}/P6_CHAIN_ERROR.flag"
    echo "[P6-CHAIN] Error recorded in ${RAW_DIR}/P6_CHAIN_ERROR.flag. Halting execution." >&2
    exit "${rc}"
}

is_gpu0_idle() {
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ' || echo "99999")
    if [ -n "$mem" ] && [ "$mem" -lt 1024 ]; then
        return 0
    else
        return 1
    fi
}

echo "================================================================================"
echo "CTS S7 P6 Pipeline Chain (scripts/p6_chain_after_p5.sh)"
echo "Host: $(hostname) | PID: $$ | Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# ==============================================================================
# Phase 0: Wait for P5_DONE.flag and GPU 0 Idle
# ==============================================================================
if [ "${SMOKE:-0}" != "1" ]; then
    echo "[P6-CHAIN] Monitoring for P5_DONE.flag and GPU 0 idle (< 1024 MiB)..."
    while true; do
        if [ -f "${RAW_DIR}/P5_DONE.flag" ]; then
            if is_gpu0_idle; then
                echo "[P6-CHAIN] P5_DONE.flag present and GPU 0 is idle. Commencing chain execution."
                break
            else
                echo "[P6-CHAIN $(date '+%Y-%m-%d %H:%M:%S')] P5_DONE.flag exists, but GPU 0 memory is occupied. Waiting 300s..."
            fi
        else
            echo "[P6-CHAIN $(date '+%Y-%m-%d %H:%M:%S')] Waiting for P5_DONE.flag... (300s sleep)"
        fi
        sleep 300
    done
fi

# ==============================================================================
# Phase 1: z_src(600) Launch
# ==============================================================================
ZSRC_DONE_FLAG="${RAW_DIR}/ZSRC600_DONE.flag"
if [ -f "${ZSRC_DONE_FLAG}" ]; then
    echo "[P6-CHAIN] Stage 1 (z_src 600) already completed (${ZSRC_DONE_FLAG} exists). Skipping."
else
    echo "[P6-CHAIN] Stage 1: Launching z_src(600) calculation..."
    LOG_ZSRC="${LOG_DIR}/zsrc_heldout600.log"
    set +e
    bash "${SCRIPTS_DIR}/zsrc_heldout600_launch.sh" > "${LOG_ZSRC}" 2>&1
    RC=$?
    set -e
    if [ "${RC}" -ne 0 ]; then
        log_error_and_halt "zsrc_heldout600" "${RC}" "${LOG_ZSRC}"
    fi
    touch "${ZSRC_DONE_FLAG}"
    echo "[P6-CHAIN] Stage 1 completed successfully. Created ${ZSRC_DONE_FLAG}."
fi

# Helper function to execute a single generation condition
run_generation_condition() {
    local cond="$1"
    local shift="$2"
    local flag_name="$3"
    local done_flag="${RAW_DIR}/${flag_name}"
    local log_file="${LOG_DIR}/p6_gen_${cond}${shift}.log"

    if [ -f "${done_flag}" ]; then
        echo "[P6-CHAIN] Generation Condition ${cond} (shift=${shift}) already completed (${done_flag} exists). Skipping."
        return 0
    fi

    echo "[P6-CHAIN] Running Generation Condition ${cond} (shift=${shift})..."
    local extra_args=()
    if [ -n "${shift}" ] && [ "${shift}" -gt 0 ]; then
        extra_args+=(--shift "${shift}")
    fi
    if [ "${SMOKE:-0}" == "1" ]; then
        extra_args+=(--smoke)
    fi

    set +e
    "${PYTHON}" "${SCRIPTS_DIR}/s7_eval_gate_p6_v36b.py" --cond "${cond}" "${extra_args[@]}" > "${log_file}" 2>&1
    local rc=$?
    set -e

    if [ "${rc}" -ne 0 ]; then
        log_error_and_halt "p6_gen_${cond}${shift}" "${rc}" "${log_file}"
    fi

    touch "${done_flag}"
    echo "[P6-CHAIN] Generation Condition ${cond} (shift=${shift}) completed successfully. Created ${done_flag}."
}

# ==============================================================================
# Phase 2: Main Gate Generation Set (C, A, B1, E1)
# ==============================================================================
run_generation_condition "C" "" "P6_GEN_C_DONE.flag"
run_generation_condition "A" "" "P6_GEN_A_DONE.flag"
run_generation_condition "B" "1" "P6_GEN_B1_DONE.flag"
run_generation_condition "E1" "" "P6_GEN_E1_DONE.flag"

# ==============================================================================
# Phase 3: Intermediate Scoring — Main Gates Table
# ==============================================================================
SCORE_MAIN_FLAG="${RAW_DIR}/P6_SCORE_MAIN_DONE.flag"
if [ -f "${SCORE_MAIN_FLAG}" ]; then
    echo "[P6-CHAIN] Stage 3 (Main Gates Scoring) already completed. Skipping."
else
    echo "[P6-CHAIN] Stage 3: Scoring Main Gates (C, A, B1, E1)..."
    LOG_SCORE_MAIN="${LOG_DIR}/p6_score_main.log"
    SCORE_EXTRA=()
    if [ "${SMOKE:-0}" == "1" ]; then
        SCORE_EXTRA+=(--smoke)
    fi

    set +e
    "${PYTHON_EVAL}" "${SCRIPTS_DIR}/s7_eval_score_p6_v36.py" \
        --bootstraps 10000 \
        --seed 2026 \
        --out_json "${RESULTS_DIR}/p6_gate_table_main.json" \
        --out_md "${RESULTS_DIR}/p6_gate_table_main.md" \
        "${SCORE_EXTRA[@]}" > "${LOG_SCORE_MAIN}" 2>&1
    RC=$?
    set -e

    if [ "${RC}" -ne 0 ]; then
        log_error_and_halt "p6_score_main" "${RC}" "${LOG_SCORE_MAIN}"
    fi
    touch "${SCORE_MAIN_FLAG}"
    echo "[P6-CHAIN] Stage 3 completed successfully. Created ${SCORE_MAIN_FLAG}."
fi

# ==============================================================================
# Phase 4: Remaining Generation Conditions
# F -> D1 -> D2(s=1) -> E2(s=1) -> B(s=2) -> B(s=3) -> D2(s=2) -> D2(s=3)
# ==============================================================================
run_generation_condition "F" "" "P6_GEN_F_DONE.flag"
run_generation_condition "D1" "" "P6_GEN_D1_DONE.flag"
run_generation_condition "D2" "1" "P6_GEN_D21_DONE.flag"
run_generation_condition "E2" "1" "P6_GEN_E21_DONE.flag"
run_generation_condition "B" "2" "P6_GEN_B2_DONE.flag"
run_generation_condition "B" "3" "P6_GEN_B3_DONE.flag"
run_generation_condition "D2" "2" "P6_GEN_D22_DONE.flag"
run_generation_condition "D2" "3" "P6_GEN_D23_DONE.flag"

# ==============================================================================
# Phase 5: Final Full Scoring Table
# ==============================================================================
SCORE_FULL_FLAG="${RAW_DIR}/P6_SCORE_FULL_DONE.flag"
if [ -f "${SCORE_FULL_FLAG}" ]; then
    echo "[P6-CHAIN] Stage 5 (Full Gates Scoring) already completed. Skipping."
else
    echo "[P6-CHAIN] Stage 5: Scoring Full Gates (all conditions)..."
    LOG_SCORE_FULL="${LOG_DIR}/p6_score_full.log"
    SCORE_EXTRA=()
    if [ "${SMOKE:-0}" == "1" ]; then
        SCORE_EXTRA+=(--smoke)
    fi

    set +e
    "${PYTHON_EVAL}" "${SCRIPTS_DIR}/s7_eval_score_p6_v36.py" \
        --bootstraps 10000 \
        --seed 2026 \
        --out_json "${RESULTS_DIR}/p6_gate_table_full.json" \
        --out_md "${RESULTS_DIR}/p6_gate_table_full.md" \
        "${SCORE_EXTRA[@]}" > "${LOG_SCORE_FULL}" 2>&1
    RC=$?
    set -e

    if [ "${RC}" -ne 0 ]; then
        log_error_and_halt "p6_score_full" "${RC}" "${LOG_SCORE_FULL}"
    fi
    touch "${SCORE_FULL_FLAG}"
    echo "[P6-CHAIN] Stage 5 completed successfully. Created ${SCORE_FULL_FLAG}."
fi

# ==============================================================================
# Phase 6: Auxiliary NLL Evaluations (A, B1, C, E1)
# ==============================================================================
run_nll_condition() {
    local cond="$1"
    local shift="$2"
    local flag_name="$3"
    local done_flag="${RAW_DIR}/${flag_name}"
    local log_file="${LOG_DIR}/p6_nll_${cond}${shift}.log"

    if [ -f "${done_flag}" ]; then
        echo "[P6-CHAIN] Auxiliary NLL for ${cond} (shift=${shift}) already completed. Skipping."
        return 0
    fi

    echo "[P6-CHAIN] Running Auxiliary NLL Evaluation for ${cond} (shift=${shift})..."
    local extra_args=(--nll)
    if [ -n "${shift}" ] && [ "${shift}" -gt 0 ]; then
        extra_args+=(--shift "${shift}")
    fi
    if [ "${SMOKE:-0}" == "1" ]; then
        extra_args+=(--smoke)
    fi

    set +e
    "${PYTHON}" "${SCRIPTS_DIR}/s7_eval_gate_p6_v36b.py" \
        --cond "${cond}" \
        --out "${RESULTS_DIR}/p6_nll_${cond}${shift}.jsonl" \
        "${extra_args[@]}" > "${log_file}" 2>&1
    local rc=$?
    set -e

    if [ "${rc}" -ne 0 ]; then
        log_error_and_halt "p6_nll_${cond}${shift}" "${rc}" "${log_file}"
    fi

    touch "${done_flag}"
    echo "[P6-CHAIN] Auxiliary NLL for ${cond} completed successfully. Created ${done_flag}."
}

run_nll_condition "A" "" "P6_NLL_A_DONE.flag"
run_nll_condition "B" "1" "P6_NLL_B1_DONE.flag"
run_nll_condition "C" "" "P6_NLL_C_DONE.flag"
run_nll_condition "E1" "" "P6_NLL_E1_DONE.flag"

# ==============================================================================
# Phase 7: Chain Complete
# ==============================================================================
CHAIN_DONE_FLAG="${RAW_DIR}/P6_CHAIN_DONE.flag"
touch "${CHAIN_DONE_FLAG}"
echo "================================================================================"
echo "CTS S7 P6 Pipeline Chain Completed Successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Flag created: ${CHAIN_DONE_FLAG}"
echo "================================================================================"

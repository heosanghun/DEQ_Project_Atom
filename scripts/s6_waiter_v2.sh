#!/usr/bin/env bash
# scripts/s6_waiter_v2.sh - S6 Sequential Execution Waiter Chain (v2)
# Directive: EXE-260926-37 Task 4 (System 2 / Ledger #353, #354)

set -eo pipefail

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

PYTHON_EXEC="/home/shoon/system1_5/.venv/bin/python"
LOG_DIR="${BASE_DIR}/raw_s6"
RESULTS_DIR="${BASE_DIR}/results_s6"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

WAITER_LOG="${LOG_DIR}/s6_waiter_v2.log"
exec > >(tee -a "${WAITER_LOG}") 2>&1

echo "================================================================================"
echo "S6 WAITER CHAIN V2 INITIALIZED"
echo "Directive: EXE-260926-37 Task 4 (System 2 / Ledger #353, #354)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname) | User: $(whoami) | PID: $$"
echo "Target Eval Log: raw_s5/phase2_eval_run4b.log"
echo "Target Eval JSON: results_s5/phase2_gate.json"
echo "Chain Sequence on GPU 0: ① EXE-35 Task 2 -> ② Control 20 a0.50 -> ③ a0.35 74q -> ④ a0.25 74q -> ⑤ Evaluation"
echo "================================================================================"

EVAL_LOG="raw_s5/phase2_eval_run4b.log"
EVAL_JSON="results_s5/phase2_gate.json"
CHECK_INTERVAL=30

echo "[WAITER_V2] Monitoring Phase 2 evaluation run4b every ${CHECK_INTERVAL}s..."
loop_count=0

while true; do
    loop_count=$((loop_count + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if evaluation process is running
    is_eval_running=false
    if pgrep -f "s5_eval_gate_v3.py" > /dev/null 2>&1; then
        is_eval_running=true
    fi
    if pgrep -f "run_phase2_eval_run4b.sh" > /dev/null 2>&1; then
        is_eval_running=true
    fi

    # Check if EVAL COMPLETED SUCCESSFULLY appears in eval log
    has_completed_successfully=false
    if [ -f "${EVAL_LOG}" ] && grep -q "EVAL COMPLETED SUCCESSFULLY" "${EVAL_LOG}"; then
        has_completed_successfully=true
    fi

    # Check if results_s5/phase2_gate.json exists and is non-empty
    has_gate_json=false
    if [ -f "${EVAL_JSON}" ] && [ -s "${EVAL_JSON}" ]; then
        has_gate_json=true
    fi

    # Check for success trigger:
    # EVAL COMPLETED SUCCESSFULLY in log AND eval process dead AND phase2_gate.json exists
    if [ "${has_completed_successfully}" = true ] && [ "${is_eval_running}" = false ] && [ "${has_gate_json}" = true ]; then
        echo ""
        echo "================================================================================"
        echo "[WAITER_V2 SUCCESS] Phase 2 Evaluation COMPLETED SUCCESSFULLY detected!"
        echo "  - Eval log status: Confirmed ('EVAL COMPLETED SUCCESSFULLY' found)"
        echo "  - Gate JSON: Found (${EVAL_JSON})"
        echo "  - Eval process: Terminated"
        echo "Timestamp: ${timestamp}"
        echo "================================================================================"
        break
    fi

    # Check for failure condition:
    # Eval process dead but success condition NOT met
    if [ "${is_eval_running}" = false ]; then
        # Give a 5-second grace period for file flushing
        sleep 5
        if [ -f "${EVAL_LOG}" ] && grep -q "EVAL COMPLETED SUCCESSFULLY" "${EVAL_LOG}" && [ -f "${EVAL_JSON}" ]; then
            echo "[WAITER_V2 SUCCESS] Phase 2 Evaluation COMPLETED SUCCESSFULLY detected on final check."
            break
        else
            echo ""
            echo "================================================================================"
            echo "[WAITER_V2 ALERT] Phase 2 Evaluation process exited but success condition was NOT met!"
            echo "  - Completed log line: ${has_completed_successfully}"
            echo "  - Gate JSON exists: ${has_gate_json}"
            echo "Timestamp: ${timestamp}"
            echo "Aborting S6 chain execution. Zero GPU execution triggered."
            echo "================================================================================"
            exit 1
        fi
    fi

    # Heartbeat log every 10 iterations (5 minutes)
    if [ $((loop_count % 10)) -eq 1 ]; then
        latest_chk=$(grep "\[CHK\]" "${EVAL_LOG}" 2>/dev/null | tail -n 1 || echo "initializing...")
        echo "[${timestamp}] Waiting for Phase 2 evaluation... (Process active, latest: ${latest_chk})"
    fi

    sleep "${CHECK_INTERVAL}"
done

# Cooldown and verify GPU 0 is clean
echo "[WAITER_V2] Waiting 20s for GPU 0 cleanup..."
sleep 20
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -i 0

# ------------------------------------------------------------------------------
# Step ①: EXE-35 Task 2 GPU Noise Floor & Parity (≈10 min)
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "[CHAIN 1/5] EXE-35 Task 2: GPU bf16 Noise Floor & Parity Measurement"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
RUN_ID_T2=$(date '+%Y%m%d_%H%M%S')
LOG_T2="raw_s5/parity_gpu_noise_${RUN_ID_T2}.log"

nice -n 10 "${PYTHON_EXEC}" exe_scratch/s5/run_gpu_noise_parity.py \
    --device cuda:0 \
    --id "${RUN_ID_T2}" 2>&1 | tee "${LOG_T2}"

echo "[CHAIN 1/5 COMPLETED] Saved: ${LOG_T2} (SHA: $(sha256sum ${LOG_T2} | awk '{print $1}'))"

# ------------------------------------------------------------------------------
# Step ②: Control 20 with alpha=0.50 (NFE 2000, tol 0.0)
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "[CHAIN 2/5] S6 Control 20 with alpha=0.50 (NFE 2000, tol 0.0)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
"${PYTHON_EXEC}" scripts/s6_damping_run.py \
    --alpha 0.50 \
    --items configs/s6_control_20.json \
    --out results_s6/zstar_s6_a0.50_control20.pt \
    --max_nfe 2000 \
    --tol 0.0 \
    --device cuda:0

echo "[CHAIN 2/5 COMPLETED] Saved: results_s6/zstar_s6_a0.50_control20.pt"

# ------------------------------------------------------------------------------
# Step ③: 74 Items with alpha=0.35 (NFE 2000, tol 0.0)
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "[CHAIN 3/5] S6 74 Items with alpha=0.35 (NFE 2000, tol 0.0)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
"${PYTHON_EXEC}" scripts/s6_damping_run.py \
    --alpha 0.35 \
    --items configs/s6_items_74.json \
    --out results_s6/zstar_s6_a0.35.pt \
    --max_nfe 2000 \
    --tol 0.0 \
    --device cuda:0

echo "[CHAIN 3/5 COMPLETED] Saved: results_s6/zstar_s6_a0.35.pt"

# ------------------------------------------------------------------------------
# Step ④: 74 Items with alpha=0.25 (NFE 2000, tol 0.0)
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "[CHAIN 4/5] S6 74 Items with alpha=0.25 (NFE 2000, tol 0.0)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
"${PYTHON_EXEC}" scripts/s6_damping_run.py \
    --alpha 0.25 \
    --items configs/s6_items_74.json \
    --out results_s6/zstar_s6_a0.25.pt \
    --max_nfe 2000 \
    --tol 0.0 \
    --device cuda:0

echo "[CHAIN 4/5 COMPLETED] Saved: results_s6/zstar_s6_a0.25.pt"

# ------------------------------------------------------------------------------
# Step ⑤: Final Evaluation & Decision Report
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "[CHAIN 5/5] S6 Damping Factor Comprehensive Evaluation"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
"${PYTHON_EXEC}" scripts/s6_eval_damping.py

echo ""
echo "================================================================================"
echo "S6 WAITER V2 CHAIN EXECUTION FINISHED SUCCESSFULLY"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Artifacts Generated:"
echo "  - ${LOG_T2}"
echo "  - results_s6/zstar_s6_a0.50_control20.pt"
echo "  - results_s6/zstar_s6_a0.35.pt"
echo "  - results_s6/zstar_s6_a0.25.pt"
echo "  - results_s6/s6_eval_summary.json"
echo "  - results_s6/report_S6_damping_results.md"
echo "================================================================================"

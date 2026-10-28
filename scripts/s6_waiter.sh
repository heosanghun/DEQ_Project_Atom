#!/usr/bin/env bash
# scripts/s6_waiter.sh - S6 Sequential Execution Waiter Chain
# Directive: EXE-260926-36 Task 3 (System 2 / 원장 #348)

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

WAITER_LOG="${LOG_DIR}/s6_waiter.log"
exec > >(tee -a "${WAITER_LOG}") 2>&1

echo "================================================================================"
echo "S6 WAITER CHAIN INITIALIZED"
echo "Directive: EXE-260926-36 Task 3 (System 2 / 원장 #348)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname) | User: $(whoami) | PID: $$"
echo "Target Pipeline Log: raw_s5/phase2_pipeline_run4.log"
echo "Chain Sequence on GPU 0: ① EXE-35 Task 2 -> ② Control 20 a0.5 -> ③ a0.35 74q -> ④ a0.25 74q"
echo "================================================================================"

PIPELINE_LOG="raw_s5/phase2_pipeline_run4.log"
PIPELINE_PID=2127626
CHECK_INTERVAL=60

# ------------------------------------------------------------------------------
# Monitoring Loop
# ------------------------------------------------------------------------------
echo "[WAITER] Monitoring Phase 2 run4 pipeline (PID ${PIPELINE_PID}) every ${CHECK_INTERVAL}s..."
loop_count=0

while true; do
    loop_count=$((loop_count + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if pipeline PID is running
    is_pid_running=false
    if ps -p "${PIPELINE_PID}" > /dev/null 2>&1; then
        is_pid_running=true
    fi

    # Also check if child python process is running
    if pgrep -f "s5_train_decoder_v2.py.*results_s5/decoder_l" > /dev/null 2>&1; then
        is_pid_running=true
    fi
    if pgrep -f "s5_eval_gate_v2.py" > /dev/null 2>&1; then
        is_pid_running=true
    fi

    # Check if COMPLETED SUCCESSFULLY appears in pipeline log
    has_completed_successfully=false
    if [ -f "${PIPELINE_LOG}" ] && grep -q "COMPLETED SUCCESSFULLY" "${PIPELINE_LOG}"; then
        has_completed_successfully=true
    fi

    # Check for success condition:
    # COMPLETED SUCCESSFULLY present AND pipeline processes not running
    if [ "${has_completed_successfully}" = true ] && [ "${is_pid_running}" = false ]; then
        echo ""
        echo "================================================================================"
        echo "[WAITER SUCCESS] Phase 2 Pipeline run4 COMPLETED SUCCESSFULLY detected!"
        echo "Timestamp: ${timestamp}"
        echo "================================================================================"
        break
    fi

    # Check for failure condition:
    # Pipeline process exited but COMPLETED SUCCESSFULLY is NOT present
    if [ "${is_pid_running}" = false ]; then
        echo ""
        echo "================================================================================"
        echo "[WAITER ALERT] Phase 2 Pipeline process exited but 'COMPLETED SUCCESSFULLY' was NOT found!"
        echo "Timestamp: ${timestamp}"
        echo "Aborting S6 chain execution. Zero GPU execution triggered."
        echo "================================================================================"
        exit 1
    fi

    # Heartbeat log every 10 iterations (10 minutes)
    if [ $((loop_count % 10)) -eq 1 ]; then
        latest_step=$(grep -o "Step [0-9]*/[0-9]*" "${PIPELINE_LOG}" 2>/dev/null | tail -n 1 || echo "unknown")
        echo "[${timestamp}] Waiting for Phase 2 pipeline... (PID active, latest logged: ${latest_step})"
    fi

    sleep "${CHECK_INTERVAL}"
done

# Cooldown and verify GPU 0 is clean
echo "[WAITER] Waiting 30s for GPU 0 cleanup..."
sleep 30
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -i 0

# ------------------------------------------------------------------------------
# Step ①: EXE-35 Task 2 GPU Noise Floor & Parity (≈10 min)
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "[CHAIN 1/4] EXE-35 Task 2: GPU bf16 Noise Floor & Parity Measurement"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
RUN_ID_T2=$(date '+%Y%m%d_%H%M%S')
LOG_T2="raw_s5/parity_gpu_noise_${RUN_ID_T2}.log"

nice -n 10 "${PYTHON_EXEC}" exe_scratch/s5/run_gpu_noise_parity.py \
    --device cuda:0 \
    --id "${RUN_ID_T2}" 2>&1 | tee "${LOG_T2}"

echo "[CHAIN 1/4 COMPLETED] Saved: ${LOG_T2} (SHA: $(sha256sum ${LOG_T2} | awk '{print $1}'))"

# ------------------------------------------------------------------------------
# Step ②: Control 20 with alpha=0.50 (NFE 2000, tol 0.0)
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "[CHAIN 2/4] S6 Control 20 with alpha=0.50 (NFE 2000, tol 0.0)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
"${PYTHON_EXEC}" scripts/s6_damping_run.py \
    --alpha 0.50 \
    --items configs/s6_control_20.json \
    --out results_s6/zstar_s6_a0.50_control20.pt \
    --max_nfe 2000 \
    --tol 0.0 \
    --device cuda:0

echo "[CHAIN 2/4 COMPLETED] Saved: results_s6/zstar_s6_a0.50_control20.pt"

# ------------------------------------------------------------------------------
# Step ③: 74 Items with alpha=0.35 (NFE 2000, tol 0.0)
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "[CHAIN 3/4] S6 74 Items with alpha=0.35 (NFE 2000, tol 0.0)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
"${PYTHON_EXEC}" scripts/s6_damping_run.py \
    --alpha 0.35 \
    --items configs/s6_items_74.json \
    --out results_s6/zstar_s6_a0.35.pt \
    --max_nfe 2000 \
    --tol 0.0 \
    --device cuda:0

echo "[CHAIN 3/4 COMPLETED] Saved: results_s6/zstar_s6_a0.35.pt"

# ------------------------------------------------------------------------------
# Step ④: 74 Items with alpha=0.25 (NFE 2000, tol 0.0)
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "[CHAIN 4/4] S6 74 Items with alpha=0.25 (NFE 2000, tol 0.0)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
"${PYTHON_EXEC}" scripts/s6_damping_run.py \
    --alpha 0.25 \
    --items configs/s6_items_74.json \
    --out results_s6/zstar_s6_a0.25.pt \
    --max_nfe 2000 \
    --tol 0.0 \
    --device cuda:0

echo "[CHAIN 4/4 COMPLETED] Saved: results_s6/zstar_s6_a0.25.pt"

# ------------------------------------------------------------------------------
# Step ⑤: Final Evaluation & Decision Report
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "[EVALUATION] S6 Damping Factor Comprehensive Evaluation"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
"${PYTHON_EXEC}" scripts/s6_eval_damping.py

echo ""
echo "================================================================================"
echo "S6 WAITER CHAIN EXECUTION FINISHED SUCCESSFULLY"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Artifacts Generated:"
echo "  - ${LOG_T2}"
echo "  - results_s6/zstar_s6_a0.50_control20.pt"
echo "  - results_s6/zstar_s6_a0.35.pt"
echo "  - results_s6/zstar_s6_a0.25.pt"
echo "  - results_s6/s6_eval_summary.json"
echo "  - results_s6/report_S6_damping_results.md"
echo "================================================================================"

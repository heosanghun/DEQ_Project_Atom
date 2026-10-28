#!/usr/bin/env bash
# scripts/run_phase2_eval_train40.sh
# Phase 2 Train 40 Items Calibration Evaluation
# Directive: EXE-39 Task 3 (Ledger #357)
set -eo pipefail

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

mkdir -p raw_s5 results_s5

EVAL_LOG="${BASE_DIR}/raw_s5/phase2_eval_train40.log"
exec > >(tee -a "${EVAL_LOG}") 2>&1

echo "================================================================================"
echo "Starting Phase 2 Train 40 Items Calibration Evaluation (EXE-39 Task 3)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname) | PID: $$"
echo "Target Items: configs/s5_train_40.json (40 items)"
echo "================================================================================"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/shoon/system1_5/.venv/bin/python scripts/s5_eval_gate_v3.py \
    --zstar results_s5/zstar_s5_400.pt \
    --decoder results_s5/decoder_l1.pt \
    --control results_s5/decoder_l0.pt \
    --items configs/s5_train_40.json \
    --out results_s5/phase2_train40_calibration_gate.json \
    --outputs results_s5/phase2_train40_calibration_outputs.jsonl \
    --device cuda:0

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "================================================================================"
    echo "TRAIN 40 CALIBRATION EVAL COMPLETED SUCCESSFULLY"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
else
    echo "TRAIN 40 CALIBRATION EVAL FAILED with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

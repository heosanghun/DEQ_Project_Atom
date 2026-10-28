#!/usr/bin/env bash
# scripts/run_phase2_eval_run4b.sh
# Phase 2 Gate Evaluation (Run 4b)
# Directive: EXE-260926-37 (Ledger #353, #354)
set -eo pipefail

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

mkdir -p raw_s5 results_s5

EVAL_LOG="${BASE_DIR}/raw_s5/phase2_eval_run4b.log"
exec > >(tee -a "${EVAL_LOG}") 2>&1

echo "================================================================================"
echo "Starting Phase 2 Gate Evaluation (Run 4b)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname) | PID: $$"
echo "================================================================================"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/shoon/system1_5/.venv/bin/python scripts/s5_eval_gate_v3.py \
    --zstar results_s5/zstar_s5_400.pt \
    --decoder results_s5/decoder_l1.pt \
    --control results_s5/decoder_l0.pt \
    --items configs/s5_heldout_80.json \
    --out results_s5/phase2_gate.json \
    --outputs results_s5/phase2_outputs.jsonl \
    --device cuda:0

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "================================================================================"
    echo "EVAL COMPLETED SUCCESSFULLY"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
else
    echo "EVAL FAILED with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

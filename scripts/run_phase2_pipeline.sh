#!/usr/bin/env bash
# scripts/run_phase2_pipeline.sh - S5 Phase 2 Continuous Pipeline
# Directive: EXE-260923-29 (System 2 / Ledger #333)

set -eo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/shoon/cts_e1_s5:${PYTHONPATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_EXEC="/home/shoon/system1_5/.venv/bin/python"

cd "${BASE_DIR}"

mkdir -p results_s5 raw_s5 configs

LOG_FILE="raw_s5/phase2_pipeline.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================================"
echo "S5 PHASE 2 CONTINUOUS PIPELINE LAUNCH"
echo "Directive: EXE-260924-30 (System 2 / Ledger #334, #335, #336)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname) | User: $(whoami) | PID: $$"
echo "GPU Device: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Log File: ${LOG_FILE}"
echo "================================================================================"

echo ""
echo "--------------------------------------------------------------------------------"
echo "[STEP 1/4] Harness Anchor Verification (B1 text_only on 10 Item 4 questions)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "--------------------------------------------------------------------------------"
"${PYTHON_EXEC}" scripts/s5_eval_gate_v2.py --harness-anchor

echo ""
echo "--------------------------------------------------------------------------------"
echo "[STEP 2/4] Primary Decoder Training (lambda=1.0, 320 train questions, 3000 steps)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "--------------------------------------------------------------------------------"
"${PYTHON_EXEC}" scripts/s5_train_decoder_v2.py \
    --lambda 1.0 \
    --seed 2026 \
    --max_steps 3000 \
    --batch_size 8 \
    --micro_batch_size 2 \
    --grad_accum_steps 4 \
    --out results_s5/decoder_l1.pt

echo ""
echo "--------------------------------------------------------------------------------"
echo "[STEP 3/4] Control Decoder Training (lambda=0.0, 320 train questions, 3000 steps)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "--------------------------------------------------------------------------------"
"${PYTHON_EXEC}" scripts/s5_train_decoder_v2.py \
    --lambda 0.0 \
    --seed 2026 \
    --max_steps 3000 \
    --batch_size 8 \
    --micro_batch_size 2 \
    --grad_accum_steps 4 \
    --out results_s5/decoder_l0.pt

echo ""
echo "--------------------------------------------------------------------------------"
echo "[STEP 4/4] Phase 2 Full Gate Evaluation (3 Strata x 5 Conditions on Held-out 80)"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "--------------------------------------------------------------------------------"
"${PYTHON_EXEC}" scripts/s5_eval_gate_v2.py \
    --zstar results_s5/zstar_s5_400.pt \
    --decoder results_s5/decoder_l1.pt \
    --control results_s5/decoder_l0.pt \
    --items configs/s5_heldout_80.json \
    --out results_s5/phase2_gate.json \
    --outputs results_s5/phase2_outputs.jsonl

echo ""
echo "================================================================================"
echo "S5 PHASE 2 CONTINUOUS PIPELINE COMPLETED SUCCESSFULLY"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Artifacts:"
echo "  - results_s5/decoder_l1.pt"
echo "  - results_s5/decoder_l0.pt"
echo "  - results_s5/phase2_gate.json"
echo "  - results_s5/phase2_outputs.jsonl"
echo "  - raw_s5/phase2_pipeline.log"
echo "================================================================================"

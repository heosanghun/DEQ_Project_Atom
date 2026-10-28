#!/usr/bin/env bash
# scripts/s7_train_p4_prime_resume_v36.sh - Resume Serial Decoder Training Pipeline for CTS P4' v3.6
# Directive: CTS P4' (PREREG_S7_D_v3_6 §2·§3·§7 / Ledger #419 / TASK-0024)
set -eo pipefail

cd /home/shoon/cts_e1_s5
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
PYTHON="/home/shoon/system1_5/.venv/bin/python"
SCRIPT="scripts/s7_train_decoder_v36.py"
ITEMS="configs/s7_train_993.json"
ZSTAR="results_s7/zstar_train1000_P4.pt"
ZSRC="results_s7/zsrc_train1000_P4.pt"

echo "================================================================================"
echo "CTS P4' Serial Decoder Training Suite (v3.6 Pure Gemma Base, GPU 0, RESUME)"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname) | CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "================================================================================"

# Decoder A: lambda=1.0, input=z*, resume from step 2500
echo ""
echo "=== [1/3] Resuming Decoder A from step 2500 (lambda=1.0, input=z*) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
$PYTHON $SCRIPT \
  --zstar $ZSTAR \
  --items $ITEMS \
  --loss_lambda 1.0 \
  --max_steps 3000 \
  --batch_size 8 \
  --micro_batch_size 2 \
  --grad_accum_steps 4 \
  --lr_wproj 1e-4 \
  --lr_adapter 3e-4 \
  --weight_decay 0.01 \
  --resume results_s5/decoder_A_v36_step2500.pt \
  --out results_s7/decoder_A_v36.pt \
  2>&1 | tee -a raw_s7/train_decoder_A_v36.log
echo "=== [1/3] Decoder A Completed at $(date '+%Y-%m-%d %H:%M:%S') ==="

# Decoder D: lambda=0.0, input=z*
echo ""
echo "=== [2/3] Starting Decoder D (lambda=0.0, input=z*) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
$PYTHON $SCRIPT \
  --zstar $ZSTAR \
  --items $ITEMS \
  --loss_lambda 0.0 \
  --max_steps 3000 \
  --batch_size 8 \
  --micro_batch_size 2 \
  --grad_accum_steps 4 \
  --lr_wproj 1e-4 \
  --lr_adapter 3e-4 \
  --weight_decay 0.01 \
  --out results_s7/decoder_D_v36.pt \
  2>&1 | tee raw_s7/train_decoder_D_v36.log
echo "=== [2/3] Decoder D Completed at $(date '+%Y-%m-%d %H:%M:%S') ==="

# Decoder E: lambda=1.0, input=z_src
echo ""
echo "=== [3/3] Starting Decoder E (lambda=1.0, input=z_src) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
$PYTHON $SCRIPT \
  --zstar $ZSRC \
  --items $ITEMS \
  --loss_lambda 1.0 \
  --max_steps 3000 \
  --batch_size 8 \
  --micro_batch_size 2 \
  --grad_accum_steps 4 \
  --lr_wproj 1e-4 \
  --lr_adapter 3e-4 \
  --weight_decay 0.01 \
  --out results_s7/decoder_E_v36.pt \
  2>&1 | tee raw_s7/train_decoder_E_v36.log
echo "=== [3/3] Decoder E Completed at $(date '+%Y-%m-%d %H:%M:%S') ==="

echo ""
echo "================================================================================"
echo "CTS P4' Serial Decoder Training Suite (v3.6) FINISHED at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

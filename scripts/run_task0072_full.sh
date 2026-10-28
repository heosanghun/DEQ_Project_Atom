#!/usr/bin/env bash
# scripts/run_task0072_full.sh - G2v3 M3 Full Joint Training & Dev 600 Evaluation Pipeline
# Directives: TASK-0072 / PREREG_S7_G2v3 / Node N8 / proposals/G2v3_PLAN_20261009.md
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python

GPU5="GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
GPU6="GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c"

mkdir -p results_s7/g2v3 results_s7/g2v2 results_s7/p6r raw_s7
LOG="raw_s7/task0072_full_master.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== TASK-0072 G2v3 M3 Full Pipeline Started at $(date '+%F %T') ==="

# 1. SHA verification of scripts and prerequisites
echo "[SHA-256 Check]"
sha256sum scripts/g2v3_m3_full_train.py scripts/g2v3_m3_eval_latents.py configs/s7_train_993.json results_s7/p6_item_order.json results_s7/p6r/p6_gen_C.jsonl

# 2. Check GPU states
for u in $GPU5 $GPU6; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$u" | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$u" | grep -c '^[0-9]' || true)
  echo "[CHECK] $u memory=${m}MiB apps=${a}"
  if [ "$m" -ge 1024 ] || [ "$a" -ne 0 ]; then
    echo "[FATAL] GPU $u is busy"
    exit 1
  fi
done

# 3. Server CPU smoke test
echo "[SMOKE] Running in-memory smoke test on server..."
$PY scripts/g2v3_m3_full_train.py --smoke --device cpu --max_steps 5 --batch_size 2
echo "[SMOKE] Server smoke test PASSED."

# 4. Phase 1: Launch Full Joint Training (1,500 steps)
echo "[LAUNCH] Mode A (M3 Phantom Gradient) on GPU 5..."
CUDA_VISIBLE_DEVICES=$GPU5 $PY scripts/g2v3_m3_full_train.py \
  --mode A \
  --device cuda:0 \
  --items configs/s7_train_993.json \
  --max_items 1000 \
  --max_steps 1500 \
  --batch_size 4 \
  --lr_projector 1e-4 \
  --lr_adapter 3e-4 \
  --lr_deq 3e-5 \
  --lambda_ift 0.1 \
  --out results_s7/g2v3/full_model_A.pt \
  --decoder_out results_s7/g2v3/decoder_A_v36.pt \
  --deq_out results_s7/g2v3/deq_backbone_A.pt \
  --manifest results_s7/g2v3/full_manifest_A.json > raw_s7/g2v3_full_train_A.log 2>&1 &
PID_TRAIN_A=$!

echo "[LAUNCH] Mode E (E''' Contrast Baseline) on GPU 6..."
CUDA_VISIBLE_DEVICES=$GPU6 $PY scripts/g2v3_m3_full_train.py \
  --mode E \
  --device cuda:0 \
  --items configs/s7_train_993.json \
  --max_items 1000 \
  --max_steps 1500 \
  --batch_size 4 \
  --lr_projector 1e-4 \
  --lr_adapter 3e-4 \
  --lr_deq 3e-5 \
  --lambda_ift 0.0 \
  --out results_s7/g2v3/full_model_E.pt \
  --decoder_out results_s7/g2v3/decoder_E_v36.pt \
  --deq_out results_s7/g2v3/deq_backbone_E.pt \
  --manifest results_s7/g2v3/full_manifest_E.json > raw_s7/g2v3_full_train_E.log 2>&1 &
PID_TRAIN_E=$!

echo "[WAIT] Waiting for Full Training A (PID $PID_TRAIN_A) and E (PID $PID_TRAIN_E)..."
wait $PID_TRAIN_A
RC_TRAIN_A=$?
echo "[DONE] Training Mode A finished with exit code $RC_TRAIN_A at $(date '+%T')"

wait $PID_TRAIN_E
RC_TRAIN_E=$?
echo "[DONE] Training Mode E finished with exit code $RC_TRAIN_E at $(date '+%T')"

if [ "$RC_TRAIN_A" -ne 0 ] || [ "$RC_TRAIN_E" -ne 0 ]; then
  echo "[FATAL] Training failed (A: $RC_TRAIN_A, E: $RC_TRAIN_E)"
  tail -n 25 raw_s7/g2v3_full_train_A.log
  tail -n 25 raw_s7/g2v3_full_train_E.log
  exit 1
fi

echo "=== Phase 1 Artifacts ==="
# Mirror decoder checkpoints to results_s7/g2v2/ for v36e path compatibility
cp results_s7/g2v3/decoder_A_v36.pt results_s7/g2v2/decoder_Am3_v36.pt
cp results_s7/g2v3/decoder_E_v36.pt results_s7/g2v2/decoder_Em3_v36.pt
ls -lh results_s7/g2v3/decoder_*.pt results_s7/g2v2/decoder_*m3_v36.pt results_s7/g2v3/deq_backbone_*.pt
sha256sum results_s7/g2v3/decoder_*.pt results_s7/g2v2/decoder_*m3_v36.pt results_s7/g2v3/deq_backbone_*.pt

# 5. Phase 2: Compute Latents for Dev 600
echo "[LATENTS] Computing latents for dev 600 in canonical item order..."
CUDA_VISIBLE_DEVICES=$GPU5 $PY scripts/g2v3_m3_eval_latents.py \
  --mode A \
  --device cuda:0 \
  --items results_s7/p6_item_order.json \
  --deq_ckpt results_s7/g2v3/deq_backbone_A.pt \
  --zstar results_s7/g2v2/zstar_zsrcinj_dev600.pt \
  --zsrc results_s7/zsrc_heldout600.pt \
  --out results_s7/g2v3/dev600_zstar_A.pt > raw_s7/g2v3_latents_dev_A.log 2>&1 &
PID_LAT_A=$!

CUDA_VISIBLE_DEVICES=$GPU6 $PY scripts/g2v3_m3_eval_latents.py \
  --mode E \
  --device cuda:0 \
  --items results_s7/p6_item_order.json \
  --deq_ckpt results_s7/g2v3/deq_backbone_E.pt \
  --zsrc results_s7/zsrc_heldout600.pt \
  --out results_s7/g2v3/dev600_zsrc_E.pt > raw_s7/g2v3_latents_dev_E.log 2>&1 &
PID_LAT_E=$!

wait $PID_LAT_A
RC_LAT_A=$?
wait $PID_LAT_E
RC_LAT_E=$?

if [ "$RC_LAT_A" -ne 0 ] || [ "$RC_LAT_E" -ne 0 ]; then
  echo "[FATAL] Latent computation failed (A: $RC_LAT_A, E: $RC_LAT_E)"
  exit 1
fi
echo "[DONE] Latents computed for dev 600."

# 6. Phase 3: Dev 600 Evaluation Generation
ORDER_SHA="6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c"

echo "[GENERATE] Dev 600 Condition A on GPU 5..."
CUDA_VISIBLE_DEVICES=$GPU5 $PY scripts/s7_eval_gate_p6_v36e.py \
  --cond A \
  --device cuda:0 \
  --decoder_ckpt results_s7/g2v2/decoder_Am3_v36.pt \
  --zstar results_s7/g2v3/dev600_zstar_A.pt \
  --order_sha $ORDER_SHA \
  --out results_s7/p6r/g2v3dev_gen_A.jsonl > raw_s7/g2v3_dev_gen_A.log 2>&1 &
PID_GEN_A=$!

echo "[GENERATE] Dev 600 Condition E on GPU 6..."
CUDA_VISIBLE_DEVICES=$GPU6 $PY scripts/s7_eval_gate_p6_v36e.py \
  --cond E1 \
  --device cuda:0 \
  --decoder_ckpt results_s7/g2v2/decoder_Em3_v36.pt \
  --zsrc results_s7/g2v3/dev600_zsrc_E.pt \
  --order_sha $ORDER_SHA \
  --out results_s7/p6r/g2v3dev_gen_E.jsonl > raw_s7/g2v3_dev_gen_E.log 2>&1 &
PID_GEN_E=$!

wait $PID_GEN_A
RC_GEN_A=$?
wait $PID_GEN_E
RC_GEN_E=$?

if [ "$RC_GEN_A" -ne 0 ] || [ "$RC_GEN_E" -ne 0 ]; then
  echo "[FATAL] Dev 600 generation failed (A: $RC_GEN_A, E: $RC_GEN_E)"
  exit 1
fi
echo "[DONE] Dev 600 generation finished."

# 7. Phase 4: Dev 600 Scoring
echo "[SCORE] Running official bootstrap scoring on dev 600..."
$PY scripts/s7_eval_score_p6_v36.py \
  --gen_A results_s7/p6r/g2v3dev_gen_A.jsonl \
  --gen_C results_s7/p6r/p6_gen_C.jsonl \
  --gen_E1 results_s7/p6r/g2v3dev_gen_E.jsonl \
  --gen_B1 results_s7/p6r/p6_gen_B1.jsonl \
  --order results_s7/p6_item_order.json \
  --out_json results_s7/p6r/g2v3dev_score_AE.json \
  --out_md results_s7/p6r/g2v3dev_score_AE.md

echo "=== Dev 600 Gate Table ==="
cat results_s7/p6r/g2v3dev_score_AE.md

echo "=== TASK-0072 Completed at $(date '+%F %T') ==="

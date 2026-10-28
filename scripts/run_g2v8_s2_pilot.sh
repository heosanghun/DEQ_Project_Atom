#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; echo "PIPELINE_EXITED_CODE_${rc}"; if [ $rc -ne 0 ]; then tail -n 25 /tmp/cts_g2v8_v2_s1/*.log /tmp/cts_g2v8_v2_s2/*.log 2>/dev/null || true; fi' EXIT

cd /home/shoon/cts_e1_s5
PY="/home/shoon/system1_5/.venv/bin/python"
OUT_S1="/tmp/cts_g2v8_v2_s1"
OUT_S2="/tmp/cts_g2v8_v2_s2"
rm -rf "$OUT_S1" "$OUT_S2"
mkdir -p "$OUT_S1" "$OUT_S2"

# Exact Physical GPU UUIDs from nvidia-smi for GPUs 5, 6, 7 (Physical GPU 4 is NEVER touched!)
U5="GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a"
U6="GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c"
U7="GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b"

echo "=== [0] GPU 4 Safety Check & Unit Self-Tests ==="
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v8_gca_bridge.py
CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v8_deq_train_eval.py --action selftest

echo "=== [1] Stage 1 (S2 Oracle) Training: Seeds 2047 (U5), 2049 (U6), 2050 (U7) ==="
PIDS=()
if [ ! -f "$OUT_S1/diag_s2_2047.json" ]; then
  CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v8_s2_train_eval.py \
    --action train --device cuda:0 --seed 2047 \
    --max_steps 200 --warmup_steps 25 --batch_size 4 \
    --eval_every 100 --eval_n_dev 60 --diag_n 120 \
    --decoder_out "$OUT_S1/decoder_s2_2047.pt" \
    --diag_out "$OUT_S1/diag_s2_2047.json" > "$OUT_S1/train_2047.log" 2>&1 &
  PIDS+=($!)
else
  echo "  [SKIP] Stage 1 Seed 2047 already trained ($OUT_S1/diag_s2_2047.json exists)"
fi

if [ ! -f "$OUT_S1/diag_s2_2049.json" ]; then
  CUDA_VISIBLE_DEVICES=$U6 $PY scripts/g2v8_s2_train_eval.py \
    --action train --device cuda:0 --seed 2049 \
    --max_steps 200 --warmup_steps 25 --batch_size 4 \
    --eval_every 100 --eval_n_dev 60 --diag_n 120 \
    --decoder_out "$OUT_S1/decoder_s2_2049.pt" \
    --diag_out "$OUT_S1/diag_s2_2049.json" > "$OUT_S1/train_2049.log" 2>&1 &
  PIDS+=($!)
else
  echo "  [SKIP] Stage 1 Seed 2049 already trained"
fi

if [ ! -f "$OUT_S1/diag_s2_2050.json" ]; then
  CUDA_VISIBLE_DEVICES=$U7 $PY scripts/g2v8_s2_train_eval.py \
    --action train --device cuda:0 --seed 2050 \
    --max_steps 200 --warmup_steps 25 --batch_size 4 \
    --eval_every 100 --eval_n_dev 60 --diag_n 120 \
    --decoder_out "$OUT_S1/decoder_s2_2050.pt" \
    --diag_out "$OUT_S1/diag_s2_2050.json" > "$OUT_S1/train_2050.log" 2>&1 &
  PIDS+=($!)
else
  echo "  [SKIP] Stage 1 Seed 2050 already trained"
fi

if [ ${#PIDS[@]} -gt 0 ]; then
  wait "${PIDS[@]}"
fi

echo "=== [1b] Stage 1 Training Complete! ==="
for s in 2047 2049 2050; do
  echo "--- Stage 1 Train Summary Seed $s ---"
  grep -E "AUDIT-GATE-C|MATCHER-INIT|W_TOK-INIT|DEV-60|SAVE-BEST|DIAG-DONE" "$OUT_S1/train_${s}.log" || true
done

echo "=== [2] Stage 1 Full Dev-600 Generation & Official Grading Across All 3 Seeds ==="
for COND in matched shuffled; do
  rm -f "$OUT_S1"/gen_${COND}_*_s*.jsonl
  for SHARD in 0 1; do
    CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v8_s2_train_eval.py \
      --action eval_gen --device cuda:0 --skip_sha \
      --decoder_ckpt "$OUT_S1/decoder_s2_2047.pt" \
      --cond $COND --shard_idx $SHARD --num_shards 2 --max_eval_items 600 \
      --out "$OUT_S1/gen_${COND}_2047_s${SHARD}.jsonl" > "$OUT_S1/gen_${COND}_2047_s${SHARD}.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=$U6 $PY scripts/g2v8_s2_train_eval.py \
      --action eval_gen --device cuda:0 --skip_sha \
      --decoder_ckpt "$OUT_S1/decoder_s2_2049.pt" \
      --cond $COND --shard_idx $SHARD --num_shards 2 --max_eval_items 600 \
      --out "$OUT_S1/gen_${COND}_2049_s${SHARD}.jsonl" > "$OUT_S1/gen_${COND}_2049_s${SHARD}.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=$U7 $PY scripts/g2v8_s2_train_eval.py \
      --action eval_gen --device cuda:0 --skip_sha \
      --decoder_ckpt "$OUT_S1/decoder_s2_2050.pt" \
      --cond $COND --shard_idx $SHARD --num_shards 2 --max_eval_items 600 \
      --out "$OUT_S1/gen_${COND}_2050_s${SHARD}.jsonl" > "$OUT_S1/gen_${COND}_2050_s${SHARD}.log" 2>&1 &
  done
  wait
  for s in 2047 2049 2050; do
    cat "$OUT_S1/gen_${COND}_${s}_s0.jsonl" "$OUT_S1/gen_${COND}_${s}_s1.jsonl" > "$OUT_S1/gen_${COND}_${s}.jsonl"
  done
done

for s in 2047 2049 2050; do
  echo "--- Stage 1 Official Grade Seed $s (Dev-600) ---"
  $PY scripts/g2v8_s2_train_eval.py \
    --action grade --max_eval_items 600 \
    --matched_jsonl "$OUT_S1/gen_matched_${s}.jsonl" \
    --shuffled_jsonl "$OUT_S1/gen_shuffled_${s}.jsonl" \
    --cfmt_jsonl "results_s7/p6r/g2v6s1/gen_C.jsonl" \
    --grade_out "$OUT_S1/grade_s2_${s}.json"
done

echo "=== [3] Stage 2 (G2 Contractive DEQ) Training: Seeds 2047 (U5), 2049 (U6), 2050 (U7) ==="
CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v8_deq_train_eval.py \
  --action train --device cuda:0 --seed 2047 \
  --init_bridge "$OUT_S1/decoder_s2_2047.pt" \
  --max_steps 200 --warmup_steps 25 --batch_size 4 \
  --eval_every 100 --eval_n_dev 60 --diag_n 120 \
  --ckpt_out "$OUT_S2/deq_bridge_2047.pt" \
  --diag_out "$OUT_S2/diag_deq_2047.json" > "$OUT_S2/train_2047.log" 2>&1 &
DPID1=$!

CUDA_VISIBLE_DEVICES=$U6 $PY scripts/g2v8_deq_train_eval.py \
  --action train --device cuda:0 --seed 2049 \
  --init_bridge "$OUT_S1/decoder_s2_2049.pt" \
  --max_steps 200 --warmup_steps 25 --batch_size 4 \
  --eval_every 100 --eval_n_dev 60 --diag_n 120 \
  --ckpt_out "$OUT_S2/deq_bridge_2049.pt" \
  --diag_out "$OUT_S2/diag_deq_2049.json" > "$OUT_S2/train_2049.log" 2>&1 &
DPID2=$!

CUDA_VISIBLE_DEVICES=$U7 $PY scripts/g2v8_deq_train_eval.py \
  --action train --device cuda:0 --seed 2050 \
  --init_bridge "$OUT_S1/decoder_s2_2050.pt" \
  --max_steps 200 --warmup_steps 25 --batch_size 4 \
  --eval_every 100 --eval_n_dev 60 --diag_n 120 \
  --ckpt_out "$OUT_S2/deq_bridge_2050.pt" \
  --diag_out "$OUT_S2/diag_deq_2050.json" > "$OUT_S2/train_2050.log" 2>&1 &
DPID3=$!

wait $DPID1 $DPID2 $DPID3
echo "=== [3b] Stage 2 DEQ Training Complete! ==="
for s in 2047 2049 2050; do
  echo "--- Stage 2 DEQ Train Summary Seed $s ---"
  grep -E "W_TOK-INIT|DEQ-MATCHER-INIT|DEV-60|SAVE-BEST-DEQ|DEQ-DIAG-DONE" "$OUT_S2/train_${s}.log" || true
done

echo "=== [4] Stage 2 Full Dev-600 Generation & Official Grading Across All 3 Seeds (Conditions A, B1, E, C) ==="
for COND in A B1 E; do
  rm -f "$OUT_S2"/gen_${COND}_*_s*.jsonl
  for SHARD in 0 1; do
    CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v8_deq_train_eval.py \
      --action eval_gen --device cuda:0 --skip_sha \
      --ckpt_out "$OUT_S2/deq_bridge_2047.pt" \
      --cond $COND --shard_idx $SHARD --num_shards 2 --max_eval_items 600 \
      --out "$OUT_S2/gen_${COND}_2047_s${SHARD}.jsonl" > "$OUT_S2/gen_${COND}_2047_s${SHARD}.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=$U6 $PY scripts/g2v8_deq_train_eval.py \
      --action eval_gen --device cuda:0 --skip_sha \
      --ckpt_out "$OUT_S2/deq_bridge_2049.pt" \
      --cond $COND --shard_idx $SHARD --num_shards 2 --max_eval_items 600 \
      --out "$OUT_S2/gen_${COND}_2049_s${SHARD}.jsonl" > "$OUT_S2/gen_${COND}_2049_s${SHARD}.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=$U7 $PY scripts/g2v8_deq_train_eval.py \
      --action eval_gen --device cuda:0 --skip_sha \
      --ckpt_out "$OUT_S2/deq_bridge_2050.pt" \
      --cond $COND --shard_idx $SHARD --num_shards 2 --max_eval_items 600 \
      --out "$OUT_S2/gen_${COND}_2050_s${SHARD}.jsonl" > "$OUT_S2/gen_${COND}_2050_s${SHARD}.log" 2>&1 &
  done
  wait
  for s in 2047 2049 2050; do
    cat "$OUT_S2/gen_${COND}_${s}_s0.jsonl" "$OUT_S2/gen_${COND}_${s}_s1.jsonl" > "$OUT_S2/gen_${COND}_${s}.jsonl"
  done
done

for s in 2047 2049 2050; do
  echo "--- Stage 2 DEQ Official Grade Seed $s (Dev-600) ---"
  $PY scripts/g2v8_deq_train_eval.py \
    --action grade --max_eval_items 600 \
    --a_jsonl "$OUT_S2/gen_A_${s}.jsonl" \
    --b1_jsonl "$OUT_S2/gen_B1_${s}.jsonl" \
    --e_jsonl "$OUT_S2/gen_E_${s}.jsonl" \
    --cfmt_jsonl "results_s7/p6r/g2v6s1/gen_C.jsonl" \
    --grade_out "$OUT_S2/grade_deq_${s}.json"
done

echo "=== [5] Final GPU 4 Verification & Pipeline Complete ==="
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
echo "ALL_G2V8_V2_STAGES_DONE"

#!/usr/bin/env bash
# scripts/cc_g2v4_cap_pilot.sh - capacity pilot (CC, 2026-10-11): can the latent->LM interface carry item info at all?
# Decoder + Phi_theta trained FROM SCRATCH (no warm start) with text_drop 1.0 (z-only training), 1500 steps, train 993 only; dev600 NLL incl. no-text own vs shift. Not a gate.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c
P=results_s7/g2v4cap; mkdir -p $P raw_s7
echo "=== cap pilot start $(date '+%F %T') ==="
sha256sum scripts/g2v4_m4_pilot_train.py scripts/g2v4_diag_iterate.py
for u in $U5 $U6; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' '); a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true); echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }; done
run() { # mode gpu lambda
  CUDA_VISIBLE_DEVICES=$2 $PY scripts/g2v4_m4_pilot_train.py --mode $1 --device cuda:0 --items configs/s7_train_993.json --max_items 1000 --max_steps 1500 --batch_size 4 \
    --lr_projector 1e-4 --lr_adapter 3e-4 --lr_deq 3e-5 --lambda_ift $3 --text_drop 1.0 --seed 2042 \
    --out $P/cap_model_$1.pt --decoder_out $P/decoder_$1.pt --deq_out $P/deq_$1.pt --manifest $P/manifest_$1.json > raw_s7/g2v4_cap_train_$1.log 2>&1
  echo "[TR-$1] rc=$? $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$2 $PY scripts/g2v4_diag_iterate.py --mode $1 --n 600 --ks 1 --notext --deq_ckpt $P/deq_$1.pt --decoder_ckpt $P/decoder_$1.pt --out $P/diag_cap_$1.json > $P/diag_cap_$1.log 2>&1
  echo "[DG-$1] rc=$? $(date '+%T')"
}
run A $U5 0.1 & PA=$!
run E $U6 0.0 & PE=$!
wait $PA; wait $PE
grep -h "SUM" $P/*.log
sha256sum $P/*.json
touch raw_s7/G2V4_CAP_DONE.flag
echo "=== cap pilot end $(date '+%F %T') ==="

#!/usr/bin/env bash
# scripts/cc_g2v4_m4_pilot.sh - G2 4th-attempt M4 pilot (CC, 2026-10-11). Train items only (s7_train_993) + dev600 NLL. No fresh items. Not a gate.
# M4 = M3 + text-dropout 0.5 (decoder must read the latent), warm start from M3 ckpts, 600 steps. GPU 5 (A), GPU 6 (E), GPU 7 (M3 baseline no-text diag).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
P=results_s7/g2v4pilot; mkdir -p $P raw_s7
echo "=== m4 pilot start $(date '+%F %T') ==="
sha256sum scripts/g2v4_m4_pilot_train.py scripts/g2v4_diag_iterate.py scripts/g2v3_m3_full_train.py configs/s7_train_993.json results_s7/g2v2/decoder_Am3_v36.pt results_s7/g2v2/decoder_Em3_v36.pt results_s7/g2v3/deq_backbone_A.pt results_s7/g2v3/deq_backbone_E.pt
for u in $U5 $U6 $U7; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' '); a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true); echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }; done
trn() { # mode gpu init_dec init_deq lambda
  CUDA_VISIBLE_DEVICES=$2 $PY scripts/g2v4_m4_pilot_train.py --mode $1 --device cuda:0 --items configs/s7_train_993.json --max_items 1000 --max_steps 600 --batch_size 4 \
    --lr_projector 1e-4 --lr_adapter 3e-4 --lr_deq 3e-5 --lambda_ift $5 --text_drop 0.5 --seed 2041 --init_decoder $3 --init_deq $4 \
    --out $P/m4_model_$1.pt --decoder_out $P/decoder_$1.pt --deq_out $P/deq_$1.pt --manifest $P/manifest_$1.json > raw_s7/g2v4_m4_train_$1.log 2>&1
}
dg() { # mode gpu deq dec out
  CUDA_VISIBLE_DEVICES=$2 $PY scripts/g2v4_diag_iterate.py --mode $1 --n 600 --ks 1 --notext --deq_ckpt $3 --decoder_ckpt $4 --out $5 > ${5%.json}.log 2>&1
}
( dg A $U7 results_s7/g2v3/deq_backbone_A.pt results_s7/g2v2/decoder_Am3_v36.pt $P/diag_m3_A.json; echo "[DG-M3-A] rc=$?"
  dg E $U7 results_s7/g2v3/deq_backbone_E.pt results_s7/g2v2/decoder_Em3_v36.pt $P/diag_m3_E.json; echo "[DG-M3-E] rc=$?" ) &
B=$!
( trn A $U5 results_s7/g2v2/decoder_Am3_v36.pt results_s7/g2v3/deq_backbone_A.pt 0.1; echo "[TR-A] rc=$? $(date '+%T')"
  dg A $U5 $P/deq_A.pt $P/decoder_A.pt $P/diag_m4_A.json; echo "[DG-M4-A] rc=$? $(date '+%T')" ) & PA=$!
( trn E $U6 results_s7/g2v2/decoder_Em3_v36.pt results_s7/g2v3/deq_backbone_E.pt 0.0; echo "[TR-E] rc=$? $(date '+%T')"
  dg E $U6 $P/deq_E.pt $P/decoder_E.pt $P/diag_m4_E.json; echo "[DG-M4-E] rc=$? $(date '+%T')" ) & PE=$!
wait $B; wait $PA; wait $PE
grep -h "SUM" $P/*.log
sha256sum $P/*.pt $P/*.json
touch raw_s7/G2V4_M4_PILOT_DONE.flag
echo "=== m4 pilot end $(date '+%F %T') ==="

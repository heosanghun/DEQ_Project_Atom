#!/usr/bin/env bash
# scripts/cc_g2v4_m7_rep.sh - CC 2026-10-11: M7 candidate replication (V2 projector + strong Phi_theta lr_deq 3e-4), new seeds, 3000 steps, train 993; dev600 NLL. Not a gate.
# GPU 5: A td0.5 (s2047) | GPU 6: E td0.5 (s2047) | GPU 7: A td1.0 (s2048) then E td1.0 (s2048)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
P=results_s7/g2v4m7rep; mkdir -p $P
echo "=== m7rep start $(date '+%F %T') ==="; sha256sum scripts/g2v4_m4_pilot_train.py scripts/g2v4_diag_iterate.py scripts/g2v4_proj_v2.py
for u in $U5 $U6 $U7; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' '); a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true); echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }; done
run() { # tag mode gpu lambda td seed
  CUDA_VISIBLE_DEVICES=$3 $PY scripts/g2v4_m4_pilot_train.py --mode $2 --device cuda:0 --items configs/s7_train_993.json --max_items 1000 --max_steps 3000 --batch_size 4 \
    --lr_projector 1e-4 --lr_adapter 3e-4 --lr_deq 3e-4 --lambda_ift $4 --text_drop $5 --seed $6 --proj_v2 \
    --out $P/model_$1.pt --decoder_out $P/decoder_$1.pt --deq_out $P/deq_$1.pt --manifest $P/manifest_$1.json > $P/train_$1.log 2>&1
  echo "[TR-$1] rc=$? $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$3 $PY scripts/g2v4_diag_iterate.py --mode $2 --n 600 --ks 1 --notext --deq_ckpt $P/deq_$1.pt --decoder_ckpt $P/decoder_$1.pt --out $P/diag_$1.json > $P/diag_$1.log 2>&1
  echo "[DG-$1] rc=$? $(date '+%T')"
}
run Ah A $U5 0.1 0.5 2047 &
run Eh E $U6 0.0 0.5 2047 &
( run Az A $U7 0.1 1.0 2048; run Ez E $U7 0.0 1.0 2048 ) &
wait
for t in Ah Eh Az Ez; do grep SUM $P/diag_$t.log | grep -v "res\|mean" | sed "s/^/$t /"; done
echo "=== m7rep end $(date '+%F %T') ==="

#!/usr/bin/env bash
# scripts/cc_g2v4_posctl.sh - CC 2026-10-11 positive control for the latent->LM path (PrefixProjectorV2, z-only training). Not a gate.
set -uo pipefail
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a
P=results_s7/g2v4posctl; mkdir -p $P
echo "=== posctl start $(date '+%F %T') ==="; sha256sum scripts/g2v4_posctl_latents.py scripts/g2v4_m4_pilot_train.py scripts/g2v4_diag_iterate.py
CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v4_posctl_latents.py > $P/lat.log 2>&1; echo "[LAT] rc=$?"; tail -n 1 $P/lat.log
CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v4_m4_pilot_train.py --mode E --device cuda:0 --items configs/s7_train_993.json --max_items 1000 --max_steps 1000 --batch_size 4 \
  --lr_projector 1e-4 --lr_adapter 3e-4 --lr_deq 3e-5 --lambda_ift 0.0 --text_drop 1.0 --seed 2045 --proj_v2 --zsrc $P/posctl_latents.pt \
  --out $P/model.pt --decoder_out $P/decoder.pt --deq_out $P/deq.pt --manifest $P/manifest.json > $P/train.log 2>&1; echo "[TR] rc=$?"
CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v4_diag_iterate.py --mode E --n 300 --ks 1 --notext --zsrc $P/posctl_latents.pt --deq_ckpt $P/deq.pt --decoder_ckpt $P/decoder.pt --out $P/diag.json > $P/diag.log 2>&1; echo "[DG] rc=$?"
grep SUM $P/diag.log | grep -v res
echo "=== posctl end $(date '+%F %T') ==="

#!/usr/bin/env bash
# scripts/cc_g2v4_traindiag.sh - CC 2026-10-11: own vs shifted latent NLL on TRAIN items (first 300 of s7_train_993) for M3 and LoRA decoders. Diagnostic.
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c
P=results_s7/g2v4train; mkdir -p $P
echo "=== traindiag start $(date '+%F %T') ==="; sha256sum scripts/g2v4_diag_iterate.py
Z="--items configs/s7_train_993.json --n 300 --ks 1 --notext --zstar results_s7/g2v2/zstar_zsrcinj_train1000.pt --zsrc results_s7/zsrc_train1000_P4.pt"
( CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v4_diag_iterate.py --mode E $Z --deq_ckpt results_s7/g2v4lora/deq_E.pt --decoder_ckpt results_s7/g2v4lora/decoder_E.pt --out $P/train_lora_E.json > $P/train_lora_E.log 2>&1; echo "[lora_E] rc=$?"
  CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v4_diag_iterate.py --mode E $Z --deq_ckpt results_s7/g2v4cap/deq_E.pt --decoder_ckpt results_s7/g2v4cap/decoder_E.pt --out $P/train_cap_E.json > $P/train_cap_E.log 2>&1; echo "[cap_E] rc=$?" ) &
( CUDA_VISIBLE_DEVICES=$U6 $PY scripts/g2v4_diag_iterate.py --mode E $Z --deq_ckpt results_s7/g2v3/deq_backbone_E.pt --decoder_ckpt results_s7/g2v2/decoder_Em3_v36.pt --out $P/train_m3_E.json > $P/train_m3_E.log 2>&1; echo "[m3_E] rc=$?" ) &
wait
grep -H "SUM" $P/*.log | grep "shift1\|zsrc_k1_nll\|notext\|mean_nll"
echo "=== traindiag end $(date '+%F %T') ==="

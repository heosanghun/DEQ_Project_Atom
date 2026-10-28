#!/usr/bin/env bash
# scripts/cc_g2v4_diag.sh - G2 4th-attempt diagnostic on dev600 all 600 items (CC, 2026-10-11). Not a gate. GPU 5 (A), GPU 6 (E).
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c
echo "=== g2v4 diag start $(date '+%F %T') ==="
sha256sum scripts/g2v4_diag_iterate.py scripts/g2v3_m3_full_train.py scripts/g2v3_m3_eval_latents_v2.py results_s7/g2v2/zstar_zsrcinj_dev600.pt results_s7/zsrc_heldout600.pt
for u in $U5 $U6; do echo "[CHECK] $u mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u) apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]')"; done
mkdir -p results_s7/g2v4diag
CUDA_VISIBLE_DEVICES=$U5 $PY scripts/g2v4_diag_iterate.py --mode A --n 600 --deq_ckpt results_s7/g2v3/deq_backbone_A.pt --decoder_ckpt results_s7/g2v2/decoder_Am3_v36.pt --out results_s7/g2v4diag/diag_A.json > raw_s7/g2v4_diag_A.log 2>&1 & PA=$!
CUDA_VISIBLE_DEVICES=$U6 $PY scripts/g2v4_diag_iterate.py --mode E --n 600 --deq_ckpt results_s7/g2v3/deq_backbone_E.pt --decoder_ckpt results_s7/g2v2/decoder_Em3_v36.pt --out results_s7/g2v4diag/diag_E.json > raw_s7/g2v4_diag_E.log 2>&1 & PE=$!
wait $PA; echo "[A] rc=$?"; wait $PE; echo "[E] rc=$?"
grep -h "SUM\|DONE\|Error\|FATAL" raw_s7/g2v4_diag_A.log raw_s7/g2v4_diag_E.log
touch raw_s7/G2V4_DIAG_DONE.flag
echo "=== g2v4 diag end $(date '+%F %T') ==="

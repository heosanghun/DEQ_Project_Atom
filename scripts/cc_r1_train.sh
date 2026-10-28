#!/usr/bin/env bash
# scripts/cc_r1_train.sh - stage-1 robustness R1 (CC 2026-10-12). Plan: shared/proposals/R1_ROBUSTNESS_PLAN_20261012.md. Train 993 + dev600 NLL. Not a gate.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_r1_stage; W=/tmp/cts_r1b
export PYTHONPATH=/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}  # try1 (/tmp/cts_r1) failed: g2v4_proj_v2 not importable when the trainer runs from /tmp
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== R1 train start $(date '+%F %T') ==="
[ -e $W ] && { echo "[FATAL] $W exists"; exit 1; }
mkdir -m 700 $W
sha256sum $S/g2v5_train.py scripts/g2v4_diag_iterate.py scripts/g2v4_proj_v2.py configs/s7_train_993.json
df -h /home /tmp | cat
AT=$(df --output=avail -k /tmp | tail -n 1 | tr -d ' '); [ "$AT" -gt 15728640 ] || { echo "[FATAL] /tmp < 15 GiB"; exit 1; }
for u in $U5 $U6 $U7; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' '); a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true); echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }; done
run() { # tag mode lambda seed gpu
  local x=""; [ "$2" = "E" ] && [ "$3" != "0.0" ] && x="--ift_e"
  CUDA_VISIBLE_DEVICES=$5 $PY $S/g2v5_train.py --mode $2 --device cuda:0 --items configs/s7_train_993.json --max_items 1000 --max_steps 3000 --batch_size 4 \
    --lr_projector 1e-4 --lr_adapter 3e-4 --lr_deq 3e-4 --lambda_ift $3 --text_drop 0.5 --seed $4 --proj_v2 $x \
    --out $W/model_$1.pt --decoder_out $W/decoder_$1.pt --deq_out $W/deq_$1.pt --manifest $W/manifest_$1.json > $W/train_$1.log 2>&1
  echo "[TR-$1] rc=$? $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$5 $PY scripts/g2v4_diag_iterate.py --mode $2 --n 600 --ks 1 --notext --deq_ckpt $W/deq_$1.pt --decoder_ckpt $W/decoder_$1.pt --out $W/diag_$1.json > $W/diag_$1.log 2>&1
  echo "[DG-$1] rc=$? $(date '+%T')"
}
( run R1A0s47 A 0.0 2047 $U5; run R1A0s49 A 0.0 2049 $U5; run R1A0s50 A 0.0 2050 $U5; run R1E1s47 E 0.1 2047 $U5 ) &
( run R1E1s49 E 0.1 2049 $U6; run R1E1s50 E 0.1 2050 $U6; run R1A1s49 A 0.1 2049 $U6 ) &
( run R1A1s50 A 0.1 2050 $U7; run R1E0s49 E 0.0 2049 $U7; run R1E0s50 E 0.0 2050 $U7 ) &
wait
grep -h "n_text_dropped\|\"ift_e\"" $W/manifest_*.json | sort | uniq -c
sha256sum $W/*.json
touch $W/R1_TRAIN_DONE.flag
echo "=== R1 train end $(date '+%F %T') ==="

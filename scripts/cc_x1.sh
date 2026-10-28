#!/usr/bin/env bash
# scripts/cc_x1.sh - X1: prefix interface + LM LoRA r16 with ORACLE latents (CC 2026-10-19). Plan: shared/proposals/X_INTERFACE_PLAN_20261019.md.
# Exploratory; train 993 + dev 600 only. Reuses S2 oracle latents (/tmp/cts_s2/oracle_latents.pt, 568f84e6).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_x_stage; W=/tmp/cts_x1; O=results_s7/p6r/g2v8x1; ORC=/tmp/cts_s2/oracle_latents.pt; ORD=6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== X1 start $(date '+%F %T') ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
[ "$(sha256sum $ORC | cut -d' ' -f1)" = "568f84e615c9f97b164dcb016767e96927dc6b55452499cf1749d52baffc1429" ] || { echo "[FATAL] oracle latents SHA"; exit 1; }
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v7_train.py $S/g2v7_diag.py $S/make_s0spec_x.py scripts/s7_eval_gate_p6_v36h.py scripts/g2v4_proj_v2.py
df -h /home /tmp | cat
AT=$(df --output=avail -k /tmp | tail -n 1 | tr -d ' '); [ "$AT" -gt 6291456 ] || { echo "[FATAL] /tmp < 6 GiB"; exit 1; }
for u in $U5 $U6 $U7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
run() { # tag seed gpu
  CUDA_VISIBLE_DEVICES=$3 $PY $S/g2v7_train.py --mode E --no_step --device cuda:0 --items configs/s7_train_993.json --max_items 1000 --max_steps 3000 --batch_size 4 \
    --lr_projector 1e-4 --lr_adapter 3e-4 --lr_deq 3e-4 --lambda_ift 0.0 --text_drop 0.5 --seed $2 --proj_v2 --zsrc $ORC \
    --lm_lora_rank 16 --lr_lm_lora 1e-4 \
    --out $W/model_$1.pt --decoder_out $W/decoder_$1.pt --deq_out $W/deq_$1.pt --manifest $W/manifest_$1.json > $W/train_$1.log 2>&1
  echo "[TR-$1] rc=$? $(date '+%T')"
  grep -h "LM-LORA" $W/train_$1.log | head -n 1
  CUDA_VISIBLE_DEVICES=$3 $PY $S/g2v7_diag.py --mode E --no_step --n 600 --ks 1 --notext --deq_ckpt $W/deq_$1.pt --decoder_ckpt $W/decoder_$1.pt \
    --zsrc $ORC --zstar results_s7/g2v2/zstar_zsrcinj_dev600.pt --out $W/diag_$1.json > $W/diag_$1.log 2>&1
  echo "[DG-$1] rc=$? $(date '+%T')"
  cp -n $W/decoder_$1.pt results_s7/g2v2/decoder_${1}_v36.pt
  CUDA_VISIBLE_DEVICES=$3 $PY scripts/s7_eval_gate_p6_v36h.py --cond E1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_${1}_v36.pt --zsrc $ORC --order_sha $ORD --out $O/gen_$1.jsonl > $W/gen_$1.log 2>&1
  echo "[GEN-$1] rc=$? rows=$(wc -l < $O/gen_$1.jsonl) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$3 $PY scripts/s7_eval_gate_p6_v36h.py --cond E2 --shift 1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_${1}_v36.pt --zsrc $ORC --order_sha $ORD --out $O/genB1_$1.jsonl > $W/genB1_$1.log 2>&1
  echo "[GENS-$1] rc=$? rows=$(wc -l < $O/genB1_$1.jsonl) $(date '+%T')"
}
run X1O47 2047 $U5 &
run X1O49 2049 $U6 &
run X1O50 2050 $U7 &
wait
grep -h "LM-LORA\|PROJ-V2" $W/gen_X1O47.log | head -n 2
$PY $S/make_s0spec_x.py /tmp/cts_x1/s0spec.json $O X1O47,X1O49,X1O50
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py /tmp/cts_x1/s0spec.json /tmp/cts_x1/x1_items.json" > $W/grade.log 2>&1
grep "\[S0\]" $W/grade.log
sha256sum $O/*.jsonl $W/x1_items.json $W/diag_*.json $W/manifest_*.json
touch $W/X1_DONE.flag
echo "=== X1 end $(date '+%F %T') ==="

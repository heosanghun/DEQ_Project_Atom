#!/usr/bin/env bash
# scripts/cc_s2.sh - S2 oracle-latent interface test (CC 2026-10-14). Plan: shared/proposals/S2_ORACLE_PLAN_20261014.md. Exploratory; train 993 + dev 600 only.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_s2_stage; W=/tmp/cts_s2; O=results_s7/p6r/g2v7s2; ORD=6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== S2 start $(date '+%F %T') ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v7_train.py $S/g2v7_diag.py $S/g2v7_oracle_latents.py scripts/s7_eval_gate_p6_v36g.py scripts/g2v4_proj_v2.py
df -h /home /tmp | cat
AT=$(df --output=avail -k /tmp | tail -n 1 | tr -d ' '); [ "$AT" -gt 8388608 ] || { echo "[FATAL] /tmp < 8 GiB"; exit 1; }
for u in $U5 $U6 $U7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
CUDA_VISIBLE_DEVICES=$U5 $PY $S/g2v7_oracle_latents.py $W/oracle_latents.pt > $W/oracle.log 2>&1; echo "[ORC] rc=$? $(tail -n 1 $W/oracle.log)"
[ -s $W/oracle_latents.pt ] || { echo "[FATAL] no oracle latents"; exit 1; }
sha256sum $W/oracle_latents.pt
run() { # tag seed gpu
  CUDA_VISIBLE_DEVICES=$3 $PY $S/g2v7_train.py --mode E --no_step --device cuda:0 --items configs/s7_train_993.json --max_items 1000 --max_steps 3000 --batch_size 4 \
    --lr_projector 1e-4 --lr_adapter 3e-4 --lr_deq 3e-4 --lambda_ift 0.0 --text_drop 0.5 --seed $2 --proj_v2 --zsrc $W/oracle_latents.pt \
    --out $W/model_$1.pt --decoder_out $W/decoder_$1.pt --deq_out $W/deq_$1.pt --manifest $W/manifest_$1.json > $W/train_$1.log 2>&1
  echo "[TR-$1] rc=$? $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$3 $PY $S/g2v7_diag.py --mode E --no_step --n 600 --ks 1 --notext --deq_ckpt $W/deq_$1.pt --decoder_ckpt $W/decoder_$1.pt \
    --zsrc $W/oracle_latents.pt --zstar results_s7/g2v2/zstar_zsrcinj_dev600.pt --out $W/diag_$1.json > $W/diag_$1.log 2>&1
  echo "[DG-$1] rc=$? $(date '+%T')"
  cp -n $W/decoder_$1.pt results_s7/g2v2/decoder_${1}_v36.pt
  CUDA_VISIBLE_DEVICES=$3 $PY scripts/s7_eval_gate_p6_v36g.py --cond E1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_${1}_v36.pt --zsrc $W/oracle_latents.pt --order_sha $ORD --out $O/gen_$1.jsonl > $W/gen_$1.log 2>&1
  echo "[GEN-$1] rc=$? rows=$(wc -l < $O/gen_$1.jsonl) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$3 $PY scripts/s7_eval_gate_p6_v36g.py --cond E2 --shift 1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_${1}_v36.pt --zsrc $W/oracle_latents.pt --order_sha $ORD --out $O/genB1_$1.jsonl > $W/genB1_$1.log 2>&1
  echo "[GENS-$1] rc=$? rows=$(wc -l < $O/genB1_$1.jsonl) $(date '+%T')"
}
run S2O47 2047 $U5 &
run S2O49 2049 $U6 &
run S2O50 2050 $U7 &
wait
grep -h "PROJ-V2" $W/gen_*.log | sort | uniq -c
$PY $S/make_s0spec_s2.py /tmp/cts_s2/s0spec.json
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py /tmp/cts_s2/s0spec.json /tmp/cts_s2/s2_items.json" > $W/grade.log 2>&1
grep "\[S0\]" $W/grade.log
sha256sum $O/*.jsonl $W/s2_items.json $W/diag_*.json $W/manifest_*.json
touch $W/S2_DONE.flag
echo "=== S2 end $(date '+%F %T') ==="

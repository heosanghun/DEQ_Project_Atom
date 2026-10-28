#!/usr/bin/env bash
# scripts/cc_s1.sh - S1 format-instruction-fixed experiment (CC 2026-10-13). Plan: shared/proposals/S1_FORMAT_FIXED_PLAN_20261013.md. Train 993 + dev600 only.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_s1_stage; W=/tmp/cts_s1; O=results_s7/p6r/g2v6s1; ORD=6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== S1 start $(date '+%F %T') ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v6_train.py $S/g2v6_diag.py scripts/s7_eval_gate_p6_v36g.py scripts/g2v4_proj_v2.py scripts/g2v3_m3_eval_latents_v2.py
df -h /home /tmp | cat
AV=$(df --output=avail -k /home | tail -n 1 | tr -d ' '); [ "$AV" -gt 1048576 ] || { echo "[FATAL] /home < 1 GiB"; exit 1; }
for u in $U5 $U6 $U7; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' '); a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true); echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }; done
tr1() { # tag mode seed gpu
  CUDA_VISIBLE_DEVICES=$4 $PY $S/g2v6_train.py --mode $2 --device cuda:0 --items configs/s7_train_993.json --max_items 1000 --max_steps 3000 --batch_size 4 \
    --lr_projector 1e-4 --lr_adapter 3e-4 --lr_deq 3e-4 --lambda_ift 0.0 --text_drop 0.5 --seed $3 --proj_v2 \
    --out $W/model_$1.pt --decoder_out $W/decoder_$1.pt --deq_out $W/deq_$1.pt --manifest $W/manifest_$1.json > $W/train_$1.log 2>&1
  echo "[TR-$1] rc=$? $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$4 $PY $S/g2v6_diag.py --mode $2 --n 600 --ks 1 --notext --deq_ckpt $W/deq_$1.pt --decoder_ckpt $W/decoder_$1.pt --out $W/diag_$1.json > $W/diag_$1.log 2>&1
  echo "[DG-$1] rc=$? $(date '+%T')"
  cp -n $W/decoder_$1.pt results_s7/g2v2/decoder_${1}_v36.pt
  if [ $2 = A ]; then
    CUDA_VISIBLE_DEVICES=$4 $PY scripts/g2v3_m3_eval_latents_v2.py --mode A --device cuda:0 --items results_s7/p6_item_order.json --deq_ckpt $W/deq_$1.pt \
      --zstar results_s7/g2v2/zstar_zsrcinj_dev600.pt --zsrc results_s7/zsrc_heldout600.pt --out $W/lat_$1.pt > $W/lat_$1.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$4 $PY scripts/g2v3_m3_eval_latents_v2.py --mode E --device cuda:0 --items results_s7/p6_item_order.json --deq_ckpt $W/deq_$1.pt \
      --zsrc results_s7/zsrc_heldout600.pt --out $W/lat_$1.pt > $W/lat_$1.log 2>&1
  fi
  grep -q "applied 436 DEQ tensors, unexpected 0" $W/lat_$1.log && echo "[LAT-OK] $1" || echo "[LAT-FAIL] $1"
}
gm() { # tag mode gpu  -> matched and shuffled generation
  local zarg; [ $2 = A ] && zarg="--zstar $W/lat_$1.pt" || zarg="--zsrc $W/lat_$1.pt"
  local mc; [ $2 = A ] && mc=A || mc=E1
  CUDA_VISIBLE_DEVICES=$3 $PY scripts/s7_eval_gate_p6_v36g.py --cond $mc --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_${1}_v36.pt $zarg --order_sha $ORD --out $O/gen_$1.jsonl > $W/gen_$1.log 2>&1
  echo "[GEN-$1] rc=$? rows=$(wc -l < $O/gen_$1.jsonl) $(date '+%T')"
  if [ $2 = A ]; then
    CUDA_VISIBLE_DEVICES=$3 $PY scripts/s7_eval_gate_p6_v36g.py --cond B --shift 1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_${1}_v36.pt --zstar $W/lat_$1.pt --order_sha $ORD --out $O/genB1_$1.jsonl > $W/genB1_$1.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$3 $PY scripts/s7_eval_gate_p6_v36g.py --cond E2 --shift 1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_${1}_v36.pt --zsrc $W/lat_$1.pt --order_sha $ORD --out $O/genB1_$1.jsonl > $W/genB1_$1.log 2>&1
  fi
  echo "[GENS-$1] rc=$? rows=$(wc -l < $O/genB1_$1.jsonl) $(date '+%T')"
}
( tr1 S1A47 A 2047 $U5; tr1 S1E47 E 2047 $U5; gm S1A47 A $U5; gm S1E47 E $U5
  CUDA_VISIBLE_DEVICES=$U5 $PY scripts/s7_eval_gate_p6_v36g.py --cond C --device cuda:0 --order_sha $ORD --out $O/gen_C.jsonl > $W/gen_C.log 2>&1; echo "[GEN-C] rc=$? $(date '+%T')" ) &
( tr1 S1A49 A 2049 $U6; tr1 S1E49 E 2049 $U6; gm S1A49 A $U6; gm S1E49 E $U6 ) &
( tr1 S1A50 A 2050 $U7; tr1 S1E50 E 2050 $U7; gm S1A50 A $U7; gm S1E50 E $U7 ) &
wait
grep -h "PROJ-V2" $W/gen_S1*.log | sort | uniq -c
# per-item grading with the locked grader
$PY - <<'PY'
import json
O = "results_s7/p6r/g2v6s1/"; D = "results_s7/p6_item_order.json"; spec = {"C": [O + "gen_C.jsonl", D]}
for t in ["S1A47", "S1A49", "S1A50", "S1E47", "S1E49", "S1E50"]:
    spec[t] = [O + f"gen_{t}.jsonl", D]; spec["shuf_" + t] = [O + f"genB1_{t}.jsonl", D]
json.dump(spec, open("/tmp/cts_s1/s0spec.json", "w"))
PY
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py /tmp/cts_s1/s0spec.json /tmp/cts_s1/s1_items.json" > $W/grade.log 2>&1
grep "\[S0\]" $W/grade.log
sha256sum $O/*.jsonl $W/s1_items.json $W/diag_*.json
touch $W/S1_DONE.flag
echo "=== S1 end $(date '+%F %T') ==="

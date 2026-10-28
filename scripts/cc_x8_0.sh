#!/usr/bin/env bash
# scripts/cc_x8_0.sh - X8-0 control (CC 2026-10-21): problem-only latents (pure-LM last-64 hidden states of the PROBLEM text) through the
# X5 recipe (style-matched targets + answer_weight 5, gen_budget 480, 1500 steps). Seeds 2047/2049/2050 on GPUs 5/6/7, matched + shuffled.
# Plan: shared/proposals/X8_READABLE_LATENT_DEQ_20261021.md §2 (rule: pooled matched - shuffled lenient CI lower > 0). Checks as cc_x7.sh.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/tmp/cts_x_stage:/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_x_stage; W=/tmp/cts_x8_0; O=results_s7/p6r/g2v9x8_0
TGT=results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl; ZP=$W/problem_latents.pt
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== X8-0 start $(date '+%F %T') launcher_sha $(sha256sum "$0" | cut -d' ' -f1) ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v9_xattn.py $S/g2v10_train.py $S/g2v9_gen.py $S/g2v8x_problem_latents.py $S/make_s0spec_x.py $TGT
[ "$(wc -l < $TGT)" -eq 993 ] || { echo "[FATAL] targets rows"; exit 1; }
[ "$($PY -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}))" $TGT)" -eq 993 ] || { echo "[FATAL] targets unique ids"; exit 1; }
$PY $S/g2v9_xattn.py --selftest || { echo "[FATAL] selftest"; exit 1; }
for u in $U5 $U6 $U7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
# Step 1: problem-only latents (train 993 + dev 600) on GPU 5
CUDA_VISIBLE_DEVICES=$U5 $PY $S/g2v8x_problem_latents.py $ZP > $W/latents.log 2>&1; rc=$?
echo "[LAT] rc=$rc $(date '+%T') $(grep -h 'SAVE' $W/latents.log | cut -c1-120)"; [ "$rc" -eq 0 ] && [ -s $ZP ] || { echo "[FATAL] latents rc=$rc"; exit 1; }
sha256sum $ZP
run() { # tag seed gpu
  local T=$1 seed=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v10_train.py --zfile $ZP --targets $TGT --seed $seed --max_steps 1500 --text_drop 0.0 --latent_drop 0.0 --answer_weight 5 --gen_budget 480 --out $W/bridge_$T.pt --manifest $W/manifest_$T.json > $W/train_$T.log 2>&1
  local rc=$?; echo "[TR-$T] rc=$rc $(date '+%T') $(grep -h 'Step' $W/train_$T.log | tail -n 1 | cut -c1-110)"; grep -h 'DATA\]' $W/train_$T.log | cut -c1-230
  [ "$rc" -eq 0 ] && [ -s $W/bridge_$T.pt ] || { echo "[FATAL-$T] training rc=$rc"; echo 1 > $W/rc_$T; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v9_gen.py --ckpt $W/bridge_$T.pt --zfile $ZP --cond matched --out $O/gen_$T.jsonl > $W/gen_$T.log 2>&1; rc=$?
  echo "[GEN-$T] rc=$rc rows=$(wc -l < $O/gen_$T.jsonl) $(date '+%T')"
  [ "$rc" -eq 0 ] || { echo 1 > $W/rc_$T; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v9_gen.py --ckpt $W/bridge_$T.pt --zfile $ZP --cond shuffled --out $O/genB1_$T.jsonl > $W/genB1_$T.log 2>&1; rc=$?
  echo "[GENS-$T] rc=$rc rows=$(wc -l < $O/genB1_$T.jsonl) $(date '+%T')"
  echo $rc > $W/rc_$T
}
run X80P47 2047 $U5 &
run X80P49 2049 $U6 &
run X80P50 2050 $U7 &
wait
ok=1
for T in X80P47 X80P49 X80P50; do
  rc=$(cat $W/rc_$T 2>/dev/null || echo 9)
  for f in gen genB1; do
    n=$(wc -l < $O/${f}_$T.jsonl 2>/dev/null || echo 0); u=$($PY -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}))" $O/${f}_$T.jsonl 2>/dev/null || echo 0)
    echo "[CHECK-$f-$T] rc=$rc rows=$n unique_ids=$u"
    [ "$rc" -eq 0 ] && [ "$n" -eq 600 ] && [ "$u" -eq 600 ] || ok=0
  done
done
[ "$ok" -eq 1 ] || { echo "[FATAL] one or more runs incomplete; no DONE flag"; exit 1; }
$PY $S/make_s0spec_x.py $W/s0spec.json $O X80P47,X80P49,X80P50
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py $W/s0spec.json $W/x8_0_items.json" > $W/grade.log 2>&1; rc=$?; grep "S0" $W/grade.log
[ "$rc" -eq 0 ] && [ "$(grep -c '^\[S0\] ' $W/grade.log)" -eq 7 ] && [ "$(grep -c 'n 600 ' $W/grade.log)" -eq 7 ] || { echo "[FATAL] grading rc=$rc"; exit 1; }
sha256sum $O/*.jsonl $W/x8_0_items.json $W/manifest_*.json $W/bridge_*.pt $ZP
touch $W/X8_0_DONE.flag
echo "=== X8-0 end $(date '+%F %T') ==="

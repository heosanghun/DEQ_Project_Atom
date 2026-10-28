#!/usr/bin/env bash
# scripts/cc_x6.sh - X6 (CC 2026-10-20 13:3x). Plan: shared/proposals/X_INTERFACE_PLAN_20261019.md "X5 결과와 X6 사전등록" (rules fixed before results).
# X6-R: oracle seeds 2049 (GPU 5) and 2050 (GPU 6), X5 recipe (1500 steps, answer_weight 5, gen_budget 480).
# X6-S: oracle seed 2047, answer_weight 20, 3000 steps (GPU 7).
# Same checks as cc_x5.sh rev3: rc propagation, 600 rows + 600 unique ids per generation, grading labels, DONE flag last.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/tmp/cts_x_stage:/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_x_stage; W=/tmp/cts_x6; O=results_s7/p6r/g2v9x6; ORC=/tmp/cts_s2/oracle_latents.pt
TGT=results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== X6 start $(date '+%F %T') ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v9_xattn.py $S/g2v10_train.py $S/g2v9_gen.py $ORC $TGT
[ "$(wc -l < $TGT)" -eq 993 ] || { echo "[FATAL] targets rows $(wc -l < $TGT)"; exit 1; }
[ "$($PY -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}))" $TGT)" -eq 993 ] || { echo "[FATAL] targets unique ids"; exit 1; }
$PY $S/g2v9_xattn.py --selftest || { echo "[FATAL] selftest"; exit 1; }
for u in $U5 $U6 $U7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
run() { # tag seed gpu steps answer_weight  -> train, gen matched, gen shuffled; writes $W/rc_$tag (0 = all ok)
  local T=$1 seed=$2 gpu=$3 steps=$4 aw=$5
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v10_train.py --zfile $ORC --targets $TGT --seed $seed --max_steps $steps --text_drop 0.0 --latent_drop 0.0 --answer_weight $aw --gen_budget 480 --out $W/bridge_$T.pt --manifest $W/manifest_$T.json > $W/train_$T.log 2>&1
  local rc=$?; echo "[TR-$T] rc=$rc $(date '+%T') $(grep -h 'Step' $W/train_$T.log | tail -n 1 | cut -c1-110)"; grep -h 'DATA\]' $W/train_$T.log | cut -c1-230
  [ "$rc" -eq 0 ] && [ -s $W/bridge_$T.pt ] || { echo "[FATAL-$T] training rc=$rc"; echo 1 > $W/rc_$T; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v9_gen.py --ckpt $W/bridge_$T.pt --zfile $ORC --cond matched --out $O/gen_$T.jsonl > $W/gen_$T.log 2>&1; rc=$?
  echo "[GEN-$T] rc=$rc rows=$(wc -l < $O/gen_$T.jsonl) $(date '+%T')"
  [ "$rc" -eq 0 ] || { echo 1 > $W/rc_$T; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v9_gen.py --ckpt $W/bridge_$T.pt --zfile $ORC --cond shuffled --out $O/genB1_$T.jsonl > $W/genB1_$T.log 2>&1; rc=$?
  echo "[GENS-$T] rc=$rc rows=$(wc -l < $O/genB1_$T.jsonl) $(date '+%T')"
  echo $rc > $W/rc_$T
}
run X6R49 2049 $U5 1500 5 &
run X6R50 2050 $U6 1500 5 &
run X6S47 2047 $U7 3000 20 &
wait
ok=1
for T in X6R49 X6R50 X6S47; do
  rc=$(cat $W/rc_$T 2>/dev/null || echo 9)
  for f in gen genB1; do
    n=$(wc -l < $O/${f}_$T.jsonl 2>/dev/null || echo 0); u=$($PY -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}))" $O/${f}_$T.jsonl 2>/dev/null || echo 0)
    echo "[CHECK-$f-$T] rc=$rc rows=$n unique_ids=$u"
    [ "$rc" -eq 0 ] && [ "$n" -eq 600 ] && [ "$u" -eq 600 ] || ok=0
  done
done
[ "$ok" -eq 1 ] || { echo "[FATAL] one or more runs incomplete; no DONE flag"; exit 1; }
$PY $S/make_s0spec_x.py $W/s0spec.json $O X6R49,X6R50,X6S47
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py $W/s0spec.json $W/x6_items.json" > $W/grade.log 2>&1; rc=$?; grep "S0" $W/grade.log
[ "$rc" -eq 0 ] && [ "$(grep -c '^\[S0\] ' $W/grade.log)" -eq 7 ] && [ "$(grep -c 'n 600 ' $W/grade.log)" -eq 7 ] || { echo "[FATAL] grading rc=$rc"; exit 1; }
sha256sum $O/*.jsonl $W/x6_items.json $W/manifest_*.json $W/bridge_*.pt
touch $W/X6_DONE.flag
echo "=== X6 end $(date '+%F %T') ==="

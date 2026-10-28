#!/usr/bin/env bash
# scripts/cc_x7.sh - X7 Stage AE under the X5 recipe (CC 2026-10-20 20:3x). Plan: X_INTERFACE_PLAN_20261019.md "X6 결과와 X7 사전등록" (rules fixed before results).
# A = canonical z*' (zstar_zsrcinj train1000 / dev600), E = z_src (zsrc_train1000_P4 / zsrc_heldout600). Seeds 2047/2049/2050. matched + shuffled(shift 1).
# Recipe = X5: style-matched targets (own C-fmt generations on train 993) + answer_weight 5, gen_budget 480, 1500 steps, text_drop 0, latent_drop 0.
# Round 1: A47 (GPU5), E47 (GPU6), A49 (GPU7). Round 2: E49 (GPU5), A50 (GPU6), E50 (GPU7). Checks as cc_x6.sh. Prints its own SHA.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/tmp/cts_x_stage:/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_x_stage; W=/tmp/cts_x7; O=results_s7/p6r/g2v9x7
TGT=results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl
ZA_TR=results_s7/g2v2/zstar_zsrcinj_train1000.pt; ZA_DEV=results_s7/g2v2/zstar_zsrcinj_dev600.pt
ZE_TR=results_s7/zsrc_train1000_P4.pt; ZE_DEV=results_s7/zsrc_heldout600.pt
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== X7 start $(date '+%F %T') launcher_sha $(sha256sum "$0" | cut -d' ' -f1) ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v9_xattn.py $S/g2v10_train.py $S/g2v9_gen.py $S/make_s0spec_x.py $TGT $ZA_TR $ZA_DEV $ZE_TR $ZE_DEV
[ "$(wc -l < $TGT)" -eq 993 ] || { echo "[FATAL] targets rows $(wc -l < $TGT)"; exit 1; }
[ "$($PY -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}))" $TGT)" -eq 993 ] || { echo "[FATAL] targets unique ids"; exit 1; }
$PY $S/g2v9_xattn.py --selftest || { echo "[FATAL] selftest"; exit 1; }
for u in $U5 $U6 $U7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
run() { # tag ztrain zdev seed gpu -> train, gen matched, gen shuffled; writes $W/rc_$tag
  local T=$1 ztr=$2 zdev=$3 seed=$4 gpu=$5
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v10_train.py --zfile $ztr --targets $TGT --seed $seed --max_steps 1500 --text_drop 0.0 --latent_drop 0.0 --answer_weight 5 --gen_budget 480 --out $W/bridge_$T.pt --manifest $W/manifest_$T.json > $W/train_$T.log 2>&1
  local rc=$?; echo "[TR-$T] rc=$rc $(date '+%T') $(grep -h 'Step' $W/train_$T.log | tail -n 1 | cut -c1-110)"; grep -h 'DATA\]' $W/train_$T.log | cut -c1-230
  [ "$rc" -eq 0 ] && [ -s $W/bridge_$T.pt ] || { echo "[FATAL-$T] training rc=$rc"; echo 1 > $W/rc_$T; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v9_gen.py --ckpt $W/bridge_$T.pt --zfile $zdev --cond matched --out $O/gen_$T.jsonl > $W/gen_$T.log 2>&1; rc=$?
  echo "[GEN-$T] rc=$rc rows=$(wc -l < $O/gen_$T.jsonl) $(date '+%T')"
  [ "$rc" -eq 0 ] || { echo 1 > $W/rc_$T; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v9_gen.py --ckpt $W/bridge_$T.pt --zfile $zdev --cond shuffled --out $O/genB1_$T.jsonl > $W/genB1_$T.log 2>&1; rc=$?
  echo "[GENS-$T] rc=$rc rows=$(wc -l < $O/genB1_$T.jsonl) $(date '+%T')"
  echo $rc > $W/rc_$T
}
run X7A47 $ZA_TR $ZA_DEV 2047 $U5 &
run X7E47 $ZE_TR $ZE_DEV 2047 $U6 &
run X7A49 $ZA_TR $ZA_DEV 2049 $U7 &
wait
echo "=== X7 round 1 done $(date '+%F %T') ==="
run X7E49 $ZE_TR $ZE_DEV 2049 $U5 &
run X7A50 $ZA_TR $ZA_DEV 2050 $U6 &
run X7E50 $ZE_TR $ZE_DEV 2050 $U7 &
wait
ok=1
for T in X7A47 X7E47 X7A49 X7E49 X7A50 X7E50; do
  rc=$(cat $W/rc_$T 2>/dev/null || echo 9)
  for f in gen genB1; do
    n=$(wc -l < $O/${f}_$T.jsonl 2>/dev/null || echo 0); u=$($PY -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}))" $O/${f}_$T.jsonl 2>/dev/null || echo 0)
    echo "[CHECK-$f-$T] rc=$rc rows=$n unique_ids=$u"
    [ "$rc" -eq 0 ] && [ "$n" -eq 600 ] && [ "$u" -eq 600 ] || ok=0
  done
done
[ "$ok" -eq 1 ] || { echo "[FATAL] one or more runs incomplete; no DONE flag"; exit 1; }
$PY $S/make_s0spec_x.py $W/s0spec.json $O X7A47,X7E47,X7A49,X7E49,X7A50,X7E50
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py $W/s0spec.json $W/x7_items.json" > $W/grade.log 2>&1; rc=$?; grep "S0" $W/grade.log
[ "$rc" -eq 0 ] && [ "$(grep -c '^\[S0\] ' $W/grade.log)" -eq 13 ] && [ "$(grep -c 'n 600 ' $W/grade.log)" -eq 13 ] || { echo "[FATAL] grading rc=$rc"; exit 1; }
sha256sum $O/*.jsonl $W/x7_items.json $W/manifest_*.json $W/bridge_*.pt
touch $W/X7_DONE.flag
echo "=== X7 end $(date '+%F %T') ==="

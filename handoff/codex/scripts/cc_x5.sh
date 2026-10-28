#!/usr/bin/env bash
# scripts/cc_x5.sh (rev2, Codex REPORT-C003 D6: rc propagation, row/unique-id checks before the DONE flag) - X5 Stage O: style-matched targets + answer-span weighting, oracle latents, seed 2047 (CC 2026-10-20).
# Plan: shared/proposals/X_INTERFACE_PLAN_20261019.md ("X5 설계"; latent_drop set to 0.0 after Codex REPORT-C001 D1: the no-latent path carries no gradient, so it cannot regularise). Rules fixed before results: Stage O (matched-shuffled CI>0 AND matched-C >= +5%p)
# plus harm check (shuffled-C CI includes 0). Requires /tmp/cts_x5base/X5BASE_DONE.flag (own C-fmt generations on train 993). GPU 5 (train+gen) and GPU 6 (shuffled gen).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/tmp/cts_x_stage:/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_x_stage; W=/tmp/cts_x5; O=results_s7/p6r/g2v9x5; ORC=/tmp/cts_s2/oracle_latents.pt
TGT=results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c
until [ -e /tmp/cts_x5base/X5BASE_DONE.flag ]; do sleep 60; done
echo "=== X5 start $(date '+%F %T') ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v9_xattn.py $S/g2v10_train.py $S/g2v9_gen.py $ORC $TGT
[ "$(wc -l < $TGT)" -eq 993 ] || { echo "[FATAL] targets rows $(wc -l < $TGT)"; exit 1; }
[ "$($PY -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}))" $TGT)" -eq 993 ] || { echo "[FATAL] targets unique ids"; exit 1; }
$PY $S/g2v9_xattn.py --selftest || { echo "[FATAL] selftest"; exit 1; }
for u in $U5 $U6; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
T=X5O47
CUDA_VISIBLE_DEVICES=$U5 $PY $S/g2v10_train.py --zfile $ORC --targets $TGT --seed 2047 --max_steps 1500 --text_drop 0.0 --latent_drop 0.0 --answer_weight 5 --gen_budget 480 --out $W/bridge_$T.pt --manifest $W/manifest_$T.json > $W/train_$T.log 2>&1
rc=$?; echo "[TR-$T] rc=$rc $(date '+%T') $(grep -h 'Step' $W/train_$T.log | tail -n 1 | cut -c1-110)"; grep -h 'DATA\]\|SAVE\]' $W/train_$T.log
[ "$rc" -eq 0 ] && [ -s $W/bridge_$T.pt ] || { echo "[FATAL] training rc=$rc"; exit 1; }
( CUDA_VISIBLE_DEVICES=$U5 $PY $S/g2v9_gen.py --ckpt $W/bridge_$T.pt --zfile $ORC --cond matched --out $O/gen_$T.jsonl > $W/gen_$T.log 2>&1; echo $? > $W/rc_gen_$T ) &
( CUDA_VISIBLE_DEVICES=$U6 $PY $S/g2v9_gen.py --ckpt $W/bridge_$T.pt --zfile $ORC --cond shuffled --out $O/genB1_$T.jsonl > $W/genB1_$T.log 2>&1; echo $? > $W/rc_genB1_$T ) &
wait
for f in gen genB1; do
  rc=$(cat $W/rc_${f}_$T); n=$(wc -l < $O/${f}_$T.jsonl); u=$($PY -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}))" $O/${f}_$T.jsonl)
  echo "[GEN-$f-$T] rc=$rc rows=$n unique_ids=$u $(date '+%T')"
  [ "$rc" -eq 0 ] && [ "$n" -eq 600 ] && [ "$u" -eq 600 ] || { echo "[FATAL] generation $f rc=$rc rows=$n unique=$u"; exit 1; }
done
$PY $S/make_s0spec_x.py $W/s0spec.json $O $T
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py $W/s0spec.json $W/x5_items.json" > $W/grade.log 2>&1; rc=$?; grep "S0" $W/grade.log
[ "$rc" -eq 0 ] && [ "$(grep -c '^\[S0\] ' $W/grade.log)" -eq 3 ] && [ "$(grep -c 'n 600 ' $W/grade.log)" -eq 3 ] || { echo "[FATAL] grading rc=$rc"; exit 1; }
sha256sum $O/*.jsonl $W/x5_items.json $W/manifest_*.json $W/bridge_*.pt
touch $W/X5_DONE.flag
echo "=== X5 end $(date '+%F %T') ==="

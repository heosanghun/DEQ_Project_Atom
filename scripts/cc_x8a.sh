#!/usr/bin/env bash
# scripts/cc_x8a.sh - X8-A: DEQ (LoRA/routing/gamma) + bridge joint training under the X5 readout objective (CC 2026-10-23).
# Plan: shared/proposals/X8_READABLE_LATENT_DEQ_20261021.md §1-2, §7 (rules fixed before results). MODE=smoke runs a 5-step, 16-item
# check on GPU 5 only (no generation). MODE=full: seeds 2047/2049/2050 on GPUs 5/6/7: train -> dev latents with trained DEQ -> gen matched/shuffled -> grade.
# E condition = X7E (same recipe, z_src, already graded: shared x7_items). C = C-fmt. Checks as cc_x7.sh. Prints its own SHA.
set -uo pipefail
MODE=${1:-full}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/tmp/cts_x_stage:/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_x_stage; TGT=results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl
ZA_TR=results_s7/g2v2/zstar_zsrcinj_train1000.pt; ZA_DEV=results_s7/g2v2/zstar_zsrcinj_dev600.pt
ZE_TR=results_s7/zsrc_train1000_P4.pt; ZE_DEV=results_s7/zsrc_heldout600.pt
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
if [ "$MODE" = "smoke" ]; then W=/tmp/cts_x8a_smoke; O=results_s7/p6r/g2v9x8a_smoke; else W=/tmp/cts_x8a; O=results_s7/p6r/g2v9x8a; fi
echo "=== X8-A $MODE start $(date '+%F %T') launcher_sha $(sha256sum "$0" | cut -d' ' -f1) ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v9_xattn.py $S/g2v10_train.py $S/g2v11_x8a_train.py $S/g2v11_x8a_latents.py $S/g2v9_gen.py $S/make_s0spec_x.py $TGT $ZA_TR $ZA_DEV $ZE_TR $ZE_DEV /home/shoon/checkpoints/m1_final_frozen.pt
$PY $S/g2v9_xattn.py --selftest || { echo "[FATAL] selftest"; exit 1; }
gpus="$U5 $U6 $U7"; [ "$MODE" = "smoke" ] && gpus="$U5"
for u in $gpus; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
if [ "$MODE" = "smoke" ]; then
  CUDA_VISIBLE_DEVICES=$U5 $PY $S/g2v11_x8a_train.py --targets $TGT --seed 2047 --max_steps 5 --batch_size 2 --max_items 16 --out_bridge $W/bridge_S.pt --out_deq $W/deq_S.pt --manifest $W/manifest_S.json > $W/train_S.log 2>&1
  rc=$?; echo "[TR-smoke] rc=$rc $(date '+%T')"; grep -h 'DATA\]\|DEQ\]\|BRIDGE\]\|Step\|SAVE\|Error\|error' $W/train_S.log | cut -c1-220 | tail -n 14
  [ "$rc" -eq 0 ] || exit 1
  $PY - <<'PYEOF'
import json; j=json.load(open("configs/s7_train_993.json"))[:8]; json.dump(j, open("/tmp/cts_x8a_smoke/items8.json","w"))
PYEOF
  CUDA_VISIBLE_DEVICES=$U5 $PY $S/g2v11_x8a_latents.py --items $W/items8.json --zstar $ZA_TR --zsrc $ZE_TR --deq_state $W/deq_S.pt --out $W/lat_S.pt > $W/lat_S.log 2>&1
  rc=$?; echo "[LAT-smoke] rc=$rc $(date '+%T') $(grep -h 'SAVE' $W/lat_S.log | cut -c1-160)"; [ "$rc" -eq 0 ] || { tail -n 8 $W/lat_S.log; exit 1; }
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n 6p
  echo "=== X8-A smoke end $(date '+%F %T') ==="; exit 0
fi
run() { # tag seed gpu
  local T=$1 seed=$2 gpu=$3
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v11_x8a_train.py --targets $TGT --seed $seed --max_steps 1500 --out_bridge $W/bridge_$T.pt --out_deq $W/deq_$T.pt --manifest $W/manifest_$T.json > $W/train_$T.log 2>&1
  local rc=$?; echo "[TR-$T] rc=$rc $(date '+%T') $(grep -h 'Step' $W/train_$T.log | tail -n 1 | cut -c1-140)"; grep -h 'DATA\]' $W/train_$T.log | cut -c1-230
  [ "$rc" -eq 0 ] && [ -s $W/bridge_$T.pt ] && [ -s $W/deq_$T.pt ] || { echo "[FATAL-$T] training rc=$rc"; echo 1 > $W/rc_$T; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v11_x8a_latents.py --items results_s7/p6_item_order.json --zstar $ZA_DEV --zsrc $ZE_DEV --deq_state $W/deq_$T.pt --out $W/latdev_$T.pt > $W/lat_$T.log 2>&1; rc=$?
  echo "[LAT-$T] rc=$rc $(date '+%T') $(grep -h 'SAVE' $W/lat_$T.log | cut -c1-160)"
  [ "$rc" -eq 0 ] && [ -s $W/latdev_$T.pt ] || { echo 1 > $W/rc_$T; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v9_gen.py --ckpt $W/bridge_$T.pt --zfile $W/latdev_$T.pt --cond matched --out $O/gen_$T.jsonl > $W/gen_$T.log 2>&1; rc=$?
  echo "[GEN-$T] rc=$rc rows=$(wc -l < $O/gen_$T.jsonl) $(date '+%T')"; [ "$rc" -eq 0 ] || { echo 1 > $W/rc_$T; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v9_gen.py --ckpt $W/bridge_$T.pt --zfile $W/latdev_$T.pt --cond shuffled --out $O/genB1_$T.jsonl > $W/genB1_$T.log 2>&1; rc=$?
  echo "[GENS-$T] rc=$rc rows=$(wc -l < $O/genB1_$T.jsonl) $(date '+%T')"; echo $rc > $W/rc_$T
}
run X8A47 2047 $U5 &
run X8A49 2049 $U6 &
run X8A50 2050 $U7 &
wait
ok=1
for T in X8A47 X8A49 X8A50; do
  rc=$(cat $W/rc_$T 2>/dev/null || echo 9)
  for f in gen genB1; do
    n=$(wc -l < $O/${f}_$T.jsonl 2>/dev/null || echo 0); u=$($PY -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}))" $O/${f}_$T.jsonl 2>/dev/null || echo 0)
    echo "[CHECK-$f-$T] rc=$rc rows=$n unique_ids=$u"; [ "$rc" -eq 0 ] && [ "$n" -eq 600 ] && [ "$u" -eq 600 ] || ok=0
  done
done
[ "$ok" -eq 1 ] || { echo "[FATAL] one or more runs incomplete; no DONE flag"; exit 1; }
$PY $S/make_s0spec_x.py $W/s0spec.json $O X8A47,X8A49,X8A50
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py $W/s0spec.json $W/x8a_items.json" > $W/grade.log 2>&1; rc=$?; grep "S0" $W/grade.log
[ "$rc" -eq 0 ] && [ "$(grep -c '^\[S0\] ' $W/grade.log)" -eq 7 ] && [ "$(grep -c 'n 600 ' $W/grade.log)" -eq 7 ] || { echo "[FATAL] grading rc=$rc"; exit 1; }
sha256sum $O/*.jsonl $W/x8a_items.json $W/manifest_*.json $W/bridge_*.pt $W/deq_*.pt $W/latdev_*.pt
touch $W/X8A_DONE.flag
echo "=== X8-A end $(date '+%F %T') ==="

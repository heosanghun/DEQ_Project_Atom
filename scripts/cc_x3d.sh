#!/usr/bin/env bash
# scripts/cc_x3d.sh - X3-min rev3 (b) rerun: X3Od47 oracle text_drop 0.0 + latent_drop 0.3 on GPU 7 (CC 2026-10-20 06:0x). (b) crashed in cc_x3c.sh (no grad path when latent dropped); trainer fixed (loss.requires_grad guard). Rules unchanged: X_INTERFACE_PLAN rev3.
# (a) X3Oc47 oracle text_drop 0.0 ; (b) X3Od47 oracle text_drop 0.0 + latent_drop 0.3 ; D2: rev2 O47 bridge on 200 TRAIN items (diagnostic).
# Starts after rev2 Stage AE jobs on GPUs 6/7 finish (GENS-X3E49 and GENS-X3A50 lines in /tmp/cts_x3b_master.log). GPU 5 left to rev2 (E50).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/tmp/cts_x_stage:/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_x_stage; W=/tmp/cts_x3d; O=results_s7/p6r/g2v9x3d; ORC=/tmp/cts_s2/oracle_latents.pt
B2=/tmp/cts_x3b/bridge_X3O47.pt; T200=configs/x3_train200.json; T200SHA=3206c7bdeccd9d300ec963256bd5e7432f68e9ce8cc76bd32860f96f67eaafc2
U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
echo "=== X3d start $(date '+%F %T') ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v9_xattn.py $S/g2v9_train.py $S/g2v9_gen.py $ORC $B2 $T200
[ "$(sha256sum $T200 | cut -d' ' -f1)" = "$T200SHA" ] || { echo "[FATAL] train200 sha"; exit 1; }
$PY $S/g2v9_xattn.py --selftest || { echo "[FATAL] selftest"; exit 1; }
df -h /home /tmp | cat
for u in $U7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
tr() { # tag zfile_train seed gpu steps extra...
  local tag=$1 zf=$2 seed=$3 gpu=$4 steps=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/g2v9_train.py --zfile $zf --seed $seed --max_steps $steps --out $W/bridge_$tag.pt --manifest $W/manifest_$tag.json "$@" > $W/train_$tag.log 2>&1
  echo "[TR-$tag] rc=$? $(date '+%T') $(grep -h 'Step' $W/train_$tag.log | tail -n 1 | cut -c1-80)"
}
gn() { # tag ckpt zfile_dev gpu
  CUDA_VISIBLE_DEVICES=$4 $PY $S/g2v9_gen.py --ckpt $2 --zfile $3 --cond matched --out $O/gen_$1.jsonl > $W/gen_$1.log 2>&1
  echo "[GEN-$1] rc=$? rows=$(wc -l < $O/gen_$1.jsonl) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$4 $PY $S/g2v9_gen.py --ckpt $2 --zfile $3 --cond shuffled --out $O/genB1_$1.jsonl > $W/genB1_$1.log 2>&1
  echo "[GENS-$1] rc=$? rows=$(wc -l < $O/genB1_$1.jsonl) $(date '+%T')"
}
d2() { # D2 diagnostic: rev2 O47 bridge on train200, gpu
  CUDA_VISIBLE_DEVICES=$1 $PY $S/g2v9_gen.py --ckpt $B2 --zfile $ORC --cond matched --items $T200 --order_sha $T200SHA --out $O/gen_D2m.jsonl > $W/gen_D2m.log 2>&1
  echo "[GEN-D2m] rc=$? rows=$(wc -l < $O/gen_D2m.jsonl) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$1 $PY $S/g2v9_gen.py --ckpt $B2 --zfile $ORC --cond shuffled --items $T200 --order_sha $T200SHA --out $O/genB1_D2m.jsonl > $W/genB1_D2m.log 2>&1
  echo "[GENS-D2m] rc=$? rows=$(wc -l < $O/genB1_D2m.jsonl) $(date '+%T')"
}
( tr X3Od47 $ORC 2047 $U7 1500 --text_drop 0.0 --latent_drop 0.3; gn X3Od47 $W/bridge_X3Od47.pt $ORC $U7 ) &
wait
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py $S/s0spec_x3d.json $W/x3d_items.json" > $W/grade_x3d.log 2>&1; grep "S0" $W/grade_x3d.log
sha256sum $O/*.jsonl $W/*_items.json $W/manifest_*.json $W/bridge_*.pt
touch $W/X3D_DONE.flag
echo "=== X3d end $(date '+%F %T') ==="

#!/usr/bin/env bash
# scripts/cc_x3b.sh - X3-min rev2 (bridge O default init, gate 0). rev1 cc_x3.sh had dead gates (O=0 and gate=0) and was stopped 10/19 17:4x; rev1 outputs kept at /tmp/cts_x3. Plan: shared/proposals/X_INTERFACE_PLAN_20261019.md (X3 rules).
# Waits for X1 to finish (GPUs 5-7). Stage O: oracle sanity (seed 2047). Stage AE: A (canonical z*' M2') vs E (z_src), seeds 3, matched + shuffled.
# Train 993 + dev 600 only. Exploratory.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/tmp/cts_x_stage:/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_x_stage; W=/tmp/cts_x3b; O=results_s7/p6r/g2v9x3b; ORC=/tmp/cts_s2/oracle_latents.pt
ZA_TR=results_s7/g2v2/zstar_zsrcinj_train1000.pt; ZA_DEV=results_s7/g2v2/zstar_zsrcinj_dev600.pt
ZE_TR=results_s7/zsrc_train1000_P4.pt; ZE_DEV=results_s7/zsrc_heldout600.pt
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
until [ -e /tmp/cts_x1/X1_DONE.flag ]; do sleep 120; done
echo "=== X3b start $(date '+%F %T') ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v9_xattn.py $S/g2v9_train.py $S/g2v9_gen.py $ZA_TR $ZA_DEV $ZE_TR $ZE_DEV $ORC
$PY $S/g2v9_xattn.py --selftest || { echo "[FATAL] selftest"; exit 1; }
df -h /home /tmp | cat
for u in $U5 $U6 $U7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
tr() { # tag zfile_train seed gpu steps
  CUDA_VISIBLE_DEVICES=$4 $PY $S/g2v9_train.py --zfile $2 --seed $3 --max_steps $5 --out $W/bridge_$1.pt --manifest $W/manifest_$1.json > $W/train_$1.log 2>&1
  echo "[TR-$1] rc=$? $(date '+%T') $(grep -h 'Step' $W/train_$1.log | tail -n 1 | cut -c1-80)"
}
gn() { # tag zfile_dev gpu
  CUDA_VISIBLE_DEVICES=$3 $PY $S/g2v9_gen.py --ckpt $W/bridge_$1.pt --zfile $2 --cond matched --out $O/gen_$1.jsonl > $W/gen_$1.log 2>&1
  echo "[GEN-$1] rc=$? rows=$(wc -l < $O/gen_$1.jsonl) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$3 $PY $S/g2v9_gen.py --ckpt $W/bridge_$1.pt --zfile $2 --cond shuffled --out $O/genB1_$1.jsonl > $W/genB1_$1.log 2>&1
  echo "[GENS-$1] rc=$? rows=$(wc -l < $O/genB1_$1.jsonl) $(date '+%T')"
}
# Stage O (oracle sanity, 1 seed, 1500 steps) on GPU 5; in parallel start A/E seed 2047 on GPUs 6/7
( tr X3O47 $ORC 2047 $U5 1500; gn X3O47 $ORC $U5 ) &
( tr X3A47 $ZA_TR 2047 $U6 3000; gn X3A47 $ZA_DEV $U6 ) &
( tr X3E47 $ZE_TR 2047 $U7 3000; gn X3E47 $ZE_DEV $U7 ) &
wait
$PY $S/make_s0spec_x.py $W/s0spec_1.json $O X3O47,X3A47,X3E47
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py $W/s0spec_1.json $W/x3_items_1.json" > $W/grade_1.log 2>&1; grep "\[S0\]" $W/grade_1.log
touch $W/X3B_STAGE1_DONE.flag
# Stage AE seeds 2049, 2050
( tr X3A49 $ZA_TR 2049 $U5 3000; gn X3A49 $ZA_DEV $U5; tr X3E50 $ZE_TR 2050 $U5 3000; gn X3E50 $ZE_DEV $U5 ) &
( tr X3E49 $ZE_TR 2049 $U6 3000; gn X3E49 $ZE_DEV $U6 ) &
( tr X3A50 $ZA_TR 2050 $U7 3000; gn X3A50 $ZA_DEV $U7 ) &
wait
$PY $S/make_s0spec_x.py $W/s0spec_2.json $O X3O47,X3A47,X3E47,X3A49,X3E49,X3A50,X3E50
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py $W/s0spec_2.json $W/x3_items.json" > $W/grade_2.log 2>&1; grep "\[S0\]" $W/grade_2.log
sha256sum $O/*.jsonl $W/x3_items.json $W/manifest_*.json $W/bridge_*.pt
touch $W/X3B_DONE.flag
echo "=== X3b end $(date '+%F %T') ==="

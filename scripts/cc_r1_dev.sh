#!/usr/bin/env bash
# scripts/cc_r1_dev.sh - R1 dev600 accuracy (CC 2026-10-12). Waits for /tmp/cts_r1b/R1_TRAIN_DONE.flag. Not a gate (dev items only).
# Per new model: decoder copy -> results_s7/g2v2/decoder_<tag>_v36.pt (path guard), one-step dev latents -> /tmp/cts_r1b, v36f generation -> results_s7/p6r/g2v4r1/.
# Then scorer pairs (A vs E at equal lambda, and lambda 0.1 vs 0 within mode), per seed. Ah/Eh reuse results_s7/p6r/g2v4dev/dev_gen_{A,E1}.jsonl.
set -uo pipefail
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
W=/tmp/cts_r1b; O=results_s7/p6r/g2v4r1; ORD=6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
until [ -e $W/R1_TRAIN_DONE.flag ]; do sleep 120; done
echo "=== R1 dev start $(date '+%F %T') ==="
[ -e $O ] && { echo "[FATAL] $O exists"; exit 1; }
mkdir -p $O
sha256sum scripts/s7_eval_gate_p6_v36f.py scripts/g2v3_m3_eval_latents_v2.py scripts/s7_eval_score_p6_v36.py scripts/s5_eval_grader_v2.py
df -h /home /tmp | cat
ev() { # tag mode gpu
  local t=$1 m=$2 u=$3
  [ -s $W/decoder_$t.pt ] || { echo "[SKIP] $t no decoder"; return 0; }
  cp -n $W/decoder_$t.pt results_s7/g2v2/decoder_${t}_v36.pt
  if [ $m = A ]; then
    CUDA_VISIBLE_DEVICES=$u $PY scripts/g2v3_m3_eval_latents_v2.py --mode A --device cuda:0 --items results_s7/p6_item_order.json --deq_ckpt $W/deq_$t.pt \
      --zstar results_s7/g2v2/zstar_zsrcinj_dev600.pt --zsrc results_s7/zsrc_heldout600.pt --out $W/lat_$t.pt > $W/lat_$t.log 2>&1 || { echo "[FAIL] lat $t"; return 0; }
    CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36f.py --cond A --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_${t}_v36.pt --zstar $W/lat_$t.pt \
      --order_sha $ORD --out $O/gen_$t.jsonl > $W/gen_$t.log 2>&1 || { echo "[FAIL] gen $t"; return 0; }
  else
    CUDA_VISIBLE_DEVICES=$u $PY scripts/g2v3_m3_eval_latents_v2.py --mode E --device cuda:0 --items results_s7/p6_item_order.json --deq_ckpt $W/deq_$t.pt \
      --zsrc results_s7/zsrc_heldout600.pt --out $W/lat_$t.pt > $W/lat_$t.log 2>&1 || { echo "[FAIL] lat $t"; return 0; }
    CUDA_VISIBLE_DEVICES=$u $PY scripts/s7_eval_gate_p6_v36f.py --cond E1 --device cuda:0 --decoder_ckpt results_s7/g2v2/decoder_${t}_v36.pt --zsrc $W/lat_$t.pt \
      --order_sha $ORD --out $O/gen_$t.jsonl > $W/gen_$t.log 2>&1 || { echo "[FAIL] gen $t"; return 0; }
  fi
  grep -q "applied 436 DEQ tensors, unexpected 0" $W/lat_$t.log && grep -q "\[PROJ-V2\]" $W/gen_$t.log && echo "[EV-OK] $t rows=$(wc -l < $O/gen_$t.jsonl) $(date '+%T')" || echo "[EV-CHECK-FAIL] $t"
}
( ev R1A0s47 A $U5; ev R1A0s49 A $U5; ev R1A0s50 A $U5; ev R1E1s47 E $U5 ) &
( ev R1E1s49 E $U6; ev R1E1s50 E $U6; ev R1A1s49 A $U6 ) &
( ev R1A1s50 A $U7; ev R1E0s49 E $U7; ev R1E0s50 E $U7 ) &
wait
G=$O; D=results_s7/p6r/g2v4dev
f() { case $1 in Ah) echo $D/dev_gen_A.jsonl;; Eh) echo $D/dev_gen_E1.jsonl;; *) echo $G/gen_$1.jsonl;; esac; }
sc() { # name first second   -> gate_4 = first - second
  nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/s7_eval_score_p6_v36.py --gen_A $(f $2) --gen_E1 $(f $3) --gen_C results_s7/p6r/p6_gen_C.jsonl --gen_B1 results_s7/p6r/p6_gen_B1.jsonl --gen_dir $O --order results_s7/p6_item_order.json --out_json $O/score_$1.json --out_md $O/score_$1.md" > $O/score_$1.log 2>&1
  $PY -c "import json;d=json.load(open('$O/score_$1.json'));s=d['gates']['Gate_4']['strict'];print('[PAIR] $1 ($2 - $3): acc1 %.4f acc2 %.4f delta %+.4f ci %s b/c %d/%d p %.4f'%(s['acc_c1'],s['acc_c2'],s['delta'],s['ci_95'],s['mcnemar_b'],s['mcnemar_c'],s['mcnemar_one_sided_p']))"
}
sc AE_l0_s47 R1A0s47 Eh; sc AE_l0_s49 R1A0s49 R1E0s49; sc AE_l0_s50 R1A0s50 R1E0s50
sc AE_l1_s47 Ah R1E1s47; sc AE_l1_s49 R1A1s49 R1E1s49; sc AE_l1_s50 R1A1s50 R1E1s50
sc AA_s47 Ah R1A0s47; sc AA_s49 R1A1s49 R1A0s49; sc AA_s50 R1A1s50 R1A0s50
sc EE_s47 R1E1s47 Eh; sc EE_s49 R1E1s49 R1E0s49; sc EE_s50 R1E1s50 R1E0s50
sha256sum $O/*.json $O/*.jsonl
touch $W/R1_DEV_DONE.flag
echo "=== R1 dev end $(date '+%F %T') ==="

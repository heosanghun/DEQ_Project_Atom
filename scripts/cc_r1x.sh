#!/usr/bin/env bash
# scripts/cc_r1x.sh - R1x (CC 2026-10-12, CTO CTS-432): same-decoder input swaps. Waits for /tmp/cts_r1b/R1_DEV_DONE.flag. Dev600 only. Not a gate.
# (a) NLL for all 12 decoders: matched / shuffled / other start (z*' vs z_src) / mean / zero / no-injection (g2v5_diag.py)
# (b) accuracy: shuffled z* (cond B shift 1) for the 6 A decoders, then matched-vs-shuffled McNemar per decoder.
set -uo pipefail
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_r1_stage; W=/tmp/cts_r1b; X=/tmp/cts_r1x; O=results_s7/p6r/g2v4r1x; ORD=6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c; U7=GPU-0eb58a21-c10d-3ad0-14c1-9e8b8b1ebf5b
until [ -e $W/R1_DEV_DONE.flag ]; do sleep 120; done
echo "=== R1x start $(date '+%F %T') ==="
for f in $X $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $X; mkdir -p $O
sha256sum $S/g2v5_diag.py scripts/s7_eval_gate_p6_v36f.py
R=results_s7/g2v4m7rep
deq() { case $1 in Ah|Eh) echo $R/deq_$1.pt;; *) echo $W/deq_$1.pt;; esac; }
dec() { case $1 in Ah|Eh) echo $R/decoder_$1.pt;; *) echo $W/decoder_$1.pt;; esac; }
dg() { CUDA_VISIBLE_DEVICES=$3 $PY $S/g2v5_diag.py --mode $2 --n 600 --ks 1 --notext --deq_ckpt $(deq $1) --decoder_ckpt $(dec $1) --out $X/diag5_$1.json > $X/diag5_$1.log 2>&1; echo "[DG5-$1] rc=$? $(date '+%T')"; }
# (b) shuffled generation for A decoders
gdec() { case $1 in Ah) echo results_s7/g2v2/decoder_M7A_v36.pt;; *) echo results_s7/g2v2/decoder_${1}_v36.pt;; esac; }
glat() { case $1 in Ah) echo $R/dev600_latent_A.pt;; *) echo $W/lat_$1.pt;; esac; }
gb() { CUDA_VISIBLE_DEVICES=$2 $PY scripts/s7_eval_gate_p6_v36f.py --cond B --shift 1 --device cuda:0 --decoder_ckpt $(gdec $1) --zstar $(glat $1) --order_sha $ORD --out $O/genB1_$1.jsonl > $X/genB1_$1.log 2>&1
  grep -q "\[PROJ-V2\]" $X/genB1_$1.log && echo "[GB-OK] $1 rows=$(wc -l < $O/genB1_$1.jsonl) $(date '+%T')" || echo "[GB-FAIL] $1"; }
( dg Ah A $U5; dg R1A0s47 A $U5; dg R1A0s49 A $U5; dg R1A0s50 A $U5; gb Ah $U5; gb R1A0s47 $U5 ) &
( dg R1A1s49 A $U6; dg R1A1s50 A $U6; dg Eh E $U6; dg R1E0s49 E $U6; gb R1A0s49 $U6; gb R1A0s50 $U6 ) &
( dg R1E0s50 E $U7; dg R1E1s47 E $U7; dg R1E1s49 E $U7; dg R1E1s50 E $U7; gb R1A1s49 $U7; gb R1A1s50 $U7 ) &
wait
mgen() { case $1 in Ah) echo results_s7/p6r/g2v4dev/dev_gen_A.jsonl;; *) echo results_s7/p6r/g2v4r1/gen_$1.jsonl;; esac; }
for t in Ah R1A0s47 R1A0s49 R1A0s50 R1A1s49 R1A1s50; do
  nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/s7_eval_score_p6_v36.py --gen_A $(mgen $t) --gen_E1 $O/genB1_$t.jsonl --gen_C results_s7/p6r/p6_gen_C.jsonl --gen_B1 $O/genB1_$t.jsonl --gen_dir $O --order results_s7/p6_item_order.json --out_json $O/score_MS_$t.json --out_md $O/score_MS_$t.md" > $O/score_MS_$t.log 2>&1
  $PY -c "import json;d=json.load(open('$O/score_MS_$t.json'));s=d['gates']['Gate_4']['strict'];l=d['gates']['Gate_4'].get('lenient',{});print('[MS] $t matched-shuffled strict %+.4f ci %s b/c %d/%d p %.4f | lenient %s'%(s['delta'],s['ci_95'],s['mcnemar_b'],s['mcnemar_c'],s['mcnemar_one_sided_p'],l.get('delta')))"
done
sha256sum $X/diag5_*.json $O/*.json
touch $X/R1X_DONE.flag
echo "=== R1x end $(date '+%F %T') ==="

#!/usr/bin/env bash
# scripts/cc_x5base.sh - C-fmt (pure Gemma + format instruction, NO bridge) generations on TRAIN 993 (CC 2026-10-20 06:3x).
# Purpose: (1) baseline for D2 (train-item accuracy of pure model), (2) style-matched targets for X5 readout redesign (plan: X_INTERFACE_PLAN rev3/X5).
# No training. Uses g2v9_gen.py --cond none (bridge not attached). GPUs 5 and 6, two shards. Train 993 only; no dev/test.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shoon/cts_e1_s5
export PYTHONPATH=/tmp/cts_x_stage:/home/shoon/cts_e1_s5/scripts:${PYTHONPATH:-}
PY=/home/shoon/system1_5/.venv/bin/python
S=/tmp/cts_x_stage; W=/tmp/cts_x5base; O=results_s7/p6r/g2v9x5base; ORC=/tmp/cts_s2/oracle_latents.pt; CK=/tmp/cts_x3b/bridge_X3O47.pt
U5=GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a; U6=GPU-f99bbac4-fb7d-cc1f-3edd-3495951ff44c
echo "=== X5base start $(date '+%F %T') ==="
for f in $W $O; do [ -e $f ] && { echo "[FATAL] exists $f"; exit 1; }; done
mkdir -m 700 $W; mkdir -p $O
sha256sum $S/g2v9_gen.py configs/x5_train993_shard0.json configs/x5_train993_shard1.json
[ "$(sha256sum configs/x5_train993_shard0.json | cut -d' ' -f1)" = "53fb3d719fe09e5cb08a3e2db1cd70c22587b9234de62141c51c4b16d1a1a826" ] || { echo "[FATAL] shard0 sha"; exit 1; }
[ "$(sha256sum configs/x5_train993_shard1.json | cut -d' ' -f1)" = "9632732b586f13386ac2ff8dd1af5e372095c891f6fdbca5133701ad066c1f92" ] || { echo "[FATAL] shard1 sha"; exit 1; }
for u in $U5 $U6; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $u | tr -d ' ')
  a=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $u | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=$m apps=$a"; [ "$m" -lt 1024 ] && [ "$a" -eq 0 ] || { echo "[FATAL] busy $u"; exit 1; }
done
( CUDA_VISIBLE_DEVICES=$U5 $PY $S/g2v9_gen.py --ckpt $CK --zfile $ORC --cond none --items configs/x5_train993_shard0.json --order_sha 53fb3d719fe09e5cb08a3e2db1cd70c22587b9234de62141c51c4b16d1a1a826 --out $O/gen_Ctrain_s0.jsonl > $W/gen_s0.log 2>&1; echo "[GEN-s0] rc=$? rows=$(wc -l < $O/gen_Ctrain_s0.jsonl) $(date '+%T')" ) &
( CUDA_VISIBLE_DEVICES=$U6 $PY $S/g2v9_gen.py --ckpt $CK --zfile $ORC --cond none --items configs/x5_train993_shard1.json --order_sha 9632732b586f13386ac2ff8dd1af5e372095c891f6fdbca5133701ad066c1f92 --out $O/gen_Ctrain_s1.jsonl > $W/gen_s1.log 2>&1; echo "[GEN-s1] rc=$? rows=$(wc -l < $O/gen_Ctrain_s1.jsonl) $(date '+%T')" ) &
wait
cat $O/gen_Ctrain_s0.jsonl $O/gen_Ctrain_s1.jsonl > $O/gen_Ctrain.jsonl
$PY - <<'PYEOF'
import json
spec={"Ctrain":["results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl","configs/s7_train_993.json"],"Ctrain200":["results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl","configs/x3_train200.json"]}
json.dump(spec,open("/tmp/cts_x5base/s0spec.json","w"))
PYEOF
nice -n 19 bash -c "ulimit -v 62914560; $PY scripts/g2v5_s0_grade.py $W/s0spec.json $W/ctrain_items.json" > $W/grade.log 2>&1; grep "S0" $W/grade.log
sha256sum $O/gen_Ctrain.jsonl $W/ctrain_items.json
touch $W/X5BASE_DONE.flag
echo "=== X5base end $(date '+%F %T') ==="

#!/usr/bin/env bash
# scripts/run_task0070_diag_v2.sh - TASK-0070 diagnostics D1/D2/D4 (v2, CC 2026-10-08)
# v1 (run_task0070_diag.sh) aborted: whole-node compute-app query counted GPU 2's "[N/A]" row (GPU 2 hardware ERR!).
# v2 changes: GPUs addressed by UUID only; per-GPU pre-check by UUID; GPU 2 and GPU 4 never used.
#   GPU 0: A (reuse diag_nll_smoke_A.jsonl if 600 rows) -> C -> E1   (direct comparison pairs on one device)
#   GPU 3: B1 -> D1 -> F
#   GPU 1: D4 trajectories (anchor check inside diag_d4_traj.py)
#   CPU  : D2 geometry
set -uo pipefail

cd /home/shoon/cts_e1_s5
PY="/home/shoon/system1_5/.venv/bin/python"
GEN="scripts/s7_eval_gate_p6_v36d.py"
ORDER_SHA="6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c"
GPU0="GPU-71be98b0-6b81-9257-c622-97dc97d46480"
GPU1="GPU-9410691f-6857-1642-ec7c-7596970e2f48"
GPU3="GPU-a9fed959-7113-e747-dac4-a515ac50c160"
mkdir -p raw_s7 results_s7/p6r results_s7/diag
LOG=raw_s7/diag_v2_master.log
exec > >(tee -a "$LOG") 2>&1

echo "=== TASK-0070 diag v2 start $(date '+%Y-%m-%d %H:%M:%S') host $(hostname) ==="
[ "$(sha256sum $GEN | cut -d' ' -f1)" = "b2df5112727597e028f842afb9c8fb42083d211bba5f137a7e125de9b0165ecd" ] || { echo "[FATAL] v36d SHA mismatch"; exit 1; }

check_gpu() {
  local u="$1" mem apps
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$u" | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$u" | tr -d ' ' | grep -c '^[0-9]' || true)
  echo "[CHECK] $u mem=${mem}MiB apps=${apps}"
  [ "$mem" -lt 1024 ] && [ "$apps" -eq 0 ]
}
for u in $GPU0 $GPU1 $GPU3; do check_gpu "$u" || { echo "[FATAL] GPU busy: $u"; exit 1; }; done

nll() {  # $1 gpu uuid, $2 out name, rest: generator args
  local u="$1" name="$2"; shift 2
  local out="results_s7/p6r/diag_nll_${name}.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -eq 600 ]; then echo "[SKIP] $name already 600 rows"; return 0; fi
  echo "[RUN] $name on $u $(date '+%H:%M:%S')"
  CUDA_VISIBLE_DEVICES="$u" "$PY" "$GEN" "$@" --max_new_tokens 1 --nll --order_sha "$ORDER_SHA" --out "$out" > "raw_s7/diag_nll_${name}.log" 2>&1
  local rc=$?
  echo "[DONE] $name rc=$rc rows=$(wc -l < "$out" 2>/dev/null) $(date '+%H:%M:%S')"
  return $rc
}

# A: reuse the earlier full run if complete (diag_nll_smoke_A.jsonl was a 600-row run on index 0 in v1)
if [ ! -f results_s7/p6r/diag_nll_A.jsonl ] && [ -f results_s7/p6r/diag_nll_smoke_A.jsonl ] && [ "$(wc -l < results_s7/p6r/diag_nll_smoke_A.jsonl)" -eq 600 ]; then
  echo "[NOTE] diag_nll_smoke_A.jsonl has 600 rows but ran with CUDA_VISIBLE_DEVICES=0 (index). Rerunning A by UUID for device certainty."
fi

nice -n 19 bash -c "ulimit -v 62914560 && $PY scripts/diag_d2_geometry.py --out results_s7/diag/d2_geometry.json" > raw_s7/diag_d2_geometry.log 2>&1 &
P_D2=$!
CUDA_VISIBLE_DEVICES="$GPU1" "$PY" scripts/diag_d4_traj.py --device cuda:0 --items configs/diag_d4_train40.json --out results_s7/diag/d4_traj_train40.pt > raw_s7/diag_d4_traj.log 2>&1 &
P_D4=$!
( nll "$GPU3" B1 --cond B --shift 1; nll "$GPU3" D1 --cond D1; nll "$GPU3" F --cond F ) &
P_G3=$!
nll "$GPU0" A --cond A; nll "$GPU0" C --cond C; nll "$GPU0" E1 --cond E1

wait $P_G3; echo "[WAIT] GPU3 chain rc=$?"
wait $P_D2; echo "[WAIT] D2 rc=$?"
wait $P_D4; echo "[WAIT] D4 rc=$?"

"$PY" scripts/diag_d1_stats.py > raw_s7/diag_d1_stats.log 2>&1; echo "[D1-STATS] rc=$?"
ls -l results_s7/p6r/diag_nll_*.jsonl results_s7/diag/ 2>&1
sha256sum results_s7/p6r/diag_nll_*.jsonl results_s7/diag/* 2>&1
touch raw_s7/TASK0070_V2_DONE.flag
echo "=== TASK-0070 diag v2 end $(date '+%Y-%m-%d %H:%M:%S') ==="

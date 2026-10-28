#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_FILE="/home/shoon/cts_e1/harness/vram_profiling.log"

# Preflight check
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID | tr -d ' ')
PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU_ID | wc -l)

if [ "$PROCS" -ne 0 ]; then
    echo "[PREFLIGHT-FAIL] GPU $GPU_ID active procs=$PROCS != 0 at $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_FILE"
    exit 1
fi

if [ "$FREE_MB" -lt 40000 ]; then
    echo "[PREFLIGHT-FAIL] GPU $GPU_ID free=${FREE_MB}MB < 40000MB at $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_FILE"
    exit 1
fi

/home/shoon/system1_5/.venv/bin/python -u /home/shoon/cts_e1/harness/run_vram_profiling.py >> "$LOG_FILE" 2>&1
RC=$?

TS=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
echo "[EXIT] rc=$RC $TS" >> "$LOG_FILE"
exit $RC

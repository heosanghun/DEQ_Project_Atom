#!/usr/bin/env bash
# exe_scratch/s5/phase2_monitor.sh - Read-only Server Monitor for Phase 2 Pipeline
# Directive: EXE-260925-31 Task 1 (System 2 / Ledger #339)

BASE_DIR="/home/shoon/cts_e1_s5"
LOG_DIR="${BASE_DIR}/raw_s5"
MONITOR_LOG="${LOG_DIR}/phase2_monitor.log"

mkdir -p "${LOG_DIR}"

# Find initial pipeline PID (wait up to 15 seconds if just launched)
PIPE_PID=""
for _ in {1..15}; do
    PIPE_PID=$(pgrep -f "scripts/run_phase2" | head -n 1)
    if [ -z "$PIPE_PID" ]; then
        PIPE_PID=$(pgrep -f "s5_train_decoder_v2|s5_eval_gate_v2" | head -n 1)
    fi
    if [ -n "$PIPE_PID" ]; then
        break
    fi
    sleep 1
done

# Target log file detection
get_target_log() {
    if [ -f "${LOG_DIR}/phase2_pipeline_retry1.log" ]; then
        echo "${LOG_DIR}/phase2_pipeline_retry1.log"
    else
        echo "${LOG_DIR}/phase2_pipeline.log"
    fi
}

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    TARGET_LOG=$(get_target_log)

    # 1. PID Liveness
    if [ -n "$PIPE_PID" ] && kill -0 "$PIPE_PID" 2>/dev/null; then
        PID_ALIVE="Y"
    else
        CUR_PID=$(pgrep -f "scripts/run_phase2|s5_train_decoder_v2|s5_eval_gate_v2" | head -n 1)
        if [ -n "$CUR_PID" ] && kill -0 "$CUR_PID" 2>/dev/null; then
            PIPE_PID="$CUR_PID"
            PID_ALIVE="Y"
        else
            PID_ALIVE="N"
        fi
    fi

    GPU0_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ' || echo "N/A")
    GPU4_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | tr -d ' ' || echo "N/A")

    # If PID is dead, record final line and exit
    if [ "$PID_ALIVE" = "N" ]; then
        echo "${TIMESTAMP} | N | EXIT 감지 | - | - | - | ${GPU0_MIB} MiB | ${GPU4_MIB} MiB" >> "${MONITOR_LOG}"
        echo "--- 로그 마지막 5행 (${TARGET_LOG}) ---" >> "${MONITOR_LOG}"
        if [ -f "$TARGET_LOG" ]; then
            tail -n 5 "$TARGET_LOG" >> "${MONITOR_LOG}"
        fi
        exit 0
    fi

    # 2. Last [STEP n/4] header
    LAST_HEADER="-"
    if [ -f "$TARGET_LOG" ]; then
        H=$(grep -E "^\[STEP [1-4]/4\]" "$TARGET_LOG" | tail -n 1)
        if [ -n "$H" ]; then
            LAST_HEADER="$H"
        fi
    fi

    # 3. Last "[Step xxxx/3000]" line
    LAST_STEP="-"
    if [ -f "$TARGET_LOG" ]; then
        S=$(grep -E "^\[Step [0-9]{4}/3000\]" "$TARGET_LOG" | tail -n 1)
        if [ -n "$S" ]; then
            LAST_STEP="$S"
        fi
    fi

    # 4. Minutes since last update
    ELAPSED_MIN=0
    if [ -f "$TARGET_LOG" ]; then
        NOW=$(date +%s)
        MOD_TIME=$(stat -c %Y "$TARGET_LOG" 2>/dev/null || echo "$NOW")
        ELAPSED_MIN=$(( (NOW - MOD_TIME) / 60 ))
    fi

    # 5. NaN/inf count
    NAN_COUNT=$(grep -ciE "nan|inf" "$TARGET_LOG" 2>/dev/null)
    if [ -z "$NAN_COUNT" ]; then
        NAN_COUNT=0
    fi

    # Format: 시각 | 파이프라인 PID 생존(Y/N) | 마지막 [STEP n/4] 헤더 | 마지막 "[Step xxxx/3000]" 행 원문(없으면 -) | 로그 마지막 갱신 후 경과 분 | NaN/inf 행 수(grep -ciE "nan|inf") | GPU0 MiB | GPU4 MiB
    LOG_LINE="${TIMESTAMP} | ${PID_ALIVE} | ${LAST_HEADER} | ${LAST_STEP} | ${ELAPSED_MIN} | ${NAN_COUNT} | ${GPU0_MIB} MiB | ${GPU4_MIB} MiB"
    echo "${LOG_LINE}" >> "${MONITOR_LOG}"

    # Sleep 600s
    sleep 600
done

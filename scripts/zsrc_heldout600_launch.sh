#!/usr/bin/env bash
# scripts/zsrc_heldout600_launch.sh - CTS S7 z_src Held-Out 600 Launcher
# Directive: TASK-0009 Section 3 / PREREG_S7_D_v3_7 §4·§7
# Master Sequence Step: 9 (P5_DONE.flag AND GPU 0 idle)

set -euo pipefail

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

SMOKE="${SMOKE:-0}"
export MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=4
export TORCH_NUM_THREADS=4

PYTHON_BIN="/home/shoon/system1_5/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python3"
fi

echo "========================================================================"
echo "CTS S7 z_src Held-Out 600 Launcher"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo "Mode: $(if [[ "${SMOKE}" == "1" ]]; then echo "SMOKE (CPU 2-item)"; else echo "FULL (Held-Out 600 on GPU 0)"; fi)"
echo "========================================================================"

# SMOKE Mode: CPU execution for 2 items
if [[ "${SMOKE}" == "1" ]]; then
    echo "[LAUNCH] Running in SMOKE mode on CPU (first 2 train items)..."
    mkdir -p scratch
    ulimit -v 62914560 || true
    exec nice -n 19 "${PYTHON_BIN}" scripts/s7_compute_zsrc.py \
        --smoke \
        --device cpu \
        --out scratch/zsrc_smoke.pt
fi

# FULL Mode: Held-out 600 items
echo "[PRE-CHECK 1/3] Checking P5 completion flag (raw_s7/P5_DONE.flag)..."
if [[ ! -f "raw_s7/P5_DONE.flag" ]]; then
    echo "[ERROR] raw_s7/P5_DONE.flag does not exist. P5 must finish before z_src computation (Master Step 9)." >&2
    exit 1
fi
echo "[PRE-CHECK 1/3] PASS: raw_s7/P5_DONE.flag found."

echo "[PRE-CHECK 2/3] Checking GPU 0 idle state..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ' || echo "99999")
    if [[ "${GPU0_MEM}" -gt 1024 ]]; then
        echo "[ERROR] GPU 0 is busy (${GPU0_MEM} MiB used > 1024 MiB limit). Aborting z_src computation." >&2
        exit 1
    fi
    echo "[PRE-CHECK 2/3] PASS: GPU 0 is idle (${GPU0_MEM} MiB used)."
else
    echo "[WARN] nvidia-smi not available, skipping GPU memory check."
fi

echo "[PRE-CHECK 3/3] Checking target output existence..."
TARGET_OUT="results_s7/zsrc_heldout600.pt"
if [[ -f "${TARGET_OUT}" ]]; then
    echo "[NOTICE] Target output ${TARGET_OUT} already exists ($(stat -c %s "${TARGET_OUT}") bytes)."
    echo "[NOTICE] If recomputation is desired, remove or rename it first."
    exit 0
fi
echo "[PRE-CHECK 3/3] PASS: Target output is clean."

# Governance Flags
mkdir -p raw_s7
FLAG_STARTED="raw_s7/ZSRC600_STARTED.flag"
FLAG_DONE="raw_s7/ZSRC600_DONE.flag"

if [[ -f "${FLAG_STARTED}" && ! -f "${FLAG_DONE}" ]]; then
    echo "[ERROR] ${FLAG_STARTED} exists and ${FLAG_DONE} is absent. Double-run prevented." >&2
    exit 1
fi

echo "START_TIME=\"$(date '+%Y-%m-%d %H:%M:%S')\" HOST=\"$(hostname)\" TARGET=\"${TARGET_OUT}\"" > "${FLAG_STARTED}"

echo "[LAUNCH] Executing z_src computation on GPU 0 for 600 heldout items..."
mkdir -p results_s7
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/s7_compute_zsrc.py \
    --heldout600 \
    --device cuda:0 \
    --out "${TARGET_OUT}"

RC=$?
if [[ ${RC} -eq 0 && -f "${TARGET_OUT}" ]]; then
    SHA=$(sha256sum "${TARGET_OUT}" | awk '{print $1}')
    echo "DONE_TIME=\"$(date '+%Y-%m-%d %H:%M:%S')\" HOST=\"$(hostname)\" TARGET=\"${TARGET_OUT}\" SHA256=\"${SHA}\"" > "${FLAG_DONE}"
    echo "[COMPLETE] z_src heldout 600 computation finished successfully. Flag written: ${FLAG_DONE}"
else
    echo "[FAIL] z_src computation failed with returncode ${RC}" >&2
    exit ${RC}
fi

#!/usr/bin/env bash
# scripts/run_verify_frozen_parity.sh
# Runner for verify_frozen_parity_cpu.py adhering strictly to Ledger #381 rules:
#   (a) torch.no_grad() and 0 computation graph retention enforced in python script.
#   (b) ulimit -v 62914560 (60 GB virtual memory limit) enforced.
#   (c) free -g logged every 5 minutes by background monitor.

set -euo pipefail

# 1. Enforce 60 GB virtual memory limit (60 * 1024 * 1024 KB = 62914560 KB)
ulimit -v 62914560
echo "[SAFETY] ulimit -v set to: $(ulimit -v) KB"

LOG_FILE="/home/shoon/cts_e1_s5/raw_s7/parity_verification_TASK0002.log"
MEM_LOG="/home/shoon/cts_e1_s5/raw_s7/parity_verification_memory_TASK0002.log"

echo "[SAFETY] Starting background memory monitor (every 5 minutes)..."
(
    while true; do
        echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] free -g ===" >> "$MEM_LOG"
        free -g >> "$MEM_LOG"
        sleep 300
    done
) &
MONITOR_PID=$!

trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

# Initial memory log
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] INITIAL free -g ===" | tee -a "$MEM_LOG"
free -g | tee -a "$MEM_LOG"

# 2. Run Python script with unbuffered output under nice -n 19
echo "[SAFETY] Launching verify_frozen_parity_cpu.py with unbuffered python and nice -n 19..."
nice -n 19 /home/shoon/system1_5/.venv/bin/python -u /home/shoon/cts_e1_s5/scripts/verify_frozen_parity_cpu.py 2>&1 | tee "$LOG_FILE"

# Final memory log
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] FINAL free -g ===" | tee -a "$MEM_LOG"
free -g | tee -a "$MEM_LOG"
echo "[SAFETY] Parity verification runner completed successfully."

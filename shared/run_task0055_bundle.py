#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0055_bundle.py - TASK-0055 Automated Inspection Runner
Executes single bundled SSH call to inspect:
0. Uptime and tmux sessions
1. Soft switch progress (raw_s7/s7deq_soft_v2.log, results files)
2. P6 chain progress (raw_s7/p6_chain.log, results files)
3. GPU 0~7 telemetry, compute apps, PID command lines, free -h
4. Output saved to scratch/task0055_out.txt
"""

import datetime
import math
import os
import subprocess
import sys
import time

REMOTE_SCRIPT = r"""
set -euo pipefail
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0055: INTERMEDIATE INSPECTION (S7DEQ, P6, REBOOT) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

# 0. Uptime and tmux sessions
echo "=== 0. UPTIME & TMUX SESSIONS ==="
echo -n "Boot timestamp (uptime -s): "
uptime -s
echo -n "Current uptime (uptime): "
uptime
echo "--- Tmux Sessions ---"
tmux ls

# 1. Soft Switch (s7deq_soft_v2)
echo "=== 1. SOFT SWITCH (S7DEQ_SOFT_V2) PROGRESS ==="
LOG_FILE="raw_s7/s7deq_soft_v2.log"
if [ -f "${LOG_FILE}" ]; then
  echo "--- 1.1 RUNNING EPS Lines ---"
  grep "=== RUNNING EPS=" "${LOG_FILE}" || echo "None"

  echo "--- 1.2 Completed eps Lines ---"
  grep "Completed eps=" "${LOG_FILE}" || echo "None"

  echo "--- 1.3 Item Header Counts & Latest Progress (NO metrics) ---"
  TOTAL_HEADERS=$(grep -cE '^\[[0-9]{4}/0074\]' "${LOG_FILE}" || true)
  echo "Total item headers in log: ${TOTAL_HEADERS}"
  echo "Last 5 item headers (indices only):"
  grep -oE '^\[[0-9]{4}/0074\]\s+q[0-9]+_a[0-9.]+' "${LOG_FILE}" | tail -5 || echo "None"
  echo -n "Latest item header: "
  grep -oE '^\[[0-9]{4}/0074\]' "${LOG_FILE}" | tail -1 || echo "None"

  echo "--- 1.4 Errors / Traceback Check ---"
  grep -inE "error|traceback|exception" "${LOG_FILE}" | tail -10 || echo "No errors or tracebacks detected."
else
  echo "[ERROR] ${LOG_FILE} missing!"
fi

echo "--- 1.5 Results Files Listing (s7deq_soft_v2_eps*) ---"
ls -la --time-style=full-iso results_s7/s7deq_soft_v2_eps* 2>/dev/null || echo "No eps files yet"

# 2. P6 Progress
echo "=== 2. P6 CHAIN PROGRESS ==="
echo "--- 2.1 P6 Log Last 8 Lines ---"
if [ -f "raw_s7/p6_chain.log" ]; then
  tail -n 8 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log missing"
fi

echo "--- 2.2 P6 Results Files Listing & Line Counts ---"
ls -la --time-style=full-iso results_s7/p6_gen_*.jsonl results_s7/p6_score* results_s7/p6_*.json 2>/dev/null || true
wc -l results_s7/p6_gen_*.jsonl results_s7/p6_score* 2>/dev/null || true

# 3. GPU 0~7 & System Resources
echo "=== 3. GPU 0~7 & SYSTEM RESOURCES ==="
echo "--- 3.1 GPU 0~7 Telemetry ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv

echo "--- 3.2 GPU Compute Apps ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

echo "--- 3.3 Process Details (PID Command Lines) ---"
ps aux | grep -E "python|launch_v5|s7deq|p6" | grep -v grep || true

echo "--- 3.4 System Memory (free -h) ---"
free -h

echo "================================================================================"
echo "=== TASK-0055 INSPECTION COMPLETED ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"
"""

def enforce_cooldown(log_path, min_seconds=65.0):
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) > 1:
            last_ts = lines[-1].split(",")[0]
            try:
                last_dt = datetime.datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                diff = (datetime.datetime.now() - last_dt).total_seconds()
                if diff < min_seconds:
                    wait_s = math.ceil(min_seconds - diff)
                    print(f"Waiting {wait_s}s to satisfy R7-1 cooldown ({diff:.1f}s elapsed)...", flush=True)
                    time.sleep(wait_s)
            except Exception as e:
                pass

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    log_path = os.path.join(root, "shared", "ssh_log.csv")
    out_path = os.path.join(root, "scratch", "task0055_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=== Step 1: Enforcing R7-1 Cooldown before SSH Execution ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    print("=== Step 2: Executing Bundled SSH (TASK-0055 Inspection) ===", flush=True)
    cmd_ssh = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT.strip(), "--purpose", "task0055_intermediate_inspection"]
    res = subprocess.run(cmd_ssh, capture_output=True, text=True, encoding="utf-8", errors="replace")

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(res.stdout)
        if res.stderr:
            f.write("\n=== STDERR ===\n")
            f.write(res.stderr)

    print(f"\nSaved raw output to {out_path} ({len(res.stdout.splitlines())} lines, exit code: {res.returncode})")
    print(res.stdout)

    if res.returncode != 0:
        print(f"[ERROR] SSH command failed with code {res.returncode}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()

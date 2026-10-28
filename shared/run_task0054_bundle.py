#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0054_bundle.py - TASK-0054 Automated Runner
Executes:
1. Enforce 65s cooldown (R7-1 compliance)
2. Single bundled SSH execution:
   - Pre-launch checks (launch_v5 SHA, reference SHA, absence of pre-existing output files)
   - Record start timestamp & launch tmux session s7deq_soft_v21 (GPU 1)
   - Wait 20 minutes (1200 seconds) in-server (R7-4 compliance)
   - Collect post-20min telemetry, GPU 0~7 stats, compute apps, filtered s7deq_soft_v2.log, p6_chain stats
Output saved to scratch/task0054_out.txt.
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
echo "=== TASK-0054: LAUNCH_V5 S7DEQ_SOFT_V21 LAUNCH & 20-MIN TELEMETRY ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

# 1. Pre-launch checks
echo "=== 1. PRE-LAUNCH FILE & SHA CHECKS ==="
V5_FILE="scripts/s7deq_soft_v2_launch_v5.sh"
REF_FILE="results_s7/anchor_ref_gpu1_check.pt"
SOFT_FILE="results_s7/anchor_soft_eps1e-12_check.pt"
EXTRA_REF="results_s7/anchor_ref_gpu1_check_v21_extra.pt"

EXPECTED_V5_SHA="7771f43f7f5e5d4e632923b5a70d080eb3c49aaecd75650a3fa954923cc4352c"
EXPECTED_REF_SHA="5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6"

if [ ! -f "${V5_FILE}" ]; then
  echo "[FATAL] ${V5_FILE} missing!"
  exit 1
fi
ACTUAL_V5_SHA=$(sha256sum "${V5_FILE}" | awk '{print $1}')
echo "V5 SHA: ${ACTUAL_V5_SHA}"
if [ "${ACTUAL_V5_SHA}" != "${EXPECTED_V5_SHA}" ]; then
  echo "[FATAL] V5 SHA mismatch! Expected ${EXPECTED_V5_SHA}, got ${ACTUAL_V5_SHA}"
  exit 1
fi

if [ ! -f "${REF_FILE}" ]; then
  echo "[FATAL] ${REF_FILE} missing!"
  exit 1
fi
ACTUAL_REF_SHA=$(sha256sum "${REF_FILE}" | awk '{print $1}')
echo "Ref SHA: ${ACTUAL_REF_SHA}"
if [ "${ACTUAL_REF_SHA}" != "${EXPECTED_REF_SHA}" ]; then
  echo "[FATAL] Ref SHA mismatch! Expected ${EXPECTED_REF_SHA}, got ${ACTUAL_REF_SHA}"
  exit 1
fi

# Must NOT exist:
if [ -f "${SOFT_FILE}" ]; then
  echo "[FATAL] ${SOFT_FILE} already exists!"
  exit 1
fi

if [ -f "${EXTRA_REF}" ]; then
  echo "[FATAL] ${EXTRA_REF} already exists!"
  exit 1
fi

if compgen -G "results_s7/s7deq_soft_v2_eps*.pt" > /dev/null; then
  echo "[FATAL] results_s7/s7deq_soft_v2_eps*.pt already exists!"
  ls -la results_s7/s7deq_soft_v2_eps*.pt
  exit 1
fi

echo "Pre-launch checks completed: All preconditions satisfied."

# 2. Record pre-launch log marker
LOG_FILE="raw_s7/s7deq_soft_v2.log"
PRE_LOG_LINES=0
if [ -f "${LOG_FILE}" ]; then
  PRE_LOG_LINES=$(wc -l < "${LOG_FILE}")
fi
echo "Pre-launch log lines: ${PRE_LOG_LINES}"

# 3. Launch in tmux session s7deq_soft_v21
echo "=== 2. LAUNCHING S7DEQ_SOFT_V21 TMUX SESSION ==="
LAUNCH_START_TS=$(date '+%Y-%m-%d %H:%M:%S KST')
echo "Launch start timestamp: ${LAUNCH_START_TS}"

env -u TEST_MODE -u TEST_ALTER_REF -u TEST_PASS_ANCHOR tmux new-session -d -s s7deq_soft_v21 'bash scripts/s7deq_soft_v2_launch_v5.sh'

sleep 3
tmux ls

# 4. In-server wait: 20 minutes (1200 seconds)
echo "=== 3. IN-SERVER WAIT (1200 SECONDS / 20 MINUTES) ==="
echo "Wait started at $(date '+%Y-%m-%d %H:%M:%S KST')..."
for i in 1 2 3 4; do
  sleep 300
  echo "Elapsed $((i * 300))s / 1200s at $(date '+%Y-%m-%d %H:%M:%S KST')..."
done
echo "Wait completed at $(date '+%Y-%m-%d %H:%M:%S KST')."

# 5. Post-20min Telemetry & Log Collection
echo "=== 4. POST-20MIN TELEMETRY & LOG COLLECTION ==="
COLLECTION_TS=$(date '+%Y-%m-%d %H:%M:%S KST')
echo "Collection timestamp: ${COLLECTION_TS}"

echo "--- 4.1 TMUX SESSIONS ---"
tmux ls

echo "--- 4.2 GPU 0~7 TELEMETRY ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv

echo "--- 4.3 GPU COMPUTE APPS & PROCESS DETAILS ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
ps aux | grep -E "python|launch_v5|s7deq" | grep -v grep || true

echo "--- 4.4 S7DEQ_SOFT_V2.LOG PARSED SECTION (v5 execution) ---"
if [ -f "${LOG_FILE}" ]; then
  POST_LOG_LINES=$(wc -l < "${LOG_FILE}")
  echo "Post-launch log total lines: ${POST_LOG_LINES}, Pre-launch lines: ${PRE_LOG_LINES}"
  tail -n +$((PRE_LOG_LINES + 1)) "${LOG_FILE}" > scratch/task0054_raw_new_log.txt
  /home/shoon/system1_5/.venv/bin/python - << 'PYEOF'
import re

with open("scratch/task0054_raw_new_log.txt", "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

in_main = False
main_count = 0
for line in lines:
    line_s = line.strip()
    if "=== RUNNING EPS=0.005" in line_s:
        in_main = True
        print(line_s)
        continue
    if not in_main:
        print(line_s)
    else:
        # In main run: print banner and progress item key without numerical metrics (수치 0)
        m = re.match(r"^(\[\d+/\d+\]\s+q\d+_a\S+):", line_s)
        if m:
            main_count += 1
            print(f"{m.group(1)} (in progress)")
        elif line_s.startswith("===") or "Finished" in line_s or "Analyzing" in line_s or "Completed" in line_s:
            print(line_s)

if in_main:
    print(f"Total main run progress items logged: {main_count}")
PYEOF
else
  echo "[ERROR] ${LOG_FILE} not found!"
fi

echo "--- 4.5 RESULTS_S7 DIRECTORY LISTING ---"
ls -la --time-style=full-iso results_s7/

echo "--- 4.6 P6_CHAIN PROGRESS ---"
echo "P6 log last 3 lines:"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 3 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log not found"
fi
echo "P6 jsonl line counts:"
wc -l results_s7/p6_gen_*.jsonl 2>/dev/null || true

echo "================================================================================"
echo "=== TASK-0054 BUNDLED RUN COMPLETED ==="
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
    out_path = os.path.join(root, "scratch", "task0054_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=== Step 1: Enforcing R7-1 Cooldown before SSH Execution ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    print("=== Step 2: Executing Bundled SSH (Pre-check, Launch, 20min Wait, Telemetry) ===", flush=True)
    cmd_ssh = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT.strip(), "--purpose", "task0054_launch_and_collect_20min"]
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

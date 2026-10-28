#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0050_bundle.py - TASK-0050 Automated Runner (Part B)
Waits until 20:25:05 KST, then executes:
1. SFTP Upload (scripts/p5_promote_1a_replay_v2.py)
2. Enforce 65s cooldown (R7-1 compliance)
3. Bundled SSH execution:
   - Server SHA verification of scripts/p5_promote_1a_replay_v2.py
   - raw_s7/s7deq_soft_v2.log last 60 lines and anchor status check
   - tmux ls & GPU 0~7 telemetry, compute-apps & PID command lines
   - P6 chain log last 5 lines, results_s7/p6_gen_*.jsonl line count (no content read)
Output saved to scratch/task0050_out.txt.
"""

import os
import sys
import subprocess
import time
import datetime
import math

REMOTE_SCRIPT = r"""
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0050: SERVER COLLECTION & VERIFICATION (20:25+ KST) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. UPLOADED FILE SHA-256 VERIFICATION ==="
f="scripts/p5_promote_1a_replay_v2.py"
if [ -f "$f" ]; then
  sha256sum "$f"
  ls -la --time-style=full-iso "$f"
else
  echo "[ERROR] Missing uploaded file: $f"
fi

echo "=== 2. RAW_S7 S7DEQ_SOFT_V2 LOG LAST 60 LINES & ANCHOR STATUS ==="
LOG_FILE="raw_s7/s7deq_soft_v2.log"
if [ -f "${LOG_FILE}" ]; then
  echo "--- 2.1. ANCHOR STATUS GREP ---"
  grep -E "Zero boundary anchors found|§5 Anchor Check passed|ERROR|exit" "${LOG_FILE}" || true
  
  echo "--- 2.2. RUNNING EPS HEADER & ITEM PROGRESS ---"
  grep -n "=== RUNNING EPS=" "${LOG_FILE}" || echo "RUNNING EPS not reached yet"
  echo "Progress item lines (line numbers and item index only, r-values omitted):"
  grep -n -E "\[[0-9]{4}/[0-9]{4}\]" "${LOG_FILE}" | awk -F: '{print $1 ":" substr($2, 1, 13)}' | tail -n 25 || true

  echo "--- 2.3. LAST 60 LINES OF RAW_S7/S7DEQ_SOFT_V2.LOG ---"
  tail -n 60 "${LOG_FILE}"
else
  echo "[ERROR] ${LOG_FILE} not found"
fi

echo "=== 3. TMUX SESSIONS & GPU STATUS ==="
echo "--- 3.1. TMUX SESSIONS ---"
tmux ls

echo "--- 3.2. GPU 0~7 STATUS ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv

echo "--- 3.3. GPU COMPUTE APPS ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

echo "--- 3.4. COMPUTE APP PROCESS COMMAND LINES ---"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u); do
  ps -p "$pid" -o pid,lstart,cmd --no-headers || true
done

echo "=== 4. P6_CHAIN STATUS & GENERATION FILES ==="
echo "--- 4.1. P6_CHAIN LOG LAST 5 LINES ---"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 5 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log not found"
fi

echo "--- 4.2. P6 GENERATION JSONL FILES LISTING & LINE COUNTS (NO CONTENT READ) ---"
ls -la --time-style=full-iso results_s7/p6_gen_*.jsonl 2>/dev/null || echo "No results_s7/p6_gen_*.jsonl files found"
wc -l results_s7/p6_gen_*.jsonl 2>/dev/null || true

echo "=== TASK-0050 COLLECTION FINISHED ==="
"""

def wait_until_target(target_hour=20, target_minute=25, target_second=5):
    now = datetime.datetime.now()
    target = now.replace(hour=target_hour, minute=target_minute, second=target_second, microsecond=0)
    if now < target:
        diff = (target - now).total_seconds()
        print(f"Waiting until {target.strftime('%H:%M:%S')} KST ({diff:.1f}s remaining)...", flush=True)
        time.sleep(diff)
    print(f"Current time is now {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')} (>= 20:25:00). Proceeding.", flush=True)

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
    sftp_batch = os.path.join(root, "scratch", "upload_task50.sftp")
    out_path = os.path.join(root, "scratch", "task0050_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1. Enforce time constraint: >= 20:25 KST
    print("=== Step 0: Enforcing 20:25 KST Time Constraint ===", flush=True)
    wait_until_target(20, 25, 5)

    # 2. Step 1: SFTP Upload
    print("=== Step 1: SFTP Upload of scripts/p5_promote_1a_replay_v2.py ===", flush=True)
    enforce_cooldown(log_path, 65.0)
    cmd_sftp = [sys.executable, cts_ssh, "sftp", sftp_batch, "--purpose", "task0050_upload_1a_replay_v2"]
    res1 = subprocess.run(cmd_sftp)
    if res1.returncode != 0:
        print(f"[ERROR] SFTP upload failed with exit code {res1.returncode}")
        sys.exit(res1.returncode)
    print("SFTP upload finished successfully.", flush=True)

    # 3. Step 2: Enforce Cooldown before Step 3
    print("=== Step 2: Enforcing 65s Cooldown (R7-1 compliance) ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    # 4. Step 3: Remote Collection Execution
    print("=== Step 3: Remote Collection Execution ===", flush=True)
    cmd_ssh = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", "task0050_collect_and_verify"]
    with open(out_path, "w", encoding="utf-8") as f:
        res2 = subprocess.run(cmd_ssh, stdout=f, stderr=subprocess.STDOUT)

    print(f"Remote execution finished with exit code {res2.returncode}. Output saved to {out_path}", flush=True)
    sys.exit(res2.returncode)

if __name__ == "__main__":
    main()

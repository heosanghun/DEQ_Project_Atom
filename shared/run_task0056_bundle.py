#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0056_bundle.py - TASK-0056 Server State Inspection Runner
Executes single bundled SSH call to inspect:
0. Uptime and tmux sessions
1. Soft switch progress (raw_s7/s7deq_soft_v2.log, results files, SHA-256)
2. JSON input SHA records
3. GPU 0~7 telemetry, compute apps, PID command lines, free -h
4. P6 chain progress (p6_chain.log, p6_gen_*.jsonl, flags)
5. ①-a Replay prerequisite file SHAs and flags
Saves raw stdout to scratch/task0056_out.txt.
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
echo "=== TASK-0056: SERVER STATE & S7 DEQ FINAL INSPECTION ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

# 0. Uptime and tmux sessions
echo "=== 0. UPTIME & TMUX SESSIONS ==="
echo -n "Boot timestamp (uptime -s): "
uptime -s
echo -n "Current uptime (uptime): "
uptime
echo "--- Tmux Sessions ---"
tmux ls || echo "No active tmux sessions"

# 1. Soft Switch (s7deq_soft_v2)
echo "=== 1. SOFT SWITCH (S7DEQ_SOFT_V2) PROGRESS ==="
LOG_FILE="raw_s7/s7deq_soft_v2.log"
if [ -f "${LOG_FILE}" ]; then
  echo "--- 1.1 RUNNING EPS Lines ---"
  grep "=== RUNNING EPS=" "${LOG_FILE}" || echo "None"

  echo "--- 1.2 Completed eps Lines ---"
  grep "Completed eps=" "${LOG_FILE}" || echo "None"

  echo "--- 1.3 ALL S7 DEQ SOFT V2 RUNS FINISHED Line ---"
  grep "ALL S7 DEQ SOFT V2 RUNS FINISHED" "${LOG_FILE}" || echo "Not yet finished"

  echo "--- 1.4 Item Header Counts & Latest Progress (NO metrics) ---"
  TOTAL_HEADERS=$(grep -cE '^\[[0-9]{4}/0074\]' "${LOG_FILE}" || true)
  echo "Total item headers in log: ${TOTAL_HEADERS}"
  echo "Last 5 item headers (indices only):"
  grep -oE '^\[[0-9]{4}/0074\]\s+q[0-9]+_a[0-9.]+' "${LOG_FILE}" | tail -5 || echo "None"
  echo -n "Latest item header: "
  grep -oE '^\[[0-9]{4}/0074\]' "${LOG_FILE}" | tail -1 || echo "None"

  echo "--- 1.5 Errors / Traceback in v5 Execution Interval (line 599+) ---"
  tail -n +599 "${LOG_FILE}" | grep -inE "error|traceback|exception" || echo "No errors or tracebacks detected in v5."
else
  echo "[ERROR] ${LOG_FILE} missing!"
fi

echo "--- 1.6 Results Files Listing (s7deq_soft_v2_eps*) ---"
ls -la --time-style=full-iso results_s7/s7deq_soft_v2_eps* 2>/dev/null || echo "No eps files"

echo "--- 1.7 Results Files SHA-256 Hashes ---"
sha256sum results_s7/s7deq_soft_v2_eps* 2>/dev/null || echo "No eps files for sha256"

# 2. JSON SHA Records
echo "=== 2. JSON SHA RECORDS ==="
/home/shoon/system1_5/.venv/bin/python -c '
import json, glob, sys
files = sorted(glob.glob("results_s7/s7deq_soft_v2_eps*.json"))
if not files:
    print("No JSON files found.")
for p in files:
    print(f"=== {p} ===")
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        for k in ["sha_records", "inputs_sha", "shas", "input_files", "metadata"]:
            if k in d:
                print(f"  {k}: {json.dumps(d[k], ensure_ascii=False)}")
    except Exception as e:
        print(f"  Error reading {p}: {e}")
' 2>&1 || true

# 3. GPU 0~7 & System Resources
echo "=== 3. GPU 0~7 & SYSTEM RESOURCES ==="
echo "--- 3.1 GPU 0~7 Telemetry ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv

echo "--- 3.2 GPU Compute Apps ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv || echo "No compute apps"

echo "--- 3.3 Process Details (PID Command Lines) ---"
ps aux | grep -E "python|launch_v5|s7deq|p6" | grep -v grep || true

echo "--- 3.4 System Memory (free -h) ---"
free -h

# 4. P6 Progress
echo "=== 4. P6 CHAIN PROGRESS ==="
echo "--- 4.1 P6 Log Last 5 Lines ---"
if [ -f "raw_s7/p6_chain.log" ]; then
  tail -n 5 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log missing"
fi

echo "--- 4.2 P6 Results Files Listing & Line Counts ---"
ls -la --time-style=full-iso results_s7/p6_gen_*.jsonl results_s7/p6_score* results_s7/p6_*.json results_s7/P6_GEN_*.flag 2>/dev/null || true
wc -l results_s7/p6_gen_*.jsonl 2>/dev/null || true

# 5. ①-a Replay Prerequisite Files & Flags Check
echo "=== 5. 1A REPLAY PREREQUISITE FILES & FLAGS ==="
echo "--- 5.1 Prerequisite File SHAs ---"
sha256sum \
  scripts/s7_zstar_lowest_run.py \
  cts/deq/zstar_canonical_lowest.py \
  configs/s7_train1000_kstar_replay_325.json \
  configs/zstar_canonical.yaml \
  /home/shoon/cts_e1/scratch/harness_A_v1.py \
  scripts/p5_promote_1a_replay_v2.py \
  results_s7/zstar_train1000_P4.pt \
  results_s7/p5_inputs_train1000_v3.json 2>&1 || true

echo "--- 5.2 Flag & Precheck Existence Checks ---"
for f in raw_s7/KSTAR_REPLAY_STARTED.flag raw_s7/KSTAR_REPLAY_DONE.flag results_s7/kstar_replay_precheck5_gpu0.pt results_s7/kstar_replay_precheck5_gpu1.pt; do
  if [ -e "$f" ]; then
    echo "EXISTS: $f"
    ls -la "$f"
  else
    echo "NOT_EXISTS: $f"
  fi
done

echo "================================================================================"
echo "=== TASK-0056 INSPECTION COMPLETED ==="
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
    out_path = os.path.join(root, "scratch", "task0056_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=== Step 1: Enforcing R7-1 Cooldown before SSH Execution ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    print("=== Step 2: Executing Bundled SSH (TASK-0056 Inspection) ===", flush=True)
    cmd_ssh = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT.strip(), "--purpose", "task0056_inspection_and_telemetry"]
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

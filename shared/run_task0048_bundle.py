#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0048_bundle.py - TASK-0048 Automated Runner
Enforces R7 cooldown, executes bundled SSH on server:
1. SHA and conflict verification
2. Launch tmux session s7deq_soft_v2 (GPU 1)
3. Wait 15 minutes (900s) on server side (R7-4 compliance)
4. Collect all required telemetry into scratch/task0048_out.txt
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
echo "=== TASK-0048: PRE-RUN SHA VERIFICATION & LAUNCH IN TMUX ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. PRE-RUN SHA AND CONFLICT VERIFICATION ==="
EXPECTED_LAUNCH="cb3ad22f61b9604da92414e3bedbf9b7c38bc387b316a59c5dd56fe6cb7c2784"
EXPECTED_RUN="2687a54e28b73c15e2980ad204724fb4f8903fb7d3b8f3d33d4e3ae11283db17"
EXPECTED_ZSTAR="d1942c3cf5defbad6c148e69102b1ca6e8c64d1da5dd731434782364fd689f07"
EXPECTED_ANALYZE="07d63b1e9fdce284d93216cc00586bb2f4e6223a7aee13007b650dc94ec73aec"
EXPECTED_CONFIG_PREFIX="42df1f49"

SHA_LAUNCH=$(sha256sum scripts/s7deq_soft_v2_launch_v2.sh | awk '{print $1}')
SHA_RUN=$(sha256sum scripts/s7deq_soft_v2_run_v2.py | awk '{print $1}')
SHA_ZSTAR=$(sha256sum exe_scratch/s7/soft_v2/zstar_soft_lowest.py | awk '{print $1}')
SHA_ANALYZE=$(sha256sum scripts/s7deq_soft_v2_analyze.py | awk '{print $1}')
SHA_CONFIG=$(sha256sum configs/s6_items_74.json | awk '{print $1}')

echo "SHA_LAUNCH:  $SHA_LAUNCH"
echo "SHA_RUN:     $SHA_RUN"
echo "SHA_ZSTAR:   $SHA_ZSTAR"
echo "SHA_ANALYZE: $SHA_ANALYZE"
echo "SHA_CONFIG:  $SHA_CONFIG"

FAIL=0
if [ "$SHA_LAUNCH" != "$EXPECTED_LAUNCH" ]; then echo "[MISMATCH] launch_v2 SHA mismatch"; FAIL=1; fi
if [ "$SHA_RUN" != "$EXPECTED_RUN" ]; then echo "[MISMATCH] run_v2 SHA mismatch"; FAIL=1; fi
if [ "$SHA_ZSTAR" != "$EXPECTED_ZSTAR" ]; then echo "[MISMATCH] zstar SHA mismatch"; FAIL=1; fi
if [ "$SHA_ANALYZE" != "$EXPECTED_ANALYZE" ]; then echo "[MISMATCH] analyze SHA mismatch"; FAIL=1; fi
if [[ "$SHA_CONFIG" != "$EXPECTED_CONFIG_PREFIX"* ]]; then echo "[MISMATCH] config SHA mismatch"; FAIL=1; fi

# Check conflicting target output files
echo "Checking conflicting output files in results_s7/..."
for f in \
  results_s7/anchor_ref_gpu1_check.pt \
  results_s7/anchor_soft_eps0.02_check.pt \
  results_s7/s7deq_soft_v2_eps0.005.pt \
  results_s7/s7deq_soft_v2_eps0.01.pt \
  results_s7/s7deq_soft_v2_eps0.02.pt \
  results_s7/s7deq_soft_v2_eps0.005.json \
  results_s7/s7deq_soft_v2_eps0.01.json \
  results_s7/s7deq_soft_v2_eps0.02.json; do
  if [ -f "$f" ]; then
    echo "[CONFLICT] Target file already exists: $f"
    FAIL=1
  else
    echo "[CLEAN] Target file absent: $f"
  fi
done

if tmux has-session -t s7deq_soft_v2 2>/dev/null; then
  echo "[CONFLICT] tmux session s7deq_soft_v2 already exists!"
  FAIL=1
fi

if [ $FAIL -ne 0 ]; then
  echo "[ABORT] Pre-run check failed. Aborting launch."
  exit 1
fi
echo "[OK] Pre-run SHA and conflict checks all verified."

echo "=== 2. LAUNCH IN TMUX s7deq_soft_v2 ==="
START_TIME=$(date '+%Y-%m-%d %H:%M:%S KST')
echo "Launch timestamp: ${START_TIME}"
echo "Launch command: tmux new-session -d -s s7deq_soft_v2 'bash scripts/s7deq_soft_v2_launch_v2.sh'"

tmux new-session -d -s s7deq_soft_v2 "bash scripts/s7deq_soft_v2_launch_v2.sh"
echo "tmux session created successfully."
tmux ls

echo "=== 3. WAITING 15 MINUTES (900s) (R7-4 compliance) ==="
echo "Entering 15-minute wait at $(date '+%Y-%m-%d %H:%M:%S KST')..."
sleep 900
echo "15-minute wait completed at $(date '+%Y-%m-%d %H:%M:%S KST')."

echo "=== 4. POST-15MIN COLLECTION ==="

echo "--- 4.1. TMUX STATUS ---"
tmux ls

echo "--- 4.2. ANCHOR CHECK SECTION FROM LOG ---"
if [ -f raw_s7/s7deq_soft_v2.log ]; then
  awk '/=== §5 PRE-LAUNCH ANCHOR CHECK/,/=== RUNNING EPS=0.005/' raw_s7/s7deq_soft_v2.log
else
  echo "[ERROR] Log file raw_s7/s7deq_soft_v2.log not found!"
fi

echo "--- 4.3. RUNNING EPS=0.005 HEADER AND PROGRESS LINE NUMBERS ---"
if [ -f raw_s7/s7deq_soft_v2.log ]; then
  grep -n "=== RUNNING EPS=0.005" raw_s7/s7deq_soft_v2.log || echo "EPS=0.005 not reached"
  echo "Progress item lines (line numbers and item count only, r-values omitted):"
  grep -n -E "\[[0-9]{4}/[0-9]{4}\]" raw_s7/s7deq_soft_v2.log | awk -F: '{print $1 ":" substr($2, 1, 13)}' | tail -n 15 || true
fi

echo "--- 4.4. GPU USAGE METRICS ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv

echo "--- 4.5. GPU COMPUTE APPS ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

echo "--- 4.6. PID PROCESS DETAILS ---"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u); do
  ps -p "$pid" -o pid,lstart,cmd --no-headers || true
done

echo "--- 4.7. P6_CHAIN LAST 3 LINES ---"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 3 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log not found"
fi

echo "--- 4.8. TMUX PANE CAPTURE ---"
tmux capture-pane -p -t s7deq_soft_v2 2>/dev/null | tail -n 25 || echo "tmux pane capture not available"

echo "=== TASK-0048 COLLECTION FINISHED ==="
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
    out_path = os.path.join(root, "scratch", "task0048_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=== Step 1: Enforcing Cooldown (R7-1 compliance) ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    print("=== Step 2: Executing Bundled SSH (Launch + 15min Wait + Telemetry) ===", flush=True)
    cmd_ssh = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", "task0048_launch_and_collect_15min"]
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd_ssh, stdout=f, stderr=subprocess.STDOUT)

    print(f"Execution finished with exit code {res.returncode}. Output saved to {out_path}", flush=True)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

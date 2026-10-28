#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0052_bundle.py - TASK-0052 Automated Runner (SFTP upload + bundled SSH verification)
Executes:
1. SFTP Upload (scripts/s7deq_soft_v2_launch_v4.sh)
2. Enforce 65s cooldown (R7-1 compliance)
3. Bundled SSH execution:
   - File SHA-256, size, mtime checks for launch_v4.sh and reference file
   - File existence checks for results_s7/anchor_soft_eps1e-12_check.pt and anchor_ref_gpu1_check_v21_extra.pt
   - tmux ls, GPU 0~7 telemetry, P6 chain log & jsonl line count
   - GPU execution count: 0 (launch_v4.sh is strictly not executed)
Output saved to scratch/task0052_out.txt.
"""

import datetime
import math
import os
import subprocess
import sys
import time

REMOTE_SCRIPT = r"""
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0052: SERVER SHA VERIFICATION & TELEMETRY COLLECTION ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. LAUNCH_V4.SH SHA-256, SIZE, AND PERMISSIONS ==="
if [ -f "scripts/s7deq_soft_v2_launch_v4.sh" ]; then
  sha256sum "scripts/s7deq_soft_v2_launch_v4.sh"
  ls -la --time-style=full-iso "scripts/s7deq_soft_v2_launch_v4.sh"
else
  echo "[ERROR] scripts/s7deq_soft_v2_launch_v4.sh missing!"
fi

echo "=== 2. REFERENCE AND PRE-CHECK FILES STATUS ==="
echo "--- 2.1 Canonical reference file: results_s7/anchor_ref_gpu1_check.pt ---"
if [ -f "results_s7/anchor_ref_gpu1_check.pt" ]; then
  sha256sum "results_s7/anchor_ref_gpu1_check.pt"
  ls -la --time-style=full-iso "results_s7/anchor_ref_gpu1_check.pt"
else
  echo "[ERROR] results_s7/anchor_ref_gpu1_check.pt missing!"
fi

echo "--- 2.2 Target soft check file: results_s7/anchor_soft_eps1e-12_check.pt ---"
if [ -f "results_s7/anchor_soft_eps1e-12_check.pt" ]; then
  echo "results_s7/anchor_soft_eps1e-12_check.pt: EXISTS (WARNING: exit 3 condition)"
  ls -la --time-style=full-iso "results_s7/anchor_soft_eps1e-12_check.pt"
else
  echo "results_s7/anchor_soft_eps1e-12_check.pt: NOT_EXISTS (Clean pre-condition for launch_v4)"
fi

echo "--- 2.3 Extra reference file: results_s7/anchor_ref_gpu1_check_v21_extra.pt ---"
if [ -f "results_s7/anchor_ref_gpu1_check_v21_extra.pt" ]; then
  echo "results_s7/anchor_ref_gpu1_check_v21_extra.pt: EXISTS (WARNING: exit 3 condition)"
  ls -la --time-style=full-iso "results_s7/anchor_ref_gpu1_check_v21_extra.pt"
else
  echo "results_s7/anchor_ref_gpu1_check_v21_extra.pt: NOT_EXISTS (Clean pre-condition for launch_v4)"
fi

echo "=== 3. TMUX SESSIONS & GPU STATUS ==="
echo "--- 3.1 TMUX SESSIONS ---"
tmux ls

echo "--- 3.2 GPU 0~7 TELEMETRY ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv

echo "--- 3.3 GPU COMPUTE APPS ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

echo "=== 4. P6_CHAIN LOG & JSONL LINE COUNTS ==="
echo "--- 4.1 P6_CHAIN LOG LAST 5 LINES ---"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 5 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log not found"
fi

echo "--- 4.2 P6 JSONL FILES LISTING & LINE COUNTS (NO CONTENT READ) ---"
ls -la --time-style=full-iso results_s7/p6_gen_*.jsonl 2>/dev/null || true
wc -l results_s7/p6_gen_*.jsonl 2>/dev/null || true

echo "=== TASK-0052 COLLECTION FINISHED (GPU EXECUTION COUNT: 0) ==="
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
    sftp_batch = os.path.join(root, "scratch", "upload_task52.sftp")
    out_path = os.path.join(root, "scratch", "task0052_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Step 1: SFTP Upload of scripts/s7deq_soft_v2_launch_v4.sh
    print("=== Step 1: SFTP Upload of scripts/s7deq_soft_v2_launch_v4.sh ===", flush=True)
    enforce_cooldown(log_path, 65.0)
    cmd_sftp = [sys.executable, cts_ssh, "sftp", sftp_batch, "--purpose", "task0052_upload_launch_v4"]
    res1 = subprocess.run(cmd_sftp)
    if res1.returncode != 0:
        print(f"[ERROR] SFTP upload failed with exit code {res1.returncode}")
        sys.exit(res1.returncode)
    print("SFTP upload finished successfully.", flush=True)

    # Step 2: Enforce Cooldown before Step 3
    print("=== Step 2: Enforcing 65s Cooldown (R7-1 compliance) ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    # Step 3: Remote Collection Execution
    print("=== Step 3: Remote Collection Execution ===", flush=True)
    cmd_ssh = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", "task0052_verify_sha_and_telemetry"]
    with open(out_path, "w", encoding="utf-8") as f:
        res2 = subprocess.run(cmd_ssh, stdout=f, stderr=subprocess.STDOUT)

    print(f"Remote execution finished with exit code {res2.returncode}. Output saved to {out_path}", flush=True)
    sys.exit(res2.returncode)

if __name__ == "__main__":
    main()

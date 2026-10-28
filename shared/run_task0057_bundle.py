#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0057_bundle.py - TASK-0057 Automated Inspection, Import Verification, and Launch Runner
Executes:
1. R7-1 Cooldown enforcement
2. Single bundled SSH execution:
   (A) Soft switch completion check, results listing & SHA-256
   (B) CPU import check & launch_v1 SHA verification
   (C) kstar_replay launch on GPU 1, 15-min in-server monitoring, telemetry collection
3. SFTP download of the 6 soft_v2 JSON/MD files (if finished)
4. Local vs Server SHA-256 bit-exact verification
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
echo "=== TASK-0057: BUNDLED EXECUTION (A, B, C) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

LOG_FILE="raw_s7/s7deq_soft_v2.log"

echo "=== 1. (A) SOFT SWITCH COMPLETION CHECK ==="
FINISHED=0
if [ -f "${LOG_FILE}" ]; then
  if grep -q "ALL S7 DEQ SOFT V2 RUNS FINISHED" "${LOG_FILE}" && grep -q "Completed eps=0.02" "${LOG_FILE}"; then
    FINISHED=1
  fi
fi

if [ "${FINISHED}" -eq 1 ]; then
  echo "[STATUS] ALL S7 DEQ SOFT V2 RUNS FINISHED DETECTED."
  grep "Completed eps=" "${LOG_FILE}" || true
  grep "ALL S7 DEQ SOFT V2 RUNS FINISHED" "${LOG_FILE}" || true
else
  echo "[STATUS] NOT YET FINISHED."
  TOTAL_HEADERS=$(grep -cE '^\[[0-9]{4}/0074\]' "${LOG_FILE}" 2>/dev/null || echo 0)
  echo "Total item headers in log: ${TOTAL_HEADERS}"
  echo -n "Latest item header: "
  grep -oE '^\[[0-9]{4}/0074\]\s+q[0-9]+_a[0-9.]+' "${LOG_FILE}" 2>/dev/null | tail -1 || echo "None"
fi

echo "--- Tmux s7deq_soft_v21 Status ---"
if tmux has-session -t s7deq_soft_v21 2>/dev/null; then
  echo "Session s7deq_soft_v21 is STILL ACTIVE."
else
  echo "Session s7deq_soft_v21 is TERMINATED (closed)."
fi

echo "--- GPU 1 Compute Apps ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv | grep "GPU-9410691f" || echo "GPU 1 has NO active compute apps (IDLE)."

if [ "${FINISHED}" -eq 1 ]; then
  echo "--- Results Files Listing (s7deq_soft_v2_eps*) ---"
  ls -la --time-style=full-iso results_s7/s7deq_soft_v2_eps*

  echo "--- Results Files SHA-256 Hashes ---"
  sha256sum results_s7/s7deq_soft_v2_eps*

  echo "================================================================================"
  echo "=== 2. (B) IMPORT CHECK & LAUNCH_V1 SHA VERIFICATION (CPU) ==="
  echo "================================================================================"

  echo "--- 2.1 CPU Import Verification ---"
  CUDA_VISIBLE_DEVICES="" /home/shoon/system1_5/.venv/bin/python -c "import scripts.p5_promote_1a_replay_v2 as m, inspect; print('MODULE_FILE:', m.__file__); print('SIGNATURE:', inspect.signature(m.evaluate_replay))"

  echo "--- 2.2 scripts/kstar_replay_launch_v1.sh SHA-256 Verification ---"
  sha256sum scripts/kstar_replay_launch_v1.sh

  echo "================================================================================"
  echo "=== 3. (C) REPLAY LAUNCH ON GPU 1 & 15-MINUTE IN-SERVER WAIT ==="
  echo "================================================================================"

  LAUNCH_CMD="bash scripts/kstar_replay_launch_v1.sh --gpu 1"
  echo "Starting kstar_replay tmux session: ${LAUNCH_CMD}"
  START_KST=$(date '+%Y-%m-%d %H:%M:%S KST')
  echo "Launch timestamp: ${START_KST}"

  env -u TEST_MODE -u TEST_PRECHECK_FAIL -u TEST_ALTER_SHA -u TEST_MOCK_RUN tmux new-session -d -s kstar_replay "${LAUNCH_CMD}"

  echo "Waiting 15 minutes (900s) in server with live 60s progress heartbeat..."
  for i in $(seq 1 15); do
    sleep 60
    echo "[Heartbeat ${i}/15 min - $(date '+%H:%M:%S')] Log tail:"
    tail -n 3 raw_s7/kstar_replay_gpu1.log 2>/dev/null || echo "(log not created yet)"
  done

  echo "=== 15-MINUTE WAIT ELAPSED: COLLECTING TELEMETRY & FULL LOG ==="
  echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"

  echo "--- 3.1 raw_s7/kstar_replay_gpu1.log Full Content ---"
  if [ -f "raw_s7/kstar_replay_gpu1.log" ]; then
    cat raw_s7/kstar_replay_gpu1.log
  else
    echo "[ERROR] raw_s7/kstar_replay_gpu1.log not found!"
  fi

  echo "--- 3.2 Started Flag Check ---"
  if [ -f "raw_s7/KSTAR_REPLAY_STARTED.flag" ]; then
    echo "EXISTS: raw_s7/KSTAR_REPLAY_STARTED.flag"
    cat raw_s7/KSTAR_REPLAY_STARTED.flag
  else
    echo "NOT_EXISTS: raw_s7/KSTAR_REPLAY_STARTED.flag"
  fi

  echo "--- 3.3 Precheck Output File Check ---"
  ls -la results_s7/kstar_replay_precheck5_gpu1.pt 2>/dev/null || echo "precheck5_gpu1.pt not found"

  echo "--- 3.4 GPU 0~7 Telemetry ---"
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv

  echo "--- 3.5 GPU Compute Apps ---"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv || echo "No compute apps"

  echo "--- 3.6 Tmux Sessions ---"
  tmux ls

  echo "--- 3.7 Process List ---"
  ps aux | grep -E "python|launch_v1|kstar|p6" | grep -v grep || true

else
  echo "[ABORT] Soft switch not yet finished. Skipping (B) and (C) as per TASK-0057 instruction."
fi

echo "================================================================================"
echo "=== TASK-0057 BUNDLED EXECUTION FINISHED ==="
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
    out_path = os.path.join(root, "scratch", "task0057_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=== Step 1: Enforcing R7-1 Cooldown before Bundled SSH Execution ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    print("=== Step 2: Executing Bundled SSH (TASK-0057 A, B, C) ===", flush=True)
    cmd_ssh = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT.strip(), "--purpose", "task0057_bundled_execution"]
    res_ssh = subprocess.run(cmd_ssh, capture_output=True, text=True, encoding="utf-8", errors="replace")

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(res_ssh.stdout)
        if res_ssh.stderr:
            f.write("\n=== STDERR ===\n")
            f.write(res_ssh.stderr)

    print(f"\nSaved raw output to {out_path} ({len(res_ssh.stdout.splitlines())} lines, exit code: {res_ssh.returncode})")
    print(res_ssh.stdout)

    if res_ssh.returncode != 0:
        print(f"[ERROR] SSH command failed with code {res_ssh.returncode}", file=sys.stderr)
        sys.exit(res_ssh.returncode)

    # Check if soft v2 was finished in output
    if "[STATUS] ALL S7 DEQ SOFT V2 RUNS FINISHED DETECTED." in res_ssh.stdout:
        print("\n=== Step 3: Enforcing Cooldown before SFTP Download ===", flush=True)
        time.sleep(65)

        print("=== Step 4: Downloading 6 soft_v2 JSON/MD files via SFTP ===", flush=True)
        sftp_batch = os.path.join(root, "scratch", "download_soft_v2.sftp")
        cmd_sftp = [sys.executable, cts_ssh, "sftp", sftp_batch, "--purpose", "task0057_download_soft_v2"]
        res_sftp = subprocess.run(cmd_sftp, capture_output=True, text=True, encoding="utf-8", errors="replace")

        with open(out_path, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n=== SFTP DOWNLOAD OUTPUT ===\n")
            f.write(f"SFTP Return Code: {res_sftp.returncode}\n")
            f.write(res_sftp.stdout + "\n")
            if res_sftp.stderr:
                f.write(res_sftp.stderr + "\n")

        print("SFTP download returncode:", res_sftp.returncode)
        print(res_sftp.stdout)

        if res_sftp.returncode != 0:
            print("[ERROR] SFTP download failed!", file=sys.stderr)
            sys.exit(res_sftp.returncode)

        # Local SHA verification
        import hashlib
        print("\n=== Step 5: Local vs Server SHA Verification for Downloaded Files ===")
        soft_v2_dir = os.path.join(root, "shared", "reports", "soft_v2")
        files_to_check = [
            "s7deq_soft_v2_eps0.005.json",
            "s7deq_soft_v2_eps0.005.md",
            "s7deq_soft_v2_eps0.01.json",
            "s7deq_soft_v2_eps0.01.md",
            "s7deq_soft_v2_eps0.02.json",
            "s7deq_soft_v2_eps0.02.md",
        ]
        with open(out_path, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n=== LOCAL DOWNLOADED FILES SHA-256 ===\n")
            for fname in files_to_check:
                fpath = os.path.join(soft_v2_dir, fname)
                if os.path.exists(fpath):
                    h = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
                    size = os.path.getsize(fpath)
                    line_str = f"{h}  {fname} ({size} B)\n"
                    print(line_str.strip())
                    f.write(line_str)
                else:
                    err_str = f"MISSING: {fname}\n"
                    print(err_str.strip())
                    f.write(err_str)

    print(f"\nAll operations completed. Output recorded in {out_path}.")

if __name__ == "__main__":
    main()

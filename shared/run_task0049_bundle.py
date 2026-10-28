#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0049_bundle.py - TASK-0049 Automated Runner (Part B)
Waits until 19:50:00 KST, then executes:
1. SFTP Upload (new configs & scripts/p5_promote_1a_replay.py)
2. Enforce 65s cooldown
3. Bundled SSH execution:
   - Server SHA verification of uploaded files
   - raw_s7/s7deq_soft_v2.log anchor check & running EPS extraction
   - tmux ls & GPU 0~7 telemetry, compute-apps & PID command lines
   - P6 chain log last 5 lines, results_s7/p6_gen_*.jsonl line count
   - zstar_train1000_P4.pt 325 keys existence & z_star shape verification on CPU
Output saved to scratch/task0049_out.txt.
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
echo "=== TASK-0049: SERVER COLLECTION & VERIFICATION (19:50+ KST) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. UPLOADED FILES SHA-256 VERIFICATION ==="
for f in configs/s7_train1000_kstar_replay_325.json scripts/p5_promote_1a_replay.py; do
  if [ -f "$f" ]; then
    sha256sum "$f"
    ls -la --time-style=full-iso "$f"
  else
    echo "[ERROR] Missing uploaded file: $f"
  fi
done

echo "=== 2. RAW_S7 S7DEQ_SOFT_V2 LOG ANCHOR & RUNNING STATUS ==="
LOG_FILE="raw_s7/s7deq_soft_v2.log"
if [ -f "${LOG_FILE}" ]; then
  echo "--- 2.1. ANCHOR CHECK SUMMARY LINES ---"
  grep -E "Zero boundary anchors found|§5 Anchor Check passed|ERROR|exit" "${LOG_FILE}" || true

  echo "--- 2.2. ANCHOR CHECK FULL SECTION ---"
  awk '/=== §5 PRE-LAUNCH ANCHOR CHECK/,/=== RUNNING EPS=/' "${LOG_FILE}" || true

  echo "--- 2.3. RUNNING EPS SECTION HEADER & PROGRESS LINES ---"
  grep -n "=== RUNNING EPS=" "${LOG_FILE}" || echo "RUNNING EPS not reached yet"
  echo "Progress item lines (line numbers and item index only, r-values omitted):"
  grep -n -E "\[[0-9]{4}/[0-9]{4}\]" "${LOG_FILE}" | awk -F: '{print $1 ":" substr($2, 1, 13)}' | tail -n 25 || true
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

echo "--- 4.2. P6 GENERATION JSONL FILES LISTING & LINE COUNTS ---"
ls -la --time-style=full-iso results_s7/p6_gen_*.jsonl 2>/dev/null || echo "No results_s7/p6_gen_*.jsonl files found"
wc -l results_s7/p6_gen_*.jsonl 2>/dev/null || true

echo "=== 5. ZSTAR_TRAIN1000_P4.PT 325 KEYS & SHAPE CHECK (CPU, NICE 19) ==="
export MALLOC_ARENA_MAX=1
export MALLOC_TRIM_THRESHOLD_=131072
export TORCH_NUM_THREADS=4
ulimit -v 62914560

nice -n 19 /home/shoon/system1_5/.venv/bin/python - << 'EOF'
import json
import sys
import torch

cfg_path = "configs/s7_train1000_kstar_replay_325.json"
p4_path = "results_s7/zstar_train1000_P4.pt"

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
indices = cfg["indices"] if isinstance(cfg, dict) and "indices" in cfg else cfg

print(f"Loading {p4_path} on CPU with torch.no_grad()...")
with torch.no_grad():
    p4_dict = torch.load(p4_path, map_location="cpu")

missing = []
wrong_shape = []
for q in indices:
    k = f"q{q:04d}_a0.25"
    if k not in p4_dict:
        missing.append(k)
        continue
    z = p4_dict[k].get("z_star")
    if z is None or tuple(z.shape) != (64, 2560):
        wrong_shape.append((k, None if z is None else str(z.shape)))

print(f"P4 325 keys check result: total={len(indices)}, missing={len(missing)}, wrong_shape={len(wrong_shape)}")
if missing:
    print(f"Missing keys: {missing}")
if wrong_shape:
    print(f"Wrong shape keys: {wrong_shape}")

if not missing and not wrong_shape:
    print("P4 325 keys strictly present with shape (64, 2560). Check PASSED.")
else:
    print("P4 check FAILED.")
    sys.exit(1)
EOF

echo "=== TASK-0049 COLLECTION FINISHED ==="
"""

def wait_until_target(target_hour=19, target_minute=50, target_second=5):
    now = datetime.datetime.now()
    target = now.replace(hour=target_hour, minute=target_minute, second=target_second, microsecond=0)
    if now < target:
        diff = (target - now).total_seconds()
        print(f"Waiting until {target.strftime('%H:%M:%S')} KST ({diff:.1f}s remaining)...", flush=True)
        time.sleep(diff)
    print(f"Current time is now {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')} (>= 19:50:00). Proceeding.", flush=True)

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
    sftp_batch = os.path.join(root, "scratch", "upload_task49.sftp")
    out_path = os.path.join(root, "scratch", "task0049_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1. Enforce time constraint: >= 19:50 KST
    print("=== Step 0: Enforcing 19:50 KST Time Constraint ===", flush=True)
    wait_until_target(19, 50, 5)

    # 2. Step 1: SFTP Upload
    print("=== Step 1: SFTP Upload of Replay Config & scripts/p5_promote_1a_replay.py ===", flush=True)
    enforce_cooldown(log_path, 65.0)
    cmd_sftp = [sys.executable, cts_ssh, "sftp", sftp_batch, "--purpose", "task0049_upload_1a_replay"]
    res1 = subprocess.run(cmd_sftp)
    if res1.returncode != 0:
        print(f"[ERROR] SFTP upload failed with exit code {res1.returncode}")
        sys.exit(res1.returncode)
    print("SFTP upload finished successfully.", flush=True)

    # 3. Step 2: Enforce Cooldown before Step 3
    print("=== Step 2: Enforcing 65s Cooldown (R7-1 compliance) ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    # 4. Step 3: Remote Collection & Verification
    print("=== Step 3: Remote Collection & Verification Execution ===", flush=True)
    cmd_ssh = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", "task0049_collect_and_verify"]
    with open(out_path, "w", encoding="utf-8") as f:
        res2 = subprocess.run(cmd_ssh, stdout=f, stderr=subprocess.STDOUT)

    print(f"Remote execution finished with exit code {res2.returncode}. Output saved to {out_path}", flush=True)
    sys.exit(res2.returncode)

if __name__ == "__main__":
    main()

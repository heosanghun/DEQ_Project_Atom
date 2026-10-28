import os
import sys
import subprocess
import time
import datetime
import math

REMOTE_SCRIPT = r"""
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0047: SHA VERIFICATION & SERVER CPU SMOKE ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 0. SYSTEM HEALTH & P6_CHAIN INTEGRITY ==="
uptime -s
uptime
tmux ls 2>&1 || echo "tmux ls failed"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 15 raw_s7/p6_chain.log
fi

echo "=== 1. UPLOADED FILES SHA-256 VERIFICATION ==="
for f in \
  exe_scratch/__init__.py \
  exe_scratch/s7/__init__.py \
  exe_scratch/s7/soft_v2/__init__.py \
  exe_scratch/s7/soft_v2/zstar_soft_lowest.py \
  scripts/s7deq_soft_v2_run_v2.py \
  scripts/s7deq_soft_v2_analyze.py \
  scripts/s7deq_soft_v2_launch_v2.sh; do
  if [ -f "$f" ]; then
    sha256sum "$f"
    ls -la --time-style=full-iso "$f"
  else
    echo "[ERROR] Missing uploaded file: $f"
  fi
done

echo "=== 2. RUN SERVER CPU SMOKE VIA launch_v2.sh ==="
export SMOKE=1
export CUDA_VISIBLE_DEVICES=""
ulimit -v 62914560
export MALLOC_ARENA_MAX=1
export MALLOC_TRIM_THRESHOLD_=131072
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export RAYON_NUM_THREADS=4

nice -n 19 bash scripts/s7deq_soft_v2_launch_v2.sh
SMOKE_RC=$?
echo "SMOKE_RC=${SMOKE_RC}"

echo "=== 3. SMOKE ARTIFACTS IN scratch/ ==="
for f in scratch/s7deq_soft_v2_smoke_eps0.01.pt scratch/s7deq_soft_v2_smoke_eps0.01.json scratch/s7deq_soft_v2_smoke_eps0.01.md; do
  if [ -f "$f" ]; then
    sha256sum "$f"
    ls -la --time-style=full-iso "$f"
  else
    echo "Smoke artifact not found: $f"
  fi
done

echo "=== BEGIN FILE: scratch/s7deq_soft_v2_smoke_eps0.01.json ==="
if [ -f scratch/s7deq_soft_v2_smoke_eps0.01.json ]; then
  cat scratch/s7deq_soft_v2_smoke_eps0.01.json
fi
echo "=== END FILE: scratch/s7deq_soft_v2_smoke_eps0.01.json ==="

echo "=== BEGIN FILE: scratch/s7deq_soft_v2_smoke_eps0.01.md ==="
if [ -f scratch/s7deq_soft_v2_smoke_eps0.01.md ]; then
  cat scratch/s7deq_soft_v2_smoke_eps0.01.md
fi
echo "=== END FILE: scratch/s7deq_soft_v2_smoke_eps0.01.md ==="

echo "=== 4. CURRENT GPU STATUS & COMPUTE APPS ==="
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
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
    sftp_batch = os.path.join(root, "scratch", "upload_task47.sftp")
    out_path = os.path.join(root, "scratch", "task0047_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Step 1: SFTP Upload
    print("=== Step 1: SFTP Upload of Soft V2 Scripts & __init__.py ===", flush=True)
    enforce_cooldown(log_path, 65.0)
    cmd_sftp = [sys.executable, cts_ssh, "sftp", sftp_batch, "--purpose", "task0047_upload_soft_v2_scripts"]
    res1 = subprocess.run(cmd_sftp)
    if res1.returncode != 0:
        print(f"[ERROR] SFTP upload failed with exit code {res1.returncode}")
        sys.exit(res1.returncode)
    print("SFTP upload finished successfully.", flush=True)

    # Step 2: Enforce Cooldown before Step 3
    print("=== Step 2: Enforcing 65s Cooldown (R7-1 compliance) ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    # Step 3: Remote SHA Verification & CPU SMOKE
    print("=== Step 3: Remote SHA Verification & CPU SMOKE Execution ===", flush=True)
    cmd_ssh = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", "task0047_verify_and_cpu_smoke"]
    with open(out_path, "w", encoding="utf-8") as f:
        res2 = subprocess.run(cmd_ssh, stdout=f, stderr=subprocess.STDOUT)

    print(f"Remote execution finished with exit code {res2.returncode}. Output saved to {out_path}", flush=True)
    sys.exit(res2.returncode)

if __name__ == "__main__":
    main()

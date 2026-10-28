import os
import sys
import subprocess
import time
import datetime
import math

REMOTE_SCRIPT = r"""
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0045: GPU 0 ANCHOR COMPARISON & P6_CHAIN PROGRESS INSPECTION ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 0. REBOOT STATUS & TMUX SESSIONS ==="
echo "--- uptime -s ---"
uptime -s
echo "--- uptime ---"
uptime
echo "--- tmux ls ---"
tmux ls 2>&1 || echo "tmux ls failed"

echo "=== 1. P6_CHAIN LOG (LAST 25 LINES) ==="
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 25 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log does not exist"
fi

echo "=== 2. ANCHOR 20 COMPARISON & FILES ==="
echo "--- SHA-256 and attributes of anchor20_gpu0.pt, anchor20_gpu1.pt, anchor20_gpu2.pt ---"
for f in results_s7/anchor20_gpu0.pt results_s7/anchor20_gpu1.pt results_s7/anchor20_gpu2.pt; do
  if [ -f "$f" ]; then
    sha256sum "$f"
    ls -la --time-style=full-iso "$f"
  else
    echo "$f does not exist"
  fi
done

echo "--- raw_s7/anchor20_gpu0_compare.log last 30 lines ---"
if [ -f raw_s7/anchor20_gpu0_compare.log ]; then
  tail -n 30 raw_s7/anchor20_gpu0_compare.log
else
  echo "raw_s7/anchor20_gpu0_compare.log does not exist"
fi

echo "--- results_s7/anchor20_gpu0_vs_12.json ---"
if [ -f results_s7/anchor20_gpu0_vs_12.json ]; then
  cat results_s7/anchor20_gpu0_vs_12.json
else
  echo "results_s7/anchor20_gpu0_vs_12.json does not exist. Checking other anchor json files in results_s7:"
  ls -la results_s7/*anchor* 2>&1 || true
fi

echo "=== 3. ANCHOR 20 LOG PROGRESS (IF ONGOING) ==="
if [ -f raw_s7/anchor20_gpu0.log ]; then
  echo -n "TOTAL_LOG_LINES: "
  wc -l < raw_s7/anchor20_gpu0.log
  echo -n "LAST_ITEM_HEADER: "
  grep -oE '^\[[0-9]{4}/0020\]' raw_s7/anchor20_gpu0.log | tail -1 || echo "None"
  echo "--- Tail 10 lines of anchor20_gpu0.log ---"
  tail -n 10 raw_s7/anchor20_gpu0.log
else
  echo "raw_s7/anchor20_gpu0.log does not exist"
fi

echo "=== 4. P6_CHAIN STAGE 1 (ZSRC 600) & NEW FILES ==="
if [ -f raw_s7/P5_DONE.flag ]; then
  echo "--- find raw_s7 results_s7 scratch -newer raw_s7/P5_DONE.flag -type f ---"
  find raw_s7 results_s7 scratch -newer raw_s7/P5_DONE.flag -type f 2>/dev/null | xargs ls -la --time-style=full-iso 2>/dev/null || echo "NO_NEWER_FILES"
else
  echo "P5_DONE.flag not found"
fi

echo "--- Check ZSRC600 files & flags ---"
ls -la raw_s7/*zsrc* results_s7/*zsrc* 2>&1 || true
if [ -f raw_s7/zsrc_heldout600.log ]; then
  echo "--- tail 15 raw_s7/zsrc_heldout600.log ---"
  tail -n 15 raw_s7/zsrc_heldout600.log
fi

echo "=== 5. GPU 0~7 HARDWARE & COMPUTE APPS ==="
echo "--- nvidia-smi --query-gpu ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv

echo "--- nvidia-smi --query-compute-apps ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

echo "--- nvidia-smi -q -i 1 (GPU 1 diagnostics for Task 3) ---"
nvidia-smi -q -i 1 | grep -i -E "pending|retired|remapped|xid|reset|ecc" | head -20 || echo "NVIDIA_SMI_Q_FAILED"
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    log_path = os.path.join(root, "shared", "ssh_log.csv")
    out_path = os.path.join(root, "scratch", "task0045_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) > 1:
            last_ts = lines[-1].split(",")[0]
            try:
                last_dt = datetime.datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                diff = (datetime.datetime.now() - last_dt).total_seconds()
                if diff < 65.0:
                    wait_s = math.ceil(65.0 - diff)
                    print(f"Waiting {wait_s}s to satisfy R7-1 cooldown...")
                    time.sleep(wait_s)
            except Exception as e:
                pass

    purpose = "task0045_anchor_compare_and_p6_progress"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]

    print("Executing TASK-0045 bundle query via cts_ssh.py...")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"Task 0045 bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

import os
import sys
import subprocess

REMOTE_SCRIPT = """
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0037 EXECUTION: DECODER E, P5 RESUME & SYSTEM STATUS ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 0. REBOOT STATUS (FIRST PRIORITY) ==="
echo "--- uptime -s ---"
uptime -s
echo "--- uptime ---"
uptime
echo "--- last -x reboot | head -3 ---"
last -x reboot | head -3
echo "--- tmux ls ---"
tmux ls 2>&1 || echo "tmux ls failed"

echo "=== 1. DECODER E STATUS & CHECKPOINT ==="
echo "--- Checkpoint existence and details ---"
if [ -f results_s7/decoder_E_v36.pt ]; then
  ls -lh results_s7/decoder_E_v36.pt
  ls -l results_s7/decoder_E_v36.pt
  sha256sum results_s7/decoder_E_v36.pt
  echo "--- Running check_p6_decoder_version.sh results_s7/decoder_E_v36.pt ---"
  bash scripts/check_p6_decoder_version.sh results_s7/decoder_E_v36.pt
  echo "CHECK_VERSION_RC=$?"
else
  echo "RESULTS: results_s7/decoder_E_v36.pt NOT FOUND"
fi

echo "--- train_decoder_E_v36.log: final checkpoint save line ---"
grep -C 3 "Saved final model to results_s7/decoder_E_v36.pt" raw_s7/train_decoder_E_v36.log 2>/dev/null || echo "[NOT FOUND] Saved final model line not found in log"

echo "--- train_decoder_E_v36.log: last 5 step lines ---"
grep "\\[Step " raw_s7/train_decoder_E_v36.log 2>/dev/null | tail -n 5 || echo "No Step lines found in train_decoder_E_v36.log"

echo "--- train_decoder_E_v36.log: Training completed line ---"
grep -C 2 "Training completed" raw_s7/train_decoder_E_v36.log 2>/dev/null || echo "[NOT FOUND] Training completed line not found"

echo "--- train_decoder_E_v36.log: tail 15 lines ---"
tail -n 15 raw_s7/train_decoder_E_v36.log 2>/dev/null || echo "raw_s7/train_decoder_E_v36.log not found"

echo "=== 2. WAITER STATUS & FLAGS ==="
echo "--- raw_s7/p5_resume_waiter.log: tail 6 lines ---"
if [ -f raw_s7/p5_resume_waiter.log ]; then
  tail -n 6 raw_s7/p5_resume_waiter.log
else
  echo "raw_s7/p5_resume_waiter.log does not exist"
fi

echo "--- P4PRIME_ABNORMAL.flag and P5_RESUME_CANCEL.flag check ---"
ls -la raw_s7/P4PRIME_ABNORMAL.flag raw_s7/P5_RESUME_CANCEL.flag 2>&1 || true

echo "=== 3. P5 RESUME STATUS (SEALED: VALUE INSPECTION ZERO) ==="
echo "--- raw_s7/P5_RESUMED.flag ---"
if [ -f raw_s7/P5_RESUMED.flag ]; then
  cat raw_s7/P5_RESUMED.flag
else
  echo "raw_s7/P5_RESUMED.flag does not exist"
fi

echo "--- P5 Log Path & Line Counts (Values Sealed) ---"
P5_LOG="raw_s7/heldout520_P5.log"
if [ -f "${P5_LOG}" ]; then
  echo "P5 Log file: ${P5_LOG}"
  echo -n "Last item header: "
  grep -oE '^\\[[0-9]{4}/0520\\]' "${P5_LOG}" 2>/dev/null | tail -n 1 || echo "None"
  echo -n "Item headers count: "
  grep -cE '^\\[[0-9]{4}/0520\\]' "${P5_LOG}" 2>/dev/null || echo "0"
  echo -n "Skipping count: "
  grep -c "Skipping" "${P5_LOG}" 2>/dev/null || echo "0"
else
  echo "${P5_LOG} does not exist"
fi

echo "--- P5 Output File & Key Count (CPU, nice 19, Values Sealed) ---"
P5_OUT="results_s7/zstar_s7_a0.25_heldout520_lowest.pt"
if [ -f "${P5_OUT}" ]; then
  ls -lh "${P5_OUT}"
  ls -l --time-style=full-iso "${P5_OUT}"
  nice -n 19 /home/shoon/system1_5/.venv/bin/python -c "
import torch
data = torch.load('${P5_OUT}', map_location='cpu', weights_only=False)
keys = list(data.keys()) if hasattr(data, 'keys') else []
print('P5_OUT_KEY_COUNT:', len(keys))
"
else
  echo "${P5_OUT} does not exist"
fi

echo "--- P5_DONE.flag existence ---"
ls -la raw_s7/P5_DONE.flag 2>&1 || true

echo "=== 4. P6 CHAIN LOG ==="
echo "--- raw_s7/p6_chain.log: tail 5 lines ---"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 5 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log does not exist"
fi

echo "=== 5. GPU & MEMORY STATUS ==="
echo "--- nvidia-smi --query-gpu ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu --format=csv

echo "--- nvidia-smi --query-compute-apps ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

echo "--- free -h ---"
free -h
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0037_check_decoder_e_p5_resume_and_reboot"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "task0037_out.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Task 0037 bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

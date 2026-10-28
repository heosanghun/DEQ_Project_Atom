import os
import sys
import subprocess

REMOTE_SCRIPT = """
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0034 EXECUTION: DECODER D/E STATUS & WAITER INSPECTION ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. DECODER D STATUS & CHECKPOINT ==="
echo "--- Checkpoint existence and details ---"
if [ -f results_s7/decoder_D_v36.pt ]; then
  ls -lh results_s7/decoder_D_v36.pt
  ls -l results_s7/decoder_D_v36.pt
  sha256sum results_s7/decoder_D_v36.pt
  echo "--- Running check_p6_decoder_version.sh results_s7/decoder_D_v36.pt ---"
  bash scripts/check_p6_decoder_version.sh results_s7/decoder_D_v36.pt
  echo "CHECK_VERSION_RC=$?"
else
  echo "RESULTS: results_s7/decoder_D_v36.pt NOT FOUND (Training may still be in progress)"
fi

echo "--- train_decoder_D_v36.log: final checkpoint save line ---"
grep -C 3 "Saved final model to results_s7/decoder_D_v36.pt" raw_s7/train_decoder_D_v36.log 2>/dev/null || echo "[NOT FOUND] Saved final model line not found in log"

echo "--- train_decoder_D_v36.log: last 5 step lines ---"
grep "\\[Step " raw_s7/train_decoder_D_v36.log 2>/dev/null | tail -n 5 || echo "No Step lines found in train_decoder_D_v36.log"

echo "--- train_decoder_D_v36.log: tail 15 lines ---"
tail -n 15 raw_s7/train_decoder_D_v36.log 2>/dev/null || echo "raw_s7/train_decoder_D_v36.log not found"

echo "=== 2. DECODER E STATUS ==="
echo "--- Checkpoint existence ---"
if [ -f results_s7/decoder_E_v36.pt ]; then
  ls -lh results_s7/decoder_E_v36.pt
  sha256sum results_s7/decoder_E_v36.pt
else
  echo "results_s7/decoder_E_v36.pt does not exist"
fi

if [ -f raw_s7/train_decoder_E_v36.log ]; then
  ls -lh raw_s7/train_decoder_E_v36.log
  echo "--- train_decoder_E_v36.log: head 40 lines (config block) ---"
  head -n 40 raw_s7/train_decoder_E_v36.log
  echo "--- train_decoder_E_v36.log: first 3 step lines ---"
  grep "\\[Step " raw_s7/train_decoder_E_v36.log 2>/dev/null | head -n 3 || echo "No Step lines found"
  echo "--- train_decoder_E_v36.log: last 5 step lines ---"
  grep "\\[Step " raw_s7/train_decoder_E_v36.log 2>/dev/null | tail -n 5 || echo "No Step lines found"
  echo "--- train_decoder_E_v36.log: tail 15 lines ---"
  tail -n 15 raw_s7/train_decoder_E_v36.log
else
  echo "raw_s7/train_decoder_E_v36.log does not exist"
fi

echo "=== 3. WAITER & CHAIN LOGS ==="
echo "--- raw_s7/p5_resume_waiter.log: tail 5 lines ---"
if [ -f raw_s7/p5_resume_waiter.log ]; then
  tail -n 5 raw_s7/p5_resume_waiter.log
else
  echo "raw_s7/p5_resume_waiter.log does not exist"
fi

echo "--- raw_s7/p6_chain.log: tail 5 lines ---"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 5 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log does not exist"
fi

echo "=== 4. SYSTEM PROCESSES & TMUX SESSIONS ==="
echo "--- tmux sessions ---"
tmux ls 2>&1 || echo "tmux ls failed"

echo "--- Active relevant processes ---"
ps -ef | grep -E "train_decoder|p4_prime|p5_|p6_|waiter" | grep -v grep || echo "No active matching processes"

echo "=== 5. GPU & MEMORY STATUS ==="
echo "--- nvidia-smi summary ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

echo "--- free memory (free -h) ---"
free -h

echo "--- free memory (free -m) ---"
free -m
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0034_check_decoder_d_e_and_waiter"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "task0034_out.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Task 0034 bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

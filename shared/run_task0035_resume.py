import os
import sys
import subprocess

REMOTE_SCRIPT = """
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0035 RESUMPTION: DECODER D (STEP 1500) + WAITER + P6 CHAIN ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. DEPLOYING scripts/s7_train_p4_prime_resume2_v36.sh ==="
cat << 'EOF' > scripts/s7_train_p4_prime_resume2_v36.sh
#!/usr/bin/env bash
# scripts/s7_train_p4_prime_resume2_v36.sh - Resume Serial Decoder Training Pipeline for CTS P4' v3.6 (D from 1500 + E)
# Directive: CTS P4' (PREREG_S7_D_v3_6 §2·§3·§7 / Ledger #419 / TASK-0035)
set -eo pipefail

cd /home/shoon/cts_e1_s5
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="GPU-71be98b0-6b81-9257-c622-97dc97d46480"
PYTHON="/home/shoon/system1_5/.venv/bin/python"
SCRIPT="scripts/s7_train_decoder_v36.py"
ITEMS="configs/s7_train_993.json"
ZSTAR="results_s7/zstar_train1000_P4.pt"
ZSRC="results_s7/zsrc_train1000_P4.pt"

echo "================================================================================"
echo "CTS P4' Serial Decoder Training Suite (v3.6 Pure Gemma Base, GPU 0, RESUME 2)"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname) | CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "================================================================================"

# Decoder D: lambda=0.0, input=z*, resume from step 1500
echo ""
echo "=== [1/2] Resuming Decoder D from step 1500 (lambda=0.0, input=z*) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
$PYTHON $SCRIPT \
  --zstar $ZSTAR \
  --items $ITEMS \
  --loss_lambda 0.0 \
  --max_steps 3000 \
  --batch_size 8 \
  --micro_batch_size 2 \
  --grad_accum_steps 4 \
  --lr_wproj 1e-4 \
  --lr_adapter 3e-4 \
  --weight_decay 0.01 \
  --resume results_s5/decoder_D_v36_step1500.pt \
  --out results_s7/decoder_D_v36.pt \
  2>&1 | tee -a raw_s7/train_decoder_D_v36.log
echo "=== [1/2] Decoder D Completed at $(date '+%Y-%m-%d %H:%M:%S') ==="

# Decoder E: lambda=1.0, input=z_src
echo ""
echo "=== [2/2] Starting Decoder E (lambda=1.0, input=z_src) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
$PYTHON $SCRIPT \
  --zstar $ZSRC \
  --items $ITEMS \
  --loss_lambda 1.0 \
  --max_steps 3000 \
  --batch_size 8 \
  --micro_batch_size 2 \
  --grad_accum_steps 4 \
  --lr_wproj 1e-4 \
  --lr_adapter 3e-4 \
  --weight_decay 0.01 \
  --out results_s7/decoder_E_v36.pt \
  2>&1 | tee raw_s7/train_decoder_E_v36.log
echo "=== [2/2] Decoder E Completed at $(date '+%Y-%m-%d %H:%M:%S') ==="

echo ""
echo "================================================================================"
echo "CTS P4' Serial Decoder Training Suite (v3.6) FINISHED at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
EOF

chmod +x scripts/s7_train_p4_prime_resume2_v36.sh
sha256sum scripts/s7_train_p4_prime_resume2_v36.sh

echo "=== 2. LAUNCHING TMUX SESSIONS ==="
tmux new-session -d -s s7_train_p4_prime "bash scripts/s7_train_p4_prime_resume2_v36.sh"
echo "Launched s7_train_p4_prime"

tmux new-session -d -s p5_resume_waiter "bash scripts/p5_resume_after_p4prime_waiter.sh"
echo "Launched p5_resume_waiter"

tmux new-session -d -s p6_chain "bash scripts/p6_chain_after_p5_v2.sh 2>&1 | tee -a raw_s7/p6_chain.log"
echo "Launched p6_chain"

echo "=== 3. INITIAL INITIALIZATION WAIT (15s) ==="
sleep 15

echo "--- Active tmux sessions ---"
tmux ls

echo "--- Resumption log lines from train_decoder_D_v36.log ---"
grep -C 3 "\\[RESUME\\]" raw_s7/train_decoder_D_v36.log 2>/dev/null | tail -n 10 || echo "[NOTICE] RESUME line not found yet"

echo "--- Last 5 lines of raw_s7/train_decoder_D_v36.log ---"
tail -n 5 raw_s7/train_decoder_D_v36.log

echo "--- First check row of raw_s7/p5_resume_waiter.log ---"
tail -n 3 raw_s7/p5_resume_waiter.log

echo "--- First check row of raw_s7/p6_chain.log ---"
tail -n 3 raw_s7/p6_chain.log

echo "=== 4. 5-MINUTE MONITORING INTERVAL (300s) ==="
echo "Entering 300s sleep at $(date '+%Y-%m-%d %H:%M:%S KST')..."
sleep 300
echo "Woke up from 300s sleep at $(date '+%Y-%m-%d %H:%M:%S KST')"

echo "=== 5. 5-MINUTE STATUS REPORT ==="
echo "--- GPU 0~7 status (nvidia-smi) ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

echo "--- train_decoder_D_v36.log: last 5 step lines ---"
grep "\\[Step " raw_s7/train_decoder_D_v36.log 2>/dev/null | tail -n 5 || echo "No Step lines found"

echo "--- train_decoder_D_v36.log: tail 10 lines ---"
tail -n 10 raw_s7/train_decoder_D_v36.log

echo "--- p5_resume_waiter.log: tail 3 lines ---"
tail -n 3 raw_s7/p5_resume_waiter.log

echo "--- p6_chain.log: tail 3 lines ---"
tail -n 3 raw_s7/p6_chain.log
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0035_resume_p4_waiter_chain_5min"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "task0035_resume_out.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Task 0035 resume bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

import os
import sys
import subprocess

SCRIPT_CONTENT = """#!/usr/bin/env bash
# scripts/s7_train_p4_prime_resume_v36.sh - Resume Serial Decoder Training Pipeline for CTS P4' v3.6
# Directive: CTS P4' (PREREG_S7_D_v3_6 §2·§3·§7 / Ledger #419 / TASK-0024)
set -eo pipefail

cd /home/shoon/cts_e1_s5
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
PYTHON="/home/shoon/system1_5/.venv/bin/python"
SCRIPT="scripts/s7_train_decoder_v36.py"
ITEMS="configs/s7_train_993.json"
ZSTAR="results_s7/zstar_train1000_P4.pt"
ZSRC="results_s7/zsrc_train1000_P4.pt"

echo "================================================================================"
echo "CTS P4' Serial Decoder Training Suite (v3.6 Pure Gemma Base, GPU 0, RESUME)"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname) | CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "================================================================================"

# Decoder A: lambda=1.0, input=z*, resume from step 2500
echo ""
echo "=== [1/3] Resuming Decoder A from step 2500 (lambda=1.0, input=z*) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
$PYTHON $SCRIPT \\
  --zstar $ZSTAR \\
  --items $ITEMS \\
  --loss_lambda 1.0 \\
  --max_steps 3000 \\
  --batch_size 8 \\
  --micro_batch_size 2 \\
  --grad_accum_steps 4 \\
  --lr_wproj 1e-4 \\
  --lr_adapter 3e-4 \\
  --weight_decay 0.01 \\
  --resume results_s5/decoder_A_v36_step2500.pt \\
  --out results_s7/decoder_A_v36.pt \\
  2>&1 | tee -a raw_s7/train_decoder_A_v36.log
echo "=== [1/3] Decoder A Completed at $(date '+%Y-%m-%d %H:%M:%S') ==="

# Decoder D: lambda=0.0, input=z*
echo ""
echo "=== [2/3] Starting Decoder D (lambda=0.0, input=z*) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
$PYTHON $SCRIPT \\
  --zstar $ZSTAR \\
  --items $ITEMS \\
  --loss_lambda 0.0 \\
  --max_steps 3000 \\
  --batch_size 8 \\
  --micro_batch_size 2 \\
  --grad_accum_steps 4 \\
  --lr_wproj 1e-4 \\
  --lr_adapter 3e-4 \\
  --weight_decay 0.01 \\
  --out results_s7/decoder_D_v36.pt \\
  2>&1 | tee raw_s7/train_decoder_D_v36.log
echo "=== [2/3] Decoder D Completed at $(date '+%Y-%m-%d %H:%M:%S') ==="

# Decoder E: lambda=1.0, input=z_src
echo ""
echo "=== [3/3] Starting Decoder E (lambda=1.0, input=z_src) ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
$PYTHON $SCRIPT \\
  --zstar $ZSRC \\
  --items $ITEMS \\
  --loss_lambda 1.0 \\
  --max_steps 3000 \\
  --batch_size 8 \\
  --micro_batch_size 2 \\
  --grad_accum_steps 4 \\
  --lr_wproj 1e-4 \\
  --lr_adapter 3e-4 \\
  --weight_decay 0.01 \\
  --out results_s7/decoder_E_v36.pt \\
  2>&1 | tee raw_s7/train_decoder_E_v36.log
echo "=== [3/3] Decoder E Completed at $(date '+%Y-%m-%d %H:%M:%S') ==="

echo ""
echo "================================================================================"
echo "CTS P4' Serial Decoder Training Suite (v3.6) FINISHED at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
"""

REMOTE_COMMANDS = f"""
cd /home/shoon/cts_e1_s5

echo "=== 1. DEPLOY scripts/s7_train_p4_prime_resume_v36.sh ==="
cat << 'EOF' > scripts/s7_train_p4_prime_resume_v36.sh
{SCRIPT_CONTENT.strip()}
EOF
chmod +x scripts/s7_train_p4_prime_resume_v36.sh
ls -lh scripts/s7_train_p4_prime_resume_v36.sh
sha256sum scripts/s7_train_p4_prime_resume_v36.sh

echo "=== 2. LAUNCH TMUX SESSIONS ==="
# P4' on GPU 0
tmux new-session -d -s s7_train_p4_prime "bash -c 'bash /home/shoon/cts_e1_s5/scripts/s7_train_p4_prime_resume_v36.sh >> raw_s7/train_p4_prime_suite_v36.log 2>&1'"

# P5 on GPU 1
tmux new-session -d -s p5_gpu1 "bash -c 'P5_RESUME=1 bash /home/shoon/cts_e1_s5/scripts/p5_heldout520_launch_v5.sh --device cuda:1'"

# P5b on GPU 2
tmux new-session -d -s p5b_gpu2 "bash -c 'P5B_RESUME=1 bash /home/shoon/cts_e1_s5/scripts/p5b_seed2031_launch_v2.sh --device cuda:2'"

# p6_chain v2
tmux new-session -d -s p6_chain "bash -c 'bash /home/shoon/cts_e1_s5/scripts/p6_chain_after_p5_v2.sh >> raw_s7/p6_chain.log 2>&1'"

echo "=== 3. IMMEDIATE TMUX & PROCESS STATUS ==="
tmux list-sessions
ps -ef | grep -E "s7_train_decoder|s7_zstar_lowest_run|p6_chain" | grep -v grep || true
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo "=== 4. WAITING 300 SECONDS (5 MINUTES) FOR PROGRESS 계측 ==="
sleep 300

echo "=== 5. 5-MINUTE STATUS 계측 ==="
echo "--- TMUX SESSIONS ---"
tmux list-sessions

echo "--- P4' LOG HEAD (RESUME LINE) ---"
grep -C 2 "\\[RESUME\\]" raw_s7/train_decoder_A_v36.log raw_s7/train_p4_prime_suite_v36.log 2>/dev/null | tail -n 10 || echo "no resume line yet"

echo "--- P4' LOG TAIL (LATEST STEPS) ---"
tail -n 15 raw_s7/train_decoder_A_v36.log 2>/dev/null

echo "--- P5 LOG (SKIPPING COUNT & FIRST NEW QUESTION) ---"
echo -n "Skipping count: "
grep -c "already present" raw_s7/heldout520_P5.log 2>/dev/null || echo "0"
echo "--- Last 15 lines of P5 log ---"
tail -n 15 raw_s7/heldout520_P5.log 2>/dev/null

echo "--- P5B LOG (SKIPPING COUNT & FIRST NEW QUESTION) ---"
echo -n "Skipping count: "
grep -c "already present" raw_s7/heldout520_P5b_seed2031.log 2>/dev/null || echo "0"
echo "--- Last 15 lines of P5b log ---"
tail -n 15 raw_s7/heldout520_P5b_seed2031.log 2>/dev/null

echo "--- P6_CHAIN LOG TAIL ---"
tail -n 15 raw_s7/p6_chain.log 2>/dev/null

echo "--- GPU 0~3 STATUS ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader -i 0,1,2,3
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    purpose = "task0024_deploy_resume_and_monitor_5min"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_COMMANDS, "--purpose", purpose]
    print(f"Launching resume & 5-min monitor via cts_ssh: {purpose}...")
    out_path = os.path.join(root, "scratch", "task0024_resume.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Output saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

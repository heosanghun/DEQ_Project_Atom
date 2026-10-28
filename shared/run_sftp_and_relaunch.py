import os
import sys
import time
import subprocess

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    sftp_batch = os.path.join(root, "scratch", "upload_decoder_v36.sftp")

    # Step 1: SFTP Upload
    print("--- Step 1: Uploading patched s7_train_decoder_v36.py via SFTP ---")
    res1 = subprocess.run([sys.executable, cts_ssh, "sftp", sftp_batch, "--purpose", "task0024_upload_decoder_v36_fix"])
    if res1.returncode != 0:
        print(f"SFTP failed with exit code {res1.returncode}")
        sys.exit(res1.returncode)
    print("SFTP completed successfully.")

    # Step 2: Enforce 65-second cooldown
    print("--- Step 2: Waiting 65s cooldown (R7-1 compliance) ---")
    time.sleep(65)

    # Step 3: Remote execution & 5-minute monitoring
    print("--- Step 3: Launching P4' resume and monitoring 300s on server ---")
    remote_cmd = """
cd /home/shoon/cts_e1_s5

echo "=== 1. VERIFY UPLOADED DECODER V36 ==="
ls -lh scripts/s7_train_decoder_v36.py
sha256sum scripts/s7_train_decoder_v36.py
python3 -m py_compile scripts/s7_train_decoder_v36.py && echo "Compilation: OK"

echo "=== 2. LAUNCH P4' RESUME IN TMUX ==="
tmux kill-session -t s7_train_p4_prime 2>/dev/null || true
tmux new-session -d -s s7_train_p4_prime "bash -c 'bash /home/shoon/cts_e1_s5/scripts/s7_train_p4_prime_resume_v36.sh >> raw_s7/train_p4_prime_suite_v36.log 2>&1'"
echo "Launched s7_train_p4_prime"

echo "=== 3. VERIFY P6_CHAIN IN TMUX ==="
if ! tmux has-session -t p6_chain 2>/dev/null; then
    tmux new-session -d -s p6_chain "bash -c 'bash /home/shoon/cts_e1_s5/scripts/p6_chain_after_p5_v2.sh >> raw_s7/p6_chain.log 2>&1'"
    echo "Launched p6_chain"
else
    echo "p6_chain is already running"
fi

echo "=== 4. IMMEDIATE TMUX & PROCESS STATUS ==="
tmux list-sessions
ps -ef | grep -E "s7_train_decoder|p6_chain" | grep -v grep || true
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader -i 0,1,2,3

echo "=== 5. WAITING 300 SECONDS (5 MINUTES) FOR PROGRESS 계측 ==="
sleep 300

echo "=== 6. 5-MINUTE STATUS REPORT ==="
echo "--- TMUX SESSIONS ---"
tmux list-sessions

echo "--- P4' LOG HEAD (RESUME LINE) ---"
grep -C 2 "\\[RESUME\\]" raw_s7/train_decoder_A_v36.log raw_s7/train_p4_prime_suite_v36.log 2>/dev/null | tail -n 10 || echo "no resume line"

echo "--- P4' LOG TAIL (LATEST STEPS) ---"
tail -n 15 raw_s7/train_decoder_A_v36.log 2>/dev/null

echo "--- P5 LOG STATUS ---"
echo -n "Skipping count: "
grep -c "already present" raw_s7/heldout520_P5.log 2>/dev/null || echo "0"
echo "--- Last 10 lines of P5 log ---"
tail -n 10 raw_s7/heldout520_P5.log 2>/dev/null

echo "--- P5B LOG STATUS ---"
echo -n "Skipping count: "
grep -c "already present" raw_s7/heldout520_P5b_seed2031.log 2>/dev/null || echo "0"
echo "--- Last 10 lines of P5b log ---"
tail -n 10 raw_s7/heldout520_P5b_seed2031.log 2>/dev/null

echo "--- P6_CHAIN LOG TAIL ---"
tail -n 10 raw_s7/p6_chain.log 2>/dev/null

echo "--- GPU 0~3 STATUS ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader -i 0,1,2,3
"""
    out_path = os.path.join(root, "scratch", "task0024_relaunch_out.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        res2 = subprocess.run([sys.executable, cts_ssh, "ssh", remote_cmd, "--purpose", "task0024_relaunch_p4_and_monitor"], stdout=f, stderr=subprocess.STDOUT)
    print(f"Relaunch output saved to {out_path} with exit code {res2.returncode}")
    sys.exit(res2.returncode)

if __name__ == "__main__":
    main()

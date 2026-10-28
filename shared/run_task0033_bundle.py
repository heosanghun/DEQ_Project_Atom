import os
import sys
import time
import subprocess

REMOTE_SCRIPT = """
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0033: VERIFICATION, SMOKE TEST AND WAITER LAUNCH ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. VERIFY UPLOADED SCRIPTS SHA-256 ==="
chmod +x scripts/p5_heldout520_launch_v8.sh scripts/p5_resume_after_p4prime_waiter.sh
sha256sum scripts/p5_heldout520_launch_v8.sh scripts/p5_resume_after_p4prime_waiter.sh scripts/p5_heldout520_launch_v7.sh

echo "=== 2. RUN SMOKE TEST ON p5_heldout520_launch_v8.sh (CPU, Q#14) ==="
SMOKE=1 bash scripts/p5_heldout520_launch_v8.sh
SMOKE_RC=$?
echo "SMOKE_RC=${SMOKE_RC}"
echo "--- Smoke log tail ---"
tail -n 15 scratch/p5_smoke.log 2>/dev/null || echo "Smoke log not found"
echo "--- Smoke output file ---"
ls -lh scratch/p5_smoke.pt 2>/dev/null || echo "Smoke pt not found"
sha256sum scratch/p5_smoke.pt 2>/dev/null || true

echo "=== 3. LAUNCH WAITER IN TMUX (p5_resume_waiter) ==="
tmux kill-session -t p5_resume_waiter 2>/dev/null || true
tmux new-session -d -s p5_resume_waiter "bash scripts/p5_resume_after_p4prime_waiter.sh"
sleep 5

echo "=== 4. INSPECT FIRST CHECK ROW OF WAITER ==="
cat raw_s7/p5_resume_waiter.log 2>/dev/null || echo "Waiter log not found"

echo "=== 5. CURRENT DECODER D PROGRESS ==="
ls -lh raw_s7/train_decoder_D_v36.log
grep "\\[Step " raw_s7/train_decoder_D_v36.log 2>/dev/null | tail -n 5 || echo "No step lines found"

echo "=== 6. SYSTEM STATUS & ACTIVE TMUX SESSIONS ==="
tmux list-sessions
ps -ef | grep -E "s7_train_decoder|train_p4_prime|p5_resume_waiter|p6_chain" | grep -v grep || true
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo "=== 7. P6_CHAIN V2 CONDITION CHECK ==="
grep -n -C 5 "P5_DONE.flag" scripts/p6_chain_after_p5_v2.sh
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    sftp_batch = os.path.join(root, "scratch", "upload_task33.sftp")

    # Step 1: SFTP Upload
    print("--- Step 1: Uploading v8 launcher and waiter script via SFTP ---")
    res1 = subprocess.run([sys.executable, cts_ssh, "sftp", sftp_batch, "--purpose", "task0033_upload_waiter_and_v8"])
    if res1.returncode != 0:
        print(f"SFTP failed with exit code {res1.returncode}")
        sys.exit(res1.returncode)
    print("SFTP upload finished cleanly.")

    # Step 2: Enforce 65-second cooldown
    print("--- Step 2: Waiting 65s cooldown (R7-1 compliance) ---")
    time.sleep(65)

    # Step 3: Remote execution
    print("--- Step 3: Remote smoke test and waiter launch ---")
    out_path = os.path.join(root, "scratch", "task0033_out.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        res2 = subprocess.run([sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", "task0033_smoke_and_launch_waiter"], stdout=f, stderr=subprocess.STDOUT)
    print(f"Task 0033 remote execution saved to {out_path} with exit code {res2.returncode}")
    sys.exit(res2.returncode)

if __name__ == "__main__":
    main()

import os
import sys
import time
import subprocess

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    sftp_batch = os.path.join(root, "scratch", "upload_task25.sftp")

    # Step 1: SFTP Upload
    print("--- Step 1: Uploading p5b_v3 and p5_v6 via SFTP ---")
    res1 = subprocess.run([sys.executable, cts_ssh, "sftp", sftp_batch, "--purpose", "task0025_upload_scripts"])
    if res1.returncode != 0:
        print(f"SFTP failed with exit code {res1.returncode}")
        sys.exit(res1.returncode)
    print("SFTP completed successfully.")

    # Step 2: Enforce 65-second cooldown
    print("--- Step 2: Waiting 65s cooldown (R7-1 compliance) ---")
    time.sleep(65)

    # Step 3: Remote execution & 5-minute monitoring
    print("--- Step 3: Remote search, smoke, P5b resume and 5-min monitor ---")
    remote_cmd = """
cd /home/shoon/cts_e1_s5

echo "=== 1. SEARCH FOR ORIGINAL DECODER V36 (05bb5fa6) ==="
mkdir -p scripts/archive
ORIG_FOUND=""
for p in $(find /home/shoon/cts_e1_s5 /home/shoon/cts_e1 /home/shoon/checkpoints /home/shoon/system1_5 -maxdepth 4 -type f \\( -name "*decoder_v36*" -o -name "*train_decoder*" -o -name "*.bak" -o -name "*orig*" \\) 2>/dev/null); do
    sha=$(sha256sum "$p" 2>/dev/null | awk '{print $1}')
    if [[ "$sha" =~ ^05bb5fa6 ]]; then
        echo "FOUND MATCH: $p ($sha)"
        ORIG_FOUND="$p"
        cp "$p" scripts/archive/s7_train_decoder_v36_orig_05bb5fa6.py
        break
    elif [[ "$p" =~ s7_train_decoder_v36 ]]; then
        echo "Found related file: $p ($sha)"
    fi
done

if [[ -n "$ORIG_FOUND" ]]; then
    echo "Copied $ORIG_FOUND to scripts/archive/s7_train_decoder_v36_orig_05bb5fa6.py"
    sha256sum scripts/archive/s7_train_decoder_v36_orig_05bb5fa6.py
else
    echo "ORIGINAL 05bb5fa6 NOT FOUND on server search paths."
fi

# Check git log if present
git log -n 5 --oneline 2>/dev/null || echo "git log not available"

echo "=== 2. VERIFY AND PERM SCRIPTS ==="
chmod +x scripts/p5b_seed2031_launch_v3.sh scripts/p5_heldout520_launch_v6.sh
sha256sum scripts/p5b_seed2031_launch_v3.sh scripts/p5_heldout520_launch_v6.sh

echo "=== 3. SMOKE TEST P5 V6 (CPU ONLY, DO NOT RUN PRODUCTION) ==="
SMOKE=1 bash scripts/p5_heldout520_launch_v6.sh || true
echo "P5 v6 smoke exit code: $?"
cat scratch/p5_smoke.log 2>/dev/null | tail -n 10 || true

echo "=== 4. SMOKE TEST P5B V3 (CPU ONLY) ==="
SMOKE=1 bash scripts/p5b_seed2031_launch_v3.sh || true
echo "P5b v3 smoke exit code: $?"
cat scratch/p5b_smoke.log 2>/dev/null | tail -n 10 || true

echo "=== 5. LAUNCH P5B ON GPU 2 IN TMUX ==="
tmux kill-session -t p5b_gpu2 2>/dev/null || true
tmux new-session -d -s p5b_gpu2 "bash -c 'P5B_RESUME=1 bash /home/shoon/cts_e1_s5/scripts/p5b_seed2031_launch_v3.sh --device cuda:2'"
echo "Launched p5b_gpu2 in tmux with P5B_RESUME=1 on cuda:2"

echo "=== 6. IMMEDIATE PROCESS AND GPU STATUS ==="
tmux list-sessions
ps -ef | grep -E "s7_train_decoder|s7_zstar_lowest_run|p6_chain" | grep -v grep || true
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader -i 0,1,2,3

echo "=== 7. GPU 1 DETAILED STATUS (TASK-0025 §3) ==="
nvidia-smi -q -i 1 | grep -i -E "throttle|temperature|power|retired|remapped|ecc|reset" -A 4 || true

echo "=== 8. PRE-REBOOT P5 LOG DURATION ANALYSIS (TASK-0025 §3) ==="
python3 -c "
import re
try:
    lines = open('raw_s7/heldout520_P5.log', 'r', errors='ignore').readlines()
    durations = []
    for line in lines:
        m = re.search(r'\\(([0-9.]+)s\\)', line)
        if m:
            durations.append(float(m.group(1)))
    if len(durations) >= 40:
        first20 = durations[:20]
        last20 = durations[-20:]
        print(f'Total measured items: {len(durations)}')
        print(f'First 20 items: avg={sum(first20)/20:.2f}s, min={min(first20):.2f}s, max={max(first20):.2f}s')
        print(f'Last 20 items: avg={sum(last20)/20:.2f}s, min={min(last20):.2f}s, max={max(last20):.2f}s')
    else:
        print(f'Durations parsed: {len(durations)}')
except Exception as e:
    print(f'Error reading P5 log: {e}')
"

echo "=== 9. WAITING 300 SECONDS (5 MINUTES) FOR P5B 계측 ==="
sleep 300

echo "=== 10. 5-MINUTE STATUS 계측 ==="
echo "--- TMUX SESSIONS ---"
tmux list-sessions

echo "--- P5B RESUMED FLAG ---"
cat raw_s7/P5B_RESUMED.flag 2>/dev/null || echo "no P5B_RESUMED.flag"

echo "--- P5B LOG TAIL ---"
tail -n 25 raw_s7/heldout520_P5b_seed2031.log 2>/dev/null

echo "--- GPU 0~3 STATUS ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader -i 0,1,2,3
"""
    out_path = os.path.join(root, "scratch", "task0025_out.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        res2 = subprocess.run([sys.executable, cts_ssh, "ssh", remote_cmd, "--purpose", "task0025_p5b_resume_and_monitor"], stdout=f, stderr=subprocess.STDOUT)
    print(f"TASK-0025 output saved to {out_path} with exit code {res2.returncode}")
    sys.exit(res2.returncode)

if __name__ == "__main__":
    main()

import os
import sys
import subprocess

REMOTE_SCRIPT = """
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0038 EXECUTION: DECODER E FINAL, P5 RESUME & SYSTEM STATUS ==="
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
grep -C 2 "Saved final model to results_s7/decoder_E_v36.pt" raw_s7/train_decoder_E_v36.log 2>/dev/null || echo "[NOT FOUND] Saved final model line not found in log"

echo "--- train_decoder_E_v36.log: last 5 step lines ---"
grep "\\[Step " raw_s7/train_decoder_E_v36.log 2>/dev/null | tail -n 5 || echo "No Step lines found in train_decoder_E_v36.log"

echo "--- train_decoder_E_v36.log: Training completed line ---"
grep -C 2 "Training completed" raw_s7/train_decoder_E_v36.log 2>/dev/null || echo "[NOT FOUND] Training completed line not found"

echo "=== 2. WAITER STATUS & FLAGS ==="
echo "--- raw_s7/p5_resume_waiter.log: tail 8 lines ---"
if [ -f raw_s7/p5_resume_waiter.log ]; then
  tail -n 8 raw_s7/p5_resume_waiter.log
else
  echo "raw_s7/p5_resume_waiter.log does not exist"
fi

echo "--- P4PRIME_ABNORMAL.flag and P5_RESUME_CANCEL.flag check ---"
ls -la raw_s7/P4PRIME_ABNORMAL.flag raw_s7/P5_RESUME_CANCEL.flag 2>&1 || true

echo "--- tmux p5_resume_waiter session status ---"
tmux has-session -t p5_resume_waiter 2>&1
echo "WAITER_TMUX_RC=$?"

echo "=== 3. P5 RESUME STATUS (SEALED: VALUE INSPECTION ZERO) ==="
echo "--- raw_s7/P5_RESUMED.flag ---"
if [ -f raw_s7/P5_RESUMED.flag ]; then
  cat raw_s7/P5_RESUMED.flag
else
  echo "raw_s7/P5_RESUMED.flag does not exist"
fi

echo "--- P5 Runner Process & SHA ---"
ps -eo pid,lstart,cmd | grep -E "p5_heldout|s7_zstar" | grep -v grep || echo "NO_P5_PROCESS"
sha256sum scripts/p5_heldout520_launch_v8.sh 2>&1 || true

echo "--- P5 Process CUDA_VISIBLE_DEVICES ---"
P5_PIDS=$(pgrep -f "s7_zstar_lowest_run.py" || true)
if [ -n "$P5_PIDS" ]; then
  for p in $P5_PIDS; do
    echo "PID: $p"
    cat /proc/$p/environ 2>/dev/null | tr '\\0' '\\n' | grep CUDA_VISIBLE || true
  done
else
  echo "NO_ZSTAR_PIDS"
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
  stat -c "MTIME: %y (%Y)" "${P5_OUT}"
  nice -n 19 /home/shoon/system1_5/.venv/bin/python -c "
import torch, os, time, datetime, re

pt = torch.load('${P5_OUT}', map_location='cpu', weights_only=False)
keys = list(pt.keys()) if hasattr(pt, 'keys') else []
print('P5_OUT_KEY_COUNT:', len(keys))

# Sealed Timing & ETA calculation (No value inspection)
resumed_flag = 'raw_s7/P5_RESUMED.flag'
p5_log = 'raw_s7/heldout520_P5.log'
resume_ts = None
init_keys = 150
if os.path.exists(resumed_flag):
    with open(resumed_flag, 'r') as f:
        for line in f:
            if line.startswith('resumed_at:'):
                ts_str = line.split('resumed_at:')[1].strip()
                resume_ts = datetime.datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            elif line.startswith('key_count:'):
                try:
                    init_keys = int(line.split('key_count:')[1].strip())
                except:
                    pass

cur_header_count = 0
if os.path.exists(p5_log):
    with open(p5_log, 'r', errors='ignore') as f:
        for line in f:
            if re.match(r'^\\[[0-9]{4}/0520\\]', line):
                cur_header_count += 1

now_ts = datetime.datetime.now()
print('NOW_TIMESTAMP:', now_ts.strftime('%Y-%m-%d %H:%M:%S'))
if resume_ts:
    elapsed_sec = (now_ts - resume_ts).total_seconds()
    new_items = max(0, cur_header_count - init_keys)
    print('RESUME_TIMESTAMP:', resume_ts.strftime('%Y-%m-%d %H:%M:%S'))
    print('INITIAL_KEYS:', init_keys)
    print('CURRENT_HEADER_COUNT:', cur_header_count)
    print('NEW_ITEMS_PROCESSED:', new_items)
    print('ELAPSED_SECONDS:', f'{elapsed_sec:.1f}')
    if new_items > 0:
        sec_per_item = elapsed_sec / new_items
        rem_items = max(0, 520 - cur_header_count)
        rem_sec = rem_items * sec_per_item
        eta_ts = now_ts + datetime.timedelta(seconds=rem_sec)
        print('SEC_PER_ITEM:', f'{sec_per_item:.2f}')
        print('REMAINING_ITEMS:', rem_items)
        print('PROJECTED_ETA_KST:', eta_ts.strftime('%Y-%m-%d %H:%M:%S'))
    else:
        print('SEC_PER_ITEM: N/A (0 new items yet)')
"
else
  echo "${P5_OUT} does not exist"
fi

echo "--- P5_DONE.flag existence ---"
ls -la raw_s7/P5_DONE.flag 2>&1 || true

echo "=== 4. P6 CHAIN LOG ==="
echo "--- raw_s7/p6_chain.log: tail 4 lines ---"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 4 raw_s7/p6_chain.log
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
    purpose = "task0038_check_decoder_e_final_and_p5_resume"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "task0038_out.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Executing TASK-0038 bundle query via cts_ssh.py...")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Task 0038 bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

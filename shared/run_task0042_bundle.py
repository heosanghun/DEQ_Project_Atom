import os
import sys
import subprocess
import time
import datetime
import math

REMOTE_SCRIPT = r"""
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0042: P5b HEADERS RETRIEVAL & P5 PROGRESS INSPECTION ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 0. REBOOT STATUS ==="
echo "--- uptime -s ---"
uptime -s
echo "--- uptime ---"
uptime
echo "--- tmux ls ---"
tmux ls 2>&1 || echo "tmux ls failed"

echo "=== 1. P5B LOG HEADERS & LAUNCH SCRIPT (TASK-0042 §2, §3) ==="
echo "--- grep -a -n -E '^Command:|S7 Lowest-Residual DEQ Run|^Device:' raw_s7/heldout520_P5b_seed2031.log | head -6 ---"
grep -a -n -E "^Command:|S7 Lowest-Residual DEQ Run|^Device:" raw_s7/heldout520_P5b_seed2031.log 2>/dev/null | head -6 || echo "GREP_FAILED"

echo "--- SHA-256 of scripts/p5b_seed2031_launch_v4.sh ---"
sha256sum scripts/p5b_seed2031_launch_v4.sh 2>/dev/null || echo "SHA_FAILED"

echo "--- OUT_PT and surrounding lines (128-138) in scripts/p5b_seed2031_launch_v4.sh ---"
sed -n '128,138p' scripts/p5b_seed2031_launch_v4.sh 2>/dev/null || echo "SED_FAILED"

echo "--- P5b output file existence check (results_s7/zstar_s7_a0.25_heldout520_seed2031.pt) ---"
ls -la results_s7/zstar_s7_a0.25_heldout520_seed2031.pt 2>&1 || true
echo "--- All files matching seed2031 in results_s7 ---"
ls -la results_s7/*seed2031* 2>&1 || true

echo "=== 2. P5 PROGRESS & SPEED (VALUES SEALED) (TASK-0042 §5) ==="
P5_LOG="raw_s7/heldout520_P5.log"
echo -n "MEASURED_AT: "
date '+%Y-%m-%d %H:%M:%S KST'
echo -n "LAST_HEADER: "
grep -oE '^\[[0-9]{4}/0520\]' "${P5_LOG}" 2>/dev/null | tail -1 || echo "None"
echo -n "TOTAL_HEADERS_COUNT: "
grep -cE '^\[[0-9]{4}/0520\]' "${P5_LOG}" 2>/dev/null || echo "0"
echo -n "SKIPPING_COUNT: "
grep -c "Skipping" "${P5_LOG}" 2>/dev/null || echo "0"

P5_OUT="results_s7/zstar_s7_a0.25_heldout520_lowest.pt"
if [ -f "${P5_OUT}" ]; then
  ls -lh "${P5_OUT}"
  ls -l --time-style=full-iso "${P5_OUT}"
  stat -c "MTIME: %y (%Y)" "${P5_OUT}"
fi

echo "--- Python Speed Calculation (CPU, nice 19, Values Sealed) ---"
nice -n 19 /home/shoon/system1_5/.venv/bin/python -c "
import os, datetime, re, torch

p5_log = 'raw_s7/heldout520_P5.log'
p5_out = 'results_s7/zstar_s7_a0.25_heldout520_lowest.pt'
resumed_flag = 'raw_s7/P5_RESUMED.flag'

if os.path.exists(p5_out):
    pt = torch.load(p5_out, map_location='cpu', weights_only=False)
    keys = list(pt.keys()) if hasattr(pt, 'keys') else []
    print('P5_OUT_KEY_COUNT:', len(keys))
else:
    print('P5_OUT_KEY_COUNT: 0')

resume_ts = datetime.datetime.strptime('2026-10-06 10:01:49', '%Y-%m-%d %H:%M:%S')
if os.path.exists(resumed_flag):
    with open(resumed_flag, 'r') as f:
        for line in f:
            if line.startswith('resumed_at:'):
                ts_str = line.split('resumed_at:')[1].strip()
                resume_ts = datetime.datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')

now_ts = datetime.datetime.now()
total_elapsed_sec = (now_ts - resume_ts).total_seconds()

computed_items = []
last_item_idx = 150
if os.path.exists(p5_log):
    with open(p5_log, 'r', errors='ignore') as f:
        for line in f:
            m = re.match(r'^\[([0-9]{4})/0520\]', line)
            if m:
                idx = int(m.group(1))
                if 'Skipping' not in line:
                    m_sec = re.search(r'\(([0-9]+\.[0-9]+)s\)', line)
                    dur = float(m_sec.group(1)) if m_sec else None
                    computed_items.append((idx, dur))
                    last_item_idx = max(last_item_idx, idx)

new_items_since_resume = [item for item in computed_items if item[0] > 150]
new_items_count = len(new_items_since_resume)

print('RESUME_TIMESTAMP:', resume_ts.strftime('%Y-%m-%d %H:%M:%S'))
print('NOW_TIMESTAMP:', now_ts.strftime('%Y-%m-%d %H:%M:%S'))
print('TOTAL_ELAPSED_SEC:', f'{total_elapsed_sec:.1f}')
print('LAST_COMPLETED_ITEM_IDX:', last_item_idx)
print('NEW_ITEMS_COUNT_SINCE_RESUME:', new_items_count)

rem_items = max(0, 520 - last_item_idx)
print('REMAINING_ITEMS:', rem_items)

if new_items_count > 0:
    overall_sec_per_item = total_elapsed_sec / new_items_count
    rem_sec_overall = rem_items * overall_sec_per_item
    eta_overall = now_ts + datetime.timedelta(seconds=rem_sec_overall)
    print('OVERALL_SEC_PER_ITEM:', f'{overall_sec_per_item:.2f}')
    print('ETA_OVERALL_KST:', eta_overall.strftime('%Y-%m-%d %H:%M:%S'))
else:
    print('OVERALL_SEC_PER_ITEM: N/A')

# Recent 60 minutes calculation (3600 seconds window)
recent_items_60m = 0
acc_time_60m = 0.0
for idx, dur in reversed(new_items_since_resume):
    if dur is not None:
        if acc_time_60m + dur <= 3600.0 or recent_items_60m == 0:
            acc_time_60m += dur
            recent_items_60m += 1
        else:
            break

print('RECENT_60MIN_ITEMS_COUNT:', recent_items_60m)
print('RECENT_60MIN_ACCUMULATED_SEC:', f'{acc_time_60m:.1f}')
if recent_items_60m > 0 and acc_time_60m > 0:
    recent_sec_per_item = acc_time_60m / recent_items_60m
    rem_sec_recent = rem_items * recent_sec_per_item
    eta_recent = now_ts + datetime.timedelta(seconds=rem_sec_recent)
    print('RECENT_60MIN_SEC_PER_ITEM:', f'{recent_sec_per_item:.2f}')
    print('ETA_RECENT_60MIN_KST:', eta_recent.strftime('%Y-%m-%d %H:%M:%S'))
else:
    print('RECENT_60MIN_SEC_PER_ITEM: N/A')
"

echo "=== 3. GPU HARDWARE & PROCESSES ==="
echo "--- nvidia-smi --query-gpu ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv

echo "--- nvidia-smi --query-compute-apps ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

echo "--- P5_DONE.flag existence ---"
ls -la raw_s7/P5_DONE.flag 2>&1 || true

echo "--- raw_s7/p6_chain.log: tail 1 line ---"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 1 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log does not exist"
fi
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    log_path = os.path.join(root, "shared", "ssh_log.csv")
    out_path = os.path.join(root, "scratch", "task0042_out.txt")

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

    purpose = "task0042_p5b_headers_and_p5_progress"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]

    print(f"Executing TASK-0042 bundle query via cts_ssh.py...")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"Task 0042 bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

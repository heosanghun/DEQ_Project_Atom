import os
import sys
import subprocess
import time
import datetime
import math

REMOTE_SCRIPT = r"""
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0044: P5 DONE CHECK & P6_CHAIN STAGE 0 START INSPECTION ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 0. REBOOT STATUS & TMUX SESSIONS ==="
echo "--- uptime -s ---"
uptime -s
echo "--- uptime ---"
uptime
echo "--- tmux ls ---"
tmux ls 2>&1 || echo "tmux ls failed"

echo "=== 1. P5_DONE.FLAG INSPECTION ==="
if [ -f raw_s7/P5_DONE.flag ]; then
  echo "P5_DONE_FLAG_EXISTS: YES"
  ls -la --time-style=full-iso raw_s7/P5_DONE.flag
  echo "--- P5_DONE.flag full content ---"
  cat raw_s7/P5_DONE.flag
else
  echo "P5_DONE_FLAG_EXISTS: NO"
fi

echo "=== 2. P5 OUTPUT FILE & EXPECTED KEYS CHECK ==="
P5_OUT="results_s7/zstar_s7_a0.25_heldout520_lowest.pt"
if [ -f "${P5_OUT}" ]; then
  ls -la --time-style=full-iso "${P5_OUT}"
  stat -c "SIZE_BYTES: %s, MTIME: %y (%Y)" "${P5_OUT}"
  echo -n "SHA256: "
  sha256sum "${P5_OUT}" | awk '{print $1}'
  echo "--- Python Inspection of P5 Output Keys vs Expected 520 (CPU, nice 19, Values Sealed) ---"
  nice -n 19 /home/shoon/system1_5/.venv/bin/python -c "
import os, json, re, torch

p = 'results_s7/zstar_s7_a0.25_heldout520_lowest.pt'
items_json = 'configs/s7_new_heldout_520.json'

pt = torch.load(p, map_location='cpu', weights_only=False)
keys = list(pt.keys()) if hasattr(pt, 'keys') else []
total_keys = len(keys)

pattern = re.compile(r'^q\d{4}_a0\.25$')
matching_keys = sum(1 for k in keys if pattern.match(str(k)))
non_matching_keys = total_keys - matching_keys

with open(items_json, 'r') as f:
    items_data = json.load(f)

indices = items_data['indices'] if isinstance(items_data, dict) and 'indices' in items_data else items_data
expected_keys = set(f'q{idx:04d}_a0.25' for idx in indices)

actual_keys_set = set(keys)
missing_keys = expected_keys - actual_keys_set
extra_keys = actual_keys_set - expected_keys

print('TOTAL_KEYS_COUNT:', total_keys)
print('MATCHING_REGEX_COUNT:', matching_keys)
print('NON_MATCHING_REGEX_COUNT:', non_matching_keys)
print('EXPECTED_KEYS_COUNT:', len(expected_keys))
print('MISSING_KEYS_COUNT:', len(missing_keys))
print('EXTRA_KEYS_COUNT:', len(extra_keys))
"
else
  echo "P5_OUT_FILE_EXISTS: NO"
fi

echo "=== 3. P5 LOG & p5_gpu0 SESSION STATUS ==="
P5_LOG="raw_s7/heldout520_P5.log"
if [ -f "${P5_LOG}" ]; then
  echo -n "TOTAL_LOG_LINES: "
  wc -l < "${P5_LOG}"
  echo -n "LAST_HEADER: "
  grep -oE '^\[[0-9]{4}/0520\]' "${P5_LOG}" | tail -1 || echo "None"
  echo "--- Last 15 lines with result lines ([NNNN/0520]) excluded ---"
  tail -n 15 "${P5_LOG}" | grep -v -E '^\[[0-9]{4}/0520\]' || echo "ALL_FILTERED"
else
  echo "P5_LOG_EXISTS: NO"
fi

echo "=== 4. DEVICE HISTORY VERIFICATION (NON-RESULT LINES & COUNTS) ==="
echo "--- [4.1] P5: sed -n '4,161p;164,233p' raw_s7/heldout520_P5.log | grep -a -v -E '^\[[0-9]{4}/0520\]' | head -60 ---"
sed -n '4,161p;164,233p' raw_s7/heldout520_P5.log 2>/dev/null | grep -a -v -E '^\[[0-9]{4}/0520\]' | head -60 || echo "NONE"

echo "--- [4.2] P5 result lines count per segment ---"
echo -n "Segment 1 (4..161) result lines count: "
sed -n '4,161p' raw_s7/heldout520_P5.log 2>/dev/null | grep -c -E '^\[[0-9]{4}/0520\]' || echo "0"
echo -n "Segment 2 (164..233) result lines count: "
sed -n '164,233p' raw_s7/heldout520_P5.log 2>/dev/null | grep -c -E '^\[[0-9]{4}/0520\]' || echo "0"
echo -n "Segment 3 (236..end) result lines count: "
sed -n '236,$p' raw_s7/heldout520_P5.log 2>/dev/null | grep -c -E '^\[[0-9]{4}/0520\]' || echo "0"

echo "--- [4.3] P5b: sed -n '400,469p;472,541p' raw_s7/heldout520_P5b_seed2031.log | grep -a -v -E '^\[[0-9]{4}/0520\]' | head -60 ---"
sed -n '400,469p;472,541p' raw_s7/heldout520_P5b_seed2031.log 2>/dev/null | grep -a -v -E '^\[[0-9]{4}/0520\]' | head -60 || echo "NONE"

echo "--- [4.4] P5b result lines count per segment ---"
echo -n "P5b Segment 1 (4..397) result lines count: "
sed -n '4,397p' raw_s7/heldout520_P5b_seed2031.log 2>/dev/null | grep -c -E '^\[[0-9]{4}/0520\]' || echo "0"
echo -n "P5b Segment 2 (400..469) result lines count: "
sed -n '400,469p' raw_s7/heldout520_P5b_seed2031.log 2>/dev/null | grep -c -E '^\[[0-9]{4}/0520\]' || echo "0"
echo -n "P5b Segment 3 (472..541) result lines count: "
sed -n '472,541p' raw_s7/heldout520_P5b_seed2031.log 2>/dev/null | grep -c -E '^\[[0-9]{4}/0520\]' || echo "0"
echo -n "P5b Segment 4 (544..end) result lines count: "
sed -n '544,$p' raw_s7/heldout520_P5b_seed2031.log 2>/dev/null | grep -c -E '^\[[0-9]{4}/0520\]' || echo "0"

echo "=== 5. P6_CHAIN LOG & NEWLY CREATED FILES ==="
echo "--- raw_s7/p6_chain.log last 25 lines ---"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 25 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log does not exist"
fi

echo "--- Newly created files newer than P5_DONE.flag ---"
if [ -f raw_s7/P5_DONE.flag ]; then
  find raw_s7 results_s7 scratch -newer raw_s7/P5_DONE.flag -type f 2>/dev/null | xargs ls -la --time-style=full-iso 2>/dev/null || echo "NO_NEWER_FILES"
fi

echo "--- raw_s7/anchor20_gpu0.log (if exists, tail 20) ---"
if [ -f raw_s7/anchor20_gpu0.log ]; then
  tail -n 20 raw_s7/anchor20_gpu0.log
else
  echo "raw_s7/anchor20_gpu0.log does not exist"
fi

echo "=== 6. GPU HARDWARE & COMPUTE APPS ==="
echo "--- nvidia-smi --query-gpu ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv

echo "--- nvidia-smi --query-compute-apps ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    log_path = os.path.join(root, "shared", "ssh_log.csv")
    out_path = os.path.join(root, "scratch", "task0044_out.txt")

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

    purpose = "task0044_p5_done_and_p6_stage0_start"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]

    print("Executing TASK-0044 bundle query via cts_ssh.py...")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"Task 0044 bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

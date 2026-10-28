import os
import sys
import subprocess

REMOTE_SCRIPT = r"""
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0039: DIAGNOSIS & P5 RESUMPTION ON GPU 0 ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. DIAGNOSIS (READ-ONLY) ==="
echo "--- pgrep -af s7_train_p4_prime ---"
PGREP_OUT=$(pgrep -af "[s]7_train_p4_prime.*\.sh|[s]7_train_decoder" | grep -v "bash -c" | grep -v "$$" || true)
echo "${PGREP_OUT}"

echo "--- ps -o pid,ppid,lstart,cmd ---"
if [ -n "${PGREP_OUT}" ]; then
  PIDS=$(echo "${PGREP_OUT}" | awk '{print $1}')
  for p in ${PIDS}; do
    ps -o pid,ppid,lstart,cmd -p "$p" || true
  done
fi

echo "--- python decoder processes ---"
PY_PROCS=$(pgrep -af "python.*[s]7_train_decoder" || true)
echo "PYTHON_DECODER_PROCS: '${PY_PROCS}'"

echo "--- nvidia-smi compute-apps & GPU 0 memory ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ' || echo "99999")
echo "GPU0_MEM_MIB: ${GPU0_MEM}"

# Check condition (가):
# Only tmux server (cmd starts with tmux), python decoder is empty, GPU 0 mem < 1024 MiB
IS_TMUX_ONLY=1
while IFS= read -r line; do
  [ -z "$line" ] && continue
  cmd=$(echo "$line" | cut -d' ' -f2-)
  if [[ "$cmd" != tmux* ]]; then
    IS_TMUX_ONLY=0
  fi
done <<< "${PGREP_OUT}"

if [ "${IS_TMUX_ONLY}" -eq 1 ] && [ -z "${PY_PROCS}" ] && [ "${GPU0_MEM}" -lt 1024 ]; then
  echo "--- CONDITION (가) SATISFIED: Proceeding to Step 2 ---"

  # Step 2.1: Write CANCEL flag
  CANCEL_FLAG="raw_s7/P5_RESUME_CANCEL.flag"
  echo "cancelled_at: $(date '+%Y-%m-%d %H:%M:%S')" > "${CANCEL_FLAG}"
  echo "reason: TASK-0039 manual resume (waiter procs=1 false positive: tmux server cmdline)" >> "${CANCEL_FLAG}"
  echo "Created CANCEL flag:"
  cat "${CANCEL_FLAG}"

  # Step 2.2: Wait for waiter loop to catch CANCEL (max 360 seconds)
  echo "Waiting for p5_resume_waiter to detect CANCEL and exit (max 360s)..."
  WAITER_EXITED=0
  for i in {1..72}; do
    if grep -q "\[CANCEL\]" raw_s7/p5_resume_waiter.log 2>/dev/null; then
      echo "Detected [CANCEL] in p5_resume_waiter.log!"
      WAITER_EXITED=1
      break
    fi
    if ! tmux has-session -t p5_resume_waiter 2>/dev/null; then
      echo "tmux p5_resume_waiter session ended!"
      WAITER_EXITED=1
      break
    fi
    sleep 5
  done

  echo "--- p5_resume_waiter.log tail 6 lines ---"
  tail -n 6 raw_s7/p5_resume_waiter.log

  # Step 2.3: Check SHA of scripts/p5_heldout520_launch_v8.sh
  EXPECTED_V8_SHA="6f0cbcd282a909ca97ed557672feecb02415fda3d4aed3933f22871d1e2b2041"
  ACTUAL_V8_SHA=$(sha256sum scripts/p5_heldout520_launch_v8.sh | awk '{print $1}')
  echo "V8_EXPECTED_SHA: ${EXPECTED_V8_SHA}"
  echo "V8_ACTUAL_SHA:   ${ACTUAL_V8_SHA}"

  if [ "${ACTUAL_V8_SHA}" != "${EXPECTED_V8_SHA}" ]; then
    echo "[ERROR] SHA mismatch for p5_heldout520_launch_v8.sh! Halting."
    exit 1
  fi

  # Step 2.4: Launch P5 resume in new tmux session p5_gpu0
  echo "Launching P5 resume in tmux session p5_gpu0..."
  tmux new-session -d -s p5_gpu0 'cd /home/shoon/cts_e1_s5 && P5_RESUME=1 bash scripts/p5_heldout520_launch_v8.sh --device "cuda:0" 2>&1 | tee -a raw_s7/p5_resume_waiter.log'
  echo "Launched p5_gpu0 at $(date '+%Y-%m-%d %H:%M:%S')"

  # Step 2.5: Wait 5 minutes (300 seconds) for monitoring
  echo "Waiting 300 seconds (5 minutes) for execution progress..."
  sleep 300

  echo "================================================================================"
  echo "=== 3. 5-MINUTE STATUS ==="
  echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
  echo "================================================================================"

  echo "--- raw_s7/P5_RESUMED.flag ---"
  if [ -f raw_s7/P5_RESUMED.flag ]; then
    cat raw_s7/P5_RESUMED.flag
  else
    echo "raw_s7/P5_RESUMED.flag does not exist"
  fi

  echo "--- P5 Process & CUDA_VISIBLE_DEVICES ---"
  P5_PIDS=$(pgrep -f "s7_zstar_lowest_run.py" || true)
  if [ -n "$P5_PIDS" ]; then
    for p in $P5_PIDS; do
      echo "PID: $p"
      cat /proc/$p/environ 2>/dev/null | tr '\0' '\n' | grep CUDA_VISIBLE || true
    done
  else
    echo "NO_ZSTAR_PIDS"
  fi

  echo "--- P5 Log Path & Line Counts (Values Sealed) ---"
  P5_LOG="raw_s7/heldout520_P5.log"
  if [ -f "${P5_LOG}" ]; then
    echo "P5 Log file: ${P5_LOG}"
    echo -n "Skipping count: "
    grep -c "Skipping" "${P5_LOG}" 2>/dev/null || echo "0"
    echo -n "Last item header: "
    grep -oE '^\[[0-9]{4}/0520\]' "${P5_LOG}" 2>/dev/null | tail -n 1 || echo "None"
    echo -n "Item headers count: "
    grep -cE '^\[[0-9]{4}/0520\]' "${P5_LOG}" 2>/dev/null || echo "0"
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
        print('SEC_PER_ITEM: N/A (0 new items completed yet)')
"
  else
    echo "${P5_OUT} does not exist"
  fi

  echo "--- p6_chain.log tail 2 lines ---"
  if [ -f raw_s7/p6_chain.log ]; then
    tail -n 2 raw_s7/p6_chain.log
  else
    echo "raw_s7/p6_chain.log does not exist"
  fi

  echo "--- nvidia-smi --query-gpu ---"
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu --format=csv

  echo "--- nvidia-smi --query-compute-apps ---"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

  echo "--- free -h ---"
  free -h

else
  echo "--- CONDITION (나) TRIGGERED: Did not meet condition (가) ---"
  echo "IS_TMUX_ONLY: ${IS_TMUX_ONLY}"
  echo "PY_PROCS: '${PY_PROCS}'"
  echo "GPU0_MEM: ${GPU0_MEM}"
fi
"""

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    log_path = os.path.join(root, "shared", "ssh_log.csv")
    if os.path.exists(log_path):
        import time, datetime, math
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
    purpose = "task0039_diag_and_manual_resume_p5_gpu0"
    cmd = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", purpose]
    out_path = os.path.join(root, "scratch", "task0039_out.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Executing TASK-0039 bundle query via cts_ssh.py...")
    with open(out_path, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Task 0039 bundle execution saved to {out_path} with exit code {res.returncode}")
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()

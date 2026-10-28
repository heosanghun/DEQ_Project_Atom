import os
import sys
import time
import subprocess
from datetime import datetime

LOG_PATH = r"D:\AI\cts\shared\net_probe.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

print(f"[NET-PROBE] Starting automated SSH probe loop (60s interval, max 15 attempts)...")
print(f"[NET-PROBE] Logging to: {LOG_PATH}")

success = False
cmd = ["ssh", "-p", "10022", "-o", "ConnectTimeout=5", "shoon@<GPU_SERVER_IP>", "date"]

with open(LOG_PATH, "a", encoding="utf-8") as log_f:
    log_f.write(f"\n=== PROBE SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_f.flush()

for attempt in range(1, 16):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        elapsed = time.time() - t0
        exit_code = proc.returncode
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        log_line = f"[{now_str}] Attempt {attempt:02d}/15: exit={exit_code} ({elapsed:.2f}s) | stdout={out} | stderr={err}\n"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        exit_code = -1
        log_line = f"[{now_str}] Attempt {attempt:02d}/15: exit=TIMEOUT ({elapsed:.2f}s) | stderr=subprocess timeout\n"
    except Exception as e:
        elapsed = time.time() - t0
        exit_code = -2
        log_line = f"[{now_str}] Attempt {attempt:02d}/15: exit=ERROR ({elapsed:.2f}s) | stderr={str(e)}\n"

    print(log_line.strip(), flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        log_f.write(log_line)
        log_f.flush()

    if exit_code == 0:
        success = True
        print(f"[NET-PROBE] SSH CONNECTION SUCCEEDED at attempt {attempt}!")
        break

    if attempt < 15:
        time.sleep(60)

print(f"[NET-PROBE] Probe session ended. Succeeded: {success}")
sys.exit(0 if success else 1)

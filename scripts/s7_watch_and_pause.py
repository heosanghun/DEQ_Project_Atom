#!/usr/bin/env python3
"""scripts/s7_watch_and_pause.py - Watcher script for S7 z* pipeline
Directive: EXE-260927-38 / 원장 #370, #371
Monitors s7_zstar_pipeline.log for Step 2 (train 680 items) completion.
When detected, terminates the tmux session s7_zstar_gpu0 to pause the pipeline
before Step 3 (eval 520 items) begins, freeing GPU 0 for decoder training (Recommendation 가).
"""

import hashlib
import os
import subprocess
import sys
import time

BASE_DIR = "/home/shoon/cts_e1_s5"
LOG_FILE = os.path.join(BASE_DIR, "raw_s7", "s7_zstar_pipeline.log")
TARGET_FILE = os.path.join(BASE_DIR, "results_s7", "zstar_s7_a0.25_train680.pt")
WATCH_LOG = os.path.join(BASE_DIR, "raw_s7", "watch_and_pause.log")
FLAG_FILE = os.path.join(BASE_DIR, "raw_s7", "step2_paused.flag")


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(WATCH_LOG, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    log("S7 Watcher started. Monitoring for Step 2 completion per Recommendation (가)...")

    while True:
        step2_done = False

        # Check 1: Existence of final train680 pt file
        if os.path.exists(TARGET_FILE):
            # Check if file has finished writing (no growth over 2 seconds)
            size1 = os.path.getsize(TARGET_FILE)
            time.sleep(2)
            size2 = os.path.getsize(TARGET_FILE)
            if size1 == size2 and size1 > 1000000:  # > 1MB
                step2_done = True
                log(f"Detected final target file: {TARGET_FILE} (size: {size2} bytes)")

        # Check 2: Check pipeline log for completion message
        if not step2_done and os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if "Step 2 (train 680 items) completed" in content:
                    step2_done = True
                    log("Detected '[STEP 2/3] Step 2 (train 680 items) completed' in log file.")
            except Exception as e:
                log(f"Warning reading log file: {e}")

        if step2_done:
            log("Executing pause: terminating tmux session 's7_zstar_gpu0'...")
            # Kill tmux session
            subprocess.run(["tmux", "kill-session", "-t", "s7_zstar_gpu0"], check=False)
            time.sleep(3)

            # Verification
            res400 = os.path.join(BASE_DIR, "results_s7", "zstar_s7_a0.25_400.pt")
            sha400 = compute_sha256(res400) if os.path.exists(res400) else "NOT_FOUND"
            sha680 = compute_sha256(TARGET_FILE) if os.path.exists(TARGET_FILE) else "NOT_FOUND"

            log("Pipeline paused successfully.")
            log(f"zstar_s7_a0.25_400.pt SHA-256: {sha400}")
            log(f"zstar_s7_a0.25_train680.pt SHA-256: {sha680}")

            with open(FLAG_FILE, "w", encoding="utf-8") as f:
                f.write(f"STEP2_COMPLETED_PAUSED\n400_sha={sha400}\n680_sha={sha680}\n")

            log("Flag file created. Ready for §0 report and decoder training on GPU 0.")
            break

        time.sleep(15)


if __name__ == "__main__":
    main()

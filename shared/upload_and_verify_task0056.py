#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/upload_and_verify_task0056.py - Upload & Verify Runner for scripts/kstar_replay_launch_v1.sh
Executes:
1. R7-1 cooldown check (>=65s)
2. SFTP upload via scratch/upload_task56.sftp
3. R7-1 cooldown wait (>=65s)
4. SSH verification of SHA-256, file size, permissions
Appends results to scratch/task0056_upload_out.txt and scratch/task0056_out.txt.
"""

import datetime
import math
import os
import subprocess
import sys
import time

def enforce_cooldown(log_path, min_seconds=65.0):
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) > 1:
            last_ts = lines[-1].split(",")[0]
            try:
                last_dt = datetime.datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                diff = (datetime.datetime.now() - last_dt).total_seconds()
                if diff < min_seconds:
                    wait_s = math.ceil(min_seconds - diff)
                    print(f"Waiting {wait_s}s to satisfy R7-1 cooldown ({diff:.1f}s elapsed)...", flush=True)
                    time.sleep(wait_s)
            except Exception as e:
                pass

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cts_ssh = os.path.join(root, "shared", "cts_ssh.py")
    log_path = os.path.join(root, "shared", "ssh_log.csv")
    sftp_batch = os.path.join(root, "scratch", "upload_task56.sftp")
    out_path = os.path.join(root, "scratch", "task0056_upload_out.txt")

    print("=== Step 1: Enforcing Cooldown before SFTP Upload ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    print("=== Step 2: Uploading scripts/kstar_replay_launch_v1.sh via SFTP ===", flush=True)
    cmd_sftp = [sys.executable, cts_ssh, "sftp", sftp_batch, "--purpose", "task0056_upload_launch_v1"]
    res_sftp = subprocess.run(cmd_sftp, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("SFTP returncode:", res_sftp.returncode)
    print(res_sftp.stdout)
    if res_sftp.returncode != 0:
        print("[ERROR] SFTP upload failed!", file=sys.stderr)
        if res_sftp.stderr:
            print(res_sftp.stderr, file=sys.stderr)
        sys.exit(res_sftp.returncode)

    print("=== Step 3: Waiting 65s Cooldown before SSH Verification ===", flush=True)
    time.sleep(65)

    print("=== Step 4: Verifying SHA-256 and Permissions on Server via SSH ===", flush=True)
    remote_cmd = (
        "cd /home/shoon/cts_e1_s5 && "
        "echo '=== SERVER LAUNCH_V1 SHA-256 & STAT ===' && "
        "sha256sum scripts/kstar_replay_launch_v1.sh && "
        "ls -la --time-style=full-iso scripts/kstar_replay_launch_v1.sh"
    )
    cmd_ssh = [sys.executable, cts_ssh, "ssh", remote_cmd, "--purpose", "task0056_verify_sha_and_telemetry"]
    res_ssh = subprocess.run(cmd_ssh, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("SSH returncode:", res_ssh.returncode)
    print(res_ssh.stdout)
    if res_ssh.returncode != 0:
        print("[ERROR] SSH verify failed!", file=sys.stderr)
        if res_ssh.stderr:
            print(res_ssh.stderr, file=sys.stderr)
        sys.exit(res_ssh.returncode)

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("=== TASK-0056 UPLOAD & VERIFY OUTPUT ===\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}\n")
        f.write(f"SFTP Return Code: {res_sftp.returncode}\n")
        f.write("--- SFTP STDOUT ---\n" + res_sftp.stdout + "\n")
        f.write(f"SSH Return Code: {res_ssh.returncode}\n")
        f.write("--- SSH STDOUT ---\n" + res_ssh.stdout + "\n")

    print(f"Results written to {out_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0058_bundle.py - Bundled execution for TASK-0058 inspection.
Executes a single bundled SSH command via shared/cts_ssh.py (v2b).
Saves output to scratch/task0058_out.txt.
"""

import os
import subprocess
import sys

def main():
    root = r"d:\AI\cts"
    out_file = os.path.join(root, "scratch", "task0058_out.txt")
    os.makedirs(os.path.join(root, "scratch"), exist_ok=True)

    remote_cmd = (
        "set -u; "
        "echo '================================================================================'; "
        "echo '=== TASK-0058: BUNDLED INSPECTION (REPLAY 325, P6, HOST) ==='; "
        "echo \"Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')\"; "
        "echo '================================================================================'; "
        "echo '=== 0. HOST & TMUX STATUS ==='; "
        "echo '--- Uptime and Boot Time ---'; "
        "uptime -s; "
        "uptime; "
        "echo '--- Tmux Sessions ---'; "
        "tmux ls; "
        "echo '================================================================================'; "
        "echo '=== 1. ①-a REPLAY (GPU 1) PROGRESS INSPECTION ==='; "
        "cd /home/shoon/cts_e1_s5; "
        "echo '--- Last 5 Header Lines (Numbers Only) ---'; "
        "grep -oE '^\\[[0-9]{4}/0325\\] q[0-9]+_a[0-9.]+' raw_s7/kstar_replay_gpu1.log | tail -5; "
        "echo '--- Last Progress Header Number ---'; "
        "grep -oE '^\\[[0-9]{4}/0325\\]' raw_s7/kstar_replay_gpu1.log | tail -1; "
        "echo '--- Total Header Count ---'; "
        "grep -cE '^\\[[0-9]{4}/0325\\]' raw_s7/kstar_replay_gpu1.log; "
        "echo '--- Errors or Tracebacks in kstar_replay_gpu1.log ---'; "
        "grep -inE 'error|traceback' raw_s7/kstar_replay_gpu1.log || echo 'NO_ERRORS_DETECTED'; "
        "echo '--- Replay Output File Check ---'; "
        "ls -la --full-time results_s7/zstar_s7_a0.25_train325_kstar_replay.pt 2>/dev/null || echo 'OUTPUT_PT_NOT_EXISTS'; "
        "echo '--- KSTAR_REPLAY_DONE.flag Check ---'; "
        "ls -la raw_s7/KSTAR_REPLAY_DONE.flag 2>/dev/null || echo 'DONE_FLAG_NOT_EXISTS'; "
        "echo '--- Precheck Output File Check ---'; "
        "ls -la --full-time results_s7/kstar_replay_precheck5_gpu1.pt 2>/dev/null || echo 'PRECHECK_NOT_EXISTS'; "
        "echo '--- Started Flag Content ---'; "
        "cat raw_s7/KSTAR_REPLAY_STARTED.flag 2>/dev/null || echo 'STARTED_FLAG_NOT_EXISTS'; "
        "echo '--- Duration Stats from Log ---'; "
        "grep -oE '\\([0-9.]+s\\)' raw_s7/kstar_replay_gpu1.log | tr -d '()s' | awk '{sum+=$1; count++} END {if (count>0) printf \"count=%d, sum_seconds=%.2f, avg_seconds=%.2f\\n\", count, sum, sum/count}'; "
        "echo '================================================================================'; "
        "echo '=== 2. P6 CHAIN (GPU 0) PROGRESS INSPECTION ==='; "
        "echo '--- raw_s7/p6_chain.log Last 12 Lines ---'; "
        "tail -n 12 raw_s7/p6_chain.log; "
        "echo '--- P6 DONE Flags ---'; "
        "ls -la raw_s7/P6_*DONE.flag 2>/dev/null || echo 'NO_P6_DONE_FLAGS'; "
        "echo '--- P6 Gen JSONL Files (Line Counts) ---'; "
        "wc -l results_s7/p6_gen_*.jsonl 2>/dev/null || echo 'NO_P6_GEN_FILES'; "
        "echo '--- P6 Score Files (Line Counts) ---'; "
        "wc -l results_s7/p6_score* 2>/dev/null || echo 'NO_P6_SCORE_FILES'; "
        "echo '================================================================================'; "
        "echo '=== 3. HARDWARE & PROCESS TELEMETRY ==='; "
        "echo '--- GPU 0~7 Telemetry ---'; "
        "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv; "
        "echo '--- GPU Compute Apps ---'; "
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv; "
        "echo '--- Process Details ---'; "
        "ps -u shoon -o user,pid,%cpu,%mem,vsz,rss,tty,stat,start,time,cmd; "
        "echo '--- Memory Status ---'; "
        "free -h; "
        "echo '================================================================================'; "
        "echo '=== TASK-0058 BUNDLE FINISHED ==='; "
        "echo \"Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')\"; "
        "echo '================================================================================'"
    )

    cmd = [
        sys.executable,
        os.path.join(root, "shared", "cts_ssh.py"),
        "ssh",
        remote_cmd,
        "--purpose",
        "task0058_bundled_inspection",
    ]

    print(f"[TASK-0058] Executing bundled inspection SSH...")
    p = subprocess.run(cmd, cwd=root, capture_output=True, text=False)

    stdout_bytes = p.stdout
    stderr_bytes = p.stderr

    with open(out_file, "wb") as f:
        f.write(stdout_bytes)

    print(f"[TASK-0058] Exit code: {p.returncode}")
    print(f"[TASK-0058] Output bytes: {len(stdout_bytes)}, saved to {out_file}")
    if stderr_bytes:
        print(f"[TASK-0058] Stderr:\n{stderr_bytes.decode('utf-8', errors='replace')}")

    if p.returncode != 0:
        print("[ERROR] SSH command failed!")
        sys.exit(p.returncode)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0059_bundle.py - Bundled inspection for TASK-0059.
Collects P6 main gate scoring artifacts, log blocks, file SHAs,
and Replay/GPU telemetry via a single SSH call.
"""

import os
import subprocess
import sys

def main():
    root = r"d:\AI\cts"
    out_file = os.path.join(root, "scratch", "task0059_out.txt")
    os.makedirs(os.path.join(root, "scratch"), exist_ok=True)

    remote_cmd = (
        "set -u; "
        "echo '================================================================================'; "
        "echo '=== TASK-0059: BUNDLED INSPECTION (P6 MAIN GATES, REPLAY, GPU) ==='; "
        "echo \"Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')\"; "
        "echo '================================================================================'; "
        "echo '=== 1. STAGE 3 BLOCK IN raw_s7/p6_chain.log ==='; "
        "cd /home/shoon/cts_e1_s5; "
        "echo '--- Stage 3 Block Line Numbers & Context ---'; "
        "grep -n -C 5 'Stage 3' raw_s7/p6_chain.log; "
        "echo '--- raw_s7/p6_score_main.log Head (Command & Setup) ---'; "
        "head -n 25 raw_s7/p6_score_main.log 2>/dev/null || echo 'NO_SCORE_MAIN_LOG'; "
        "echo '--- raw_s7/p6_score_main.log Tail (Completion & Output Files) ---'; "
        "tail -n 25 raw_s7/p6_score_main.log 2>/dev/null || echo 'NO_SCORE_MAIN_LOG'; "
        "echo '================================================================================'; "
        "echo '=== 2. P6 SCORE / GATE / MAIN ARTIFACTS LISTING & SHAS ==='; "
        "echo '--- File Listing (results_s7 & raw_s7 matching p6_score*, p6_gate*, p6_main*) ---'; "
        "find results_s7 raw_s7 -maxdepth 2 -type f \\( -name '*p6_score*' -o -name '*p6_gate*' -o -name '*p6_main*' \\) -exec ls -la --full-time {} \\; | sort -k9; "
        "echo '--- SHA-256 Hashes of Artifacts ---'; "
        "find results_s7 raw_s7 -maxdepth 2 -type f \\( -name '*p6_score*' -o -name '*p6_gate*' -o -name '*p6_main*' \\) -exec sha256sum {} \\; | sort -k2; "
        "echo '--- Evaluator Script SHA (scripts/s7_eval_score_p6_v36.py) ---'; "
        "sha256sum scripts/s7_eval_score_p6_v36.py; "
        "echo '--- P6 Generation Files Line Counts & SHAs ---'; "
        "wc -l results_s7/p6_gen_*.jsonl 2>/dev/null; "
        "sha256sum results_s7/p6_gen_*.jsonl 2>/dev/null; "
        "echo '================================================================================'; "
        "echo '=== 3. ①-a REPLAY (GPU 1) PROGRESS & GPU TELEMETRY ==='; "
        "echo '--- Replay Header Progress ---'; "
        "grep -oE '^\\[[0-9]{4}/0325\\] q[0-9]+_a[0-9.]+' raw_s7/kstar_replay_gpu1.log | tail -5; "
        "echo '--- Replay Last Header Number ---'; "
        "grep -oE '^\\[[0-9]{4}/0325\\]' raw_s7/kstar_replay_gpu1.log | tail -1; "
        "echo '--- Replay Total Header Count ---'; "
        "grep -cE '^\\[[0-9]{4}/0325\\]' raw_s7/kstar_replay_gpu1.log; "
        "echo '--- Replay Output Tensor Info ---'; "
        "ls -la --full-time results_s7/zstar_s7_a0.25_train325_kstar_replay.pt 2>/dev/null; "
        "echo '--- GPU 0~7 Telemetry ---'; "
        "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv; "
        "echo '--- GPU Compute Apps ---'; "
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv; "
        "echo '================================================================================'; "
        "echo '=== TASK-0059 BUNDLE FINISHED ==='; "
        "echo \"Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')\"; "
        "echo '================================================================================'"
    )

    cmd = [
        sys.executable,
        os.path.join(root, "shared", "cts_ssh.py"),
        "ssh",
        remote_cmd,
        "--purpose",
        "task0059_bundled_inspection",
    ]

    print(f"[TASK-0059] Executing bundled inspection SSH...")
    p = subprocess.run(cmd, cwd=root, capture_output=True, text=False)

    stdout_bytes = p.stdout
    stderr_bytes = p.stderr

    with open(out_file, "wb") as f:
        f.write(stdout_bytes)

    print(f"[TASK-0059] Exit code: {p.returncode}")
    print(f"[TASK-0059] Output bytes: {len(stdout_bytes)}, saved to {out_file}")
    if stderr_bytes:
        print(f"[TASK-0059] Stderr:\n{stderr_bytes.decode('utf-8', errors='replace')}")

    if p.returncode != 0:
        print("[ERROR] SSH command failed!")
        sys.exit(p.returncode)

if __name__ == "__main__":
    main()

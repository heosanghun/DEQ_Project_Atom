#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0061_bundle.py - Bundled inspection for TASK-0061.
Checks ①-a replay completion, analyzer outputs, P6 status, and GPU telemetry via single SSH call.
Uses /home/shoon/system1_5/.venv/bin/python (contains PyTorch).
"""

import base64
import os
import subprocess
import sys

def main():
    root = r"d:\AI\cts"
    out_file = os.path.join(root, "scratch", "task0061_out.txt")
    os.makedirs(os.path.join(root, "scratch"), exist_ok=True)

    remote_python_code = """
import os, sys, glob, subprocess, hashlib

print("================================================================================")
print("=== 1. UPTIME & TMUX ===")
print("Timestamp:", subprocess.getoutput("date '+%Y-%m-%d %H:%M:%S KST'"))
print("Uptime:", subprocess.getoutput("uptime -s"))
print("Tmux sessions:")
subprocess.run(["tmux", "ls"])

print("================================================================================")
print("=== 2. KSTAR_REPLAY_DONE.flag ===")
flag_path = "raw_s7/KSTAR_REPLAY_DONE.flag"
if os.path.exists(flag_path):
    with open(flag_path, "r", encoding="utf-8") as f:
        print(f.read().strip())
else:
    print("KSTAR_REPLAY_DONE.flag NOT FOUND")

print("================================================================================")
print("=== 3. kstar_replay_gpu1.log (tail 40 lines with result masking) ===")
log_path = "raw_s7/kstar_replay_gpu1.log"
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    total_lines = len(lines)
    start_idx = max(0, total_lines - 40)
    for i, line in enumerate(lines[start_idx:], start=start_idx+1):
        line_str = line.strip()
        # If line contains item result [NNNN/0325], mask details to prevent metric leakage
        if line_str.startswith("[") and "/0325]" in line_str:
            parts = line_str.split()
            header_str = parts[0] if parts else "[item]"
            print(f"Line {i:05d}: {header_str} (item result line masked)")
        else:
            print(f"Line {i:05d}: {line_str}")
else:
    print("kstar_replay_gpu1.log NOT FOUND")

print("================================================================================")
print("=== 4. REPLAY OUTPUT FILES & KEYS COUNT ===")
pt_path = "results_s7/zstar_s7_a0.25_train325_kstar_replay.pt"
if os.path.exists(pt_path):
    sz = os.path.getsize(pt_path)
    h_obj = hashlib.sha256()
    with open(pt_path, "rb") as f:
        while chunk := f.read(65536):
            h_obj.update(chunk)
    h = h_obj.hexdigest()
    print(f"PT_FILE: {pt_path} | SIZE: {sz} bytes | SHA256: {h}")
    try:
        import torch
        with torch.no_grad():
            d = torch.load(pt_path, map_location="cpu")
            keys_list = list(d.keys())
            print(f"PT_KEYS_COUNT: {len(keys_list)}")
            del d
    except Exception as e:
        print(f"PT_KEYS_COUNT_ERROR: {e}")
else:
    print(f"PT_FILE NOT FOUND: {pt_path}")

for p in ["results_s7/p5_promote_1a_replay_v2.json", "results_s7/p5_promote_1a_replay_v2.md"]:
    if os.path.exists(p):
        sz = os.path.getsize(p)
        h_obj = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h_obj.update(chunk)
        h = h_obj.hexdigest()
        print(f"ANALYSIS_FILE: {p} | SIZE: {sz} bytes | SHA256: {h}")
    else:
        print(f"ANALYSIS_FILE NOT FOUND: {p}")

print("================================================================================")
print("=== 5. P6 CHAIN STATUS ===")
p6_log = "raw_s7/p6_chain.log"
if os.path.exists(p6_log):
    with open(p6_log, "r", encoding="utf-8", errors="replace") as f:
        p6_lines = f.readlines()
    print(f"--- p6_chain.log tail 6 (total lines: {len(p6_lines)}) ---")
    for l in p6_lines[-6:]:
        print(l.rstrip())
else:
    print("p6_chain.log NOT FOUND")

print("--- p6_gen_*.jsonl lines count ---")
for jp in sorted(glob.glob("results_s7/p6_gen_*.jsonl")):
    with open(jp, "r", encoding="utf-8", errors="replace") as f:
        c = sum(1 for _ in f)
    print(f"{c:6d} {jp}")

print("--- p6_gate_table_* files ---")
for gp in sorted(glob.glob("results_s7/p6_gate_table_*")):
    sz = os.path.getsize(gp)
    print(f"{sz:8d} bytes | {gp}")

print("================================================================================")
print("=== 6. GPU TELEMETRY ===")
subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw", "--format=csv"])
print("--- GPU Compute Apps ---")
subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv"])

print("================================================================================")
print("=== TASK-0061 BUNDLE FINISHED ===")
print("Timestamp:", subprocess.getoutput("date '+%Y-%m-%d %H:%M:%S KST'"))
print("================================================================================")
"""

    b64_code = base64.b64encode(remote_python_code.encode("utf-8")).decode("utf-8")

    remote_cmd = (
        "cd /home/shoon/cts_e1_s5; "
        "ulimit -v 62914560; "
        f"nice -n 19 env CUDA_VISIBLE_DEVICES='' /home/shoon/system1_5/.venv/bin/python -c \"import base64; exec(base64.b64decode('{b64_code}').decode('utf-8'))\""
    )

    cmd = [
        sys.executable,
        os.path.join(root, "shared", "cts_ssh.py"),
        "ssh",
        remote_cmd,
        "--purpose",
        "task0061_bundled_inspection",
    ]

    print("[TASK-0061] Executing bundled inspection SSH via cts_ssh.py...")
    p = subprocess.run(cmd, cwd=root, capture_output=True, text=False)

    stdout_bytes = p.stdout
    stderr_bytes = p.stderr

    with open(out_file, "wb") as f:
        f.write(stdout_bytes)

    print(f"[TASK-0061] Exit code: {p.returncode}")
    print(f"[TASK-0061] Output bytes: {len(stdout_bytes)}, saved to {out_file}")
    if stderr_bytes:
        print(f"[TASK-0061] Stderr:\n{stderr_bytes.decode('utf-8', errors='replace')}")

    if p.returncode != 0:
        print("[ERROR] SSH command failed!")
        sys.exit(p.returncode)

if __name__ == "__main__":
    main()

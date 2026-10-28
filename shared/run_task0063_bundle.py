#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0063_bundle.py - Bundled inspection for TASK-0063.
Checks P6 Stage 4 finish, Stage 5 full scoring artifacts, Stage 6 NLL progress/timing,
and GPU telemetry via single SSH call.
NOTE: Execution permitted strictly after 12:30 KST per TASK-0063 directive.
"""

import base64
import os
import subprocess
import sys

def main():
    root = r"d:\AI\cts"
    out_file = os.path.join(root, "scratch", "task0063_out.txt")
    os.makedirs(os.path.join(root, "scratch"), exist_ok=True)

    remote_python_code = """
import os, sys, glob, subprocess, hashlib, time

print("================================================================================")
print("=== 1. UPTIME & TMUX ===")
print("Timestamp:", subprocess.getoutput("date '+%Y-%m-%d %H:%M:%S KST'"))
print("Uptime:", subprocess.getoutput("uptime -s"))
print("Tmux sessions:")
subprocess.run(["tmux", "ls"])

print("================================================================================")
print("=== 2. p6_chain.log TAIL 12 LINES ===")
chain_log = "raw_s7/p6_chain.log"
if os.path.exists(chain_log):
    with open(chain_log, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    for l in lines[-12:]:
        print(l.rstrip())
else:
    print("p6_chain.log NOT FOUND")

print("================================================================================")
print("=== 3. STAGE 5 FULL GATES SCORING ARTIFACTS ===")
for p in [
    "raw_s7/P6_SCORE_FULL_DONE.flag",
    "raw_s7/p6_score_full.log",
    "results_s7/p6_gate_table_full.json",
    "results_s7/p6_gate_table_full.md",
]:
    if os.path.exists(p):
        sz = os.path.getsize(p)
        h_obj = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h_obj.update(chunk)
        h = h_obj.hexdigest()
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(p)))
        print(f"STAGE5_FILE: {p} | SIZE: {sz} bytes | MTIME: {mtime} | SHA256: {h}")
    else:
        print(f"STAGE5_FILE NOT FOUND: {p}")

print("================================================================================")
print("=== 4. STAGE 6 NLL ARTIFACTS & SPEED ===")
print("--- NLL Done Flags ---")
for f in sorted(glob.glob("raw_s7/P6_NLL_*_DONE.flag")):
    sz = os.path.getsize(f)
    with open(f, "r", encoding="utf-8", errors="replace") as fp:
        c = fp.read().strip()
    print(f"FLAG: {f} | CONTENT: {c}")

print("--- NLL JSONL Files (Progress) ---")
for jf in sorted(glob.glob("results_s7/p6_nll_*.jsonl")):
    sz = os.path.getsize(jf)
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(jf)))
    with open(jf, "r", encoding="utf-8", errors="replace") as fp:
        line_cnt = sum(1 for _ in fp)
    print(f"NLL_JSONL: {jf} | LINES: {line_cnt} | SIZE: {sz} bytes | MTIME: {mtime}")

print("--- NLL Log Files ---")
for lf in sorted(glob.glob("raw_s7/p6_nll_*.log")):
    sz = os.path.getsize(lf)
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(lf)))
    with open(lf, "r", encoding="utf-8", errors="replace") as fp:
        log_lines = fp.readlines()
    print(f"NLL_LOG: {lf} | TOTAL_LINES: {len(log_lines)} | SIZE: {sz} bytes | MTIME: {mtime}")
    if log_lines:
        print("  Head 3:")
        for hl in log_lines[:3]:
            print("   ", hl.rstrip())
        print("  Tail 5 (Timing & Item headers):")
        for tl in log_lines[-5:]:
            tl_str = tl.rstrip()
            if "nll" in tl_str.lower() or "loss" in tl_str.lower():
                print("    [NLL/LOSS LINE MASKED per TASK-0063 directive]")
            else:
                print("   ", tl_str)

print("================================================================================")
print("=== 5. P6 GENERATION JSONL LINE COUNTS ===")
for gp in sorted(glob.glob("results_s7/p6_gen_*.jsonl")):
    with open(gp, "r", encoding="utf-8", errors="replace") as fp:
        c = sum(1 for _ in fp)
    print(f"{c:6d} {gp}")

print("================================================================================")
print("=== 6. GPU TELEMETRY ===")
subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw", "--format=csv"])
print("--- GPU Compute Apps ---")
subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv"])

print("================================================================================")
print("=== TASK-0063 BUNDLE FINISHED ===")
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
        "task0063_bundled_inspection",
    ]

    print("[TASK-0063] Executing bundled inspection SSH via cts_ssh.py...")
    p = subprocess.run(cmd, cwd=root, capture_output=True, text=False)

    stdout_bytes = p.stdout
    stderr_bytes = p.stderr

    with open(out_file, "wb") as f:
        f.write(stdout_bytes)

    print(f"[TASK-0063] Exit code: {p.returncode}")
    print(f"[TASK-0063] Output bytes: {len(stdout_bytes)}, saved to {out_file}")
    if stderr_bytes:
        print(f"[TASK-0063] Stderr:\n{stderr_bytes.decode('utf-8', errors='replace')}")

    if p.returncode != 0:
        print("[ERROR] SSH command failed!")
        sys.exit(p.returncode)

if __name__ == "__main__":
    main()

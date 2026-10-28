#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shared/run_task0051_bundle.py - TASK-0051 Automated Runner (SFTP upload + bundled SSH collection)
Executes:
1. SFTP Upload (scripts/s7deq_soft_v2_launch_v3.sh)
2. Enforce 65s cooldown (R7-1 compliance)
3. Bundled SSH execution:
   - File SHA-256, size, mtime checks for reference, launch_v3, scripts, backbone
   - Python CPU evaluation (nice 19, ulimit 60GB):
     * Key counts and sorted lists of both .pt files
     * eps fields and fields list
     * Existence of results_s7/anchor_soft_eps1e-12_check.pt
     * R1 prefix comparison: rho_traj[:k1-1] vs canonical reference across 20 anchors
   - tmux ls, GPU 0~7 telemetry, P6 chain log & jsonl line count
Output saved to scratch/task0051_out.txt.
"""

import datetime
import math
import os
import subprocess
import sys
import time

REMOTE_SCRIPT = r"""
cd /home/shoon/cts_e1_s5

echo "================================================================================"
echo "=== TASK-0051: SERVER FACT COLLECTION & VERIFICATION ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')"
echo "================================================================================"

echo "=== 1. FILE SHA-256, SIZE, AND MTIME VERIFICATION ==="
FILES_TO_CHECK=(
  "scripts/s7deq_soft_v2_launch_v3.sh"
  "results_s7/anchor_ref_gpu1_check.pt"
  "results_s7/anchor_soft_eps0.02_check.pt"
  "scripts/s7_zstar_lowest_run.py"
  "cts/deq/zstar_canonical_lowest.py"
  "exe_scratch/s7/soft_v2/zstar_soft_lowest.py"
  "scripts/s7deq_soft_v2_run_v2.py"
  "data/anchor20.json"
  "configs/zstar_canonical.yaml"
)

for f in "${FILES_TO_CHECK[@]}"; do
  if [ -f "$f" ]; then
    sha256sum "$f"
    ls -la --time-style=full-iso "$f"
  else
    echo "[ERROR] Missing file: $f"
  fi
done

echo "--- 1.1 BACKBONE SNAPSHOT DIRECTORY FILES ---"
BB_DIR="/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
if [ -d "${BB_DIR}" ]; then
  for f in "${BB_DIR}"/*.safetensors "${BB_DIR}"/config.json; do
    if [ -f "$f" ]; then
      sha256sum "$f"
      ls -la --time-style=full-iso "$f"
    fi
  done
else
  echo "[ERROR] Backbone directory not found: ${BB_DIR}"
fi

echo "=== 2. PT FILES INSPECTION & R1 PREFIX COMPARISON (CPU, NICE 19) ==="
export CUDA_VISIBLE_DEVICES=""
export MALLOC_ARENA_MAX=1
export MALLOC_TRIM_THRESHOLD_=131072
export TORCH_NUM_THREADS=4
ulimit -v 62914560

nice -n 19 /home/shoon/system1_5/.venv/bin/python - << 'EOF'
import json
import os
import sys
import torch

ref_path = "results_s7/anchor_ref_gpu1_check.pt"
soft_path = "results_s7/anchor_soft_eps0.02_check.pt"
soft_eps1e12_path = "results_s7/anchor_soft_eps1e-12_check.pt"

print("--- 2.1 Existence of anchor_soft_eps1e-12_check.pt ---")
exists_1e12 = os.path.exists(soft_eps1e12_path)
print(f"results_s7/anchor_soft_eps1e-12_check.pt exists: {exists_1e12}")

print("\n--- 2.2 Loading .pt files on CPU with torch.no_grad() ---")
with torch.no_grad():
    ref_dict = torch.load(ref_path, map_location="cpu")
    soft_dict = torch.load(soft_path, map_location="cpu")

ref_keys = sorted(list(ref_dict.keys()))
soft_keys = sorted(list(soft_dict.keys()))

print(f"anchor_ref_gpu1_check.pt key count: {len(ref_keys)}")
print(f"anchor_ref_gpu1_check.pt keys: {ref_keys}")

print(f"\nanchor_soft_eps0.02_check.pt key count: {len(soft_keys)}")
print(f"anchor_soft_eps0.02_check.pt keys: {soft_keys}")

if soft_keys:
    first_k = soft_keys[0]
    print(f"\nFields of first key ({first_k}) in anchor_soft_eps0.02_check.pt:")
    fields = sorted(list(soft_dict[first_k].keys()))
    print(f"Field count: {len(fields)}, Fields: {fields}")

print("\neps field values for each key in anchor_soft_eps0.02_check.pt:")
eps_vals = {k: soft_dict[k].get("eps") for k in soft_keys}
for k, eps in eps_vals.items():
    print(f"  {k}: eps = {eps}")

print("\n=== 2.3 R1 (REPORT ONLY): PREFIX COMPARISON (rho_traj[:k1-1]) ===")
print("Evaluating 20 anchors: finding first boundary step k1 (d <= 0.02), then checking max|Delta| on prefix rho_traj[:k1-1]")

anchor_config = "data/anchor20.json"
with open(anchor_config, "r", encoding="utf-8") as f:
    anchor_data = json.load(f)
if isinstance(anchor_data, list):
    anchors = [int(x) for x in anchor_data]
elif isinstance(anchor_data, dict) and "indices" in anchor_data:
    anchors = [int(x) for x in anchor_data["indices"]]
else:
    anchors = [int(x) for x in list(anchor_data.values())[0]]

initial = [20, 24, 32]
ordered_anchors = list(initial)
for a in anchors:
    if a not in ordered_anchors:
        ordered_anchors.append(a)

print(f"{'Anchor':<8} | {'k1 (1-based)':<12} | {'Prefix Len':<10} | {'max|Delta| prefix':<20} | {'|Delta| at k1':<18} | {'Status'}")
print("-" * 85)

for q in ordered_anchors:
    k = f"q{q:04d}_a0.25"
    if k not in soft_dict or k not in ref_dict:
        print(f"Q#{q:<6} | KEY_MISSING in dicts")
        continue

    s_item = soft_dict[k]
    r_item = ref_dict[k]

    d_traj = s_item.get("d_traj", [])
    rho_soft = s_item.get("rho_traj", [])
    rho_ref = r_item.get("rho_traj", [])

    # Find k1: first 1-based step where d <= 0.02
    k1 = None
    for step_idx, d_val in enumerate(d_traj):
        if d_val <= 0.02:
            k1 = step_idx + 1  # 1-based
            break

    if k1 is None:
        k1 = len(d_traj) + 1
        prefix_len = len(rho_soft)
        diff_prefix = torch.max(torch.abs(torch.tensor(rho_soft) - torch.tensor(rho_ref))).item()
        delta_k1 = 0.0
        status = "NEVER_ENTERED"
    elif k1 == 1:
        prefix_len = 0
        diff_prefix = 0.0
        delta_k1 = abs(rho_soft[0] - rho_ref[0])
        status = "BOUNDARY_AT_STEP1"
    else:
        prefix_len = k1 - 1
        s_pref = torch.tensor(rho_soft[:prefix_len], dtype=torch.float64)
        r_pref = torch.tensor(rho_ref[:prefix_len], dtype=torch.float64)
        diff_prefix = torch.max(torch.abs(s_pref - r_pref)).item()
        delta_k1 = abs(rho_soft[k1 - 1] - rho_ref[k1 - 1])
        status = "BIT_EXACT" if diff_prefix == 0.0 else "DIFF_DETECTED"

    print(f"Q#{q:<6} | {k1:<12} | {prefix_len:<10} | {diff_prefix:<20.6e} | {delta_k1:<18.6e} | {status}")

print("-" * 85)
EOF

echo "=== 3. TMUX SESSIONS & GPU STATUS ==="
echo "--- 3.1 TMUX SESSIONS ---"
tmux ls

echo "--- 3.2 GPU 0~7 TELEMETRY ---"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv

echo "--- 3.3 GPU COMPUTE APPS ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv

echo "=== 4. P6_CHAIN LOG & JSONL LINE COUNTS ==="
echo "--- 4.1 P6_CHAIN LOG LAST 5 LINES ---"
if [ -f raw_s7/p6_chain.log ]; then
  tail -n 5 raw_s7/p6_chain.log
else
  echo "raw_s7/p6_chain.log not found"
fi

echo "--- 4.2 P6 JSONL FILES LISTING & LINE COUNTS (NO CONTENT READ) ---"
ls -la --time-style=full-iso results_s7/p6_gen_*.jsonl 2>/dev/null || true
wc -l results_s7/p6_gen_*.jsonl 2>/dev/null || true

echo "=== TASK-0051 COLLECTION FINISHED ==="
"""

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
    sftp_batch = os.path.join(root, "scratch", "upload_task51.sftp")
    out_path = os.path.join(root, "scratch", "task0051_out.txt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Step 1: SFTP Upload of scripts/s7deq_soft_v2_launch_v3.sh
    print("=== Step 1: SFTP Upload of scripts/s7deq_soft_v2_launch_v3.sh ===", flush=True)
    enforce_cooldown(log_path, 65.0)
    cmd_sftp = [sys.executable, cts_ssh, "sftp", sftp_batch, "--purpose", "task0051_upload_launch_v3"]
    res1 = subprocess.run(cmd_sftp)
    if res1.returncode != 0:
        print(f"[ERROR] SFTP upload failed with exit code {res1.returncode}")
        sys.exit(res1.returncode)
    print("SFTP upload finished successfully.", flush=True)

    # Step 2: Enforce Cooldown before Step 3
    print("=== Step 2: Enforcing 65s Cooldown (R7-1 compliance) ===", flush=True)
    enforce_cooldown(log_path, 65.0)

    # Step 3: Remote Collection Execution
    print("=== Step 3: Remote Collection Execution ===", flush=True)
    cmd_ssh = [sys.executable, cts_ssh, "ssh", REMOTE_SCRIPT, "--purpose", "task0051_collect_facts_and_r1"]
    with open(out_path, "w", encoding="utf-8") as f:
        res2 = subprocess.run(cmd_ssh, stdout=f, stderr=subprocess.STDOUT)

    print(f"Remote execution finished with exit code {res2.returncode}. Output saved to {out_path}", flush=True)
    sys.exit(res2.returncode)

if __name__ == "__main__":
    main()

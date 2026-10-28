#!/usr/bin/env bash
# ==============================================================================
# scripts/s7deq_soft_v2_launch_v2.sh - S7 DEQ Smooth Top-3 Switch Launcher (v2)
# Directive: CTS-428 Task 3 / PREREG_S7_DEQ_v2 §2·§5·§6 / AUDIT-0046 F2~F5
# Pinned strictly to physical GPU 1 UUID: GPU-9410691f-6857-1642-ec7c-7596970e2f48
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Memory arena and thread controls to respect ulimit -v 62914560 (Rule #381)
# Prevents glibc multi-arena virtual memory explosion during model loading
# ------------------------------------------------------------------------------
export MALLOC_ARENA_MAX=1
export MALLOC_TRIM_THRESHOLD_=131072
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export RAYON_NUM_THREADS=4

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

LOG_FILE="raw_s7/s7deq_soft_v2.log"
mkdir -p raw_s7 results_s7 scratch

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "=== S7 DEQ SMOOTH TOP-3 SWITCH LAUNCH (v2) ===" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# 1. Verification of Items Config SHA-256
CONFIG_74="configs/s6_items_74.json"
EXPECTED_74_SHA="42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3"

if [ ! -f "${CONFIG_74}" ]; then
  echo "[ERROR] Config file ${CONFIG_74} not found. Aborting." | tee -a "${LOG_FILE}"
  exit 1
fi

ACTUAL_74_SHA=$(sha256sum "${CONFIG_74}" | awk '{print $1}')
echo "Config 74 SHA: ${ACTUAL_74_SHA}" | tee -a "${LOG_FILE}"
if [ "${ACTUAL_74_SHA}" != "${EXPECTED_74_SHA}" ]; then
  echo "[ERROR] Config 74 SHA mismatch. Expected: ${EXPECTED_74_SHA}, Got: ${ACTUAL_74_SHA}. Aborting." | tee -a "${LOG_FILE}"
  exit 1
fi

# 2. Hardware Device Binding
GPU1_UUID="GPU-9410691f-6857-1642-ec7c-7596970e2f48"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "[MODE] SMOKE=1 CPU execution mode enabled." | tee -a "${LOG_FILE}"
  export CUDA_VISIBLE_DEVICES=""
  DEVICE_ARG="cpu"
  MAX_NFE=5
  SMOKE_OUT_PT="scratch/s7deq_soft_v2_smoke_eps0.01.pt"
  SMOKE_OUT_JSON="scratch/s7deq_soft_v2_smoke_eps0.01.json"
  SMOKE_OUT_MD="scratch/s7deq_soft_v2_smoke_eps0.01.md"

  echo "Running CPU SMOKE with 2 items from ${CONFIG_74} (max_nfe=5)..." | tee -a "${LOG_FILE}"
  /home/shoon/system1_5/.venv/bin/python scripts/s7deq_soft_v2_run_v2.py \
    --items "${CONFIG_74}" \
    --max_items 2 \
    --out "${SMOKE_OUT_PT}" \
    --eps 0.01 \
    --alpha 0.25 \
    --g0 0.25 \
    --seed 2030 \
    --sigma 0.020 \
    --tol 0.0 \
    --max_nfe "${MAX_NFE}" \
    --device "${DEVICE_ARG}" \
    --config configs/zstar_canonical.yaml 2>&1 | tee -a "${LOG_FILE}"

  echo "Running analyzer on SMOKE output..." | tee -a "${LOG_FILE}"
  /home/shoon/system1_5/.venv/bin/python scripts/s7deq_soft_v2_analyze.py \
    --input_pt "${SMOKE_OUT_PT}" \
    --items_json "${CONFIG_74}" \
    --eps 0.01 \
    --alpha 0.25 \
    --out_json "${SMOKE_OUT_JSON}" \
    --out_md "${SMOKE_OUT_MD}" 2>&1 | tee -a "${LOG_FILE}"

  echo "[SMOKE-COMPLETE] SMOKE execution finished successfully." | tee -a "${LOG_FILE}"
  exit 0
fi

# 정규 GPU 1 실행 경로 (TASK-0048 전용)
export CUDA_VISIBLE_DEVICES="${GPU1_UUID}"
DEVICE_ARG="cuda:0"
MAX_NFE=1000

echo "=== GPU STATUS PRE-LAUNCH ===" | tee -a "${LOG_FILE}"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv | tee -a "${LOG_FILE}"
nvidia-smi -i 1 --query-gpu=uuid,utilization.gpu,memory.used,temperature.gpu --format=csv | tee -a "${LOG_FILE}"

# 3. Section 5 Pre-Launch Anchor Check (Code Equivalence on GPU 1, F2~F5)
echo "=== §5 PRE-LAUNCH ANCHOR CHECK (GPU 1, F2~F5) ===" | tee -a "${LOG_FILE}"
ANCHOR_REF_PT="results_s7/anchor_ref_gpu1_check.pt"
ANCHOR_SOFT_PT="results_s7/anchor_soft_eps0.02_check.pt"
ANCHOR_BASE_JSON="data/anchor20.json"

/home/shoon/system1_5/.venv/bin/python - << 'EOF' | tee -a "${LOG_FILE}"
import json
import os
import subprocess
import sys
import torch

anchor_config = "data/anchor20.json"
with open(anchor_config, "r", encoding="utf-8") as f:
    anchor_data = json.load(f)

if isinstance(anchor_data, list):
    anchors = [int(x) for x in anchor_data]
elif isinstance(anchor_data, dict) and "indices" in anchor_data:
    anchors = [int(x) for x in anchor_data["indices"]]
else:
    anchors = [int(x) for x in list(anchor_data.values())[0]]

print(f"Loaded {len(anchors)} anchor candidates from {anchor_config}")

initial_anchors = [20, 24, 32]
test_queue = list(initial_anchors)
for a in anchors:
    if a not in test_queue:
        test_queue.append(a)

zero_boundary_items = []
evaluated_items = []

ref_out = "results_s7/anchor_ref_gpu1_check.pt"
soft_out = "results_s7/anchor_soft_eps0.02_check.pt"

common_args = [
    "--alpha", "0.25",
    "--g0", "0.25",
    "--seed", "2030",
    "--sigma", "0.020",
    "--tol", "0.0",
    "--max_nfe", "1000",
    "--device", "cuda:0",
    "--config", "configs/zstar_canonical.yaml",
]

for q in test_queue:
    if len(zero_boundary_items) >= 3:
        break

    print(f"\n--- Testing Anchor Q#{q} on GPU 1 ---")
    evaluated_items.append(q)

    # 1. Canonical run
    cmd_canon = [
        "/home/shoon/system1_5/.venv/bin/python",
        "scripts/s7_zstar_lowest_run.py",
        "--items", str(q),
        "--out", ref_out,
    ] + common_args
    rc_canon = subprocess.run(cmd_canon).returncode
    if rc_canon != 0:
        print(f"[ERROR] Canonical anchor execution failed on Q#{q}")
        sys.exit(1)

    # 2. Soft runner run (eps=0.02)
    cmd_soft = [
        "/home/shoon/system1_5/.venv/bin/python",
        "scripts/s7deq_soft_v2_run_v2.py",
        "--items", str(q),
        "--out", soft_out,
        "--eps", "0.02",
    ] + common_args
    rc_soft = subprocess.run(cmd_soft).returncode
    if rc_soft != 0:
        print(f"[ERROR] Soft switch anchor execution failed on Q#{q}")
        sys.exit(1)

    # Load and check boundary step count
    ref_dict = torch.load(ref_out, map_location="cpu")
    soft_dict = torch.load(soft_out, map_location="cpu")
    key = f"q{q:04d}_a0.25"
    b_steps = soft_dict[key].get("boundary_step_count", 0)
    print(f"Anchor Q#{q}: boundary_step_count = {b_steps}")

    if b_steps == 0:
        zero_boundary_items.append(q)
        # F3 check: z_star, z_lowest, and rho_traj
        d_star = torch.max(torch.abs(ref_dict[key]["z_star"] - soft_dict[key]["z_star"])).item()
        d_lowest = torch.max(torch.abs(ref_dict[key]["z_lowest"] - soft_dict[key]["z_lowest"])).item()
        
        rho_ref = torch.tensor(ref_dict[key]["rho_traj"])
        rho_soft = torch.tensor(soft_dict[key]["rho_traj"])
        d_rho = torch.max(torch.abs(rho_ref - rho_soft)).item()

        print(f"Anchor Q#{q} (boundary=0):")
        print(f"  max |Delta_star|     = {d_star}")
        print(f"  max |Delta_lowest|   = {d_lowest}")
        print(f"  max |Delta_rho_traj| = {d_rho}")

        if d_star != 0.0 or d_lowest != 0.0 or d_rho != 0.0:
            print(f"[ERROR] Non-zero difference detected on zero boundary item Q#{q}! Aborting.")
            sys.exit(1)

print(f"\nTotal anchors evaluated: {len(evaluated_items)}")
print(f"Zero boundary anchors found ({len(zero_boundary_items)}/3): {zero_boundary_items}")

# F2 check: If fewer than 3 items have boundary_step_count == 0, exit 2
if len(zero_boundary_items) < 3:
    print(f"[ERROR] Fewer than 3 anchors with boundary_step_count == 0 found. Exiting with code 2.")
    sys.exit(2)

# F5 check: Comparison against results_s7/anchor20_gpu1.pt (reporting only)
hist_ref_pt = "results_s7/anchor20_gpu1.pt"
if os.path.exists(hist_ref_pt):
    print("\n--- F5 Reporting: Comparison with historical anchor20_gpu1.pt ---")
    hist_dict = torch.load(hist_ref_pt, map_location="cpu")
    ref_dict = torch.load(ref_out, map_location="cpu")
    for q in evaluated_items:
        key = f"q{q:04d}_a0.25"
        if key in hist_dict and key in ref_dict:
            z_hist = hist_dict[key]["z_star"].reshape(-1).to(torch.float64)
            z_curr = ref_dict[key]["z_star"].reshape(-1).to(torch.float64)
            d_hist = torch.max(torch.abs(z_hist - z_curr)).item()
            cos_hist = float(torch.nn.functional.cosine_similarity(z_hist.unsqueeze(0), z_curr.unsqueeze(0)).item())
            print(f"Anchor Q#{q} vs anchor20_gpu1.pt: max |Delta| = {d_hist:.6e}, cos = {cos_hist:.8f}")

print("\n§5 Anchor Check passed successfully.")
EOF

# 4. Main Sequence: Run eps in order {0.005, 0.01, 0.02}
EPS_VALUES=(0.005 0.01 0.02)

for EPS in "${EPS_VALUES[@]}"; do
  OUT_PT="results_s7/s7deq_soft_v2_eps${EPS}.pt"
  OUT_JSON="results_s7/s7deq_soft_v2_eps${EPS}.json"
  OUT_MD="results_s7/s7deq_soft_v2_eps${EPS}.md"

  echo "================================================================================" | tee -a "${LOG_FILE}"
  echo "=== RUNNING EPS=${EPS} AT $(date '+%Y-%m-%d %H:%M:%S KST') ===" | tee -a "${LOG_FILE}"
  echo "================================================================================" | tee -a "${LOG_FILE}"

  /home/shoon/system1_5/.venv/bin/python scripts/s7deq_soft_v2_run_v2.py \
    --items "${CONFIG_74}" \
    --out "${OUT_PT}" \
    --eps "${EPS}" \
    --alpha 0.25 \
    --g0 0.25 \
    --seed 2030 \
    --sigma 0.020 \
    --tol 0.0 \
    --max_nfe "${MAX_NFE}" \
    --device "${DEVICE_ARG}" \
    --config configs/zstar_canonical.yaml >> "${LOG_FILE}" 2>&1

  echo "Analyzing results for eps=${EPS}..." | tee -a "${LOG_FILE}"
  /home/shoon/system1_5/.venv/bin/python scripts/s7deq_soft_v2_analyze.py \
    --input_pt "${OUT_PT}" \
    --items_json "${CONFIG_74}" \
    --ref_pt "results_s6/zstar_s6_a0.25.pt" \
    --eps "${EPS}" \
    --alpha 0.25 \
    --out_json "${OUT_JSON}" \
    --out_md "${OUT_MD}" >> "${LOG_FILE}" 2>&1

  echo "Completed eps=${EPS} at $(date '+%Y-%m-%d %H:%M:%S KST')" | tee -a "${LOG_FILE}"
done

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "=== ALL S7 DEQ SOFT V2 RUNS FINISHED ===" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

#!/usr/bin/env bash
# ==============================================================================
# scripts/s7deq_soft_v2_launch_v4.sh - S7 DEQ Smooth Top-3 Switch Launcher (v4)
# Directive: CTS-428 Task 3 / PREREG_S7_DEQ_v2.1 §2·§5·§6 / AUDIT-P009 FIX-5~7
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

# ------------------------------------------------------------------------------
# FIX-6 & FIX-7: Environment and constant bindings
# In regular GPU path, all parameters are strictly readonly constants.
# Overrides are permitted ONLY when TEST_MODE=1.
# ------------------------------------------------------------------------------
if [ "${TEST_MODE:-0}" = "1" ]; then
  BASE_DIR="${BASE_DIR:-/home/shoon/cts_e1_s5}"
  EXPECTED_74_SHA="${EXPECTED_74_SHA:-42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3}"
  EXPECTED_REF_SHA="${EXPECTED_REF_SHA:-5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6}"
  PYTHON="${PYTHON:-/home/shoon/system1_5/.venv/bin/python}"
else
  readonly BASE_DIR="/home/shoon/cts_e1_s5"
  readonly EXPECTED_74_SHA="42df1f498648a57e228eca9fadb5cf10c1a0a72a78ad490acdc6821b2e5048b3"
  readonly EXPECTED_REF_SHA="5103642de5877932392db122fa81359004af27102c3cca6958613df0d565f2a6"
  readonly PYTHON="/home/shoon/system1_5/.venv/bin/python"
fi

cd "${BASE_DIR}"

LOG_FILE="raw_s7/s7deq_soft_v2.log"
mkdir -p raw_s7 results_s7 scratch

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "=== S7 DEQ SMOOTH TOP-3 SWITCH LAUNCH (v4) ===" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# FIX-6: Log Python environment
echo "Python environment:" | tee -a "${LOG_FILE}"
${PYTHON} -c "import sys,torch;print(sys.executable, torch.__version__, torch.version.cuda)" | tee -a "${LOG_FILE}"

# 1. Verification of Items Config SHA-256
CONFIG_74="configs/s6_items_74.json"

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
  ${PYTHON} scripts/s7deq_soft_v2_run_v2.py \
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
  ${PYTHON} scripts/s7deq_soft_v2_analyze.py \
    --input_pt "${SMOKE_OUT_PT}" \
    --items_json "${CONFIG_74}" \
    --eps 0.01 \
    --alpha 0.25 \
    --out_json "${SMOKE_OUT_JSON}" \
    --out_md "${SMOKE_OUT_MD}" 2>&1 | tee -a "${LOG_FILE}"

  echo "[SMOKE-COMPLETE] SMOKE execution finished successfully." | tee -a "${LOG_FILE}"
  exit 0
fi

# Regular GPU 1 execution path (CTS-428 Task 3)
export CUDA_VISIBLE_DEVICES="${GPU1_UUID}"
DEVICE_ARG="cuda:0"
MAX_NFE=1000

echo "=== GPU STATUS PRE-LAUNCH ===" | tee -a "${LOG_FILE}"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv | tee -a "${LOG_FILE}" || true
nvidia-smi -i 1 --query-gpu=uuid,utilization.gpu,memory.used,temperature.gpu --format=csv | tee -a "${LOG_FILE}" || true

# 3. Section 5 Pre-Launch Anchor Check (v2.1: eps=1e-12, FIX-1~FIX-7)
echo "=== §5 PRE-LAUNCH ANCHOR CHECK (GPU 1, v2.1 eps=1e-12) ===" | tee -a "${LOG_FILE}"
ANCHOR_REF_PT="results_s7/anchor_ref_gpu1_check.pt"
ANCHOR_SOFT_PT="results_s7/anchor_soft_eps1e-12_check.pt"
ANCHOR_EXTRA_REF_PT="results_s7/anchor_ref_gpu1_check_v21_extra.pt"
ANCHOR_BASE_JSON="data/anchor20.json"

# FIX-2: Check that results_s7/anchor_soft_eps1e-12_check.pt does NOT exist prior to run
if [ -f "${ANCHOR_SOFT_PT}" ]; then
  echo "[ERROR] Pre-check output file ${ANCHOR_SOFT_PT} already exists. Aborting with exit code 3." | tee -a "${LOG_FILE}"
  exit 3
fi

# Recommendation / FIX-5: Check that extra reference file does NOT exist prior to run
if [ -f "${ANCHOR_EXTRA_REF_PT}" ]; then
  echo "[ERROR] Pre-check extra reference file ${ANCHOR_EXTRA_REF_PT} already exists. Aborting with exit code 3." | tee -a "${LOG_FILE}"
  exit 3
fi

# FIX-1 & FIX-5: Check that reference file exists and matches expected SHA pre-check
if [ ! -f "${ANCHOR_REF_PT}" ]; then
  echo "[ERROR] Reference file ${ANCHOR_REF_PT} not found. Aborting with exit code 4." | tee -a "${LOG_FILE}"
  exit 4
fi

ACTUAL_PRE_REF_SHA=$(sha256sum "${ANCHOR_REF_PT}" | awk '{print $1}')
echo "Reference file SHA pre-check: ${ACTUAL_PRE_REF_SHA}" | tee -a "${LOG_FILE}"
if [ "${ACTUAL_PRE_REF_SHA}" != "${EXPECTED_REF_SHA}" ]; then
  echo "[ERROR] Reference file SHA mismatch pre-check. Expected: ${EXPECTED_REF_SHA}, Got: ${ACTUAL_PRE_REF_SHA}. Aborting with exit code 4." | tee -a "${LOG_FILE}"
  exit 4
fi

# Python anchor check execution
${PYTHON} - << 'EOF' | tee -a "${LOG_FILE}"
import json
import os
import subprocess
import sys
import torch

anchor_config = "data/anchor20.json"
ref_pt = "results_s7/anchor_ref_gpu1_check.pt"
extra_ref_pt = "results_s7/anchor_ref_gpu1_check_v21_extra.pt"
soft_out = "results_s7/anchor_soft_eps1e-12_check.pt"

# TEST_MODE mock handling for local tests
if os.environ.get("TEST_MODE") == "1":
    if os.environ.get("TEST_ALTER_REF") == "1":
        print("[TEST_MODE] Simulating reference file modification during run...")
        with open(ref_pt, "ab") as f:
            f.write(b"\n# modified_during_anchor_check\n")
        sys.exit(0)
    elif os.environ.get("TEST_PASS_ANCHOR") == "1":
        print("[TEST_MODE] Simulating successful anchor check...")
        sys.exit(0)

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

# FIX-1: Load reference dictionary in READ-ONLY mode
ref_dict = torch.load(ref_pt, map_location="cpu")
print(f"Loaded {len(ref_dict)} keys from read-only reference {ref_pt}")

for q in test_queue:
    if len(zero_boundary_items) >= 3:
        break

    print(f"\n--- Testing Anchor Q#{q} on GPU 1 (eps=1e-12) ---")
    evaluated_items.append(q)
    key = f"q{q:04d}_a0.25"

    # 1. Canonical reference resolution (DO NOT modify ref_pt)
    if key in ref_dict:
        canon_item = ref_dict[key]
    else:
        print(f"Key {key} not found in {ref_pt}. Computing into {extra_ref_pt}...")
        cmd_canon = [
            sys.executable,
            "scripts/s7_zstar_lowest_run.py",
            "--items", str(q),
            "--out", extra_ref_pt,
        ] + common_args
        rc_canon = subprocess.run(cmd_canon).returncode
        if rc_canon != 0:
            print(f"[ERROR] Canonical anchor execution failed on Q#{q}")
            sys.exit(1)
        extra_dict = torch.load(extra_ref_pt, map_location="cpu")
        canon_item = extra_dict[key]

    # 2. Soft runner run (eps=1e-12)
    cmd_soft = [
        sys.executable,
        "scripts/s7deq_soft_v2_run_v2.py",
        "--items", str(q),
        "--out", soft_out,
        "--eps", "1e-12",
    ] + common_args
    rc_soft = subprocess.run(cmd_soft).returncode
    if rc_soft != 0:
        print(f"[ERROR] Soft switch anchor execution failed on Q#{q}")
        sys.exit(1)

    # 3. Load and inspect soft output
    soft_dict = torch.load(soft_out, map_location="cpu")
    if key not in soft_dict:
        print(f"[ERROR] Key {key} missing from soft output {soft_out}")
        sys.exit(1)

    # FIX-2: Check that eps field is 1e-12
    key_eps = soft_dict[key].get("eps")
    if key_eps is None or abs(float(key_eps) - 1e-12) > 1e-20:
        print(f"[ERROR] Anchor Q#{q} eps field is {key_eps}, expected 1e-12! Aborting.")
        sys.exit(1)

    b_steps = soft_dict[key].get("boundary_step_count", 0)
    d_final = soft_dict[key].get("d_final")
    in_b_final = soft_dict[key].get("in_boundary_final", False)

    print(f"Anchor Q#{q}: boundary_step_count = {b_steps}, eps = {key_eps}")

    # Recommendation: Log whether final evaluation (line 381) had tie / entered boundary
    # Note: A tie at line 381 is recorded for information only; exit 1 is triggered ONLY if max |Delta| != 0.
    if in_b_final or (d_final is not None and d_final <= 1e-12):
        print(f"Anchor Q#{q} note: final evaluation (line 381) had tie/boundary condition (d_final={d_final}). Recorded only; check continues unless max|Delta| != 0.")

    if b_steps == 0:
        zero_boundary_items.append(q)
        # Check equivalence on zero boundary item
        d_star = torch.max(torch.abs(canon_item["z_star"] - soft_dict[key]["z_star"])).item()
        d_lowest = torch.max(torch.abs(canon_item["z_lowest"] - soft_dict[key]["z_lowest"])).item()

        rho_ref = torch.tensor(canon_item["rho_traj"])
        rho_soft = torch.tensor(soft_dict[key]["rho_traj"])
        d_rho = torch.max(torch.abs(rho_ref - rho_soft)).item()

        print(f"Anchor Q#{q} (boundary=0):")
        print(f"  max |Delta_star|     = {d_star}")
        print(f"  max |Delta_lowest|   = {d_lowest}")
        print(f"  max |Delta_rho_traj| = {d_rho}")

        if d_star != 0.0 or d_lowest != 0.0 or d_rho != 0.0:
            print(f"[ERROR] Non-zero difference detected on zero boundary item Q#{q}! Aborting with exit code 1.")
            if in_b_final or (d_final is not None and d_final <= 1e-12):
                print(f"[DIAGNOSTIC] Item Q#{q} non-zero difference occurred alongside final evaluation tie-breaking (d_final={d_final}).")
            sys.exit(1)

print(f"\nTotal anchors evaluated: {len(evaluated_items)}")
print(f"Zero boundary anchors found ({len(zero_boundary_items)}/3): {zero_boundary_items}")

# Exit 2: If fewer than 3 items have boundary_step_count == 0, exit 2
if len(zero_boundary_items) < 3:
    print(f"[ERROR] Fewer than 3 anchors with boundary_step_count == 0 found. Exiting with code 2.")
    sys.exit(2)

print("\n§5 Anchor Check passed successfully.")
EOF

# FIX-1 & FIX-5 post-check: Verify reference file was strictly read-only and matches expected SHA
ACTUAL_POST_REF_SHA=$(sha256sum "${ANCHOR_REF_PT}" | awk '{print $1}')
echo "Reference file SHA post-check: ${ACTUAL_POST_REF_SHA}" | tee -a "${LOG_FILE}"
if [ "${ACTUAL_POST_REF_SHA}" != "${EXPECTED_REF_SHA}" ]; then
  echo "[ERROR] Post-check reference SHA mismatch/altered! Expected: ${EXPECTED_REF_SHA}, Got: ${ACTUAL_POST_REF_SHA}. Aborting with exit code 4." | tee -a "${LOG_FILE}"
  exit 4
fi

# 4. Main Sequence: Run eps in order {0.005, 0.01, 0.02}
# FIX-7: Guard preventing GPU regular execution in TEST_MODE=1
if [ "${TEST_MODE:-0}" = "1" ]; then
  echo "[TEST_MODE] Regular GPU execution path blocked in TEST_MODE=1. Exiting with code 5." | tee -a "${LOG_FILE}"
  exit 5
fi

EPS_VALUES=(0.005 0.01 0.02)

for EPS in "${EPS_VALUES[@]}"; do
  OUT_PT="results_s7/s7deq_soft_v2_eps${EPS}.pt"
  OUT_JSON="results_s7/s7deq_soft_v2_eps${EPS}.json"
  OUT_MD="results_s7/s7deq_soft_v2_eps${EPS}.md"

  echo "================================================================================" | tee -a "${LOG_FILE}"
  echo "=== RUNNING EPS=${EPS} AT $(date '+%Y-%m-%d %H:%M:%S KST') ===" | tee -a "${LOG_FILE}"
  echo "================================================================================" | tee -a "${LOG_FILE}"

  ${PYTHON} scripts/s7deq_soft_v2_run_v2.py \
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
  ${PYTHON} scripts/s7deq_soft_v2_analyze.py \
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

#!/usr/bin/env bash
# ==============================================================================
# scripts/kstar_replay_launch_v1.sh - ①-a k* Replay Launcher (v1)
# Directive: PREREG_S7_DEQ_v1.4 §10(h) / AUDIT-0050 / TASK-0056 (B)
# Pinned to physical GPU 0 or GPU 1 (CTS-428 Principal Permitted)
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
# Argument Parsing: --gpu 0|1 (Required)
# ------------------------------------------------------------------------------
GPU_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)
      if [ $# -lt 2 ]; then
        echo "[ERROR] Missing value for --gpu. Aborting." >&2
        exit 1
      fi
      GPU_ARG="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown option: $1. Aborting." >&2
      exit 1
      ;;
  esac
done

if [ -z "${GPU_ARG}" ]; then
  echo "[ERROR] Missing required argument --gpu 0|1. Aborting." >&2
  exit 1
fi

if [ "${GPU_ARG}" != "0" ] && [ "${GPU_ARG}" != "1" ]; then
  echo "[ERROR] --gpu must be 0 or 1, got '${GPU_ARG}'. Aborting." >&2
  exit 1
fi

# Physical GPU UUID Mapping (PREREG_S7_DEQ_v1.4 §10(h) FIX-4)
readonly GPU0_UUID="GPU-71be98b0-6b81-9257-c622-97dc97d46480"
readonly GPU1_UUID="GPU-9410691f-6857-1642-ec7c-7596970e2f48"

if [ "${GPU_ARG}" = "0" ]; then
  TARGET_GPU_UUID="${GPU0_UUID}"
else
  TARGET_GPU_UUID="${GPU1_UUID}"
fi

# ------------------------------------------------------------------------------
# Environment and constant bindings
# In regular GPU path, all parameters are strictly readonly constants.
# Overrides are permitted ONLY when TEST_MODE=1 with an explicit non-server BASE_DIR.
# ------------------------------------------------------------------------------
if [ "${TEST_MODE:-0}" = "1" ]; then
  # Requirement 1: BASE_DIR must be explicitly provided and cannot be server path
  if [ -z "${BASE_DIR:-}" ]; then
    echo "[ERROR] TEST_MODE=1 requires an explicit BASE_DIR. Aborting with exit code 5." >&2
    exit 5
  fi
  case "${BASE_DIR}" in
    /home/shoon/cts_e1_s5|/home/shoon/cts_e1_s5/*)
      echo "[ERROR] TEST_MODE=1 cannot use server production directory ${BASE_DIR}. Aborting with exit code 5." >&2
      exit 5
      ;;
  esac

  PYTHON="${PYTHON:-python}"
  # Zero GPU usage in TEST_MODE
  TARGET_GPU_UUID=""
  CUDA_DEVICE="cpu"

  EXP_RUNNER_SHA="${EXP_RUNNER_SHA:-779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba}"
  EXP_CANON_SHA="${EXP_CANON_SHA:-9f73fb272353e8a307b9e628c65d1f17eaff164e2138fa3d232917958d6a6b2f}"
  EXP_CONFIG325_SHA="${EXP_CONFIG325_SHA:-550250e399ea96560b8360cdbdd539a82983a25fb435f994d0319da10a8a7862}"
  EXP_YAML_SHA="${EXP_YAML_SHA:-bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157}"
  EXP_HARNESS_SHA="${EXP_HARNESS_SHA:-5fb768a8f56add15c7cdca069f5e1a7f80d16fd231f8ecae1fca4fcf6920bf52}"
  EXP_ANALYZER_SHA="${EXP_ANALYZER_SHA:-a1d74cf6cd3c4773aa92a2d18400eb67f8a347938243fe7c8a3a36f215a47da5}"
  EXP_P4_SHA="${EXP_P4_SHA:-${EXPECTED_P4_SHA:-37ea6807d112e44f55ab401f48b6c8e7051f3ca2800aa7cd4caa04a546fb4696}}"
  EXP_INPUTS_SHA="${EXP_INPUTS_SHA:-186d5b6cc6678b4df2e999919cc39b85d77b9b437dce10da5c553841d3bea1f1}"
else
  # Requirement 2: In regular mode, strip test hooks from environment
  unset TEST_MODE TEST_PRECHECK_FAIL TEST_ALTER_SHA TEST_MOCK_RUN 2>/dev/null || true

  readonly BASE_DIR="/home/shoon/cts_e1_s5"
  readonly PYTHON="/home/shoon/system1_5/.venv/bin/python"
  CUDA_DEVICE="cuda:0"

  readonly EXP_RUNNER_SHA="779ea1f1acfe85a9801175f85d0d6f339d7179af10230c57040383c6df62bcba"
  readonly EXP_CANON_SHA="9f73fb272353e8a307b9e628c65d1f17eaff164e2138fa3d232917958d6a6b2f"
  readonly EXP_CONFIG325_SHA="550250e399ea96560b8360cdbdd539a82983a25fb435f994d0319da10a8a7862"
  readonly EXP_YAML_SHA="bd8e4e96cf971de37c518dd09bf90dae2a38c22e59da664a5786ff64c7ef2157"
  readonly EXP_HARNESS_SHA="5fb768a8f56add15c7cdca069f5e1a7f80d16fd231f8ecae1fca4fcf6920bf52"
  readonly EXP_ANALYZER_SHA="a1d74cf6cd3c4773aa92a2d18400eb67f8a347938243fe7c8a3a36f215a47da5"
  readonly EXP_P4_SHA="37ea6807d112e44f55ab401f48b6c8e7051f3ca2800aa7cd4caa04a546fb4696"
  readonly EXP_INPUTS_SHA="186d5b6cc6678b4df2e999919cc39b85d77b9b437dce10da5c553841d3bea1f1"
fi

cd "${BASE_DIR}"

LOG_FILE="raw_s7/kstar_replay_gpu${GPU_ARG}.log"
mkdir -p raw_s7 results_s7 scratch configs scripts cts/deq

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "=== ①-a k* REPLAY LAUNCH (v1) on GPU ${GPU_ARG} ===" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')" | tee -a "${LOG_FILE}"
echo "Target GPU UUID: ${TARGET_GPU_UUID:-NONE (TEST_MODE)}" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# Log Python environment
echo "Python environment:" | tee -a "${LOG_FILE}"
${PYTHON} -c "import sys; print(sys.executable, sys.version.split()[0])" | tee -a "${LOG_FILE}"

# Locate harness_A_v1.py (production path or workspace fallback)
if [ -f "/home/shoon/cts_e1/scratch/harness_A_v1.py" ]; then
  HARNESS_FILE="/home/shoon/cts_e1/scratch/harness_A_v1.py"
elif [ -f "scratch/harness_A_v1.py" ]; then
  HARNESS_FILE="scratch/harness_A_v1.py"
else
  HARNESS_FILE="/home/shoon/cts_e1/scratch/harness_A_v1.py"
fi

check_file_sha() {
  local fpath="$1"
  local expected="$2"
  local label="$3"

  if [ ! -f "${fpath}" ]; then
    echo "[ERROR] ${label} file missing: ${fpath}. Aborting with exit code 4." | tee -a "${LOG_FILE}"
    exit 4
  fi

  local actual
  actual=$(sha256sum "${fpath}" | awk '{print $1}')
  echo "${label} SHA: ${actual}" | tee -a "${LOG_FILE}"

  if [ "${actual}" != "${expected}" ]; then
    echo "[ERROR] ${label} SHA mismatch (${fpath}). Expected: ${expected}, Got: ${actual}. Aborting with exit code 4." | tee -a "${LOG_FILE}"
    exit 4
  fi
}

echo "=== Verifying 8 Prerequisite File SHAs ===" | tee -a "${LOG_FILE}"
check_file_sha "scripts/s7_zstar_lowest_run.py" "${EXP_RUNNER_SHA}" "Runner (s7_zstar_lowest_run.py)"
check_file_sha "cts/deq/zstar_canonical_lowest.py" "${EXP_CANON_SHA}" "Canonical solver (zstar_canonical_lowest.py)"
check_file_sha "configs/s7_train1000_kstar_replay_325.json" "${EXP_CONFIG325_SHA}" "Replay 325 config"
check_file_sha "configs/zstar_canonical.yaml" "${EXP_YAML_SHA}" "Canonical YAML config"
check_file_sha "${HARNESS_FILE}" "${EXP_HARNESS_SHA}" "Harness (harness_A_v1.py)"
check_file_sha "scripts/p5_promote_1a_replay_v2.py" "${EXP_ANALYZER_SHA}" "Analyzer v2"
check_file_sha "results_s7/zstar_train1000_P4.pt" "${EXP_P4_SHA}" "P4 reference pt"
check_file_sha "results_s7/p5_inputs_train1000_v3.json" "${EXP_INPUTS_SHA}" "Inputs train1000 v3 JSON"

# ------------------------------------------------------------------------------
# 2. Flag and Precheck Existence Check (PREREG_S7_DEQ_v1.4 §10(h) / O-2)
# If raw_s7/KSTAR_REPLAY_STARTED.flag exists -> exit 3
# If results_s7/kstar_replay_precheck5_gpu{0,1}.pt exists -> exit 3 (ESC target)
# ------------------------------------------------------------------------------
STARTED_FLAG="raw_s7/KSTAR_REPLAY_STARTED.flag"
DONE_FLAG="raw_s7/KSTAR_REPLAY_DONE.flag"
export PRECHECK_OUT="results_s7/kstar_replay_precheck5_gpu${GPU_ARG}.pt"

if [ -f "${STARTED_FLAG}" ]; then
  echo "[ERROR] ${STARTED_FLAG} already exists. Aborting with exit code 3." | tee -a "${LOG_FILE}"
  exit 3
fi

if [ -f "${PRECHECK_OUT}" ]; then
  echo "[ERROR] Precheck file ${PRECHECK_OUT} already exists (v1.4 Observation O-2: ESC target). Aborting with exit code 3." | tee -a "${LOG_FILE}"
  exit 3
fi

# ------------------------------------------------------------------------------
# 3. Step 1: Pre-check (5 items: Q#31, 277, 406, 711, 858) on Target GPU
# Evaluates items with canonical runner and analyzer v2 precheck capability.
# Failure of any item triggers exit code 6.
# ------------------------------------------------------------------------------
echo "================================================================================" | tee -a "${LOG_FILE}"
echo "=== Step 1: Pre-check (5 items) on GPU ${GPU_ARG} ===" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

# Observation O-1: Log full precheck command to terminal and log file
PRECHECK_CMD="CUDA_VISIBLE_DEVICES=${TARGET_GPU_UUID} ${PYTHON} scripts/s7_zstar_lowest_run.py --items 31,277,406,711,858 --out ${PRECHECK_OUT} --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --device ${CUDA_DEVICE} --config configs/zstar_canonical.yaml"
echo "Pre-check command: ${PRECHECK_CMD}" | tee -a "${LOG_FILE}"

if [ "${TEST_MODE:-0}" = "1" ]; then
  if [ "${TEST_PRECHECK_FAIL:-0}" = "1" ]; then
    echo "[TEST_MODE] Simulating pre-check failure on item Q#31..." | tee -a "${LOG_FILE}"
    ${PYTHON} -c "
import torch
pre = {}
for q, exp_k in [(31, 10), (277, 709), (406, 999), (711, 809), (858, 999)]:
    k = f'q{q:04d}_a0.25'
    pre[k] = {'z_star': torch.zeros(64, 2560), 'k_star': exp_k, 'rho_traj': [1e-3]*1000}
torch.save(pre, '${PRECHECK_OUT}')
"
  else
    echo "[TEST_MODE] Simulating pre-check passing all 5 items..." | tee -a "${LOG_FILE}"
    ${PYTHON} -c "
import torch
p4 = torch.load('results_s7/zstar_train1000_P4.pt', map_location='cpu')
pre = {}
for q, exp_k in [(31, 829), (277, 709), (406, 999), (711, 809), (858, 999)]:
    k = f'q{q:04d}_a0.25'
    min_r = float(p4[k]['min_rho'])
    traj = [min_r * 2.0] * 1000
    traj[exp_k] = min_r
    pre[k] = {'z_star': p4[k]['z_star'].clone(), 'k_star': exp_k, 'rho_traj': traj}
torch.save(pre, '${PRECHECK_OUT}')
"
  fi
else
  CUDA_VISIBLE_DEVICES="${TARGET_GPU_UUID}" ${PYTHON} scripts/s7_zstar_lowest_run.py \
    --items "31,277,406,711,858" \
    --out "${PRECHECK_OUT}" \
    --alpha 0.25 \
    --g0 0.25 \
    --seed 2030 \
    --sigma 0.020 \
    --max_nfe 1000 \
    --tol 0.0 \
    --device "${CUDA_DEVICE}" \
    --config configs/zstar_canonical.yaml 2>&1 | tee -a "${LOG_FILE}"
fi

# Pre-check evaluation via analyzer v2
echo "Evaluating pre-check using scripts/p5_promote_1a_replay_v2.py..." | tee -a "${LOG_FILE}"

${PYTHON} - << 'EOF' | tee -a "${LOG_FILE}"
import os
import sys
from scripts.p5_promote_1a_replay_v2 import evaluate_replay

pre_pt = os.environ.get("PRECHECK_OUT", "")
if not pre_pt or not os.path.exists(pre_pt):
    print(f"[ERROR] Precheck output file {pre_pt} missing. Aborting with exit code 6.")
    sys.exit(6)

p4_pt = "results_s7/zstar_train1000_P4.pt"
inputs_json = "results_s7/p5_inputs_train1000_v3.json"
cfg_325 = "configs/s7_train1000_kstar_replay_325.json"

skip_sha = (os.environ.get("TEST_MODE") == "1")

try:
    res = evaluate_replay(
        replay_pt_path=pre_pt,
        inputs_json_path=inputs_json,
        p4_pt_path=p4_pt,
        replay_config_path=cfg_325,
        precheck_pt_path=pre_pt,
        skip_sha_check=skip_sha,
    )
except Exception as e:
    print(f"[ERROR] Exception during evaluate_replay: {e}")
    sys.exit(6)

pre = res.get("precheck")
if not pre:
    print("[ERROR] Precheck results missing from evaluate_replay return. Aborting with exit code 6.")
    sys.exit(6)

all_pass = pre.get("all_pre_pass", False)
items = pre.get("items", [])

print("--- Pre-check Evaluation Table (5 items) ---")
for it in items:
    q = it.get("q_idx")
    exp_k = it.get("expected_k")
    act_k = it.get("actual_k")
    km = it.get("k_match")
    c1 = it.get("cond_i")
    c2 = it.get("cond_ii")
    p = it.get("pass")
    d_rel = it.get("diff_rel")
    d_abs = it.get("diff_abs")
    cos = it.get("cos")
    d_rel_str = f"{d_rel:.4e}" if d_rel is not None else "-"
    d_abs_str = f"{d_abs:.4e}" if d_abs is not None else "-"
    cos_str = f"{cos:.8f}" if cos is not None else "-"
    print(f"  Q#{q:04d}: expected_k={exp_k}, actual_k={act_k}, k_match={km}, cond_i={c1} (rel={d_rel_str}, abs={d_abs_str}), cond_ii={c2} (cos={cos_str}), pass={p}")

if not all_pass:
    print("[ERROR] Pre-check criterion not met for all 5 items. Aborting with exit code 6.")
    sys.exit(6)

print("[PRECHECK-PASS] Pre-check criterion met for all 5 items.")
EOF

# Check exit code of pre-check evaluation
PRECHECK_RC=$?
if [ ${PRECHECK_RC} -ne 0 ]; then
  echo "[ERROR] Pre-check failed with exit code ${PRECHECK_RC}." | tee -a "${LOG_FILE}"
  exit ${PRECHECK_RC}
fi

# ------------------------------------------------------------------------------
# 4. Step 2: Main Replay (325 items)
# ------------------------------------------------------------------------------
# Guard for TEST_MODE: block regular GPU execution unless mock test
if [ "${TEST_MODE:-0}" = "1" ] && [ "${TEST_MOCK_RUN:-0}" != "1" ]; then
  echo "[TEST_MODE] Regular GPU execution path blocked in TEST_MODE=1. Exiting with code 5." | tee -a "${LOG_FILE}"
  exit 5
fi

# Query PCI bus ID
PCI_BUS_ID=""
if command -v nvidia-smi >/dev/null 2>&1 && [ -n "${TARGET_GPU_UUID}" ]; then
  PCI_BUS_ID=$(nvidia-smi --query-gpu=gpu_uuid,pci.bus_id --format=csv,noheader 2>/dev/null | grep "${TARGET_GPU_UUID}" | awk -F', ' '{print $2}' || true)
fi
if [ -z "${PCI_BUS_ID}" ]; then
  PCI_BUS_ID="PCI-N/A"
fi

# Write KSTAR_REPLAY_STARTED.flag (Timestamp, UUID, PCI, Input SHAs)
echo "Writing ${STARTED_FLAG}..." | tee -a "${LOG_FILE}"
cat << FLAG_EOF > "${STARTED_FLAG}"
STARTED_AT: $(date '+%Y-%m-%d %H:%M:%S KST')
GPU_ARG: ${GPU_ARG}
GPU_UUID: ${TARGET_GPU_UUID:-NONE}
PCI_BUS_ID: ${PCI_BUS_ID}
INPUT_SHAS:
  scripts/s7_zstar_lowest_run.py: ${EXP_RUNNER_SHA}
  cts/deq/zstar_canonical_lowest.py: ${EXP_CANON_SHA}
  configs/s7_train1000_kstar_replay_325.json: ${EXP_CONFIG325_SHA}
  configs/zstar_canonical.yaml: ${EXP_YAML_SHA}
  ${HARNESS_FILE}: ${EXP_HARNESS_SHA}
  scripts/p5_promote_1a_replay_v2.py: ${EXP_ANALYZER_SHA}
  results_s7/zstar_train1000_P4.pt: ${EXP_P4_SHA}
  results_s7/p5_inputs_train1000_v3.json: ${EXP_INPUTS_SHA}
  ${PRECHECK_OUT}: $(sha256sum "${PRECHECK_OUT}" | awk '{print $1}')
FLAG_EOF

echo "Flag ${STARTED_FLAG} content:" | tee -a "${LOG_FILE}"
cat "${STARTED_FLAG}" | tee -a "${LOG_FILE}"

REPLAY_OUT="results_s7/zstar_s7_a0.25_train325_kstar_replay.pt"

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "=== Step 2: Main Replay (325 items) on GPU ${GPU_ARG} ===" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

REPLAY_CMD="CUDA_VISIBLE_DEVICES=${TARGET_GPU_UUID} ${PYTHON} scripts/s7_zstar_lowest_run.py --items configs/s7_train1000_kstar_replay_325.json --out ${REPLAY_OUT} --alpha 0.25 --g0 0.25 --seed 2030 --sigma 0.020 --max_nfe 1000 --tol 0.0 --device ${CUDA_DEVICE} --config configs/zstar_canonical.yaml"
echo "Replay command: ${REPLAY_CMD}" | tee -a "${LOG_FILE}"

if [ "${TEST_MODE:-0}" = "1" ]; then
  echo "[TEST_MODE] Simulating main 325-item replay run..." | tee -a "${LOG_FILE}"
  ${PYTHON} -c "
import json, torch
p4 = torch.load('results_s7/zstar_train1000_P4.pt', map_location='cpu')
with open('configs/s7_train1000_kstar_replay_325.json') as f:
    cfg = json.load(f)
inds = cfg['indices'] if isinstance(cfg, dict) and 'indices' in cfg else cfg
rep = {}
for q in inds:
    k = f'q{q:04d}_a0.25'
    min_r = float(p4[k]['min_rho'])
    traj = [min_r * 2.0] * 1000
    traj[500] = min_r
    rep[k] = {'z_star': p4[k]['z_star'].clone(), 'k_star': 500, 'rho_traj': traj}
torch.save(rep, '${REPLAY_OUT}')
"
else
  CUDA_VISIBLE_DEVICES="${TARGET_GPU_UUID}" ${PYTHON} scripts/s7_zstar_lowest_run.py \
    --items configs/s7_train1000_kstar_replay_325.json \
    --out "${REPLAY_OUT}" \
    --alpha 0.25 \
    --g0 0.25 \
    --seed 2030 \
    --sigma 0.020 \
    --max_nfe 1000 \
    --tol 0.0 \
    --device "${CUDA_DEVICE}" \
    --config configs/zstar_canonical.yaml 2>&1 | tee -a "${LOG_FILE}"
fi

# Write KSTAR_REPLAY_DONE.flag
REPLAY_OUT_SHA=$(sha256sum "${REPLAY_OUT}" | awk '{print $1}')
REPLAY_OUT_SIZE=$(wc -c < "${REPLAY_OUT}")

cat << DONE_EOF > "${DONE_FLAG}"
DONE_AT: $(date '+%Y-%m-%d %H:%M:%S KST')
OUTPUT_FILE: ${REPLAY_OUT}
OUTPUT_SHA256: ${REPLAY_OUT_SHA}
OUTPUT_BYTES: ${REPLAY_OUT_SIZE}
DONE_EOF

echo "Created ${DONE_FLAG} (SHA: ${REPLAY_OUT_SHA}, bytes: ${REPLAY_OUT_SIZE})" | tee -a "${LOG_FILE}"

# ------------------------------------------------------------------------------
# 5. Step 3: Run Analyzer v2
# ------------------------------------------------------------------------------
ANALYSIS_JSON="results_s7/p5_promote_1a_replay_v2.json"
ANALYSIS_MD="results_s7/p5_promote_1a_replay_v2.md"

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "=== Step 3: Running Analyzer v2 ===" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

ANALYZER_ARGS=(
  --replay_pt "${REPLAY_OUT}"
  --inputs_json results_s7/p5_inputs_train1000_v3.json
  --p4_pt results_s7/zstar_train1000_P4.pt
  --replay_config configs/s7_train1000_kstar_replay_325.json
  --precheck_pt "${PRECHECK_OUT}"
  --out_json "${ANALYSIS_JSON}"
  --out_md "${ANALYSIS_MD}"
)

if [ "${TEST_MODE:-0}" = "1" ]; then
  ANALYZER_ARGS+=(--skip_sha_check)
fi

${PYTHON} scripts/p5_promote_1a_replay_v2.py "${ANALYZER_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

echo "================================================================================" | tee -a "${LOG_FILE}"
echo "=== ①-a k* REPLAY COMPLETED ON GPU ${GPU_ARG} ===" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S KST')" | tee -a "${LOG_FILE}"
echo "================================================================================" | tee -a "${LOG_FILE}"

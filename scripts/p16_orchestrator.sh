#!/usr/bin/env bash
# scripts/p16_orchestrator.sh - Master Orchestrator for PREREG_S7_DEQ_v1.3 §10(f)(g)
# Directive: Principal Directive 2026-10-05 (REPORT-0016)

set -euo pipefail

BASE_DIR="/home/shoon/cts_e1_s5"
cd "${BASE_DIR}"

PYTHON="/home/shoon/system1_5/.venv/bin/python"
RAW_DIR="${BASE_DIR}/raw_s7"
RESULTS_DIR="${BASE_DIR}/results_s7"
SCRATCH_DIR="${BASE_DIR}/scratch"
CONFIGS_DIR="${BASE_DIR}/configs"

mkdir -p "${RAW_DIR}" "${RESULTS_DIR}" "${SCRATCH_DIR}" "${CONFIGS_DIR}"

LOG="${RAW_DIR}/p16_orchestrator.log"
exec > >(tee -a "${LOG}") 2>&1

echo "================================================================================"
echo "P16 MASTER ORCHESTRATOR START"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo "================================================================================"

# ------------------------------------------------------------------------------
# STEP 1: Verify PREREG_S7_DEQ_v1.3.md SHA-256
# ------------------------------------------------------------------------------
echo "--- [Step 1] Verifying PREREG_S7_DEQ_v1.3.md SHA-256 ---"
EXPECTED_V13_SHA="254beb6f9ba891723a2feb79661708a2886f974b4ab3be30800e290f82567beb"

for target in "${BASE_DIR}/prereg/PREREG_S7_DEQ_v1.3.md" "${BASE_DIR}/PREREG_S7_DEQ_v1.3.md"; do
    if [[ ! -f "${target}" ]]; then
        echo "[CRITICAL ERROR] Prerequisite file missing: ${target}" >&2
        exit 1
    fi
    ACT_SHA=$(sha256sum "${target}" | awk '{print $1}')
    if [[ "${ACT_SHA}" != "${EXPECTED_V13_SHA}" ]]; then
        echo "[CRITICAL ERROR] SHA mismatch for ${target}! Expected ${EXPECTED_V13_SHA}, got ${ACT_SHA}" >&2
        exit 1
    fi
    echo "  [OK] ${target} (${ACT_SHA})"
done

# Ensure smoke config exists
if [[ ! -f "${CONFIGS_DIR}/smoke_train1_p5.json" ]]; then
    echo "[INFO] Creating smoke_train1_p5.json with train item [0]..."
    echo "[0]" > "${CONFIGS_DIR}/smoke_train1_p5.json"
fi

# ------------------------------------------------------------------------------
# STEP 2: SMOKE=1 CPU Tests for Launchers (Sequential, ulimit -v 62914560)
# ------------------------------------------------------------------------------
echo "--- [Step 2] SMOKE=1 CPU Tests for Launchers ---"

# 2a. P5 v5 SMOKE Test
echo "Running SMOKE=1 for p5_heldout520_launch_v5.sh..."
rm -f "${SCRATCH_DIR}/P5_STARTED.flag" "${SCRATCH_DIR}/P5_DONE.flag" "${SCRATCH_DIR}/p5_smoke.pt" "${SCRATCH_DIR}/p5_smoke.log"
export MALLOC_ARENA_MAX=1
SMOKE=1 bash scripts/p5_heldout520_launch_v5.sh

if [[ ! -f "${SCRATCH_DIR}/P5_DONE.flag" ]] || [[ ! -f "${SCRATCH_DIR}/p5_smoke.pt" ]]; then
    echo "[CRITICAL ERROR] P5 v5 SMOKE test failed to produce output!" >&2
    exit 1
fi
echo "  [OK] P5 v5 SMOKE test passed successfully."

# 2b. P5b v2 SMOKE Test
echo "Running SMOKE=1 for p5b_seed2031_launch_v2.sh..."
rm -f "${SCRATCH_DIR}/P5B_STARTED.flag" "${SCRATCH_DIR}/P5B_DONE.flag" "${SCRATCH_DIR}/p5b_smoke.pt" "${SCRATCH_DIR}/p5b_smoke.log"
SMOKE=1 bash scripts/p5b_seed2031_launch_v2.sh

if [[ ! -f "${SCRATCH_DIR}/P5B_DONE.flag" ]] || [[ ! -f "${SCRATCH_DIR}/p5b_smoke.pt" ]]; then
    echo "[CRITICAL ERROR] P5b v2 SMOKE test failed to produce output!" >&2
    exit 1
fi
echo "  [OK] P5b v2 SMOKE test passed successfully."

# ------------------------------------------------------------------------------
# STEP 3: Verify GPU Availability and Sealing Rules
# ------------------------------------------------------------------------------
echo "--- [Step 3] Checking GPU Device States ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

GPU1_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
GPU2_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' ')

if [[ "${GPU1_MEM}" -gt 1024 ]]; then
    echo "[CRITICAL ERROR] GPU 1 is busy (${GPU1_MEM} MiB used)!" >&2
    exit 1
fi
if [[ "${GPU2_MEM}" -gt 1024 ]]; then
    echo "[CRITICAL ERROR] GPU 2 is busy (${GPU2_MEM} MiB used)!" >&2
    exit 1
fi
echo "  [OK] GPU 1 and GPU 2 are confirmed idle."

# ------------------------------------------------------------------------------
# STEP 4: Parallel Anchor 20 Evaluation on GPU 1 and GPU 2
# ------------------------------------------------------------------------------
echo "--- [Step 4] Parallel Anchor 20 Evaluation (Seed 2030) ---"
rm -f "${RESULTS_DIR}/anchor20_gpu1.json" "${RESULTS_DIR}/anchor20_gpu2.json"
rm -f "${RESULTS_DIR}/anchor20_gpu1.pt" "${RESULTS_DIR}/anchor20_gpu2.pt"
rm -f "${RAW_DIR}/anchor20_gpu1.log" "${RAW_DIR}/anchor20_gpu2.log"

echo "Launching Anchor 20 on GPU 1 (cuda:1)..."
${PYTHON} scripts/s7_anchor_equiv_gpu.py \
    --device cuda:1 \
    --out_json "${RESULTS_DIR}/anchor20_gpu1.json" \
    --out_pt "${RESULTS_DIR}/anchor20_gpu1.pt" \
    > "${RAW_DIR}/anchor20_gpu1.log" 2>&1 &
PID_G1=$!

echo "Launching Anchor 20 on GPU 2 (cuda:2)..."
${PYTHON} scripts/s7_anchor_equiv_gpu.py \
    --device cuda:2 \
    --out_json "${RESULTS_DIR}/anchor20_gpu2.json" \
    --out_pt "${RESULTS_DIR}/anchor20_gpu2.pt" \
    > "${RAW_DIR}/anchor20_gpu2.log" 2>&1 &
PID_G2=$!

echo "Anchor 20 PIDs: GPU 1 = ${PID_G1}, GPU 2 = ${PID_G2}"
echo "Monitoring execution progress every 30s..."

while kill -0 ${PID_G1} 2>/dev/null || kill -0 ${PID_G2} 2>/dev/null; do
    sleep 30
    G1_L=$(tail -n 1 "${RAW_DIR}/anchor20_gpu1.log" 2>/dev/null || echo "starting...")
    G2_L=$(tail -n 1 "${RAW_DIR}/anchor20_gpu2.log" 2>/dev/null || echo "starting...")
    echo "[$(date '+%H:%M:%S')] G1: ${G1_L} | G2: ${G2_L}"
done

wait ${PID_G1}
RC_G1=$?
wait ${PID_G2}
RC_G2=$?

echo "Anchor 20 Finished: GPU 1 RC=${RC_G1}, GPU 2 RC=${RC_G2}"

if [[ ${RC_G1} -ne 0 ]]; then
    echo "[CRITICAL ERROR] GPU 1 Anchor evaluation failed with RC ${RC_G1}!" >&2
    exit 1
fi
if [[ ${RC_G2} -ne 0 ]]; then
    echo "[CRITICAL ERROR] GPU 2 Anchor evaluation failed with RC ${RC_G2}!" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# STEP 5: Verification of Anchor 20 Qualification Criteria & Cross-GPU Match
# ------------------------------------------------------------------------------
echo "--- [Step 5] Evaluating Anchor Determinism Qualification ---"
${PYTHON} -c "
import json, sys, torch

def verify_qual(path, name):
    with open(path, 'r') as f:
        d = json.load(f)
    print(f'=== {name} DETERMINISM REPORT ===')
    print(f'Total items evaluated:        {d[\"total_items\"]}')
    print(f'Bitwise match (max|Δ|==0):    {d[\"bitwise_match_count\"]}')
    print(f'cos >= 0.9999 only match:     {d[\"cos_ge_09999_only_count\"]}')
    print(f'Sub-threshold (<0.9999):      {d[\"sub_threshold_count\"]}')
    print(f'Total qualified:              {d[\"total_qualified\"]}')
    print(f'Criteria met 20/20:           {d[\"criteria_met_20_of_20\"]}')
    if not d[\"criteria_met_20_of_20\"] or d[\"sub_threshold_count\"] > 0:
        print(f'[FAIL] {name} failed determinism qualification criteria!', file=sys.stderr)
        sys.exit(1)

verify_qual('${RESULTS_DIR}/anchor20_gpu1.json', 'GPU 1')
verify_qual('${RESULTS_DIR}/anchor20_gpu2.json', 'GPU 2')

# Cross-GPU Check (GPU 1 vs GPU 2)
pt1 = torch.load('${RESULTS_DIR}/anchor20_gpu1.pt', map_location='cpu')
pt2 = torch.load('${RESULTS_DIR}/anchor20_gpu2.pt', map_location='cpu')
cross_bitwise = 0
cross_cos9999 = 0
for k in pt1:
    z1 = pt1[k]['z_star'].double()
    z2 = pt2[k]['z_star'].double()
    delta = float(torch.max(torch.abs(z1 - z2)).item())
    cos = float(torch.nn.functional.cosine_similarity(z1.view(1, -1), z2.view(1, -1)).item())
    if delta == 0.0:
        cross_bitwise += 1
    elif cos >= 0.9999:
        cross_cos9999 += 1
print(f'=== CROSS-GPU (GPU 1 vs GPU 2) DETERMINISM ===')
print(f'Cross-GPU Bitwise Match:      {cross_bitwise} / {len(pt1)}')
print(f'Cross-GPU cos >= 0.9999 only: {cross_cos9999} / {len(pt1)}')
"

# ------------------------------------------------------------------------------
# STEP 6: Launch Production P5 (GPU 1) and P5b (GPU 2)
# ------------------------------------------------------------------------------
echo "--- [Step 6] Launching Production P5 and P5b ---"

# 6a. Launch P5 on GPU 1
echo "Launching P5 (held-out 520, seed 2030) on GPU 1 in tmux 'p5_heldout520_gpu1'..."
tmux new-session -d -s p5_heldout520_gpu1 "bash scripts/p5_heldout520_launch_v5.sh --device cuda:1 2>&1 | tee -a raw_s7/heldout520_P5_gpu1.log"

# Wait up to 30 seconds for P5_STARTED.flag
P5_STARTED=0
for i in {1..30}; do
    if [[ -f "${RAW_DIR}/P5_STARTED.flag" ]]; then
        echo "P5_STARTED.flag detected for GPU 1! (wait ${i}s)"
        P5_STARTED=1
        break
    fi
    sleep 1
done

if [[ ${P5_STARTED} -ne 1 ]]; then
    echo "[CRITICAL ERROR] P5_STARTED.flag was not generated on GPU 1!" >&2
    exit 1
fi

# 6b. Terminate ONLY tmux p5_waiter
echo "Terminating tmux session 'p5_waiter' (0 contact with s7_train_p4_prime)..."
if tmux has-session -t p5_waiter 2>/dev/null; then
    tmux kill-session -t p5_waiter
    echo "tmux session p5_waiter successfully terminated."
else
    echo "tmux session p5_waiter was not running."
fi

# 6c. Launch P5b on GPU 2
echo "Launching P5b (held-out 520, seed 2031) on GPU 2 in tmux 'p5b_seed2031_gpu2'..."
tmux new-session -d -s p5b_seed2031_gpu2 "bash scripts/p5b_seed2031_launch_v2.sh --device cuda:2 2>&1 | tee -a raw_s7/heldout520_P5b_seed2031_gpu2.log"

# Wait up to 30 seconds for P5B_STARTED.flag
P5B_STARTED=0
for i in {1..30}; do
    if [[ -f "${RAW_DIR}/P5B_STARTED.flag" ]]; then
        echo "P5B_STARTED.flag detected for GPU 2! (wait ${i}s)"
        P5B_STARTED=1
        break
    fi
    sleep 1
done

if [[ ${P5B_STARTED} -ne 1 ]]; then
    echo "[CRITICAL ERROR] P5B_STARTED.flag was not generated on GPU 2!" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# STEP 7: Final Execution Status Summary
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "P16 ORCHESTRATION COMPLETED SUCCESSFULLY"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Active tmux sessions:"
tmux ls
echo "Active GPU Compute Processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "================================================================================"

touch "${RESULTS_DIR}/P16_ORCHESTRATOR_DONE.flag"

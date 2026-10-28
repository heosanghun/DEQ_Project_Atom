#!/usr/bin/env bash
# ==============================================================================
# scripts/p6r_launch_v2.sh - P6R Main Gate Generation Launcher (v2)
# Directive: PREREG_S7_D_v3_8 §5-3(b)·(d) / TASK-0067 Section C / AUDIT-0067
# v2 (CC, AUDIT-0067): B1 argument fix (name vs generator args), no default launch,
#   nvidia-smi queried by UUID, refuse resume when the last output line is incomplete.
# Pinned GPUs:
#   - GPU 0 (UUID: GPU-71be98b0-6b81-9257-c622-97dc97d46480): Cond A -> C -> E1 (tmux: p6r_gpu0)
#   - GPU 1 (UUID: GPU-9410691f-6857-1642-ec7c-7596970e2f48): Cond B1 (shift=1) (tmux: p6r_gpu1)
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Arena and threading controls (Rule #381 / GLIBC virtual memory safety)
# ------------------------------------------------------------------------------
export MALLOC_ARENA_MAX=1
export MALLOC_TRIM_THRESHOLD_=131072
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export RAYON_NUM_THREADS=4

# ------------------------------------------------------------------------------
# Fixed Paths and Constants (PREREG_S7_D_v3_8 §5-3)
# ------------------------------------------------------------------------------
BASE_DIR="/home/shoon/cts_e1_s5"
RAW_DIR="${BASE_DIR}/raw_s7"
RESULTS_P6R_DIR="${BASE_DIR}/results_s7/p6r"
SCRIPTS_DIR="${BASE_DIR}/scripts"
CONFIGS_DIR="${BASE_DIR}/configs"

PYTHON="/home/shoon/system1_5/.venv/bin/python"
GENERATOR="${SCRIPTS_DIR}/s7_eval_gate_p6_v36d.py"
ORDER_FILE="${BASE_DIR}/results_s7/p6_item_order.json"

EXPECTED_V36D_SHA="b2df5112727597e028f842afb9c8fb42083d211bba5f137a7e125de9b0165ecd"
EXPECTED_ORDER_SHA="6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c"

GPU0_UUID="GPU-71be98b0-6b81-9257-c622-97dc97d46480"
GPU1_UUID="GPU-9410691f-6857-1642-ec7c-7596970e2f48"

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
compute_sha() {
    local target="$1"
    sha256sum "${target}" | awk '{print $1}'
}

verify_preconditions_common() {
    echo "[PRE-CHECK] Verifying Python interpreter..."
    if [ ! -x "${PYTHON}" ]; then
        echo "[FATAL] Python interpreter not found or not executable: ${PYTHON}" >&2
        exit 1
    fi

    echo "[PRE-CHECK] Verifying generator v36d..."
    if [ ! -f "${GENERATOR}" ]; then
        echo "[FATAL] Generator script not found: ${GENERATOR}" >&2
        exit 1
    fi
    local v36d_sha
    v36d_sha=$(compute_sha "${GENERATOR}")
    if [ "${v36d_sha}" != "${EXPECTED_V36D_SHA}" ]; then
        echo "[FATAL] Generator v36d SHA mismatch: expected ${EXPECTED_V36D_SHA}, got ${v36d_sha}" >&2
        exit 1
    fi
    echo "  [PASS] Generator v36d SHA: ${v36d_sha}"

    echo "[PRE-CHECK] Verifying canonical item order file..."
    if [ ! -f "${ORDER_FILE}" ]; then
        echo "[FATAL] Canonical order file not found: ${ORDER_FILE}" >&2
        exit 1
    fi
    local order_sha
    order_sha=$(compute_sha "${ORDER_FILE}")
    if [ "${order_sha}" != "${EXPECTED_ORDER_SHA}" ]; then
        echo "[FATAL] Canonical order file SHA mismatch: expected ${EXPECTED_ORDER_SHA}, got ${order_sha}" >&2
        exit 1
    fi
    echo "  [PASS] Canonical order file SHA: ${order_sha}"

    mkdir -p "${RAW_DIR}" "${RESULTS_P6R_DIR}"
}

check_gpu_cleanliness() {
    local gpu_idx="$1"
    local gpu_uuid="$2"
    echo "[PRE-CHECK] Checking GPU ${gpu_idx} (${gpu_uuid}) memory and compute apps..."
    
    local vram_used
    vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_uuid}" | tr -d ' ')
    if [ "${vram_used}" -ge 1024 ]; then
        echo "[FATAL] GPU ${gpu_idx} VRAM usage >= 1024 MiB (${vram_used} MiB). Aborting." >&2
        exit 1
    fi

    local compute_apps
    compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_uuid}" | tr -d ' ')
    if [ -n "${compute_apps}" ]; then
        echo "[FATAL] Active compute apps on GPU ${gpu_idx}: ${compute_apps}. Aborting." >&2
        exit 1
    fi
    echo "  [PASS] GPU ${gpu_idx} clean (VRAM: ${vram_used} MiB, apps: 0)"
}

post_condition_audit() {
    local cond="$1"
    local out_file="$2"
    local done_flag="${RAW_DIR}/P6R_GEN_${cond}_DONE.flag"

    echo "[POST-AUDIT] Auditing Condition ${cond} output: ${out_file}..."
    if [ ! -f "${out_file}" ]; then
        echo "[FATAL] Condition ${cond} output file missing: ${out_file}" >&2
        exit 1
    fi

    # Audit lines, unique IDs, JSON errors (NO content inspection)
    local audit_result
    audit_result=$("${PYTHON}" -c "
import json, sys
path = '${out_file}'
total = 0
ids = set()
errors = 0
try:
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                total += 1
                ids.add(d['id'])
            except Exception as e:
                errors += 1
    if total != 600 or len(ids) != 600 or errors != 0:
        print(f'FAIL total={total} unique_ids={len(ids)} errors={errors}')
        sys.exit(1)
    else:
        print(f'PASS total={total} unique_ids={len(ids)} errors={errors}')
except Exception as e:
    print(f'FATAL {e}')
    sys.exit(1)
")
    if [ "${audit_result%% *}" != "PASS" ]; then
        echo "[FATAL] Post-audit failed for Condition ${cond}: ${audit_result}" >&2
        exit 1
    fi
    echo "  [PASS] Post-audit passed for Condition ${cond}: ${audit_result}"

    # Touch completion flag
    echo "$(date '+%Y-%m-%d %H:%M:%S') KST Condition ${cond} completed" > "${done_flag}"
    echo "  [PASS] Created flag: ${done_flag}"
}

run_condition() {
    local cond="$1"
    local gpu_idx="$2"
    local gpu_uuid="$3"
    local gen_args=("${@:4}")
    
    local out_file="${RESULTS_P6R_DIR}/p6_gen_${cond}.jsonl"
    local log_file="${RAW_DIR}/p6r_gen_${cond}.log"
    local done_flag="${RAW_DIR}/P6R_GEN_${cond}_DONE.flag"

    if [ -f "${done_flag}" ]; then
        echo "[SKIP] Condition ${cond} already completed (${done_flag} exists)."
        return 0
    fi

    # Resume rule (§5-3(d))
    if [ -f "${out_file}" ]; then
        local existing_rows
        existing_rows=$(grep -c '^{' "${out_file}" || true)
        if [ -s "${out_file}" ] && [ "$(tail -c 1 "${out_file}" | od -An -tx1 | tr -d ' ')" != "0a" ]; then
            echo "[FATAL] Last line of ${out_file} is incomplete. Move it to scratch/ and run the 2-item precheck first (§5-3(d)). Halting." >&2
            exit 1
        fi
        echo "[RESUME] Output file ${out_file} already exists (${existing_rows} rows). Applying §5-3(d) resume rule."
    fi

    echo "================================================================================"
    echo "Starting Generation: Condition ${cond} on GPU ${gpu_idx} (${gpu_uuid})"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Output: ${out_file}"
    echo "Log: ${log_file}"
    echo "================================================================================"

    set +e
    CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${PYTHON}" "${GENERATOR}" \
        "${gen_args[@]}" \
        --order_sha "${EXPECTED_ORDER_SHA}" \
        --out "${out_file}" > "${log_file}" 2>&1
    local rc=$?
    set -e

    if [ "${rc}" -ne 0 ]; then
        echo "[FATAL] Condition ${cond} failed with return code ${rc}. Halting without retry." >&2
        echo "Tail 20 lines of ${log_file}:" >&2
        tail -n 20 "${log_file}" >&2
        exit "${rc}"
    fi

    # Run post-condition audit
    post_condition_audit "${cond}" "${out_file}"
}

# ------------------------------------------------------------------------------
# Worker Routines
# ------------------------------------------------------------------------------
run_worker_gpu0() {
    echo "=== P6R WORKER GPU 0: Condition Sequence A -> C -> E1 ==="
    verify_preconditions_common
    check_gpu_cleanliness 0 "${GPU0_UUID}"

    # Condition A
    run_condition "A" 0 "${GPU0_UUID}" --cond A

    # Condition C
    run_condition "C" 0 "${GPU0_UUID}" --cond C

    # Condition E1
    run_condition "E1" 0 "${GPU0_UUID}" --cond E1

    echo "=== P6R WORKER GPU 0 ALL CONDITIONS COMPLETE ==="
}

run_worker_gpu1() {
    echo "=== P6R WORKER GPU 1: Condition B1 (shift=1) ==="
    verify_preconditions_common
    check_gpu_cleanliness 1 "${GPU1_UUID}"

    # Condition B1 (shift=1)
    run_condition "B1" 1 "${GPU1_UUID}" --cond B --shift 1

    echo "=== P6R WORKER GPU 1 CONDITION B1 COMPLETE ==="
}

# ------------------------------------------------------------------------------
# Dry-run Routine
# ------------------------------------------------------------------------------
do_dry_run() {
    echo "================================================================================"
    echo "P6R MAIN GATE GENERATION LAUNCHER (v2) — DRY-RUN"
    echo "================================================================================"
    echo "Base Directory:     ${BASE_DIR}"
    echo "Generator Script:   ${GENERATOR}"
    echo "Expected v36d SHA:  ${EXPECTED_V36D_SHA}"
    echo "Canonical Order:    ${ORDER_FILE}"
    echo "Expected Order SHA: ${EXPECTED_ORDER_SHA}"
    echo "Output Directory:   ${RESULTS_P6R_DIR}"
    echo "Python Interpreter: ${PYTHON}"
    echo ""
    echo "GPU Assignments & Pipeline:"
    echo "  [GPU 0] UUID: ${GPU0_UUID} (tmux: p6r_gpu0)"
    echo "    Sequence: Condition A -> Condition C -> Condition E1"
    echo "    Cond A Command:"
    echo "      CUDA_VISIBLE_DEVICES=${GPU0_UUID} ${PYTHON} ${GENERATOR} --cond A --order_sha ${EXPECTED_ORDER_SHA} --out ${RESULTS_P6R_DIR}/p6_gen_A.jsonl"
    echo "    Cond C Command:"
    echo "      CUDA_VISIBLE_DEVICES=${GPU0_UUID} ${PYTHON} ${GENERATOR} --cond C --order_sha ${EXPECTED_ORDER_SHA} --out ${RESULTS_P6R_DIR}/p6_gen_C.jsonl"
    echo "    Cond E1 Command:"
    echo "      CUDA_VISIBLE_DEVICES=${GPU0_UUID} ${PYTHON} ${GENERATOR} --cond E1 --order_sha ${EXPECTED_ORDER_SHA} --out ${RESULTS_P6R_DIR}/p6_gen_E1.jsonl"
    echo ""
    echo "  [GPU 1] UUID: ${GPU1_UUID} (tmux: p6r_gpu1)"
    echo "    Sequence: Condition B1 (shift=1)"
    echo "    Cond B1 Command:"
    echo "      CUDA_VISIBLE_DEVICES=${GPU1_UUID} ${PYTHON} ${GENERATOR} --cond B --shift 1 --order_sha ${EXPECTED_ORDER_SHA} --out ${RESULTS_P6R_DIR}/p6_gen_B1.jsonl"
    echo ""
    echo "Safety & Governance:"
    echo "  - Pre-run VRAM check: < 1024 MiB and compute apps == 0 per GPU"
    echo "  - Pre-run SHA checks: v36d and p6_item_order.json validated before execution"
    echo "  - Resume rule: §5-3(d) compliant, preserves existing lines and logs resume notice"
    echo "  - Post-audit: exactly 600 lines, 600 unique IDs, 0 JSON parse errors (NO content read)"
    echo "  - Zero auto-retry on non-zero exit code"
    echo "  - Scoring is strictly excluded from launcher"
    echo "================================================================================"
    echo "[DRY-RUN COMPLETE] No files modified. No GPUs touched."
}

# ------------------------------------------------------------------------------
# Main Dispatch
# ------------------------------------------------------------------------------
ACTION="usage"
if [ $# -gt 0 ]; then
    case "$1" in
        --dry-run)
            ACTION="dry-run"
            ;;
        --worker-gpu0)
            ACTION="worker-gpu0"
            ;;
        --worker-gpu1)
            ACTION="worker-gpu1"
            ;;
        --launch)
            ACTION="launch"
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--dry-run | --launch | --worker-gpu0 | --worker-gpu1]" >&2
            exit 1
            ;;
    esac
fi

case "${ACTION}" in
    usage)
        echo "Usage: $0 [--dry-run | --launch | --worker-gpu0 | --worker-gpu1]" >&2
        exit 1
        ;;
    dry-run)
        do_dry_run
        ;;
    worker-gpu0)
        run_worker_gpu0
        ;;
    worker-gpu1)
        run_worker_gpu1
        ;;
    launch)
        verify_preconditions_common
        check_gpu_cleanliness 0 "${GPU0_UUID}"
        check_gpu_cleanliness 1 "${GPU1_UUID}"

        echo "[LAUNCH] Launching tmux session 'p6r_gpu0' on GPU 0..."
        tmux new-session -d -s p6r_gpu0 "bash ${SCRIPTS_DIR}/p6r_launch_v2.sh --worker-gpu0"

        echo "[LAUNCH] Launching tmux session 'p6r_gpu1' on GPU 1..."
        tmux new-session -d -s p6r_gpu1 "bash ${SCRIPTS_DIR}/p6r_launch_v2.sh --worker-gpu1"

        echo "[LAUNCH SUCCESS] Both pipelines launched in background tmux sessions."
        ;;
esac

#!/usr/bin/env bash
# scripts/check_p6_decoder_version.sh - P6 Decoder Checkpoint Version Gate
# Directive: TASK-0005 F3 / AUDIT-CTS-419
# Ensures that only valid v3.6 decoder checkpoints are accepted for P6 evaluation.

set -euo pipefail

verify_decoder_checkpoint() {
    local ckpt_path="$1"
    local condition_label="${2:-DECODER}"

    if [[ -z "$ckpt_path" ]]; then
        echo "[ERROR] [P6-GATE] No checkpoint path specified for ${condition_label}." >&2
        exit 1
    fi

    local filename
    filename=$(basename "$ckpt_path")

    # Enforce strictly that the file name matches _v36 pattern
    if [[ ! "$filename" =~ _v36(\.pt|_step[0-9]{4}\.pt)$ ]]; then
        echo "[ERROR] [P6-GATE] Invalid decoder checkpoint for ${condition_label}: '${ckpt_path}'" >&2
        echo "[ERROR] [P6-GATE] P6 evaluation strictly requires v3.6 checkpoints (*_v36.pt or *_v36_step*.pt)." >&2
        echo "[ERROR] [P6-GATE] Legacy v3.5/P4 checkpoints (*_P4*.pt, *_v35*.pt, or non-v36 files) are invalid and prohibited." >&2
        exit 1
    fi

    if [[ ! -f "$ckpt_path" ]]; then
        echo "[ERROR] [P6-GATE] Checkpoint file does not exist: '${ckpt_path}'" >&2
        exit 1
    fi

    echo "[PASS] [P6-GATE] Verified ${condition_label} checkpoint: '${ckpt_path}' (v3.6)"
}

# Gate verification for P6 evaluation runs (Conditions A, D, E)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -eq 0 ]]; then
        echo "Usage: $0 <path_to_decoder_checkpoint> [condition_label]"
        exit 1
    fi
    verify_decoder_checkpoint "$1" "${2:-DECODER}"
fi


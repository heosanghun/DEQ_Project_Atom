#!/usr/bin/env python3
"""scripts/s5_diag_nonconv.py - Phase 1 Non-converged Diagnostic & 1-d Determinism (Task 3)
Directive: EXE-260924-30 / 원장 #334

Specifications:
- 54 non-converged items (from Phase 1) + 6 converged items (randomly sampled) = 60 items.
- Evaluated at NFE 2000 under canonical Picard solver (tol=0.0).
- rho recorded every 10 steps -> raw_s5/diag_nonconv_trace_<runid>.log, results_s5/diag_nonconv.pt.
- Shape classification for each item:
  * monotonic decrease (단조감소)
  * plateau (정체)
  * oscillation (진동)
  * whether rho < 5e-3 was reached at step 2000.
- Criterion 1-d Verification:
  * Random 20 items out of 400 recomputed twice under canonical settings (NFE 1000, seed 2030, tol=0.0).
  * Exact bit-for-bit check: max|Delta| 20/20 == 0.0.
- Zero judgment words in output JSON.
- Diagnostic only; Phase 1 gate verdict remains immutable FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from harness_A_v1 import get_model, get_dataset
from cts.deq.zstar_canonical import compute_zstar, load_canonical_config

DEFAULT_CONFIG_PATH = os.path.join(BASE_S5_DIR, "configs", "zstar_canonical.yaml")
DEFAULT_DATASET_PATH = os.path.join(SYS_DIR, "data", "stage2", "math_train_prompts_5000.jsonl")
DEFAULT_ITEMS_PATH = os.path.join(BASE_S5_DIR, "data", "s5_items_400.json")
DEFAULT_PHASE1_DIST_PATH = os.path.join(BASE_S5_DIR, "results_s5", "phase1_dist.json")
DEFAULT_ZSTAR_PATH = os.path.join(BASE_S5_DIR, "results_s5", "zstar_s5_400.pt")
DEFAULT_RESULTS_DIR = os.path.join(BASE_S5_DIR, "results_s5")
DEFAULT_RAW_DIR = os.path.join(BASE_S5_DIR, "raw_s5")
DEFAULT_MANIFEST_PATH = os.path.join(DEFAULT_RESULTS_DIR, "MANIFEST_s5.sha256")


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def append_to_manifest(manifest_path: str, target_file: str) -> str:
    sha = compute_sha256(target_file)
    rel_name = os.path.basename(target_file)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(f"{sha}  {rel_name}\n")
    return sha


def classify_trajectory(rho_traj: List[float], burn_in: int = 100) -> str:
    """Classify trajectory shape into monotonic_decrease, plateau, or oscillation."""
    if len(rho_traj) <= burn_in:
        arr = np.array(rho_traj)
    else:
        arr = np.array(rho_traj[burn_in:])

    n = len(arr)
    if n < 10:
        return "plateau"

    diffs = np.diff(arr)
    # Signs of changes
    sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
    oscillation_rate = sign_changes / max(len(diffs) - 1, 1)

    # Relative drop from start of analysis window to end
    start_val = arr[0]
    end_val = arr[-1]
    rel_change = (start_val - end_val) / max(start_val, 1e-12)

    # Standard deviation in the second half
    half_idx = n // 2
    second_half_std = np.std(arr[half_idx:])
    second_half_mean = np.mean(arr[half_idx:])
    cv_second_half = second_half_std / max(second_half_mean, 1e-12)

    if oscillation_rate > 0.35 and cv_second_half > 0.05:
        return "oscillation"
    elif rel_change > 0.20 and arr[-1] < arr[0]:
        return "monotonic_decrease"
    else:
        return "plateau"


def run_diagnostic(
    config_path: str = DEFAULT_CONFIG_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    items_path: str = DEFAULT_ITEMS_PATH,
    phase1_dist_path: str = DEFAULT_PHASE1_DIST_PATH,
    zstar_400_path: str = DEFAULT_ZSTAR_PATH,
    results_dir: str = DEFAULT_RESULTS_DIR,
    raw_dir: str = DEFAULT_RAW_DIR,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    device_str: str = "cuda:0",
    seed: int = 2026,
    max_nfe: int = 2000,
) -> Dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    trace_log_path = os.path.join(raw_dir, f"diag_nonconv_trace_{run_id}.log")
    out_pt_path = os.path.join(results_dir, "diag_nonconv.pt")
    summary_json_path = os.path.join(results_dir, "diag_nonconv_summary.json")

    print("=" * 80)
    print("S5 Non-converged Diagnostic & 1-d Determinism (Task 3 / Directive EXE-260924-30)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device_str} | Seed: {seed} | NFE: {max_nfe}")
    print(f"Trace Log: {trace_log_path}")
    print(f"Output PT: {out_pt_path}")
    print("=" * 80, flush=True)

    # 1. Load Question Data
    with open(items_path, "r", encoding="utf-8") as f:
        items_data = json.load(f)
    train_indices = [int(x) for x in items_data.get("train_indices", [])]
    heldout_indices = [int(x) for x in items_data.get("heldout_indices", [])]
    all_400 = train_indices + heldout_indices

    with open(phase1_dist_path, "r", encoding="utf-8") as f:
        dist_data = json.load(f)

    non_conv_items = dist_data["non_converged_items"]
    non_conv_indices = sorted([int(item["q_idx"]) for item in non_conv_items])
    assert len(non_conv_indices) == 54, f"Expected 54 non-converged items, got {len(non_conv_indices)}"

    converged_indices = sorted([q for q in all_400 if q not in set(non_conv_indices)])
    assert len(converged_indices) == 346, f"Expected 346 converged items, got {len(converged_indices)}"

    # Sample 6 converged control items deterministically
    rng = np.random.default_rng(seed)
    sampled_6_conv = sorted(rng.choice(converged_indices, size=6, replace=False).tolist())
    target_60 = sorted(non_conv_indices + sampled_6_conv)
    assert len(target_60) == 60

    print(f"[SETUP] 54 non-converged items loaded.")
    print(f"[SETUP] 6 converged control items sampled (seed={seed}): {sampled_6_conv}")
    print(f"[SETUP] Total diagnostic items: {len(target_60)}", flush=True)

    # Sample 20 random items for 1-d determinism recompute
    sampled_20_det = sorted(rng.choice(all_400, size=20, replace=False).tolist())
    print(f"[SETUP] 20 random items for 1-d determinism (seed={seed}): {sampled_20_det}", flush=True)

    # 2. Load Model & Dataset
    device = torch.device(device_str)
    t0_load = time.time()
    bb = get_model(str(device))
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
    dataset = get_dataset()
    print(f"[BACKBONE] Loaded in {time.time() - t0_load:.2f}s.", flush=True)

    # 3. Open Trace Log
    trace_f = open(trace_log_path, "w", encoding="utf-8")
    trace_f.write("=== S5 PHASE 1 NON-CONVERGED DIAGNOSTIC NFE 2000 TRACE ===\n")
    trace_f.write(f"Run ID: {run_id}\n")
    trace_f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    trace_f.write(f"Device: {device_str}\n")
    trace_f.write(f"Seed: {seed}\n")
    trace_f.write(f"Items: 54 non-converged + 6 converged = 60 items\n")
    trace_f.write(f"Sampled 6 Converged Control Items: {sampled_6_conv}\n")
    trace_f.write("-" * 80 + "\n")
    trace_f.flush()

    diag_records: List[Dict[str, Any]] = []
    out_pt_dict: Dict[str, Any] = {}

    t0_total = time.time()
    for idx_i, q_idx in enumerate(target_60, 1):
        is_control = q_idx in sampled_6_conv
        p_text = dataset[q_idx]["prompt"]
        t0_item = time.time()

        z_star, rho_fp, nfe, log = compute_zstar(
            x=p_text,
            cfg=config_path,
            backbone=bb,
            w_g=w_g,
            device=device,
            tol=0.0,
            max_nfe=max_nfe,
            sigma_noise=0.020,
        )
        elapsed_item = time.time() - t0_item
        rho_traj = log["rho_traj"]

        rho_1000 = float(rho_traj[999]) if len(rho_traj) >= 1000 else float(rho_traj[-1])
        rho_2000 = float(rho_traj[1999]) if len(rho_traj) >= 2000 else float(rho_traj[-1])
        reached_conv = bool(rho_2000 < 5e-3)
        shape_type = classify_trajectory(rho_traj)

        norm_z = float(torch.norm(z_star).item())
        key = f"q{q_idx:04d}_diag"

        # Log trace every 10 steps to trace file
        trace_f.write(f"\n--- Item {idx_i:02d}/60 | Q#{q_idx:04d} | Type: {'CONTROL' if is_control else 'NON-CONV'} | Norm: {norm_z:.4f} | Time: {elapsed_item:.2f}s ---\n")
        trace_f.write(f"rho@1000: {rho_1000:.8e} | rho@2000: {rho_2000:.8e} | reached_5e-3: {reached_conv} | shape: {shape_type}\n")
        trace_f.write("Step\tRho\n")
        for s_idx in range(9, len(rho_traj), 10):
            trace_f.write(f"{s_idx + 1}\t{rho_traj[s_idx]:.8e}\n")
        trace_f.flush()

        rec = {
            "idx": idx_i,
            "q_idx": q_idx,
            "is_control": is_control,
            "norm_z": norm_z,
            "rho_at_1000": rho_1000,
            "rho_at_2000": rho_2000,
            "delta_rho": float(rho_1000 - rho_2000),
            "reached_5e_3_at_2000": reached_conv,
            "shape": shape_type,
            "wall_time_sec": float(elapsed_item),
        }
        diag_records.append(rec)

        out_pt_dict[key] = {
            "question_idx": q_idx,
            "z_star": z_star.detach().cpu(),
            "rho_traj_step10": [float(rho_traj[s]) for s in range(9, len(rho_traj), 10)],
            "rho_1000": rho_1000,
            "rho_2000": rho_2000,
            "shape": shape_type,
            "is_control": is_control,
            "reached_5e_3": reached_conv,
        }

        print(
            f"[{idx_i:02d}/60 Q#{q_idx:04d}] {'CTRL' if is_control else 'NCONV'} | "
            f"rho@1000={rho_1000:.6e} -> rho@2000={rho_2000:.6e} | "
            f"<5e-3: {'YES' if reached_conv else 'NO'} | shape={shape_type} ({elapsed_item:.2f}s)",
            flush=True,
        )

    trace_f.write("\n=== END NFE 2000 TRACE ===\n")
    trace_f.flush()

    # Save diag tensor checkpoint
    torch.save(out_pt_dict, out_pt_path)
    sha_pt = append_to_manifest(manifest_path, out_pt_path)
    sha_log = append_to_manifest(manifest_path, trace_log_path)
    print(f"[DIAG-PT] Saved {len(out_pt_dict)} items to {out_pt_path} (SHA: {sha_pt})")
    print(f"[DIAG-LOG] Saved trace log to {trace_log_path} (SHA: {sha_log})")

    # 4. Criterion 1-d Determinism Recomputation (20 random items out of 400)
    print("\n" + "=" * 80)
    print("[1-d DETERMINISM] Recomputing 20 random items twice to verify max|Delta| == 0.0")
    print("=" * 80, flush=True)

    trace_f.write("\n=== 1-D DETERMINISM RECOMPUTATION (20 ITEMS x 2 PASSES) ===\n")
    det_records: List[Dict[str, Any]] = []
    max_delta_all = 0.0
    exact_count = 0

    # Also load reference Phase 1 tensor to verify bit-for-bit against phase 1
    phase1_zstar = torch.load(zstar_400_path, map_location="cpu") if os.path.exists(zstar_400_path) else None

    for d_idx, q_idx in enumerate(sampled_20_det, 1):
        p_text = dataset[q_idx]["prompt"]
        # Pass 1
        z1, _, _, _ = compute_zstar(
            x=p_text, cfg=config_path, backbone=bb, w_g=w_g, device=device, tol=0.0, max_nfe=1000, sigma_noise=0.020
        )
        # Pass 2
        z2, _, _, _ = compute_zstar(
            x=p_text, cfg=config_path, backbone=bb, w_g=w_g, device=device, tol=0.0, max_nfe=1000, sigma_noise=0.020
        )

        diff = float(torch.max(torch.abs(z1 - z2)).item())
        if diff > max_delta_all:
            max_delta_all = diff
        is_exact = bool(diff == 0.0)
        if is_exact:
            exact_count += 1

        # Check against Phase 1 canonical tensor if available
        diff_p1 = None
        for k_cand in [f"q{q_idx:02d}_s2030", f"q{q_idx:04d}_s2030", str(q_idx)]:
            if phase1_zstar and k_cand in phase1_zstar:
                z_p1 = phase1_zstar[k_cand]["z_star"] if isinstance(phase1_zstar[k_cand], dict) else phase1_zstar[k_cand]
                diff_p1 = float(torch.max(torch.abs(z1.cpu() - z_p1.cpu())).item())
                break

        det_rec = {
            "idx": d_idx,
            "q_idx": q_idx,
            "max_abs_delta_pass1_pass2": diff,
            "max_abs_delta_vs_phase1": diff_p1,
            "exact_zero": is_exact,
        }
        det_records.append(det_rec)
        line_msg = f"[{d_idx:02d}/20 Q#{q_idx:04d}] max|Delta|={diff:.10e} (vs Phase 1: {diff_p1}) -> {'PASS' if is_exact else 'FAIL'}"
        print(line_msg, flush=True)
        trace_f.write(line_msg + "\n")

    trace_f.write(f"\n1-d Final: exact={exact_count}/20, max_delta={max_delta_all:.10e}\n")
    trace_f.write("=== END 1-D DETERMINISM ===\n")
    trace_f.close()

    # 5. Build Summary Statistics & Histogram
    shape_counts = {"monotonic_decrease": 0, "plateau": 0, "oscillation": 0}
    for r in diag_records:
        shape_counts[r["shape"]] = shape_counts.get(r["shape"], 0) + 1

    reached_counts = {
        "non_converged_reached": sum(1 for r in diag_records if not r["is_control"] and r["reached_5e_3_at_2000"]),
        "non_converged_unreached": sum(1 for r in diag_records if not r["is_control"] and not r["reached_5e_3_at_2000"]),
        "control_reached": sum(1 for r in diag_records if r["is_control"] and r["reached_5e_3_at_2000"]),
        "control_unreached": sum(1 for r in diag_records if r["is_control"] and not r["reached_5e_3_at_2000"]),
    }

    # Histogram of rho@2000 for 54 non-converged items
    nonconv_rho2000 = [r["rho_at_2000"] for r in diag_records if not r["is_control"]]
    hist_bins = {
        "<1e-3": sum(1 for v in nonconv_rho2000 if v < 1e-3),
        "[1e-3, 5e-3)": sum(1 for v in nonconv_rho2000 if 1e-3 <= v < 5e-3),
        "[5e-3, 1e-2)": sum(1 for v in nonconv_rho2000 if 5e-3 <= v < 1e-2),
        "[1e-2, 5e-2)": sum(1 for v in nonconv_rho2000 if 1e-2 <= v < 5e-2),
        ">=5e-2": sum(1 for v in nonconv_rho2000 if v >= 5e-2),
    }

    summary = {
        "metadata": {
            "directive": "EXE-260924-30 / 원장 #334 Task 3",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": device_str,
            "seed": seed,
            "max_nfe": max_nfe,
            "total_diag_items": len(target_60),
            "non_converged_count": len(non_conv_indices),
            "control_converged_count": len(sampled_6_conv),
            "determinism_recompute_count": len(sampled_20_det),
            "elapsed_total_seconds": float(time.time() - t0_total),
        },
        "shape_distribution": shape_counts,
        "reached_5e_3_summary": reached_counts,
        "rho_at_2000_histogram": hist_bins,
        "items_table_60": diag_records,
        "determinism_1d": {
            "exact_zero_count": exact_count,
            "total_tested": 20,
            "max_abs_delta": max_delta_all,
            "records": det_records,
        },
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sha_summary = append_to_manifest(manifest_path, summary_json_path)
    print(f"\n[SUMMARY] Saved Task 3 summary JSON to {summary_json_path} (SHA: {sha_summary})")

    return summary


def main():
    parser = argparse.ArgumentParser(description="S5 Task 3 Non-converged Diagnostic & 1-d Determinism.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--items", type=str, default=DEFAULT_ITEMS_PATH)
    parser.add_argument("--phase1_dist", type=str, default=DEFAULT_PHASE1_DIST_PATH)
    parser.add_argument("--zstar", type=str, default=DEFAULT_ZSTAR_PATH)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_nfe", type=int, default=2000)
    args = parser.parse_args()

    run_diagnostic(
        config_path=args.config,
        dataset_path=args.dataset,
        items_path=args.items,
        phase1_dist_path=args.phase1_dist,
        zstar_400_path=args.zstar,
        results_dir=args.results_dir,
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        device_str=args.device,
        seed=args.seed,
        max_nfe=args.max_nfe,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""scripts/s6_damping_run.py - S6 Damping Factor (alpha) DEQ Computation Script
Directive: EXE-260926-36 Task 2 (System 2 / 원장 #348)

Imports canonical entry points directly from canonical DEQ solver:
- get_model, get_dataset from harness_A_v1
- compute_zstar, load_canonical_config from cts.deq.zstar_canonical

Executes DEQ solver with damping factor alpha over designated item set.
Logs: rho@1000, rho@2000, min_rho, trajectory every 10 steps, and 4-way classification.
Outputs: results_s6/zstar_s6_a{alpha}.pt, raw_s6/s6_a{alpha}_<ID>.log
"""

import argparse
import hashlib
import json
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
DEFAULT_RESULTS_DIR = os.path.join(BASE_S5_DIR, "results_s6")
DEFAULT_RAW_DIR = os.path.join(BASE_S5_DIR, "raw_s6")
DEFAULT_MANIFEST_PATH = os.path.join(DEFAULT_RESULTS_DIR, "MANIFEST_s6.sha256")


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


def classify_trajectory_4way(rho_traj: List[float], burn_in: int = 100) -> str:
    """Classify trajectory into 4 shapes: monotonic_decrease, oscillation, plateau, increase."""
    if len(rho_traj) <= burn_in:
        arr = np.array(rho_traj, dtype=np.float64)
    else:
        arr = np.array(rho_traj[burn_in:], dtype=np.float64)

    n = len(arr)
    if n < 10:
        return "plateau"

    diffs = np.diff(arr)
    sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
    oscillation_rate = sign_changes / max(len(diffs) - 1, 1)

    start_val = float(arr[0])
    end_val = float(arr[-1])
    rel_change_drop = (start_val - end_val) / max(start_val, 1e-12)
    rel_change_rise = (end_val - start_val) / max(start_val, 1e-12)

    half_idx = n // 2
    second_half_std = float(np.std(arr[half_idx:]))
    second_half_mean = float(np.mean(arr[half_idx:]))
    cv_second_half = second_half_std / max(second_half_mean, 1e-12)

    if oscillation_rate > 0.35 and cv_second_half > 0.05:
        return "oscillation"
    elif rel_change_rise > 0.20 and end_val > start_val:
        return "increase"
    elif rel_change_drop > 0.20 and end_val < start_val:
        return "monotonic_decrease"
    else:
        return "plateau"


def load_items_from_source(items_source: str) -> List[int]:
    """Parse item indices from path or comma-separated list."""
    if os.path.exists(items_source):
        with open(items_source, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [int(x) for x in data]
        elif isinstance(data, dict):
            if "all_74" in data:
                return [int(x) for x in data["all_74"]]
            elif "control_20" in data:
                return [int(x) for x in data["control_20"]]
            elif "nonconv_54" in data:
                return [int(x) for x in data["nonconv_54"]]
            elif "indices" in data:
                return [int(x) for x in data["indices"]]
            elif "items" in data:
                return [int(x) for x in data["items"]]
            first_key = next(iter(data.keys()))
            if isinstance(data[first_key], list):
                return [int(x) for x in data[first_key]]
    else:
        return [int(x.strip()) for x in items_source.split(",") if x.strip()]
    raise ValueError(f"Unable to parse item list from {items_source}")


def run_damping_experiment(
    alpha: float,
    items: List[int],
    out_pt_path: str,
    raw_dir: str = DEFAULT_RAW_DIR,
    config_path: str = DEFAULT_CONFIG_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    device_str: str = "cuda:0",
    max_nfe: int = 2000,
    tol: float = 0.0,
    sigma_noise: float = 0.020,
    resume: bool = True,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(os.path.dirname(out_pt_path), exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    log_file_path = os.path.join(raw_dir, f"s6_a{alpha:.2f}_{run_id}.log")

    print("=" * 80)
    print(f"S6 Damping Experiment: alpha={alpha}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device_str} | Max NFE: {max_nfe} | Tol: {tol} | Sigma: {sigma_noise}")
    print(f"Total Items: {len(items)}")
    print(f"Output PT: {out_pt_path}")
    print(f"Log File: {log_file_path}")
    print("=" * 80, flush=True)

    # 1. Load Model & Dataset
    device = torch.device(device_str)
    t0_load = time.time()
    bb = get_model(str(device))
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
    dataset = get_dataset()
    print(f"Model and dataset loaded in {time.time() - t0_load:.2f}s.", flush=True)

    # 2. Setup Resume & Output Dict
    out_dict: Dict[str, Any] = {}
    partial_path = out_pt_path.replace(".pt", ".partial.pt")

    if resume:
        if os.path.exists(out_pt_path):
            out_dict = torch.load(out_pt_path, map_location="cpu")
            print(f"Loaded existing output tensor: {len(out_dict)} items", flush=True)
        elif os.path.exists(partial_path):
            out_dict = torch.load(partial_path, map_location="cpu")
            print(f"Loaded partial output tensor: {len(out_dict)} items", flush=True)

    # 3. Open Log File
    log_f = open(log_file_path, "a" if resume else "w", encoding="utf-8")
    if not resume or log_f.tell() == 0:
        log_f.write(f"=== S6 DAMPING EXPERIMENT TRACE (alpha={alpha}) ===\n")
        log_f.write(f"Run ID: {run_id}\n")
        log_f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Device: {device_str} | Max NFE: {max_nfe} | Tol: {tol} | Sigma: {sigma_noise}\n")
        log_f.write(f"Target Items Count: {len(items)}\n")
        log_f.write("-" * 80 + "\n")
        log_f.flush()

    records: List[Dict[str, Any]] = []
    t0_all = time.time()

    for idx, q_idx in enumerate(items, 1):
        key = f"q{q_idx:04d}_a{alpha:.2f}"
        if resume and key in out_dict:
            print(f"[{idx:03d}/{len(items):03d}] Skipping already computed Q#{q_idx:04d}", flush=True)
            existing_rec = out_dict[key]
            records.append({
                "idx": idx,
                "q_idx": q_idx,
                "norm_z": float(existing_rec["meta"]["norm_zstar"]),
                "rho_at_1000": float(existing_rec["rho_1000"]),
                "rho_at_2000": float(existing_rec["rho_2000"]),
                "min_rho": float(existing_rec["min_rho"]),
                "shape": existing_rec["shape"],
                "converged": bool(existing_rec["converged"]),
                "wall_time_sec": float(existing_rec["meta"]["wall_time_sec"]),
            })
            continue

        p_text = dataset[q_idx]["prompt"]
        t0_q = time.time()

        # Strict canonical entry point: pass alpha via kwargs override
        z_star, rho_fp, nfe, log = compute_zstar(
            x=p_text,
            cfg=config_path,
            backbone=bb,
            w_g=w_g,
            device=device,
            tol=tol,
            max_nfe=max_nfe,
            sigma_noise=sigma_noise,
            alpha=alpha,
        )
        elapsed_q = time.time() - t0_q

        rho_traj = log["rho_traj"]
        rho_1000 = float(rho_traj[999]) if len(rho_traj) >= 1000 else float(rho_traj[-1])
        rho_2000 = float(rho_traj[1999]) if len(rho_traj) >= 2000 else float(rho_traj[-1])
        min_rho = float(min(rho_traj))
        shape_type = classify_trajectory_4way(rho_traj)
        converged = bool(rho_fp < 5e-3 and nfe > 0)
        norm_z = float(torch.norm(z_star).item())

        # Write to log file
        log_f.write(f"\n--- Item {idx:03d}/{len(items):03d} | Q#{q_idx:04d} | Alpha: {alpha:.2f} | Norm: {norm_z:.4f} | Time: {elapsed_q:.2f}s ---\n")
        log_f.write(f"rho@1000: {rho_1000:.8e} | rho@2000: {rho_2000:.8e} | min_rho: {min_rho:.8e} | reached_5e-3: {converged} | shape: {shape_type}\n")
        log_f.write("Step\tRho\n")
        for s_idx in range(9, len(rho_traj), 10):
            log_f.write(f"{s_idx + 1}\t{rho_traj[s_idx]:.8e}\n")
        if len(rho_traj) % 10 != 0:
            log_f.write(f"{len(rho_traj)}\t{rho_traj[-1]:.8e}\n")
        log_f.flush()

        print(
            f"[{idx:03d}/{len(items):03d}] Q#{q_idx:04d} (alpha={alpha:.2f}) | "
            f"rho@1000={rho_1000:.4e} | rho@2000={rho_2000:.4e} | min={min_rho:.4e} | "
            f"shape={shape_type} | norm={norm_z:.2f} | {elapsed_q:.1f}s",
            flush=True,
        )

        meta = {
            "question_idx": q_idx,
            "key": key,
            "alpha": float(alpha),
            "max_nfe": int(max_nfe),
            "nfe": int(nfe),
            "sigma": float(sigma_noise),
            "rho_fp": float(rho_fp),
            "final_rho_fp": float(rho_fp),
            "norm_zstar": norm_z,
            "converged": converged,
            "wall_time_sec": float(elapsed_q),
        }

        entry = {
            "z_star": z_star.detach().cpu(),
            "rho": float(rho_fp),
            "final_rho_fp": float(rho_fp),
            "nfe": int(nfe),
            "converged": converged,
            "rho_1000": rho_1000,
            "rho_2000": rho_2000,
            "min_rho": min_rho,
            "shape": shape_type,
            "alpha": float(alpha),
            "meta": meta,
        }
        out_dict[key] = entry

        rec = {
            "idx": idx,
            "q_idx": q_idx,
            "norm_z": norm_z,
            "rho_at_1000": rho_1000,
            "rho_at_2000": rho_2000,
            "min_rho": min_rho,
            "shape": shape_type,
            "converged": converged,
            "wall_time_sec": float(elapsed_q),
        }
        records.append(rec)

        # Save partial tensor every item
        torch.save(out_dict, partial_path)

    log_f.close()

    # Save final tensor
    torch.save(out_dict, out_pt_path)
    if os.path.exists(partial_path):
        try:
            os.remove(partial_path)
        except Exception:
            pass

    # Save summary JSON
    summary_path = out_pt_path.replace(".pt", "_summary.json")
    shapes = [r["shape"] for r in records]
    summary_data = {
        "metadata": {
            "alpha": float(alpha),
            "total_items": len(items),
            "max_nfe": max_nfe,
            "device": device_str,
            "run_id": run_id,
            "total_time_sec": time.time() - t0_all,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "shape_distribution": {
            "monotonic_decrease": sum(1 for s in shapes if s == "monotonic_decrease"),
            "oscillation": sum(1 for s in shapes if s == "oscillation"),
            "plateau": sum(1 for s in shapes if s == "plateau"),
            "increase": sum(1 for s in shapes if s == "increase"),
        },
        "reached_5e_3_at_1000": sum(1 for r in records if r["rho_at_1000"] < 5e-3),
        "reached_5e_3_at_2000": sum(1 for r in records if r["rho_at_2000"] < 5e-3),
        "items": records,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Compute SHAs
    sha_pt = compute_sha256(out_pt_path)
    sha_log = compute_sha256(log_file_path)
    sha_sum = compute_sha256(summary_path)

    print("-" * 80)
    print(f"Completed S6 run for alpha={alpha} in {time.time() - t0_all:.1f}s.")
    print(f"Output PT SHA-256: {sha_pt} ({out_pt_path})")
    print(f"Log File SHA-256: {sha_log} ({log_file_path})")
    print(f"Summary JSON SHA-256: {sha_sum} ({summary_path})")
    print("-" * 80, flush=True)

    return {
        "alpha": alpha,
        "out_pt_path": out_pt_path,
        "sha_pt": sha_pt,
        "log_file_path": log_file_path,
        "sha_log": sha_log,
        "summary_path": summary_path,
        "sha_sum": sha_sum,
        "total_items": len(items),
        "shape_distribution": summary_data["shape_distribution"],
    }


def main():
    parser = argparse.ArgumentParser(description="S6 Damping Factor Experiment (EXE-260926-36)")
    parser.add_argument("--alpha", type=float, required=True, help="Damping factor alpha (e.g. 0.35, 0.25, 0.5)")
    parser.add_argument("--items", type=str, default=os.path.join(BASE_S5_DIR, "configs", "s6_items_74.json"),
                        help="Path to JSON file containing items or comma-separated indices")
    parser.add_argument("--max_nfe", type=int, default=2000, help="Maximum Picard iterations (default 2000)")
    parser.add_argument("--tol", type=float, default=0.0, help="Convergence tolerance (default 0.0)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (cuda:0 or cpu)")
    parser.add_argument("--out", type=str, default=None, help="Output .pt file path")
    parser.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_DIR, help="Raw log directory")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to canonical yaml config")
    parser.add_argument("--sigma", type=float, default=0.020, help="Perturbation sigma noise (default 0.020)")
    parser.add_argument("--no_resume", action="store_true", help="Disable resuming from partial results")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST_PATH, help="Manifest file path")

    args = parser.parse_args()

    items = load_items_from_source(args.items)
    if args.out is None:
        args.out = os.path.join(DEFAULT_RESULTS_DIR, f"zstar_s6_a{args.alpha:.2f}.pt")

    res = run_damping_experiment(
        alpha=args.alpha,
        items=items,
        out_pt_path=args.out,
        raw_dir=args.raw_dir,
        config_path=args.config,
        device_str=args.device,
        max_nfe=args.max_nfe,
        tol=args.tol,
        sigma_noise=args.sigma,
        resume=not args.no_resume,
    )

    # Append to manifest
    append_to_manifest(args.manifest, res["out_pt_path"])
    print(f"Recorded to manifest: {args.manifest}")


if __name__ == "__main__":
    main()

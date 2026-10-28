#!/usr/bin/env python3
"""scripts/run_local_s7_zstar.py - Local RTX 4090 S7 z* Runner with Continuous SSH Polling
User Directive:
- Run S7 z* calculation locally on RTX 4090 GPU (alpha=0.25, max_nfe=1000, seed=2030).
- Concurrently poll remote SSH server (shoon@<GPU_SERVER_IP>:<SSH_PORT>).
- As soon as SSH connectivity is restored, save state, log flag, and exit cleanly for server transition.
- Zero overwriting of server destination files: saves to results_s7/local_zstar_s7_a0.25_train680.pt.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.model.gemma_loader import load_gemma4_e4b
from cts.train.lora_compat import apply_paper_lora
from cts.deq.zstar_canonical import compute_zstar

BASE_DIR = str(Path(__file__).resolve().parent.parent)
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "zstar_canonical.yaml")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "stage2", "math_train_prompts_5000.jsonl")
DEFAULT_SNAPSHOT_PATH = os.path.join(BASE_DIR, "snapshots", "58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377")
DEFAULT_CKPT_PATH = os.path.join(BASE_DIR, "artifacts", "stage1_last.lora_only.pt")
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results_s7")
DEFAULT_RAW_DIR = os.path.join(BASE_DIR, "raw_s7")


def check_ssh_reachability(timeout: int = 3) -> bool:
    """Check if remote SSH server is reachable."""
    cmd = [
        "ssh", "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        "-p", "10022",
        "shoon@<GPU_SERVER_IP>",
        "echo alive"
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        return (proc.returncode == 0 and "alive" in proc.stdout)
    except Exception:
        return False


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
    """Parse item indices from JSON path or comma-separated string."""
    if os.path.exists(items_source):
        with open(items_source, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [int(x) for x in data]
        elif isinstance(data, dict):
            if "indices" in data:
                return [int(x) for x in data["indices"]]
            elif "items" in data:
                return [int(x) for x in data["items"]]
            first_key = next(iter(data.keys()))
            if isinstance(data[first_key], list):
                return [int(x) for x in data[first_key]]
    else:
        return [int(x.strip()) for x in items_source.split(",") if x.strip()]
    raise ValueError(f"Unable to parse item list from {items_source}")


def load_local_model(device_str: str = "cuda:0") -> tuple[GemmaCTSBackbone, torch.Tensor]:
    """Load canonical GemmaCTSBackbone with Stage 1 LoRA and routing weights on RTX 4090."""
    print("Loading base Gemma-4-E4B on CPU in torch.bfloat16...", flush=True)
    t0 = time.time()
    m, tok = load_gemma4_e4b(
        model_id=DEFAULT_SNAPSHOT_PATH,
        device_map=None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    bb = GemmaCTSBackbone(m, tok)
    apply_paper_lora(
        bb,
        rank=8,
        target_modules=("q_proj", "v_proj", "o_proj"),
        dropout=0.0,
        require_match=False,
        verbose=False,
    )

    print("Loading Stage-1 LoRA and routing checkpoint...", flush=True)
    ckpt = torch.load(DEFAULT_CKPT_PATH, map_location="cpu")
    sd = ckpt.get("backbone_state_dict", ckpt)
    new_sd = {}
    for k, v in sd.items():
        if k == "routing_proj":
            new_sd["routing_proj.parametrizations.weight.original"] = v
        elif k.endswith(".lora_A.weight"):
            new_sd[k[:-len(".weight")] + ".parametrizations.weight.original"] = v
        else:
            new_sd[k] = v

    bb.load_state_dict(new_sd, strict=False)

    print(f"Moving model to {device_str}...", flush=True)
    bb.to(device=device_str)
    bb.eval()
    w_g = bb.routing_matrix().to(device=device_str, dtype=torch.float32)
    print(f"Model and W_g ready on {device_str} in {time.time() - t0:.2f}s.", flush=True)
    return bb, w_g


def run_local_pipeline(
    alpha: float = 0.25,
    items_source: str = os.path.join(BASE_DIR, "configs", "s7_new_train_680.json"),
    out_pt_path: str = os.path.join(DEFAULT_RESULTS_DIR, "local_zstar_s7_a0.25_train680.pt"),
    raw_dir: str = DEFAULT_RAW_DIR,
    max_nfe: int = 1000,
    tol: float = 0.0,
    sigma_noise: float = 0.020,
    device_str: str = "cuda:0",
    start_q_idx: Optional[int] = None,
) -> None:
    os.makedirs(os.path.dirname(out_pt_path), exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(raw_dir, f"local_s7_a{alpha:.2f}_{run_id}.log")
    partial_path = out_pt_path.replace(".pt", ".partial.pt")
    flag_path = os.path.join(raw_dir, "ssh_server_available.flag")

    items = load_items_from_source(items_source)
    if start_q_idx is not None and start_q_idx in items:
        cut_idx = items.index(start_q_idx)
        items = items[cut_idx:]
        print(f"Filter: Starting from Q#{start_q_idx:04d} (remaining: {len(items)} items)", flush=True)

    print("=" * 80, flush=True)
    print("LOCAL RTX 4090 S7 z* PIPELINE LAUNCH", flush=True)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Device: {device_str} | Alpha: {alpha} | Max NFE: {max_nfe} | Tol: {tol}", flush=True)
    print(f"Total Target Items: {len(items)}", flush=True)
    print(f"Output: {out_pt_path}", flush=True)
    print(f"Log: {log_file_path}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Dataset
    print("Loading dataset...", flush=True)
    with open(DEFAULT_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    # 2. Load Model
    bb, w_g = load_local_model(device_str)

    # 3. Setup Resume
    out_dict: Dict[str, Any] = {}
    if os.path.exists(out_pt_path):
        out_dict = torch.load(out_pt_path, map_location="cpu")
        print(f"Loaded existing output tensor: {len(out_dict)} items", flush=True)
    elif os.path.exists(partial_path):
        out_dict = torch.load(partial_path, map_location="cpu")
        print(f"Loaded partial output tensor: {len(out_dict)} items", flush=True)

    # 4. Open Log File
    log_f = open(log_file_path, "a", encoding="utf-8")
    log_f.write(f"=== LOCAL RTX 4090 S7 z* TRACE (alpha={alpha}) ===\n")
    log_f.write(f"Run ID: {run_id} | Device: {device_str} | Items: {len(items)}\n")
    log_f.write("-" * 80 + "\n")
    log_f.flush()

    device = torch.device(device_str)
    t0_all = time.time()

    for idx, q_idx in enumerate(items, 1):
        key = f"q{q_idx:04d}_a{alpha:.2f}"
        if key in out_dict:
            print(f"[{idx:03d}/{len(items):03d}] Skipping already computed Q#{q_idx:04d}", flush=True)
            continue

        # Check SSH status before starting item
        if check_ssh_reachability(timeout=2):
            msg = f"\n[SSH RESTORED] Server <GPU_SERVER_IP>:<SSH_PORT> reachable at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}! Pausing local runner for server transition."
            print(msg, flush=True)
            log_f.write(msg + "\n")
            log_f.flush()
            with open(flag_path, "w", encoding="utf-8") as ff:
                ff.write(f"SSH_RESTORED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} during item {idx}/{len(items)} (Q#{q_idx:04d})\n")
            break

        p_text = dataset[q_idx]["prompt"]
        t0_q = time.time()

        # Compute z* with alpha=0.25 and max_nfe=1000
        z_star, rho_fp, nfe, log = compute_zstar(
            x=p_text,
            cfg=DEFAULT_CONFIG_PATH,
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
            f"rho@1000={rho_1000:.4e} | min={min_rho:.4e} | "
            f"shape={shape_type} | norm={norm_z:.2f} | {elapsed_q:.1f}s | VRAM={torch.cuda.memory_allocated(0)/(1024**2):.0f}MB",
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

        # Incremental save
        torch.save(out_dict, partial_path)

        # Check SSH status after item
        if check_ssh_reachability(timeout=2):
            msg = f"\n[SSH RESTORED] Server <GPU_SERVER_IP>:<SSH_PORT> reachable after Q#{q_idx:04d}! Pausing local runner for server transition."
            print(msg, flush=True)
            log_f.write(msg + "\n")
            log_f.flush()
            with open(flag_path, "w", encoding="utf-8") as ff:
                ff.write(f"SSH_RESTORED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} after Q#{q_idx:04d}\n")
            break

    log_f.close()

    # Final save if completed all items
    if len(out_dict) == len(items):
        torch.save(out_dict, out_pt_path)
        print(f"All {len(items)} items completed! Saved to {out_pt_path}", flush=True)
        if os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Run S7 z* locally on RTX 4090 with SSH Polling")
    parser.add_argument("--alpha", type=float, default=0.25, help="Damping factor alpha (default 0.25)")
    parser.add_argument("--items", type=str, default=os.path.join(BASE_DIR, "configs", "s7_new_train_680.json"))
    parser.add_argument("--max_nfe", type=int, default=1000, help="Max iterations (default 1000)")
    parser.add_argument("--out", type=str, default=os.path.join(DEFAULT_RESULTS_DIR, "local_zstar_s7_a0.25_train680.pt"))
    parser.add_argument("--start_q", type=int, default=None, help="Question index to start from (e.g. 2018 or 2022)")
    args = parser.parse_args()

    run_local_pipeline(
        alpha=args.alpha,
        items_source=args.items,
        out_pt_path=args.out,
        max_nfe=args.max_nfe,
        start_q_idx=args.start_q,
    )


if __name__ == "__main__":
    main()

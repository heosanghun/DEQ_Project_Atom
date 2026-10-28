#!/usr/bin/env python3
"""scripts/s5_compute_zstar_v2.py - S5 Canonical z* Computation Script (v2)
Directive: EXE-260923-28 rev1 (System 2 / Ledger #330)

Imports canonical entry points directly from frozen script:
diag_task20_t1_v4.py (ac313e4a84580f3df57014339e808c919afe1e1d50bdf688254f9da9b0502334)
- get_model, get_dataset from harness_A_v1
- compute_zstar, load_canonical_config from cts.deq.zstar_canonical

Zero lock stubs, zero broyden, zero hardcoded seed routes.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import torch

# Base directories
BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Direct imports matching diag_task20_t1_v4.py:38-39
from harness_A_v1 import get_model, get_dataset
from cts.deq.zstar_canonical import compute_zstar, load_canonical_config

DEFAULT_CONFIG_PATH = os.path.join(BASE_S5_DIR, "configs", "zstar_canonical.yaml")
DEFAULT_DATASET_PATH = os.path.join(SYS_DIR, "data", "stage2", "math_train_prompts_5000.jsonl")
DEFAULT_RESULTS_DIR = os.path.join(BASE_S5_DIR, "results_s5")
DEFAULT_RAW_DIR = os.path.join(BASE_S5_DIR, "raw_s5")
DEFAULT_ANCHOR_REF = os.path.join(SYS_DIR, "results", "auto24h", "zinj3_g0.25_a0.5_20q_nfe1000_20260913_084927.pt")

STAGE1_QUESTIONS: Set[int] = set(range(0, 40, 4))   # [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
STAGE6_QUESTIONS: Set[int] = set(range(40, 80, 4))  # [40, 44, 48, 52, 56, 60, 64, 68, 72, 76]
ALL_ANCHOR_20: List[int] = sorted(list(STAGE1_QUESTIONS | STAGE6_QUESTIONS))


def load_question_indices(indices_json_path: str) -> List[int]:
    """Load list of question indices from JSON file."""
    if not os.path.exists(indices_json_path):
        raise FileNotFoundError(f"Indices file not found: {indices_json_path}")
    with open(indices_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [int(x) for x in data]
    elif isinstance(data, dict):
        if "indices" in data:
            return [int(x) for x in data["indices"]]
        elif "anchor" in data:
            return [int(x) for x in data["anchor"]]
        elif "train_indices" in data and "heldout_indices" in data:
            return [int(x) for x in data["train_indices"]] + [int(x) for x in data["heldout_indices"]]
        elif "train" in data and "heldout" in data:
            return [int(x) for x in data["train"]] + [int(x) for x in data["heldout"]]
        elif "items" in data:
            return [int(x) for x in data["items"]]
        else:
            first_key = next(iter(data.keys()))
            if isinstance(data[first_key], list):
                return [int(x) for x in data[first_key]]
    raise ValueError(f"Unable to parse indices from {indices_json_path}")


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


def run_compute(
    indices: List[int],
    config_path: str,
    output_pt_path: str,
    raw_dir: str,
    device_str: str,
    is_anchor: bool = False,
    anchor_ref_path: str = DEFAULT_ANCHOR_REF,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run z* computation across the provided indices."""
    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    device = torch.device(device_str)
    cfg = load_canonical_config(config_path)
    seed = int(cfg.get("seed", 2030))
    tol = 0.0  # 정본 규약: diag_task20_t1_v4.py:106 과 동일, 1000 NFE 고정. yaml tol 은 무시(사유: 원장 #331)
    max_nfe = int(cfg.get("max_nfe", 1000))
    default_sigma = float(cfg.get("sigma_noise", 0.020))

    print("=" * 80)
    print(f"S5 Compute z* v2 (Canonical DEQ Pipeline)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Output: {output_pt_path}")
    print(f"Total items: {len(indices)} | Is Anchor: {is_anchor} | Resume: {resume}")
    print("=" * 80, flush=True)

    # Load model and dataset
    t0_load = time.time()
    bb = get_model(str(device))
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
    dataset = get_dataset()
    print(f"Model and dataset loaded in {time.time() - t0_load:.2f}s.", flush=True)

    # If anchor validation, load reference tensor
    ref_data: Optional[Dict[str, Any]] = None
    if is_anchor:
        if not os.path.exists(anchor_ref_path):
            raise FileNotFoundError(f"Anchor reference tensor not found: {anchor_ref_path}")
        ref_data = torch.load(anchor_ref_path, map_location="cpu")
        print(f"Anchor reference tensor loaded from {anchor_ref_path} (Keys: {len(ref_data)})", flush=True)

    # Resume handling
    out_dict: Dict[str, Any] = {}
    partial_path = output_pt_path.replace(".pt", ".partial.pt")
    if resume:
        if os.path.exists(output_pt_path):
            out_dict = torch.load(output_pt_path, map_location="cpu")
            print(f"Resumed {len(out_dict)} items from existing {output_pt_path}", flush=True)
        elif os.path.exists(partial_path):
            out_dict = torch.load(partial_path, map_location="cpu")
            print(f"Resumed {len(out_dict)} items from partial {partial_path}", flush=True)

    repro_records: List[Dict[str, Any]] = []
    anchor_log_lines: List[str] = []
    anchor_all_match = True

    t0_start = time.time()
    completed_count = 0

    for idx, q_idx in enumerate(indices, 1):
        key = f"q{q_idx:02d}_s{seed}"

        # If already computed, skip
        if resume and key in out_dict:
            continue

        p_text = dataset[q_idx]["prompt"]

        # Determine sigma: Anchor Stage 6 uses 1.0; otherwise canonical 0.020
        if is_anchor:
            sigma = 1.0 if q_idx in STAGE6_QUESTIONS else 0.020
        else:
            sigma = default_sigma

        t0_q = time.time()
        z_star, rho_fp, nfe, log = compute_zstar(
            x=p_text,
            cfg=config_path,
            backbone=bb,
            w_g=w_g,
            device=device,
            tol=tol,
            max_nfe=max_nfe,
            sigma_noise=sigma,
        )
        elapsed_q = time.time() - t0_q

        converged = bool(rho_fp < 5e-3 and nfe > 0)
        norm_z = float(torch.norm(z_star).item())
        z_star_cpu = z_star.detach().cpu()
        z0_cpu = log.get("z0", z_star).detach().cpu()

        meta = {
            "question_idx": q_idx,
            "key": key,
            "seed": seed,
            "sigma": sigma,
            "rho_fp": float(rho_fp),
            "final_rho_fp": float(rho_fp),
            "nfe": int(nfe),
            "converged": converged,
            "lambda1": "[?]",
            "norm_zstar": norm_z,
            "wall_time_sec": float(elapsed_q),
        }

        entry = {
            "z_star": z_star_cpu,
            "z_1000": z_star_cpu,
            "z_0": z0_cpu,
            "rho": float(rho_fp),
            "final_rho_fp": float(rho_fp),
            "nfe": int(nfe),
            "converged": converged,
            "lambda1": "[?]",
            "meta": meta,
        }
        out_dict[key] = entry
        completed_count += 1

        # Anchor comparison
        if is_anchor and ref_data is not None:
            if key not in ref_data:
                raise KeyError(f"Key {key} not found in reference tensor!")
            ref_z1000 = ref_data[key]["z_1000"].to(device)
            ref_rho = float(ref_data[key]["final_rho_fp"])
            diff_norm = float(torch.norm(z_star.to(torch.float64) - ref_z1000.to(torch.float64)).item())
            max_delta = float(torch.max(torch.abs(z_star.to(torch.float64) - ref_z1000.to(torch.float64))).item())
            cos_sim = float(torch.cosine_similarity(
                z_star.flatten().unsqueeze(0).to(torch.float64),
                ref_z1000.flatten().unsqueeze(0).to(torch.float64)
            ).item())
            is_exact = (diff_norm < 1e-4 and abs(cos_sim - 1.0) < 1e-6)
            if not is_exact:
                anchor_all_match = False
            status_str = "EXACT" if diff_norm == 0.0 else ("PASS" if is_exact else "FAIL")
            rec = {
                "idx": idx,
                "q_idx": q_idx,
                "key": key,
                "sigma": sigma,
                "norm_canon": norm_z,
                "norm_saved": float(torch.norm(ref_z1000).item()),
                "diff_norm": diff_norm,
                "max_delta": max_delta,
                "cos_sim": cos_sim,
                "rho_canon": float(rho_fp),
                "rho_saved": ref_rho,
                "time_sec": elapsed_q,
                "match": status_str,
            }
            repro_records.append(rec)
            line = f"Q#{q_idx:02d} | {key:<11} | {sigma:<5.3f} | {norm_z:<11.2f} | {rec['norm_saved']:<11.2f} | {max_delta:<10.2e} | {diff_norm:<10.2e} | {cos_sim:<12.8f} | {float(rho_fp):<10.4e} | {ref_rho:<10.4e} | {elapsed_q:<5.1f}s | {status_str}"
            anchor_log_lines.append(line)
            print(f"[{idx:02d}/{len(indices):02d} Q#{q_idx:02d}] rho={rho_fp:.4e} | max_delta={max_delta:.2e} | diff={diff_norm:.2e} | cos={cos_sim:.8f} | {status_str} | t={elapsed_q:.1f}s", flush=True)
        else:
            # Standard question logging (idx, rho, nfe, converged, seconds)
            print(f"[{idx:03d}/{len(indices):03d} Q#{q_idx:04d}] rho={rho_fp:.4e} | nfe={nfe:4d} | converged={converged} | t={elapsed_q:.1f}s", flush=True)

        # 50-item periodic partial checkpoint
        if completed_count > 0 and completed_count % 50 == 0:
            torch.save(out_dict, partial_path)
            print(f"[CHECKPOINT] Saved {len(out_dict)} items to {partial_path}", flush=True)

    # Save final tensor
    torch.save(out_dict, output_pt_path)
    total_time = time.time() - t0_start
    print(f"\nCompleted {len(out_dict)} items in {total_time:.2f}s ({total_time/60:.2f}m).", flush=True)
    print(f"Final tensor saved: {output_pt_path}", flush=True)

    # Write anchor verification log if anchor run
    if is_anchor:
        anchor_log_path = os.path.join(raw_dir, "anchor20.log")
        with open(anchor_log_path, "w", encoding="utf-8") as f:
            f.write("=== S5 ANCHOR 20 DETERMINISTIC REPRODUCTION LOG ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Reference pt: {anchor_ref_path}\n")
            f.write(f"Target pt: {output_pt_path}\n")
            f.write(f"Total items: {len(repro_records)} | All exact match: {anchor_all_match}\n")
            f.write("-" * 148 + "\n")
            f.write(f"{'Item':<5} | {'Key':<11} | {'sigma':<5} | {'||z*_canon||':<11} | {'||z*_saved||':<11} | {'max|Delta|':<10} | {'||Delta||':<10} | {'cos_sim':<12} | {'rho_canon':<10} | {'rho_saved':<10} | {'Time':<6} | {'Status'}\n")
            f.write("-" * 148 + "\n")
            for l in anchor_log_lines:
                f.write(l + "\n")
            f.write("-" * 148 + "\n")
            f.write(f"Anchor Verdict: {'SUCCESS' if anchor_all_match else 'FAIL'}\n")
            f.write("=== END ANCHOR LOG ===\n")
        print(f"Anchor log saved to: {anchor_log_path}", flush=True)

    return {
        "output_pt": output_pt_path,
        "total_items": len(out_dict),
        "total_time_sec": total_time,
        "is_anchor": is_anchor,
        "anchor_all_match": anchor_all_match if is_anchor else None,
    }


def main():
    parser = argparse.ArgumentParser(description="S5 Canonical z* Computation Script v2 (EXE-260923-28).")
    parser.add_argument("--indices_json", "--items", type=str, default=os.path.join(BASE_S5_DIR, "data", "s5_items_400.json"), help="Path to indices JSON file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Results output directory")
    parser.add_argument("--out", type=str, default="", help="Direct path to output .pt file (overrides output_dir)")
    parser.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_DIR, help="Raw logs directory")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to zstar_canonical.yaml")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device (cuda:0)")
    parser.add_argument("--anchor", action="store_true", help="Run in anchor validation mode")
    parser.add_argument("--anchor_ref", type=str, default=DEFAULT_ANCHOR_REF, help="Path to reference 20q .pt file")
    parser.add_argument("--resume", action="store_true", help="Resume from partial checkpoint")
    args = parser.parse_args()

    indices = load_question_indices(args.indices_json)

    # Auto-detect anchor run if file has anchor in name
    is_anchor = bool(args.anchor or "anchor" in os.path.basename(args.indices_json))

    # Resolve output pt path
    if args.out:
        out_pt = args.out
    else:
        if is_anchor:
            out_pt = os.path.join(args.output_dir, "zstar_s5_anchor20.pt")
        else:
            out_pt = os.path.join(args.output_dir, "zstar_s5_400.pt")

    res = run_compute(
        indices=indices,
        config_path=args.config,
        output_pt_path=out_pt,
        raw_dir=args.raw_dir,
        device_str=args.device,
        is_anchor=is_anchor,
        anchor_ref_path=args.anchor_ref,
        resume=args.resume,
    )

    if is_anchor and not res.get("anchor_all_match", False):
        print("[ERROR] Anchor verification failed: max|Delta| != 0 detected! Exit 1.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

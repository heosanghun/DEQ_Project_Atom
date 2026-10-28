#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/s7_compute_zsrc.py - CTS S7 z_src Computation for Held-Out 600 / Smoke
Directive: TASK-0009 / PREREG_S7_D_v3_7 §4·§7

Computes z_src via canonical root_fwdpass(ctx, K=64) on canonical backbone.
Supports:
  - Held-out 600 items (configs/s5_heldout_80.json + configs/s7_new_heldout_520.json)
  - Train smoke test (2 items on CPU)
Outputs:
  - results_s7/zsrc_heldout600.pt (or specified --out)
"""

import os
import sys
import time
import json
import hashlib
import argparse
from typing import Dict, Any, List
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_heldout_600_indices() -> List[int]:
    """Load and merge 80 + 520 heldout indices, sorted ascending."""
    p80 = os.path.join(BASE_S5_DIR, "configs", "s5_heldout_80.json")
    p520 = os.path.join(BASE_S5_DIR, "configs", "s7_new_heldout_520.json")
    
    with open(p80, "r", encoding="utf-8") as f:
        d80 = json.load(f)
    idx80 = [int(x) for x in d80["indices"]]
    assert len(idx80) == 80, f"Expected 80, got {len(idx80)}"
    
    with open(p520, "r", encoding="utf-8") as f:
        d520 = json.load(f)
    idx520 = [int(x) for x in d520["indices"]]
    assert len(idx520) == 520, f"Expected 520, got {len(idx520)}"
    
    combined = sorted(list(set(idx80) | set(idx520)))
    assert len(combined) == 600, f"Expected 600 unique items, got {len(combined)}"
    return combined


def load_smoke_indices() -> List[int]:
    """Load first 2 train items for CPU smoke test."""
    p_train = os.path.join(BASE_S5_DIR, "configs", "s7_train_993.json")
    if os.path.exists(p_train):
        with open(p_train, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, list):
            return [int(x) for x in d[:2]]
        elif "indices" in d:
            return [int(x) for x in d["indices"][:2]]
    # Fallback to known first 2 train indices
    return [4298, 3236]


def main():
    parser = argparse.ArgumentParser(description="Compute z_src for S7 Held-Out 600 or Smoke")
    parser.add_argument("--items", type=str, default=None, help="Path to items JSON (optional)")
    parser.add_argument("--heldout600", action="store_true", help="Load canonical 600 heldout items")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode: first 2 train items on CPU")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0 or cpu)")
    parser.add_argument("--out", type=str, default=None, help="Output .pt path")
    args = parser.parse_args()

    t0_start = time.time()
    print("=" * 80)
    print("CTS S7 — z_src Computation (scripts/s7_compute_zsrc.py)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80, flush=True)

    if args.smoke:
        target_indices = load_smoke_indices()
        default_out = os.path.join(BASE_S5_DIR, "scratch", "zsrc_smoke.pt")
        default_device = "cpu"
        print(f"[SMOKE MODE] Selected {len(target_indices)} train items: {target_indices}")
    elif args.heldout600 or args.items is None:
        target_indices = load_heldout_600_indices()
        default_out = os.path.join(BASE_S5_DIR, "results_s7", "zsrc_heldout600.pt")
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[HELDOUT-600 MODE] Loaded 600 heldout items (range: {target_indices[0]}..{target_indices[-1]})")
    else:
        with open(args.items, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, list):
            target_indices = [int(x) for x in d]
        elif "indices" in d:
            target_indices = [int(x) for x in d["indices"]]
        else:
            raise ValueError(f"Cannot parse indices from {args.items}")
        default_out = os.path.join(BASE_S5_DIR, "results_s7", f"zsrc_custom_{len(target_indices)}.pt")
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[CUSTOM MODE] Loaded {len(target_indices)} items from {args.items}")

    device_str = args.device or default_device
    out_path = args.out or default_out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Device: {device_str}")
    print(f"Target Output: {out_path}")
    print(f"Total Items: {len(target_indices)}")

    # Load canonical backbone
    from harness_A_v1 import get_model, get_dataset
    print(f"\nLoading canonical backbone on {device_str}...")
    device = torch.device(device_str)
    bb = get_model(device_str)
    bb.eval()
    dataset = get_dataset()

    zsrc_dict: Dict[str, Any] = {}
    t0_calc = time.time()

    with torch.no_grad():
        for idx, q in enumerate(target_indices, 1):
            prompt = dataset[q]["prompt"]
            ctx = bb.encode_context(prompt)
            if hasattr(ctx, "to"):
                ctx = ctx.to(device=device)
            z_src = bb.root_fwdpass(ctx, K=64, output_dtype=torch.float32).cpu()
            assert z_src.shape == (64, 2560), f"Unexpected shape {z_src.shape} for q={q}"
            assert z_src.dtype == torch.float32, f"Unexpected dtype {z_src.dtype} for q={q}"

            k_canonical = f"q{q:04d}_a0.25"
            k_short = f"q{q:04d}"
            entry = {
                "z_star": z_src,
                "meta": {
                    "question_idx": q,
                    "k_latent": 64,
                    "type": "z_src_pre_noise",
                },
            }
            zsrc_dict[k_canonical] = entry
            zsrc_dict[k_short] = entry

            if idx % 50 == 0 or idx == len(target_indices) or idx <= 5:
                elapsed = time.time() - t0_calc
                print(f"[{idx}/{len(target_indices)}] q={q:04d} done | elapsed: {elapsed:.2f}s", flush=True)

    torch.save(zsrc_dict, out_path)
    sha_out = compute_sha256(out_path)
    file_size = os.path.getsize(out_path)
    total_time = time.time() - t0_start

    print("\n" + "=" * 80)
    print("z_src Computation Complete")
    print(f"Output File: {out_path}")
    print(f"Size: {file_size:,} bytes")
    print(f"SHA-256: {sha_out}")
    print(f"Total Time: {total_time:.2f}s")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""scripts/select_s6_items.py - Select 54 Non-converged + 20 Control Converged Items for S6
Directive: EXE-260926-36 Task 1 (System 2 / 원장 #348)
"""

import os
import sys
import re
import json
import hashlib
import subprocess
from typing import Dict, Any, List
import numpy as np
import torch

BASE_DIR = "/home/shoon/cts_e1_s5"
ZSTAR_S5_400_PATH = os.path.join(BASE_DIR, "results_s5", "zstar_s5_400.pt")
DIAG_SUMMARY_PATH = os.path.join(BASE_DIR, "results_s5", "diag_nonconv_summary.json")
S5_ITEMS_400_PATH = os.path.join(BASE_DIR, "data", "s5_items_400.json")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
RAW_S6_DIR = os.path.join(BASE_DIR, "raw_s6")

def get_git_head(repo_path: str = "/home/shoon/audit_package_v1") -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "1293f2789a31a8ae449f9d10924e2bbc3d06c459"

def main():
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    os.makedirs(RAW_S6_DIR, exist_ok=True)

    # 1. Load zstar_s5_400.pt
    assert os.path.exists(ZSTAR_S5_400_PATH), f"File not found: {ZSTAR_S5_400_PATH}"
    z_dict = torch.load(ZSTAR_S5_400_PATH, map_location="cpu")
    print(f"Loaded {len(z_dict)} items from {ZSTAR_S5_400_PATH}")
    assert len(z_dict) == 400, f"Expected 400 items, got {len(z_dict)}"

    nonconv_54 = []
    conv_346 = []

    for k, v in z_dict.items():
        m = re.match(r"^q(\d+)", str(k))
        if m:
            q_idx = int(m.group(1))
        elif isinstance(v, dict) and "meta" in v and "question_idx" in v["meta"]:
            q_idx = int(v["meta"]["question_idx"])
        else:
            raise ValueError(f"Cannot parse question index from key {k}")

        converged = v.get("converged", False) if isinstance(v, dict) else False
        rho = float(v.get("rho", 1.0)) if isinstance(v, dict) else 1.0

        if not converged or rho >= 5e-3:
            nonconv_54.append(q_idx)
        else:
            conv_346.append(q_idx)

    nonconv_54 = sorted(list(set(nonconv_54)))
    conv_346 = sorted(list(set(conv_346)))

    print(f"Non-converged count: {len(nonconv_54)}")
    print(f"Converged count: {len(conv_346)}")
    assert len(nonconv_54) == 54, f"Expected 54 nonconv, got {len(nonconv_54)}"
    assert len(conv_346) == 346, f"Expected 346 conv, got {len(conv_346)}"
    assert set(nonconv_54).isdisjoint(set(conv_346)), "Overlap between nonconv and conv sets"

    # Cross-check with diag_nonconv_summary.json if exists
    if os.path.exists(DIAG_SUMMARY_PATH):
        with open(DIAG_SUMMARY_PATH, "r", encoding="utf-8") as f:
            diag_summary = json.load(f)
        diag_keys = sorted([x["q_idx"] for x in diag_summary.get("items_table_60", []) if not x.get("is_control", False)])
        assert diag_keys == nonconv_54, f"Mismatch between zstar_s5_400 nonconv ({len(nonconv_54)}) and diag_nonconv_summary ({len(diag_keys)})"
        print("Cross-check with diag_nonconv_summary.json: EXACT MATCH (54/54)")

    # Load split info from s5_items_400.json
    train_set = set()
    heldout_set = set()
    if os.path.exists(S5_ITEMS_400_PATH):
        with open(S5_ITEMS_400_PATH, "r", encoding="utf-8") as f:
            s5_items = json.load(f)
        train_set = set(s5_items.get("train_indices", []))
        heldout_set = set(s5_items.get("heldout_indices", []))
        print(f"Loaded split info: train {len(train_set)}, heldout {len(heldout_set)}")

    # 2. Sample 20 Control Converged items using EXE-25 Seed Convention
    head_commit = get_git_head()
    nonce = "S6_DAMPING_CONTROL20"
    entropy = f"CTS-random-select-v1|{head_commit}|{nonce}"
    entropy_sha = hashlib.sha256(entropy.encode("utf-8")).hexdigest()
    seed = int(entropy_sha[:16], 16)

    rng = np.random.default_rng(seed)
    sampled_control_20 = sorted([int(x) for x in rng.choice(conv_346, size=20, replace=False)])

    print(f"Head commit: {head_commit}")
    print(f"Nonce: {nonce}")
    print(f"Entropy string: {entropy}")
    print(f"Entropy SHA-256: {entropy_sha}")
    print(f"Derived seed: {seed}")
    print(f"Selected control 20: {sampled_control_20}")

    all_74 = sorted(nonconv_54 + sampled_control_20)
    assert len(all_74) == 74, f"Expected 74 total items, got {len(all_74)}"
    assert len(set(all_74)) == 74, "Duplicate items in 74 selection"

    items_detail = []
    for q in all_74:
        is_ctrl = q in sampled_control_20
        is_nc = q in nonconv_54
        split = "train" if q in train_set else ("heldout" if q in heldout_set else "unknown")
        items_detail.append({
            "q_idx": q,
            "is_control": is_ctrl,
            "is_nonconv": is_nc,
            "split": split,
        })

    # 3. Save configs
    items_74_dict = {
        "metadata": {
            "directive": "EXE-260926-36 (원장 #348)",
            "head_commit": head_commit,
            "nonce": nonce,
            "entropy": entropy,
            "entropy_sha256": entropy_sha,
            "seed": seed,
            "numpy_version": np.__version__,
            "total_items": 74,
            "nonconv_54_count": 54,
            "control_20_count": 20,
        },
        "nonconv_54": nonconv_54,
        "control_20": sampled_control_20,
        "all_74": all_74,
        "items_detail": items_detail,
    }

    path_74 = os.path.join(CONFIGS_DIR, "s6_items_74.json")
    with open(path_74, "w", encoding="utf-8") as f:
        json.dump(items_74_dict, f, indent=2)
    print(f"Saved: {path_74}")

    path_control20 = os.path.join(CONFIGS_DIR, "s6_control_20.json")
    with open(path_control20, "w", encoding="utf-8") as f:
        json.dump(sampled_control_20, f, indent=2)
    print(f"Saved: {path_control20}")

    path_nonconv54 = os.path.join(CONFIGS_DIR, "s6_nonconv_54.json")
    with open(path_nonconv54, "w", encoding="utf-8") as f:
        json.dump(nonconv_54, f, indent=2)
    print(f"Saved: {path_nonconv54}")

    # 4. Save Selection Log
    log_path = os.path.join(RAW_S6_DIR, "s6_item_selection.log")
    timestamp_str = subprocess.check_output(["date"], cwd=BASE_DIR).decode().strip()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("S6 DAMPING EXPERIMENT ITEM SELECTION LOG (EXE-260926-36 / 원장 #348)\n")
        f.write("================================================================================\n")
        f.write(f"Timestamp: {timestamp_str}\n")
        f.write(f"Git HEAD Commit: {head_commit}\n")
        f.write(f"Nonce: {nonce}\n")
        f.write(f"Entropy: {entropy}\n")
        f.write(f"Entropy SHA-256: {entropy_sha}\n")
        f.write(f"Derived Seed (uint64): {seed}\n")
        f.write(f"NumPy Version: {np.__version__}\n")
        f.write(f"Candidate Pool (Converged items in zstar_s5_400): {len(conv_346)} items\n")
        f.write(f"Non-converged items in zstar_s5_400: {len(nonconv_54)} items\n")
        f.write("-" * 80 + "\n")
        f.write(f"Selected Control 20 items (N=20):\n{sampled_control_20}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Target Non-converged 54 items (N=54):\n{nonconv_54}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Target Set (N=74):\n{all_74}\n")
        f.write("=" * 80 + "\n")
    print(f"Saved selection log: {log_path}")

    # Calculate SHAs
    sha_log = hashlib.sha256(open(log_path, "rb").read()).hexdigest()
    sha_74 = hashlib.sha256(open(path_74, "rb").read()).hexdigest()
    sha_ctrl20 = hashlib.sha256(open(path_control20, "rb").read()).hexdigest()
    sha_nonconv54 = hashlib.sha256(open(path_nonconv54, "rb").read()).hexdigest()
    print(f"Selection log SHA-256: {sha_log}")
    print(f"s6_items_74.json SHA-256: {sha_74}")
    print(f"s6_control_20.json SHA-256: {sha_ctrl20}")
    print(f"s6_nonconv_54.json SHA-256: {sha_nonconv54}")

if __name__ == "__main__":
    main()

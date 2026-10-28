#!/usr/bin/env python3
"""scripts/s5_phase1_gate.py - S5 Phase 1 Gate Verification Script
Directive: EXE-260923-28 rev2 (System 2 / Ledger #331)

Evaluates Phase 1 z* computation against PREREG_S5_v4.md Section 5 criteria:
- ①-a: Train 320 items converged ratio >= 95% (parsed from PREREG)
- ①-b: Held-out 80 items converged ratio >= 90% (parsed from PREREG)
- ①-c: Held-out pairwise mean cosine similarity <= 0.75 (parsed from PREREG)
- ①-d: 20-item determinism max|Delta| == 0 (parsed from PREREG)
- ①-e: Held-out lambda1 distribution report

Zero judgment words in JSON output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
DEFAULT_ZSTAR_PATH = os.path.join(BASE_S5_DIR, "results_s5", "zstar_s5_400.pt")
DEFAULT_PREREG_PATH = os.path.join(BASE_S5_DIR, "PREREG_S5_v4.md")
DEFAULT_ITEMS_PATH = os.path.join(BASE_S5_DIR, "data", "s5_items_400.json")
DEFAULT_OUT_PATH = os.path.join(BASE_S5_DIR, "results_s5", "phase1_gate.json")
DEFAULT_ANCHOR_LOG = os.path.join(BASE_S5_DIR, "raw_s5", "anchor20_v2.log")
FALLBACK_ANCHOR_LOG = os.path.join(BASE_S5_DIR, "raw_s5", "anchor20.log")
DEFAULT_MANIFEST = os.path.join(BASE_S5_DIR, "results_s5", "MANIFEST_s5.sha256")


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


def parse_prereg_thresholds(prereg_path: str) -> Dict[str, Any]:
    """Dynamically parse Section 5 official criteria table from PREREG markdown."""
    if not os.path.exists(prereg_path):
        raise FileNotFoundError(f"PREREG file not found: {prereg_path}")

    with open(prereg_path, "r", encoding="utf-8") as f:
        text = f.read()

    thresholds: Dict[str, Any] = {}

    for line in text.splitlines():
        if "①-a" in line and "train" in line:
            m = re.search(r"(\d+)\s*\\?%", line)
            if m:
                thresholds["train_converged_min"] = float(m.group(1)) / 100.0
        elif "①-b" in line and "held-out" in line:
            m = re.search(r"(\d+)\s*\\?%", line)
            if m:
                thresholds["heldout_converged_min"] = float(m.group(1)) / 100.0
        elif "①-c" in line and "cos" in line:
            m = re.search(r"(\d+\.\d+)", line)
            if m:
                thresholds["heldout_mean_cos_max"] = float(m.group(1))
        elif "①-d" in line and "Delta" in line:
            m = re.search(r"=\s*(\d+)", line)
            if m:
                thresholds["determinism_max_delta"] = float(m.group(1))

    required_keys = ["train_converged_min", "heldout_converged_min", "heldout_mean_cos_max", "determinism_max_delta"]
    for k in required_keys:
        if k not in thresholds:
            raise ValueError(f"Unable to parse threshold for '{k}' from {prereg_path}")

    return thresholds


def parse_anchor_log(log_path: str) -> Dict[str, Any]:
    """Parse anchor verification log to check 20/20 EXACT match and max delta."""
    if not os.path.exists(log_path):
        if os.path.exists(FALLBACK_ANCHOR_LOG):
            log_path = FALLBACK_ANCHOR_LOG
        else:
            return {
                "log_found": False,
                "log_path": log_path,
                "total_items": 0,
                "exact_count": 0,
                "max_delta_overall": None,
                "all_match": False,
            }

    exact_count = 0
    total_items = 0
    max_delta_overall = 0.0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            # Match lines like: [01/20 Q#00] rho=... | max_delta=0.00e+00 | ... | EXACT
            if line_str.startswith("[") and "Q#" in line_str:
                total_items += 1
                if "EXACT" in line_str:
                    exact_count += 1
                # Parse max_delta
                m_delta = re.search(r"max_delta=([0-9.eE+-]+)", line_str)
                if m_delta:
                    d_val = float(m_delta.group(1))
                    if d_val > max_delta_overall:
                        max_delta_overall = d_val
            elif line_str.startswith("Q#") and "|" in line_str:
                total_items += 1
                if "EXACT" in line_str:
                    exact_count += 1
                parts = [p.strip() for p in line_str.split("|")]
                for p in parts:
                    if "e" in p or "E" in p:
                        try:
                            val = float(p)
                            if val > max_delta_overall and val < 1e5:
                                pass
                        except ValueError:
                            pass

    return {
        "log_found": True,
        "log_path": log_path,
        "total_items": total_items,
        "exact_count": exact_count,
        "max_delta_overall": max_delta_overall,
        "all_match": (exact_count == 20 and total_items == 20 and max_delta_overall == 0.0),
    }


def evaluate_gate(
    zstar_pt_path: str,
    prereg_path: str,
    items_json_path: str,
    out_json_path: str,
    anchor_log_path: str,
    manifest_path: str = DEFAULT_MANIFEST,
) -> Dict[str, Any]:
    """Execute gate evaluation and produce phase1_gate.json."""
    print("=" * 80)
    print("S5 Phase 1 Gate Evaluation (EXE-260923-28 rev2)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"z* Tensor: {zstar_pt_path}")
    print(f"PREREG: {prereg_path}")
    print(f"Items: {items_json_path}")
    print(f"Output: {out_json_path}")
    print("=" * 80, flush=True)

    # 1. Parse thresholds
    thresholds = parse_prereg_thresholds(prereg_path)
    print(f"Parsed PREREG thresholds: {thresholds}", flush=True)

    # 2. Load items
    with open(items_json_path, "r", encoding="utf-8") as f:
        items_data = json.load(f)
    train_indices: List[int] = [int(x) for x in items_data["train_indices"]]
    heldout_indices: List[int] = [int(x) for x in items_data["heldout_indices"]]

    # 3. Load zstar tensor
    if not os.path.exists(zstar_pt_path):
        raise FileNotFoundError(f"z* tensor not found: {zstar_pt_path}")
    zstar_dict = torch.load(zstar_pt_path, map_location="cpu")
    print(f"Loaded z* tensor with {len(zstar_dict)} keys.", flush=True)

    # Helper to find entry for a question index
    def get_entry(q_idx: int) -> Dict[str, Any]:
        candidates = [
            f"q{q_idx:02d}_s2030",
            f"q{q_idx:04d}_s2030",
            f"q{q_idx}_s2030",
        ]
        for c in candidates:
            if c in zstar_dict:
                return zstar_dict[c]
        raise KeyError(f"Question index {q_idx} not found in z* dict keys (tried: {candidates})")

    # 4. ①-a Train convergence (320 items)
    train_converged_count = 0
    for q in train_indices:
        entry = get_entry(q)
        if entry.get("converged", False):
            train_converged_count += 1
    train_converged_ratio = float(train_converged_count / len(train_indices))
    pass_1a = bool(train_converged_ratio >= thresholds["train_converged_min"])

    # 5. ①-b Held-out convergence (80 items)
    heldout_converged_count = 0
    heldout_z_list: List[torch.Tensor] = []
    heldout_lambda1_list: List[Any] = []
    for q in heldout_indices:
        entry = get_entry(q)
        if entry.get("converged", False):
            heldout_converged_count += 1
        z = entry.get("z_star", entry.get("z_1000"))
        heldout_z_list.append(z.flatten().float())
        l1 = entry.get("lambda1", "[?]")
        heldout_lambda1_list.append(l1)
    heldout_converged_ratio = float(heldout_converged_count / len(heldout_indices))
    pass_1b = bool(heldout_converged_ratio >= thresholds["heldout_converged_min"])

    # 6. ①-c Held-out pairwise cosine similarity
    Z = torch.stack(heldout_z_list, dim=0)  # [80, D]
    norms = torch.norm(Z, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    Z_norm = Z / norms
    cos_matrix = (Z_norm @ Z_norm.T).numpy()

    # Upper triangular off-diagonal pairs (3160 pairs)
    triu_idx = np.triu_indices(len(heldout_indices), k=1)
    pairwise_cos = cos_matrix[triu_idx]
    mean_cos = float(pairwise_cos.mean())
    max_cos = float(pairwise_cos.max())
    min_cos = float(pairwise_cos.min())
    std_cos = float(pairwise_cos.std())
    pass_1c = bool(mean_cos <= thresholds["heldout_mean_cos_max"])

    # 7. ①-d Determinism (Anchor check)
    anchor_info = parse_anchor_log(anchor_log_path)
    pass_1d = bool(anchor_info.get("all_match", False))

    # 8. ①-e lambda1 report
    lambda1_numeric = [x for x in heldout_lambda1_list if isinstance(x, (int, float))]
    lambda1_report = {
        "total_heldout": len(heldout_lambda1_list),
        "numeric_count": len(lambda1_numeric),
        "raw_field": "[?]" if len(lambda1_numeric) == 0 else "NUMERIC",
    }

    # Overall verdict
    overall_pass = bool(pass_1a and pass_1b and pass_1c and pass_1d)

    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "directive": "EXE-260923-28 rev2 (Ledger #331)",
            "zstar_pt": zstar_pt_path,
            "prereg": prereg_path,
            "items_json": items_json_path,
            "total_items": len(zstar_dict),
            "train_items": len(train_indices),
            "heldout_items": len(heldout_indices),
        },
        "prereg_thresholds": thresholds,
        "criteria": {
            "1a_train_convergence": {
                "metric": "train_converged_ratio",
                "measured": train_converged_ratio,
                "converged_count": train_converged_count,
                "total_count": len(train_indices),
                "threshold": thresholds["train_converged_min"],
                "operator": ">=",
                "verdict": "PASS" if pass_1a else "FAIL",
            },
            "1b_heldout_convergence": {
                "metric": "heldout_converged_ratio",
                "measured": heldout_converged_ratio,
                "converged_count": heldout_converged_count,
                "total_count": len(heldout_indices),
                "threshold": thresholds["heldout_converged_min"],
                "operator": ">=",
                "verdict": "PASS" if pass_1b else "FAIL",
            },
            "1c_heldout_discrimination": {
                "metric": "heldout_pairwise_mean_cos",
                "measured_mean": mean_cos,
                "measured_max": max_cos,
                "measured_min": min_cos,
                "measured_std": std_cos,
                "threshold": thresholds["heldout_mean_cos_max"],
                "operator": "<=",
                "verdict": "PASS" if pass_1c else "FAIL",
            },
            "1d_determinism": {
                "metric": "anchor_20_repro_max_delta",
                "measured_max_delta": anchor_info.get("max_delta_overall"),
                "exact_count": anchor_info.get("exact_count"),
                "total_anchor": anchor_info.get("total_items"),
                "threshold": thresholds["determinism_max_delta"],
                "operator": "==",
                "verdict": "PASS" if pass_1d else "FAIL",
            },
            "1e_lambda1_distribution": lambda1_report,
        },
        "gate_verdict": "PASS" if overall_pass else "FAIL",
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    append_to_manifest(manifest_path, out_json_path)

    print("-" * 80)
    print(f"①-a Train Convergence:    {train_converged_ratio:.4f} (>= {thresholds['train_converged_min']:.2f}) -> {'PASS' if pass_1a else 'FAIL'}")
    print(f"①-b Held-out Convergence: {heldout_converged_ratio:.4f} (>= {thresholds['heldout_converged_min']:.2f}) -> {'PASS' if pass_1b else 'FAIL'}")
    print(f"①-c Held-out Mean Cos:    {mean_cos:.4f} (<= {thresholds['heldout_mean_cos_max']:.2f}) -> {'PASS' if pass_1c else 'FAIL'}")
    print(f"①-d Determinism Anchor:   exact={anchor_info.get('exact_count')}/20 -> {'PASS' if pass_1d else 'FAIL'}")
    print(f"GATE OVERALL VERDICT:     {'PASS' if overall_pass else 'FAIL'}")
    print(f"Output saved to:          {out_json_path}")
    print("-" * 80, flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="S5 Phase 1 Gate Verification (EXE-260923-28 rev2).")
    parser.add_argument("--zstar", type=str, default=DEFAULT_ZSTAR_PATH, help="Path to zstar_s5_400.pt")
    parser.add_argument("--prereg", type=str, default=DEFAULT_PREREG_PATH, help="Path to PREREG_S5_v4.md")
    parser.add_argument("--items", type=str, default=DEFAULT_ITEMS_PATH, help="Path to s5_items_400.json")
    parser.add_argument("--out", "--output", type=str, default=DEFAULT_OUT_PATH, help="Path to output phase1_gate.json")
    parser.add_argument("--anchor_log", type=str, default=DEFAULT_ANCHOR_LOG, help="Path to anchor log")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help="Path to MANIFEST_s5.sha256")
    args = parser.parse_args()

    results = evaluate_gate(
        zstar_pt_path=args.zstar,
        prereg_path=args.prereg,
        items_json_path=args.items,
        out_json_path=args.out,
        anchor_log_path=args.anchor_log,
        manifest_path=args.manifest,
    )

    if results["gate_verdict"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()

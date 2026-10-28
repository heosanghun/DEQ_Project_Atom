#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diag_d2_geometry.py - Latent State Geometry Analysis (D2)
Directive: TASK-0070 §D2 (Ledger #430)

Computes geometry metrics on:
1. Held-out 600:
   - z*: 80 from results_s7/zstar_s7_a0.25_400.pt + 520 from results_s7/zstar_s7_a0.25_heldout520_lowest.pt (field: z_star)
   - z_src: results_s7/zsrc_heldout600.pt (duplicate priority: qNNNN_a0.25 over qNNNN)
2. Train 1,000:
   - z*: results_s7/zstar_train1000_P4.pt
   - z_src: results_s7/zsrc_train1000_P4.pt

Representations:
  - 64-slot mean: (2,560d)
  - Flattened: (163,840d)

Metrics:
  - cos(z*_i, z_src_i) distribution (mean, 5/50/95%)
  - Cross-item mean cos (z* pairs, z_src pairs)
  - ||z*|| / ||z_src|| norm ratio distribution (mean, 5/50/95%)
  - Cross-retrieval rate (1-NN mutual recovery: z* -> z_src, and z_src -> z*)
  - 1-NN same-type classification rate (using 'type' in math_train_prompts_5000.jsonl)
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Tuple
import torch
import numpy as np

# Strict adherence to CPU memory & safety protocol (Ledger #381)
torch.set_grad_enabled(False)


def log_memory(tag: str = ""):
    """Logs current RSS memory."""
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = rusage.ru_maxrss / 1024.0  # Linux is in KiB
        print(f"[MEM-LOG {tag}] RSS: {rss_mb:.2f} MB", flush=True)
    except Exception:
        pass


def extract_tensor_entry(entry: Any, prefer_z_star: bool = True) -> torch.Tensor:
    """Extract raw tensor from dictionary entry."""
    if isinstance(entry, torch.Tensor):
        return entry
    if isinstance(entry, dict):
        if prefer_z_star and "z_star" in entry and isinstance(entry["z_star"], torch.Tensor):
            return entry["z_star"]
        if "z_lowest" in entry and isinstance(entry["z_lowest"], torch.Tensor):
            return entry["z_lowest"]
        if "z_star" in entry and isinstance(entry["z_star"], torch.Tensor):
            return entry["z_star"]
        if "z_src" in entry and isinstance(entry["z_src"], torch.Tensor):
            return entry["z_src"]
    raise ValueError(f"Unable to extract tensor from entry of type {type(entry)}")


def lookup_qid_tensor(
    tensor_dict: Dict[str, Any],
    qid: int,
    prefer_a025: bool = True,
    prefer_z_star: bool = True
) -> torch.Tensor:
    """Lookup tensor by question ID with prefix priorities."""
    if prefer_a025:
        candidates = [
            f"q{qid:04d}_a0.25",
            f"q{qid:04d}",
            f"q{qid}",
            str(qid),
            qid,
        ]
    else:
        candidates = [
            f"q{qid:04d}",
            f"q{qid:04d}_a0.25",
            f"q{qid}",
            str(qid),
            qid,
        ]

    for c in candidates:
        if c in tensor_dict:
            return extract_tensor_entry(tensor_dict[c], prefer_z_star=prefer_z_star)
    raise KeyError(f"Question ID {qid} not found in tensor dict (tried: {candidates})")


def compute_geometry_metrics(z_star: torch.Tensor, z_src: torch.Tensor, types: List[str]) -> Dict[str, Any]:
    """
    Computes all D2 geometry metrics.
    z_star: Tensor [N, D] in float32 or float64
    z_src: Tensor [N, D] in float32 or float64
    types: List of length N
    """
    N, D = z_star.shape
    assert z_src.shape == (N, D), f"Shape mismatch: {z_star.shape} vs {z_src.shape}"
    assert len(types) == N, f"Type length mismatch: {len(types)} vs {N}"

    # Vector norms
    z_star_norm = torch.norm(z_star.double(), p=2, dim=1, keepdim=True)
    z_src_norm = torch.norm(z_src.double(), p=2, dim=1, keepdim=True)

    norm_ratio = (z_star_norm / torch.clamp(z_src_norm, min=1e-12)).squeeze(1).cpu().numpy()

    # Unit vectors
    z_star_u = (z_star.double() / torch.clamp(z_star_norm, min=1e-12))
    z_src_u = (z_src.double() / torch.clamp(z_src_norm, min=1e-12))

    # 1. Item-wise cos(z*_i, z_src_i)
    cos_i = torch.sum(z_star_u * z_src_u, dim=1).cpu().numpy()
    cos_i_dist = {
        "mean": float(np.mean(cos_i)),
        "p5": float(np.percentile(cos_i, 5)),
        "p50": float(np.percentile(cos_i, 50)),
        "p95": float(np.percentile(cos_i, 95)),
    }

    # 2. Norm ratio distribution
    norm_ratio_dist = {
        "mean": float(np.mean(norm_ratio)),
        "p5": float(np.percentile(norm_ratio, 5)),
        "p50": float(np.percentile(norm_ratio, 50)),
        "p95": float(np.percentile(norm_ratio, 95)),
    }

    # 3. Cross-item cosine: cos(z*_i, z*_j) and cos(z_src_i, z_src_j) for i != j
    S_star = torch.mm(z_star_u, z_star_u.t())
    mask_offdiag = ~torch.eye(N, dtype=torch.bool, device=S_star.device)
    cross_cos_star = S_star[mask_offdiag].cpu().numpy()

    S_src = torch.mm(z_src_u, z_src_u.t())
    cross_cos_src = S_src[mask_offdiag].cpu().numpy()

    cross_cos = {
        "zstar_cross_mean": float(np.mean(cross_cos_star)),
        "zsrc_cross_mean": float(np.mean(cross_cos_src)),
    }

    # 4. Cross-retrieval rate (1-NN mutual recovery)
    S_cross = torch.mm(z_star_u, z_src_u.t()) # S_cross[i, j] = cos(z*_i, z_src_j)

    # Forward: for item i, find argmax_j cos(z*_i, z_src_j)
    pred_src = torch.argmax(S_cross, dim=1).cpu().numpy()
    fwd_retrieval = float(np.mean(pred_src == np.arange(N)))

    # Backward: for item j, find argmax_i cos(z*_i, z_src_j)
    pred_star = torch.argmax(S_cross, dim=0).cpu().numpy()
    bwd_retrieval = float(np.mean(pred_star == np.arange(N)))

    retrieval = {
        "zstar_to_zsrc": fwd_retrieval,
        "zsrc_to_zstar": bwd_retrieval,
    }

    # 5. 1-NN same-type classification rate
    types_arr = np.array(types)

    S_star_no_diag = S_star.clone()
    S_star_no_diag.fill_diagonal_(-float("inf"))
    nn_star_idx = torch.argmax(S_star_no_diag, dim=1).cpu().numpy()
    nn_star_match = float(np.mean(types_arr[nn_star_idx] == types_arr))

    S_src_no_diag = S_src.clone()
    S_src_no_diag.fill_diagonal_(-float("inf"))
    nn_src_idx = torch.argmax(S_src_no_diag, dim=1).cpu().numpy()
    nn_src_match = float(np.mean(types_arr[nn_src_idx] == types_arr))

    type_1nn = {
        "zstar_1nn_same_type": nn_star_match,
        "zsrc_1nn_same_type": nn_src_match,
    }

    return {
        "cos_i_dist": cos_i_dist,
        "norm_ratio_dist": norm_ratio_dist,
        "cross_cos": cross_cos,
        "retrieval": retrieval,
        "type_1nn": type_1nn,
    }


def load_dataset_types(prompts_path: str) -> Dict[int, str]:
    """Load 'type' metadata from math_train_prompts_5000.jsonl without reading prompt/solution texts."""
    types_map = {}
    with open(prompts_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                types_map[idx] = data.get("type", data.get("subject", "Unknown"))
    return types_map


def main():
    parser = argparse.ArgumentParser(description="TASK-0070 D2: Latent State Geometry Analysis")
    parser.add_argument("--base_dir", type=str, default="/home/shoon/cts_e1_s5")
    parser.add_argument("--out", type=str, default="results_s7/diag/d2_geometry.json")
    args = parser.parse_args()

    base_dir = args.base_dir
    os.makedirs(os.path.join(base_dir, os.path.dirname(args.out)), exist_ok=True)
    out_file = os.path.join(base_dir, args.out)

    print("=" * 80)
    print("TASK-0070 D2: Latent State Geometry Analysis (CPU)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Directory: {base_dir}")
    print("=" * 80, flush=True)

    log_memory("INIT")

    # Load question types
    prompts_path = os.path.join(base_dir, "data", "stage2", "math_train_prompts_5000.jsonl")
    if not os.path.exists(prompts_path):
        prompts_path = "data/stage2/math_train_prompts_5000.jsonl"
    print(f"Loading metadata types from {prompts_path}...")
    types_map = load_dataset_types(prompts_path)
    print(f"Loaded {len(types_map)} types.")

    results_all = {}

    # =========================================================================
    # PART 1: Held-out 600 Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: Held-out 600 Analysis")
    print("=" * 60, flush=True)

    p80 = os.path.join(base_dir, "results_s7", "zstar_s7_a0.25_400.pt")
    p520 = os.path.join(base_dir, "results_s7", "zstar_s7_a0.25_heldout520_lowest.pt")
    psrc600 = os.path.join(base_dir, "results_s7", "zsrc_heldout600.pt")
    porder = os.path.join(base_dir, "results_s7", "p6_item_order.json")

    print(f"Loading 80 z* from {p80}...")
    d80 = torch.load(p80, map_location="cpu")
    print(f"Loading 520 z* from {p520}...")
    d520 = torch.load(p520, map_location="cpu")
    zstar_dict_600 = {**d80, **d520}

    print(f"Loading 600 z_src from {psrc600}...")
    zsrc_dict_600 = torch.load(psrc600, map_location="cpu")

    print(f"Loading canonical item order from {porder}...")
    with open(porder, "r", encoding="utf-8") as f:
        order_obj = json.load(f)
    qids_600 = order_obj["indices"] if isinstance(order_obj, dict) else order_obj
    assert len(qids_600) == 600, f"Expected 600 items, got {len(qids_600)}"

    # Stack tensors
    zstar_tensors_600 = []
    zsrc_tensors_600 = []
    types_600 = []

    for qid in qids_600:
        # z* with field z_star
        z_s = lookup_qid_tensor(zstar_dict_600, qid, prefer_a025=True, prefer_z_star=True)
        # z_src with duplicate priority qNNNN_a0.25
        z_r = lookup_qid_tensor(zsrc_dict_600, qid, prefer_a025=True, prefer_z_star=False)

        if z_s.dim() == 3:
            z_s = z_s.squeeze(0)  # [64, 2560]
        if z_r.dim() == 3:
            z_r = z_r.squeeze(0)  # [64, 2560]

        assert z_s.shape == (64, 2560), f"Unexpected z_star shape {z_s.shape} for Q#{qid}"
        assert z_r.shape == (64, 2560), f"Unexpected z_src shape {z_r.shape} for Q#{qid}"

        zstar_tensors_600.append(z_s.float())
        zsrc_tensors_600.append(z_r.float())
        types_600.append(types_map[qid])

    zstar_stack_600 = torch.stack(zstar_tensors_600)  # [600, 64, 2560]
    zsrc_stack_600 = torch.stack(zsrc_tensors_600)    # [600, 64, 2560]

    # Free raw dicts
    del d80, d520, zstar_dict_600, zsrc_dict_600
    log_memory("HELD-OUT 600 STACKED")

    # Representation 1: 64-slot mean (2,560d)
    print("\nComputing Held-out 600: 64-slot mean (2,560d)...")
    zstar_mean_600 = torch.mean(zstar_stack_600, dim=1)  # [600, 2560]
    zsrc_mean_600 = torch.mean(zsrc_stack_600, dim=1)    # [600, 2560]
    m_heldout_mean = compute_geometry_metrics(zstar_mean_600, zsrc_mean_600, types_600)

    # Representation 2: Flattened (163,840d)
    print("Computing Held-out 600: Flattened (163,840d)...")
    zstar_flat_600 = zstar_stack_600.reshape(600, -1)   # [600, 163840]
    zsrc_flat_600 = zsrc_stack_600.reshape(600, -1)     # [600, 163840]
    m_heldout_flat = compute_geometry_metrics(zstar_flat_600, zsrc_flat_600, types_600)

    results_all["heldout_600"] = {
        "slot_mean_2560d": m_heldout_mean,
        "flattened_163840d": m_heldout_flat,
    }

    del zstar_stack_600, zsrc_stack_600, zstar_mean_600, zsrc_mean_600, zstar_flat_600, zsrc_flat_600
    log_memory("HELD-OUT 600 FINISHED")

    # =========================================================================
    # PART 2: Train 1,000 Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: Train 1,000 Analysis")
    print("=" * 60, flush=True)

    pstar_train = os.path.join(base_dir, "results_s7", "zstar_train1000_P4.pt")
    psrc_train = os.path.join(base_dir, "results_s7", "zsrc_train1000_P4.pt")
    ptrain_cfg = os.path.join(base_dir, "configs", "s7_train_1000.json")

    print(f"Loading Train 1000 z* from {pstar_train}...")
    zstar_dict_train = torch.load(pstar_train, map_location="cpu")
    print(f"Loading Train 1000 z_src from {psrc_train}...")
    zsrc_dict_train = torch.load(psrc_train, map_location="cpu")

    print(f"Loading Train 1000 config from {ptrain_cfg}...")
    with open(ptrain_cfg, "r", encoding="utf-8") as f:
        train_cfg = json.load(f)
    qids_train = train_cfg["indices"] if isinstance(train_cfg, dict) else train_cfg
    valid_train_qids = []
    missing_qids = []
    for qid in qids_train:
        try:
            lookup_qid_tensor(zstar_dict_train, qid, prefer_a025=True, prefer_z_star=True)
            lookup_qid_tensor(zsrc_dict_train, qid, prefer_a025=True, prefer_z_star=False)
            valid_train_qids.append(qid)
        except KeyError:
            missing_qids.append(qid)

    print(f"Train items present in both z* and z_src: {len(valid_train_qids)} / {len(qids_train)} (Missing: {len(missing_qids)})")
    if missing_qids:
        print(f"Sample missing QIDs: {missing_qids[:10]}")

    zstar_tensors_train = []
    zsrc_tensors_train = []
    types_train = []

    for qid in valid_train_qids:
        z_s = lookup_qid_tensor(zstar_dict_train, qid, prefer_a025=True, prefer_z_star=True)
        z_r = lookup_qid_tensor(zsrc_dict_train, qid, prefer_a025=True, prefer_z_star=False)

        if z_s.dim() == 3:
            z_s = z_s.squeeze(0)
        if z_r.dim() == 3:
            z_r = z_r.squeeze(0)

        assert z_s.shape == (64, 2560), f"Unexpected z_star shape {z_s.shape} for Q#{qid}"
        assert z_r.shape == (64, 2560), f"Unexpected z_src shape {z_r.shape} for Q#{qid}"

        zstar_tensors_train.append(z_s.float())
        zsrc_tensors_train.append(z_r.float())
        types_train.append(types_map[qid])

    zstar_stack_train = torch.stack(zstar_tensors_train)  # [1000, 64, 2560]
    zsrc_stack_train = torch.stack(zsrc_tensors_train)    # [1000, 64, 2560]

    del zstar_dict_train, zsrc_dict_train
    log_memory("TRAIN 1000 STACKED")

    # Representation 1: 64-slot mean (2,560d)
    print("\nComputing Train 1,000: 64-slot mean (2,560d)...")
    zstar_mean_train = torch.mean(zstar_stack_train, dim=1)
    zsrc_mean_train = torch.mean(zsrc_stack_train, dim=1)
    m_train_mean = compute_geometry_metrics(zstar_mean_train, zsrc_mean_train, types_train)

    # Representation 2: Flattened (163,840d)
    n_train_actual = len(valid_train_qids)
    print(f"Computing Train ({n_train_actual} items): Flattened (163,840d)...")
    zstar_flat_train = zstar_stack_train.reshape(n_train_actual, -1)
    zsrc_flat_train = zsrc_stack_train.reshape(n_train_actual, -1)
    m_train_flat = compute_geometry_metrics(zstar_flat_train, zsrc_flat_train, types_train)

    results_all["train_1000"] = {
        "slot_mean_2560d": m_train_mean,
        "flattened_163840d": m_train_flat,
    }

    del zstar_stack_train, zsrc_stack_train, zstar_mean_train, zsrc_mean_train, zstar_flat_train, zsrc_flat_train
    log_memory("TRAIN 1000 FINISHED")

    # Write output JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)

    print(f"\n[D2 COMPLETE] Saved geometry metrics to {out_file}")
    print(json.dumps(results_all, indent=2))


if __name__ == "__main__":
    main()

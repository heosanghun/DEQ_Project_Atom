#!/usr/bin/env python3
"""scripts/verify_local_equivalence.py - Equivalence verification between Server A100 and Local RTX 4090
Directive: Ledger #394 Task 1
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

BASE_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, BASE_DIR)

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.model.gemma_loader import load_gemma4_e4b
from cts.train.lora_compat import apply_paper_lora
from cts.deq.zstar_canonical import compute_zstar

DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "zstar_canonical.yaml")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "stage2", "math_train_prompts_5000.jsonl")
DEFAULT_SNAPSHOT_PATH = os.path.join(BASE_DIR, "snapshots", "58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377")
DEFAULT_CKPT_PATH = os.path.join(BASE_DIR, "artifacts", "stage1_last.lora_only.pt")
OUT_JSON_PATH = os.path.join(BASE_DIR, "raw_s7", "equivalence_test_TASK0001.json")
OUT_LOG_PATH = os.path.join(BASE_DIR, "raw_s7", "equivalence_test_TASK0001.log")


SERVER_GROUND_TRUTH = {
    583: {
        "rho_1000": 4.9559e-03,
        "min_rho": 2.4905e-03,
        "shape": "oscillation",
        "norm": 5038.54,
        "source": "shared/logs/server_s0_TASK0001.log:47",
    },
    587: {
        "rho_1000": 4.3494e-07,
        "min_rho": 4.3494e-07,
        "shape": "monotonic_decrease",
        "norm": 5028.50,
        "source": "shared/logs/server_s0_TASK0001.log:49",
    },
    597: {
        "rho_1000": 4.7282e-03,
        "min_rho": 3.7708e-04,
        "shape": "plateau",
        "norm": 4538.46,
        "source": "shared/logs/server_s0_TASK0001.log:51",
    },
    610: {
        "rho_1000": 9.4428e-05,
        "min_rho": 9.4428e-05,
        "shape": "monotonic_decrease",
        "norm": None,
        "source": "shared/logs/driver_2026-09-28.log:1",
    },
    780: {
        "rho_1000": 7.7222e-05,
        "min_rho": 7.7222e-05,
        "shape": "monotonic_decrease",
        "norm": 4900.55,
        "source": "shared/logs/server_s0_TASK0001_fix.log:4",
    },
    1022: {
        "rho_1000": 1.6604e-04,
        "min_rho": 1.6604e-04,
        "shape": "monotonic_decrease",
        "norm": None,
        "source": "shared/logs/driver_2026-09-28.log:9",
    },
    1099: {
        "rho_1000": 5.6496e-03,
        "min_rho": 7.9858e-04,
        "shape": "plateau",
        "norm": 4249.6,
        "source": "shared/logs/driver_2026-09-28.log:10",
    },
    1169: {
        "rho_1000": 5.7583e-06,
        "min_rho": 5.7583e-06,
        "shape": "monotonic_decrease",
        "norm": None,
        "source": "shared/logs/driver_2026-09-28.log:11",
    },
    1352: {
        "rho_1000": 8.9987e-04,
        "min_rho": 8.0051e-04,
        "shape": "oscillation",
        "norm": 4647.42,
        "source": "shared/logs/server_status_TASK0002_fix.log:36",
    },
    1355: {
        "rho_1000": 6.0670e-04,
        "min_rho": 6.0670e-04,
        "shape": "monotonic_decrease",
        "norm": 3878.48,
        "source": "shared/logs/server_status_TASK0002_fix.log:37",
    },
    1364: {
        "rho_1000": 2.8664e-05,
        "min_rho": 2.8664e-05,
        "shape": "monotonic_decrease",
        "norm": 4997.08,
        "source": "shared/logs/server_status_TASK0002_fix.log:38",
    },
    1365: {
        "rho_1000": 2.7639e-03,
        "min_rho": 1.2788e-03,
        "shape": "oscillation",
        "norm": 4856.92,
        "source": "shared/logs/server_status_TASK0002_fix.log:39",
    },
    1367: {
        "rho_1000": 3.2070e-04,
        "min_rho": 2.7595e-04,
        "shape": "monotonic_decrease",
        "norm": 4934.91,
        "source": "shared/logs/server_status_TASK0002_fix.log:40",
    },
    1403: {
        "rho_1000": 5.4682e-06,
        "min_rho": 5.4682e-06,
        "shape": "monotonic_decrease",
        "norm": None,
        "source": "shared/logs/driver_2026-09-28.log:13",
    },
    1470: {
        "rho_1000": 1.4103e-03,
        "min_rho": 1.3104e-03,
        "shape": "oscillation",
        "norm": 5000.0,
        "source": "shared/logs/driver_2026-09-28.log:14",
    },
    1614: {
        "rho_1000": 2.0368e-03,
        "min_rho": 8.0715e-04,
        "shape": "monotonic_decrease",
        "norm": None,
        "source": "shared/logs/driver_2026-09-28.log:16",
    },
    1661: {
        "rho_1000": 4.9919e-03,
        "min_rho": 5.5851e-04,
        "shape": "monotonic_decrease",
        "norm": None,
        "source": "shared/logs/driver_2026-09-28.log:18",
    },
    1756: {
        "rho_1000": 3.8179e-07,
        "min_rho": 3.8179e-07,
        "shape": "monotonic_decrease",
        "norm": None,
        "source": "shared/logs/driver_2026-09-28.log:20",
    },
}


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


def get_interval_name(r: float) -> str:
    """Classify true residual r into PREREG_S7_D_v3_5 4 intervals."""
    if r < 1.0e-03:
        return "초고정밀 수렴"
    elif 1.0e-03 <= r < 5.0e-03:
        return "엄격 수렴"
    elif 5.0e-03 <= r < 1.0e-02:
        return "S5 동치 완보 수렴"
    else:
        return "미수렴"


def load_local_model(device_str: str = "cuda:0") -> tuple[GemmaCTSBackbone, torch.Tensor]:
    print("Loading base Gemma-4-E4B on CPU in torch.bfloat16...", flush=True)
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
    bb.to(device=device_str)
    bb.eval()
    w_g = bb.routing_matrix().to(device=device_str, dtype=torch.float32)
    return bb, w_g


def main():
    device_str = "cuda:0"
    device = torch.device(device_str)
    alpha = 0.25
    max_nfe = 1000
    tol = 0.0
    sigma_noise = 0.020
    seed = 2030
    g0 = 0.25

    print(f"Loading dataset from {DEFAULT_DATASET_PATH}...", flush=True)
    with open(DEFAULT_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    bb, w_g = load_local_model(device_str)

    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    log_f = open(OUT_LOG_PATH, "w", encoding="utf-8")
    log_f.write(f"=== EQUIVALENCE TEST (RTX 4090 vs A100 SERVER) ===\n")
    log_f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"Device: {device_str} | Alpha: {alpha} | Max NFE: {max_nfe} | Tol: {tol} | Sigma: {sigma_noise} | Seed: {seed} | g0: {g0}\n")
    log_f.write("-" * 80 + "\n")
    log_f.flush()

    results: List[Dict[str, Any]] = []

    print(f"Starting verification of {len(SERVER_GROUND_TRUTH)} items...", flush=True)
    for q_idx, gt in SERVER_GROUND_TRUTH.items():
        p_text = dataset[q_idx]["prompt"]
        t0 = time.time()
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
        elapsed = time.time() - t0

        rho_traj = log["rho_traj"]
        rho_1000_local = float(rho_traj[999]) if len(rho_traj) >= 1000 else float(rho_traj[-1])
        min_rho_local = float(min(rho_traj))
        shape_local = classify_trajectory_4way(rho_traj)
        norm_z_local = float(torch.norm(z_star).item())

        rho_1000_server = gt["rho_1000"]
        r_local = rho_1000_local / alpha  # 4 * rho_1000
        r_server = rho_1000_server / alpha  # 4 * rho_1000

        interval_local = get_interval_name(r_local)
        interval_server = get_interval_name(r_server)
        interval_match = (interval_local == interval_server)

        rel_diff_rho = abs(rho_1000_local - rho_1000_server) / max(rho_1000_server, 1e-12)

        entry = {
            "q_idx": q_idx,
            "rho_1000_local": rho_1000_local,
            "rho_1000_server": rho_1000_server,
            "r_local": r_local,
            "r_server": r_server,
            "interval_local": interval_local,
            "interval_server": interval_server,
            "interval_match": interval_match,
            "shape_local": shape_local,
            "shape_server": gt["shape"],
            "norm_local": norm_z_local,
            "norm_server": gt["norm"],
            "rel_diff_rho": rel_diff_rho,
            "min_rho_local": min_rho_local,
            "min_rho_server": gt["min_rho"],
            "elapsed_sec": elapsed,
            "server_source": gt["source"],
        }
        results.append(entry)

        line = (
            f"Q#{q_idx:04d}: rho_local={rho_1000_local:.6e}, rho_server={rho_1000_server:.6e}, "
            f"rel_diff={rel_diff_rho:.4e}, r_loc={r_local:.6e} ({interval_local}), "
            f"r_srv={r_server:.6e} ({interval_server}), match={interval_match}, shape_loc={shape_local}, shape_srv={gt['shape']}"
        )
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log_f.close()
    print(f"\nEquivalence test finished! Results saved to {OUT_JSON_PATH}", flush=True)


if __name__ == "__main__":
    main()

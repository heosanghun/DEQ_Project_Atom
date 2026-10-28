"""
cts/deq/zstar_inject_v2.py - G2 개선판(M2′) 고정점 계산기 (CC, 2026-10-08)
정본 cts/deq/zstar_canonical_lowest.py의 compute_zstar_lowest를 복사하고 두 가지만 더했다:
  inject_mode="zsrc"이면 주입 벡터 c_hat_0 대신 z_src(64칸 전체)를 쓴다.
  save_steps에 적은 걸음의 상태 z_k를 log["z_k"]로 돌려준다.
inject_mode="mean", save_steps=()이면 정본과 같은 계산이다(앵커 max|Δ|=0으로 확인).
"""
import os
import sys
import yaml
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
import torch

SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"
BASE_S5_DIR = "/home/shoon/cts_e1_s5"
for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cts.deq.transition_inject import (
    compute_broadcast_context_c,
    compute_c_hat_0,
    InjectedTransitionStep,
)
from cts.deq.zstar_canonical import load_canonical_config

DEFAULT_CONFIG_PATH = os.path.join(SYS_DIR, "configs", "zstar_canonical.yaml")
if not os.path.exists(DEFAULT_CONFIG_PATH):
    DEFAULT_CONFIG_PATH = os.path.join(BASE_S5_DIR, "configs", "zstar_canonical.yaml")


@torch.no_grad()
def compute_zstar_lowest_v2(
    x: Any,
    cfg: Optional[Union[str, Path, Dict[str, Any]]] = None,
    backbone: Optional[Any] = None,
    w_g: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
    inject_mode: str = "mean",
    save_steps: Tuple[int, ...] = (),
    **kwargs: Any,
) -> Tuple[torch.Tensor, float, int, Dict[str, Any]]:
    """Same as compute_zstar_lowest, plus (a) inject_mode: "mean" = canonical c_hat_0 (prompt-mean
    embedding broadcast to K slots), "zsrc" = per-slot z_src as the injection vector; (b) save_steps:
    iteration indices k whose state z_k (k=0 is z0) is returned in log["z_k"]. With inject_mode="mean"
    and save_steps=() the computation is identical to compute_zstar_lowest."""
    config = load_canonical_config(cfg if cfg is not None else DEFAULT_CONFIG_PATH)
    config.update(kwargs)

    g0 = float(config.get("g0", 0.25))
    alpha = float(config.get("alpha", 0.25))
    seed = int(config.get("seed", 2030))
    tol = float(config.get("tol", 0.0))
    max_nfe = int(config.get("max_nfe", 1000))
    solver = str(config.get("solver", "picard")).lower()

    k_latent = int(config.get("k_latent", 64))
    sigma_noise = float(config.get("sigma_noise", 0.020))
    top_k = int(config.get("top_k", 3))
    routing_mode = str(config.get("routing_mode", "sparse"))

    if solver != "picard":
        raise ValueError(f"compute_zstar_lowest strictly supports picard, got: {solver}")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if backbone is None:
        from harness_A_v1 import get_model
        backbone = get_model(str(device))

    orig_mode = backbone.training
    try:
        backbone.eval()
        d = backbone.hidden_size

        if w_g is None:
            w_g = backbone.routing_matrix().to(device=device, dtype=torch.float32)
        else:
            w_g = w_g.to(device=device, dtype=torch.float32)

        if isinstance(x, str):
            ctx = backbone.encode_context(x)
            if hasattr(ctx, "to"):
                ctx = ctx.to(device=device)
        else:
            ctx = x
            if hasattr(ctx, "to"):
                ctx = ctx.to(device=device)

        # 1. Compute z_src via root_fwdpass
        z_src = backbone.root_fwdpass(ctx, K=k_latent, output_dtype=torch.float32).to(device=device)

        # 2. Compute context broadcast c_x
        c_x = compute_broadcast_context_c(ctx, K=k_latent, device=device, dtype=torch.float32)

        # 3. Initialize z0 with canonical seed
        torch.manual_seed(seed)
        z0 = (
            z_src
            + torch.randn(k_latent, d, device=device, dtype=torch.float32) * sigma_noise
        ).clone()

        # 4. Compute constant injection vector c_hat_0
        c_hat_0 = compute_c_hat_0(c_x, z0).to(device=device, dtype=torch.float32)
        if inject_mode == "zsrc":
            c_hat_0 = z_src.clone()
        elif inject_mode != "mean":
            raise ValueError(f"unknown inject_mode: {inject_mode}")

        # 5. Build step function T(z)
        step_fn = InjectedTransitionStep(
            backbone=backbone,
            context=ctx,
            w_g=w_g,
            c_hat_0=c_hat_0,
            g0=g0,
            alpha=alpha,
            top_k=top_k,
            routing_mode=routing_mode,
        )

        t0_start = time.time()
        res_traj: List[float] = []
        rho_traj: List[float] = []
        checkpoints_100: Dict[int, Dict[str, float]] = {}

        # 6. Execute Picard iteration and track lowest residual state
        z_cur = z0.clone()
        final_nfe = max_nfe

        # Tracking variables
        k_star = 0
        r_lowest = float("inf")
        z_lowest = z0.clone()
        top3_lowest: List[int] = []
        save_set = set(int(s) for s in save_steps)
        z_k_saved: Dict[int, torch.Tensor] = {}
        if 0 in save_set:
            z_k_saved[0] = z0.detach().cpu().clone()

        for k in range(1, max_nfe + 1):
            # Evaluate transition on z_cur (which is z_{k-1})
            z_next = step_fn(z_cur)

            # Trajectory metrics matching canonical zstar_canonical.py
            diff = float(torch.norm(z_next - z_cur).item())
            norm_z = float(torch.norm(z_cur).item())
            rho_fp = diff / max(norm_z, 1e-12)

            res_traj.append(diff)
            rho_traj.append(rho_fp)

            if k % 100 == 0:
                checkpoints_100[k] = {"norm_z": float(torch.norm(z_next).item()), "diff": diff, "rho_fp": rho_fp}

            # Float64 residual calculation for z_{k-1}: r(z_{k-1}) = ||T(z_{k-1}) - z_{k-1}|| / (alpha * ||z_{k-1}||)
            diff_64 = z_next.to(torch.float64) - z_cur.to(torch.float64)
            diff_norm_64 = float(torch.norm(diff_64, p=2).item())
            norm_cur_64 = float(torch.norm(z_cur.to(torch.float64), p=2).item())
            r_k_prev = diff_norm_64 / (alpha * max(norm_cur_64, 1e-12))
            top3_k_prev = [int(x) for x in step_fn.last_r_ids]

            # Update lowest residual state if strictly smaller (tie-breaking: min k)
            if r_k_prev < r_lowest:
                r_lowest = r_k_prev
                k_star = k - 1
                z_lowest = z_cur.clone()
                top3_lowest = list(top3_k_prev)

            z_cur = z_next
            if k in save_set:
                z_k_saved[k] = z_cur.detach().cpu().clone()

            if tol is not None and tol > 0 and rho_fp < tol:
                final_nfe = k
                break

        # Final fixed-point state at termination
        z_star = z_cur
        final_diff = res_traj[-1]
        final_norm_z = float(torch.norm(z_star).item())
        final_rho_fp = rho_traj[-1]

        # Evaluate T(z_star) to compute r_1000 and top3_final
        z_next_final = step_fn(z_star)
        diff_final_64 = z_next_final.to(torch.float64) - z_star.to(torch.float64)
        diff_final_norm_64 = float(torch.norm(diff_final_64, p=2).item())
        norm_star_64 = float(torch.norm(z_star.to(torch.float64), p=2).item())
        r_1000 = diff_final_norm_64 / (alpha * max(norm_star_64, 1e-12))
        top3_final = [int(x) for x in step_fn.last_r_ids]

        # Compare r_1000 against r_lowest (if strictly less, k_star = max_nfe)
        if r_1000 < r_lowest:
            r_lowest = r_1000
            k_star = final_nfe
            z_lowest = z_star.clone()
            top3_lowest = list(top3_final)

        t_elapsed = time.time() - t0_start

        log: Dict[str, Any] = {
            "solver": "picard",
            "nfe": final_nfe,
            "norm_z": final_norm_z,
            "final_diff": final_diff,
            "final_rho_fp": final_rho_fp,
            "converged": bool(final_rho_fp < tol) if tol is not None and tol > 0 else False,
            "time_seconds": t_elapsed,
            "steps_per_sec": float(final_nfe / max(t_elapsed, 1e-6)),
            "z0": z0,
            "z_src": z_src,
            "c_hat_0": c_hat_0,
            "c_x": c_x,
            "res_traj": res_traj,
            "rho_traj": rho_traj,
            "checkpoints_100": checkpoints_100,
            "config": config,
            # Lowest-residual fields
            "z_lowest": z_lowest,
            "k_star": k_star,
            "r_lowest": r_lowest,
            "r_1000": r_1000,
            "top3_final": top3_final,
            "top3_lowest": top3_lowest,
            "inject_mode": inject_mode,
            "z_k": z_k_saved,
        }

        return z_star, final_rho_fp, final_nfe, log
    finally:
        if orig_mode:
            backbone.train()

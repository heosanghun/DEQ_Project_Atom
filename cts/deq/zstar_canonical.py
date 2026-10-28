"""
cts/deq/zstar_canonical.py - Canonical Fixed-Point Generator (Single Entry Point)
Directive: EXE-260913-07 Task 1 (Phase 4-1)
Ledger: #229

Purpose:
Unify z* generation across training, harness evaluation, and inference.
Resolves historical discrepancy (#213) where training used broyden30/hardcoded seed
while inference/harness used picard300/picard1000.

Formulation:
  z0 = z_src(x) + N(seed)     (imported from canonical path diag_task13.py:150-156)
  c_hat_0 = (c_x / ||c_x[0]||) * mean_i(||z_0[i]||)
  T(z) = (1 - alpha) * z + alpha * (Phi(z) + g0 * c_hat_0)
  stopping criterion: rho_fp < tol or step == max_nfe
  All hyperparameters loaded from configs/zstar_canonical.yaml (zero hardcoding).
"""

import os
import sys
import yaml
import time
import torch
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"
for p in [SYS_DIR, SCRATCH_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cts.deq.transition_inject import (
    compute_broadcast_context_c,
    compute_c_hat_0,
    InjectedTransitionStep,
)

DEFAULT_CONFIG_PATH = os.path.join(SYS_DIR, "configs", "zstar_canonical.yaml")


def load_canonical_config(cfg_path: Optional[Union[str, Path, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Load canonical configuration from YAML file or return dictionary."""
    if isinstance(cfg_path, dict):
        return dict(cfg_path)
    path_to_use = cfg_path if cfg_path is not None else DEFAULT_CONFIG_PATH
    if not os.path.exists(path_to_use):
        raise FileNotFoundError(f"Configuration file not found: {path_to_use}")
    with open(path_to_use, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


@torch.no_grad()
def compute_zstar(
    x: Any,
    cfg: Optional[Union[str, Path, Dict[str, Any]]] = None,
    backbone: Optional[Any] = None,
    w_g: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, float, int, Dict[str, Any]]:
    """Compute canonical fixed-point state z* for a given input x (prompt text or pre-encoded context).

    Args:
        x: String prompt or pre-encoded context object.
        cfg: Path to config YAML or dictionary of configuration parameters.
        backbone: GemmaCTSBackbone instance (optional; retrieved if not provided).
        w_g: Routing matrix tensor (optional; extracted from backbone if not provided).
        device: Torch device (optional; defaults to CUDA if available, else CPU).
        **kwargs: Parameter overrides for config (e.g., solver="anderson", tol=1e-4, max_nfe=1000).

    Returns:
        z_star: Canonical fixed point tensor [K_LATENT, hidden_size]
        final_rho_fp: Relative fixed point residual norm at termination
        nfe: Number of function evaluations / iterations executed
        log: Dictionary of trajectory metrics, timing, intermediate checkpoints, etc.
    """
    config = load_canonical_config(cfg)
    config.update(kwargs)

    # Read hyperparameters from config (strictly no hardcoding)
    g0 = float(config["g0"])
    alpha = float(config["alpha"])
    seed = int(config["seed"])
    tol = float(config.get("tol", 1e-4))
    max_nfe = int(config.get("max_nfe", 1000))
    solver = str(config.get("solver", "picard")).lower()

    k_latent = int(config.get("k_latent", 64))
    sigma_noise = float(config.get("sigma_noise", 0.020))
    top_k = int(config.get("top_k", 3))
    routing_mode = str(config.get("routing_mode", "sparse"))

    anderson_m = int(config.get("anderson_m", 5))
    anderson_beta = float(config.get("anderson_beta", 1.0))
    anderson_safeguard = bool(config.get("anderson_safeguard", True))

    # Resolve device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Resolve backbone
    if backbone is None:
        from harness_A_v1 import get_model
        backbone = get_model(str(device))

    d = backbone.hidden_size

    # Resolve routing matrix
    if w_g is None:
        w_g = backbone.routing_matrix().to(device=device, dtype=torch.float32)
    else:
        w_g = w_g.to(device=device, dtype=torch.float32)

    # Resolve context
    if isinstance(x, str):
        ctx = backbone.encode_context(x)
        if hasattr(ctx, "to"):
            ctx = ctx.to(device=device)
    else:
        ctx = x
        if hasattr(ctx, "to"):
            ctx = ctx.to(device=device)

    # 1. Compute z_src via root_fwdpass (diag_task13.py:150-156 canonical path)
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

    # 5. Build step function T(z) = (1 - alpha) * z + alpha * (Phi(z) + g0 * c_hat_0)
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

    # 6. Execute iteration according to solver
    if solver == "picard":
        z_cur = z0.clone()
        final_nfe = max_nfe
        for k in range(1, max_nfe + 1):
            z_next = step_fn(z_cur)
            diff = float(torch.norm(z_next - z_cur).item())
            norm_z = float(torch.norm(z_cur).item())
            rho_fp = diff / max(norm_z, 1e-12)

            res_traj.append(diff)
            rho_traj.append(rho_fp)

            if k % 100 == 0:
                checkpoints_100[k] = {"norm_z": float(torch.norm(z_next).item()), "diff": diff, "rho_fp": rho_fp}

            z_cur = z_next

            if tol is not None and tol > 0 and rho_fp < tol:
                final_nfe = k
                break

        z_star = z_cur
        final_diff = res_traj[-1]
        final_norm_z = float(torch.norm(z_star).item())
        final_rho_fp = rho_traj[-1]

    elif solver == "anderson":
        z_cur = z0.clone()
        final_nfe = max_nfe

        # Initial point evaluation
        T_z = step_fn(z_cur)
        f_cur = (T_z - z_cur).reshape(-1)
        diff = float(torch.norm(f_cur).item())
        norm_z = float(torch.norm(z_cur).item())
        rho_fp = diff / max(norm_z, 1e-12)

        res_traj.append(diff)
        rho_traj.append(rho_fp)

        if tol is not None and tol > 0 and rho_fp < tol:
            final_nfe = 1
            z_star = z_cur
            final_diff = diff
            final_norm_z = norm_z
            final_rho_fp = rho_fp
        else:
            Z_hist = [z_cur.reshape(-1).clone()]
            F_hist = [f_cur.clone()]
            z_cur = T_z  # step 1 is Picard step

            for k in range(2, max_nfe + 1):
                T_z = step_fn(z_cur)
                f_cur = (T_z - z_cur).reshape(-1)
                diff = float(torch.norm(f_cur).item())
                norm_z = float(torch.norm(z_cur).item())
                rho_fp = diff / max(norm_z, 1e-12)

                res_traj.append(diff)
                rho_traj.append(rho_fp)

                if k % 100 == 0:
                    checkpoints_100[k] = {"norm_z": norm_z, "diff": diff, "rho_fp": rho_fp}

                if tol is not None and tol > 0 and rho_fp < tol:
                    final_nfe = k
                    break

                # Safeguard: if residual increases, fallback to Picard step and reset history
                if anderson_safeguard and len(res_traj) >= 2 and diff > res_traj[-2]:
                    Z_hist = [z_cur.reshape(-1).clone()]
                    F_hist = [f_cur.clone()]
                    z_cur = T_z
                    continue

                Z_hist.append(z_cur.reshape(-1).clone())
                F_hist.append(f_cur.clone())

                if len(Z_hist) > anderson_m + 1:
                    Z_hist.pop(0)
                    F_hist.pop(0)

                m_k = len(Z_hist) - 1
                if m_k >= 1:
                    dF = torch.stack([F_hist[j + 1] - F_hist[j] for j in range(m_k)], dim=1)
                    dZ = torch.stack([Z_hist[j + 1] - Z_hist[j] for j in range(m_k)], dim=1)
                    gamma = torch.linalg.lstsq(dF, f_cur.unsqueeze(1)).solution
                    corr = ((dZ + anderson_beta * dF) @ gamma).squeeze(1).reshape_as(z_cur)
                    z_next = T_z - corr
                    z_cur = z_next
                else:
                    z_cur = T_z

            z_star = z_cur
            final_diff = res_traj[-1]
            final_norm_z = float(torch.norm(z_star).item())
            final_rho_fp = rho_traj[-1]
    else:
        raise ValueError(f"Unsupported solver: {solver}. Choose 'picard' or 'anderson'.")

    t_elapsed = time.time() - t0_start

    log: Dict[str, Any] = {
        "solver": solver,
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
    }

    return z_star, final_rho_fp, final_nfe, log

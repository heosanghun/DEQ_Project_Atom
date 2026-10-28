"""Hybrid DEQ equilibrium solver: Official Broyden + Dump/Uncompressed Solver fallback.

Specification:
1. Run Official Limited-Memory Broyden (memory_limit=16) up to budget_total.
2. If Broyden converges (rel < tol), return immediately with selected_solver="broyden".
3. If Broyden does not converge, run Phase 2 (Uncompressed/Dump Solver) from initial point z0
   with budget_total.
4. Compare best_rel from both runs, select the state with lower best_rel,
   and stamp selected_solver ("broyden" or "simple") and exact evaluation counts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Any
import torch

from cts.deq.broyden_forward import (
    _relative_residual,
    _limited_memory_broyden_batch,
    _lm_apply,
    _lm_apply_transpose,
    BroydenInfo,
)


@dataclass
class HybridSolverInfo:
    iterations: int
    residual_norm: float
    converged: bool
    selected_solver: str  # "broyden" or "simple"
    broyden_best_res: float
    simple_best_res: float
    all_residuals: List[float] = field(default_factory=list)
    broyden_iters: int = 0
    simple_iters: int = 0
    total_evals: int = 0


def solve_hybrid(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    budget_total: int = 1000,
    tol: float = 5e-3,
    memory_limit: int = 16,
    parent_inv_jacobian: Optional[Any] = None,
    is_root: bool = True,
) -> Tuple[torch.Tensor, HybridSolverInfo]:
    """Execute hybrid equilibrium solve for a single latent z0 [1, K, D] or [K, D].
    
    Note: Single-instance only; batch extension requires separate review.
    """
    orig_shape = z0.shape
    if z0.dim() == 2:
        z0 = z0.unsqueeze(0)
    batch = int(z0.shape[0])
    branch_shape = z0.shape[1:]
    
    # L2: Batch guard (single-instance only)
    assert batch == 1, (
        f"solve_hybrid is single-instance only (got batch={batch}); "
        "batch extension requires separate review"
    )
    
    # 1. Broyden phase (Official limited-memory Broyden, up to budget_total)
    z_broyden, broyden_infos = _limited_memory_broyden_batch(
        phi=phi,
        z0_batch=z0,
        tol=tol,
        max_iter=budget_total,
        parent_inv_jacobian=parent_inv_jacobian,
        fp32_buffer=True,
        memory_limit=memory_limit,
        is_root=is_root,
    )
    b_info = broyden_infos[0]
    b_res = float(b_info.residual_norm)
    b_iters = int(b_info.iterations)
    b_history = list(b_info.all_residuals)
    
    if b_info.converged or b_res < tol:
        info = HybridSolverInfo(
            iterations=b_iters,
            residual_norm=b_res,
            converged=True,
            selected_solver="broyden",
            broyden_best_res=b_res,
            simple_best_res=float("inf"),
            all_residuals=b_history,
            broyden_iters=b_iters,
            simple_iters=0,
            total_evals=b_iters,
        )
        return z_broyden.view(orig_shape), info

    # 2. Phase 2: Dump / Uncompressed solver from initial z0
    simple_budget = budget_total
    z = z0.detach().reshape(batch, -1).clone().float()
    n = int(z.shape[1])
    
    def G(vec: torch.Tensor) -> torch.Tensor:
        shaped = vec.to(z0.dtype).view(batch, *branch_shape)
        return phi(shaped).reshape(batch, -1).float()
    
    scale = 0.1
    work_capacity = simple_budget
    u_buf = torch.empty(batch, work_capacity, n, device=z0.device, dtype=torch.float32)
    v_buf = torch.empty_like(u_buf)
    work_rank = 0
    
    phi_z = G(z)
    residual = z - phi_z
    
    best_simple_res = float("inf")
    best_simple_z = z.clone()
    simple_history: List[float] = []
    simple_converged = False
    simple_used_iters = 0
    
    with torch.no_grad():
        for s_idx in range(1, simple_budget + 1):
            simple_used_iters = s_idx
            rel_norm, abs_norm = _relative_residual(residual, z, phi_z, dim=1)
            rel_val = float(rel_norm.item())
            simple_history.append(rel_val)
            
            if rel_val < best_simple_res:
                best_simple_res = rel_val
                best_simple_z = z.clone()
                
            if rel_val < tol:
                simple_converged = True
                break
                
            if not torch.isfinite(rel_norm):
                break
                
            step = -_lm_apply(residual, u_buf, v_buf, work_rank, scale)
            z_new = z + step
            phi_new = G(z_new)
            residual_new = z_new - phi_new
            
            s = z_new - z
            y = residual_new - residual
            hy = _lm_apply(y, u_buf, v_buf, work_rank, scale)
            ht_s = _lm_apply_transpose(s, u_buf, v_buf, work_rank, scale)
            denom = torch.sum(s * hy, dim=1)
            scale_denom = (
                torch.linalg.vector_norm(s, dim=1)
                * torch.linalg.vector_norm(hy, dim=1)
            ).clamp_min(1e-20)
            curvature = denom.abs() / scale_denom
            valid = torch.isfinite(denom) & (curvature > 1e-7)
            if bool(valid.item()) and work_rank < work_capacity:
                u_new = s - hy
                v_new = ht_s / denom
                u_buf[:, work_rank] = u_new
                v_buf[:, work_rank] = v_new
                work_rank += 1
                
            z = z_new
            phi_z = phi_new
            residual = residual_new

    # 3. Best selection between Broyden and Simple (Dump)
    if simple_converged:
        final_z = best_simple_z
        selected = "simple"
        final_res = best_simple_res
        is_conv = True
    elif b_res <= best_simple_res:
        final_z = z_broyden
        selected = "broyden"
        final_res = b_res
        is_conv = False
    else:
        final_z = best_simple_z
        selected = "simple"
        final_res = best_simple_res
        is_conv = False
        
    total_evals = b_iters + simple_used_iters
    info = HybridSolverInfo(
        iterations=total_evals,
        residual_norm=final_res,
        converged=is_conv,
        selected_solver=selected,
        broyden_best_res=b_res,
        simple_best_res=best_simple_res,
        all_residuals=b_history + simple_history,
        broyden_iters=b_iters,
        simple_iters=simple_used_iters,
        total_evals=total_evals,
    )
    return final_z.view(orig_shape), info

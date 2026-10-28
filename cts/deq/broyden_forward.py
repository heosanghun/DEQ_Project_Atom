"""L-Broyden solver: limited-memory FP32 inverse Jacobian (paper §5.2).

Paper Remark 2: Inherited Jacobians warm-start non-root nodes,
yielding average 11.2+-2.8 iterations (root: 14.8; non-root: 8.9).

For small n (< MAX_DENSE_N), uses dense n×n Jacobian (fast, exact).
For large n, uses a limited-memory inverse-Broyden operator (O(m*n)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

MAX_DENSE_N = 8192


@dataclass
class BroydenInfo:
    iterations: int
    residual_norm: float
    converged: bool
    all_residuals: List[float] = field(default_factory=list)
    jacobian_state: Optional[Any] = None
    # Absolute norm is diagnostic only.  ``residual_norm`` and
    # ``all_residuals`` follow the paper's scale-free relative-tolerance
    # contract.
    absolute_residual_norm: float = float("inf")
    # Exact logical number of Phi evaluations required by this branch.  This
    # is deliberately separate from ``iterations``: the small dense solver
    # evaluates the trial point after its final update, whereas the large
    # inverse-Broyden loop tests the already-cached Phi value.
    phi_eval_count: int = 0
    failure_reason: Optional[str] = None


@dataclass
class LimitedMemoryInverseJacobian:
    """Compact ``H = scale*I + sum_r u_r v_r^T`` inverse Jacobian.

    The tensors are ``[rank, n]`` for one branch.  Unlike a dense ``n x n``
    matrix this is O(rank*n), so Gemma's K=64,H=2560 state can inherit the
    parent quasi-Newton operator.  ``detach`` mirrors the Tensor API used by
    tree nodes.
    """

    scale: float
    u: torch.Tensor
    v: torch.Tensor
    validated: bool = False

    @property
    def rank(self) -> int:
        return int(self.u.shape[0])

    def detach(self) -> "LimitedMemoryInverseJacobian":
        return LimitedMemoryInverseJacobian(
            float(self.scale), self.u.detach(), self.v.detach(), self.validated
        )

    def to(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> "LimitedMemoryInverseJacobian":
        return LimitedMemoryInverseJacobian(
            float(self.scale),
            self.u.to(device=device, dtype=dtype),
            self.v.to(device=device, dtype=dtype),
            self.validated,
        )

    def for_tree_storage(self) -> "LimitedMemoryInverseJacobian":
        """Offload inherited factors from scarce VRAM between expansions."""
        return LimitedMemoryInverseJacobian(
            float(self.scale),
            # Preserve the quasi-Newton operator exactly between nodes.  FP16
            # can overflow valid factors and changes the inherited solve; the
            # latency gain is not worth silently changing the algorithm.
            self.u.detach().to(device="cpu", dtype=torch.float32),
            self.v.detach().to(device="cpu", dtype=torch.float32),
            True,
        )


@dataclass
class BroydenConvergenceStats:
    """Aggregate statistics matching paper Appendix C / Table 12."""

    total_solves: int = 0
    converged_count: int = 0
    fallback_count: int = 0
    iteration_counts: List[int] = field(default_factory=list)
    root_iterations: List[int] = field(default_factory=list)
    nonroot_iterations: List[int] = field(default_factory=list)

    @property
    def convergence_rate(self) -> float:
        return self.converged_count / max(self.total_solves, 1)

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / max(self.total_solves, 1)

    @property
    def avg_iterations(self) -> float:
        return sum(self.iteration_counts) / max(len(self.iteration_counts), 1)

    @property
    def avg_root_iterations(self) -> float:
        return sum(self.root_iterations) / max(len(self.root_iterations), 1)

    @property
    def avg_nonroot_iterations(self) -> float:
        return sum(self.nonroot_iterations) / max(len(self.nonroot_iterations), 1)

    def update(self, info: BroydenInfo, is_root: bool = True) -> None:
        self.total_solves += 1
        self.iteration_counts.append(info.iterations)
        if is_root:
            self.root_iterations.append(info.iterations)
        else:
            self.nonroot_iterations.append(info.iterations)
        if info.converged:
            self.converged_count += 1
        else:
            self.fallback_count += 1

    def report(self) -> Dict[str, float]:
        return {
            "convergence_rate": self.convergence_rate,
            "fallback_rate": self.fallback_rate,
            "avg_iterations": self.avg_iterations,
            "avg_root_iterations": self.avg_root_iterations,
            "avg_nonroot_iterations": self.avg_nonroot_iterations,
            "total_solves": float(self.total_solves),
        }


_global_stats: Optional[BroydenConvergenceStats] = None


def enable_convergence_tracking() -> BroydenConvergenceStats:
    global _global_stats
    _global_stats = BroydenConvergenceStats()
    return _global_stats


def get_convergence_stats() -> Optional[BroydenConvergenceStats]:
    return _global_stats


def map_nu_tol_to_tol(nu_tol: float, tol_min: float, tol_max: float) -> float:
    """Clamp the runtime relative tolerance to the declared paper range.

    ``MetaPolicy.raw_to_nu`` performs the normalized-to-physical mapping.
    Keeping a second affine mapping here broke fixed controls such as the
    documented Stage-1 mean ``5e-3`` (it became roughly ``1.5e-4``).
    """
    return max(float(tol_min), min(float(tol_max), float(nu_tol)))


def _relative_residual(
    residual: torch.Tensor,
    z: torch.Tensor,
    phi_z: torch.Tensor,
    *,
    dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(relative, absolute)`` fixed-point residual norms.

    The denominator is the larger of ``||z||`` and ``||phi(z)||``.  This is
    scale invariant, well-defined near zero, and avoids comparing the global
    L2 norm of a 163,840-D Gemma latent directly with a scalar tolerance.
    """
    absolute = torch.linalg.vector_norm(residual, dim=dim)
    z_norm = torch.linalg.vector_norm(z, dim=dim)
    phi_norm = torch.linalg.vector_norm(phi_z, dim=dim)
    # Standard mixed absolute/relative scaling: large latent states use a
    # scale-free residual, while roots near zero can still converge by the
    # absolute tolerance instead of being stuck at a constant ratio.
    denom = torch.maximum(z_norm, phi_norm).clamp_min(1.0)
    relative = absolute / denom
    # Fail closed when any norm overflows.  Otherwise a finite residual over
    # an infinite state norm would look like an exact root (relative=0).
    finite_norms = (
        torch.isfinite(absolute)
        & torch.isfinite(z_norm)
        & torch.isfinite(phi_norm)
    )
    relative = torch.where(
        finite_norms, relative, torch.full_like(relative, float("inf"))
    )
    return relative, absolute


def map_nu_ne_to_tol(nu_ne: float, tol_min: float, tol_max: float) -> float:
    return map_nu_tol_to_tol(nu_ne, tol_min, tol_max)


def _dense_broyden(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    tol: float,
    max_iter: int,
    parent_inv_jacobian: Optional[torch.Tensor],
    fp32_buffer: bool,
    memory_limit: int,
    is_root: bool,
) -> Tuple[torch.Tensor, BroydenInfo]:
    """Dense inverse-Broyden reference for small states.

    This path deliberately uses the same inverse-Jacobian contract as the
    Gemma-scale limited-memory solver.  The previous implementation stored a
    *direct* Jacobian while exposing it as ``inv_jacobian``, making inherited
    small-state tests exercise different mathematics from production.
    """
    orig_shape = z0.shape
    device = z0.device
    compute_dtype = torch.float32 if fp32_buffer else z0.dtype
    z = z0.detach().reshape(-1).to(device=device, dtype=compute_dtype).clone()
    n = z.numel()

    def F(vec: torch.Tensor) -> torch.Tensor:
        zz = vec.to(z0.dtype).view(orig_shape)
        p = phi(zz).reshape(-1).to(compute_dtype)
        return vec - p

    Fv = F(z)
    residuals: List[float] = []

    if (
        isinstance(parent_inv_jacobian, torch.Tensor)
        and parent_inv_jacobian.shape == (n, n)
    ):
        H = parent_inv_jacobian.to(device=device, dtype=compute_dtype).clone()
    else:
        # Algorithm 1 initial inverse-Jacobian scale.
        H = 0.1 * torch.eye(n, device=device, dtype=compute_dtype)

    for it in range(max_iter):
        phi_z = z - Fv
        rel_t, abs_t = _relative_residual(Fv, z, phi_z)
        res = float(rel_t.item())
        abs_res = float(abs_t.item())
        residuals.append(res)
        if res < tol:
            z_out = z.view(orig_shape).to(z0.dtype)
            info = BroydenInfo(
                it + 1,
                res,
                True,
                residuals,
                jacobian_state=H.detach().clone(),
                absolute_residual_norm=abs_res,
                phi_eval_count=it + 1,
            )
            if _global_stats is not None:
                _global_stats.update(info, is_root=is_root)
            return z_out, info

        # ``max_iter`` is a Phi-evaluation budget.  Do not make an untested
        # final update and then label it non-convergent.
        if it + 1 >= max_iter:
            break

        step = -(H @ Fv)

        z_new = z + step
        F_new = F(z_new)
        s = z_new - z
        y = F_new - Fv
        hy = H @ y
        ht_s = H.transpose(0, 1) @ s
        denom = torch.dot(s, hy)
        denom_floor = (
            1e-12 * torch.linalg.vector_norm(s) * torch.linalg.vector_norm(hy)
        ).clamp_min(1e-20)
        if bool(torch.isfinite(denom).item()) and bool(
            (denom.abs() > denom_floor).item()
        ):
            H = H + torch.outer(s - hy, ht_s / denom)
        z, Fv = z_new, F_new

    z_out = z.view(orig_shape).to(z0.dtype)
    phi_z = z - Fv
    rel_t, abs_t = _relative_residual(Fv, z, phi_z)
    info = BroydenInfo(
        max_iter,
        float(rel_t.item()),
        False,
        residuals,
        jacobian_state=H.detach().clone(),
        absolute_residual_norm=float(abs_t.item()),
        phi_eval_count=max_iter,
    )
    if _global_stats is not None:
        _global_stats.update(info, is_root=is_root)
    return z_out, info


def _anderson_broyden(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    tol: float,
    max_iter: int,
    fp32_buffer: bool,
    memory_limit: int,
    is_root: bool,
) -> Tuple[torch.Tensor, BroydenInfo]:
    """Anderson acceleration for large n. O(m*n) memory."""
    orig_shape = z0.shape
    device = z0.device
    compute_dtype = torch.float32 if fp32_buffer else z0.dtype
    z = z0.detach().reshape(-1).to(device=device, dtype=compute_dtype).clone()

    def G(vec: torch.Tensor) -> torch.Tensor:
        zz = vec.to(z0.dtype).view(orig_shape)
        p = phi(zz).reshape(-1).to(compute_dtype)
        return p

    residuals: List[float] = []
    x_hist: List[torch.Tensor] = []
    f_hist: List[torch.Tensor] = []
    beta = 1.0

    for it in range(max_iter):
        fz = G(z)
        res_vec = fz - z
        rel_t, abs_t = _relative_residual(res_vec, z, fz)
        res = float(rel_t.item())
        abs_res = float(abs_t.item())
        residuals.append(res)
        if res < tol:
            z_out = z.view(orig_shape).to(z0.dtype)
            info = BroydenInfo(
                it + 1,
                res,
                True,
                residuals,
                absolute_residual_norm=abs_res,
                phi_eval_count=it + 1,
            )
            if _global_stats is not None:
                _global_stats.update(info, is_root=is_root)
            return z_out, info

        x_hist.append(z.clone())
        f_hist.append(fz.clone())
        if len(x_hist) > memory_limit + 1:
            x_hist.pop(0)
            f_hist.pop(0)

        m = len(f_hist)
        if m < 2:
            z = beta * fz + (1 - beta) * z
            continue

        dF = torch.stack([f_hist[i] - f_hist[i - 1] for i in range(1, m)], dim=0)  # [m-1, n]
        dX = torch.stack([x_hist[i] - x_hist[i - 1] for i in range(1, m)], dim=0)  # [m-1, n]

        gram = dF @ dF.t()
        gram += 1e-6 * torch.eye(m - 1, device=device, dtype=compute_dtype)
        rhs = dF @ res_vec

        try:
            gamma = torch.linalg.solve(gram, rhs)  # [m-1], small solve
        except RuntimeError:
            z = beta * fz + (1 - beta) * z
            continue

        z_new = fz - (gamma @ (dF - dX))
        z = z_new

    z_out = z.view(orig_shape).to(z0.dtype)
    fz = G(z.reshape(-1).to(compute_dtype))
    final_z = z.reshape(-1).to(compute_dtype)
    final_delta = fz - final_z
    rel_t, abs_t = _relative_residual(final_delta, final_z, fz)
    info = BroydenInfo(
        max_iter,
        float(rel_t.item()),
        False,
        residuals,
        absolute_residual_norm=float(abs_t.item()),
        phi_eval_count=max_iter + 1,
    )
    if _global_stats is not None:
        _global_stats.update(info, is_root=is_root)
    return z_out, info


def broyden_fixed_point(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    tol: float,
    max_iter: int = 30,
    *,
    parent_inv_jacobian: Optional[Any] = None,
    fp32_buffer: bool = True,
    memory_limit: int = 16,
) -> Tuple[torch.Tensor, BroydenInfo]:
    """L-Broyden fixed-point solver (paper §5.2).

    Automatically selects dense (small n) or limited-memory (large n) mode.
    Dense: O(n^2) memory, uses full Jacobian, proven stable.
    Limited-memory inverse Broyden: O(m*n), scalable and inheritable.

    Paper rank = 16 (memory_limit default).
    """
    n = z0.numel()
    is_root = parent_inv_jacobian is None

    if n <= MAX_DENSE_N:
        return _dense_broyden(
            phi, z0, tol, max_iter, parent_inv_jacobian,
            fp32_buffer, memory_limit, is_root,
        )
    else:
        z_batch, infos = _limited_memory_broyden_batch(
            lambda zz: phi(zz[0]).unsqueeze(0),
            z0.unsqueeze(0),
            tol=tol,
            max_iter=max_iter,
            parent_inv_jacobian=parent_inv_jacobian,
            fp32_buffer=fp32_buffer,
            memory_limit=memory_limit,
            is_root=is_root,
        )
        return z_batch[0], infos[0]


def broyden_fixed_point_batch(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0_batch: torch.Tensor,
    tol: float,
    max_iter: int = 30,
    *,
    parent_inv_jacobian: Optional[Any] = None,
    fp32_buffer: bool = True,
    memory_limit: int = 16,
) -> Tuple[torch.Tensor, List[BroydenInfo]]:
    """Batch Broyden for W sibling branches (paper §4.1).

    Each branch inherits the same parent_inv_jacobian.
    """
    n_per_branch = z0_batch[0].numel()
    if n_per_branch > MAX_DENSE_N:
        return _limited_memory_broyden_batch(
            phi,
            z0_batch,
            tol=tol,
            max_iter=max_iter,
            parent_inv_jacobian=parent_inv_jacobian,
            fp32_buffer=fp32_buffer,
            memory_limit=memory_limit,
            is_root=parent_inv_jacobian is None,
        )

    from concurrent.futures import ThreadPoolExecutor

    W = z0_batch.shape[0]
    results: List[Optional[torch.Tensor]] = [None] * W
    infos: List[Optional[BroydenInfo]] = [None] * W

    def _solve_branch(i: int) -> None:
        z_star_i, info_i = broyden_fixed_point(
            phi,
            z0_batch[i],
            tol=tol,
            max_iter=max_iter,
            parent_inv_jacobian=parent_inv_jacobian,
            fp32_buffer=fp32_buffer,
            memory_limit=memory_limit,
        )
        results[i] = z_star_i
        infos[i] = info_i

    with ThreadPoolExecutor(max_workers=W) as executor:
        list(executor.map(_solve_branch, range(W)))

    return torch.stack(results), infos  # type: ignore[arg-type]


def _lm_apply(
    x: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    rank: int,
    scale: float,
) -> torch.Tensor:
    """Apply batched limited-memory inverse Jacobians to ``x=[B,N]``."""
    out = float(scale) * x
    if rank <= 0:
        return out
    ur = u[:, :rank]
    vr = v[:, :rank]
    coeff = torch.bmm(vr, x.unsqueeze(-1))
    return out + torch.bmm(ur.transpose(1, 2), coeff).squeeze(-1)


def _lm_apply_transpose(
    x: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    rank: int,
    scale: float,
) -> torch.Tensor:
    out = float(scale) * x
    if rank <= 0:
        return out
    ur = u[:, :rank]
    vr = v[:, :rank]
    coeff = torch.bmm(ur, x.unsqueeze(-1))
    return out + torch.bmm(vr.transpose(1, 2), coeff).squeeze(-1)


def _compress_lm_factors(
    u: torch.Tensor,
    v: torch.Tensor,
    max_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress ``sum u_r v_r^T`` after an exact solve, not during it.

    Batched thin-QR plus an SVD of the tiny rank-by-rank core avoids ever
    materialising the N-by-N operator.  Truncating only at the node boundary
    keeps every in-solve good-Broyden update mathematically consistent.
    """
    rank = int(u.shape[1])
    if rank <= int(max_rank):
        return u, v
    # CUDA QR/SVD kernels are not implemented for bf16 ("svd_cuda_gesvdjBatched
    # not implemented for 'BFloat16'"), and the tiny rank-by-rank core deserves
    # full precision anyway.  Merely calling ``.float()`` is insufficient when
    # the caller wraps the whole DEQ solve in BF16 autocast: QR/matmul can be
    # cast back to BF16 before SVD.  Disable autocast for this compression
    # island, factor in FP32, and return in the caller's dtype.
    in_dtype = u.dtype
    orig_device = u.device
    with torch.amp.autocast(u.device.type, enabled=False):
        u_fp32 = u.float()
        v_fp32 = v.float()
        try:
            qu, ru = torch.linalg.qr(u_fp32.transpose(1, 2), mode="reduced")
            qv, rv = torch.linalg.qr(v_fp32.transpose(1, 2), mode="reduced")
            core = torch.bmm(ru, rv.transpose(1, 2))
            left_core, singular, vh = torch.linalg.svd(
                core, full_matrices=False
            )
            keep = min(int(max_rank), int(singular.shape[-1]))
            sqrt_s = singular[:, :keep].clamp_min(0).sqrt()
            left = torch.bmm(
                qu,
                left_core[:, :, :keep] * sqrt_s.unsqueeze(1),
            )
            right_core = vh.transpose(1, 2)[:, :, :keep]
            right = torch.bmm(qv, right_core * sqrt_s.unsqueeze(1))
        except (torch.OutOfMemoryError, RuntimeError):
            if orig_device.type == "cuda":
                torch.cuda.empty_cache()
            u_cpu = u_fp32.cpu()
            v_cpu = v_fp32.cpu()
            qu, ru = torch.linalg.qr(u_cpu.transpose(1, 2), mode="reduced")
            qv, rv = torch.linalg.qr(v_cpu.transpose(1, 2), mode="reduced")
            core = torch.bmm(ru, rv.transpose(1, 2))
            left_core, singular, vh = torch.linalg.svd(
                core, full_matrices=False
            )
            keep = min(int(max_rank), int(singular.shape[-1]))
            sqrt_s = singular[:, :keep].clamp_min(0).sqrt()
            left = torch.bmm(
                qu,
                left_core[:, :, :keep] * sqrt_s.unsqueeze(1),
            ).to(device=orig_device)
            right_core = vh.transpose(1, 2)[:, :, :keep]
            right = torch.bmm(qv, right_core * sqrt_s.unsqueeze(1)).to(device=orig_device)
    return (
        left.transpose(1, 2).contiguous().to(in_dtype),
        right.transpose(1, 2).contiguous().to(in_dtype),
    )


def _limited_memory_broyden_batch(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0_batch: torch.Tensor,
    tol: float,
    max_iter: int,
    *,
    parent_inv_jacobian: Optional[Any],
    fp32_buffer: bool,
    memory_limit: int,
    is_root: bool,
) -> Tuple[torch.Tensor, List[BroydenInfo]]:
    """True W-batched limited-memory inverse-Broyden.

    Solves ``F(z)=z-phi(z)=0`` with the good inverse-Broyden rank-one update

    ``H+ = H + (s-Hy) (s^T H) / (s^T H y)``.

    Each sibling owns independent low-rank factors but all expensive ``phi``
    evaluations are issued as one W batch.  A parent compact state is cloned
    into every sibling, implementing Algorithm 1's Jacobian inheritance on
    the actual Gemma-scale path.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if memory_limit < 1:
        raise ValueError("memory_limit must be >= 1")

    branch_shape = z0_batch.shape[1:]
    batch = int(z0_batch.shape[0])
    compute_dtype = torch.float32 if fp32_buffer else z0_batch.dtype
    z = z0_batch.detach().reshape(batch, -1).to(dtype=compute_dtype).clone()
    n = int(z.shape[1])

    def G(vec: torch.Tensor) -> torch.Tensor:
        shaped = vec.to(z0_batch.dtype).view(batch, *branch_shape)
        return phi(shaped).reshape(batch, -1).to(compute_dtype)

    scale = 0.1  # Algorithm 1: B0 <- 0.1 I
    # Keep all updates exact during this solve.  Allocate only slots that can
    # actually be populated: root needs max_iter-1, while a non-root solve
    # needs parent_rank+max_iter-1.  ``empty`` avoids zero-filling ~100 MiB of
    # buffers whose used slots are overwritten before every read.
    rank = 0
    inherited = None
    if isinstance(parent_inv_jacobian, LimitedMemoryInverseJacobian):
        inherited = parent_inv_jacobian.to(device=z.device, dtype=compute_dtype)
        if inherited.u.dim() != 2 or inherited.v.dim() != 2:
            raise ValueError("inherited inverse-Broyden factors must be [rank,n]")
        if inherited.u.shape != inherited.v.shape or inherited.u.shape[1] != n:
            raise ValueError(
                "inherited inverse-Broyden state is incompatible with branch "
                f"size {n}: u={tuple(inherited.u.shape)}, "
                f"v={tuple(inherited.v.shape)}"
            )
        if not inherited.validated:
            if not bool(torch.isfinite(inherited.u).all().item()) or not bool(
                torch.isfinite(inherited.v).all().item()
            ):
                raise ValueError(
                    "inherited inverse-Broyden state contains non-finite values"
                )
        scale = float(inherited.scale)
        rank = min(int(inherited.rank), int(memory_limit))
    work_capacity = rank + max(0, int(max_iter) - 1)
    u_buf = torch.empty(
        batch, work_capacity, n, device=z.device, dtype=compute_dtype
    )
    v_buf = torch.empty_like(u_buf)
    if inherited is not None:
        if rank:
            u_buf[:, :rank] = inherited.u[-rank:].unsqueeze(0).expand(
                batch, -1, -1
            )
            v_buf[:, :rank] = inherited.v[-rank:].unsqueeze(0).expand(
                batch, -1, -1
            )
    work_rank = rank

    active = torch.ones(batch, device=z.device, dtype=torch.bool)
    converged_at = torch.zeros(batch, device=z.device, dtype=torch.long)
    failed_at = torch.zeros(batch, device=z.device, dtype=torch.long)
    converged_res = torch.full(
        (batch,), float("inf"), device=z.device, dtype=torch.float32
    )
    converged_abs = torch.full_like(converged_res, float("inf"))
    residual_buffer = torch.full(
        (max_iter, batch),
        float("nan"),
        device=z.device,
        dtype=torch.float32,
    )

    phi_z = G(z)
    residual = z - phi_z
    # A non-converged quasi-Newton trajectory can improve substantially and
    # then diverge on a later rank-one update.  Keep the best finite iterate
    # so training never receives an arbitrary last-step explosion.  Headline
    # inference still treats ``converged=False`` as a parent-state fallback.
    best_rel = torch.full(
        (batch,), float("inf"), device=z.device, dtype=torch.float32
    )
    best_abs = torch.full_like(best_rel, float("inf"))
    best_z = z.clone()

    for eval_idx in range(max_iter):
        rel_norm, abs_norm = _relative_residual(
            residual, z, phi_z, dim=1
        )
        finite_residual = active & torch.isfinite(rel_norm)
        improved = finite_residual & (rel_norm < best_rel)
        best_rel = torch.where(improved, rel_norm.float(), best_rel)
        best_abs = torch.where(improved, abs_norm.float(), best_abs)
        best_z = torch.where(improved.unsqueeze(1), z, best_z)
        numerical_failure = active & ~torch.isfinite(rel_norm)
        failed_at = torch.where(
            numerical_failure,
            torch.full_like(failed_at, eval_idx + 1),
            failed_at,
        )
        active = active & ~numerical_failure
        residual_buffer[eval_idx] = torch.where(
            active, rel_norm.float(), residual_buffer[eval_idx]
        )
        newly = active & (rel_norm < float(tol))
        converged_at = torch.where(
            newly,
            torch.full_like(converged_at, eval_idx + 1),
            converged_at,
        )
        converged_res = torch.where(newly, rel_norm.float(), converged_res)
        converged_abs = torch.where(newly, abs_norm.float(), converged_abs)
        active = active & ~newly

        if not bool(active.any().item()) or eval_idx + 1 >= max_iter:
            break

        step = -_lm_apply(residual, u_buf, v_buf, work_rank, scale)
        step = torch.where(active.unsqueeze(1), step, torch.zeros_like(step))
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
        valid = active & torch.isfinite(denom) & (curvature > 1e-7)
        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        u_new = s - hy
        v_new = ht_s / safe_denom.unsqueeze(1)
        u_new = torch.where(valid.unsqueeze(1), u_new, torch.zeros_like(u_new))
        v_new = torch.where(valid.unsqueeze(1), v_new, torch.zeros_like(v_new))
        if work_rank >= work_capacity:
            raise RuntimeError("inverse-Broyden work-factor capacity exhausted")
        u_buf[:, work_rank] = u_new
        v_buf[:, work_rank] = v_new
        work_rank += 1

        z = torch.where(active.unsqueeze(1), z_new, z)
        phi_z = torch.where(active.unsqueeze(1), phi_new, phi_z)
        residual = torch.where(active.unsqueeze(1), residual_new, residual)

    failed = converged_at == 0
    returned_z = torch.where(failed.unsqueeze(1), best_z, z)
    final_rel = torch.where(failed, best_rel, converged_res)
    final_abs = torch.where(failed, best_abs, converged_abs)
    residual_host = residual_buffer.detach().cpu()

    success_indices = torch.nonzero(
        converged_at > 0, as_tuple=False
    ).flatten()
    compressed_u = compressed_v = None
    success_position: Dict[int, int] = {}
    if success_indices.numel() > 0:
        if batch == 1 and int(success_indices[0].item()) == 0:
            u_sel = u_buf[:, :work_rank]
            v_sel = v_buf[:, :work_rank]
        else:
            u_sel = u_buf.index_select(0, success_indices)[:, :work_rank]
            v_sel = v_buf.index_select(0, success_indices)[:, :work_rank]
        compressed_u, compressed_v = _compress_lm_factors(
            u_sel,
            v_sel,
            memory_limit,
        )
        success_position = {
            int(branch): pos
            for pos, branch in enumerate(success_indices.detach().cpu().tolist())
        }

    infos: List[BroydenInfo] = []
    for bi in range(batch):
        converged = int(converged_at[bi].item()) > 0
        numerical_failure_at = int(failed_at[bi].item())
        iterations = int(converged_at[bi].item()) if converged else (
            numerical_failure_at if numerical_failure_at > 0 else int(max_iter)
        )
        vals = residual_host[:, bi]
        history = [float(v) for v in vals[torch.isfinite(vals)].tolist()]
        state = None
        if converged:
            assert compressed_u is not None and compressed_v is not None
            pos = success_position[bi]
            state = LimitedMemoryInverseJacobian(
                scale=float(scale),
                u=compressed_u[pos].detach().clone(),
                v=compressed_v[pos].detach().clone(),
                validated=True,
            )
        info = BroydenInfo(
            iterations=iterations,
            residual_norm=float(final_rel[bi].item()),
            converged=converged,
            all_residuals=history,
            jacobian_state=state,
            absolute_residual_norm=float(final_abs[bi].item()),
            phi_eval_count=iterations,
            failure_reason=(
                "non_finite_residual"
                if numerical_failure_at > 0
                else (None if converged else "max_iter")
            ),
        )
        infos.append(info)
        if _global_stats is not None:
            _global_stats.update(info, is_root=is_root)

    return returned_z.view(batch, *branch_shape).to(z0_batch.dtype), infos


def _anderson_broyden_batch(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0_batch: torch.Tensor,
    tol: float,
    max_iter: int,
    *,
    fp32_buffer: bool,
    memory_limit: int,
    is_root: bool,
) -> Tuple[torch.Tensor, List[BroydenInfo]]:
    """Independent Anderson histories with one batched ``phi([W,K,H])``."""
    branch_shape = z0_batch.shape[1:]
    batch = int(z0_batch.shape[0])
    compute_dtype = torch.float32 if fp32_buffer else z0_batch.dtype
    z = z0_batch.detach().reshape(batch, -1).to(dtype=compute_dtype).clone()

    def G(vec: torch.Tensor) -> torch.Tensor:
        shaped = vec.to(z0_batch.dtype).view(batch, *branch_shape)
        return phi(shaped).reshape(batch, -1).to(compute_dtype)

    active = torch.ones(batch, device=z.device, dtype=torch.bool)
    converged_at = torch.zeros(batch, device=z.device, dtype=torch.long)
    converged_res = torch.full(
        (batch,), float("inf"), device=z.device, dtype=torch.float32
    )
    converged_abs = torch.full(
        (batch,), float("inf"), device=z.device, dtype=torch.float32
    )
    # Keep residual logging on device and transfer it once after the solve.
    # The historical loop copied active masks and every residual to CPU at
    # every iteration, adding several GPU synchronization barriers per phi.
    residual_buffer = torch.full(
        (max_iter, batch),
        float("nan"),
        device=z.device,
        dtype=torch.float32,
    )
    x_hist: List[torch.Tensor] = []
    f_hist: List[torch.Tensor] = []
    eye_full = torch.eye(
        max(1, memory_limit), device=z.device, dtype=compute_dtype
    ).unsqueeze(0)

    for it in range(max_iter):
        fz = G(z)
        res_vec = fz - z
        res_norm, abs_norm = _relative_residual(res_vec, z, fz, dim=1)
        residual_buffer[it] = torch.where(
            active, res_norm.float(), residual_buffer[it]
        )

        newly = active & (res_norm < float(tol))
        converged_at = torch.where(
            newly,
            torch.full_like(converged_at, it + 1),
            converged_at,
        )
        converged_res = torch.where(newly, res_norm.float(), converged_res)
        converged_abs = torch.where(newly, abs_norm.float(), converged_abs)
        active = active & ~newly
        if not bool(active.any().item()):
            break

        x_hist.append(z.clone())
        f_hist.append(fz.clone())
        if len(x_hist) > memory_limit + 1:
            x_hist.pop(0)
            f_hist.pop(0)

        if len(f_hist) < 2:
            candidate = fz
        else:
            dF = torch.stack(
                [f_hist[i] - f_hist[i - 1] for i in range(1, len(f_hist))],
                dim=1,
            )
            dX = torch.stack(
                [x_hist[i] - x_hist[i - 1] for i in range(1, len(x_hist))],
                dim=1,
            )
            gram = torch.bmm(dF, dF.transpose(1, 2))
            eye = eye_full[:, : gram.shape[-1], : gram.shape[-1]]
            gram = gram + 1e-6 * eye
            rhs = torch.bmm(dF, res_vec.unsqueeze(-1))
            gamma_raw, solve_info = torch.linalg.solve_ex(gram, rhs)
            gamma = gamma_raw.squeeze(-1)
            correction = torch.bmm(
                gamma.unsqueeze(1), dF - dX
            ).squeeze(1)
            accelerated = fz - correction
            candidate = torch.where(
                (solve_info == 0).unsqueeze(1), accelerated, fz
            )
        z = torch.where(active.unsqueeze(1), candidate, z)

    failed = converged_at == 0
    if bool(failed.any().item()):
        final_f = G(z)
        failed_norm, failed_abs = _relative_residual(
            final_f - z, z, final_f, dim=1
        )
        final_norm = torch.where(failed, failed_norm, converged_res)
        final_abs_norm = torch.where(failed, failed_abs, converged_abs)
    else:
        final_norm = converged_res
        final_abs_norm = converged_abs

    residual_host = residual_buffer.detach().cpu()
    residual_histories: List[List[float]] = []
    for bi in range(batch):
        vals = residual_host[:, bi]
        residual_histories.append(
            [float(v) for v in vals[torch.isfinite(vals)].tolist()]
        )

    infos: List[BroydenInfo] = []
    for bi in range(batch):
        converged = int(converged_at[bi].item()) > 0
        iterations = int(converged_at[bi].item()) if converged else int(max_iter)
        residual = (
            float(converged_res[bi].item())
            if converged
            else float(final_norm[bi].item())
        )
        info = BroydenInfo(
            iterations,
            residual,
            converged,
            residual_histories[bi],
            absolute_residual_norm=float(final_abs_norm[bi].item()),
            phi_eval_count=iterations + (0 if converged else 1),
        )
        infos.append(info)
        if _global_stats is not None:
            _global_stats.update(info, is_root=is_root)

    return z.view(batch, *branch_shape).to(z0_batch.dtype), infos

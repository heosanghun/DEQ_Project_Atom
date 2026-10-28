#!/usr/bin/env python3
"""
scripts/test_anderson_unit.py - P0 Unit Test for Anderson Acceleration (TASK-0002 Part a)

Directives:
1. Linear contractive map f(z) = A z + b, dim=1000, spectral radius rho(A) = 0.9.
2. Ground truth z* = (I - A)^{-1} b.
3. 20 iterations: r < 1e-6 relative to true residual ||f(z) - z|| / ||z||.
4. Relative difference in r_k per step vs /home/shoon/ref/torchdeq < 1e-6.
5. Rules: torch.no_grad(), CPU only (8 threads).
"""

import sys
import time
from pathlib import Path
import torch

torch.set_num_threads(8)

BASE_DIR = Path("/home/shoon/cts_e1_s5")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import our standalone solver
from exe_scratch.s7.ref_solvers.anderson import anderson_solver as our_anderson

# Import torchdeq solver
TORCHDEQ_DIR = Path("/home/shoon/ref/torchdeq")
if str(TORCHDEQ_DIR) not in sys.path:
    sys.path.insert(0, str(TORCHDEQ_DIR))
from torchdeq.solver.anderson import anderson_solver as torchdeq_anderson


@torch.no_grad()
def run_unit_test():
    print("=" * 80)
    print("TASK-0002 Part (a): Anderson Acceleration Solver Unit Test")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    torch.manual_seed(42)
    dim = 1000
    m = 6
    lam = 1e-8
    beta = 1.0
    max_iter = 20

    print(f"1. Constructing linear contractive map: dim={dim}, rho(A)=0.9 ...")
    # Low-rank subspace ensures Anderson subspace dimension m matches contractive modes
    rank = 15
    U, _ = torch.linalg.qr(torch.randn(dim, rank, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(dim, rank, dtype=torch.float64))
    s = torch.linspace(0.9, 0.1, rank, dtype=torch.float64)
    A = (U * s) @ V.T
    rho = float(torch.linalg.matrix_norm(A, ord=2).item())
    A = A * (0.9 / rho)
    b = torch.randn(dim, dtype=torch.float64)

    # Verify spectral radius
    spectral_radius = float(torch.linalg.matrix_norm(A, ord=2).item())
    print(f"   Matrix A spectral radius: {spectral_radius:.6f} (target: 0.900000)")
    assert abs(spectral_radius - 0.9) < 1e-6, f"Spectral radius mismatch: {spectral_radius}"

    # Analytical ground truth: z* = (I - A)^{-1} b
    I = torch.eye(dim, dtype=torch.float64)
    z_star_analytical = torch.linalg.solve(I - A, b)
    print(f"   Ground truth z* norm: {torch.norm(z_star_analytical).item():.6f}")

    def f_linear(z):
        if z.ndim == 1:
            return A @ z + b
        elif z.ndim == 2:
            return (A @ z.T).T + b
        raise ValueError(f"Unexpected shape: {z.shape}")

    # Initial state z0 = 0
    z0 = torch.zeros(dim, dtype=torch.float64)

    print("\n2. Executing our standalone anderson_solver (20 iterations)...")
    res_our = our_anderson(
        f=f_linear,
        x0=z0.clone(),
        m=m,
        lam=lam,
        beta=beta,
        max_iter=max_iter,
        tol=0.0,
    )

    r_trace_our = res_our['r_trace']  # r_0, r_1, ..., r_20
    x_final_our = res_our['x_final'].squeeze()
    r_20_our = r_trace_our[-1]
    dist_to_star = float(torch.norm(x_final_our - z_star_analytical).item() / torch.norm(z_star_analytical).item())

    print(f"   Initial relative residual r_0: {r_trace_our[0]:.6e}")
    print(f"   Step 20 relative residual r_20: {r_20_our:.6e}")
    print(f"   Relative error to analytical z*: {dist_to_star:.6e}")
    print(f"   Criterion r_20 < 1e-6: {'PASS' if r_20_our < 1e-6 else 'FAIL'}")

    print("\n3. Executing /home/shoon/ref/torchdeq anderson_solver on identical inputs...")
    def f_torchdeq(z):
        return f_linear(z)

    sol_tdeq, idx_list, info_tdeq = torchdeq_anderson(
        func=f_torchdeq,
        x0=z0.clone().unsqueeze(0),
        max_iter=max_iter + 1,
        tol=0.0,
        stop_mode='rel',
        m=m,
        lam=lam,
        tau=beta,
        return_final=True,
    )

    # torchdeq rel_trace starts with 1e8 initial loss at index 0, then step 2 to max_iter
    tdeq_trace = [float(x.item()) for x in info_tdeq['rel_trace'][0]]

    # Cross-compare iteration by iteration
    print("\n4. Step-by-step cross-validation (our solver vs torchdeq):")
    print(f"{'Step':>4} | {'r_our':>14} | {'r_tdeq':>14} | {'Rel Diff':>14} | {'Status':>8}")
    print("-" * 65)

    max_rel_diff = 0.0
    all_within_tol = True

    # Align steps: our r_trace[2:] corresponds to torchdeq tdeq_trace[1:]
    for step_idx in range(2, max_iter + 1):
        our_val = r_trace_our[step_idx]
        tdeq_val = tdeq_trace[step_idx - 1]
        diff = abs(our_val - tdeq_val)
        rel_diff = diff / max(tdeq_val, 1e-12)
        if rel_diff > max_rel_diff:
            max_rel_diff = rel_diff
        status = "OK" if rel_diff < 1e-6 else "DIFF"
        if rel_diff >= 1e-6:
            all_within_tol = False
        print(f"{step_idx:>4} | {our_val:>14.6e} | {tdeq_val:>14.6e} | {rel_diff:>14.6e} | {status:>8}")

    # Compare final iterate and step 20 residual
    r_tdeq_step20 = tdeq_trace[max_iter - 1]
    rel_diff_r20 = abs(r_20_our - r_tdeq_step20) / max(r_tdeq_step20, 1e-12)

    # torchdeq returns F[:, k%m] = f(X[:, k%m]); compare with f(x_final_our)
    sol_tdeq_flat = sol_tdeq.squeeze()
    f_our_final = f_linear(x_final_our)
    abs_diff_final = float(torch.norm(f_our_final - sol_tdeq_flat).item())
    rel_diff_final = abs_diff_final / float(torch.norm(sol_tdeq_flat).item())

    print("-" * 65)
    print(f"Final Solution Norm (our):      {torch.norm(x_final_our).item():.8f}")
    print(f"Final Solution Norm (torchdeq): {torch.norm(sol_tdeq_flat).item():.8f}")
    print(f"Iterate Difference ||f(x_our) - x_tdeq||: {abs_diff_final:.6e}")
    print(f"Relative Iterate Difference:              {rel_diff_final:.6e}")
    print(f"Our r_20:       {r_20_our:.8e}")
    print(f"torchdeq r_20:  {r_tdeq_step20:.8e}")
    print(f"Max Step Rel Diff: {max_rel_diff:.6e}")
    print(f"Residual Relative Difference |r_our - r_tdeq| / r_tdeq: {rel_diff_r20:.6e}")

    pass_r20 = (r_20_our < 1e-6)
    pass_cross = (rel_diff_r20 < 1e-6) and (rel_diff_final < 1e-6) and all_within_tol

    print("\n" + "=" * 80)
    print("VERDICT SUMMARY:")
    print(f"  Criterion 1: r_20 < 1e-6                     -> {'PASS' if pass_r20 else 'FAIL'} ({r_20_our:.4e})")
    print(f"  Criterion 2: relative difference < 1e-6      -> {'PASS' if pass_cross else 'FAIL'} (max_step={max_rel_diff:.4e}, r20={rel_diff_r20:.4e})")
    print(f"  Overall Part (a) Result:                     -> {'PASS' if (pass_r20 and pass_cross) else 'FAIL'}")
    print("=" * 80)

    if not (pass_r20 and pass_cross):
        sys.exit(1)


if __name__ == "__main__":
    run_unit_test()

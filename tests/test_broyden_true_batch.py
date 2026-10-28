from __future__ import annotations

import torch

from cts.deq.broyden_forward import (
    _compress_lm_factors,
    broyden_fixed_point,
    broyden_fixed_point_batch,
)
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights


def test_large_sibling_anderson_uses_one_batched_phi_stream():
    torch.manual_seed(7)
    z0 = torch.randn(3, 8200)
    target = torch.linspace(-0.2, 0.2, 8200)
    batch_calls = 0

    def phi_batch(z):
        nonlocal batch_calls
        batch_calls += 1
        return 0.5 * z + 0.5 * target

    z_batch, infos = broyden_fixed_point_batch(
        phi_batch, z0, tol=1e-5, max_iter=20
    )

    serial = []
    serial_calls = 0
    for branch in z0:
        def phi_one(z):
            nonlocal serial_calls
            serial_calls += 1
            return 0.5 * z + 0.5 * target

        solved, _ = broyden_fixed_point(phi_one, branch, tol=1e-5, max_iter=20)
        serial.append(solved)

    assert torch.allclose(z_batch, torch.stack(serial), atol=2e-5, rtol=2e-5)
    assert batch_calls < serial_calls
    assert len(infos) == 3


def test_routing_reference_preserves_leading_batch_dimension():
    z = torch.randn(3, 4, 8)
    gate = torch.randn(19, 8)
    alpha = routing_weights(z, gate, 1.0)
    sparse = sparse_module_weights(alpha, 3)
    assert alpha.shape == (3, 19)
    assert sparse.shape == (3, 19)
    assert torch.allclose(sparse.sum(dim=-1), torch.ones(3))
    assert torch.count_nonzero(sparse, dim=-1).tolist() == [3, 3, 3]


def test_large_broyden_reports_exact_initial_convergence_count():
    z0 = torch.randn(8200)
    calls = 0

    def phi(z):
        nonlocal calls
        calls += 1
        return z

    _, info = broyden_fixed_point(phi, z0, tol=1e-6, max_iter=3)
    assert calls == 1
    assert info.converged
    assert info.iterations == info.phi_eval_count == 1
    assert info.failure_reason is None


def test_large_broyden_max_iter_is_fail_closed_and_exactly_counted():
    z0 = torch.zeros(8200)
    calls = 0

    def phi(z):
        nonlocal calls
        calls += 1
        return z + 1.0

    _, info = broyden_fixed_point(phi, z0, tol=1e-12, max_iter=3)
    assert calls == 3
    assert not info.converged
    assert info.iterations == info.phi_eval_count == 3
    assert info.failure_reason == "max_iter"
    assert info.jacobian_state is None


def test_large_broyden_returns_best_finite_iterate_after_late_divergence():
    z0 = -torch.ones(8200)

    def phi(z):
        # Deterministic nonlinear map: the first inverse-Broyden update
        # improves the relative residual and the second makes it worse.
        return 3.0 * z - z.pow(3) - 2.0

    solved, info = broyden_fixed_point(phi, z0, tol=1e-12, max_iter=3)

    assert not info.converged
    assert info.failure_reason == "max_iter"
    assert info.residual_norm == min(info.all_residuals)
    assert torch.allclose(solved, torch.full_like(z0, -1.3), atol=1e-5)


def test_large_broyden_returns_best_finite_iterate_after_late_nan():
    z0 = torch.ones(8200)

    def phi(z):
        if float(z.mean()) <= 1.005:
            return z + 0.1
        return torch.full_like(z, torch.nan)

    solved, info = broyden_fixed_point(phi, z0, tol=1e-12, max_iter=3)

    assert not info.converged
    assert info.failure_reason == "non_finite_residual"
    assert info.iterations == info.phi_eval_count == 2
    assert len(info.all_residuals) == 1
    assert info.residual_norm == info.all_residuals[0]
    assert torch.isfinite(solved).all()
    assert torch.equal(solved, z0)


def test_large_batched_broyden_tracks_early_nan_per_branch():
    z0 = torch.ones(2, 8200)
    physical_calls = 0

    def phi(z):
        nonlocal physical_calls
        physical_calls += 1
        out = 0.5 * z
        out[0] = torch.nan
        return out

    _, infos = broyden_fixed_point_batch(phi, z0, tol=1e-12, max_iter=3)
    assert infos[0].iterations == infos[0].phi_eval_count == 1
    assert infos[0].failure_reason == "non_finite_residual"
    assert infos[0].jacobian_state is None
    assert infos[1].phi_eval_count == 3
    assert physical_calls == max(info.phi_eval_count for info in infos)


def test_large_batched_broyden_isolates_failed_lane_from_converged_lane():
    z0 = torch.ones(2, 8200)
    target = torch.full((8200,), 0.25)
    physical_calls = 0

    def phi(z):
        nonlocal physical_calls
        physical_calls += 1
        out = 0.25 * z + 0.75 * target
        if float(z[0].mean()) > 1.005:
            out[0] = torch.nan
        else:
            out[0] = z[0] + 0.1
        return out

    solved, infos = broyden_fixed_point_batch(
        phi, z0, tol=1e-5, max_iter=20
    )

    assert infos[0].failure_reason == "non_finite_residual"
    assert infos[0].iterations == infos[0].phi_eval_count == 2
    assert infos[0].jacobian_state is None
    assert torch.equal(solved[0], z0[0])
    assert infos[1].converged
    assert infos[1].failure_reason is None
    assert infos[1].residual_norm < 1e-5
    assert infos[1].jacobian_state is not None
    assert infos[1].jacobian_state.validated
    assert torch.allclose(solved[1], target, atol=2e-5, rtol=2e-5)
    direct = phi(solved.clone())[1]
    direct_rel = torch.linalg.vector_norm(solved[1] - direct) / torch.maximum(
        torch.linalg.vector_norm(solved[1]),
        torch.linalg.vector_norm(direct),
    )
    assert direct_rel < 1e-5
    assert physical_calls - 1 == max(info.phi_eval_count for info in infos)


def test_large_broyden_does_not_false_converge_on_overflowed_state_norm():
    z0 = torch.full((8200,), 1e30)

    def phi(z):
        out = z.clone()
        out[0] = 0.0
        return out

    solved, info = broyden_fixed_point(phi, z0, tol=1e-2, max_iter=3)

    assert not info.converged
    assert info.failure_reason == "non_finite_residual"
    assert info.residual_norm == float("inf")
    assert torch.equal(solved, z0)


def test_factor_compression_forces_fp32_inside_outer_bf16_autocast():
    torch.manual_seed(19)
    u = torch.randn(2, 20, 64, dtype=torch.bfloat16)
    v = torch.randn(2, 20, 64, dtype=torch.bfloat16)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        compressed_u, compressed_v = _compress_lm_factors(u, v, 16)

    assert compressed_u.shape == (2, 16, 64)
    assert compressed_v.shape == (2, 16, 64)
    assert compressed_u.dtype == torch.bfloat16
    assert compressed_v.dtype == torch.bfloat16
    assert torch.isfinite(compressed_u).all()
    assert torch.isfinite(compressed_v).all()

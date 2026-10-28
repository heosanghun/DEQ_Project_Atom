from __future__ import annotations

import torch
import torch.nn as nn

import cts.deq.transition as transition_mod
from cts.deq.broyden_forward import BroydenInfo
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights
from cts.types import NuVector, RuntimeBudgetState


class _RoutingSpyBackbone(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self._hidden = hidden
        self.anchor = nn.Parameter(torch.zeros(()))
        self.gate = nn.Parameter(
            torch.arange(19 * hidden, dtype=torch.float32).reshape(19, hidden)
            / 100.0
        )
        self.seen_weights: list[torch.Tensor] = []
        self.seen_ids: list[list[int]] = []
        self.prepare_calls = 0

    @property
    def hidden_size(self):
        return self._hidden

    @property
    def supports_batched_deq(self):
        return True

    def encode_context(self, text):
        return torch.zeros(1, self._hidden)

    def routing_matrix(self):
        return self.gate

    def prepare_deq_context(self, z, context, module_ids):
        self.prepare_calls += 1
        return {"prepared": True, "shape": tuple(z.shape)}

    def deq_step(self, z, context, module_weights, extra):
        self.seen_weights.append(module_weights.detach().clone())
        self.seen_ids.append(list(extra["routing_module_ids"]))
        return 0.5 * z


def _info() -> BroydenInfo:
    return BroydenInfo(
        iterations=2,
        residual_norm=0.0,
        converged=True,
        phi_eval_count=2,
    )


def test_transition_routes_once_from_parent_fixed_point(monkeypatch):
    bb = _RoutingSpyBackbone()
    parent = torch.linspace(-1.0, 1.0, 4 * 8).reshape(4, 8)
    nu = NuVector(nu_temp=0.7)

    def fake_solver(phi, z0, **kwargs):
        phi(z0)
        phi(-z0)
        return z0, _info()

    monkeypatch.setattr(transition_mod, "broyden_fixed_point", fake_solver)
    result = transition_mod.transition(
        "problem",
        0,
        nu,
        RuntimeBudgetState(),
        bb,
        K=4,
        d=8,
        top_k=3,
        parent_z_star=parent,
        noise_sigma=0.0,
        decode_child=False,
    )

    expected = sparse_module_weights(
        routing_weights(parent, bb.gate, nu.nu_temp), 3
    )
    assert result.prune is False
    assert bb.prepare_calls == 1
    assert len(bb.seen_weights) == 2
    torch.testing.assert_close(bb.seen_weights[0], expected)
    assert torch.equal(bb.seen_weights[0], bb.seen_weights[1])
    assert bb.seen_ids[0] == bb.seen_ids[1]


def test_batched_transition_expands_one_parent_route_to_all_siblings(monkeypatch):
    bb = _RoutingSpyBackbone()
    parent = torch.linspace(-0.5, 0.5, 4 * 8).reshape(4, 8)
    nu = NuVector(nu_temp=1.1)

    def fake_solver(phi, z0, **kwargs):
        phi(z0)
        phi(-z0)
        return z0, [_info(), _info(), _info()]

    monkeypatch.setattr(
        transition_mod, "broyden_fixed_point_batch", fake_solver
    )
    results = transition_mod.transition_batch(
        "problem",
        nu,
        RuntimeBudgetState(),
        bb,
        W=3,
        K=4,
        d=8,
        top_k=3,
        parent_z_star=parent,
        noise_sigma=0.0,
        decode_child=False,
    )

    expected_row = sparse_module_weights(
        routing_weights(parent, bb.gate, nu.nu_temp), 3
    )
    expected = expected_row.unsqueeze(0).expand(3, -1)
    assert len(results) == 3
    assert bb.prepare_calls == 1
    assert len(bb.seen_weights) == 2
    torch.testing.assert_close(bb.seen_weights[0], expected)
    assert torch.equal(bb.seen_weights[0], bb.seen_weights[1])

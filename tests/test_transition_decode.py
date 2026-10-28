"""transition() passes max_decode_tokens to backbones that support it."""

from __future__ import annotations

import torch
import pytest

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.transition import transition, transition_batch
from cts.types import NuVector, RuntimeBudgetState


class _BackboneWithDecode(MockTinyBackbone):
    def __init__(self) -> None:
        super().__init__(hidden=32, num_layers=8)
        self.last_max_new_tokens: int | None = None

    def decode_from_z_star(self, z_star: torch.Tensor, *, max_new_tokens: int = 1) -> str:
        self.last_max_new_tokens = max_new_tokens
        return "ok"


class _IdentityBackbone(_BackboneWithDecode):
    def deq_step(self, z, context, module_weights, extra):
        return z


def test_transition_passes_max_decode_tokens():
    bb = _BackboneWithDecode()
    nu = NuVector(nu_tol=5e-3, nu_temp=1.0, nu_expl=1.0)
    budget = RuntimeBudgetState()
    r = transition(
        "x",
        0,
        nu,
        budget,
        bb,
        K=4,
        d=32,
        broyden_max_iter=30,
        broyden_tol_min=1e-3,
        broyden_tol_max=1e-2,
        tau_flops_budget=1e20,
        max_decode_tokens=17,
    )
    assert not r.prune
    assert bb.last_max_new_tokens == 17
    assert r.child_text == "ok"


def test_transition_can_skip_intermediate_decode_and_reuse_context():
    bb = _BackboneWithDecode()
    nu = NuVector(nu_tol=5e-3, nu_temp=1.0, nu_expl=1.0)
    context = bb.encode_context("original problem")
    r = transition(
        "original problem",
        0,
        nu,
        RuntimeBudgetState(),
        bb,
        K=4,
        d=32,
        broyden_max_iter=30,
        broyden_tol_min=1e-3,
        broyden_tol_max=1e-2,
        tau_flops_budget=1e20,
        context_override=context,
        decode_child=False,
    )
    assert not r.prune
    assert bb.last_max_new_tokens is None
    assert r.child_text == "original problem"


def test_branch_action_selects_stable_distinct_noise_stream_and_act_does_not_scale_mac():
    bb = _IdentityBackbone()
    parent = torch.zeros(4, 32)

    def run(branch: int, nu_act: float):
        return transition(
            "problem",
            branch,
            NuVector(nu_tol=5e-3, nu_temp=1.0, nu_expl=1.0, nu_act=nu_act),
            RuntimeBudgetState(),
            bb,
            K=4,
            d=32,
            broyden_max_iter=3,
            tau_flops_budget=1e20,
            parent_z_star=parent,
            noise_seed=99,
            decode_child=False,
        )

    a = run(0, 0.6)
    a_repeat = run(0, 2.0)
    b = run(1, 0.6)
    assert torch.equal(a.z_star_child, a_repeat.z_star_child)
    assert not torch.equal(a.z_star_child, b.z_star_child)
    assert a.solver_stats["noise_seed"] != b.solver_stats["noise_seed"]
    assert a.solver_stats["logical_mac_transition"] == a_repeat.solver_stats[
        "logical_mac_transition"
    ]
    assert a.solver_stats["phi_eval_count"] == 1


def test_transition_rejects_two_rng_owners():
    bb = _IdentityBackbone()
    with pytest.raises(ValueError, match="either noise_seed or generator"):
        transition(
            "problem",
            0,
            NuVector(),
            RuntimeBudgetState(),
            bb,
            K=4,
            d=32,
            generator=torch.Generator().manual_seed(1),
            noise_seed=2,
        )


def test_transition_batch_matches_scalar_branch_contract_on_dense_mock():
    bb = _IdentityBackbone()
    parent = torch.zeros(4, 32)
    context = bb.encode_context("problem")
    nu = NuVector(nu_tol=5e-3, nu_temp=1.0, nu_expl=1.0)
    common = dict(
        K=4,
        d=32,
        broyden_max_iter=3,
        tau_flops_budget=1e20,
        parent_z_star=parent,
        context_override=context,
        noise_seed=123,
        decode_child=False,
    )
    batched = transition_batch(
        "problem", nu, RuntimeBudgetState(), bb, W=3, **common
    )
    serial = [
        transition("problem", w, nu, RuntimeBudgetState(), bb, **common)
        for w in range(3)
    ]
    assert len(batched) == len(serial) == 3
    for rb, rs in zip(batched, serial):
        assert torch.equal(rb.z_star_child, rs.z_star_child)
        assert rb.solver_stats["branch_index"] == rs.solver_stats["branch_index"]
        assert rb.solver_stats["noise_seed"] == rs.solver_stats["noise_seed"]

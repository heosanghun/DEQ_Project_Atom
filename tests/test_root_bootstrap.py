from __future__ import annotations

import torch

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.root_bootstrap import solve_root_z_star


class _PromptWarmStartBackbone(MockTinyBackbone):
    def __init__(self) -> None:
        super().__init__(hidden=16, num_layers=8)
        self.root_fwdpass_calls = 0
        self.deq_step_calls = 0

    def root_fwdpass(
        self,
        context: torch.Tensor,
        *,
        K: int,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        self.root_fwdpass_calls += 1
        return context[:1].expand(K, -1).to(dtype=output_dtype)

    def deq_step(self, z, context, module_weights, extra):
        self.deq_step_calls += 1
        return z


def test_root_bootstrap_is_seed_deterministic_and_prompt_conditioned():
    torch.manual_seed(3)
    bb = MockTinyBackbone(hidden=16, num_layers=8)

    def solve(prompt: str):
        context = bb.encode_context(prompt)
        return solve_root_z_star(
            prompt,
            backbone=bb,
            context=context,
            K=4,
            d=16,
            broyden_max_iter=12,
            seed=99,
            require_converged=False,
        ).z_star_child

    a1 = solve("problem A")
    a2 = solve("problem A")
    b = solve("different problem B")
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, b)


def test_root_nonconvergence_keeps_best_prompt_conditioned_iterate() -> None:
    torch.manual_seed(5)
    bb = MockTinyBackbone(hidden=16, num_layers=8)
    result = solve_root_z_star(
        "root must stay conditioned",
        backbone=bb,
        context=bb.encode_context("root must stay conditioned"),
        K=4,
        d=16,
        broyden_max_iter=1,
        seed=17,
        require_converged=False,
    )
    assert result.solver_stats["converged"] is False
    assert result.solver_stats["failure_fallback"] == "best_finite_iterate"


def test_root_uses_prompt_fwdpass_and_is_seed_invariant_when_available() -> None:
    torch.manual_seed(7)
    bb = _PromptWarmStartBackbone()

    def solve(prompt: str, seed: int):
        return solve_root_z_star(
            prompt,
            backbone=bb,
            context=bb.encode_context(prompt),
            K=4,
            d=16,
            broyden_max_iter=2,
            seed=seed,
            require_converged=True,
        )

    a_seed_1 = solve("same prompt", 1)
    a_seed_2 = solve("same prompt", 999)
    b = solve("different prompt", 1)

    assert torch.equal(a_seed_1.z_star_child, a_seed_2.z_star_child)
    assert not torch.equal(a_seed_1.z_star_child, b.z_star_child)
    assert a_seed_1.solver_stats["root_initialization"] == "prompt_full_stack"
    assert (
        a_seed_1.solver_stats["root_bootstrap_mode"]
        == "algorithm1_direct_fwdpass"
    )
    assert a_seed_1.solver_stats["iterations"] == 0
    assert a_seed_1.solver_stats["phi_eval_count"] == 0
    assert a_seed_1.solver_stats["logical_mac_transition"] == 0.0
    assert bb.deq_step_calls == 0
    assert bb.root_fwdpass_calls == 3

from cts.eval.flops_contract import (
    SOLVER_STATS_KEYS_TRANSITION,
    iso_flop_report_keys,
    public_isoflop_report,
)
from cts.types import NuVector, RuntimeBudgetState


def test_transition_solver_stats_covers_flop_contract():
    from cts.backbone.mock_tiny import MockTinyBackbone
    from cts.deq.transition import transition

    bb = MockTinyBackbone(hidden=64, num_layers=42)
    nu = NuVector(nu_tol=5e-3, nu_temp=1.0, nu_expl=1.0)
    r = transition("x", 0, nu, RuntimeBudgetState(), bb, K=4, d=64, broyden_max_iter=5)
    keys = set(r.solver_stats.keys())
    assert SOLVER_STATS_KEYS_TRANSITION <= keys


def test_public_report_matches_format_isoflop():
    stats = {
        "iterations": 2,
        "logical_mac_phi_once": 3.0,
        "logical_mac_transition": 12.0,
        "executed_mac_phi_once_estimate": 3.0,
        "executed_mac_prefix_setup_estimate": 0.0,
        "executed_mac_transition_estimate": 12.0,
        "phi_eval_count": 4,
        "mac_unit": "MAC",
        "mac_lut_schema_version": 0,
        "mac_lut_sha256": "",
        "mac_accounting_source": "generic_non_headline_proxy",
        "mac_headline_eligible": False,
        "eq3_total_tokens": 8,
        "eq3_prefix_tokens": 4,
        "converged": True,
        "residual_norm": 0.01,
        "absolute_residual_norm": 0.04,
    }
    r1 = public_isoflop_report(stats)
    assert set(r1.keys()) == iso_flop_report_keys()

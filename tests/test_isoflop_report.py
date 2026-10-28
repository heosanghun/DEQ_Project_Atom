import pytest

from cts.eval.isoflop_matcher import format_isoflop_report


def _stats():
    return {
        "iterations": 5,
        "residual_norm": 0.01,
        "absolute_residual_norm": 0.2,
        "converged": True,
        "logical_mac_phi_once": 100.0,
        "logical_mac_transition": 1000.0,
        "executed_mac_phi_once_estimate": 40.0,
        "executed_mac_prefix_setup_estimate": 50.0,
        "executed_mac_transition_estimate": 450.0,
        "phi_eval_count": 10,
        "mac_unit": "MAC",
        "mac_lut_schema_version": 2,
        "mac_lut_sha256": "a" * 64,
        "mac_accounting_source": "gemma4_e4b_analytic_v2",
        "mac_headline_eligible": True,
        "eq3_total_tokens": 256,
        "eq3_prefix_tokens": 192,
        "routing_mac": 48_640.0,
    }


def test_format_isoflop_report_preserves_canonical_mac_v2():
    report = format_isoflop_report(_stats())
    assert report["logical_mac_transition"] == 1000.0
    assert report["executed_mac_transition_estimate"] == 450.0
    assert report["mac_headline_eligible"] is True
    assert report["mac_lut_schema_version"] == 2
    assert report["phi_eval_count"] == 10


def test_format_isoflop_report_fails_closed_without_exact_mac():
    with pytest.raises(KeyError, match="MAC-v2"):
        format_isoflop_report(
            {"iterations": 3, "flops_broyden_estimate": 60.0}
        )


def test_format_rejects_flop_unit():
    stats = _stats()
    stats["mac_unit"] = "FLOP"
    with pytest.raises(ValueError, match="MAC"):
        format_isoflop_report(stats)


def test_format_includes_batch_physical_cost_when_present():
    stats = _stats()
    stats["physical_batch_mac_total_estimate"] = 900.0
    stats["physical_batch_mac_share_estimate"] = 300.0
    report = format_isoflop_report(stats)
    assert report["physical_batch_mac_total_estimate"] == 900.0
    assert report["physical_batch_mac_share_estimate"] == 300.0

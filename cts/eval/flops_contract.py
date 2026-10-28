"""Canonical Iso-MAC field contract shared by evaluation tools.

``flops_*`` fields remain one-release compatibility aliases inside
``transition``.  New code and public reports must use the MAC-v2 names here.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet

from cts.eval.isoflop_matcher import format_isoflop_report


SOLVER_STATS_KEYS_TRANSITION: FrozenSet[str] = frozenset(
    {
        "iterations",
        "residual_norm",
        "absolute_residual_norm",
        "converged",
        "logical_mac_phi_once",
        "logical_mac_transition",
        "executed_mac_phi_once_estimate",
        "executed_mac_prefix_setup_estimate",
        "executed_mac_transition_estimate",
        "phi_eval_count",
        "mac_unit",
        "mac_lut_schema_version",
        "mac_lut_sha256",
        "mac_accounting_source",
        "mac_headline_eligible",
        "eq3_total_tokens",
        "eq3_prefix_tokens",
    }
)

SOLVER_STATS_OPTIONAL_KEYS: FrozenSet[str] = frozenset(
    {
        "act_halt",
        "routing_mac",
        "routing_mac_share",
        "physical_batch_phi_eval_count",
        "physical_batch_mac_total_estimate",
        "physical_batch_mac_share_estimate",
    }
)

SOLVER_STATS_LEGACY_ALIASES: FrozenSet[str] = frozenset(
    {"flops_used", "flops_inner_once", "flops_broyden_estimate"}
)


def iso_flop_report_keys() -> FrozenSet[str]:
    """Keys returned by the exact public Iso-MAC report."""
    sample = format_isoflop_report(
        {
            "iterations": 1,
            "residual_norm": 0.0,
            "absolute_residual_norm": 0.0,
            "converged": True,
            "logical_mac_phi_once": 1.0,
            "logical_mac_transition": 2.0,
            "executed_mac_phi_once_estimate": 1.0,
            "executed_mac_prefix_setup_estimate": 0.0,
            "executed_mac_transition_estimate": 2.0,
            "phi_eval_count": 2,
            "mac_unit": "MAC",
            "mac_lut_schema_version": 0,
            "mac_lut_sha256": "",
            "mac_accounting_source": "generic_non_headline_proxy",
            "mac_headline_eligible": False,
            "eq3_total_tokens": 4,
            "eq3_prefix_tokens": 2,
        }
    )
    return frozenset(sample.keys())


def public_isoflop_report(solver_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Return public Iso-MAC JSON, failing closed on legacy-only stats."""
    return format_isoflop_report(solver_stats)


__all__ = [
    "SOLVER_STATS_KEYS_TRANSITION",
    "SOLVER_STATS_LEGACY_ALIASES",
    "SOLVER_STATS_OPTIONAL_KEYS",
    "iso_flop_report_keys",
    "public_isoflop_report",
]

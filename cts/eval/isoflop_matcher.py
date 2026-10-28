"""Public Iso-MAC reporting for the CTS Eq. (3) operator.

The paper budget is measured in multiply-accumulates (MAC), not FLOPs.  The
canonical values are produced by :mod:`cts.eval.mac_accounting` inside
``transition``.  This module only projects those *recorded* values into a
stable report; it never reconstructs compute from Broyden iteration counts.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from cts.eval.mac_accounting import Eq3MacEstimate, estimate_eq3_mac


_REQUIRED_MAC_V2_KEYS = (
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
)


def estimate_eq3_step_mac(
    backbone: Any,
    selected_module_ids: Sequence[int],
    *,
    total_tokens: int,
    latent_tokens: int,
    suffix_cache_active: bool = False,
    shared_prefix_cache_active: bool = False,
) -> Eq3MacEstimate:
    """Use the same dynamic MAC-v2 implementation as ``transition``."""
    return estimate_eq3_mac(
        backbone,
        selected_module_ids,
        total_tokens=total_tokens,
        latent_tokens=latent_tokens,
        suffix_cache_active=suffix_cache_active,
        shared_prefix_cache_active=shared_prefix_cache_active,
    )


def logical_transition_mac_from_stats(solver_stats: Dict[str, Any]) -> float:
    """Return the exact logical transition MAC or fail closed."""
    if "logical_mac_transition" not in solver_stats:
        raise KeyError("solver_stats missing logical_mac_transition (MAC-v2)")
    value = float(solver_stats["logical_mac_transition"])
    if value < 0.0:
        raise ValueError("logical_mac_transition must be non-negative")
    return value


def format_isoflop_report(solver_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Project exact MAC-v2 stats into the stable public JSON contract."""
    missing = [key for key in _REQUIRED_MAC_V2_KEYS if key not in solver_stats]
    if missing:
        raise KeyError("solver_stats missing MAC-v2 fields: " + ", ".join(missing))
    if solver_stats["mac_unit"] != "MAC":
        raise ValueError("mac_unit must be exactly 'MAC'")

    report = {key: solver_stats[key] for key in _REQUIRED_MAC_V2_KEYS}
    report.update(
        {
            "broyden_iterations": int(solver_stats.get("iterations") or 0),
            "converged": solver_stats.get("converged"),
            "residual_norm": solver_stats.get("residual_norm"),
            "absolute_residual_norm": solver_stats.get(
                "absolute_residual_norm"
            ),
            "routing_mac": solver_stats.get(
                "routing_mac", solver_stats.get("routing_mac_share")
            ),
            "physical_batch_mac_total_estimate": solver_stats.get(
                "physical_batch_mac_total_estimate"
            ),
            "physical_batch_mac_share_estimate": solver_stats.get(
                "physical_batch_mac_share_estimate"
            ),
        }
    )
    return report


__all__ = [
    "estimate_eq3_step_mac",
    "format_isoflop_report",
    "logical_transition_mac_from_stats",
]

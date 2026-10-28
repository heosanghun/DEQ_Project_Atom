"""Versioned MAC accounting for the Gemma-4 E4B CTS operator.

The paper budget is defined on the mathematical Eq.3 operator.  Therefore
``logical_*`` values are invariant to sibling batching, suffix caching and
CUDA graphs.  ``executed_*`` values estimate the optimized implementation and
are reported only for speed attribution; they never drive ACT or tree depth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from cts.model.module_partition import layers_for_module


class MacAccountingError(RuntimeError):
    """The live backbone cannot satisfy the headline MAC-v2 contract."""


@dataclass(frozen=True)
class Eq3MacEstimate:
    logical_phi_mac: float
    executed_phi_mac: float
    executed_prefix_setup_mac: float
    routing_mac: float
    total_tokens: int
    prefix_tokens: int
    latent_tokens: int
    suffix_cache_active: bool
    shared_prefix_cache_active: bool
    schema_version: int
    lut_sha256: str
    headline_eligible: bool
    accounting_source: str

    def logical_transition_mac(self, phi_eval_count: int) -> float:
        return self.routing_mac + self.logical_phi_mac * float(phi_eval_count)

    def executed_transition_mac(self, phi_eval_count: int) -> float:
        return (
            self.routing_mac
            + self.executed_prefix_setup_mac
            + self.executed_phi_mac * float(phi_eval_count)
        )


def _lut_path() -> Path:
    return Path(__file__).resolve().parents[1] / "routing" / "lut_mac.json"


@lru_cache(maxsize=1)
def load_mac_lut_v2() -> tuple[Mapping[str, Any], str]:
    path = _lut_path()
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if int(data.get("schema_version", 0)) != 2:
        raise MacAccountingError("headline CTS requires MAC LUT schema v2")
    if bool(data.get("placeholder", True)):
        raise MacAccountingError("placeholder MAC LUT is forbidden for headline CTS")
    if data.get("unit") != "MAC":
        raise MacAccountingError("MAC LUT unit must be exactly 'MAC'")
    for key in ("geometry", "layer_coefficients", "shared_anchor_kv_mac_per_token"):
        if not isinstance(data.get(key), dict):
            raise MacAccountingError(f"MAC LUT missing mapping: {key}")
    return data, hashlib.sha256(raw).hexdigest()


def _language_model(backbone: Any) -> Any | None:
    hook = getattr(backbone, "_lm", None)
    if callable(hook):
        try:
            return hook()
        except Exception:
            return None
    cg = getattr(backbone, "cg", None)
    model = getattr(cg, "model", None)
    return getattr(model, "language_model", model)


def _validate_gemma4_geometry(config: Any, geometry: Mapping[str, Any]) -> None:
    required = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "global_head_dim",
        "num_kv_shared_layers",
        "sliding_window",
    )
    errors = []
    for key in required:
        live = getattr(config, key, None)
        expected = geometry.get(key)
        if int(live) != int(expected) if live is not None and expected is not None else True:
            errors.append(f"{key}={live!r} (expected {expected!r})")
    if str(getattr(config, "model_type", "")) != "gemma4_text":
        errors.append(f"model_type={getattr(config, 'model_type', None)!r}")
    if errors:
        raise MacAccountingError(
            "live backbone does not match Gemma-4 E4B MAC-v2 geometry: "
            + "; ".join(errors)
        )


def _coefficient_key(*, shared: bool, layer_type: str) -> str:
    if layer_type in {"sliding_attention", "sliding"}:
        suffix = "sliding"
    elif layer_type in {"full_attention", "full"}:
        suffix = "full"
    else:
        raise MacAccountingError(f"unsupported Gemma attention type: {layer_type!r}")
    return f"{'shared' if shared else 'nonshared'}_{suffix}"


def _layer_mac(
    coeff: Mapping[str, Any],
    *,
    query_tokens: int,
    key_tokens: int,
    sliding_window: int | None,
) -> float:
    effective_keys = (
        min(int(key_tokens), int(sliding_window))
        if sliding_window is not None
        else int(key_tokens)
    )
    return float(coeff["linear_mac_per_query_token"]) * float(query_tokens) + float(
        coeff["attention_mac_per_query_key"]
    ) * float(query_tokens) * float(effective_keys)


def _generic_estimate(
    backbone: Any,
    selected_module_ids: Sequence[int],
    *,
    total_tokens: int,
    latent_tokens: int,
) -> Eq3MacEstimate:
    hook = getattr(backbone, "estimate_logical_phi_mac", None)
    if callable(hook):
        value = float(
            hook(
                selected_module_ids=list(selected_module_ids),
                total_tokens=int(total_tokens),
                latent_tokens=int(latent_tokens),
            )
        )
        source = "backbone_hook"
    else:
        # Deterministic non-headline proxy for tiny unit-test backbones.  It is
        # intentionally not the E4B LUT and is marked ineligible for papers.
        value = float(max(1, len(set(selected_module_ids)))) * float(total_tokens)
        source = "generic_non_headline_proxy"
    return Eq3MacEstimate(
        logical_phi_mac=value,
        executed_phi_mac=value,
        executed_prefix_setup_mac=0.0,
        routing_mac=0.0,
        total_tokens=int(total_tokens),
        prefix_tokens=int(total_tokens) - int(latent_tokens),
        latent_tokens=int(latent_tokens),
        suffix_cache_active=False,
        shared_prefix_cache_active=False,
        schema_version=0,
        lut_sha256="",
        headline_eligible=False,
        accounting_source=source,
    )


def estimate_eq3_mac(
    backbone: Any,
    selected_module_ids: Sequence[int],
    *,
    total_tokens: int,
    latent_tokens: int,
    suffix_cache_active: bool,
    shared_prefix_cache_active: bool,
) -> Eq3MacEstimate:
    """Estimate one logical and executed Phi for the fixed routing support."""
    selected = sorted({int(x) for x in selected_module_ids})
    if not selected or any(x < 0 or x >= 19 for x in selected):
        raise MacAccountingError(f"invalid selected module ids: {selected}")
    total_tokens = int(total_tokens)
    latent_tokens = int(latent_tokens)
    prefix_tokens = total_tokens - latent_tokens
    if latent_tokens <= 0 or prefix_tokens < 0:
        raise MacAccountingError(
            f"invalid Eq.3 lengths: total={total_tokens}, K={latent_tokens}"
        )

    lm = _language_model(backbone)
    config = getattr(lm, "config", None)
    if config is None or str(getattr(config, "model_type", "")) != "gemma4_text":
        return _generic_estimate(
            backbone,
            selected,
            total_tokens=total_tokens,
            latent_tokens=latent_tokens,
        )

    lut, lut_sha = load_mac_lut_v2()
    geometry = lut["geometry"]
    _validate_gemma4_geometry(config, geometry)
    coeffs = lut["layer_coefficients"]
    shared_start = int(config.num_hidden_layers) - int(config.num_kv_shared_layers)
    window = int(config.sliding_window)

    logical_phi = 0.0
    executed_phi = 0.0
    prefix_setup = 0.0
    has_shared = False
    for module_id in selected:
        for layer_idx in layers_for_module(module_id):
            layer_type = str(config.layer_types[layer_idx])
            shared = int(layer_idx) >= shared_start
            has_shared = has_shared or shared
            coeff = coeffs[
                _coefficient_key(shared=shared, layer_type=layer_type)
            ]
            sliding = window if "sliding" in layer_type else None
            logical_phi += _layer_mac(
                coeff,
                query_tokens=total_tokens,
                key_tokens=total_tokens,
                sliding_window=sliding,
            )
            if suffix_cache_active:
                executed_phi += _layer_mac(
                    coeff,
                    query_tokens=latent_tokens,
                    key_tokens=total_tokens,
                    sliding_window=sliding,
                )
                if not shared and prefix_tokens:
                    prefix_setup += _layer_mac(
                        coeff,
                        query_tokens=prefix_tokens,
                        key_tokens=prefix_tokens,
                        sliding_window=sliding,
                    )
            else:
                executed_phi += _layer_mac(
                    coeff,
                    query_tokens=total_tokens,
                    key_tokens=total_tokens,
                    sliding_window=sliding,
                )

    if has_shared:
        anchor = lut["shared_anchor_kv_mac_per_token"]
        anchor_per_token = sum(float(x) for x in anchor.values())
        logical_phi += anchor_per_token * float(total_tokens)
        if shared_prefix_cache_active:
            prefix_setup += anchor_per_token * float(prefix_tokens)
            executed_phi += anchor_per_token * float(latent_tokens)
        else:
            executed_phi += anchor_per_token * float(total_tokens)

    routing_mac = float(config.hidden_size) * float(geometry["modules"])
    return Eq3MacEstimate(
        logical_phi_mac=logical_phi,
        executed_phi_mac=executed_phi,
        executed_prefix_setup_mac=prefix_setup,
        routing_mac=routing_mac,
        total_tokens=total_tokens,
        prefix_tokens=prefix_tokens,
        latent_tokens=latent_tokens,
        suffix_cache_active=bool(suffix_cache_active),
        shared_prefix_cache_active=bool(shared_prefix_cache_active),
        schema_version=2,
        lut_sha256=lut_sha,
        headline_eligible=True,
        accounting_source="gemma4_e4b_analytic_v2",
    )


def linear_layer_mac_per_item(module: Any) -> float:
    """Sum Linear ``in_features × out_features`` MAC for one item.

    Nonlinearities, bias additions and normalization follow the LUT convention
    and are intentionally excluded.  Nested LoRA A/B Linear modules are counted
    automatically when present.
    """
    iterator = getattr(module, "modules", None)
    if not callable(iterator):
        raise MacAccountingError("module does not expose modules() for MAC count")
    total = 0
    seen: set[int] = set()
    for child in iterator():
        if id(child) in seen:
            continue
        seen.add(id(child))
        in_features = getattr(child, "in_features", None)
        out_features = getattr(child, "out_features", None)
        if in_features is None or out_features is None:
            continue
        total += int(in_features) * int(out_features)
    if total <= 0:
        raise MacAccountingError(
            f"no Linear geometry found in {type(module).__name__}"
        )
    return float(total)


__all__ = [
    "Eq3MacEstimate",
    "MacAccountingError",
    "estimate_eq3_mac",
    "linear_layer_mac_per_item",
    "load_mac_lut_v2",
]

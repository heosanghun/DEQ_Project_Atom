from __future__ import annotations

from types import SimpleNamespace

from cts.critic.neuro_critic import NeuroCritic
from cts.eval.mac_accounting import (
    estimate_eq3_mac,
    linear_layer_mac_per_item,
    load_mac_lut_v2,
)
from cts.policy.meta_policy import MetaPolicy


def _gemma_config():
    return SimpleNamespace(
        model_type="gemma4_text",
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=42,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        num_kv_shared_layers=18,
        sliding_window=512,
        layer_types=[
            "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
            for i in range(42)
        ],
    )


class _GemmaBackbone:
    def __init__(self):
        self.lm = SimpleNamespace(config=_gemma_config())

    def _lm(self):
        return self.lm


def _estimate(modules, *, total=256, K=64, suffix=False):
    return estimate_eq3_mac(
        _GemmaBackbone(),
        modules,
        total_tokens=total,
        latent_tokens=K,
        suffix_cache_active=suffix,
        shared_prefix_cache_active=True,
    )


def test_v2_lut_is_non_placeholder_and_mac_typed():
    lut, digest = load_mac_lut_v2()
    assert lut["schema_version"] == 2
    assert lut["placeholder"] is False
    assert lut["unit"] == "MAC"
    assert len(digest) == 64


def test_nonshared_sliding_module_coefficient_is_exact():
    estimate = _estimate([0], total=64, K=64)
    per_layer = 93_159_424 * 64 + 4_096 * 64 * 64
    assert estimate.logical_phi_mac == float(2 * per_layer)
    assert estimate.headline_eligible


def test_topk_modules_are_summed_not_routing_weighted():
    combined = _estimate([0, 1, 2]).logical_phi_mac
    separate = sum(_estimate([m]).logical_phi_mac for m in (0, 1, 2))
    assert combined == separate


def test_shared_anchor_is_charged_once_per_phi():
    estimate = _estimate([12], total=256, K=64)
    # Module 12 = shared sliding layers 24 and 25, plus both anchor K/V
    # projections exactly once for any shared consumer set.
    layer = 90_513_408 * 256 + 4_096 * 256 * 256
    anchors = (2_646_016 + 5_271_552) * 256
    assert estimate.logical_phi_mac == float(2 * layer + anchors)


def test_suffix_cache_changes_executed_not_logical_mac():
    uncached = _estimate([0, 11, 12], suffix=False)
    cached = _estimate([0, 11, 12], suffix=True)
    assert cached.logical_phi_mac == uncached.logical_phi_mac
    assert cached.executed_phi_mac < uncached.executed_phi_mac
    assert cached.executed_prefix_setup_mac > 0
    assert (
        cached.executed_transition_mac(10)
        < uncached.executed_transition_mac(10)
    )


def test_dynamic_sequence_length_increases_mac():
    short = _estimate([0, 11, 12], total=128)
    long = _estimate([0, 11, 12], total=512)
    assert long.logical_phi_mac > short.logical_phi_mac


def test_actual_meta_and_critic_linear_geometry():
    meta = MetaPolicy(text_dim=2560, hidden=256, W=3)
    critic = NeuroCritic(z_dim=2560)
    assert linear_layer_mac_per_item(meta) == 657_152.0
    assert linear_layer_mac_per_item(critic) == 6_556_160.0


def test_non_gemma_backbone_never_uses_e4b_lut():
    class Tiny:
        pass

    estimate = estimate_eq3_mac(
        Tiny(),
        [0, 1, 2],
        total_tokens=16,
        latent_tokens=4,
        suffix_cache_active=False,
        shared_prefix_cache_active=False,
    )
    assert not estimate.headline_eligible
    assert estimate.schema_version == 0
    assert estimate.accounting_source == "generic_non_headline_proxy"

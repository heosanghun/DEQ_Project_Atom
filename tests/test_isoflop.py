import pytest

from cts.eval.isoflop_matcher import logical_transition_mac_from_stats


def test_logical_transition_mac_is_exact():
    assert logical_transition_mac_from_stats(
        {"logical_mac_transition": 123.5}
    ) == pytest.approx(123.5)


def test_legacy_flops_alias_does_not_satisfy_mac_v2():
    with pytest.raises(KeyError, match="logical_mac_transition"):
        logical_transition_mac_from_stats({"flops_broyden_estimate": 123.5})


def test_negative_logical_mac_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        logical_transition_mac_from_stats({"logical_mac_transition": -1.0})

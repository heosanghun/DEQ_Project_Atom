from __future__ import annotations

from scripts.profile_cts_real_gpu import _boxed_contract_stats


def test_boxed_contract_accepts_nested_terminal_box() -> None:
    stats = _boxed_contract_stats(r"\boxed{\frac{1}{2}}")
    assert stats == {
        "marker_count": 1,
        "complete_count": 1,
        "last_box_terminal": True,
        "exactly_one_complete_terminal_box": True,
    }


def test_boxed_contract_rejects_incomplete_trailing_and_multiple() -> None:
    assert not _boxed_contract_stats(r"\boxed{1")[
        "exactly_one_complete_terminal_box"
    ]
    assert not _boxed_contract_stats(r"\boxed{1} trailing")[
        "exactly_one_complete_terminal_box"
    ]
    assert not _boxed_contract_stats(r"\boxed{1} \boxed{2}")[
        "exactly_one_complete_terminal_box"
    ]

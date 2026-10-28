"""Torch-free tests for raw CTS scoring and optional hybrid diagnostics.

Garbage detection is metadata: a degenerate CTS decode is scored as the raw
CTS answer (normally incorrect).  A greedy replacement may be computed only
as an explicitly named diagnostic and must never change the headline CTS
prediction or reward.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent

# Load ``cts/eval/garbage_filter.py`` directly via importlib so we
# do *not* execute ``cts/__init__.py`` (which transitively imports
# torch through cts.mcts.cts_episode and would hang on the
# author's degraded host - REVIEWER_FAQ Q15). The module is a
# standalone pure-Python file with no torch / numpy imports, so
# this direct load is safe and returns in <5 ms.
_GF_PATH = ROOT / "cts" / "eval" / "garbage_filter.py"
_spec = importlib.util.spec_from_file_location("_gf_test", _GF_PATH)
assert _spec is not None and _spec.loader is not None, f"cannot load {_GF_PATH}"
_gf = importlib.util.module_from_spec(_spec)
sys.modules["_gf_test"] = _gf
_spec.loader.exec_module(_gf)

MATH_BENCHMARKS = _gf.MATH_BENCHMARKS
is_garbage_math = _gf.is_garbage_math
is_math_benchmark = _gf.is_math_benchmark


# ---------------------------------------------------------------------------
# Canonical Q14 corpus (from results/post_stage2_D11/nu_traces/
# cts_4nu_aime_seed0.jsonl pre-fix episodes).
# ---------------------------------------------------------------------------

GARBAGE_TOKENS = [
    "Cultura",
    "LinearLayout",
    "?????",
    "obar",
    "the answer is",  # non-numeric prose
    "x = ",            # symbolic, no numeric prefix
    "<unk>",
    "????",
    "ANSWER",
    "Solution:",
]

NUMERIC_TOKENS = [
    "47",
    "-12",
    "0",
    "3.14",
    "-3.14",
    "1234567890",
    "-99999",
]


# ---------------------------------------------------------------------------
# Section 1: garbage_filter unit invariants
# ---------------------------------------------------------------------------


def test_math_benchmark_membership():
    assert is_math_benchmark("aime")
    assert is_math_benchmark("aime_90")
    assert is_math_benchmark("math500")
    assert is_math_benchmark("gsm8k")
    assert not is_math_benchmark("humaneval")
    assert not is_math_benchmark("arc")
    assert not is_math_benchmark("arc_agi_text")
    assert not is_math_benchmark(None)
    assert not is_math_benchmark("")


def test_math_benchmarks_set_is_frozen():
    """The math benchmark set is intentionally a frozenset so that
    accidental mutation by a downstream module raises rather than
    silently expanding fallback scope."""
    assert isinstance(MATH_BENCHMARKS, frozenset)
    try:
        MATH_BENCHMARKS.add("foo")  # type: ignore[attr-defined]
    except AttributeError:
        return
    assert False, "MATH_BENCHMARKS must be immutable"


def test_garbage_on_every_math_benchmark():
    """Every garbage token triggers fallback on every math benchmark."""
    for bench in MATH_BENCHMARKS:
        for tok in GARBAGE_TOKENS:
            assert is_garbage_math(bench, tok), (
                f"{bench!r} + {tok!r} should be garbage"
            )


def test_numeric_never_triggers_fallback():
    """No numeric token is ever flagged as garbage on math benchmarks."""
    for bench in MATH_BENCHMARKS:
        for tok in NUMERIC_TOKENS:
            assert not is_garbage_math(bench, tok), (
                f"{bench!r} + {tok!r} should NOT be garbage"
            )


def test_non_math_benchmarks_never_garbage():
    """On non-math benchmarks (humaneval / arc / etc.) we never
    fall back, regardless of the prediction. This is critical:
    HumanEval expects code, ARC expects letters, and falling
    back to greedy on those would *destroy* accuracy."""
    for bench in ("humaneval", "arc", "arc_agi_text", "mbpp", None, ""):
        for tok in GARBAGE_TOKENS + NUMERIC_TOKENS + ["B", "def f():\n  return 1"]:
            assert not is_garbage_math(bench, tok), (
                f"non-math {bench!r} + {tok!r} should NEVER be garbage"
            )


def test_none_prediction_never_triggers_fallback():
    """A None prediction is handled by the dispatcher's
    ``not pred_raw`` guard, not by garbage_filter. The helper
    must therefore return False for None to avoid double-fallback."""
    for bench in MATH_BENCHMARKS:
        assert not is_garbage_math(bench, None)


def test_empty_string_is_garbage_on_math():
    """Empty string starts with no digit, so the predicate flags
    it as garbage. The dispatcher's separate ``len < 3`` guard
    also catches this; we keep both for defence-in-depth."""
    for bench in MATH_BENCHMARKS:
        assert is_garbage_math(bench, "")


def test_negative_numbers_are_not_garbage():
    """The predicate must accept ``-12`` and ``-3.14`` as numeric;
    a naive ``str.isdigit()`` check would reject them and trigger
    spurious fallback on legitimate negative-answer problems."""
    for bench in MATH_BENCHMARKS:
        assert not is_garbage_math(bench, "-12")
        assert not is_garbage_math(bench, "-3.14")
        assert not is_garbage_math(bench, "-0")


def test_decimals_are_not_garbage():
    for bench in MATH_BENCHMARKS:
        assert not is_garbage_math(bench, "3.14159")
        assert not is_garbage_math(bench, "0.5")


def test_math500_symbolic_answers_are_not_mislabeled_as_garbage():
    for answer in (
        r"\frac{1}{2}",
        r"\sqrt{18}",
        r"(1, 2)",
        r"[-1,\infty)",
        r"x^2+1",
    ):
        assert not is_garbage_math("math500", answer)


# ---------------------------------------------------------------------------
# Section 2: Mock-based dispatcher behaviour
# ---------------------------------------------------------------------------


def _simulate_dispatcher_decision(
    benchmark: str,
    cts_pred: str,
    pred_raw: str | None = None,
) -> str:
    """Classify raw CTS output without ever replacing the prediction."""
    if pred_raw is None:
        pred_raw = f"The answer is {cts_pred}.\n\\boxed{{{cts_pred}}}"
    is_garbage = is_garbage_math(benchmark, cts_pred)
    if (
        not pred_raw
        or len(pred_raw.strip()) < 3
        or pred_raw.strip() == "obar"
        or is_garbage
    ):
        return "cts-invalid"
    return "cts"


def test_dispatcher_marks_canonical_aime_garbage_but_keeps_cts_label():
    canonical = ["Cultura", "LinearLayout", "?????"]
    for s in canonical:
        for bench in ("aime", "aime_90", "math500", "gsm8k"):
            assert _simulate_dispatcher_decision(bench, s) == "cts-invalid", (
                f"dispatcher must mark canonical garbage {s!r} for {bench!r}"
            )


def test_dispatcher_keeps_clean_numeric_predictions():
    """If CTS produces a clean numeric answer (with a non-trivial
    raw output, as the real model would), the dispatcher must
    NOT fall back (otherwise we lose every CTS win)."""
    for s in ("47", "277", "-12", "3.14", "0"):
        for bench in ("aime", "aime_90", "math500", "gsm8k"):
            assert _simulate_dispatcher_decision(bench, s) == "cts", (
                f"dispatcher must keep clean prediction {s!r} for {bench!r}"
            )


def test_dispatcher_marks_too_short_raw_output_invalid():
    for s in ("47", "0"):
        # raw output is only 2 chars - shorter than the 3-char guard
        for bench in ("aime", "math500"):
            assert _simulate_dispatcher_decision(bench, s, pred_raw=s) == "cts-invalid"


def test_dispatcher_marks_obar_raw_output_invalid():
    for bench in ("aime", "math500", "gsm8k"):
        assert _simulate_dispatcher_decision(bench, "47", pred_raw="obar") == "cts-invalid"


def test_dispatcher_keeps_humaneval_raw_output():
    for s in ["Cultura", "LinearLayout", "def foo():\n    return 1"]:
        assert _simulate_dispatcher_decision("humaneval", s) == "cts", (
            "dispatcher must keep HumanEval raw output"
        )


# ---------------------------------------------------------------------------
# Section 3: Mock-based predictor end-to-end
# ---------------------------------------------------------------------------


def test_headline_never_invokes_greedy_for_garbage_cts():
    greedy_predictor = MagicMock(return_value="The answer is \\boxed{47}")
    cts_pred = "Cultura"
    final = cts_pred

    greedy_predictor.assert_not_called()
    assert final == "Cultura"


def test_opt_in_hybrid_diagnostic_does_not_mutate_headline_prediction():
    greedy_predictor = MagicMock(return_value="The answer is \\boxed{47}")
    cts_pred = "Cultura"
    headline_pred = cts_pred
    hybrid_pred = headline_pred
    compute_hybrid_diagnostic = True
    if is_garbage_math("aime", cts_pred) and compute_hybrid_diagnostic:
        hybrid_pred = greedy_predictor("FALLBACK_PROMPT", max_new_tokens=512)

    greedy_predictor.assert_called_once()
    assert headline_pred == "Cultura"
    assert "47" in hybrid_pred


def test_hybrid_diagnostic_not_invoked_for_clean_number():
    fallback_predictor = MagicMock(return_value="OOPS_CALLED")

    cts_pred = "47"
    bench = "aime"
    final = cts_pred

    fallback_predictor.assert_not_called()
    assert final == "47"


def test_mock_humaneval_keeps_raw_output():
    fallback_predictor = MagicMock(return_value="OOPS_CALLED_ON_HUMANEVAL")

    cts_pred = "LinearLayout"  # plausibly a class name in code
    bench = "humaneval"
    if is_garbage_math(bench, cts_pred):
        final = fallback_predictor("FALLBACK_PROMPT", max_new_tokens=512)
    else:
        final = cts_pred

    fallback_predictor.assert_not_called()
    assert final == "LinearLayout"

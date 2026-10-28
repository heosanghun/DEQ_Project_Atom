"""Garbage-prediction diagnostics for math benchmarks.

This module factors out the predicate that the dispatcher in
``scripts/run_cts_eval_full.py`` uses to decide whether a CTS / DEQ
prediction on a math benchmark looks like the AIME garbage symptom
(``Cultura``, ``LinearLayout``, ``?????``).  This flag is diagnostic;
headline CTS always scores its raw prediction.

It lives here (vs. inline in ``run_cts_eval_full.py``) for two
reasons:

1. The predicate is now unit-testable in <10 ms with no torch
   dependency (see ``tests/test_dispatcher_fallback_mock.py``).
2. The dispatcher can import the helper instead of duplicating
   the regex / benchmark-membership logic in two separate places
   (CTS path + DEQ-only path).

This is a *pure function*: no side effects, no torch.
"""

from __future__ import annotations

import re

# Benchmarks routed through the math diagnostic.  MATH-500 is deliberately
# not in ``NUMERIC_ONLY_BENCHMARKS``: its valid answers include fractions,
# radicals, tuples, intervals, and symbolic expressions.
MATH_BENCHMARKS: frozenset[str] = frozenset({
    "math500",
    "gsm8k",
    "aime",
    "aime_90",
})

NUMERIC_ONLY_BENCHMARKS: frozenset[str] = frozenset({
    "gsm8k",
    "aime",
    "aime_90",
})

_NUMERIC_PREFIX = re.compile(r"^-?\d")
_KNOWN_GARBAGE = frozenset({
    "cultura",
    "linearlayout",
    "?????",
    "????",
    "obar",
    "<unk>",
    "answer",
    "solution:",
    "the answer is",
    "x =",
})


def is_math_benchmark(benchmark: str | None) -> bool:
    """Whether the given benchmark is handled by the math evaluator."""
    return benchmark in MATH_BENCHMARKS


def is_garbage_math(benchmark: str | None, prediction: str | None) -> bool:
    """Whether ``prediction`` looks like a known degenerate math decode.

    Returns True iff:

    - the benchmark is a supported math benchmark, AND
    - the prediction is not None, AND
    - the normalized prediction is a known degenerate token, or the benchmark
      is numeric-only and the prediction has no numeric prefix.

    None and empty-string predictions on math benchmarks are
    *also* considered garbage, so the dispatcher's existing
    ``not pred_raw or len(pred_raw.strip()) < 3`` check is
    redundant with this helper for those cases. We keep both
    checks in the dispatcher for defence-in-depth.

    Examples (math benchmarks):

        >>> is_garbage_math("aime", "Cultura")
        True
        >>> is_garbage_math("gsm8k", "LinearLayout")
        True
        >>> is_garbage_math("math500", "?????")
        True
        >>> is_garbage_math("aime", "47")
        False
        >>> is_garbage_math("math500", "-12.5")
        False
        >>> is_garbage_math("aime_90", "")
        True
        >>> is_garbage_math("aime", None)
        False

    Examples (non-math benchmarks - never garbage):

        >>> is_garbage_math("humaneval", "Cultura")
        False
        >>> is_garbage_math("arc", "B")
        False
        >>> is_garbage_math(None, "Cultura")
        False
    """
    if not is_math_benchmark(benchmark):
        return False
    if prediction is None:
        return False
    normalized = prediction.strip().lower()
    if normalized in _KNOWN_GARBAGE or not normalized:
        return True
    if benchmark in NUMERIC_ONLY_BENCHMARKS:
        return _NUMERIC_PREFIX.match(normalized) is None
    return False

"""
tests/test_lenient_extract.py - Unit tests for lenient_extract.py.

Specification (Directive EXE-260912-02 §2):
3 unit tests (CPU, synthetic strings):
1. \boxed present
2. No \boxed, '= 2 euros' -> '2'
3. Fraction \frac{20}{3}
"""

import pytest
from cts.eval.lenient_extract import lenient_extract


def test_boxed_present():
    # Case 1: oxed present
    text1 = "Therefore, the final answer is \\boxed{42}."
    assert lenient_extract(text1) == "42"

    text2 = "We conclude that \\boxed{\\frac{1}{2}} is the root."
    assert lenient_extract(text2) == "\\frac{1}{2}"


def test_no_boxed_euros():
    # Case 2: No oxed, "= 2 euros"
    text1 = "The calculated ticket price is = 2 euros."
    assert lenient_extract(text1) == "2"

    text2 = "Total cost = 2 euros"
    assert lenient_extract(text2) == "2"


def test_fraction_no_boxed():
    # Case 3: Fraction rac{20}{3} without oxed
    text1 = "Hence the simplified fraction is \\frac{20}{3}."
    assert lenient_extract(text1) == "\\frac{20}{3}"

    text2 = "The required ratio is $\\frac{20}{3}$."
    assert lenient_extract(text2) == "\\frac{20}{3}"

r"""
tests/test_grader_v2.py - 20 P0 Unit Tests for s5_eval_grader_v2 (TASK-0002 Part b).

Covers all Defect 9 edge cases and strict/lenient contracts:
1. "2\sqrt{3}" vs "2" (mismatch - first-number fallback eliminated)
2. "Let me compute: 1." (no answer extracted - mid-reasoning fallback eliminated)
3. "3.5." vs "3.5" (trailing period handled)
4. "1,250" vs "1250" (comma thousands separator handled)
5. "also" in prose with answer (word boundary test)
6. "also" without answer (word boundary test - no false trigger)
7. "So the answer is 42." (prose match: strict=False, lenient=True)
8. "(A)" vs "a" (choice letter and case normalization)
9. "\frac{1}{2}" vs "0.5" (math-verify symbolic equivalence)
10. "3\sqrt{2}" vs "\sqrt{18}" (math-verify radical simplification)
11. "\frac{2}{4}" vs "\frac{1}{2}" (unreduced fraction equivalence)
12. "\boxed{42}" vs "43" (strict mismatch)
13. "= 2 euros" (lenient unit equation: strict=False, lenient=True)
14. "\frac{20}{3}" in text (lenient fraction: strict=False, lenient=True)
15. "$x = 17$" in text (lenient inline math: strict=False, lenient=True)
16. "\boxed{\frac{1}{\sqrt{2}}}" vs "\frac{\sqrt{2}}{2}" (nested braces + symbolic)
17. "\boxed{99" (unclosed boxed recovery)
18. "2.00" vs "2" (float zero padding equivalence)
19. "Step 1: 5 is odd..." (pure reasoning text, no answer extracted)
20. "\boxed{042}" vs "42" (AIME-style padded integer)
"""

import sys
from pathlib import Path

# Ensure project root and scripts/ directory are on path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from cts.eval.grader_v2 import grade_prediction, answers_match, extract_answer, lenient_extract
except ImportError:
    from s5_eval_grader_v2 import grade_prediction, answers_match, extract_answer, lenient_extract


TEST_CASES = [
    # (id, name, pred, gold, exp_strict, exp_lenient)
    (1, "defect9_sqrt_prefix", "\\boxed{2\\sqrt{3}}", "2", False, False),
    (2, "defect9_mid_reasoning_let_me_compute", "Let me compute: 1. First step...", "1", False, False),
    (3, "defect9_trailing_period", "\\boxed{3.5.}", "3.5", True, True),
    (4, "defect9_comma_separator", "\\boxed{1,250}", "1250", True, True),
    (5, "defect9_also_word_boundary_answer", "We also find the solution is \\boxed{7}.", "7", True, True),
    (6, "defect9_also_word_boundary_no_answer", "We also have 5 apples in the basket.", "5", False, False),
    (7, "prose_so_the_answer_is", "So the answer is 42.", "42", False, True),
    (8, "choice_letter_normalization", "\\boxed{(A)}", "a", True, True),
    (9, "symbolic_fraction_to_decimal", "\\boxed{\\frac{1}{2}}", "0.5", True, True),
    (10, "symbolic_radical_simplification", "\\boxed{3\\sqrt{2}}", "\\sqrt{18}", True, True),
    (11, "symbolic_unreduced_fraction", "\\boxed{\\frac{2}{4}}", "\\frac{1}{2}", True, True),
    (12, "strict_direct_mismatch", "\\boxed{42}", "43", False, False),
    (13, "lenient_euros_unit_equation", "The cost of the ticket is = 2 euros.", "2", False, True),
    (14, "lenient_fraction_in_text", "The calculated ratio is \\frac{20}{3}.", "\\frac{20}{3}", False, True),
    (15, "lenient_inline_math_equation", "Solving the equation yields $x = 17$.", "17", False, True),
    (16, "nested_braces_and_symbolic", "\\boxed{\\frac{1}{\\sqrt{2}}}", "\\frac{\\sqrt{2}}{2}", True, True),
    (17, "unclosed_boxed_recovery", "Thus we get \\boxed{99", "99", True, True),
    (18, "float_zero_padding", "\\boxed{2.00}", "2", True, True),
    (19, "reasoning_only_no_answer", "Step 1: 5 is odd. Step 2: 7 is prime. Conclusion.", "5", False, False),
    (20, "padded_integer_aime", "\\boxed{042}", "42", True, True),
]


def run_all_tests():
    print("=" * 80)
    print("TASK-0002 Part (b): Grader Unit Tests (20 Test Cases)")
    print("=" * 80)
    print(f"{'#':>2} | {'Test Name':<35} | {'Strict':<12} | {'Lenient':<12} | {'Result':<6}")
    print("-" * 80)

    passed_count = 0
    total_count = len(TEST_CASES)

    for case_id, name, pred, gold, exp_strict, exp_lenient in TEST_CASES:
        strict_match, lenient_match, b_ext, l_ext = grade_prediction(pred, gold)

        strict_ok = (strict_match == exp_strict)
        lenient_ok = (lenient_match == exp_lenient)
        case_passed = strict_ok and lenient_ok

        if case_passed:
            passed_count += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        strict_str = f"{strict_match} (exp {exp_strict})"
        lenient_str = f"{lenient_match} (exp {exp_lenient})"
        print(f"{case_id:>2} | {name:<35} | {strict_str:<12} | {lenient_str:<12} | {verdict:<6}")

        if not case_passed:
            print(f"     DETAILS: pred={pred!r}, gold={gold!r}")
            print(f"     boxed_ext={b_ext!r}, lenient_ext={l_ext!r}")
            print(f"     strict_ok={strict_ok}, lenient_ok={lenient_ok}")

    print("-" * 80)
    print(f"Total: {passed_count}/{total_count} test cases passed ({passed_count/total_count*100:.1f}%)")
    print("=" * 80)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

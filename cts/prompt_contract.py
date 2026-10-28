"""Single-source prompt contracts used by training and headline evaluation."""

from __future__ import annotations


MATH_CONCISE_BOXED_CONTRACT = "math_concise_boxed_v1"
MATH_CONCISE_BOXED_PREFIX = (
    "Solve the problem. Give only the essential reasoning in at most four "
    "short sentences. Then finish with exactly one line "
    "\\boxed{FINAL_ANSWER}. Write nothing after the box.\n\n"
)


def build_math_decode_prompt(question: str) -> str:
    """Return user content only; the model adapter owns chat templating."""
    return MATH_CONCISE_BOXED_PREFIX + str(question)


__all__ = [
    "MATH_CONCISE_BOXED_CONTRACT",
    "MATH_CONCISE_BOXED_PREFIX",
    "build_math_decode_prompt",
]

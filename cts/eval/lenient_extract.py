"""
cts/eval/lenient_extract.py - Lenient mathematical answer extraction module.

Specification (Directive EXE-260912-02 §2 / Ledger #196):
- Priority 1: Extract content of the final \boxed{...} (handling balanced braces).
- Priority 2: If no \boxed{...} is present, extract the final mathematical expression or numeric value.
- Handles edge cases: LaTeX fractions (\frac{num}{den}), prose units ("= 2 euros" -> "2"), etc.
"""

from __future__ import annotations

import re
from typing import Optional


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content of the last \boxed{...} handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        idx = text.rfind("\boxed{")
    if idx == -1:
        return None

    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1].strip()
    return text[start:].strip()


def lenient_extract(text: str) -> str:
    """Extract final answer from model-generated text leniently.

    1. If \boxed{...} exists, return the content of the last \boxed{...}.
    2. Otherwise, extract the final mathematical expression (e.g. \frac{20}{3}, $...$)
       or numeric value (e.g. '= 2 euros' -> '2').
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.strip()

    # Priority 1: Last \boxed{...}
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    # Priority 2: Final LaTeX fraction \frac{...}{...}
    fracs = list(re.finditer(r"\\frac\{[^{}]+\}\{[^{}]+\}", text))
    if fracs:
        return fracs[-1].group(0).strip()

    # Priority 3: Math formula in $...$
    math_matches = list(re.finditer(r"\$([^$]+)\$", text))
    if math_matches:
        val = math_matches[-1].group(1).strip()
        if "=" in val:
            val = val.split("=")[-1].strip()
        if val:
            return val

    # Priority 4: Equation pattern like '= 2 euros' or '= 42'
    eq_match = list(re.finditer(r"=\s*([-+]?\d+(?:\.\d+)?)", text))
    if eq_match:
        return eq_match[-1].group(1).strip()

    # Priority 5: Fallback to last number in text
    num_matches = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", text))
    if num_matches:
        return num_matches[-1].group(0).strip()

    return ""


__all__ = ["lenient_extract", "_extract_boxed"]

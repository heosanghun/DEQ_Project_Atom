r"""
scripts/s5_eval_grader_v2.py - Upgraded Mathematical Answer Grader for S5/S7.

Key Improvements (Ledger #381 / TASK-0002 Part b / Defect 9 Fixes):
1. Integrated math-verify (symbolic equivalence via sympy).
2. Completely removed first-number substring fallback (_legacy_answers_match).
   Prevents "2\sqrt{3}" from matching "2", etc.
3. Added word boundaries (\b) to prose patterns (prevents "also" from matching "so").
4. Fixed \boxed{ raw string regex and robust nested brace matching.
5. Handles trailing period ("3.5." -> "3.5"), commas ("1,250" -> "1250"), case normalization.
6. Eliminates arbitrary mid-reasoning numeric fallbacks ("Let me compute: 1." -> no match).
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

_logger = logging.getLogger(__name__)

# Try to import math-verify
_MATH_VERIFY_AVAILABLE = False
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    try:
        import sys
        if "/home/shoon/venv_eval/lib/python3.12/site-packages" not in sys.path:
            sys.path.insert(0, "/home/shoon/venv_eval/lib/python3.12/site-packages")
        from math_verify import parse as _mv_parse, verify as _mv_verify
        _MATH_VERIFY_AVAILABLE = True
    except Exception:
        _MATH_VERIFY_AVAILABLE = False

# Word-bounded prose answer markers to prevent false positives like "also" matching "so"
_PROSE_PATTERNS = [
    r"\b(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)",
    r"\b(?:answer|result)\s*[=:]\s*(.+?)(?:\.|$)",
    r"\b(?:therefore|thus|so|hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(.+?)(?:\.|$)",
]


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content of the last \\boxed{...} handling nested braces."""
    if not text or not isinstance(text, str):
        return None
    idx = text.rfind(r"\boxed{")
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
    res = text[start:i - 1].strip() if depth == 0 else text[start:].strip()
    return res if res else None


def normalize_answer(s: str) -> str:
    """Normalize mathematical answer strings."""
    if not s or not isinstance(s, str):
        return ""
    s = s.strip()
    # Strip optional surrounding quotes or parentheses (e.g. choice letter "(A)")
    m_choice = re.fullmatch(r"\(([a-zA-Z0-9]+)\)", s)
    if m_choice:
        s = m_choice.group(1)

    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left\s*", "", s)
    s = re.sub(r"\\right\s*", "", s)
    s = s.replace("\\,", "")
    s = s.replace("^\\circ", "")
    s = s.replace("^{\\circ}", "")

    if r"\boxed{" in s:
        boxed = _extract_boxed(s)
        if boxed:
            s = boxed
        else:
            s = s.replace(r"\boxed{", "").rstrip("}")

    s = re.sub(r"[$%]", "", s)
    s = s.replace(",", "")
    s = s.rstrip(".")  # strip trailing period from sentence end e.g. "3.5." -> "3.5"
    return s


def _as_plain_int(s: str) -> Optional[int]:
    """Return int value if string represents a pure integer."""
    t = str(s).strip().strip("$").strip()
    m = re.fullmatch(r"[-+]?\d+", t)
    return int(m.group()) if m else None


def _as_plain_float(s: str) -> Optional[float]:
    """Return float value ONLY if the entire string represents a single valid float."""
    t = str(s).strip().strip("$").replace(",", "").rstrip(".").strip()
    m = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _mv_wrap(s: str) -> str:
    """Wrap a bare answer in $...$ for math-verify parsing."""
    t = str(s).strip()
    if not t or t.startswith("$") or r"\boxed" in t:
        return t
    return f"${t}$"


def answers_match(pred: str, gold: str) -> bool:
    """Check mathematical equivalence between predicted and ground-truth answers.

    Guarantees:
    - Zero first-number substring fallback.
    - math-verify symbolic equivalence.
    - Normalized string equality.
    - Strict full-string numeric comparison.
    """
    if pred is None or gold is None:
        return False
    pred_str = str(pred).strip()
    gold_str = str(gold).strip()
    if not pred_str or not gold_str:
        return False

    # 1. Plain integer comparison (handles leading zeros like "042" == "42")
    _pi, _gi = _as_plain_int(pred_str), _as_plain_int(gold_str)
    if _pi is not None and _gi is not None:
        return _pi == _gi

    # 2. Plain float comparison ONLY when both are full standalone numbers
    _pf, _gf = _as_plain_float(pred_str), _as_plain_float(gold_str)
    if _pf is not None and _gf is not None:
        return abs(_pf - _gf) < 1e-6

    # 3. Normalized string match
    pn = normalize_answer(pred_str)
    gn = normalize_answer(gold_str)
    if pn and gn and pn == gn:
        return True

    # 4. math-verify symbolic equivalence
    if _MATH_VERIFY_AVAILABLE:
        try:
            p_parsed = _mv_parse(_mv_wrap(pred_str))
            g_parsed = _mv_parse(_mv_wrap(gold_str))
            if p_parsed and g_parsed and bool(_mv_verify(g_parsed, p_parsed)):
                return True
        except Exception:
            pass

    return False


def extract_answer(text: str) -> str:
    """Extract answer following strict explicit markers only (no arbitrary fallback)."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()

    # Priority 1: GSM8K hash format
    gsm_match = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
    if gsm_match:
        return gsm_match.group(1).strip()

    # Priority 2: \boxed{...}
    boxed = _extract_boxed(text)
    if boxed:
        return boxed.strip()

    # Priority 3: Word-bounded prose patterns
    for pat in _PROSE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip().rstrip(".")
            if ans:
                return ans

    # Strict: NO fallback to mid-reasoning text or numbers
    return ""


def lenient_extract(text: str) -> str:
    """Extract answer leniently without mid-reasoning false positive fallback."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()

    # Priority 1: \boxed{...}
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    # Priority 2: Final LaTeX fraction \frac{...}{...}
    fracs = list(re.finditer(r"\\frac\{[^{}]+\}\{[^{}]+\}", text))
    if fracs:
        return fracs[-1].group(0).strip()

    # Priority 3: Math formula $...=...$
    math_matches = list(re.finditer(r"\$([^$]+)\$", text))
    for m in reversed(math_matches):
        val = m.group(1).strip()
        if "=" in val:
            ans = val.split("=")[-1].strip()
            if ans:
                return ans

    # Priority 4: Explicit equation like '= 2 euros' or '= 42'
    eq_matches = list(re.finditer(r"=\s*([-+]?\d+(?:\.\d+)?)\s*(?:[a-zA-Z]+)?", text))
    if eq_matches:
        return eq_matches[-1].group(1).strip()

    # Priority 5: Word-bounded prose patterns
    for pat in _PROSE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip().rstrip(".")
            if ans:
                return ans

    # Strict: NO fallback to arbitrary numbers in text
    return ""


def grade_prediction(pred_text: str, gold_text: str) -> Tuple[bool, bool, str, str]:
    """Grade a model prediction against gold answer.

    Returns:
        (strict_match, lenient_match, boxed_extract, lenient_extract)
    """
    gold_clean = _extract_boxed(gold_text) or extract_answer(gold_text) or gold_text.strip()

    # Strict: requires explicit \boxed{} in prediction
    boxed_val = _extract_boxed(pred_text)
    strict_match = bool(answers_match(boxed_val, gold_clean)) if boxed_val else False

    # Lenient: tries boxed, fractions, equations, prose markers
    lenient_val = lenient_extract(pred_text)
    lenient_match = bool(answers_match(lenient_val, gold_clean)) if lenient_val else False

    return strict_match, lenient_match, (boxed_val or ""), (lenient_val or "")

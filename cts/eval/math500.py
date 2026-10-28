"""
MATH-500 style evaluation hook.

Wire a dataset JSONL (e.g. HuggingFace `HuggingFaceH4/MATH-500` export) with fields:
  `problem` or `question`, `answer` (final boxed or numeric), optional `id`.

Grader: for a minimal path, compare `normalize_answer(pred) == normalize_answer(gold)`.
Full sympy integration can be added later.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

# Prose answer markers, shared by extract_answer() and extract_answer_method()
# so the reported extraction path can never diverge from the actual one.
_PROSE_PATTERNS = [
    r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)",
    r"(?:answer|result)\s*[=:]\s*(.+?)(?:\.|$)",
    r"(?:therefore|thus|so|hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(.+?)(?:\.|$)",
]


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    # Apply LaTeX macro substitutions FIRST so the closing "}" in
    # patterns like \text{cm} and ^{\circ} is still present when the
    # regex tries to match. (Earlier code did rstrip("}") here, which
    # silently broke both patterns; that bug was caught by
    # tests/test_answers_match.py and is fixed now.)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left\s*", "", s)
    s = re.sub(r"\\right\s*", "", s)
    s = s.replace("\\,", "")
    s = s.replace("^\\circ", "")
    s = s.replace("^{\\circ}", "")
    # Now handle \boxed{...} -- prefer balanced extraction; fall back to
    # the legacy "strip prefix and trailing brace" only if extraction
    # finds nothing.
    if "\\boxed{" in s:
        boxed = _extract_boxed(s)
        if boxed:
            s = boxed
        else:
            s = s.replace("\\boxed{", "").rstrip("}")
    s = re.sub(r"[$%]", "", s)
    s = s.replace(",", "")
    return s


try:  # optional dependency: HF math-verify (used by lighteval for MATH/AIME)
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _MATH_VERIFY_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _MATH_VERIFY_AVAILABLE = False


def _as_plain_int(s: str) -> Optional[int]:
    """Return int value if the answer is a bare integer (allows $, +/-, leading zeros)."""
    t = str(s).strip().strip("$").strip()
    m = re.fullmatch(r"[-+]?\d+", t)
    return int(m.group()) if m else None


def extract_answer_method(text: str) -> str:
    """Which path extract_answer() takes: 'gsm_hash' | 'boxed' | 'prose' | 'none'.

    Logged per sample so that "answers lost to extraction failure" is a
    first-class, reportable quantity rather than an invisible source of error.
    """
    t = (text or "").strip()
    if re.search(r"####\s*(.+?)$", t, re.MULTILINE):
        return "gsm_hash"
    if _extract_boxed(t):
        return "boxed"
    for pat in _PROSE_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return "prose"
    return "none"


def _mv_wrap(s: str) -> str:
    """Wrap a bare answer in $...$ so math-verify will parse it as LaTeX."""
    t = str(s).strip()
    if not t or t.startswith("$") or "\\boxed" in t:
        return t
    return f"${t}$"


def _legacy_answers_match(pred: str, gold: str) -> bool:
    """Original string/first-number comparison. Kept only as a parse-failure fallback.

    NOTE: this has known false negatives (fraction<->decimal, unreduced fractions,
    sqrt simplification) because it compares only the FIRST numeric substring.
    """
    pn = normalize_answer(pred)
    gn = normalize_answer(gold)
    if pn == gn:
        return True
    try:
        pv = float(re.search(r"-?[\d.]+", pn).group())
        gv = float(re.search(r"-?[\d.]+", gn).group())
        if abs(pv - gv) < 1e-6:
            return True
    except (AttributeError, ValueError, TypeError):
        pass
    return False


def answers_match(pred: str, gold: str) -> bool:
    """Mathematical-equivalence answer matching.

    Uses ``math-verify`` (symbolic equivalence) so that e.g. ``\\frac{1}{2}``==``0.5``,
    ``\\frac{2}{4}``==``\\frac{1}{2}`` and ``3\\sqrt{2}``==``\\sqrt{18}`` all match.
    The previous first-numeric-substring comparison produced systematic FALSE
    NEGATIVES on exactly these cases (5/10 grader self-tests failed), deflating
    every reported accuracy. Falls back to the legacy comparison only when
    ``math-verify`` is unavailable or cannot parse an expression.
    """
    if pred is None or not str(pred).strip():
        return False
    # AIME-style integer answers are 0-999 and models often pad ("042"); compare
    # numerically so leading zeros do not create false negatives.
    _pi, _gi = _as_plain_int(pred), _as_plain_int(gold)
    if _pi is not None and _gi is not None:
        return _pi == _gi
    if _MATH_VERIFY_AVAILABLE:
        try:
            # math-verify only parses LaTeX-delimited expressions: bare "\sqrt{4}"
            # yields an empty parse, so wrap both sides in $...$ first.
            if bool(_mv_verify(_mv_parse(_mv_wrap(gold)), _mv_parse(_mv_wrap(pred)))):
                return True
        except Exception:
            pass  # unparseable -> fall through to legacy
    # Union semantics: legacy string normalization still catches formatting-only
    # differences that math-verify declines to parse.
    return _legacy_answers_match(pred, gold)


def _extract_boxed(text: str) -> str:
    """Extract content of the last \\boxed{...} with balanced braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else text[start:]


def extract_gold(gold: str) -> str:
    """Extract final answer from gold, handling GSM8K '#### <num>' format."""
    gsm_match = re.search(r"####\s*(.+?)$", gold, re.MULTILINE)
    if gsm_match:
        return gsm_match.group(1).strip()
    return gold


def extract_answer(text: str) -> str:
    """Extract the final answer from model-generated text.

    Handles: \\boxed{...}, '#### <num>' (GSM8K), 'the answer is ...', last number.
    """
    text = text.strip()

    gsm_match = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
    if gsm_match:
        return gsm_match.group(1).strip()

    boxed = _extract_boxed(text)
    if boxed:
        return boxed.strip()

    for pat in _PROSE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # NO last-number / whole-text fallback.
    #
    # The previous fallbacks returned the last number found anywhere in the text
    # (or the entire completion) whenever no explicit answer marker was present.
    # When generation is truncated before the final answer -- which was happening
    # on 100% of samples (0/250 completions contained \boxed{}) -- that silently
    # fed a mid-reasoning fragment to the grader and scored it as the model's
    # answer. An explicit extraction failure is strictly more informative than a
    # polluted value, so we return "" and log instead.
    _logger.debug("EXTRACT_FAILED no answer marker in completion (len=%d): %r",
                  len(text), text[-200:])
    return ""


def load_math_samples(path: Path | str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_pass_at_1(
    samples: List[Dict[str, Any]],
    predict_fn: Callable[[str], str],
    *,
    gold_key: str = "answer",
    question_key: str = "problem",
    include_items: bool = False,
    pred_max_chars: int = 4096,
) -> Dict[str, Any]:
    """`predict_fn(question)` returns model string; compared to `gold_key` after normalize."""
    ok = 0
    n = 0
    items: List[Dict[str, Any]] = []
    for ex in samples:
        q = ex.get(question_key) or ex.get("question") or ""
        gold = ex.get(gold_key, "")
        if not q:
            continue
        pred = predict_fn(str(q))
        n += 1
        match = normalize_answer(str(pred)) == normalize_answer(str(gold))
        if match:
            ok += 1
        if include_items:
            rid = ex.get("unique_id", ex.get("id", ""))
            items.append(
                {
                    "id": str(rid) if rid is not None else "",
                    "match": match,
                    "gold": str(gold)[:512],
                    "pred": str(pred)[:pred_max_chars],
                }
            )
    out: Dict[str, Any] = {
        "pass_at_1": (ok / n) if n else 0.0,
        "n": n,
        "correct": ok,
    }
    if include_items:
        out["items"] = items
    return out


def evaluate_stub() -> dict:
    return {
        "pass_at_1": None,
        "note": "Set MATH500_JSONL and run evaluate_pass_at_1(load_math_samples(path), predict_fn)",
    }

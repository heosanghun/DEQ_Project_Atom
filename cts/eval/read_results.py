"""Canonical reader for harness result JSONL files.

ALL analysis and reporting must go through this module. Ad-hoc re-parsing of
result files caused three separate misreadings in a single day:

  1. looking for a ``pred`` key when the harness writes ``extracted_pred``
  2. looking for a ``match`` key when the harness writes ``graded_match``
  3. searching for ``"\\\\boxed"`` from inside a raw-string block, which looks
     for two backslashes and therefore reported a 0% \\boxed rate when the
     true rate was 10-60%

The fix is structural: this module owns the key names and the escaping
conventions the harness writes, so a temporary script cannot get them wrong.

Counts that the harness itself records (e.g. ``extract_method``) are read from
the recorded field and are NOT recomputed by re-parsing raw text.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

# Literal backslash followed by "boxed". Built by concatenation so that it is
# correct regardless of whether a caller embeds this module's source in a raw
# string, an f-string, or a shell heredoc.
BOXED_MARKER = chr(92) + "boxed"


@dataclass
class Sample:
    """One graded problem, using the harness's own field names."""

    problem_id: str = ""
    gold_ans: str = ""
    raw_output_text: str = ""
    cleaned_output: str = ""
    extracted_pred: str = ""
    graded_match: bool = False
    oracle_grader_check: bool = False
    extract_method: Optional[str] = None
    n_output_tokens: int = 0
    wall_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_boxed(self) -> bool:
        return BOXED_MARKER in (self.raw_output_text or "")

    @property
    def is_empty(self) -> bool:
        return not (self.raw_output_text or "").strip()

    def truncated(self, cap: int) -> bool:
        return self.n_output_tokens >= cap


@dataclass
class Cell:
    name: str
    meta: Dict[str, Any] = field(default_factory=dict)
    samples: List[Sample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def weights(self) -> str:
        return str(self.meta.get("weights", "?"))

    @property
    def fmt(self) -> str:
        return str(self.meta.get("fmt", "?"))

    def boxed_rate(self) -> float:
        return _pct(sum(s.has_boxed for s in self.samples), len(self.samples))

    def truncation_rate(self, cap: int) -> float:
        return _pct(sum(s.truncated(cap) for s in self.samples), len(self.samples))

    def accuracy(self) -> float:
        """Accuracy as recorded by the harness (graded_match)."""
        return _pct(sum(s.graded_match for s in self.samples), len(self.samples))

    def mean_tokens(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.n_output_tokens for s in self.samples) / len(self.samples)

    def extract_methods(self) -> Dict[str, int]:
        """Counts of the harness-recorded extract_method field (never recomputed)."""
        return dict(Counter(s.extract_method or "unrecorded" for s in self.samples))

    def extract_method_recorded(self) -> bool:
        """True only if the harness wrote extract_method for every sample.

        Older result files predate that field. Callers must check this before
        trusting extract_methods(); when False, boxed presence is still
        available via Sample.has_boxed (owned by this reader, not re-parsed
        ad hoc), but the boxed-vs-prose-vs-none split is NOT recoverable and
        must be reported as unavailable rather than guessed ([FILL] rule).
        """
        return bool(self.samples) and all(
            s.extract_method is not None for s in self.samples
        )

    def by_id(self) -> Dict[str, Sample]:
        return {s.problem_id: s for s in self.samples}

    def wrong_samples(self) -> List[Sample]:
        """Samples the harness graded incorrect (graded_match is False)."""
        return [s for s in self.samples if not s.graded_match]


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def _to_sample(d: Dict[str, Any]) -> Sample:
    tok = d.get("raw_output_token_ids") or []
    return Sample(
        problem_id=str(d.get("problem_id", "")),
        gold_ans=str(d.get("gold_ans") or d.get("gold_raw") or ""),
        raw_output_text=str(d.get("raw_output_text") or ""),
        cleaned_output=str(d.get("cleaned_output") or ""),
        extracted_pred=str(d.get("extracted_pred") or ""),
        graded_match=bool(d.get("graded_match")),
        oracle_grader_check=bool(d.get("oracle_grader_check")),
        extract_method=d.get("extract_method"),
        n_output_tokens=len(tok) if isinstance(tok, list) else 0,
        wall_ms=float(d.get("wall_ms") or 0.0),
        raw=d,
    )


def read_cell(path: str) -> Cell:
    """Read one harness JSONL file (a leading ``_meta`` line plus samples)."""
    name = os.path.basename(path)
    if name.endswith(".json"):
        name = name[:-5]
    cell = Cell(name=name)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in d:
                cell.meta = d["_meta"]
            else:
                cell.samples.append(_to_sample(d))
    return cell


def read_cells(pattern: str) -> Dict[str, Cell]:
    """Read every result file matching a glob, keyed by cell name."""
    return {c.name: c for c in (read_cell(p) for p in sorted(glob.glob(pattern)))}


def iter_samples(pattern: str) -> Iterator[Sample]:
    for cell in read_cells(pattern).values():
        yield from cell.samples


def matched_comparison(a: Cell, b: Cell) -> Dict[str, Any]:
    """Compare two cells on the problem ids they share (fair subset comparison)."""
    ia, ib = a.by_id(), b.by_id()
    shared = sorted(set(ia) & set(ib))
    if not shared:
        return {"n_shared": 0}
    return {
        "n_shared": len(shared),
        "a_name": a.name,
        "b_name": b.name,
        "a_acc": _pct(sum(ia[i].graded_match for i in shared), len(shared)),
        "b_acc": _pct(sum(ib[i].graded_match for i in shared), len(shared)),
        "a_boxed": _pct(sum(ia[i].has_boxed for i in shared), len(shared)),
        "b_boxed": _pct(sum(ib[i].has_boxed for i in shared), len(shared)),
        "a_mean_tok": sum(ia[i].n_output_tokens for i in shared) / len(shared),
        "b_mean_tok": sum(ib[i].n_output_tokens for i in shared) / len(shared),
        "shared_ids": shared,
    }

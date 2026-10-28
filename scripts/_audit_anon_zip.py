#!/usr/bin/env python3
"""Identity-leak audit for the anonymous submission ZIP.

Scans every text-like file in ``anonymous_submission_neurips2026.zip`` for
known author signatures, GitHub handles, emails, and domains that would
violate NeurIPS double-blind review. Prints a per-pattern hit table and
a final PASS / FAIL verdict so the agent can gate the submission.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = REPO_ROOT / "anonymous_submission_neurips2026.zip"

# Patterns that, if found, indicate an identity leak. Word-boundary anchors
# avoid matching innocuous substrings (e.g. "heritage" would not flag
# "hero", but "heosanghun" is a unique handle that has no false positives).
# Identity leaks: real handles, real emails, and ABSOLUTE local paths that
# embed the Windows username. The narrow path patterns avoid matching the
# innocuous lowercase ``sims`` token used as MCTS-simulation abbreviation
# throughout the codebase (``args.sims``, ``--sims``, ``sims = []``).
HARD_FAIL_PATTERNS = [
    r"\bheosanghun\b",
    r"49586773",
    r"github\.com/[A-Za-z0-9_-]*heosanghun",
    r"[A-Za-z0-9._%+-]+@(?:gmail|outlook|hotmail|naver|kakao|daum|nate)\.com",
    # Windows local-path leaks (case-sensitive on "Sims" the proper noun).
    r"[Cc]:[\\/]+Users[\\/]+Sims",
    r"[\\/]Users[\\/]+Sims[\\/]",
    r"\bSims\b(?=\s*[\\/])",
]

# These should not appear at all (we rewrote git history but double-check).
HARD_FAIL_PATHS = [
    re.compile(r"\.git(?:/|$)"),
    re.compile(r"\.git_backup_pre_anonymize"),
    re.compile(r"\.mailmap_anonymize"),
    re.compile(r"_filter_repo_callback\.py$"),
    # D-7 Apr 29 (evening): defence-in-depth against author-draft leaks.
    # These are gitignored AND in EXCLUDE_GLOBS, but the audit must
    # also catch them in case either layer is bypassed.
    re.compile(r"PROGRESS_REPORT_.*\.md$"),
    re.compile(r"OPENREVIEW_RESPONSE_PREP\.md$"),
    re.compile(r"NEXT_TASKS_.*\.md$"),
    re.compile(r"PAPER_VS_LOCAL_FINAL\.md$"),
    re.compile(r"PAPER_VS_LOCAL_INTUITIVE\.md$"),
    re.compile(r"PAPER_CONSISTENCY_AUDIT\.md$"),
    re.compile(r"ROOT_CAUSE_ANALYSIS\.md$"),
    re.compile(r"EXPERIMENTAL_RESULTS\.md$"),
]

# Files we expect to find (positive sanity check that the ZIP is non-empty).
# NOTE: Author-facing planning docs (EXPERIMENTAL_RESULTS.md, NEXT_TASKS_*.md,
# PAPER_VS_LOCAL_*.md, ROOT_CAUSE_ANALYSIS.md, SUBMISSION_GUIDE_*.md) are
# intentionally excluded from the ZIP and therefore must NOT appear here --
# they are tracked in EXCLUDE_GLOBS in scripts/make_anonymous_submission.py.
EXPECTED_PATHS = [
    "README.md",
    "REPRODUCIBILITY.md",
    "REVIEWER_FAQ.md",
    "cts/__init__.py",
    "scripts/run_cts_eval_full.py",
    "scripts/compare_to_paper_table2.py",
    "tests/test_cts_full_episode.py",
    "tests/test_public_api.py",
    "CHANGELOG.md",
]

TEXT_EXTS = {
    ".md", ".py", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".rst", ".sh", ".bat", ".ps1", ".html", ".css", ".js", ".ts", ".tsx",
    ".jsonl",
}


def is_textlike(name: str) -> bool:
    return any(name.endswith(ext) for ext in TEXT_EXTS)


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"[AUDIT] FAIL: {ZIP_PATH} does not exist.")
        return 2

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        # Strip the leading top-level dir so we can match expected_paths.
        roots = {n.split("/", 1)[0] for n in names}
        top = next(iter(roots)) if len(roots) == 1 else ""
        prefix = f"{top}/" if top else ""

        # 1) Path-level hard fails.
        path_violations: list[str] = []
        for n in names:
            for pat in HARD_FAIL_PATHS:
                if pat.search(n):
                    path_violations.append(n)
                    break

        # 2) Sanity check expected files are present.
        missing: list[str] = []
        for ep in EXPECTED_PATHS:
            if (prefix + ep) not in names:
                missing.append(ep)

        # 3) Content-level identity leaks.
        # Case-SENSITIVE: ``Sims`` (proper-noun Windows username) must not be
        # collapsed with the lowercase MCTS ``sims`` abbreviation. Other
        # patterns (handles, emails, GitHub URLs) are already mixed-case-safe
        # because they are written with explicit character classes.
        compiled = [re.compile(p) for p in HARD_FAIL_PATTERNS]
        per_pattern_hits: dict[str, list[tuple[str, int, str]]] = {p: [] for p in HARD_FAIL_PATTERNS}
        scanned = 0
        for n in names:
            if not is_textlike(n):
                continue
            try:
                raw = zf.read(n).decode("utf-8", errors="ignore")
            except Exception:
                continue
            scanned += 1
            for line_no, line in enumerate(raw.splitlines(), start=1):
                for pat, rx in zip(HARD_FAIL_PATTERNS, compiled):
                    if rx.search(line):
                        per_pattern_hits[pat].append((n, line_no, line.strip()[:160]))

    # ---------------- Report ----------------
    print(f"[AUDIT] zip            : {ZIP_PATH.name}")
    print(f"[AUDIT] total files    : {len(names)}")
    print(f"[AUDIT] text scanned   : {scanned}")
    print(f"[AUDIT] expected found : {len(EXPECTED_PATHS) - len(missing)}/{len(EXPECTED_PATHS)}")

    if missing:
        print(f"[AUDIT] WARNING: missing expected files -> {missing}")

    print()
    print("[AUDIT] path-level hard-fail patterns:")
    if path_violations:
        for v in path_violations[:20]:
            print(f"  - VIOLATION path: {v}")
    else:
        print("  - none.")

    print()
    print("[AUDIT] content-level identity-leak patterns:")
    total_hits = 0
    for pat, hits in per_pattern_hits.items():
        n_hits = len(hits)
        total_hits += n_hits
        marker = "OK " if n_hits == 0 else "HIT"
        print(f"  [{marker}] {pat:60s} hits={n_hits}")
        for path, lineno, line in hits[:5]:
            print(f"        {path}:{lineno}: {line}")

    print()
    if path_violations or total_hits > 0 or missing:
        print("[AUDIT] VERDICT: FAIL  (do NOT submit this ZIP)")
        return 1
    print("[AUDIT] VERDICT: PASS  (NeurIPS double-blind safe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

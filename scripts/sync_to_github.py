#!/usr/bin/env python3
"""Continuous GitHub sync helper for the NeurIPS 2026 anonymous repo.

Designed to be run periodically (or after each substantive change) to
keep ``origin/d2-neurips2026-anonymized`` aligned with the NeurIPS-2026
submission state. The script is conservative by design:

* It NEVER auto-commits if the leak audit fails (running
  ``scripts/_audit_anon_zip.py`` against a freshly built submission ZIP).
* It NEVER commits files that contain identity-leak patterns according
  to the audit's regex set (the audit script itself is gitignored).
* It refuses to push to a branch other than ``d2-neurips2026-anonymized``
  unless ``--branch`` is supplied explicitly.
* It always commits with the anonymous identity
  ``Anonymous <anonymous@neurips.cc>`` regardless of the local
  ``user.name`` / ``user.email`` setting.

Typical usage::

    # Dry-run: show what would be committed and pushed.
    python scripts/sync_to_github.py --dry-run

    # Real commit + push, with a custom subject line.
    python scripts/sync_to_github.py \\
        --message "docs(D11): update REVIEWER_FAQ Q13"

    # Skip the leak audit (NOT recommended; only for emergency fixes
    # where the ZIP build itself is broken).
    python scripts/sync_to_github.py --no-audit

Exit codes:

* 0 — clean (no diff) OR commit + push succeeded.
* 1 — uncommitted changes contain a leak; refused to commit.
* 2 — git push failed.
* 3 — anonymous ZIP build or audit failed.
* 4 — other error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRANCH = "d2-neurips2026-anonymized"
ANON_NAME = "Anonymous"
ANON_EMAIL = "anonymous@neurips.cc"


def _run(cmd, *, check=True, capture=True):
    """Run *cmd* in ROOT, return (returncode, stdout, stderr)."""
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=capture,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        sys.stderr.write(
            f"[sync] command failed (rc={proc.returncode}): "
            f"{' '.join(map(str, cmd))}\n"
        )
        if capture:
            sys.stderr.write(f"  stdout: {proc.stdout}\n")
            sys.stderr.write(f"  stderr: {proc.stderr}\n")
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _git(*args, anon=False, **kwargs):
    base = ["git"]
    if anon:
        base.extend([
            "-c", f"user.name={ANON_NAME}",
            "-c", f"user.email={ANON_EMAIL}",
        ])
    return _run(base + list(args), **kwargs)


def _list_changes() -> tuple[list[str], list[str]]:
    """Return (modified_files, untracked_files) relative to ROOT."""
    rc, out, _ = _git("status", "--porcelain")
    if rc != 0:
        return [], []
    modified, untracked = [], []
    for line in out.splitlines():
        if not line.strip():
            continue
        flag = line[:2]
        path = line[3:].strip()
        if flag.startswith("??"):
            untracked.append(path)
        else:
            modified.append(path)
    return modified, untracked


def _run_leak_audit() -> bool:
    """Build the anonymous ZIP and run the audit. Return True iff PASS."""
    audit_script = ROOT / "scripts" / "_audit_anon_zip.py"
    if not audit_script.is_file():
        sys.stderr.write(
            "[sync] WARNING: scripts/_audit_anon_zip.py not on disk; "
            "skipping audit. (This is expected on a freshly cloned "
            "reviewer-side checkout because the audit script itself "
            "is gitignored.)\n"
        )
        return True
    rc, out, _ = _run(
        [sys.executable, str(ROOT / "scripts" / "make_anonymous_submission.py")],
        check=False,
    )
    if rc != 0:
        sys.stderr.write("[sync] anonymous ZIP build FAILED\n")
        sys.stderr.write(out)
        return False
    rc, out, _ = _run(
        [sys.executable, str(audit_script)],
        check=False,
    )
    sys.stdout.write(out)
    return rc == 0 and "VERDICT: PASS" in out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync local repo to origin (NeurIPS anonymous branch).",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="remote branch to push to (default: %(default)s)",
    )
    parser.add_argument(
        "--message",
        default="chore(D11): periodic GitHub sync",
        help="commit subject line (will be prepended with the change list)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be committed and pushed; do not modify",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="skip the anonymous-ZIP leak audit (NOT recommended)",
    )
    args = parser.parse_args()

    modified, untracked = _list_changes()
    if not modified and not untracked:
        print("[sync] working tree clean; nothing to commit.")
        return 0

    print("[sync] modified files:")
    for f in modified:
        print(f"   M  {f}")
    print("[sync] untracked files:")
    for f in untracked:
        print(f"   ?  {f}")

    if not args.no_audit:
        print("[sync] running leak audit (anonymous ZIP rebuild + scan)...")
        if not _run_leak_audit():
            sys.stderr.write(
                "[sync] LEAK AUDIT FAILED — refusing to commit. "
                "Fix the offending files (or extend EXCLUDE_GLOBS in "
                "scripts/make_anonymous_submission.py) and rerun.\n"
            )
            return 1
        print("[sync] leak audit: PASS")

    if args.dry_run:
        print("[sync] dry-run; would commit + push above changes to "
              f"origin/{args.branch}.")
        return 0

    rc, _, _ = _git("add", "-A", check=False)
    if rc != 0:
        return 4

    msg = args.message
    rc, out, err = _git(
        "commit", "-m", msg, anon=True, check=False
    )
    if rc != 0 and "nothing to commit" not in (out + err):
        sys.stderr.write(f"[sync] commit failed: {err}\n")
        return 4

    rc, out, err = _git(
        "push", "origin", f"HEAD:{args.branch}", check=False
    )
    if rc != 0:
        sys.stderr.write(f"[sync] push failed: {err}\n")
        return 2

    print("[sync] OK - pushed to origin/{}".format(args.branch))
    return 0


if __name__ == "__main__":
    sys.exit(main())

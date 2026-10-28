"""Run the unchanged locked grading wrapper with a Windows-safe main entry point."""
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise SystemExit("Run from C:\\Project\\CTS\\shared")
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python OUTBOX/run_c002_grader.py spec.json new-output.json")
    target = Path(sys.argv[2])
    if target.exists():
        raise SystemExit("Output already exists; retained without modification")
    target.parent.mkdir(parents=True, exist_ok=True)
    wrapper = root / "scripts" / "codex_grade_local.py"
    sys.argv = [str(wrapper), sys.argv[1], sys.argv[2]]
    # A non-main run_name keeps this guarded launcher as multiprocessing's
    # main module. The original wrapper and locked grader remain byte-identical.
    runpy.run_path(str(wrapper), run_name="cts_original_grading_wrapper")

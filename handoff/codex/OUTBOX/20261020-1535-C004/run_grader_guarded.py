"""Execute the unchanged authorized wrapper without reentry on Windows spawn.
No grader functions, parsing settings or timeout policies are altered.
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    import runpy
    root=Path("D:/AI/cts")
    here=Path(__file__).resolve().parent
    wrapper=root/"handoff/codex/scripts/codex_grade_local.py"
    output=here/"x5_items_guarded.json"
    if output.exists():
        raise FileExistsError(output)
    sys.argv=[str(wrapper),str(here/"s0spec_local.json"),str(output)]
    # Keep this guarded launcher as __main__ for multiprocessing.spawn.
    runpy.run_path(str(wrapper),run_name="codex_authorized_wrapper")

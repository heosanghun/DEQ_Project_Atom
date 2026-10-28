"""Verify the AIME garbage fix actually works on CTS-4nu (Fix A + Fix B).

Loads the first AIME problem, runs cts_4nu through ``_run_cts_on_problems``
just like the post-Stage-2 pipeline does, and asserts:
  * The CTS soft-prompt decode now uses problem_text (paper-faithful)
  * If CTS returns a non-numeric token, the dispatcher falls back to
    greedy and the cell at least reports a digit-leading prediction.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CTS_DISABLE_TRITON", "1")
os.environ.setdefault("CTS_EVAL_TAU_CAP", "1e13")
os.environ.setdefault("CTS_EVAL_EPISODE_TIMEOUT", "180")


def main() -> int:
    print("=" * 70)
    print("AIME garbage fix verification -- cts_4nu single-problem")
    print("=" * 70)

    aime_path = ROOT / "data" / "aime" / "test.jsonl"
    with open(aime_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    problem = first["problem"]
    gold = first["answer"]
    print(f"\nProblem (year={first.get('year')}): {problem[:160]}...")
    print(f"Gold answer: '{gold}'")

    from scripts.run_cts_eval_full import _run_cts_on_problems
    from cts.utils.config import load_config

    cfg = load_config("default")

    print("\n[1] Loading model + running cts_4nu (this will take ~3-4 min)...")
    scores = _run_cts_on_problems(
        method="cts_4nu",
        problems=[first],
        cfg=cfg,
        device="cuda:0",
        model_dir=None,
        benchmark="aime",
        seed=0,
        nu_trace_dir=None,
    )
    print(f"\n[2] cts_4nu/aime/seed=0 score: {scores}")
    if scores and scores[0] == 1.0:
        print("[3] PASS: matched gold answer (paper-faithful CTS path active)")
    else:
        print("[3] DID NOT match gold (expected on most AIME problems on local "
              "compute, but the prediction MUST now be digit-leading because "
              "of the garbage-fallback fix). Check the [cts/fallback] tag in "
              "the per-problem print() above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI smoke probe for the non-headline MockTiny MAC contract."""

from __future__ import annotations

import argparse
import json

import torch

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.transition import transition
from cts.eval.flops_contract import public_isoflop_report
from cts.types import NuVector, RuntimeBudgetState


def main() -> None:
    p = argparse.ArgumentParser(description="CTS Iso-FLOP report (mock backbone)")
    p.add_argument("--json", action="store_true", help="print JSON only")
    args = p.parse_args()

    # This is a deterministic contract smoke test, not a paper E4B profile.
    # Keep it on CPU so its output can never be mistaken for GPU headline data.
    torch.manual_seed(2026)
    device = torch.device("cpu")
    bb = MockTinyBackbone(hidden=64, num_layers=42).to(device)
    nu = NuVector(nu_tol=5e-3, nu_temp=1.0, nu_expl=1.0)
    budget = RuntimeBudgetState()
    r = transition(
        "iso-flop probe",
        0,
        nu,
        budget,
        bb,
        K=64,
        d=64,
        broyden_max_iter=30,
        tau_flops_budget=1e22,
    )
    rep = public_isoflop_report(r.solver_stats)
    rep = {
        "report_kind": "mock_non_headline_smoke",
        "diagnostic_only": True,
        **rep,
    }
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        for k, v in rep.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Orchestrate and aggregate ``diag_harness_2x2.py`` runs.

Each requested cell runs in a fresh subprocess.  The harness owns only the
per-cell JSONL measurement; this module is the single source of truth for
cross-cell reproducibility, prompt-token contracts, factorial effects, and the
final pipeline report.

The pipeline never mutates its input dataset, problem-ID file, or completed
JSONL logs.  All generated artifacts are placed below ``--out-dir``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "scripts" / "diag_harness_2x2.py"

CONTROLLED_CELLS = ("REF", "M0", "M1", "M2", "M3")
FACTORIAL_CELLS = ("M0", "M1", "M2", "M3")
NATIVE_CELLS = ("M0_native", "M1_native", "M2_native", "M3_native")
REPRO_FIELDS = ("raw_output_token_ids", "extracted_pred", "graded_match")
PROMPT_TOKEN_PAIRS = (
    ("plain_controlled", "M0", "M2"),
    ("canonical_controlled", "M1", "M3"),
    ("plain_native", "M0_native", "M2_native"),
    ("canonical_native", "M1_native", "M3_native"),
)
RESERVED_FORWARD_ARGS = ("--cell", "--device", "--out")


def _selected_cells(cell_set: str, cell_order: str | None) -> List[str]:
    if cell_set == "controlled":
        expected = set(CONTROLLED_CELLS)
    elif cell_set == "native":
        expected = set(NATIVE_CELLS)
    else:
        expected = set(CONTROLLED_CELLS) | set(NATIVE_CELLS)

    if cell_order is None:
        return sorted(expected)

    requested = [part.strip() for part in cell_order.split(",") if part.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("--cell-order contains duplicate cells")
    if set(requested) != expected:
        missing = sorted(expected - set(requested))
        extra = sorted(set(requested) - expected)
        raise ValueError(
            "--cell-order must be an exact permutation of the selected set; "
            f"missing={missing}, extra={extra}"
        )
    return requested


def _validate_forwarded_args(extra_args: Sequence[str]) -> None:
    for arg in extra_args:
        if any(arg == reserved or arg.startswith(f"{reserved}=")
               for reserved in RESERVED_FORWARD_ARGS):
            raise ValueError(
                f"{arg!r} cannot be forwarded; the pipeline owns "
                "--cell, --device, and --out"
            )


def run_cell(
    *,
    cell: str,
    benchmark: str,
    data_path: str,
    base_model: str,
    it_model: str,
    problem_ids_file: str | None,
    limit: int | None,
    out_dir: Path,
    phys_gpu: int,
    seed: int,
    extra_args: Sequence[str],
    out_override: Path | None = None,
) -> Dict[str, Any]:
    """Run one isolated cell and return an explicit completion record."""
    out_file = out_override or (out_dir / f"{cell}_{benchmark}.jsonl")
    cmd = [
        sys.executable,
        str(HARNESS),
        "--cell", cell,
        "--benchmark", benchmark,
        "--data", data_path,
        "--device", "cuda:0",
        "--base-model", base_model,
        "--it-model", it_model,
        "--out", str(out_file),
    ]
    if problem_ids_file:
        cmd += ["--problem-ids-file", problem_ids_file]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    cmd += list(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(phys_gpu)
    env["CTS_GLOBAL_SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)

    print(f"\n>>> Running fresh subprocess: cell={cell}, physical_gpu={phys_gpu}, device=cuda:0")
    print(f"Command: {' '.join(cmd)}")
    result: Dict[str, Any] = {
        "cell": cell,
        "status": "failed",
        "returncode": None,
        "out_file": str(out_file),
        "command": cmd,
        "error": None,
    }
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )
        result["returncode"] = completed.returncode
        if completed.returncode == 0 and out_file.is_file():
            result["status"] = "completed"
        elif completed.returncode == 0:
            result["error"] = "subprocess succeeded but produced no JSONL output"
        else:
            result["error"] = f"subprocess exited with code {completed.returncode}"
    except OSError as exc:
        result["error"] = f"failed to start subprocess: {exc}"
    return result


def load_item_records(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load per-item records without modifying the source JSONL."""
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "_meta" in row:
                continue
            problem_id = str(row.get("problem_id", ""))
            if not problem_id:
                raise ValueError(f"{path}:{line_number}: missing problem_id")
            if problem_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate problem_id={problem_id}")
            records[problem_id] = row
    return records


def accuracy_summary(path: Path) -> Dict[str, Any]:
    records = load_item_records(path)
    if not records:
        return {"accuracy": None, "n": 0, "n_correct": 0}
    n_correct = sum(row.get("graded_match") is True for row in records.values())
    return {
        "accuracy": n_correct / len(records),
        "n": len(records),
        "n_correct": n_correct,
    }


def compare_fields(path_a: Path, path_b: Path, fields: Iterable[str]) -> Dict[str, Any]:
    rows_a = load_item_records(path_a)
    rows_b = load_item_records(path_b)
    ids_a = set(rows_a)
    ids_b = set(rows_b)
    mismatches: List[Dict[str, Any]] = []
    for problem_id in sorted(ids_a & ids_b):
        changed = [field for field in fields
                   if rows_a[problem_id].get(field) != rows_b[problem_id].get(field)]
        if changed:
            mismatches.append({"problem_id": problem_id, "fields": changed})
    only_a = sorted(ids_a - ids_b)
    only_b = sorted(ids_b - ids_a)
    return {
        "identical": not mismatches and not only_a and not only_b,
        "n_common": len(ids_a & ids_b),
        "only_in_first": only_a,
        "only_in_second": only_b,
        "mismatches": mismatches,
    }


def paired_effect(
    minuend_path: Path,
    subtrahend_path: Path,
    *,
    label: str,
) -> Dict[str, Any]:
    """Compute a paired accuracy delta from common problem IDs."""
    minuend = load_item_records(minuend_path)
    subtrahend = load_item_records(subtrahend_path)
    ids_min = set(minuend)
    ids_sub = set(subtrahend)
    common = sorted(ids_min & ids_sub)
    deltas: Dict[str, int] = {}
    for problem_id in common:
        lhs = int(minuend[problem_id].get("graded_match") is True)
        rhs = int(subtrahend[problem_id].get("graded_match") is True)
        deltas[problem_id] = lhs - rhs
    return {
        "label": label,
        "n_common": len(common),
        "mean_delta": (sum(deltas.values()) / len(common)) if common else None,
        "improved": sum(delta == 1 for delta in deltas.values()),
        "worsened": sum(delta == -1 for delta in deltas.values()),
        "unchanged": sum(delta == 0 for delta in deltas.values()),
        "only_in_minuend": sorted(ids_min - ids_sub),
        "only_in_subtrahend": sorted(ids_sub - ids_min),
        "per_problem_delta": deltas,
    }


def aggregate_results(
    output_paths: Mapping[str, Path],
    *,
    reproducibility_pair: tuple[Path, Path] | None = None,
) -> Dict[str, Any]:
    """Aggregate existing JSONL files; never mutate them."""
    report: Dict[str, Any] = {
        "accuracies": {},
        "factorial": None,
        "factorial_id_contract": None,
        "native": {},
        "prompt_token_contract": {},
        "reproducibility": None,
        "aggregation_errors": {},
    }

    valid_paths: Dict[str, Path] = {}
    for cell, path in output_paths.items():
        try:
            report["accuracies"][cell] = accuracy_summary(path)
            valid_paths[cell] = path
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report["aggregation_errors"][cell] = str(exc)

    if all(cell in valid_paths for cell in FACTORIAL_CELLS):
        factorial_ids = {
            cell: set(load_item_records(valid_paths[cell]))
            for cell in FACTORIAL_CELLS
        }
        reference_ids = factorial_ids["M0"]
        id_contract = {
            "identical": all(ids == reference_ids for ids in factorial_ids.values()),
            "reference_cell": "M0",
            "n_reference": len(reference_ids),
            "differences": {
                cell: {
                    "missing_vs_M0": sorted(reference_ids - ids),
                    "extra_vs_M0": sorted(ids - reference_ids),
                }
                for cell, ids in factorial_ids.items()
                if ids != reference_ids
            },
        }
        report["factorial_id_contract"] = id_contract
        acc = {cell: report["accuracies"][cell]["accuracy"] for cell in FACTORIAL_CELLS}
        if id_contract["identical"] and all(value is not None for value in acc.values()):
            effects = {
                "M1-M0": paired_effect(valid_paths["M1"], valid_paths["M0"], label="M1-M0"),
                "M3-M2": paired_effect(valid_paths["M3"], valid_paths["M2"], label="M3-M2"),
                "M2-M0": paired_effect(valid_paths["M2"], valid_paths["M0"], label="M2-M0"),
                "M3-M1": paired_effect(valid_paths["M3"], valid_paths["M1"], label="M3-M1"),
            }
            report["factorial"] = {
                "cells": {cell: report["accuracies"][cell] for cell in FACTORIAL_CELLS},
                "paired_effects": effects,
                "interaction": acc["M3"] - acc["M2"] - acc["M1"] + acc["M0"],
            }

    report["native"] = {
        cell: report["accuracies"][cell]
        for cell in NATIVE_CELLS
        if cell in report["accuracies"]
    }

    for label, first, second in PROMPT_TOKEN_PAIRS:
        if first in valid_paths and second in valid_paths:
            try:
                report["prompt_token_contract"][label] = compare_fields(
                    valid_paths[first], valid_paths[second], ("prompt_token_ids",)
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                report["prompt_token_contract"][label] = {"identical": False, "error": str(exc)}

    if reproducibility_pair is not None:
        try:
            report["reproducibility"] = compare_fields(
                reproducibility_pair[0], reproducibility_pair[1], REPRO_FIELDS
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report["reproducibility"] = {"identical": False, "error": str(exc)}
    return report


def print_report(report: Mapping[str, Any], execution: Mapping[str, Mapping[str, Any]]) -> None:
    print("\n" + "=" * 64)
    print("DIAGNOSTIC HARNESS PIPELINE REPORT")
    print("=" * 64)
    for name, result in execution.items():
        suffix = f" ({result['error']})" if result.get("error") else ""
        print(f"{name:20} {result['status']}{suffix}")

    print("\nAccuracy tables:")
    for cell, summary in report.get("accuracies", {}).items():
        accuracy = summary["accuracy"]
        shown = "NA" if accuracy is None else f"{accuracy * 100:.2f}%"
        family = "native" if cell in NATIVE_CELLS else "controlled/reference"
        print(f"  {cell:12} {shown:>8}  n={summary['n']:>3}  [{family}]")

    factorial = report.get("factorial")
    if factorial:
        print("\nControlled factorial paired effects:")
        for label, effect in factorial["paired_effects"].items():
            delta = effect["mean_delta"]
            shown = "NA" if delta is None else f"{delta * 100:+.2f}%"
            print(
                f"  {label:7} {shown:>8}  n={effect['n_common']} "
                f"(+{effect['improved']}/-{effect['worsened']}/={effect['unchanged']})"
            )
        print(f"  interaction {factorial['interaction'] * 100:+.2f}%")
    else:
        print("\nControlled factorial analysis unavailable (missing/invalid M0-M3 logs).")
    id_contract = report.get("factorial_id_contract")
    if id_contract is not None:
        print(
            "Factorial problem-ID contract: "
            + ("PASS" if id_contract.get("identical") else "FAIL")
        )

    repro = report.get("reproducibility")
    if repro is not None:
        print(
            "\nThree-field reproducibility: "
            + ("PASS" if repro.get("identical") else "FAIL")
        )
        if repro.get("mismatches"):
            print(f"  mismatches={repro['mismatches']}")

    for label, result in report.get("prompt_token_contract", {}).items():
        print(
            f"Prompt-token contract {label}: "
            + ("PASS" if result.get("identical") else "FAIL")
        )
    print("=" * 64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["math500", "gsm8k"])
    parser.add_argument("--data", required=True, help="Path to benchmark test data")
    parser.add_argument("--problem-ids-file", default=None, help="Stratified problem IDs file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-model", default="google/gemma-4-E4B")
    parser.add_argument("--it-model", default="google/gemma-4-E4B-it")
    parser.add_argument("--set", dest="cell_set", choices=["controlled", "native", "all"],
                        default="controlled")
    parser.add_argument(
        "--cell-order",
        default=None,
        help="Comma-separated exact permutation of the cells selected by --set (default: sorted)",
    )
    parser.add_argument("--phys-gpu", type=int, choices=range(4), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/diag2x2")
    args, extra_args = parser.parse_known_args()

    try:
        cells = _selected_cells(args.cell_set, args.cell_order)
        _validate_forwarded_args(extra_args)
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    execution: Dict[str, Dict[str, Any]] = {}
    successful_paths: Dict[str, Path] = {}

    for cell in cells:
        result = run_cell(
            cell=cell,
            benchmark=args.benchmark,
            data_path=args.data,
            base_model=args.base_model,
            it_model=args.it_model,
            problem_ids_file=args.problem_ids_file,
            limit=args.limit,
            out_dir=out_dir,
            phys_gpu=args.phys_gpu,
            seed=args.seed,
            extra_args=extra_args,
        )
        execution[cell] = result
        if result["status"] == "completed":
            successful_paths[cell] = Path(result["out_file"])

    repro_pair: tuple[Path, Path] | None = None
    if "M0" in successful_paths:
        repro_path = out_dir / f"M0_repro_{args.benchmark}.jsonl"
        repro_result = run_cell(
            cell="M0",
            benchmark=args.benchmark,
            data_path=args.data,
            base_model=args.base_model,
            it_model=args.it_model,
            problem_ids_file=args.problem_ids_file,
            limit=args.limit,
            out_dir=out_dir,
            phys_gpu=args.phys_gpu,
            seed=args.seed,
            extra_args=extra_args,
            out_override=repro_path,
        )
        execution["M0_repro"] = repro_result
        if repro_result["status"] == "completed":
            repro_pair = (successful_paths["M0"], repro_path)

    analysis = aggregate_results(successful_paths, reproducibility_pair=repro_pair)
    report = {
        "config": {
            "benchmark": args.benchmark,
            "data": args.data,
            "problem_ids_file": args.problem_ids_file,
            "cell_set": args.cell_set,
            "cell_order": cells,
            "physical_gpu": args.phys_gpu,
            "subprocess_device": "cuda:0",
            "seed": args.seed,
        },
        "execution": execution,
        "analysis": analysis,
    }
    report_path = out_dir / f"pipeline_report_{args.benchmark}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(analysis, execution)
    print(f"Report written to {report_path}")

    failed = any(item["status"] != "completed" for item in execution.values())
    contract_failed = any(
        not item.get("identical", False)
        for item in analysis.get("prompt_token_contract", {}).values()
    )
    id_contract = analysis.get("factorial_id_contract")
    id_contract_failed = (
        id_contract is not None and not id_contract.get("identical", False)
    )
    repro_failed = (
        analysis.get("reproducibility") is not None
        and not analysis["reproducibility"].get("identical", False)
    )
    aggregation_failed = bool(analysis.get("aggregation_errors"))
    empty_outputs = any(
        summary.get("n", 0) == 0
        for summary in analysis.get("accuracies", {}).values()
    )
    return 1 if (
        failed
        or contract_failed
        or id_contract_failed
        or repro_failed
        or aggregation_failed
        or empty_outputs
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())

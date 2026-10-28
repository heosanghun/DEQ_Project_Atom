"""TASK-C002 aggregates existing locked-grader flags; it does not grade answers.

The first local grading artifact has multiprocessing errors and is diagnostic
only. This script preserves that limitation in every generated report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GRADER = "90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13"
EXPECTED_ORDER = "6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c"
FLAGS = ("strict", "lenient", "boxed")
SEEDS = (47, 49, 50)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    keys = [int(row["id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate ids: {path}")
    return {str(row["id"]): row for row in rows}


def bootstrap(values, n_boot=10000, seed=2026):
    """One paired item cluster retains all 3 seed outcomes; same draw per seed."""
    n = len(values)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        indices = rng.choices(range(n), k=n)
        draws.append(sum(values[i] for i in indices) / n * 100)
    draws.sort()
    return {"mean_pp": sum(values) / n * 100,
            "ci95_pp": [draws[int(n_boot * .025)], draws[int(n_boot * .975)]],
            "n_boot": n_boot, "seed": seed}


def totals(matrix, order):
    result = {}
    for label, items in matrix.items():
        counts = [sum(items[q][i] for q in order) for i in range(3)]
        joint = sum(bool(items[q][0] and items[q][2]) for q in order)
        result[label] = {"n": len(order), **dict(zip(FLAGS, counts)),
                         "rates_pct": dict(zip(FLAGS, [v / len(order) * 100 for v in counts])),
                         "strict_and_boxed": joint,
                         "strict_within_boxed_pct": joint / counts[2] * 100 if counts[2] else None}
    return result


def statistics_for(matrix, order):
    result = {}
    for flag in (0, 1):
        for contrast in ("A", "B"):
            per_item = []
            per_seed = {}
            for seed in SEEDS:
                left = matrix[f"X1O{seed}"]
                right = matrix[f"shuf_X1O{seed}"] if contrast == "A" else matrix["C_fmt"]
                per_seed[str(seed)] = sum(left[q][flag] - right[q][flag] for q in order) / len(order) * 100
            for q in order:
                per_item.append(sum(
                    matrix[f"X1O{s}"][q][flag]
                    - matrix[f"shuf_X1O{s}" if contrast == "A" else "C_fmt"][q][flag]
                    for s in SEEDS) / len(SEEDS))
            result[f"{contrast}_{FLAGS[flag]}"] = {**bootstrap(per_item), "per_seed_pp": per_seed}
    return result


def intersections(matrix, order):
    result = {}
    for seed in SEEDS:
        for flag in (0, 1):
            a = {q for q in order if matrix[f"X1O{seed}"][q][flag]}
            b = {q for q in order if matrix[f"shuf_X1O{seed}"][q][flag]}
            result[f"{seed}_{FLAGS[flag]}"] = {"both": len(a & b), "matched_only": len(a - b),
                                               "shuffled_only": len(b - a), "neither": len(order) - len(a | b)}
    return result


def lengths(rows):
    values = sorted(int(row["tokens"]) for row in rows.values())
    def quantile(p):
        position = (len(values) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    stops = Counter(str(row.get("stop_id")) for row in rows.values())
    eos = sum(row.get("stop_id") in (1, 106) for row in rows.values())
    return {"n": len(values), "min": min(values), "q25": quantile(.25),
            "median": statistics.median(values), "q75": quantile(.75), "p90": quantile(.90),
            "p95": quantile(.95), "max": max(values), "mean": statistics.mean(values),
            "stop_ids": dict(stops), "eos_1_or_106": eos, "eos_pct": eos / len(values) * 100,
            "null_stop": sum(row.get("stop_id") is None for row in rows.values()),
            "meaning": "saved tokens/stop_id fields, no tokenizer replay"}


def write_new(path, text):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=Path("OUTBOX/x1_items_codex.json"))
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from shared")
    if sha("scripts/s5_eval_grader_v2.py") != EXPECTED_GRADER or sha("data/p6_item_order.json") != EXPECTED_ORDER:
        raise ValueError("Input hash mismatch; no analysis performed")
    j = load_json("data/p6_item_order.json")
    order = [str(q) for q in (j["indices"] if isinstance(j, dict) else j)]
    if len(order) != 600 or len(set(order)) != 600:
        raise ValueError("Expected 600 unique ordered ids")
    gold = load_rows("data/dev600_records.jsonl")
    if set(gold) != set(order):
        raise ValueError("Gold id coverage differs")
    spec = load_json("data/s0spec_x1.json")
    cc = load_json("data/x1_items_CC.json")
    local = load_json(args.local)
    generations = {label: load_rows(paths[0]) for label, paths in spec.items()}
    for source in (cc, local):
        if set(source) != set(spec):
            raise ValueError("Label coverage differs")
        for label, items in source.items():
            if set(items) != set(order):
                raise ValueError(f"Flag id coverage differs: {label}")
            if any(len(v) != 3 or any(type(x) is not int or x not in (0, 1) for x in v) for v in items.values()):
                raise ValueError("Expected binary flag triples")
    for label, rows in generations.items():
        if set(rows) != set(order):
            raise ValueError(f"Generation coverage differs: {label}")
    # Extract displayed answer text with the locked grader's own helpers.
    # Do not call answers_match, math-verify, or grade_prediction here.
    sys.path.insert(0, str(ROOT / "scripts"))
    from s5_eval_grader_v2 import _extract_boxed, extract_answer, lenient_extract
    differences = []
    directions = Counter()
    for label in spec:
        for q in order:
            c, l = cc[label][q], local[label][q]
            for i, flag in enumerate(FLAGS):
                if c[i] != l[i]:
                    directions[f"{flag}:CC_only" if c[i] else f"{flag}:local_only"] += 1
                    raw = generations[label][q]["generation"]
                    solution = gold[q]["solution"]
                    differences.append({"label": label, "id": int(q), "flag": flag,
                                        "CC": c[i], "local": l[i], "CC_flags": c, "local_flags": l,
                                        "prediction_boxed": _extract_boxed(raw),
                                        "prediction_lenient": lenient_extract(raw),
                                        "gold_extracted": _extract_boxed(solution) or extract_answer(solution) or solution.strip(),
                                        "generation_raw": raw, "gold_solution_raw": solution})
    input_paths = {Path(paths[0]) for paths in spec.values()} | {
        Path("data/s0spec_x1.json"), Path("data/p6_item_order.json"), Path("data/dev600_records.jsonl"),
        Path("data/x1_items_CC.json"), args.local,
        Path("scripts/s5_eval_grader_v2.py"), Path("scripts/codex_grade_local.py"),
        Path("scripts/s7_eval_score_p6_v36.py"), Path(__file__),
    }
    report = {
        "author": "Codex", "created_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "local_status": "DIAGNOSTIC_ONLY: interrupted Windows multiprocessing errors; not an independent symbolic equivalence result",
        "counts": {"labels": 7, "items_per_label": 600, "label_item_pairs": 4200, "flag_comparisons": 12600},
        "environment": {"python": sys.version, **{p: importlib.metadata.version(p) for p in ("sympy", "math-verify", "latex2sympy2-extended", "mpmath", "antlr4-python3-runtime")}},
        "CC": {"totals": totals(cc, order), "statistics": statistics_for(cc, order), "intersections": intersections(cc, order)},
        "local_diagnostic": {"totals": totals(local, order), "statistics": statistics_for(local, order), "intersections": intersections(local, order)},
        "lengths": {label: lengths(rows) for label, rows in generations.items()},
        "differences": differences, "difference_directions": dict(directions),
        "input_sha256": {str(p): sha(p) for p in sorted(input_paths, key=str)},
        "reference_CC_ci": {"A_lenient": [1.11, -.22, 2.44], "B_lenient": [-7.61, -10.44, -4.94]},
    }
    write_new("OUTBOX/x1_recheck_results.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# TASK-C002 フラグ差分 / 플래그 불일치", "", "작성자: Codex | " + report["created_kst"],
             "", "로컬 값은 Windows multiprocessing 오류가 있는 첫 시도의 진단용 산출물입니다. 수식 동등성 재채점의 확정 결과가 아닙니다.",
             "", "모든 원문은 x1_recheck_results.json differences 배열에 보존되어 있습니다.", "",
             "| label | id | flag | CC | local | prediction boxed | prediction lenient | gold |",
             "|---|---:|---|---:|---:|---|---|---|"]
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    for row in differences:
        lines.append("| " + " | ".join(cell(row[k]) for k in ("label", "id", "flag", "CC", "local", "prediction_boxed", "prediction_lenient", "gold_extracted")) + " |")
    write_new("OUTBOX/x1_discrepancies.md", "\n".join(lines) + "\n")
    print(json.dumps({"created_kst": report["created_kst"], "differences": len(differences),
                      "directions": dict(directions), "CC_stats": report["CC"]["statistics"],
                      "local_diagnostic_stats": report["local_diagnostic"]["statistics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

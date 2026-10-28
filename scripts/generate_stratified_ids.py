#!/usr/bin/env python3
"""Generate deterministic diagnostic IDs and dataset-bound manifests.

MATH-500 is selected round-robin across ``(subject, level)`` buckets.  GSM8K
uses a deterministic uniform stride.  Every ID file is accompanied by a JSON
manifest that binds the selection to the exact input dataset SHA-256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


GENERATOR_VERSION = "2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _math500_selection(
    data_path: Path,
    n_samples: int,
) -> Tuple[List[str], Dict[str, Any]]:
    if not data_path.is_file():
        raise FileNotFoundError(f"MATH-500 file not found: {data_path}")

    buckets: Dict[str, List[str]] = {}
    with data_path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            problem = json.loads(line)
            subject = problem.get("subject") or "unknown"
            level = int(problem.get("level") or 3)
            problem_id = problem.get("unique_id") or problem.get("id") or str(index)
            key = f"{subject}_L{level}"
            buckets.setdefault(key, []).append(str(problem_id))

    selected: List[str] = []
    selected_by_bucket: Dict[str, List[str]] = {key: [] for key in sorted(buckets)}
    offsets = {key: 0 for key in buckets}
    while len(selected) < n_samples:
        progressed = False
        for key in sorted(buckets):
            offset = offsets[key]
            if offset >= len(buckets[key]):
                continue
            problem_id = buckets[key][offset]
            offsets[key] += 1
            selected.append(problem_id)
            selected_by_bucket[key].append(problem_id)
            progressed = True
            if len(selected) >= n_samples:
                break
        if not progressed:
            break

    bucket_manifest = {
        key: {
            "available": len(buckets[key]),
            "selected": selected_by_bucket[key],
        }
        for key in sorted(buckets)
    }
    return selected, bucket_manifest


def stratify_math500(data_path: Path, n_samples: int = 20) -> List[str]:
    selected, _buckets = _math500_selection(data_path, n_samples)
    return selected


def _gsm8k_selection(
    data_path: Path,
    n_samples: int,
) -> Tuple[List[str], Dict[str, Any]]:
    if not data_path.is_file():
        raise FileNotFoundError(f"GSM8K file not found: {data_path}")

    rows: List[str] = []
    with data_path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            problem = json.loads(line)
            problem_id = (
                problem.get("unique_id")
                or problem.get("id")
                or problem.get("idx")
                or str(index)
            )
            rows.append(str(problem_id))

    stride = max(1, len(rows) // n_samples) if n_samples > 0 else 1
    selected = [rows[index] for index in range(0, len(rows), stride)[:n_samples]]
    buckets = {
        "uniform_stride": {
            "available": len(rows),
            "stride": stride,
            "selected": selected,
        }
    }
    return selected, buckets


def stratify_gsm8k(data_path: Path, n_samples: int = 20) -> List[str]:
    selected, _buckets = _gsm8k_selection(data_path, n_samples)
    return selected


def _relative_dataset_path(root: Path, data_path: Path) -> str:
    try:
        return data_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(data_path.resolve())


def write_selection(
    *,
    root: Path,
    benchmark: str,
    data_path: Path,
    ids: List[str],
    buckets: Dict[str, Any],
) -> Tuple[Path, Path]:
    ids_path = root / "data" / f"{benchmark}_stratified_ids.txt"
    manifest_path = root / "data" / f"{benchmark}_stratified_ids.manifest.json"
    ids_path.write_text("\n".join(ids), encoding="utf-8")
    manifest = {
        "dataset_path": _relative_dataset_path(root, data_path),
        "dataset_sha256": sha256_file(data_path),
        "n": len(ids),
        "buckets": buckets,
        "ids": ids,
        "generator_version": GENERATOR_VERSION,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ids_path, manifest_path


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    math_data = root / "data" / "math500" / "test.jsonl"
    math_ids, math_buckets = _math500_selection(math_data, 20)
    math_ids_path, math_manifest_path = write_selection(
        root=root,
        benchmark="math500",
        data_path=math_data,
        ids=math_ids,
        buckets=math_buckets,
    )
    print(
        f"Generated {len(math_ids)} MATH-500 IDs -> {math_ids_path}; "
        f"manifest -> {math_manifest_path}"
    )

    gsm_data = root / "data" / "gsm8k" / "test.jsonl"
    if gsm_data.is_file():
        gsm_ids, gsm_buckets = _gsm8k_selection(gsm_data, 20)
        gsm_ids_path, gsm_manifest_path = write_selection(
            root=root,
            benchmark="gsm8k",
            data_path=gsm_data,
            ids=gsm_ids,
            buckets=gsm_buckets,
        )
        print(
            f"Generated {len(gsm_ids)} GSM8K IDs -> {gsm_ids_path}; "
            f"manifest -> {gsm_manifest_path}"
        )
    else:
        gsm_dir = root / "data" / "gsm8k"
        present = (
            [path.name for path in gsm_dir.iterdir() if path.is_file()]
            if gsm_dir.is_dir()
            else []
        )
        print(f"[info] GSM8K test.jsonl not found; directory contains: {present}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

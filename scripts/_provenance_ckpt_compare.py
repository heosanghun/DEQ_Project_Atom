#!/usr/bin/env python3
"""DRAFT ONLY: exact checkpoint provenance comparison; do not run without approval.

This read-only helper compares only the frozen input-layernorm tensors at layers
0, 21, and 41 against local base and instruction-tuned reference files. It:

* loads the checkpoint on CPU with ``weights_only=True`` and ``mmap=True``;
* never retries with ``weights_only=False``;
* resolves checkpoint keys from the actual loaded mappings by suffix, without
  assuming a ``cg.`` prefix;
* checks shape and dtype before using ``torch.equal``;
* reports ``UNDETERMINED`` for missing, ambiguous, mixed, or non-exact results;
* prints JSON to stdout and does not write any file.

Proposed usage after separate server-execution approval:

    python scripts/_provenance_ckpt_compare.py \
      --checkpoint /path/to/stage1_last.pt \
      --base-model-file /path/to/gemma-4-E4B/model.safetensors \
      --it-model-file /path/to/gemma-4-E4B-it/model.safetensors \
      --manifest artifacts/provenance/provenance_manifest.json

This file is a provenance-preparation draft. Creating or syntax-checking it is
not approval to execute it against a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open


TARGET_SUFFIXES = (
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.21.input_layernorm.weight",
    "model.language_model.layers.41.input_layernorm.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-file", type=Path, required=True)
    parser.add_argument("--it-model-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def load_manifest_targets(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    indexed = {item["tensor_key"]: item for item in manifest["tensors"]}
    missing = [key for key in TARGET_SUFFIXES if key not in indexed]
    if missing:
        raise ValueError(f"manifest is missing target tensors: {missing}")
    return [indexed[key] for key in TARGET_SUFFIXES]


def load_reference_tensors(
    model_file: Path, targets: list[dict[str, Any]]
) -> dict[str, torch.Tensor]:
    """Load only the three small reference tensors, sequentially per model file."""
    result: dict[str, torch.Tensor] = {}
    with safe_open(str(model_file), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for item in targets:
            key = item["tensor_key"]
            if key not in available:
                raise KeyError(f"reference tensor not found in {model_file}: {key}")
            result[key] = handle.get_tensor(key).contiguous().cpu()
    return result


def walk_tensor_mappings(
    obj: Any, path: str = "root"
) -> Iterator[tuple[str, Mapping[Any, Any]]]:
    """Yield every nested mapping that directly contains at least one tensor."""
    if not isinstance(obj, Mapping):
        return
    if any(isinstance(value, torch.Tensor) for value in obj.values()):
        yield path, obj
    for key, value in obj.items():
        if isinstance(value, Mapping):
            yield from walk_tensor_mappings(value, f"{path}.{key}")


def key_matches(candidate: str, tensor_key: str) -> bool:
    suffixes = {tensor_key}
    if tensor_key.startswith("model."):
        suffixes.add(tensor_key[len("model.") :])
    return any(
        candidate == suffix or candidate.endswith("." + suffix)
        for suffix in suffixes
    )


def resolve_checkpoint_tensor(
    checkpoint: Any, tensor_key: str
) -> list[tuple[str, str, torch.Tensor]]:
    """Resolve by actual key suffix; return all matches so ambiguity is visible."""
    matches: list[tuple[str, str, torch.Tensor]] = []
    seen: set[tuple[str, str]] = set()
    for mapping_path, mapping in walk_tensor_mappings(checkpoint):
        for raw_key, value in mapping.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = str(raw_key)
            identity = (mapping_path, key)
            if identity not in seen and key_matches(key, tensor_key):
                seen.add(identity)
                matches.append((mapping_path, key, value))
    return matches


def tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def compare_one(
    checkpoint: Any,
    tensor_key: str,
    base_reference: torch.Tensor,
    it_reference: torch.Tensor,
) -> dict[str, Any]:
    matches = resolve_checkpoint_tensor(checkpoint, tensor_key)
    if len(matches) != 1:
        return {
            "tensor_key": tensor_key,
            "verdict": "UNDETERMINED",
            "reason": "missing checkpoint key" if not matches else "ambiguous checkpoint keys",
            "resolved_candidates": [
                {"mapping_path": path, "checkpoint_key": key}
                for path, key, _ in matches
            ],
        }

    mapping_path, checkpoint_key, checkpoint_tensor = matches[0]
    checkpoint_tensor = checkpoint_tensor.detach().cpu().contiguous()
    base_shape_dtype_match = (
        checkpoint_tensor.shape == base_reference.shape
        and checkpoint_tensor.dtype == base_reference.dtype
    )
    it_shape_dtype_match = (
        checkpoint_tensor.shape == it_reference.shape
        and checkpoint_tensor.dtype == it_reference.dtype
    )
    base_equal = bool(
        base_shape_dtype_match and torch.equal(checkpoint_tensor, base_reference)
    )
    it_equal = bool(it_shape_dtype_match and torch.equal(checkpoint_tensor, it_reference))

    if base_equal and not it_equal:
        verdict = "BASE"
        reason = "exactly equals the base reference only"
    elif it_equal and not base_equal:
        verdict = "IT"
        reason = "exactly equals the instruction-tuned reference only"
    else:
        verdict = "UNDETERMINED"
        reason = (
            "matches both references"
            if base_equal and it_equal
            else "matches neither reference exactly"
        )

    return {
        "tensor_key": tensor_key,
        "mapping_path": mapping_path,
        "checkpoint_key": checkpoint_key,
        "checkpoint_signature": tensor_signature(checkpoint_tensor),
        "base_reference_signature": tensor_signature(base_reference),
        "it_reference_signature": tensor_signature(it_reference),
        "base_shape_dtype_match": base_shape_dtype_match,
        "it_shape_dtype_match": it_shape_dtype_match,
        "base_equal": base_equal,
        "it_equal": it_equal,
        "verdict": verdict,
        "reason": reason,
    }


def overall_verdict(rows: list[dict[str, Any]]) -> str:
    verdicts = [row["verdict"] for row in rows]
    if verdicts and all(verdict == "BASE" for verdict in verdicts):
        return "BASE"
    if verdicts and all(verdict == "IT" for verdict in verdicts):
        return "IT"
    return "UNDETERMINED"


def main() -> int:
    args = parse_args()
    try:
        targets = load_manifest_targets(args.manifest)

        # Open and close each reference file separately to keep memory mappings
        # simple and load only the three small layernorm tensors.
        base_references = load_reference_tensors(args.base_model_file, targets)
        it_references = load_reference_tensors(args.it_model_file, targets)

        # Security boundary: there is deliberately no weights_only=False retry.
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "overall_verdict": "UNDETERMINED",
                    "status": "LOAD_ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "weights_only_false_fallback": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    rows = [
        compare_one(
            checkpoint,
            item["tensor_key"],
            base_references[item["tensor_key"]],
            it_references[item["tensor_key"]],
        )
        for item in targets
    ]
    report = {
        "status": "OK",
        "checkpoint": str(args.checkpoint),
        "compared_tensors": len(rows),
        "overall_verdict": overall_verdict(rows),
        "weights_only_false_fallback": False,
        "rows": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

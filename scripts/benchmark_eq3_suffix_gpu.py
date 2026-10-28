"""GPU A/B for the exact Eq.3 suffix cache and fixed-routing hot path.

This is a microbenchmark, not a headline accuracy run.  It holds model,
prompt, z, routing weights, selected modules, dtype, and device constant while
separating four changes:

  eager_fixed -> explicit_fixed -> suffix_dynamic -> suffix_fixed

The first delta isolates the rectangular additive mask representation, the
second isolates prefix-KV reuse, and the last isolates fixed routing support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.model.gemma_loader import load_gemma4_e4b


ARMS = (
    "eager_fixed",
    "explicit_fixed",
    "suffix_dynamic",
    "suffix_fixed",
)


def _sha_tensor(value: torch.Tensor) -> str:
    raw = value.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, fn):
    _sync(device)
    start = time.perf_counter_ns()
    value = fn()
    _sync(device)
    return value, (time.perf_counter_ns() - start) / 1e6


def _arm_flags(arm: str) -> tuple[bool, bool, bool]:
    if arm == "eager_fixed":
        return False, False, True
    if arm == "explicit_fixed":
        return True, False, True
    if arm == "suffix_dynamic":
        return True, True, False
    if arm == "suffix_fixed":
        return True, True, True
    raise ValueError(f"unknown arm: {arm}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expected-visible-gpus", type=int, default=1)
    ap.add_argument("--prompt", default="Find the integer value of 17+25.")
    ap.add_argument("--branches", type=int, default=3)
    ap.add_argument("--latent-tokens", type=int, default=64)
    ap.add_argument("--module-ids", default="0,11,12")
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("this benchmark requires CUDA")
    if torch.cuda.device_count() != int(args.expected_visible_gpus):
        raise RuntimeError(
            f"expected {args.expected_visible_gpus} visible GPU(s), got "
            f"{torch.cuda.device_count()}"
        )

    model, tokenizer = load_gemma4_e4b(
        model_id=args.model,
        device_map=str(device),
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    module_ids = sorted(int(x) for x in args.module_ids.split(",") if x)
    if len(module_ids) != 3:
        raise ValueError("this CTS-4nu benchmark requires exactly top-3 modules")

    torch.manual_seed(20260716)
    hidden = int(model.config.get_text_config().hidden_size)
    z_base = torch.randn(
        int(args.branches),
        int(args.latent_tokens),
        hidden,
        device=device,
        dtype=torch.float32,
    ) * 0.02
    direction = torch.randn_like(z_base)
    direction = direction / direction.norm().clamp_min(1e-12)
    z_probe = z_base + 1e-3 * direction

    alpha = torch.zeros(
        int(args.branches), 19, device=device, dtype=torch.float32
    )
    alpha[:, module_ids[0]] = 0.5
    alpha[:, module_ids[1]] = 0.3
    alpha[:, module_ids[2]] = 0.2
    support = torch.tensor(module_ids, device=device, dtype=torch.long)

    outputs: dict[str, torch.Tensor] = {}
    probes: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []

    for arm in ARMS:
        explicit, suffix, fixed = _arm_flags(arm)
        os.environ["CTS_DEQ_MAP_MODE"] = "parallel"
        os.environ["CTS_EQ3_EXPLICIT_MASKS"] = "1" if explicit else "0"
        os.environ["CTS_EQ3_SUFFIX_CACHE"] = "1" if suffix else "0"
        backbone = GemmaCTSBackbone(model, tokenizer).to(device)
        backbone.eval()
        context = backbone.encode_context(args.prompt)

        static, setup_ms = _timed(
            device,
            lambda: backbone.prepare_deq_context(z_base, context, module_ids),
        )
        suffix_reason = getattr(static, "suffix_cache_reason", None)
        if suffix and suffix_reason != "ready":
            raise RuntimeError(
                f"suffix arm {arm} silently failed to prepare: {suffix_reason}"
            )

        extra = {
            "top_k": 3,
            "eq3_static_inputs": static,
            "routing_module_ids": module_ids,
        }
        if fixed:
            extra["routing_module_ids_tensor"] = support

        with torch.inference_mode():
            for _ in range(int(args.warmups)):
                backbone.deq_step(z_base, context, alpha, extra)
            out, first_ms = _timed(
                device, lambda: backbone.deq_step(z_base, context, alpha, extra)
            )
            probe, probe_ms = _timed(
                device, lambda: backbone.deq_step(z_probe, context, alpha, extra)
            )
            samples_ms: list[float] = []
            for _ in range(int(args.repeats)):
                _, elapsed_ms = _timed(
                    device,
                    lambda: backbone.deq_step(z_base, context, alpha, extra),
                )
                samples_ms.append(elapsed_ms)

        outputs[arm] = out.detach().float().cpu()
        probes[arm] = probe.detach().float().cpu()
        records.append(
            {
                "arm": arm,
                "explicit_masks": explicit,
                "suffix_cache": suffix,
                "fixed_routing": fixed,
                "suffix_cache_reason": suffix_reason,
                "prefix_setup_ms": setup_ms,
                "first_phi_ms": first_ms,
                "probe_phi_ms": probe_ms,
                "phi_samples_ms": samples_ms,
                "phi_mean_ms": statistics.fmean(samples_ms),
                "phi_median_ms": statistics.median(samples_ms),
                "output_sha256_fp32": _sha_tensor(out),
                "probe_sha256_fp32": _sha_tensor(probe),
            }
        )

    reference = outputs["eager_fixed"]
    reference_probe = probes["eager_fixed"]
    for record in records:
        arm = str(record["arm"])
        delta = (outputs[arm] - reference).abs()
        probe_delta = (probes[arm] - reference_probe).abs()
        record["vs_eager_max_abs"] = float(delta.max().item())
        record["vs_eager_mean_abs"] = float(delta.mean().item())
        record["probe_vs_eager_max_abs"] = float(probe_delta.max().item())
        record["probe_vs_eager_mean_abs"] = float(probe_delta.mean().item())
        record["close_atol_2e_4_rtol_2e_4"] = bool(
            torch.allclose(outputs[arm], reference, atol=2e-4, rtol=2e-4)
            and torch.allclose(
                probes[arm], reference_probe, atol=2e-4, rtol=2e-4
            )
        )

    by_arm = {str(row["arm"]): row for row in records}
    eager_ms = float(by_arm["eager_fixed"]["phi_median_ms"])
    for row in records:
        row["speedup_vs_eager_median"] = eager_ms / float(
            row["phi_median_ms"]
        )
    payload = {
        "schema_version": 1,
        "diagnostic_only": True,
        "model": args.model,
        "device": str(device),
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "prompt": args.prompt,
        "branches": int(args.branches),
        "latent_tokens": int(args.latent_tokens),
        "module_ids": module_ids,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

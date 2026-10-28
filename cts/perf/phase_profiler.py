"""Low-perturbation, opt-in CUDA phase profiler.

The factory returns ``None`` unless ``CTS_PHASE_PROFILE=1``.  Consequently the
disabled hot path creates no CUDA events and performs no synchronization.
Enabled spans only enqueue event markers; ``finish()`` performs the profiler's
single synchronization and converts the recorded pairs into JSON-safe stats.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import math
import os
from statistics import median
from time import perf_counter_ns
from typing import Any, Dict, List, Optional

import torch


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _Record:
    name: str
    host_ms: float
    start_event: Optional[Any]
    end_event: Optional[Any]


class _PhaseSpan(AbstractContextManager["_PhaseSpan"]):
    def __init__(self, profiler: "CudaPhaseProfiler", name: str) -> None:
        self.profiler = profiler
        self.name = str(name)
        self.host_start_ns = 0
        self.start_event: Optional[Any] = None
        self.end_event: Optional[Any] = None
        self.nvtx_pushed = False

    def __enter__(self) -> "_PhaseSpan":
        if self.profiler._finished_report is not None:
            raise RuntimeError("cannot record a phase after profiler.finish()")
        self.host_start_ns = perf_counter_ns()
        if self.profiler.cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            if self.profiler.nvtx:
                torch.cuda.nvtx.range_push(self.name)
                self.nvtx_pushed = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.profiler.cuda_events and self.end_event is not None:
            self.end_event.record()
            if self.nvtx_pushed:
                torch.cuda.nvtx.range_pop()
        host_ms = (perf_counter_ns() - self.host_start_ns) / 1_000_000.0
        self.profiler._records.append(
            _Record(self.name, host_ms, self.start_event, self.end_event)
        )
        return False


class CudaPhaseProfiler:
    """Collect inclusive host/CUDA timings without mid-episode synchronizes."""

    def __init__(self, device: torch.device, *, nvtx: bool = False) -> None:
        self.device = torch.device(device)
        self.cuda_events = bool(
            self.device.type == "cuda" and torch.cuda.is_available()
        )
        self.nvtx = bool(nvtx and self.cuda_events)
        self._records: List[_Record] = []
        self._finished_report: Optional[Dict[str, Any]] = None

    @classmethod
    def from_env(cls, device: torch.device) -> Optional["CudaPhaseProfiler"]:
        if not _env_enabled("CTS_PHASE_PROFILE"):
            return None
        return cls(
            torch.device(device),
            nvtx=_env_enabled("CTS_PHASE_PROFILE_NVTX"),
        )

    def span(self, name: str) -> _PhaseSpan:
        return _PhaseSpan(self, name)

    @staticmethod
    def _summary(host_values: List[float], cuda_values: List[float]) -> Dict[str, Any]:
        host_sorted = sorted(float(x) for x in host_values)
        out: Dict[str, Any] = {
            "count": len(host_sorted),
            "host_ms_total": float(sum(host_sorted)),
            "host_ms_mean": float(sum(host_sorted) / max(1, len(host_sorted))),
            "host_ms_p50": float(median(host_sorted)) if host_sorted else 0.0,
            "host_ms_p95": (
                float(host_sorted[max(0, math.ceil(0.95 * len(host_sorted)) - 1)])
                if host_sorted
                else 0.0
            ),
        }
        if cuda_values:
            cuda_sorted = sorted(float(x) for x in cuda_values)
            out.update(
                {
                    "cuda_ms_total": float(sum(cuda_sorted)),
                    "cuda_ms_mean": float(sum(cuda_sorted) / len(cuda_sorted)),
                    "cuda_ms_p50": float(median(cuda_sorted)),
                    "cuda_ms_p95": float(
                        cuda_sorted[max(0, math.ceil(0.95 * len(cuda_sorted)) - 1)]
                    ),
                }
            )
        else:
            out.update(
                {
                    "cuda_ms_total": None,
                    "cuda_ms_mean": None,
                    "cuda_ms_p50": None,
                    "cuda_ms_p95": None,
                }
            )
        return out

    def finish(self) -> Dict[str, Any]:
        if self._finished_report is not None:
            return self._finished_report
        if self.cuda_events:
            torch.cuda.synchronize(self.device)

        grouped_host: Dict[str, List[float]] = {}
        grouped_cuda: Dict[str, List[float]] = {}
        for record in self._records:
            grouped_host.setdefault(record.name, []).append(record.host_ms)
            if (
                self.cuda_events
                and record.start_event is not None
                and record.end_event is not None
            ):
                grouped_cuda.setdefault(record.name, []).append(
                    float(record.start_event.elapsed_time(record.end_event))
                )

        phases = {
            name: self._summary(grouped_host[name], grouped_cuda.get(name, []))
            for name in sorted(grouped_host)
        }
        self._finished_report = {
            "schema_version": 1,
            "device": str(self.device),
            "cuda_events": bool(self.cuda_events),
            "nvtx": bool(self.nvtx),
            "phase_semantics": "inclusive",
            "phases": phases,
        }
        return self._finished_report

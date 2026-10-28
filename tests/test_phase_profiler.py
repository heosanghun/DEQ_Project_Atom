from __future__ import annotations

import json

import torch

from cts.perf.phase_profiler import CudaPhaseProfiler


def test_factory_disabled_returns_none_without_touching_cuda(monkeypatch) -> None:
    monkeypatch.delenv("CTS_PHASE_PROFILE", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled profiler touched CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert CudaPhaseProfiler.from_env(torch.device("cuda:0")) is None


def test_cpu_profile_is_json_safe_and_finish_is_idempotent(monkeypatch) -> None:
    monkeypatch.setenv("CTS_PHASE_PROFILE", "1")
    profiler = CudaPhaseProfiler.from_env(torch.device("cpu"))
    assert profiler is not None
    with profiler.span("repeat"):
        sum(range(50))
    with profiler.span("repeat"):
        sum(range(100))

    first = profiler.finish()
    second = profiler.finish()
    assert first is second
    assert first["cuda_events"] is False
    assert first["phases"]["repeat"]["count"] == 2
    assert first["phases"]["repeat"]["cuda_ms_total"] is None
    json.dumps(first)


def test_span_records_even_when_body_raises(monkeypatch) -> None:
    monkeypatch.setenv("CTS_PHASE_PROFILE", "1")
    profiler = CudaPhaseProfiler.from_env(torch.device("cpu"))
    assert profiler is not None
    try:
        with profiler.span("failure"):
            raise ValueError("expected")
    except ValueError:
        pass
    assert profiler.finish()["phases"]["failure"]["count"] == 1

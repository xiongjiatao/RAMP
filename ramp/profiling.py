"""Opt-in, semantics-neutral throughput profiling for the production entry."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

import torch


class ThroughputProfiler:
    def __init__(self, *, enabled: bool = False, device: torch.device | str = "cpu"):
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self.records: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "wall_seconds": 0.0, "cuda_seconds": 0.0}
        )
        self.transfers = {"cpu_to_gpu_bytes": 0, "gpu_to_cpu_bytes": 0}
        self.tensor_contracts: dict[str, dict[str, Any]] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        start_event = end_event = None
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = time.perf_counter()
        try:
            yield
        finally:
            wall = time.perf_counter() - started
            cuda_seconds = 0.0
            if use_cuda:
                assert start_event is not None and end_event is not None
                end_event.record()
                end_event.synchronize()
                cuda_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
            record = self.records[name]
            record["calls"] += 1
            record["wall_seconds"] += wall
            record["cuda_seconds"] += cuda_seconds

    def transfer(self, direction: str, byte_count: int) -> None:
        if self.enabled:
            self.transfers[direction] += int(byte_count)

    def tensor(self, name: str, value: torch.Tensor) -> None:
        if self.enabled and name not in self.tensor_contracts:
            self.tensor_contracts[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "bytes": int(value.numel() * value.element_size()),
            }

    def to_dict(self) -> dict[str, Any]:
        records = {}
        for name, values in sorted(self.records.items()):
            calls = int(values["calls"])
            records[name] = {
                "calls": calls,
                "wall_seconds": values["wall_seconds"],
                "cuda_enclosed_seconds": values["cuda_seconds"],
                "wall_seconds_per_call": values["wall_seconds"] / max(calls, 1),
                "cuda_enclosed_seconds_per_call": values["cuda_seconds"] / max(calls, 1),
            }
        return {
            "schema": "RAMP throughput profile v1",
            "device": str(self.device),
            "records": records,
            "transfers": dict(self.transfers),
            "tensor_contracts": self.tensor_contracts,
            "cuda_timing_note": (
                "CUDA events enclose each phase and therefore include stream-idle "
                "gaps caused by host Python work; they are not pure kernel time."
            ),
        }

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def profiled(name: str) -> Any:
    """Decorate a method whose owner may expose a profiler."""

    def decorate(function: Any) -> Any:
        @wraps(function)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            profiler = getattr(self, "profiler", None)
            if profiler is None:
                return function(self, *args, **kwargs)
            with profiler.phase(name):
                return function(self, *args, **kwargs)
        return wrapped
    return decorate


def tensor_bytes(values: Any) -> int:
    if isinstance(values, torch.Tensor):
        return int(values.numel() * values.element_size())
    if hasattr(values, "__dataclass_fields__"):
        return sum(tensor_bytes(getattr(values, name)) for name in values.__dataclass_fields__)
    return 0

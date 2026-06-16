from __future__ import annotations

import statistics
import time
from dataclasses import dataclass


@dataclass
class TimingResult:
    elapsed_s: float
    latencies_s: list[float]


def time_single(callable_obj, items: list[str]) -> TimingResult:
    latencies: list[float] = []
    start = time.perf_counter()
    for item in items:
        doc_start = time.perf_counter()
        callable_obj(item)
        latencies.append(time.perf_counter() - doc_start)
    elapsed = time.perf_counter() - start
    return TimingResult(elapsed_s=elapsed, latencies_s=latencies)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * q)
    return ordered[index]


def summarize_latencies(latencies: list[float]) -> dict[str, float]:
    return {
        "avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50) * 1000,
        "p95_latency_ms": percentile(latencies, 0.95) * 1000,
        "p99_latency_ms": percentile(latencies, 0.99) * 1000,
    }


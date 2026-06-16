from __future__ import annotations

import tracemalloc
from dataclasses import dataclass


@dataclass
class MemoryMeasurement:
    peak_bytes: int
    current_bytes: int


def measure_memory(callable_obj, *args, **kwargs) -> tuple[object, MemoryMeasurement]:
    tracemalloc.start()
    result = callable_obj(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, MemoryMeasurement(peak_bytes=peak, current_bytes=current)


"""Low-overhead stage timing for BYAKU live-frame profiling."""
from contextlib import contextmanager, nullcontext
from threading import Lock
from time import perf_counter_ns
from typing import Any, Iterator
import math
import statistics


class FrameProfiler:
    """Collect wall-clock milliseconds by named stage for one or more frames."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}
        self._metadata: dict[str, Any] = {}
        self._lock = Lock()

    def record(self, stage: str, elapsed_ms: float) -> None:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        value = float(elapsed_ms)
        if not math.isfinite(value) or value < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        with self._lock:
            self._samples.setdefault(stage, []).append(value)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = perf_counter_ns()
        try:
            yield
        finally:
            self.record(stage, (perf_counter_ns() - started) / 1_000_000)

    def add_metadata(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata key must be a non-empty string")
        with self._lock:
            self._metadata[key] = value

    def summary(self) -> dict[str, Any]:
        with self._lock:
            samples = {name: values.copy() for name, values in self._samples.items()}
            metadata = self._metadata.copy()
        summary = {
            name: {
                "count": len(values),
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
            for name, values in sorted(samples.items())
            if values
        }
        return {"stages": summary, "metadata": metadata}

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()
            self._metadata.clear()


def profile_stage(profiler: FrameProfiler | None, stage: str):
    """Return a timing context or a no-op context when measurement is disabled."""
    return profiler.measure(stage) if profiler is not None else nullcontext()

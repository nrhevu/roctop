from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .formatting import clamp_percent
from .models import Snapshot


@dataclass(frozen=True, slots=True)
class CpuTimes:
    idle: int
    total: int


@dataclass(frozen=True, slots=True)
class GpuMetricSample:
    index: int
    utilization_percent: float | None
    memory_percent: float | None


@dataclass(frozen=True, slots=True)
class MetricSample:
    timestamp: datetime
    avg_cpu_percent: float | None
    avg_mem_percent: float | None
    avg_gpu_percent: float | None
    avg_gpu_mem_percent: float | None
    gpu_metrics: tuple[GpuMetricSample, ...] = ()


class MetricsHistory:
    def __init__(
        self,
        max_samples: int = 120,
        stat_path: str | Path = "/proc/stat",
        meminfo_path: str | Path = "/proc/meminfo",
    ) -> None:
        self.max_samples = max(1, int(max_samples))
        self.stat_path = Path(stat_path)
        self.meminfo_path = Path(meminfo_path)
        self._samples: deque[MetricSample] = deque(maxlen=self.max_samples)
        self._previous_cpu_times: CpuTimes | None = None
        self._lock = threading.RLock()

    @property
    def samples(self) -> tuple[MetricSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def append_sample(self, sample: MetricSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def prime_cpu(self) -> None:
        cpu_times = read_cpu_times(self.stat_path)
        if cpu_times is not None:
            with self._lock:
                self._previous_cpu_times = cpu_times

    def add_snapshot(self, snapshot: Snapshot) -> MetricSample:
        with self._lock:
            cpu_times = read_cpu_times(self.stat_path)
            cpu_percent = cpu_percent_from_times(self._previous_cpu_times, cpu_times)
            if cpu_times is not None:
                self._previous_cpu_times = cpu_times

            sample = MetricSample(
                timestamp=snapshot.timestamp,
                avg_cpu_percent=cpu_percent,
                avg_mem_percent=read_mem_percent(self.meminfo_path),
                avg_gpu_percent=average_gpu_percent(snapshot),
                avg_gpu_mem_percent=average_gpu_mem_percent(snapshot),
                gpu_metrics=gpu_metric_samples(snapshot),
            )
            self._samples.append(sample)
            return sample


def read_cpu_times(path: str | Path = "/proc/stat") -> CpuTimes | None:
    try:
        return parse_cpu_times(Path(path).read_text())
    except OSError:
        return None


def parse_cpu_times(text: str) -> CpuTimes | None:
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        parts = line.split()[1:]
        if len(parts) < 4:
            return None
        try:
            values = [int(part) for part in parts]
        except ValueError:
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values[:8])
        if total <= 0:
            return None
        return CpuTimes(idle=idle, total=total)
    return None


def cpu_percent_from_times(previous: CpuTimes | None, current: CpuTimes | None) -> float | None:
    if previous is None or current is None:
        return None
    total_delta = current.total - previous.total
    idle_delta = current.idle - previous.idle
    if total_delta <= 0 or idle_delta < 0:
        return None
    busy_delta = max(0, total_delta - idle_delta)
    return clamp_percent(busy_delta / total_delta * 100.0)


def read_mem_percent(path: str | Path = "/proc/meminfo") -> float | None:
    try:
        return parse_mem_percent(Path(path).read_text())
    except OSError:
        return None


def parse_mem_percent(text: str) -> float | None:
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].rstrip(":")
        try:
            values[key] = int(parts[1])
        except ValueError:
            continue

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or total <= 0 or available is None or available < 0:
        return None
    return clamp_percent((total - available) / total * 100.0)


def average_gpu_percent(snapshot: Snapshot) -> float | None:
    return average_percent(gpu.utilization_percent for gpu in snapshot.gpus)


def average_gpu_mem_percent(snapshot: Snapshot) -> float | None:
    return average_percent(gpu.memory_percent for gpu in snapshot.gpus)


def gpu_metric_samples(snapshot: Snapshot) -> tuple[GpuMetricSample, ...]:
    return tuple(
        GpuMetricSample(
            index=gpu.index,
            utilization_percent=clamp_percent(gpu.utilization_percent),
            memory_percent=clamp_percent(gpu.memory_percent),
        )
        for gpu in sorted(snapshot.gpus, key=lambda gpu: gpu.index)
    )


def average_percent(values: Iterable[float | int | None]) -> float | None:
    percentages = [clamp_percent(value) for value in values if value is not None]
    if not percentages:
        return None
    return sum(percentages) / len(percentages)

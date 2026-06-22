from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class GpuInfo:
    index: int
    name: str = "AMD GPU"
    gfx_version: str = ""
    temperature_c: float | None = None
    fan_percent: float | None = None
    fan_rpm: int | None = None
    power_w: float | None = None
    sclk_mhz: int | None = None
    mclk_mhz: int | None = None
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0
    utilization_percent: float = 0.0

    @property
    def memory_percent(self) -> float:
        if self.memory_total_bytes <= 0:
            return 0.0
        return min(100.0, max(0.0, self.memory_used_bytes / self.memory_total_bytes * 100.0))


@dataclass(slots=True)
class ProcessInfo:
    gpu_index: int | None
    pid: int
    name: str = ""
    user: str = ""
    cpu_percent: float | None = None
    host_mem_percent: float | None = None
    elapsed: str = ""
    command: str = ""
    args: str = ""
    gpu_memory_bytes: int = 0
    gpu_memory_percent: float = 0.0
    cu_occupancy: int | None = None


@dataclass(slots=True)
class Snapshot:
    timestamp: datetime
    driver_version: str = ""
    gpus: list[GpuInfo] = field(default_factory=list)
    processes: list[ProcessInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat(timespec="seconds")
        return data

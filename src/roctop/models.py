from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class GpuInfo:
    index: int
    name: str = "AMD GPU"
    guid: str = ""
    gpu_type: str = ""
    gfx_version: str = ""
    vendor: str = ""
    vbios_version: str = ""
    pcie_bus: str = ""
    max_power_w: float | None = None
    performance_level: str = ""
    throttle_status: str = ""
    voltage_mv: float | None = None
    unique_id: str = ""
    sku: str = ""
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
    ppid: int | None = None


@dataclass(slots=True)
class ProcessDetailInfo:
    pid: int
    state: str = ""
    threads: int | None = None
    vm_rss_kib: int | None = None
    vm_size_kib: int | None = None
    vm_hwm_kib: int | None = None
    cpu_allowed_list: str = ""
    voluntary_ctxt_switches: int | None = None
    nonvoluntary_ctxt_switches: int | None = None
    cmdline: str = ""
    cwd: str = ""
    exe: str = ""
    error: str = ""


@dataclass(slots=True)
class Snapshot:
    timestamp: datetime
    node_name: str = ""
    driver_version: str = ""
    gpus: list[GpuInfo] = field(default_factory=list)
    processes: list[ProcessInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    process_ancestors: list[ProcessInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat(timespec="seconds")
        return data

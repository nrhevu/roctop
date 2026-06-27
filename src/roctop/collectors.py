from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from .formatting import clamp_percent, parse_int, parse_number
from .models import GpuInfo, ProcessDetailInfo, ProcessInfo, Snapshot
from .profiling import profile_span


ROCM_SMI_ARGS = [
    "rocm-smi",
    "--showid",
    "--showproductname",
    "--showvbios",
    "--showserial",
    "--showuniqueid",
    "--showbus",
    "--showuse",
    "--showmeminfo",
    "vram",
    "--showtemp",
    "--showfan",
    "--showclocks",
    "--showpower",
    "--showmaxpower",
    "--showperflevel",
    "--showvoltage",
    "--showmetrics",
    "--showpids",
    "--showdriverversion",
    "--json",
]

AMD_SMI_PROCESS_ARGS = ["amd-smi", "process", "-G", "--json"]
AMD_SMI_GPU_DETAIL_ARGS = (
    ["amd-smi", "static", "--json"],
    ["amd-smi", "metric", "--json"],
)
ROCM_SMI_TIMEOUT_SECONDS = 15.0
AMD_SMI_PROCESS_TIMEOUT_SECONDS = 1.5
AMD_SMI_PROCESS_BACKOFF_SECONDS = 10.0
AMD_SMI_GPU_DETAIL_TIMEOUT_SECONDS = 1.5
AMD_SMI_GPU_DETAIL_BACKOFF_SECONDS = 10.0
PS_TIMEOUT_SECONDS = 1.0
PS_CACHE_TTL_SECONDS = 2.0
AMD_PCI_VENDOR_ID = "1002"
PCI_IDS_PATHS = (
    Path("/usr/share/misc/pci.ids"),
    Path("/usr/share/hwdata/pci.ids"),
)
GPU_MODEL_BY_DEVICE_ID = {
    "0x738c": "AMD Instinct MI100",
    "0x738e": "AMD Instinct MI100",
    "0x7408": "AMD Instinct MI250X",
    "0x740c": "AMD Instinct MI200 Series",
    "0x740f": "AMD Instinct MI210",
    "0x74a0": "AMD Instinct MI300A",
    "0x74a1": "AMD Instinct MI300X",
    "0x75b0": "AMD Instinct MI350X",
}
GPU_MODEL_BY_ARCHITECTURE = {
    "gfx908": "AMD Instinct MI100",
    "gfx90a": "AMD Instinct MI200 Series",
    "gfx940": "AMD Instinct MI300 Series",
    "gfx941": "AMD Instinct MI300 Series",
    "gfx942": "AMD Instinct MI300 Series",
    "gfx950": "AMD Instinct MI350 Series",
}
MEMORY_UNIT_MULTIPLIERS = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kb": 1000,
    "kbyte": 1000,
    "kbytes": 1000,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
    "mb": 1000**2,
    "mbyte": 1000**2,
    "mbytes": 1000**2,
    "gb": 1000**3,
    "gbyte": 1000**3,
    "gbytes": 1000**3,
    "tb": 1000**4,
    "tbyte": 1000**4,
    "tbytes": 1000**4,
}
INSTINCT_MODEL_BY_TOKEN = {
    "mi6": "AMD Radeon Instinct MI6",
    "mi8": "AMD Radeon Instinct MI8",
    "mi25": "AMD Radeon Instinct MI25",
    "mi50": "AMD Radeon Instinct MI50",
    "mi60": "AMD Radeon Instinct MI60",
    "mi100": "AMD Instinct MI100",
    "mi210": "AMD Instinct MI210",
    "mi250": "AMD Instinct MI250",
    "mi250x": "AMD Instinct MI250X",
    "mi300a": "AMD Instinct MI300A",
    "mi300x": "AMD Instinct MI300X",
    "mi325x": "AMD Instinct MI325X",
    "mi350p": "AMD Instinct MI350P",
    "mi350x": "AMD Instinct MI350X",
    "mi355x": "AMD Instinct MI355X",
}
INSTINCT_SERIES_BY_PREFIX = {
    "210": "AMD Instinct MI200 Series",
    "250": "AMD Instinct MI200 Series",
    "300": "AMD Instinct MI300 Series",
    "325": "AMD Instinct MI300 Series",
    "350": "AMD Instinct MI350 Series",
    "355": "AMD Instinct MI350 Series",
}

_amd_smi_process_backoff_until = 0.0
_amd_smi_gpu_detail_backoff_until = 0.0
_ps_row_cache: dict[int, tuple[float, dict[str, str]]] = {}
PROC_ROOT = Path("/proc")


@dataclass(slots=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(slots=True)
class AmdSmiProcessCommand:
    result: CommandResult | None = None
    error: CollectionError | None = None


@dataclass(slots=True)
class AmdSmiGpuDetailCommand:
    args: list[str]
    result: CommandResult | None = None
    error: CollectionError | None = None


class CollectionError(RuntimeError):
    pass


class CommandTimeout(CollectionError):
    pass


class CommandInterrupted(CollectionError):
    pass


def run_command(args: list[str], timeout: float = 5.0) -> CommandResult:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise CollectionError(f"Command not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CommandTimeout(f"Command timed out: {' '.join(shlex.quote(a) for a in args)}") from exc

    return CommandResult(
        args=args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def collect_snapshot(now: datetime | None = None) -> Snapshot:
    with profile_span("collection"):
        return _collect_snapshot(now)


def _collect_snapshot(now: datetime | None = None) -> Snapshot:
    now = now or datetime.now()
    warnings: list[str] = []

    rocm_result, amd_gpu_detail_commands, amd_process_command = collect_smi_command_results()
    warnings.extend(_warnings_from_result(rocm_result))
    if command_was_interrupted(rocm_result):
        raise CommandInterrupted(_command_failure_message(rocm_result))
    if rocm_result.returncode != 0 and not rocm_result.stdout.strip():
        raise CollectionError(_command_failure_message(rocm_result))

    rocm_data = load_json_from_text(rocm_result.stdout)
    gpus, rocm_processes, driver_version = parse_rocm_smi_json(rocm_data)
    amd_smi_driver_version = merge_amd_smi_gpu_detail_commands(gpus, amd_gpu_detail_commands, warnings)
    if not driver_version:
        driver_version = amd_smi_driver_version

    process_rows: list[ProcessInfo] = []
    if amd_process_command is not None:
        try:
            if amd_process_command.error is not None:
                raise amd_process_command.error
            if amd_process_command.result is None:
                raise CollectionError("amd-smi process did not return a result")
            amd_result = amd_process_command.result
            warnings.extend(_warnings_from_result(amd_result))
            if command_was_interrupted(amd_result):
                raise CommandInterrupted(_command_failure_message(amd_result))
            if amd_result.returncode != 0:
                raise CollectionError(_command_failure_message(amd_result))
            if amd_result.stdout.strip():
                process_rows = parse_amd_smi_process_json(load_json_from_text(amd_result.stdout), gpus)
            record_amd_smi_process_success()
        except CommandTimeout:
            record_amd_smi_process_failure()
        except (CollectionError, json.JSONDecodeError, ValueError) as exc:
            record_amd_smi_process_failure()
            warnings.append(f"amd-smi process unavailable: {exc}")

    if not process_rows:
        process_rows = rocm_processes

    process_rows = merge_process_sources(process_rows, rocm_processes)
    enrich_processes_with_ps(process_rows)
    process_ancestors = collect_process_ancestors(process_rows)

    return Snapshot(
        timestamp=now,
        node_name=platform.node().strip(),
        driver_version=driver_version,
        gpus=sorted(gpus, key=lambda gpu: gpu.index),
        processes=sorted(
            process_rows,
            key=lambda proc: (
                proc.gpu_index is None,
                proc.gpu_index if proc.gpu_index is not None else 9999,
                -proc.gpu_memory_bytes,
                proc.pid,
            ),
        ),
        process_ancestors=process_ancestors,
        warnings=dedupe_preserving_order(warnings),
    )


def collect_smi_command_results() -> tuple[CommandResult, list[AmdSmiGpuDetailCommand], AmdSmiProcessCommand | None]:
    run_gpu_details = should_run_amd_smi_gpu_detail()
    run_processes = should_run_amd_smi_process()
    if not run_gpu_details and not run_processes:
        return run_command(ROCM_SMI_ARGS, timeout=ROCM_SMI_TIMEOUT_SECONDS), [], None

    worker_count = 1 + (len(AMD_SMI_GPU_DETAIL_ARGS) if run_gpu_details else 0) + int(run_processes)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        rocm_future = executor.submit(run_command, ROCM_SMI_ARGS, timeout=ROCM_SMI_TIMEOUT_SECONDS)
        gpu_detail_futures = []
        if run_gpu_details:
            gpu_detail_futures = [
                (args, executor.submit(run_command, args, timeout=AMD_SMI_GPU_DETAIL_TIMEOUT_SECONDS))
                for args in AMD_SMI_GPU_DETAIL_ARGS
            ]
        amd_future = None
        if run_processes:
            amd_future = executor.submit(run_command, AMD_SMI_PROCESS_ARGS, timeout=AMD_SMI_PROCESS_TIMEOUT_SECONDS)

        rocm_result = rocm_future.result()
        gpu_detail_commands: list[AmdSmiGpuDetailCommand] = []
        for args, future in gpu_detail_futures:
            try:
                result = future.result()
            except CollectionError as exc:
                gpu_detail_commands.append(AmdSmiGpuDetailCommand(args=args, error=exc))
            else:
                gpu_detail_commands.append(AmdSmiGpuDetailCommand(args=args, result=result))
        amd_command = None
        if amd_future is not None:
            try:
                amd_result = amd_future.result()
            except CollectionError as exc:
                amd_command = AmdSmiProcessCommand(error=exc)
            else:
                amd_command = AmdSmiProcessCommand(result=amd_result)

    return rocm_result, gpu_detail_commands, amd_command


def should_run_amd_smi_process() -> bool:
    return time.monotonic() >= _amd_smi_process_backoff_until


def should_run_amd_smi_gpu_detail() -> bool:
    return time.monotonic() >= _amd_smi_gpu_detail_backoff_until


def record_amd_smi_process_failure() -> None:
    global _amd_smi_process_backoff_until
    _amd_smi_process_backoff_until = time.monotonic() + AMD_SMI_PROCESS_BACKOFF_SECONDS


def record_amd_smi_process_success() -> None:
    global _amd_smi_process_backoff_until
    _amd_smi_process_backoff_until = 0.0


def record_amd_smi_gpu_detail_failure() -> None:
    global _amd_smi_gpu_detail_backoff_until
    _amd_smi_gpu_detail_backoff_until = time.monotonic() + AMD_SMI_GPU_DETAIL_BACKOFF_SECONDS


def record_amd_smi_gpu_detail_success() -> None:
    global _amd_smi_gpu_detail_backoff_until
    _amd_smi_gpu_detail_backoff_until = 0.0


def parse_rocm_smi_json(data: dict[str, Any]) -> tuple[list[GpuInfo], list[ProcessInfo], str]:
    gpus: list[GpuInfo] = []
    processes: list[ProcessInfo] = []
    driver_version = ""

    for key, value in data.items():
        if key == "system" and isinstance(value, dict):
            driver_version = str(value.get("Driver version", "") or "")
            processes.extend(parse_rocm_system_processes(value))
            continue

        if not key.startswith("card") or not isinstance(value, dict):
            continue

        match = re.search(r"\d+", key)
        if not match:
            continue
        index = int(match.group(0))
        name = first_non_empty(
            value.get("Card Series"),
            value.get("Card SKU"),
            value.get("Card Model"),
            "AMD GPU",
        )
        gfx_version = str(value.get("GFX Version", "") or "")
        gpus.append(
            GpuInfo(
                index=index,
                name=name,
                guid=first_non_empty(value.get("GUID")),
                gpu_type=read_gpu_model(value),
                gfx_version=gfx_version,
                vendor=first_non_empty(
                    value.get("Card Vendor"),
                    value.get("Card vendor"),
                    value.get("Vendor"),
                    value.get("Device Vendor"),
                ),
                vbios_version=first_non_empty(
                    value.get("VBIOS Version"),
                    value.get("VBIOS version"),
                    value.get("vbios_version"),
                ),
                pcie_bus=first_non_empty(
                    value.get("PCIe Bus"),
                    value.get("PCIe BDF"),
                    value.get("PCIe Bus ID"),
                    value.get("PCI Bus"),
                    value.get("PCI BDF"),
                    value.get("Bus"),
                ),
                max_power_w=parse_optional_float(
                    first_non_empty(
                        value.get("Max Socket Graphics Package Power (W)"),
                        value.get("Max Graphics Package Power (W)"),
                        value.get("Max Power (W)"),
                        value.get("Power Cap (W)"),
                    )
                ),
                performance_level=first_non_empty(
                    value.get("Performance Level"),
                    value.get("Performance level"),
                    value.get("perf_level"),
                    value.get("Perf"),
                ),
                throttle_status=first_non_empty(
                    value.get("Throttling Status"),
                    value.get("Throttle Status"),
                    value.get("Throttle"),
                    value.get("throttle_status"),
                    value.get("indep_throttle_status"),
                ),
                voltage_mv=parse_optional_float(
                    first_non_empty(
                        value.get("Voltage (mV)"),
                        value.get("voltage_gfx (mV)"),
                        value.get("VDDGFX (mV)"),
                        value.get("vddgfx_voltage (mV)"),
                        value.get("Voltage"),
                    )
                ),
                unique_id=first_non_empty(
                    value.get("Unique ID"),
                    value.get("Unique Id"),
                    value.get("GPU Unique ID"),
                ),
                sku=first_non_empty(value.get("Card SKU"), value.get("SKU")),
                temperature_c=parse_optional_float(
                    first_non_empty(
                        value.get("Temperature (Sensor junction) (C)"),
                        value.get("Temperature (Sensor edge) (C)"),
                        value.get("temperature_hotspot (C)"),
                    )
                ),
                fan_percent=parse_fan_percent(value),
                fan_rpm=parse_fan_rpm(value),
                power_w=parse_optional_float(
                    first_non_empty(
                        value.get("Current Socket Graphics Package Power (W)"),
                        value.get("Average Graphics Package Power (W)"),
                        value.get("average_socket_power (W)"),
                        value.get("current_socket_power (W)"),
                    )
                ),
                sclk_mhz=parse_clock_mhz(
                    first_non_empty(
                        value.get("sclk clock speed:"),
                        value.get("current_gfxclk (MHz)"),
                    )
                ),
                mclk_mhz=parse_clock_mhz(
                    first_non_empty(
                        value.get("mclk clock speed:"),
                        value.get("current_uclk (MHz)"),
                    )
                ),
                memory_used_bytes=parse_int(value.get("VRAM Total Used Memory (B)")),
                memory_total_bytes=parse_int(value.get("VRAM Total Memory (B)")),
                utilization_percent=clamp_percent(value.get("GPU use (%)")),
            )
        )

    apply_gpu_memory_percent(processes, gpus)
    return gpus, processes, driver_version


def parse_rocm_system_processes(system: dict[str, Any]) -> list[ProcessInfo]:
    processes: list[ProcessInfo] = []
    for key, raw_value in system.items():
        if not key.startswith("PID"):
            continue
        pid = parse_int(key.removeprefix("PID"), default=-1)
        if pid < 0:
            continue
        parts = [part.strip() for part in str(raw_value).split(",")]
        name = parts[0] if parts else ""
        gpu_index = parse_int(parts[1] if len(parts) > 1 else None, default=-1)
        gpu_memory = parse_int(parts[2] if len(parts) > 2 else 0)
        processes.append(
            ProcessInfo(
                gpu_index=gpu_index if gpu_index >= 0 else None,
                pid=pid,
                name=name,
                command=name,
                gpu_memory_bytes=gpu_memory,
            )
        )
    return processes


def merge_amd_smi_gpu_detail_commands(
    gpus: list[GpuInfo],
    commands: list[AmdSmiGpuDetailCommand],
    warnings: list[str],
) -> str:
    if not commands:
        return ""

    detail_gpus: list[GpuInfo] = []
    driver_version = ""
    had_failure = False
    for command in commands:
        try:
            if command.error is not None:
                raise command.error
            if command.result is None:
                raise CollectionError(f"{amd_smi_command_label(command.args)} did not return a result")
            result = command.result
            warnings.extend(_warnings_from_result(result))
            if command_was_interrupted(result):
                raise CommandInterrupted(_command_failure_message(result))
            if result.returncode != 0:
                raise CollectionError(_command_failure_message(result))
            if result.stdout.strip():
                data = load_json_from_text(result.stdout)
                driver_version = first_non_empty(driver_version, amd_smi_driver_version(data))
                detail_gpus.extend(parse_amd_smi_gpu_json(data))
        except (CollectionError, json.JSONDecodeError, ValueError) as exc:
            had_failure = True
            warnings.append(f"{amd_smi_command_label(command.args)} unavailable: {exc}")

    if detail_gpus and gpus:
        merge_amd_smi_gpu_details(gpus, detail_gpus)
    if had_failure:
        record_amd_smi_gpu_detail_failure()
    else:
        record_amd_smi_gpu_detail_success()
    return driver_version


def amd_smi_command_label(args: list[str]) -> str:
    return " ".join(args[:2]) if len(args) >= 2 else "amd-smi"


def parse_amd_smi_gpu_json(data: Any) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for index, entry in iter_amd_smi_gpu_entries(data):
        fields = flatten_amd_smi_fields(entry)
        if not fields:
            continue
        model = normalize_gpu_model(
            first_non_empty(
                amd_smi_text_field(fields, ("model", "model_name", "product_name", "market_name")),
                amd_smi_text_field(fields, ("card_model", "device_name")),
            )
        )
        device_id = normalize_device_id(amd_smi_text_field(fields, ("device_id", "asic_id")))
        if not model and device_id:
            model = GPU_MODEL_BY_DEVICE_ID.get(device_id, "") or load_amd_pci_models().get(device_id, "")
        gpus.append(
            GpuInfo(
                index=index,
                name=amd_smi_text_field(fields, ("market_name", "product_name", "card_name", "card_series")),
                guid=amd_smi_text_field(fields, ("guid",)),
                gpu_type=model,
                gfx_version=amd_smi_text_field(fields, ("gfx_version", "gfxip", "gfx_ip", "gfx")),
                vendor=amd_smi_text_field(fields, ("vendor_name", "vendor")),
                vbios_version=amd_smi_text_field(fields, ("vbios_version", "vbios")),
                pcie_bus=amd_smi_text_field(fields, ("bdf", "pci_bdf", "pcie_bdf", "pci_bus", "pcie_bus", "bus_id")),
                max_power_w=amd_smi_float_field(
                    fields,
                    ("max_power", "max_power_w", "power_cap", "power_limit", "max_socket_power"),
                ),
                performance_level=amd_smi_text_field(fields, ("performance_level", "perf_level", "perf")),
                throttle_status=amd_smi_text_field(
                    fields,
                    ("throttle_status", "indep_throttle_status", "throttling_status", "throttle"),
                ),
                voltage_mv=amd_smi_float_field(fields, ("voltage_mv", "voltage_gfx", "gfx_voltage", "vddgfx", "voltage")),
                unique_id=amd_smi_text_field(
                    fields,
                    ("unique_id", "gpu_unique_id", "uuid", "serial_number", "asic_serial"),
                ),
                sku=amd_smi_text_field(fields, ("sku", "card_sku", "product_sku")),
                temperature_c=amd_smi_float_field(
                    fields,
                    ("temperature_c", "junction_temperature", "hotspot_temperature", "edge_temperature", "temperature"),
                ),
                fan_percent=amd_smi_float_field(fields, ("fan_percent", "fan_speed_percent")),
                fan_rpm=amd_smi_int_field(fields, ("fan_rpm", "fan_speed_rpm", "current_fan_speed")),
                power_w=amd_smi_float_field(
                    fields,
                    ("average_socket_power", "current_socket_power", "socket_power", "power_usage"),
                ),
                sclk_mhz=amd_smi_clock_field(fields, ("current_gfxclk", "gfxclk", "sclk", "gfx_clock")),
                mclk_mhz=amd_smi_clock_field(fields, ("current_uclk", "uclk", "mclk", "memory_clock")),
                memory_used_bytes=amd_smi_memory_field(
                    fields,
                    ("vram_used", "used_vram", "memory_used", "used_memory"),
                ),
                memory_total_bytes=amd_smi_memory_field(
                    fields,
                    ("vram_total", "total_vram", "memory_total", "total_memory"),
                ),
                utilization_percent=clamp_percent(
                    amd_smi_float_field(fields, ("gpu_utilization", "gfx_activity", "gpu_use", "utilization"))
                ),
            )
        )
    return gpus


def amd_smi_driver_version(data: Any) -> str:
    return amd_smi_text_field(flatten_amd_smi_fields(data), ("driver_version", "driver"))


def merge_amd_smi_gpu_details(gpus: list[GpuInfo], details: list[GpuInfo]) -> None:
    details_by_index: dict[int, GpuInfo] = {}
    for detail in details:
        merged = details_by_index.get(detail.index)
        if merged is None:
            details_by_index[detail.index] = detail
        else:
            fill_missing_gpu_detail(merged, detail)

    for gpu in gpus:
        detail = details_by_index.get(gpu.index)
        if detail is not None:
            fill_missing_gpu_detail(gpu, detail)


def fill_missing_gpu_detail(target: GpuInfo, source: GpuInfo) -> None:
    fill_gpu_text_field(target, source, "name", replace_generic=True)
    for field_name in (
        "guid",
        "gpu_type",
        "gfx_version",
        "vendor",
        "vbios_version",
        "pcie_bus",
        "performance_level",
        "throttle_status",
        "unique_id",
        "sku",
    ):
        fill_gpu_text_field(target, source, field_name)
    for field_name in ("max_power_w", "voltage_mv", "temperature_c", "fan_percent", "power_w"):
        fill_gpu_optional_field(target, source, field_name)
    for field_name in ("fan_rpm", "sclk_mhz", "mclk_mhz"):
        fill_gpu_optional_field(target, source, field_name)
    if target.memory_total_bytes <= 0 and source.memory_total_bytes > 0:
        target.memory_total_bytes = source.memory_total_bytes
    if target.memory_used_bytes <= 0 and source.memory_used_bytes > 0:
        target.memory_used_bytes = source.memory_used_bytes


def fill_gpu_text_field(target: GpuInfo, source: GpuInfo, field_name: str, replace_generic: bool = False) -> None:
    source_value = first_non_empty(getattr(source, field_name))
    if not source_value:
        return
    target_value = first_non_empty(getattr(target, field_name))
    if target_value and not (replace_generic and is_generic_gpu_name(target_value)):
        return
    setattr(target, field_name, source_value)


def is_generic_gpu_name(value: str) -> bool:
    return value == "AMD GPU" or re.fullmatch(r"0x[0-9a-f]+", value, flags=re.IGNORECASE) is not None


def fill_gpu_optional_field(target: GpuInfo, source: GpuInfo, field_name: str) -> None:
    if getattr(target, field_name) is not None:
        return
    source_value = getattr(source, field_name)
    if source_value is not None:
        setattr(target, field_name, source_value)


def iter_amd_smi_gpu_entries(data: Any, fallback_index: int | None = None):
    if isinstance(data, list):
        for index, item in enumerate(data):
            yield from iter_amd_smi_gpu_entries(item, index)
        return

    if not isinstance(data, dict):
        return

    index = amd_smi_gpu_index(data, fallback_index)
    fields = flatten_amd_smi_fields(data)
    if index is not None and amd_smi_fields_have_gpu_detail(fields):
        yield index, data
        return

    for key, value in data.items():
        if not isinstance(value, (dict, list)):
            continue
        key_index = amd_smi_gpu_index_from_text(key)
        yield from iter_amd_smi_gpu_entries(value, key_index)


def amd_smi_gpu_index(data: dict[str, Any], fallback_index: int | None = None) -> int | None:
    for key, value in data.items():
        if normalize_amd_smi_key(key) in ("gpu", "gpu_id", "gpu_index", "card", "card_id"):
            index = amd_smi_gpu_index_from_text(value)
            if index is not None:
                return index
    return fallback_index


def amd_smi_gpu_index_from_text(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def amd_smi_fields_have_gpu_detail(fields: list[tuple[str, Any]]) -> bool:
    detail_aliases = (
        "market_name",
        "product_name",
        "vendor_name",
        "vbios_version",
        "bdf",
        "max_power",
        "performance_level",
        "throttle_status",
        "voltage",
        "memory_total",
        "gpu_utilization",
    )
    return any(amd_smi_key_matches_alias(key, alias) for key, _value in fields for alias in detail_aliases)


def flatten_amd_smi_fields(data: Any, path: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    fields: list[tuple[str, Any]] = []
    if path:
        fields.append((".".join(path), data))
    if isinstance(data, dict):
        for key, value in data.items():
            fields.extend(flatten_amd_smi_fields(value, (*path, normalize_amd_smi_key(key))))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            fields.extend(flatten_amd_smi_fields(value, (*path, str(index))))
    return fields


def amd_smi_text_field(fields: list[tuple[str, Any]], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        for key, value in fields:
            if isinstance(value, (dict, list)):
                continue
            if not amd_smi_key_matches_alias(key, alias):
                continue
            text = first_non_empty(value)
            if text:
                return text
    return ""


def amd_smi_float_field(fields: list[tuple[str, Any]], aliases: tuple[str, ...]) -> float | None:
    value = amd_smi_raw_field(fields, aliases)
    if isinstance(value, dict):
        value = value.get("value")
    return parse_optional_float(value)


def amd_smi_int_field(fields: list[tuple[str, Any]], aliases: tuple[str, ...]) -> int | None:
    value = amd_smi_raw_field(fields, aliases)
    if isinstance(value, dict):
        value = value.get("value")
    parsed = parse_optional_float(value)
    return int(round(parsed)) if parsed is not None else None


def amd_smi_clock_field(fields: list[tuple[str, Any]], aliases: tuple[str, ...]) -> int | None:
    value = amd_smi_raw_field(fields, aliases)
    if isinstance(value, dict):
        value = value.get("value")
    return parse_clock_mhz(value)


def amd_smi_memory_field(fields: list[tuple[str, Any]], aliases: tuple[str, ...]) -> int:
    value = amd_smi_raw_field(fields, aliases)
    return parse_memory_bytes_field(value) if value is not None else 0


def amd_smi_raw_field(fields: list[tuple[str, Any]], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        for key, value in fields:
            if amd_smi_key_matches_alias(key, alias):
                return value
    return None


def amd_smi_key_matches_alias(key: str, alias: str) -> bool:
    normalized_alias = normalize_amd_smi_key(alias)
    parts = [part for part in key.split(".") if part]
    if not parts:
        return False
    if parts[-1] == normalized_alias:
        return True
    compact = "_".join(parts)
    return compact == normalized_alias or compact.endswith(f"_{normalized_alias}")


def normalize_amd_smi_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def parse_amd_smi_process_json(data: list[dict[str, Any]], gpus: list[GpuInfo]) -> list[ProcessInfo]:
    if not isinstance(data, list):
        return []

    total_by_gpu = {gpu.index: gpu.memory_total_bytes for gpu in gpus}
    processes: list[ProcessInfo] = []
    for gpu_entry in data:
        if not isinstance(gpu_entry, dict):
            continue
        gpu_index = parse_int(gpu_entry.get("gpu"), default=-1)
        if gpu_index < 0:
            continue
        process_list = gpu_entry.get("process_list", []) or []
        if not isinstance(process_list, list):
            continue
        for process_entry in process_list:
            if not isinstance(process_entry, dict):
                continue
            info = process_entry.get("process_info", {})
            if not isinstance(info, dict):
                continue
            pid = parse_int(info.get("pid"), default=-1)
            if pid < 0:
                continue
            gpu_memory = parse_memory_bytes_field(info.get("mem_usage"))
            if gpu_memory <= 0:
                memory_usage = info.get("memory_usage", {})
                if isinstance(memory_usage, dict):
                    gpu_memory = parse_memory_bytes_field(memory_usage.get("vram_mem", {}))
            if gpu_memory <= 0:
                continue
            total = total_by_gpu.get(gpu_index, 0)
            memory_percent = gpu_memory / total * 100.0 if total > 0 else 0.0
            name = str(info.get("name") or "")
            if name.upper() == "N/A":
                name = ""
            processes.append(
                ProcessInfo(
                    gpu_index=gpu_index,
                    pid=pid,
                    name=name,
                    command=name,
                    gpu_memory_bytes=gpu_memory,
                    gpu_memory_percent=memory_percent,
                )
            )
    return processes


def apply_gpu_memory_percent(processes: list[ProcessInfo], gpus: list[GpuInfo]) -> None:
    total_by_gpu = {gpu.index: gpu.memory_total_bytes for gpu in gpus}
    for proc in processes:
        if proc.gpu_index is None:
            continue
        total = total_by_gpu.get(proc.gpu_index, 0)
        proc.gpu_memory_percent = proc.gpu_memory_bytes / total * 100.0 if total > 0 else 0.0


def merge_process_sources(primary: list[ProcessInfo], fallback: list[ProcessInfo]) -> list[ProcessInfo]:
    if not fallback:
        return primary

    fallback_by_pid: dict[int, ProcessInfo] = {}
    for proc in fallback:
        existing = fallback_by_pid.get(proc.pid)
        if existing is None or proc.gpu_memory_bytes > existing.gpu_memory_bytes:
            fallback_by_pid[proc.pid] = proc

    for proc in primary:
        fallback_proc = fallback_by_pid.get(proc.pid)
        if not fallback_proc:
            continue
        if not proc.name:
            proc.name = fallback_proc.name
        if not proc.command:
            proc.command = fallback_proc.command
        if proc.gpu_memory_bytes <= 0:
            proc.gpu_memory_bytes = fallback_proc.gpu_memory_bytes
    primary_pids = {proc.pid for proc in primary}
    primary.extend(proc for pid, proc in fallback_by_pid.items() if pid not in primary_pids)
    return primary


def enrich_processes_with_ps(processes: list[ProcessInfo]) -> None:
    if not processes:
        return
    pid_list = sorted({proc.pid for proc in processes})
    ps_rows = read_ps_rows_fresh(pid_list)
    for proc in processes:
        row = ps_rows.get(proc.pid)
        if not row:
            continue
        proc.ppid = process_ppid_from_ps_row(row, proc.ppid)
        proc.user = row.get("user", proc.user)
        proc.cpu_percent = parse_number(row.get("cpu"), proc.cpu_percent or 0.0)
        proc.host_mem_percent = parse_number(row.get("mem"), proc.host_mem_percent or 0.0)
        proc.elapsed = row.get("etime", proc.elapsed)
        proc.command = row.get("comm", proc.command)
        proc.args = row.get("args", proc.args)
        if not proc.name:
            proc.name = proc.command


def collect_process_ancestors(processes: list[ProcessInfo]) -> list[ProcessInfo]:
    known_pids = {proc.pid for proc in processes}
    seen_pids = set(known_pids)
    frontier: set[int] = {
        proc.ppid
        for proc in processes
        if proc.ppid is not None and proc.ppid > 0 and proc.ppid not in seen_pids
    }
    seen_pids.update(frontier)
    ancestors: list[ProcessInfo] = []

    while frontier:
        ps_rows = read_ps_rows_cached(sorted(frontier))
        next_frontier: set[int] = set()
        for pid in sorted(frontier):
            row = ps_rows.get(pid)
            if not row:
                continue
            ancestor = process_from_ps_row(pid, row)
            ancestors.append(ancestor)
            if ancestor.ppid is None or ancestor.ppid <= 0:
                continue
            if ancestor.ppid in seen_pids:
                continue
            seen_pids.add(ancestor.ppid)
            next_frontier.add(ancestor.ppid)
        frontier = next_frontier

    return ancestors


def process_from_ps_row(pid: int, row: dict[str, str]) -> ProcessInfo:
    command = row.get("comm", "")
    return ProcessInfo(
        gpu_index=None,
        pid=pid,
        ppid=process_ppid_from_ps_row(row),
        name=command,
        user=row.get("user", ""),
        cpu_percent=parse_number(row.get("cpu"), 0.0),
        host_mem_percent=parse_number(row.get("mem"), 0.0),
        elapsed=row.get("etime", ""),
        command=command,
        args=row.get("args", command),
    )


def process_ppid_from_ps_row(row: dict[str, str], default: int | None = None) -> int | None:
    raw = row.get("ppid")
    parsed = parse_int(raw, default=-1)
    if parsed > 0:
        return parsed
    if raw is not None and str(raw).strip():
        return None
    return default


def read_ps_rows_cached(pids: list[int]) -> dict[int, dict[str, str]]:
    if not pids:
        return {}

    now = time.monotonic()
    rows: dict[int, dict[str, str]] = {}
    missing_pids: list[int] = []
    for pid in pids:
        cached = _ps_row_cache.get(pid)
        if cached is not None and now - cached[0] < PS_CACHE_TTL_SECONDS:
            rows[pid] = cached[1]
            continue
        _ps_row_cache.pop(pid, None)
        missing_pids.append(pid)

    if missing_pids:
        fresh_rows = read_ps_rows(missing_pids)
        for pid, row in fresh_rows.items():
            _ps_row_cache[pid] = (now, row)
            rows[pid] = row

    return rows


def read_ps_rows_fresh(pids: list[int]) -> dict[int, dict[str, str]]:
    if not pids:
        return {}

    fresh_rows = read_ps_rows(pids)
    now = time.monotonic()
    for pid, row in fresh_rows.items():
        _ps_row_cache[pid] = (now, row)

    rows = dict(fresh_rows)
    for pid in pids:
        if pid in rows:
            continue
        cached = _ps_row_cache.get(pid)
        if cached is not None:
            rows[pid] = cached[1]
    return rows


def read_ps_rows(pids: list[int]) -> dict[int, dict[str, str]]:
    if not pids:
        return {}
    args = [
        "ps",
        "-o",
        "pid=,ppid=,user=,pcpu=,pmem=,etime=,comm=,args=",
        "-p",
        ",".join(str(pid) for pid in pids),
    ]
    try:
        result = run_command(args, timeout=PS_TIMEOUT_SECONDS)
    except CollectionError:
        return {}
    if result.returncode != 0:
        return {}
    rows: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # pid ppid user pcpu pmem etime comm args
        parts = line.strip().split(None, 7)
        if len(parts) < 7:
            continue
        pid = parse_int(parts[0], default=-1)
        if pid < 0:
            continue
        rows[pid] = {
            "ppid": parts[1],
            "user": parts[2],
            "cpu": parts[3],
            "mem": parts[4],
            "etime": parts[5],
            "comm": parts[6],
            "args": parts[7] if len(parts) > 7 else parts[6],
        }
    return rows


def read_process_detail(pid: int, proc_root: Path | str = PROC_ROOT) -> ProcessDetailInfo:
    detail = ProcessDetailInfo(pid=pid)
    proc_dir = Path(proc_root) / str(pid)
    if not proc_dir.exists():
        detail.error = "process exited or /proc entry unavailable"
        return detail

    errors: list[str] = []
    status_text = read_proc_text(proc_dir / "status", "status", errors)
    if status_text:
        apply_process_status(detail, status_text)

    cmdline = read_proc_bytes(proc_dir / "cmdline", "cmdline", errors)
    if cmdline:
        detail.cmdline = decode_proc_cmdline(cmdline)

    detail.cwd = read_proc_link(proc_dir / "cwd", "cwd", errors)
    detail.exe = read_proc_link(proc_dir / "exe", "exe", errors)
    detail.error = "; ".join(dedupe_preserving_order(errors))
    return detail


def read_proc_text(path: Path, label: str, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        errors.append(f"process exited while reading {label}")
    except PermissionError:
        errors.append(f"permission denied reading {label}")
    except OSError as exc:
        errors.append(f"could not read {label}: {exc}")
    return ""


def read_proc_bytes(path: Path, label: str, errors: list[str]) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        errors.append(f"process exited while reading {label}")
    except PermissionError:
        errors.append(f"permission denied reading {label}")
    except OSError as exc:
        errors.append(f"could not read {label}: {exc}")
    return b""


def read_proc_link(path: Path, label: str, errors: list[str]) -> str:
    try:
        return os.readlink(path)
    except FileNotFoundError:
        errors.append(f"process exited while reading {label}")
    except PermissionError:
        errors.append(f"permission denied reading {label}")
    except OSError as exc:
        errors.append(f"could not read {label}: {exc}")
    return ""


def apply_process_status(detail: ProcessDetailInfo, status_text: str) -> None:
    fields: dict[str, str] = {}
    for line in status_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()

    detail.state = fields.get("State", detail.state)
    detail.threads = parse_optional_status_int(fields.get("Threads"), detail.threads)
    detail.vm_rss_kib = parse_optional_status_int(fields.get("VmRSS"), detail.vm_rss_kib)
    detail.vm_size_kib = parse_optional_status_int(fields.get("VmSize"), detail.vm_size_kib)
    detail.vm_hwm_kib = parse_optional_status_int(fields.get("VmHWM"), detail.vm_hwm_kib)
    detail.cpu_allowed_list = fields.get("Cpus_allowed_list", detail.cpu_allowed_list)
    detail.voluntary_ctxt_switches = parse_optional_status_int(
        fields.get("voluntary_ctxt_switches"),
        detail.voluntary_ctxt_switches,
    )
    detail.nonvoluntary_ctxt_switches = parse_optional_status_int(
        fields.get("nonvoluntary_ctxt_switches"),
        detail.nonvoluntary_ctxt_switches,
    )


def parse_optional_status_int(value: str | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    match = re.search(r"[-+]?\d+", value.replace(",", ""))
    if not match:
        return default
    return parse_int(match.group(0), default=default if default is not None else 0)


def decode_proc_cmdline(cmdline: bytes) -> str:
    parts = [part.decode("utf-8", errors="replace") for part in cmdline.split(b"\0") if part]
    return " ".join(parts)


def load_json_from_text(text: str) -> Any:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if not stripped:
        raise json.JSONDecodeError("empty JSON", text, 0)
    for index, char in enumerate(stripped):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no JSON object found", text, 0)


def parse_memory_bytes_field(value: Any) -> int:
    if not isinstance(value, dict):
        return parse_int(value)
    amount = value.get("value")
    if isinstance(amount, str):
        amount = amount.replace(",", "")
    return int(parse_number(amount, 0.0) * memory_unit_multiplier(value.get("unit")))


def memory_unit_multiplier(unit: Any) -> int:
    text = str(unit or "b").strip().lower().replace(" ", "")
    return MEMORY_UNIT_MULTIPLIERS.get(text, 1)


def first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.upper() != "N/A":
            return text
    return ""


def read_gpu_model(data: dict[str, Any]) -> str:
    reported_model = normalize_gpu_model(
        first_non_empty(
            data.get("Card Series"),
            data.get("Card SKU"),
            data.get("Card Name"),
            data.get("Product Name"),
        )
    )
    if reported_model:
        return reported_model

    device_id = normalize_device_id(first_non_empty(data.get("Card Model"), data.get("Device ID")))
    if device_id:
        mapped_model = GPU_MODEL_BY_DEVICE_ID.get(device_id)
        if mapped_model:
            return mapped_model

        pci_model = load_amd_pci_models().get(device_id)
        if pci_model:
            return pci_model

        device_model = normalize_gpu_model(device_id)
        if device_model:
            return device_model

    architecture = str(data.get("GFX Version", "") or "").strip().lower()
    mapped_architecture_model = GPU_MODEL_BY_ARCHITECTURE.get(architecture)
    if mapped_architecture_model:
        return mapped_architecture_model

    return device_id


def normalize_gpu_model(value: Any) -> str:
    text = first_non_empty(value)
    if not text or re.fullmatch(r"0x[0-9a-f]+", text, flags=re.IGNORECASE):
        return ""
    standard_instinct_model = normalize_instinct_model(text)
    if standard_instinct_model:
        return standard_instinct_model

    bracketed_name = extract_bracketed_product_name(text)
    if bracketed_name:
        standard_instinct_model = normalize_instinct_model(bracketed_name)
        if standard_instinct_model:
            return standard_instinct_model
        text = bracketed_name

    if "radeon" in text.lower():
        return normalize_radeon_model(text)
    return text


def normalize_device_id(value: Any) -> str:
    text = first_non_empty(value).lower()
    match = re.fullmatch(r"(?:0x)?([0-9a-f]{4})", text)
    if match:
        return f"0x{match.group(1)}"
    return text


@lru_cache(maxsize=1)
def load_amd_pci_models() -> dict[str, str]:
    for path in PCI_IDS_PATHS:
        try:
            return parse_amd_pci_models(path.read_text(errors="ignore"))
        except OSError:
            continue
    return {}


def parse_amd_pci_models(text: str) -> dict[str, str]:
    models: dict[str, str] = {}
    in_amd_vendor = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith("\t"):
            vendor_id = line.split(None, 1)[0].lower()
            in_amd_vendor = vendor_id == AMD_PCI_VENDOR_ID
            continue
        if not in_amd_vendor or line.startswith("\t\t"):
            continue
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{4}", parts[0]):
            continue
        model = normalize_gpu_model(parts[1])
        if model:
            models[f"0x{parts[0].lower()}"] = model
    return models


def normalize_instinct_model(value: str) -> str:
    tokens = {
        f"mi{match.group(1).lower()}"
        for match in re.finditer(r"\bMI\s*([0-9]{1,4}[A-Z]?)\b", value, flags=re.IGNORECASE)
    }
    known_tokens = sorted(tokens & INSTINCT_MODEL_BY_TOKEN.keys())
    if not known_tokens:
        return ""
    if len(known_tokens) == 1:
        return INSTINCT_MODEL_BY_TOKEN[known_tokens[0]]

    prefixes = {re.match(r"mi([0-9]+)", token).group(1) for token in known_tokens}
    series_names = {INSTINCT_SERIES_BY_PREFIX[prefix] for prefix in prefixes if prefix in INSTINCT_SERIES_BY_PREFIX}
    if len(series_names) == 1:
        return series_names.pop()
    return ", ".join(INSTINCT_MODEL_BY_TOKEN[token] for token in known_tokens)


def extract_bracketed_product_name(value: str) -> str:
    match = re.search(r"\[([^\]]+)\]", value)
    return match.group(1).strip() if match else ""


def normalize_radeon_model(value: str) -> str:
    text = value.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bRadeon Pro\b", "Radeon PRO", text, flags=re.IGNORECASE)
    if not text.lower().startswith("amd "):
        text = f"AMD {text}"
    return text


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A" or "not supported" in text.lower():
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    return float(match.group(0))


def parse_clock_mhz(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return int(round(float(match.group(0))))


def parse_fan_percent(data: dict[str, Any]) -> float | None:
    for key in (
        "Fan Speed (%)",
        "Fan Speed",
        "Fan Level",
        "fan_speed (%)",
        "current_fan_speed (%)",
    ):
        value = data.get(key)
        parsed = parse_percent_from_text(value)
        if parsed is not None:
            return parsed
    return None


def parse_fan_rpm(data: dict[str, Any]) -> int | None:
    for key in (
        "Fan Speed (RPM)",
        "Fan RPM",
        "Current Fan Speed (RPM)",
        "current_fan_speed (rpm)",
    ):
        value = data.get(key)
        parsed = parse_int_from_text(value)
        if parsed is not None:
            return parsed
    return None


def parse_percent_from_text(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A" or "not supported" in text.lower():
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?=\s*%)", text)
    if match:
        return clamp_percent(float(match.group(0)))
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return clamp_percent(float(text))
    return None


def parse_int_from_text(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A" or "not supported" in text.lower():
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    return int(round(float(match.group(0))))


def _warnings_from_result(result: CommandResult) -> list[str]:
    warnings: list[str] = []
    for stream in (result.stderr,):
        for line in stream.splitlines():
            line = line.strip()
            if not line:
                continue
            if "Permanently added" in line:
                continue
            warnings.append(line)
    return warnings


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _command_failure_message(result: CommandResult) -> str:
    details = result.stderr.strip() or result.stdout.strip()
    command = " ".join(shlex.quote(arg) for arg in result.args)
    if details:
        return f"{command} failed with exit code {result.returncode}: {details}"
    return f"{command} failed with exit code {result.returncode}"


def command_was_interrupted(result: CommandResult) -> bool:
    return result.returncode < 0


def terminal_size() -> os.terminal_size:
    return os.get_terminal_size()

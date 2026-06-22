from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .formatting import clamp_percent, parse_int, parse_number
from .models import GpuInfo, ProcessInfo, Snapshot


ROCM_SMI_ARGS = [
    "rocm-smi",
    "--showproductname",
    "--showuse",
    "--showmeminfo",
    "vram",
    "--showtemp",
    "--showfan",
    "--showclocks",
    "--showpower",
    "--showpids",
    "--showdriverversion",
    "--json",
]

AMD_SMI_PROCESS_ARGS = ["amd-smi", "process", "-G", "--json"]
ROCM_SMI_TIMEOUT_SECONDS = 15.0
GPU_TYPE_BY_DEVICE_ID = {
    "0x75b0": "AMD MI350",
}
GPU_TYPE_BY_GFX_VERSION = {
    "gfx950": "AMD MI350",
}


@dataclass(slots=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


class CollectionError(RuntimeError):
    pass


class CommandTimeout(CollectionError):
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
    now = now or datetime.now()
    warnings: list[str] = []

    rocm_result = run_command(ROCM_SMI_ARGS, timeout=ROCM_SMI_TIMEOUT_SECONDS)
    warnings.extend(_warnings_from_result(rocm_result))
    if rocm_result.returncode != 0 and not rocm_result.stdout.strip():
        raise CollectionError(_command_failure_message(rocm_result))

    rocm_data = load_json_from_text(rocm_result.stdout)
    gpus, rocm_processes, driver_version = parse_rocm_smi_json(rocm_data)

    process_rows: list[ProcessInfo] = []
    try:
        amd_result = run_command(AMD_SMI_PROCESS_ARGS)
        warnings.extend(_warnings_from_result(amd_result))
        if amd_result.returncode == 0 and amd_result.stdout.strip():
            process_rows = parse_amd_smi_process_json(load_json_from_text(amd_result.stdout), gpus)
    except CommandTimeout:
        pass
    except (CollectionError, json.JSONDecodeError, ValueError) as exc:
        warnings.append(f"amd-smi process unavailable: {exc}")

    if not process_rows:
        process_rows = rocm_processes

    process_rows = merge_process_sources(process_rows, rocm_processes)
    enrich_processes_with_ps(process_rows)

    return Snapshot(
        timestamp=now,
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
        warnings=dedupe_preserving_order(warnings),
    )


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
                gpu_type=infer_gpu_type(value),
                gfx_version=gfx_version,
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
        gpu_memory = parse_int(parts[2] if len(parts) > 2 else 0)
        if gpu_memory <= 0:
            continue
        processes.append(
            ProcessInfo(
                gpu_index=None,
                pid=pid,
                name=name,
                command=name,
                gpu_memory_bytes=gpu_memory,
            )
        )
    return processes


def parse_amd_smi_process_json(data: list[dict[str, Any]], gpus: list[GpuInfo]) -> list[ProcessInfo]:
    total_by_gpu = {gpu.index: gpu.memory_total_bytes for gpu in gpus}
    processes: list[ProcessInfo] = []
    for gpu_entry in data:
        gpu_index = parse_int(gpu_entry.get("gpu"), default=-1)
        if gpu_index < 0:
            continue
        for process_entry in gpu_entry.get("process_list", []) or []:
            info = process_entry.get("process_info", {})
            pid = parse_int(info.get("pid"), default=-1)
            if pid < 0:
                continue
            gpu_memory = parse_int(value_field(info.get("mem_usage")))
            if gpu_memory <= 0:
                gpu_memory = parse_int(value_field(info.get("memory_usage", {}).get("vram_mem", {})))
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
                    cu_occupancy=parse_int(info.get("cu_occupancy"), default=0),
                )
            )
    return processes


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
    return primary


def enrich_processes_with_ps(processes: list[ProcessInfo]) -> None:
    if not processes:
        return
    pid_list = sorted({proc.pid for proc in processes})
    ps_rows = read_ps_rows(pid_list)
    for proc in processes:
        row = ps_rows.get(proc.pid)
        if not row:
            continue
        proc.user = row.get("user", proc.user)
        proc.cpu_percent = parse_number(row.get("cpu"), proc.cpu_percent or 0.0)
        proc.host_mem_percent = parse_number(row.get("mem"), proc.host_mem_percent or 0.0)
        proc.elapsed = row.get("etime", proc.elapsed)
        proc.command = row.get("comm", proc.command)
        proc.args = row.get("args", proc.args)
        if not proc.name:
            proc.name = proc.command


def read_ps_rows(pids: list[int]) -> dict[int, dict[str, str]]:
    if not pids:
        return {}
    args = [
        "ps",
        "-o",
        "pid=,user=,pcpu=,pmem=,etime=,comm=,args=",
        "-p",
        ",".join(str(pid) for pid in pids),
    ]
    try:
        result = run_command(args, timeout=3.0)
    except CollectionError:
        return {}
    if result.returncode != 0:
        return {}
    rows: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # pid user pcpu pmem etime comm args
        parts = line.strip().split(None, 6)
        if len(parts) < 6:
            continue
        pid = parse_int(parts[0], default=-1)
        if pid < 0:
            continue
        rows[pid] = {
            "user": parts[1],
            "cpu": parts[2],
            "mem": parts[3],
            "etime": parts[4],
            "comm": parts[5],
            "args": parts[6] if len(parts) > 6 else parts[5],
        }
    return rows


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


def value_field(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.upper() != "N/A":
            return text
    return ""


def infer_gpu_type(data: dict[str, Any]) -> str:
    product_name = compact_gpu_type(
        first_non_empty(
            data.get("Card Series"),
            data.get("Card SKU"),
            data.get("Card Name"),
            data.get("Product Name"),
        )
    )
    if product_name:
        return product_name

    device_id = first_non_empty(data.get("Card Model"), data.get("Device ID")).lower()
    if device_id in GPU_TYPE_BY_DEVICE_ID:
        return GPU_TYPE_BY_DEVICE_ID[device_id]

    gfx_version = str(data.get("GFX Version", "") or "").strip().lower()
    return GPU_TYPE_BY_GFX_VERSION.get(gfx_version, "")


def compact_gpu_type(value: Any) -> str:
    text = first_non_empty(value)
    if not text:
        return ""
    match = re.search(r"\bMI\s*([0-9]{3,4}[A-Z]?)\b", text, flags=re.IGNORECASE)
    if match:
        return f"AMD MI{match.group(1).upper()}"
    if re.fullmatch(r"0x[0-9a-f]+", text, flags=re.IGNORECASE):
        return ""
    return text


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    return parse_number(text)


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


def terminal_size() -> os.terminal_size:
    return os.get_terminal_size()

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
from .models import GpuInfo, ProcessInfo, Snapshot
from .profiling import profile_span


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
AMD_SMI_PROCESS_TIMEOUT_SECONDS = 1.5
AMD_SMI_PROCESS_BACKOFF_SECONDS = 10.0
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
_ps_row_cache: dict[int, tuple[float, dict[str, str]]] = {}


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

    rocm_result, amd_process_command = collect_smi_command_results()
    warnings.extend(_warnings_from_result(rocm_result))
    if command_was_interrupted(rocm_result):
        raise CommandInterrupted(_command_failure_message(rocm_result))
    if rocm_result.returncode != 0 and not rocm_result.stdout.strip():
        raise CollectionError(_command_failure_message(rocm_result))

    rocm_data = load_json_from_text(rocm_result.stdout)
    gpus, rocm_processes, driver_version = parse_rocm_smi_json(rocm_data)

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


def collect_smi_command_results() -> tuple[CommandResult, AmdSmiProcessCommand | None]:
    if not should_run_amd_smi_process():
        return run_command(ROCM_SMI_ARGS, timeout=ROCM_SMI_TIMEOUT_SECONDS), None

    with ThreadPoolExecutor(max_workers=2) as executor:
        rocm_future = executor.submit(run_command, ROCM_SMI_ARGS, timeout=ROCM_SMI_TIMEOUT_SECONDS)
        amd_future = executor.submit(run_command, AMD_SMI_PROCESS_ARGS, timeout=AMD_SMI_PROCESS_TIMEOUT_SECONDS)

        rocm_result = rocm_future.result()
        try:
            amd_result = amd_future.result()
        except CollectionError as exc:
            amd_command = AmdSmiProcessCommand(error=exc)
        else:
            amd_command = AmdSmiProcessCommand(result=amd_result)

    return rocm_result, amd_command


def should_run_amd_smi_process() -> bool:
    return time.monotonic() >= _amd_smi_process_backoff_until


def record_amd_smi_process_failure() -> None:
    global _amd_smi_process_backoff_until
    _amd_smi_process_backoff_until = time.monotonic() + AMD_SMI_PROCESS_BACKOFF_SECONDS


def record_amd_smi_process_success() -> None:
    global _amd_smi_process_backoff_until
    _amd_smi_process_backoff_until = 0.0


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
            gpu_memory = parse_int(value_field(info.get("mem_usage")))
            if gpu_memory <= 0:
                memory_usage = info.get("memory_usage", {})
                if isinstance(memory_usage, dict):
                    gpu_memory = parse_int(value_field(memory_usage.get("vram_mem", {})))
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
    ps_rows = read_ps_rows_cached(pid_list)
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


def command_was_interrupted(result: CommandResult) -> bool:
    return result.returncode < 0


def terminal_size() -> os.terminal_size:
    return os.get_terminal_size()

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable, Sequence

from .collectors import CollectionError, CommandResult, run_command
from .models import ProcessInfo, Snapshot


DEBUG_COUNTER_SAMPLE_MSEC = 1000
DEBUG_COUNTER_MAX_PROCESSES = 3
ROCPROF_COMMAND_TIMEOUT_SECONDS = 8.0
ROCPROF_AVAIL_TIMEOUT_SECONDS = 3.0
ROCPROF_ATTACH_CONFIG_DIR = Path("/tmp")
PROC_ROOT = Path("/proc")
ROCPROF_COUNTER_DEFINITION_PATHS = (
    Path("/opt/rocm/share/rocprofiler-sdk/counter_defs.yaml"),
    Path("/opt/rocm/share/rocprofiler-sdk/basic_counters.xml"),
    Path("/opt/rocm/share/rocprofiler-sdk/derived_counters.xml"),
)

WAVE_COUNTER_CANDIDATES = ("SQ_WAVES_sum", "SQ_WAVES")
INSTRUCTION_COUNTER_CANDIDATES = (
    "SQ_INSTS",
    "SQ_INSTS_VALU",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_VMEM_RD",
    "SQ_INSTS_VMEM_WR",
    "SQ_INSTS_SALU",
    "SQ_INSTS_SMEM",
    "SQ_INSTS_LDS",
    "SQ_INSTS_FLAT",
)
CACHE_HIT_PERCENT_CANDIDATES = ("L2CacheHit",)
CACHE_HIT_COUNTER_CANDIDATES = ("TCC_HIT_sum", "GL2C_HIT_sum", "TCC_HIT", "GL2C_HIT")
CACHE_MISS_COUNTER_CANDIDATES = ("TCC_MISS_sum", "GL2C_MISS_sum", "TCC_MISS", "GL2C_MISS")
FETCH_COUNTER_CANDIDATES = ("FETCH_SIZE", "FetchSize")
WRITE_COUNTER_CANDIDATES = ("WRITE_SIZE", "WriteSize")


RunCommand = Callable[[list[str], float], CommandResult]


@dataclass(frozen=True, slots=True)
class ResolvedDebugCounters:
    counters: tuple[str, ...]
    wave_counter: str | None = None
    instruction_counters: tuple[str, ...] = ()
    l2_hit_percent_counter: str | None = None
    cache_hit_counter: str | None = None
    cache_miss_counter: str | None = None
    fetch_counter: str | None = None
    write_counter: str | None = None
    status: str = ""


@dataclass(frozen=True, slots=True)
class KernelDebugCounters:
    name: str
    dispatches: int = 0
    duration_ns: float | None = None
    waves: float | None = None
    instructions: float | None = None
    l2_hit_percent: float | None = None
    read_bytes: float | None = None
    write_bytes: float | None = None


@dataclass(frozen=True, slots=True)
class ProcessDebugCounters:
    pid: int
    command: str = ""
    status: str = ""
    sample_seconds: float = 0.0
    dispatches: int = 0
    waves: float | None = None
    instructions: float | None = None
    l2_hit_percent: float | None = None
    read_bytes: float | None = None
    write_bytes: float | None = None
    kernels: tuple[KernelDebugCounters, ...] = ()


@dataclass(frozen=True, slots=True)
class GpuDebugCounters:
    gpu_index: int
    sampled_at: datetime
    status: str = ""
    counters: tuple[str, ...] = ()
    processes: tuple[ProcessDebugCounters, ...] = ()


@dataclass(slots=True)
class DispatchCounterRecord:
    kernel_name: str = "unknown"
    duration_ns: float | None = None
    counters: dict[str, float] = field(default_factory=dict)


def collect_gpu_debug_counters(
    snapshot: Snapshot,
    gpu_index: int,
    run_command_func: RunCommand | None = None,
    sample_msec: int = DEBUG_COUNTER_SAMPLE_MSEC,
    max_processes: int = DEBUG_COUNTER_MAX_PROCESSES,
    rocprofv3_path: str | None = None,
    rocprofv3_avail_path: str | None = None,
    now: datetime | None = None,
) -> GpuDebugCounters:
    runner = run_command_func or run_debug_command
    sampled_at = now or datetime.now()
    rocprofv3 = rocprofv3_path or shutil.which("rocprofv3")
    rocprofv3_avail = rocprofv3_avail_path or shutil.which("rocprofv3-avail")
    if rocprofv3 is None:
        return debug_unavailable(gpu_index, sampled_at, "Debug backend unavailable: rocprofv3 not found")
    if rocprofv3_avail is None:
        return debug_unavailable(gpu_index, sampled_at, "Debug backend unavailable: rocprofv3-avail not found")

    resolved = resolve_debug_counters(gpu_index, rocprofv3_avail, runner)
    if not resolved.counters:
        status = resolved.status or "No ROCm counters available"
        return GpuDebugCounters(
            gpu_index=gpu_index,
            sampled_at=sampled_at,
            status=status,
            processes=tuple(
                process_debug_error(process, 0.0, status)
                for process in top_gpu_debug_processes(snapshot.processes, gpu_index, max_processes)
            ),
        )

    processes = top_gpu_debug_processes(snapshot.processes, gpu_index, max_processes)
    if not processes:
        return GpuDebugCounters(
            gpu_index=gpu_index,
            sampled_at=sampled_at,
            counters=resolved.counters,
            status=f"No GPU processes on GPU {gpu_index}",
        )

    sample_seconds = max(0.001, sample_msec / 1000.0)
    process_samples = tuple(
        sample_process_debug_counters(
            process,
            gpu_index,
            resolved,
            rocprofv3,
            runner,
            sample_msec=sample_msec,
            sample_seconds=sample_seconds,
        )
        for process in processes
    )
    ok_count = sum(1 for sample in process_samples if sample.kernels)
    if ok_count:
        status = f"Sampled {ok_count}/{len(process_samples)} process(es)"
    else:
        status = "No active kernel counter samples"
    return GpuDebugCounters(
        gpu_index=gpu_index,
        sampled_at=sampled_at,
        counters=resolved.counters,
        processes=process_samples,
        status=status,
    )


def debug_unavailable(gpu_index: int, sampled_at: datetime, status: str) -> GpuDebugCounters:
    return GpuDebugCounters(gpu_index=gpu_index, sampled_at=sampled_at, status=status)


def run_debug_command(args: list[str], timeout: float) -> CommandResult:
    return run_command(args, timeout=timeout)


def resolve_debug_counters(
    gpu_index: int,
    rocprofv3_avail_path: str,
    run_command_func: RunCommand = run_debug_command,
) -> ResolvedDebugCounters:
    try:
        result = run_command_func(
            [rocprofv3_avail_path, "-d", str(gpu_index), "list", "--pmc"],
            ROCPROF_AVAIL_TIMEOUT_SECONDS,
        )
    except CollectionError as exc:
        return ResolvedDebugCounters((), status=f"Counter discovery failed: {exc}")

    available = parse_available_counter_names(f"{result.stdout}\n{result.stderr}")
    if not resolved_debug_counters_from_available(available).counters:
        available.update(parse_counter_definition_names())

    resolved = resolved_debug_counters_from_available(available)
    if not resolved.counters:
        return ResolvedDebugCounters((), status=f"No hardware counters available for GPU {gpu_index}")

    return validate_resolved_debug_counters(resolved, gpu_index, rocprofv3_avail_path, run_command_func)


def resolved_debug_counters_from_available(available: set[str]) -> ResolvedDebugCounters:
    wave_counter = first_available(available, WAVE_COUNTER_CANDIDATES)
    instruction_counters = resolve_instruction_counters(available)
    l2_hit_counter = first_available(available, CACHE_HIT_PERCENT_CANDIDATES)
    cache_hit_counter = None
    cache_miss_counter = None
    if l2_hit_counter is None:
        cache_hit_counter, cache_miss_counter = resolve_cache_pair(available)
    fetch_counter = first_available(available, FETCH_COUNTER_CANDIDATES)
    write_counter = first_available(available, WRITE_COUNTER_CANDIDATES)

    return ResolvedDebugCounters(
        counters=unique_non_empty(
            (
                wave_counter,
                *instruction_counters,
                l2_hit_counter,
                cache_hit_counter,
                cache_miss_counter,
                fetch_counter,
                write_counter,
            )
        ),
        wave_counter=wave_counter,
        instruction_counters=instruction_counters,
        l2_hit_percent_counter=l2_hit_counter,
        cache_hit_counter=cache_hit_counter,
        cache_miss_counter=cache_miss_counter,
        fetch_counter=fetch_counter,
        write_counter=write_counter,
    )


def validate_resolved_debug_counters(
    resolved: ResolvedDebugCounters,
    gpu_index: int,
    rocprofv3_avail_path: str,
    run_command_func: RunCommand,
) -> ResolvedDebugCounters:
    supported: list[str] = []
    last_error = ""
    for counter in resolved.counters:
        ok, error = validate_counter_group((counter,), gpu_index, rocprofv3_avail_path, run_command_func)
        if ok:
            supported.append(counter)
        elif error:
            last_error = error

    if not supported:
        status = f"Counter validation failed: {last_error}" if last_error else "Counter validation failed"
        return ResolvedDebugCounters((), status=status)

    ok, _error = validate_counter_group(tuple(supported), gpu_index, rocprofv3_avail_path, run_command_func)
    if ok:
        return resolved_debug_counters_from_available(set(supported))

    selected: list[str] = []
    for counter in supported:
        trial = (*selected, counter)
        ok, _error = validate_counter_group(trial, gpu_index, rocprofv3_avail_path, run_command_func)
        if ok:
            selected.append(counter)

    if not selected:
        return ResolvedDebugCounters((), status="Counter validation failed: counters cannot be collected together")
    return resolved_debug_counters_from_available(set(selected))


def validate_counter_group(
    counters: Sequence[str],
    gpu_index: int,
    rocprofv3_avail_path: str,
    run_command_func: RunCommand,
) -> tuple[bool, str]:
    try:
        check = run_command_func(
            [rocprofv3_avail_path, "-d", str(gpu_index), "pmc-check", *counters],
            ROCPROF_AVAIL_TIMEOUT_SECONDS,
        )
    except CollectionError as exc:
        return False, str(exc)
    if check.returncode != 0:
        return False, command_error_text(check)
    return True, ""


def parse_counter_definition_names(paths: Sequence[Path] | None = None) -> set[str]:
    names: set[str] = set()
    for path in paths or counter_definition_paths():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        names.update(parse_available_counter_names(text))
    return names


def counter_definition_paths() -> tuple[Path, ...]:
    paths = list(ROCPROF_COUNTER_DEFINITION_PATHS)
    for root in sorted(Path("/opt").glob("rocm-*")):
        paths.extend(
            (
                root / "share/rocprofiler-sdk/counter_defs.yaml",
                root / "share/rocprofiler-sdk/basic_counters.xml",
                root / "share/rocprofiler-sdk/derived_counters.xml",
            )
        )
    return tuple(dict.fromkeys(paths))


def parse_available_counter_names(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", text))


def first_available(available: set[str], candidates: Sequence[str]) -> str | None:
    return next((counter for counter in candidates if counter in available), None)


def resolve_instruction_counters(available: set[str]) -> tuple[str, ...]:
    if "SQ_INSTS" in available:
        return ("SQ_INSTS",)
    return tuple(counter for counter in INSTRUCTION_COUNTER_CANDIDATES[1:] if counter in available)


def resolve_cache_pair(available: set[str]) -> tuple[str | None, str | None]:
    for hit, miss in zip(CACHE_HIT_COUNTER_CANDIDATES, CACHE_MISS_COUNTER_CANDIDATES):
        if hit in available and miss in available:
            return hit, miss
    return None, None


def top_gpu_debug_processes(
    processes: Iterable[ProcessInfo],
    gpu_index: int,
    limit: int = DEBUG_COUNTER_MAX_PROCESSES,
) -> tuple[ProcessInfo, ...]:
    rows = [proc for proc in processes if proc.gpu_index == gpu_index]
    rows.sort(key=lambda proc: (-proc.gpu_memory_bytes, proc.pid))
    return tuple(rows[: max(0, limit)])


def sample_process_debug_counters(
    process: ProcessInfo,
    gpu_index: int,
    resolved: ResolvedDebugCounters,
    rocprofv3_path: str,
    run_command_func: RunCommand = run_debug_command,
    sample_msec: int = DEBUG_COUNTER_SAMPLE_MSEC,
    sample_seconds: float = 1.0,
) -> ProcessDebugCounters:
    with TemporaryDirectory(prefix="roctop-debug-") as temp_root:
        temp_root_path = Path(temp_root)
        output_dir = temp_root_path / f"pid-{process.pid}"
        attach_library_error = target_attach_library_error(process.pid, rocprofv3_path)
        if attach_library_error:
            return process_debug_error(process, sample_seconds, attach_library_error)
        attach_config_error = clear_rocprof_attach_config(process.pid)
        command = build_rocprofv3_command(
            rocprofv3_path,
            process.pid,
            resolved.counters,
            str(output_dir),
            sample_msec,
        )
        try:
            result = run_command_func(command, ROCPROF_COMMAND_TIMEOUT_SECONDS)
        except CollectionError as exc:
            return process_debug_error(process, sample_seconds, f"rocprofv3 failed: {exc}")

        csv_files = find_counter_collection_csvs(temp_root_path)
        target_temp_root = target_visible_path(process.pid, temp_root_path)
        if target_temp_root is not None and target_temp_root != temp_root_path:
            csv_files.extend(find_counter_collection_csvs(target_temp_root))
        if not csv_files:
            if result.returncode != 0:
                status = command_error_text(result)
                if attach_config_error:
                    status = f"{status}; could not clear stale attach config: {attach_config_error}"
                return process_debug_error(process, sample_seconds, status)
            return process_debug_error(process, sample_seconds, "No counter data collected")

        samples = [
            parse_counter_collection_csv(
                path.read_text(encoding="utf-8", errors="replace"),
                process,
                gpu_index,
                resolved,
                sample_seconds=sample_seconds,
            )
            for path in csv_files
        ]
    return merge_process_debug_samples(process, sample_seconds, samples)


def build_rocprofv3_command(
    rocprofv3_path: str,
    pid: int,
    counters: Sequence[str],
    output_dir: str,
    sample_msec: int = DEBUG_COUNTER_SAMPLE_MSEC,
) -> list[str]:
    return [
        rocprofv3_path,
        "--attach",
        str(pid),
        "--attach-duration-msec",
        str(sample_msec),
        "--pmc",
        *counters,
        "--output-format",
        "csv",
        "--output-directory",
        output_dir,
        "--output-file",
        f"roctop-{pid}",
    ]


def target_attach_library_error(
    pid: int,
    rocprofv3_path: str,
    proc_root: Path = PROC_ROOT,
) -> str:
    process_root = proc_root / str(pid) / "root"
    try:
        if not process_root.exists():
            return ""
    except OSError:
        return ""
    attach_library = rocprof_attach_library_path(rocprofv3_path)
    target_library = target_visible_path(pid, attach_library, proc_root)
    if target_library is None:
        return ""
    try:
        if target_library.exists():
            return ""
    except OSError as exc:
        return f"Cannot inspect target process mount namespace for ROCm attach library: {exc}"
    return (
        f"Target process cannot see ROCm attach library {attach_library}; "
        "run roctop in the same container/ROCm environment as the workload"
    )


def rocprof_attach_library_path(rocprofv3_path: str) -> Path:
    try:
        rocprofv3 = Path(rocprofv3_path).resolve()
    except OSError:
        rocprofv3 = Path(rocprofv3_path)
    return rocprofv3.parent.parent / "lib/rocprofiler-sdk/librocprofv3-attach.so"


def target_visible_path(pid: int, path: Path, proc_root: Path = PROC_ROOT) -> Path | None:
    if not path.is_absolute():
        return None
    try:
        relative = path.relative_to("/")
    except ValueError:
        return None
    return proc_root / str(pid) / "root" / relative


def clear_rocprof_attach_config(pid: int, config_dir: Path = ROCPROF_ATTACH_CONFIG_DIR) -> str:
    path = config_dir / f"rocprofv3_attach_{pid}.pkl"
    try:
        path.unlink()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        return str(exc)
    return ""


def find_counter_collection_csvs(output_dir: Path) -> list[Path]:
    csv_files = sorted(path for path in output_dir.rglob("*.csv") if path.is_file())
    counter_files = [path for path in csv_files if "counter" in path.name.lower()]
    return counter_files or csv_files


def parse_counter_collection_csv(
    text: str,
    process: ProcessInfo,
    gpu_index: int,
    resolved: ResolvedDebugCounters,
    sample_seconds: float = 1.0,
) -> ProcessDebugCounters:
    rows = csv.DictReader(text.splitlines())
    records: dict[str, DispatchCounterRecord] = {}
    for row_index, row in enumerate(rows):
        dispatch_key = csv_value(
            row,
            "Dispatch_Id",
            "Dispatch ID",
            "DispatchID",
            "Correlation_Id",
            "Correlation ID",
            "CorrelationID",
        )
        if not dispatch_key:
            dispatch_key = f"row-{row_index}"
        record = records.setdefault(dispatch_key, DispatchCounterRecord())
        kernel_name = csv_value(row, "Kernel_Name", "Kernel Name", "KernelName", "Kernel")
        if kernel_name:
            record.kernel_name = kernel_name
        duration_ns = parse_float(csv_value(row, "Duration_ns", "Duration (ns)", "DurationNs", "Duration"))
        if duration_ns is not None:
            record.duration_ns = duration_ns
        for counter_name, value in csv_counter_values(row, resolved.counters).items():
            record.counters[counter_name] = record.counters.get(counter_name, 0.0) + value

    dispatch_records = list(records.values())
    if not dispatch_records or not any(record.counters for record in dispatch_records):
        return ProcessDebugCounters(
            pid=process.pid,
            command=process_debug_command(process),
            status=f"No active kernels on GPU {gpu_index}",
            sample_seconds=sample_seconds,
        )
    return process_summary_from_dispatches(process, dispatch_records, resolved, sample_seconds)


def csv_counter_values(row: dict[str, str], counters: Sequence[str]) -> dict[str, float]:
    counter_name = csv_value(row, "Counter_Name", "Counter Name", "Counter", "Metric_Name", "Metric Name")
    counter_value = parse_float(csv_value(row, "Counter_Value", "Counter Value", "Value", "Metric_Value", "Metric Value"))
    if counter_name and counter_value is not None:
        return {counter_name: counter_value}

    values: dict[str, float] = {}
    for counter in counters:
        value = parse_float(csv_value(row, counter))
        if value is not None:
            values[counter] = value
    return values


def process_summary_from_dispatches(
    process: ProcessInfo,
    records: Sequence[DispatchCounterRecord],
    resolved: ResolvedDebugCounters,
    sample_seconds: float,
) -> ProcessDebugCounters:
    kernels: dict[str, list[DispatchCounterRecord]] = {}
    for record in records:
        kernels.setdefault(record.kernel_name or "unknown", []).append(record)
    kernel_summaries = tuple(
        sorted(
            (kernel_summary_from_dispatches(name, group, resolved) for name, group in kernels.items()),
            key=lambda kernel: (-(kernel.duration_ns or 0.0), -kernel.dispatches, kernel.name),
        )
    )
    total = kernel_summary_from_dispatches("all", records, resolved)
    return ProcessDebugCounters(
        pid=process.pid,
        command=process_debug_command(process),
        status="ok",
        sample_seconds=sample_seconds,
        dispatches=total.dispatches,
        waves=total.waves,
        instructions=total.instructions,
        l2_hit_percent=total.l2_hit_percent,
        read_bytes=total.read_bytes,
        write_bytes=total.write_bytes,
        kernels=kernel_summaries,
    )


def kernel_summary_from_dispatches(
    name: str,
    records: Sequence[DispatchCounterRecord],
    resolved: ResolvedDebugCounters,
) -> KernelDebugCounters:
    duration_ns = sum_optional(record.duration_ns for record in records)
    l2_values = [
        record.counters[resolved.l2_hit_percent_counter]
        for record in records
        if resolved.l2_hit_percent_counter is not None and resolved.l2_hit_percent_counter in record.counters
    ]
    return KernelDebugCounters(
        name=name,
        dispatches=len(records),
        duration_ns=duration_ns,
        waves=sum_counter(records, resolved.wave_counter),
        instructions=sum_counters(records, resolved.instruction_counters),
        l2_hit_percent=cache_hit_percent(records, resolved, l2_values),
        read_bytes=counter_kib_to_bytes(sum_counter(records, resolved.fetch_counter)),
        write_bytes=counter_kib_to_bytes(sum_counter(records, resolved.write_counter)),
    )


def cache_hit_percent(
    records: Sequence[DispatchCounterRecord],
    resolved: ResolvedDebugCounters,
    l2_values: Sequence[float],
) -> float | None:
    if l2_values:
        return sum(l2_values) / len(l2_values)
    hits = sum_counter(records, resolved.cache_hit_counter)
    misses = sum_counter(records, resolved.cache_miss_counter)
    if hits is None or misses is None:
        return None
    total = hits + misses
    if total <= 0:
        return None
    return hits / total * 100.0


def sum_counter(records: Sequence[DispatchCounterRecord], counter: str | None) -> float | None:
    if counter is None:
        return None
    found = False
    total = 0.0
    for record in records:
        if counter not in record.counters:
            continue
        found = True
        total += record.counters[counter]
    return total if found else None


def sum_counters(records: Sequence[DispatchCounterRecord], counters: Sequence[str]) -> float | None:
    values = [sum_counter(records, counter) for counter in counters]
    found = [value for value in values if value is not None]
    return sum(found) if found else None


def sum_optional(values: Iterable[float | None]) -> float | None:
    found = False
    total = 0.0
    for value in values:
        if value is None:
            continue
        found = True
        total += value
    return total if found else None


def counter_kib_to_bytes(value: float | None) -> float | None:
    return None if value is None else value * 1024.0


def merge_process_debug_samples(
    process: ProcessInfo,
    sample_seconds: float,
    samples: Sequence[ProcessDebugCounters],
) -> ProcessDebugCounters:
    populated = [sample for sample in samples if sample.kernels]
    if not populated:
        status = samples[0].status if samples else "No counter data collected"
        return process_debug_error(process, sample_seconds, status)
    if len(populated) == 1:
        return populated[0]

    kernels = tuple(kernel for sample in populated for kernel in sample.kernels)
    return ProcessDebugCounters(
        pid=process.pid,
        command=process_debug_command(process),
        status="ok",
        sample_seconds=sample_seconds,
        dispatches=sum(sample.dispatches for sample in populated),
        waves=sum_debug_values(sample.waves for sample in populated),
        instructions=sum_debug_values(sample.instructions for sample in populated),
        l2_hit_percent=avg_debug_values(sample.l2_hit_percent for sample in populated),
        read_bytes=sum_debug_values(sample.read_bytes for sample in populated),
        write_bytes=sum_debug_values(sample.write_bytes for sample in populated),
        kernels=kernels,
    )


def sum_debug_values(values: Iterable[float | None]) -> float | None:
    found = [value for value in values if value is not None]
    return sum(found) if found else None


def avg_debug_values(values: Iterable[float | None]) -> float | None:
    found = [value for value in values if value is not None]
    return sum(found) / len(found) if found else None


def process_debug_error(process: ProcessInfo, sample_seconds: float, status: str) -> ProcessDebugCounters:
    return ProcessDebugCounters(
        pid=process.pid,
        command=process_debug_command(process),
        status=status or "Counter sample failed",
        sample_seconds=sample_seconds,
    )


def process_debug_command(process: ProcessInfo) -> str:
    return process.args or process.command or process.name or ""


def command_error_text(result: CommandResult) -> str:
    output = f"{result.stderr}\n{result.stdout}"
    normalized = output.lower()
    if "agent hw architecture is not supported" in normalized:
        return "ROCm profiler reports unsupported hardware counters for this GPU architecture"
    if "cannot open /dev/kfd" in normalized:
        return "ROCm profiler cannot open /dev/kfd; check ROCm device permissions"
    fatal = first_matching_line(output, "fatal error")
    if fatal:
        return fatal
    text = first_non_empty_line(result.stderr) or first_non_empty_line(result.stdout)
    if text:
        return text
    return f"command exited {result.returncode}"


def first_matching_line(text: str, pattern: str) -> str:
    needle = pattern.lower()
    for line in text.splitlines():
        stripped = line.strip()
        if needle in stripped.lower():
            return stripped
    return ""


def first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def csv_value(row: dict[str, str], *names: str) -> str:
    values = {normalize_field_name(key): value for key, value in row.items() if key is not None}
    for name in names:
        value = values.get(normalize_field_name(name))
        if value is not None:
            return str(value).strip()
    return ""


def normalize_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def unique_non_empty(values: Iterable[str | None]) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return tuple(unique)

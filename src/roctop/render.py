from __future__ import annotations

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from .formatting import clamp_percent, format_bytes_mib, percent_text
from .models import GpuInfo, ProcessInfo, Snapshot


def render_snapshot(snapshot: Snapshot) -> Group:
    header = render_header(snapshot)
    gpu_table = render_gpu_table(snapshot.gpus)
    process_table = render_process_table(snapshot.processes)
    parts = [header, gpu_table, process_table]
    visible_warnings = ui_warnings(snapshot.warnings)
    if visible_warnings:
        parts.append(render_warnings(visible_warnings))
    return Group(*parts)


def render_header(snapshot: Snapshot) -> Panel:
    title = Text("roctop", style="bold cyan")
    timestamp = snapshot.timestamp.strftime("%a %b %d %H:%M:%S %Y")
    details = Text()
    details.append(timestamp, style="white")
    if snapshot.driver_version:
        details.append("   ROCm Driver: ", style="dim")
        details.append(snapshot.driver_version, style="green")
    details.append("   Press Ctrl-C to quit", style="dim")
    return Panel(details, title=title, border_style="bright_black", box=box.SQUARE)


def render_gpu_table(gpus: list[GpuInfo]) -> Table:
    table = Table(box=box.SQUARE, expand=True, show_lines=False, padding=(0, 1))
    table.add_column("GPU", justify="right", style="bold")
    table.add_column("Name", overflow="fold")
    table.add_column("Temp", justify="right")
    table.add_column("Memory-Usage", justify="right")
    table.add_column("GPU-Util", justify="right")
    table.add_column("MEM", ratio=2)
    table.add_column("UTL", ratio=2)

    for gpu in gpus:
        mem_style = percent_style(gpu.memory_percent)
        util_style = percent_style(gpu.utilization_percent)
        temp = f"{gpu.temperature_c:.0f}°C" if gpu.temperature_c is not None else "N/A"
        name = gpu.name
        if gpu.gfx_version and gpu.gfx_version not in name:
            name = f"{name} {gpu.gfx_version}"
        table.add_row(
            str(gpu.index),
            name,
            Text(temp, style=temp_style(gpu.temperature_c)),
            Text(
                f"{format_bytes_mib(gpu.memory_used_bytes)} / {format_bytes_mib(gpu.memory_total_bytes)}",
                style=mem_style,
            ),
            Text(percent_text(gpu.utilization_percent), style=util_style),
            progress_cell(gpu.memory_percent, mem_style),
            progress_cell(gpu.utilization_percent, util_style),
        )
    return table


def render_process_table(processes: list[ProcessInfo]) -> Table:
    table = Table(box=box.SQUARE, expand=True, show_lines=False, padding=(0, 1))
    table.add_column("GPU", justify="right", style="bold")
    table.add_column("PID", justify="right")
    table.add_column("USER")
    table.add_column("GPU-MEM", justify="right")
    table.add_column("%GPU-MEM", justify="right")
    table.add_column("%CPU", justify="right")
    table.add_column("%MEM", justify="right")
    table.add_column("TIME", justify="right")
    table.add_column("COMMAND", overflow="ellipsis", ratio=2)

    if not processes:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "No GPU processes found")
        return table

    for proc in processes:
        gpu = "-" if proc.gpu_index is None else str(proc.gpu_index)
        command = proc.args or proc.command or proc.name or "N/A"
        style = percent_style(proc.gpu_memory_percent)
        table.add_row(
            gpu,
            str(proc.pid),
            proc.user or "-",
            Text(format_bytes_mib(proc.gpu_memory_bytes), style=style),
            Text(percent_text(proc.gpu_memory_percent, digits=1), style=style),
            f"{proc.cpu_percent:.1f}" if proc.cpu_percent is not None else "-",
            f"{proc.host_mem_percent:.1f}" if proc.host_mem_percent is not None else "-",
            proc.elapsed or "-",
            command,
        )
    return table


def render_warnings(warnings: list[str]) -> Panel:
    text = Text()
    for index, warning in enumerate(warnings[:6]):
        if index:
            text.append("\n")
        text.append(warning, style="yellow")
    if len(warnings) > 6:
        text.append(f"\n... {len(warnings) - 6} more warnings", style="yellow")
    return Panel(text, title="Warnings", border_style="yellow", box=box.SQUARE)


def ui_warnings(warnings: list[str]) -> list[str]:
    hidden_fragments = (
        "_amdgpu_device_initialize: amdgpu_get_auth",
        "User is missing the following required groups",
        "Unable to open queues directory",
    )
    return [
        warning
        for warning in warnings
        if not any(fragment in warning for fragment in hidden_fragments)
    ]


def progress_cell(percent: float, style: str) -> ProgressBar:
    return ProgressBar(total=100, completed=clamp_percent(percent), width=None, style="grey23", complete_style=style)


def percent_style(percent: float | int | None) -> str:
    value = clamp_percent(percent)
    if value >= 80:
        return "bold red"
    if value >= 50:
        return "yellow"
    return "green"


def temp_style(temp_c: float | None) -> str:
    if temp_c is None:
        return "dim"
    if temp_c >= 80:
        return "bold red"
    if temp_c >= 65:
        return "yellow"
    return "green"

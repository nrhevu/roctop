from __future__ import annotations

from rich import box
from rich.console import Group
from rich.panel import Panel
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
    table.add_column("Fan", justify="right")
    table.add_column("Power", justify="right")
    table.add_column("SCLK", justify="right")
    table.add_column("MCLK", justify="right")
    table.add_column("Memory-Usage", justify="right")
    table.add_column("MEM", ratio=2)
    table.add_column("UTL", ratio=2)

    for gpu in gpus:
        mem_style = percent_style(gpu.memory_percent)
        util_style = percent_style(gpu.utilization_percent)
        temp = f"{gpu.temperature_c:.0f}°C" if gpu.temperature_c is not None else "N/A"
        fan = format_fan(gpu.fan_percent, gpu.fan_rpm)
        power = f"{gpu.power_w:.0f}W" if gpu.power_w is not None else "N/A"
        sclk = format_clock(gpu.sclk_mhz)
        mclk = format_clock(gpu.mclk_mhz)
        name = gpu.name
        if gpu.gfx_version and gpu.gfx_version not in name:
            name = f"{name} {gpu.gfx_version}"
        table.add_row(
            str(gpu.index),
            name,
            Text(temp, style=temp_style(gpu.temperature_c)),
            Text(fan, style=fan_style(gpu.fan_percent, gpu.fan_rpm)),
            Text(power, style=power_style(gpu.power_w)),
            Text(sclk, style=clock_style(gpu.sclk_mhz)),
            Text(mclk, style=clock_style(gpu.mclk_mhz)),
            Text(
                f"{format_bytes_mib(gpu.memory_used_bytes)} / {format_bytes_mib(gpu.memory_total_bytes)}",
                style=mem_style,
            ),
            bar_with_percent(gpu.memory_percent, mem_style, digits=1),
            bar_with_percent(gpu.utilization_percent, util_style),
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


def bar_with_percent(percent: float, style: str, digits: int = 0) -> Table:
    percent = clamp_percent(percent)
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right", no_wrap=True, width=7)
    grid.add_row(
        progress_text(percent, style),
        Text(percent_text(percent, digits=digits), style=style, no_wrap=True),
    )
    return grid


def progress_text(percent: float, style: str, width: int = 48) -> Text:
    percent = clamp_percent(percent)
    completed = round(width * percent / 100.0)
    text = Text(no_wrap=True, overflow="crop")
    if completed:
        text.append("━" * completed, style=style)
    if completed < width:
        text.append("━" * (width - completed), style="grey23")
    return text


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


def format_fan(fan_percent: float | None, fan_rpm: int | None) -> str:
    if fan_percent is not None:
        return percent_text(fan_percent)
    if fan_rpm is not None:
        return f"{fan_rpm}RPM"
    return "N/A"


def fan_style(fan_percent: float | None, fan_rpm: int | None) -> str:
    if fan_percent is None and fan_rpm is None:
        return "dim"
    if fan_percent is not None:
        return percent_style(fan_percent)
    return "green"


def power_style(power_w: float | None) -> str:
    if power_w is None:
        return "dim"
    if power_w >= 350:
        return "bold red"
    if power_w >= 250:
        return "yellow"
    return "green"


def format_clock(clock_mhz: int | None) -> str:
    if clock_mhz is None:
        return "N/A"
    return f"{clock_mhz}MHz"


def clock_style(clock_mhz: int | None) -> str:
    if clock_mhz is None:
        return "dim"
    return "cyan"

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .formatting import clamp_percent, format_bytes_mib, percent_text
from .history import MetricsHistory
from .models import GpuInfo, ProcessInfo, Snapshot

DRACULA_GREEN = "#50fa7b"
DRACULA_YELLOW = "#f1fa8c"
DRACULA_RED = "#ff5555"
DRACULA_CYAN = "#8be9fd"
DRACULA_PINK = "#ff79c6"
DRACULA_ORANGE = "#ffb86c"
DRACULA_TRACK = "#3a3a3a"
DRACULA_DIM = "#6272a4"
DRACULA_FG = "#f8f8f2"
GRAPH_ROWS_PER_LINE = 4
BRAILLE_DOTS_BY_SUBROW = (0x09, 0x12, 0x24, 0xC0)


def render_snapshot(snapshot: Snapshot, history: MetricsHistory | None = None) -> Group:
    header = render_header(snapshot)
    gpu_table = render_gpu_table(snapshot.gpus)
    process_table = render_process_table(snapshot.processes)
    parts = [header, gpu_table]
    if history is not None:
        parts.append(render_metrics_graphs(history))
    parts.append(process_table)
    visible_warnings = ui_warnings(snapshot.warnings)
    if visible_warnings:
        parts.append(render_warnings(visible_warnings))
    return Group(*parts)


def render_header(snapshot: Snapshot) -> Panel:
    title = Text("roctop", style=f"bold {DRACULA_CYAN}")
    timestamp = snapshot.timestamp.strftime("%a %b %d %H:%M:%S %Y")
    details = Text()
    details.append(timestamp, style=DRACULA_FG)
    if snapshot.driver_version:
        details.append("   ROCm Driver: ", style=DRACULA_DIM)
        details.append(snapshot.driver_version, style=DRACULA_GREEN)
    gpu_types = summarize_gpu_types(snapshot.gpus)
    if gpu_types:
        details.append("   Type: ", style=DRACULA_DIM)
        details.append(gpu_types, style=DRACULA_CYAN)
    gfx_versions = summarize_gfx_versions(snapshot.gpus)
    if gfx_versions:
        details.append("   GFX: ", style=DRACULA_DIM)
        details.append(gfx_versions, style=DRACULA_CYAN)
    details.append("   Press Ctrl-C to quit", style=DRACULA_DIM)
    return Panel(details, title=title, border_style=DRACULA_DIM, box=box.SQUARE)


def render_gpu_table(gpus: list[GpuInfo]) -> Table:
    table = Table(box=box.SQUARE, expand=True, show_lines=False, padding=(0, 1))
    table.add_column("GPU", justify="right", style="bold")
    table.add_column("DID", overflow="fold")
    table.add_column("GUID", overflow="fold")
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
        power = f"{gpu.power_w:.0f}W" if gpu.power_w is not None else "N/A"
        sclk = format_clock(gpu.sclk_mhz)
        mclk = format_clock(gpu.mclk_mhz)
        row = [
            str(gpu.index),
            gpu.name,
            Text(gpu.guid or "N/A", style=DRACULA_FG if gpu.guid else DRACULA_DIM),
            Text(temp, style=temp_style(gpu.temperature_c)),
            Text(
                format_fan(gpu.fan_percent, gpu.fan_rpm),
                style=fan_style(gpu.fan_percent, gpu.fan_rpm),
            ),
            Text(power, style=power_style(gpu.power_w)),
            Text(sclk, style=clock_style(gpu.sclk_mhz)),
            Text(mclk, style=clock_style(gpu.mclk_mhz)),
            Text(
                f"{format_bytes_mib(gpu.memory_used_bytes)} / {format_bytes_mib(gpu.memory_total_bytes)}",
                style=mem_style,
            ),
            bar_with_percent(gpu.memory_percent, mem_style, digits=1),
            bar_with_percent(gpu.utilization_percent, util_style),
        ]
        table.add_row(*row)
    return table


def summarize_gpu_types(gpus: list[GpuInfo]) -> str:
    return ", ".join(unique_non_empty(gpu.gpu_type for gpu in gpus))


def summarize_gfx_versions(gpus: list[GpuInfo]) -> str:
    return ", ".join(unique_non_empty(gpu.gfx_version for gpu in gpus))


def unique_non_empty(values) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique


def render_metrics_graphs(history: MetricsHistory) -> Table:
    table = Table(box=box.SQUARE, expand=True, show_header=False, padding=(0, 1))
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_row(
        MetricGraph(history, "avg_cpu_percent", "Avg %CPU", DRACULA_CYAN),
        MetricGraph(history, "avg_gpu_percent", "Avg %GPU", DRACULA_ORANGE),
    )
    table.add_row(
        MetricGraph(history, "avg_mem_percent", "Avg %MEM", DRACULA_PINK),
        MetricGraph(history, "avg_gpu_mem_percent", "Avg %GPU MEM", DRACULA_YELLOW),
    )
    return table


@dataclass(frozen=True, slots=True)
class MetricGraph:
    history: MetricsHistory
    metric_name: str
    label: str
    style: str
    height: int = 15

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = max(8, options.max_width)
        values = [getattr(sample, self.metric_name) for sample in self.history.samples]
        current = latest_value(values)
        label = Text(no_wrap=True, overflow="ellipsis")
        label.append(f"{self.label}: ", style=DRACULA_FG)
        if current is None:
            label.append("N/A", style=DRACULA_DIM)
        else:
            label.append(percent_text(current, digits=1), style=self.style)
        yield label
        yield from metric_graph_lines(values, width=width, height=self.height, style=self.style)


def latest_value(values: Sequence[float | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None


def metric_graph_lines(
    values: Sequence[float | None],
    width: int,
    height: int,
    style: str,
    rows_per_line: int = GRAPH_ROWS_PER_LINE,
) -> list[Text]:
    width = max(1, width)
    height = max(1, height)
    rows_per_line = max(1, min(rows_per_line, len(BRAILLE_DOTS_BY_SUBROW)))
    recent_values = list(values[-width:])
    padded_values: list[float | None] = [None] * (width - len(recent_values)) + recent_values
    lines: list[Text] = []
    for line_top_level in range(height, 0, -rows_per_line):
        line = Text(no_wrap=True, overflow="crop")
        for value in padded_values:
            filled_rows = graph_filled_rows(value, height)
            active_mask = braille_graph_mask(
                filled_rows=filled_rows,
                line_top_level=line_top_level,
                rows_per_line=rows_per_line,
            )
            if active_mask:
                line.append(braille_char(active_mask), style=style)
            else:
                line.append(" ")
        lines.append(line)
    return lines


def braille_graph_mask(filled_rows: int, line_top_level: int, rows_per_line: int) -> int:
    mask = 0
    for subrow in range(rows_per_line):
        level = line_top_level - subrow
        if level <= 0:
            continue
        dot_mask = BRAILLE_DOTS_BY_SUBROW[subrow]
        if filled_rows >= level:
            mask |= dot_mask
    return mask


def braille_char(mask: int) -> str:
    return chr(0x2800 + mask)


def graph_filled_rows(value: float | None, height: int) -> int:
    if value is None:
        return 0
    percent = clamp_percent(value)
    if percent <= 0:
        return 0
    return max(1, min(height, math.ceil(percent / 100.0 * height)))


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
        gpu_mem_style = percent_style(proc.gpu_memory_percent)
        table.add_row(
            gpu,
            str(proc.pid),
            proc.user or "-",
            Text(format_bytes_mib(proc.gpu_memory_bytes), style=gpu_mem_style),
            Text(percent_text(proc.gpu_memory_percent, digits=1), style=gpu_mem_style),
            metric_text(proc.cpu_percent, digits=1),
            metric_text(proc.host_mem_percent, digits=1),
            proc.elapsed or "-",
            command,
        )
    return table


def metric_text(value: float | int | None, digits: int = 1) -> Text:
    if value is None:
        return Text("-", style=DRACULA_DIM)
    return Text(f"{float(value):.{digits}f}", style=percent_style(value))


def render_warnings(warnings: list[str]) -> Panel:
    text = Text()
    for index, warning in enumerate(warnings[:6]):
        if index:
            text.append("\n")
        text.append(warning, style=DRACULA_YELLOW)
    if len(warnings) > 6:
        text.append(f"\n... {len(warnings) - 6} more warnings", style=DRACULA_YELLOW)
    return Panel(text, title="Warnings", border_style=DRACULA_YELLOW, box=box.SQUARE)


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
        text.append("━" * (width - completed), style=DRACULA_TRACK)
    return text


def percent_style(percent: float | int | None) -> str:
    value = clamp_percent(percent)
    if value >= 80:
        return f"bold {DRACULA_RED}"
    if value >= 50:
        return DRACULA_YELLOW
    return DRACULA_GREEN


def temp_style(temp_c: float | None) -> str:
    if temp_c is None:
        return DRACULA_DIM
    if temp_c >= 80:
        return f"bold {DRACULA_RED}"
    if temp_c >= 65:
        return DRACULA_YELLOW
    return DRACULA_GREEN


def format_fan(fan_percent: float | None, fan_rpm: int | None) -> str:
    if fan_percent is not None:
        return percent_text(fan_percent)
    if fan_rpm is not None:
        return f"{fan_rpm}RPM"
    return "N/A"


def fan_style(fan_percent: float | None, fan_rpm: int | None) -> str:
    if fan_percent is None and fan_rpm is None:
        return DRACULA_DIM
    if fan_percent is not None:
        return percent_style(fan_percent)
    return DRACULA_GREEN


def power_style(power_w: float | None) -> str:
    if power_w is None:
        return DRACULA_DIM
    if power_w >= 350:
        return f"bold {DRACULA_RED}"
    if power_w >= 250:
        return DRACULA_YELLOW
    return DRACULA_GREEN


def format_clock(clock_mhz: int | None) -> str:
    if clock_mhz is None:
        return "N/A"
    return f"{clock_mhz}MHz"


def clock_style(clock_mhz: int | None) -> str:
    if clock_mhz is None:
        return DRACULA_DIM
    return DRACULA_CYAN

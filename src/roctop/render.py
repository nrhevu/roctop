from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from typing import Sequence

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .formatting import clamp_percent, format_bytes_mib, percent_text
from .history import MetricsHistory
from .interaction import MODE_KILL_CONFIRM, MODE_SORT_MENU, SORT_DEFAULT, SORT_LABELS, SORT_OPTIONS, ProcessViewState
from .models import GpuInfo, ProcessInfo, Snapshot

DRACULA_GREEN = "#50fa7b"
DRACULA_YELLOW = "#f1fa8c"
DRACULA_RED = "#ff5555"
DRACULA_CYAN = "#8be9fd"
DRACULA_PURPLE = "#bd93f9"
DRACULA_PINK = "#ff79c6"
DRACULA_ORANGE = "#ffb86c"
DRACULA_TRACK = "#3a3a3a"
DRACULA_DIM = "#6272a4"
DRACULA_FG = "#f8f8f2"
DRACULA_BG = "#282a36"
DRACULA_SELECTION_BG = "#44475a"
DRACULA_SELECTION_FG = DRACULA_FG
GRAPH_ROWS_PER_LINE = 4
BRAILLE_DOTS_BY_SUBROW = (0x09, 0x12, 0x24, 0xC0)


def render_snapshot(
    snapshot: Snapshot,
    history: MetricsHistory | None = None,
    process_state: ProcessViewState | None = None,
    terminal_height: int | None = None,
    terminal_width: int | None = None,
) -> Group:
    gpu_table = render_gpu_table(snapshot.gpus)
    process_rows = estimate_process_view_rows(snapshot, history, terminal_height) if process_state else None
    process_table = render_process_table(
        snapshot.processes,
        process_state=process_state,
        max_rows=process_rows,
        terminal_width=terminal_width,
    )
    header = render_header(
        snapshot,
        process_state=process_state,
        process_count=len(snapshot.processes),
    )
    parts = [header, gpu_table]
    if history is not None:
        parts.append(render_metrics_graphs(history))
    parts.append(process_table)
    visible_warnings = ui_warnings(snapshot.warnings)
    if visible_warnings:
        parts.append(render_warnings(visible_warnings))
    return Group(*parts)


def estimate_process_view_rows(
    snapshot: Snapshot,
    history: MetricsHistory | None,
    terminal_height: int | None,
) -> int | None:
    if terminal_height is None:
        return None
    used_rows = 3
    used_rows += len(snapshot.gpus) + 4
    if history is not None:
        used_rows += 12
    visible_warnings = ui_warnings(snapshot.warnings)
    if visible_warnings:
        used_rows += min(len(visible_warnings), 6) + 4
    return max(1, terminal_height - used_rows - 4)


def render_header(
    snapshot: Snapshot,
    process_state: ProcessViewState | None = None,
    process_count: int = 0,
) -> Panel:
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
    if process_state is not None:
        details.append("\n")
        append_process_help(details)
    else:
        details.append("\n")
        details.append("Ctrl-C: ", style=f"bold {DRACULA_ORANGE}")
        details.append("quit", style=DRACULA_DIM)
    return Panel(details, title=title, border_style=DRACULA_DIM, box=box.SQUARE)


def append_process_help(details: Text) -> None:
    append_keybinding(details, "j/k", "move", leading_space=False)
    append_keybinding(details, "PgUp/PgDn", "scroll")
    append_keybinding(details, "s", "sort")
    append_keybinding(details, "x", "kill")
    append_keybinding(details, "q", "quit")


def append_keybinding(details: Text, key: str, action: str, leading_space: bool = True) -> None:
    if leading_space:
        details.append("   ")
    details.append(f"{key}: ", style=f"bold {DRACULA_ORANGE}")
    details.append(action, style=DRACULA_DIM)


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
    return trim_empty_graph_lines(lines)


def trim_empty_graph_lines(lines: list[Text]) -> list[Text]:
    first_visible = 0
    while first_visible < len(lines) and not lines[first_visible].plain.strip():
        first_visible += 1
    return lines[first_visible:] or lines[-1:]


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


def render_process_table(
    processes: list[ProcessInfo],
    process_state: ProcessViewState | None = None,
    max_rows: int | None = None,
    terminal_width: int | None = None,
) -> Table:
    display_processes = list(processes)
    title = None
    caption = None
    command_width = estimate_process_command_width(terminal_width)
    if process_state is not None:
        display_processes = process_state.sorted_processes(display_processes)
        process_state.sync(display_processes)
        title = Text(process_state.process_title(len(display_processes)), style=DRACULA_DIM)
        caption = render_process_caption(process_state)
        if caption is None:
            caption_text = process_state.caption()
        else:
            caption_text = ""
        if caption_text:
            caption_style = DRACULA_YELLOW
            if process_state.mode == MODE_KILL_CONFIRM:
                caption_style = DRACULA_RED
            caption = Text(caption_text, style=caption_style)
        display_processes = visible_process_window(display_processes, process_state, max_rows, command_width)

    table = Table(
        box=box.SQUARE,
        expand=True,
        show_lines=False,
        padding=(0, 1),
        title=title,
        title_justify="left",
        caption=caption,
        caption_justify="left",
    )
    table.add_column(process_column_header("GPU", "gpu", process_state), justify="right", style="bold")
    table.add_column(process_column_header("PID", "pid", process_state), justify="right")
    table.add_column(process_column_header("USER", "user", process_state))
    table.add_column(process_column_header("GPU-MEM", "gpu_memory", process_state), justify="right")
    table.add_column(process_column_header("%GPU-MEM", "gpu_memory_percent", process_state), justify="right")
    table.add_column(process_column_header("%CPU", "cpu", process_state), justify="right")
    table.add_column(process_column_header("%MEM", "mem", process_state), justify="right")
    table.add_column(process_column_header("TIME", "time", process_state), justify="right")
    table.add_column(
        process_column_header("COMMAND", "command", process_state),
        overflow="fold",
        ratio=1,
        min_width=12,
    )

    if not display_processes:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "No GPU processes found")
        return table

    selected_pid = process_state.selected_pid if process_state is not None else None
    for proc in display_processes:
        gpu = "-" if proc.gpu_index is None else str(proc.gpu_index)
        command = proc.args or proc.command or proc.name or "N/A"
        if process_state is not None:
            command = wrap_process_command(command, command_width)
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
            style=f"bold {DRACULA_SELECTION_FG} on {DRACULA_SELECTION_BG}" if selected_pid == proc.pid else None,
        )
    return table


def render_process_caption(process_state: ProcessViewState) -> Text | None:
    if process_state.mode != MODE_SORT_MENU:
        return None
    caption = Text()
    caption.append("Sort by: ", style=f"bold {DRACULA_CYAN}")
    for index, field in enumerate(SORT_OPTIONS):
        if index:
            caption.append("   ")
        label = SORT_LABELS[field]
        if index == process_state.sort_menu_index:
            caption.append(f" {label} ", style=f"bold {DRACULA_BG} on {DRACULA_SELECTION_BG}")
        else:
            caption.append(label, style=f"bold {DRACULA_CYAN}")
    return caption


def process_column_header(label: str, sort_field: str, process_state: ProcessViewState | None) -> Text:
    text = Text(label, style=f"bold {DRACULA_FG}")
    if (
        process_state is not None
        and process_state.sort_field != SORT_DEFAULT
        and process_state.sort_field == sort_field
    ):
        arrow = "↓" if process_state.sort_desc else "↑"
        text.append(f" {arrow}", style=f"bold {DRACULA_ORANGE}")
    return text


def visible_process_window(
    processes: list[ProcessInfo],
    process_state: ProcessViewState,
    max_visual_rows: int | None,
    command_width: int,
) -> list[ProcessInfo]:
    if not processes:
        process_state.sync(processes)
        return []
    selected_index = max(0, min(process_state.selected_index, len(processes) - 1))
    max_visual_rows = max(1, max_visual_rows or process_state.viewport_rows)
    row_heights = [process_visual_height(proc, command_width) for proc in processes]

    start = selected_index
    end = selected_index + 1
    used_rows = row_heights[selected_index]

    while start > 0 and used_rows + row_heights[start - 1] <= max_visual_rows:
        start -= 1
        used_rows += row_heights[start]
    while end < len(processes) and used_rows + row_heights[end] <= max_visual_rows:
        used_rows += row_heights[end]
        end += 1

    process_state.scroll_offset = start
    process_state.viewport_rows = max(1, end - start)
    return processes[start:end]


def process_visual_height(proc: ProcessInfo, command_width: int) -> int:
    command = proc.args or proc.command or proc.name or "N/A"
    return max(1, len(wrap_command_lines(command, command_width)))


def estimate_process_command_width(terminal_width: int | None) -> int:
    if terminal_width is None:
        return 80
    return max(18, terminal_width - 82)


def wrap_process_command(command: str, width: int) -> str:
    return "\n".join(wrap_command_lines(command, width))


def wrap_command_lines(command: str, width: int) -> list[str]:
    text = str(command or "N/A")
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        wrapped = textwrap.wrap(
            raw_line,
            width=max(1, width),
            break_long_words=True,
            break_on_hyphens=False,
            drop_whitespace=True,
        )
        lines.extend(wrapped or [""])
    return lines or [""]


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

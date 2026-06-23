from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .formatting import clamp_percent, format_bytes_mib, percent_text
from .history import MetricsHistory
from .interaction import (
    KILL_CONFIRM_LABELS,
    KILL_CONFIRM_OPTIONS,
    MODE_FILTER,
    MODE_KILL_CONFIRM,
    MODE_SEARCH,
    MODE_SORT_MENU,
    SORT_DEFAULT,
    SORT_LABELS,
    SORT_OPTIONS,
    ProcessViewState,
)
from .models import GpuInfo, ProcessInfo, Snapshot
from .profiling import profile_span

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
SORT_MENU_SELECTION_BG = DRACULA_CYAN
SORT_MENU_SELECTION_FG = DRACULA_BG
GRAPH_ROWS_PER_LINE = 4
GRAPH_COLUMNS_PER_CELL = 2
BRAILLE_DOTS_BY_COLUMN = (
    (0x01, 0x02, 0x04, 0x40),
    (0x08, 0x10, 0x20, 0x80),
)


@dataclass(slots=True)
class ProcessRenderRow:
    process: ProcessInfo
    command: str
    visual_height: int


def render_snapshot(
    snapshot: Snapshot,
    history: MetricsHistory | None = None,
    process_state: ProcessViewState | None = None,
    terminal_height: int | None = None,
    terminal_width: int | None = None,
    display_processes: list[ProcessInfo] | None = None,
    display_time: datetime | None = None,
    show_subsecond_time: bool = False,
) -> Group:
    with profile_span("render"):
        gpu_table = render_gpu_table(snapshot.gpus)
        process_rows = estimate_process_view_rows(snapshot, history, terminal_height, process_state)
        process_table = render_process_table(
            display_processes if display_processes is not None else snapshot.processes,
            process_state=process_state,
            max_rows=process_rows,
            terminal_width=terminal_width,
            processes_sorted=display_processes is not None,
        )
        header = render_header(
            snapshot,
            process_state=process_state,
            process_count=len(snapshot.processes),
            display_time=display_time,
            show_subsecond_time=show_subsecond_time,
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
    process_state: ProcessViewState | None = None,
) -> int | None:
    if terminal_height is None:
        return None
    used_rows = 4
    used_rows += len(snapshot.gpus) + 4
    if history is not None:
        used_rows += 13
    if process_state is not None and process_state.mode in (MODE_SORT_MENU, MODE_KILL_CONFIRM, MODE_SEARCH, MODE_FILTER):
        used_rows += 1
    visible_warnings = ui_warnings(snapshot.warnings)
    if visible_warnings:
        used_rows += min(len(visible_warnings), 6) + 4
    return max(1, terminal_height - used_rows - 5)


def render_header(
    snapshot: Snapshot,
    process_state: ProcessViewState | None = None,
    process_count: int = 0,
    display_time: datetime | None = None,
    show_subsecond_time: bool = False,
) -> Panel:
    title = Text("roctop", style=f"bold {DRACULA_CYAN}")
    if snapshot.node_name:
        title.append(" @ ", style=DRACULA_DIM)
        title.append(snapshot.node_name, style=f"bold {DRACULA_GREEN}")
    timestamp = format_header_timestamp(display_time or snapshot.timestamp, show_subsecond_time)
    details = Text()
    details.append(timestamp, style=DRACULA_FG)
    if snapshot.driver_version:
        details.append("   ROCm Driver: ", style=DRACULA_DIM)
        details.append(snapshot.driver_version, style=DRACULA_GREEN)
    gpu_models = summarize_gpu_models(snapshot.gpus)
    if gpu_models:
        details.append("   Model: ", style=DRACULA_DIM)
        details.append(gpu_models, style=DRACULA_CYAN)
    architectures = summarize_gpu_architectures(snapshot.gpus)
    if architectures:
        details.append("   Architecture: ", style=DRACULA_DIM)
        details.append(architectures, style=DRACULA_CYAN)
    if process_state is not None:
        details.append("\n")
        append_process_help(details)
    else:
        details.append("\n")
        details.append("Ctrl-C: ", style=f"bold {DRACULA_ORANGE}")
        details.append("quit", style=DRACULA_DIM)
    return Panel(details, title=title, border_style=DRACULA_DIM, box=box.SQUARE)


def format_header_timestamp(timestamp: datetime, show_subsecond_time: bool = False) -> str:
    if not show_subsecond_time:
        return timestamp.strftime("%a %b %d %H:%M:%S %Y")
    tenth = timestamp.microsecond // 100000
    return f"{timestamp.strftime('%a %b %d %H:%M:%S')}.{tenth} {timestamp.strftime('%Y')}"


def append_process_help(details: Text) -> None:
    append_keybinding(details, "s", "sort", leading_space=False)
    append_keybinding(details, "/", "search")
    append_keybinding(details, "f", "filter")
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
    table.add_column("GUID", overflow="fold")
    table.add_column("Temp", justify="right")
    table.add_column("Fan", justify="right")
    table.add_column("Power", justify="right")
    table.add_column("SCLK", justify="right")
    table.add_column("MCLK", justify="right")
    table.add_column("Memory-Usage", justify="right")
    table.add_column("%Memory-Usage", ratio=2)
    table.add_column("%Utilization", ratio=2)

    for gpu in gpus:
        mem_style = percent_style(gpu.memory_percent)
        util_style = percent_style(gpu.utilization_percent)
        temp = f"{gpu.temperature_c:.0f}°C" if gpu.temperature_c is not None else "N/A"
        power = f"{gpu.power_w:.0f}W" if gpu.power_w is not None else "N/A"
        sclk = format_clock(gpu.sclk_mhz)
        mclk = format_clock(gpu.mclk_mhz)
        row = [
            str(gpu.index),
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


def summarize_gpu_models(gpus: list[GpuInfo]) -> str:
    return ", ".join(unique_non_empty(gpu.gpu_type for gpu in gpus))


def summarize_gpu_architectures(gpus: list[GpuInfo]) -> str:
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
        MetricGraphPair(
            history=history,
            top_metric_name="avg_cpu_percent",
            top_label="Avg %CPU",
            top_style=DRACULA_CYAN,
            bottom_metric_name="avg_mem_percent",
            bottom_label="Avg %MEM",
            bottom_style=DRACULA_PINK,
        ),
        MetricGraphPair(
            history=history,
            top_metric_name="avg_gpu_percent",
            top_label="Avg %GPU",
            top_style=DRACULA_ORANGE,
            bottom_metric_name="avg_gpu_mem_percent",
            bottom_label="Avg %GPU MEM",
            bottom_style=DRACULA_YELLOW,
        ),
    )
    return table


@dataclass(frozen=True, slots=True)
class MetricGraphPair:
    history: MetricsHistory
    top_metric_name: str
    top_label: str
    top_style: str
    bottom_metric_name: str
    bottom_label: str
    bottom_style: str
    height: int = 15

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = max(12, options.max_width)
        top_values = [getattr(sample, self.top_metric_name) for sample in self.history.samples]
        bottom_values = [getattr(sample, self.bottom_metric_name) for sample in self.history.samples]
        yield metric_label(self.top_label, latest_value(top_values), self.top_style)
        yield from metric_graph_lines(top_values, width=width, height=self.height, style=self.top_style, trim_empty=False)
        yield time_axis_line(width)
        yield from reversed(
            metric_graph_lines(
                bottom_values,
                width=width,
                height=self.height,
                style=self.bottom_style,
                trim_empty=False,
                invert_dots=True,
            )
        )
        yield metric_label(self.bottom_label, latest_value(bottom_values), self.bottom_style)


def metric_label(label: str, value: float | None, style: str) -> Text:
    text = Text(no_wrap=True, overflow="ellipsis")
    text.append(f"{label}: ", style=DRACULA_FG)
    if value is None:
        text.append("N/A", style=DRACULA_DIM)
    else:
        text.append(percent_text(value, digits=1), style=style)
    return text


def time_axis_line(width: int) -> Text:
    width = max(1, width)
    chars = ["─"] * width
    for seconds, label in (
        (1080, "1080s"),
        (720, "720s"),
        (360, "360s"),
        (240, "240s"),
        (120, "120s"),
        (60, "60s"),
        (30, "30s"),
    ):
        marker = width - 1 - seconds // GRAPH_COLUMNS_PER_CELL
        start = marker - len(label)
        space = start - 1
        if space < 0 or start + len(label) > width:
            continue
        chars[space] = " "
        chars[start:marker] = list(label)
        chars[marker] = "│"
    return Text("".join(chars), style=DRACULA_DIM, no_wrap=True, overflow="crop")


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
    trim_empty: bool = True,
    invert_dots: bool = False,
) -> list[Text]:
    width = max(1, width)
    height = max(1, height)
    rows_per_line = max(1, min(rows_per_line, len(BRAILLE_DOTS_BY_COLUMN[0])))
    graph_columns = width * GRAPH_COLUMNS_PER_CELL
    recent_values = list(values[-graph_columns:])
    padded_values: list[float | None] = [None] * (graph_columns - len(recent_values)) + recent_values
    lines: list[Text] = []
    for line_top_level in range(height, 0, -rows_per_line):
        line = Text(no_wrap=True, overflow="crop")
        for column_start in range(0, len(padded_values), GRAPH_COLUMNS_PER_CELL):
            active_mask = 0
            for column, value in enumerate(padded_values[column_start : column_start + GRAPH_COLUMNS_PER_CELL]):
                filled_rows = graph_filled_rows(value, height)
                active_mask |= braille_graph_mask(
                    filled_rows=filled_rows,
                    line_top_level=line_top_level,
                    rows_per_line=rows_per_line,
                    column=column,
                )
            if invert_dots:
                active_mask = flip_braille_vertical(active_mask)
            if active_mask:
                line.append(braille_char(active_mask), style=style)
            else:
                line.append(" ")
        lines.append(line)
    if trim_empty:
        return trim_empty_graph_lines(lines)
    return lines


def trim_empty_graph_lines(lines: list[Text]) -> list[Text]:
    first_visible = 0
    while first_visible < len(lines) and not lines[first_visible].plain.strip():
        first_visible += 1
    return lines[first_visible:] or lines[-1:]


def braille_graph_mask(filled_rows: int, line_top_level: int, rows_per_line: int, column: int) -> int:
    mask = 0
    dot_masks = BRAILLE_DOTS_BY_COLUMN[column]
    for subrow in range(rows_per_line):
        level = line_top_level - subrow
        if level <= 0:
            continue
        dot_mask = dot_masks[subrow]
        if filled_rows >= level:
            mask |= dot_mask
    return mask


def flip_braille_vertical(mask: int) -> int:
    flipped = 0
    for source, target in (
        (0x01, 0x40),
        (0x02, 0x04),
        (0x04, 0x02),
        (0x08, 0x80),
        (0x10, 0x20),
        (0x20, 0x10),
        (0x40, 0x01),
        (0x80, 0x08),
    ):
        if mask & source:
            flipped |= target
    return flipped


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
    processes_sorted: bool = False,
) -> Table:
    display_processes = list(processes)
    process_count = len(display_processes)
    display_rows: list[ProcessRenderRow]
    title = None
    command_width = estimate_process_command_width(terminal_width)
    if process_state is not None:
        if not processes_sorted:
            display_processes = process_state.display_processes(display_processes)
        process_state.sync(display_processes, viewport_rows=max_rows)
        title = render_process_title(process_state, len(display_processes))
        display_rows = visible_process_window(display_processes, process_state, max_rows, command_width)
    else:
        if max_rows is not None and len(display_processes) > max_rows:
            display_processes = display_processes[: max(1, max_rows)]
            title = render_static_process_title(len(display_processes), process_count)
        if max_rows is None:
            display_rows = [ProcessRenderRow(proc, process_command(proc), 1) for proc in display_processes]
        else:
            display_rows = [wrapped_process_row(proc, command_width, max_lines=1) for proc in display_processes]

    table = Table(
        box=box.SQUARE,
        expand=True,
        show_lines=False,
        padding=(0, 1),
        title=title,
        title_justify="left",
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

    if not display_rows:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "No GPU processes found")
        return table

    selected_visible_index = None
    if process_state is not None:
        selected_visible_index = process_state.selected_index - process_state.scroll_offset
    for visible_index, row in enumerate(display_rows):
        proc = row.process
        gpu = "-" if proc.gpu_index is None else str(proc.gpu_index)
        command = row.command
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
            style=(
                f"bold {DRACULA_SELECTION_FG} on {DRACULA_SELECTION_BG}"
                if selected_visible_index == visible_index
                else None
            ),
        )
    return table


def render_process_title(process_state: ProcessViewState, process_count: int) -> Text:
    title = Text(process_state.process_title(process_count), style=DRACULA_DIM)
    status_text = process_state.caption()
    if status_text:
        title.append("   ")
        title.append(status_text, style=process_status_style(process_state))
    sort_menu = render_sort_menu(process_state)
    kill_menu = render_kill_confirm_menu(process_state)
    search_menu = render_search_menu(process_state)
    filter_menu = render_filter_menu(process_state)
    menu = sort_menu or kill_menu or search_menu or filter_menu
    if menu is not None:
        title.append("\n")
        title.append(menu)
    return title


def render_static_process_title(visible_count: int, process_count: int) -> Text:
    return Text(f"Processes  {visible_count}/{process_count}", style=DRACULA_DIM)


def process_status_style(process_state: ProcessViewState) -> str:
    if process_state.mode == MODE_KILL_CONFIRM:
        return DRACULA_RED
    return DRACULA_YELLOW


def render_sort_menu(process_state: ProcessViewState) -> Text | None:
    if process_state.mode != MODE_SORT_MENU:
        return None
    menu = Text(no_wrap=True, overflow="ellipsis")
    menu.append("Sort by: ", style=f"bold {DRACULA_CYAN}")
    for index, field in enumerate(SORT_OPTIONS):
        if index:
            menu.append("   ")
        label = SORT_LABELS[field]
        if index == process_state.sort_menu_index:
            menu.append(f" {label} ", style=f"bold {SORT_MENU_SELECTION_FG} on {SORT_MENU_SELECTION_BG}")
        else:
            menu.append(label, style=f"bold {DRACULA_CYAN}")
    return menu


def render_kill_confirm_menu(process_state: ProcessViewState) -> Text | None:
    if process_state.mode != MODE_KILL_CONFIRM:
        return None
    menu = Text(no_wrap=True, overflow="ellipsis")
    if process_state.selected_pid is None:
        menu.append("Kill process: ", style=f"bold {DRACULA_RED}")
    else:
        menu.append(f"Kill PID {process_state.selected_pid}: ", style=f"bold {DRACULA_RED}")
    for index, option in enumerate(KILL_CONFIRM_OPTIONS):
        if index:
            menu.append("   ")
        label = KILL_CONFIRM_LABELS[option]
        if index == process_state.kill_confirm_index:
            menu.append(f" {label} ", style=f"bold {SORT_MENU_SELECTION_FG} on {SORT_MENU_SELECTION_BG}")
        else:
            menu.append(label, style=f"bold {DRACULA_CYAN}")
    return menu


def render_search_menu(process_state: ProcessViewState) -> Text | None:
    if process_state.mode != MODE_SEARCH:
        return None
    menu = Text(no_wrap=True, overflow="ellipsis")
    menu.append("Search: ", style=f"bold {DRACULA_CYAN}")
    menu.append(process_state.search_input, style=DRACULA_FG)
    return menu


def render_filter_menu(process_state: ProcessViewState) -> Text | None:
    if process_state.mode != MODE_FILTER:
        return None
    menu = Text(no_wrap=True, overflow="ellipsis")
    menu.append("Filter: ", style=f"bold {DRACULA_CYAN}")
    menu.append(process_state.filter_input, style=DRACULA_FG)
    return menu


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
) -> list[ProcessRenderRow]:
    if not processes:
        process_state.scroll_offset = 0
        return []

    with profile_span("process-window"):
        selected_index = max(0, min(process_state.selected_index, len(processes) - 1))
        max_visual_rows = max(1, max_visual_rows or process_state.viewport_rows)
        wrapped_rows: dict[int, ProcessRenderRow] = {}

        def row_at(index: int) -> ProcessRenderRow:
            row = wrapped_rows.get(index)
            if row is None:
                row = wrapped_process_row(processes[index], command_width, max_lines=max_visual_rows)
                wrapped_rows[index] = row
            return row

        start = selected_index
        end = selected_index + 1
        used_rows = row_at(selected_index).visual_height

        while start > 0:
            row = row_at(start - 1)
            if used_rows + row.visual_height > max_visual_rows:
                break
            start -= 1
            used_rows += row.visual_height
        while end < len(processes):
            row = row_at(end)
            if used_rows + row.visual_height > max_visual_rows:
                break
            used_rows += row.visual_height
            end += 1

        process_state.scroll_offset = start
        return [row_at(index) for index in range(start, end)]


def process_command(proc: ProcessInfo) -> str:
    return proc.args or proc.command or proc.name or "N/A"


def wrapped_process_row(proc: ProcessInfo, command_width: int, max_lines: int | None = None) -> ProcessRenderRow:
    command = wrap_process_command(process_command(proc), command_width, max_lines)
    return ProcessRenderRow(proc, command, max(1, command.count("\n") + 1))


def estimate_process_command_width(terminal_width: int | None) -> int:
    if terminal_width is None:
        return 80
    return max(18, terminal_width - 82)


def wrap_process_command(command: str, width: int, max_lines: int | None = None) -> str:
    lines = wrap_command_lines(command, width)
    if max_lines is not None:
        lines = truncate_command_lines(lines, width, max_lines)
    return "\n".join(lines)


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


def truncate_command_lines(lines: list[str], width: int, max_lines: int) -> list[str]:
    max_lines = max(1, max_lines)
    if len(lines) <= max_lines:
        return lines
    visible_lines = list(lines[:max_lines])
    visible_lines[-1] = ellipsize_command_line(visible_lines[-1], width)
    return visible_lines


def ellipsize_command_line(line: str, width: int) -> str:
    width = max(1, width)
    if width <= 3:
        return "." * width
    return f"{line[: width - 3].rstrip()}..."


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

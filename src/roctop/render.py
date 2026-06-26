from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.measure import Measurement
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from .formatting import clamp_percent, format_bytes_mib, percent_text
from .history import MetricSample, MetricsHistory
from .interaction import (
    HELP_ENTRIES,
    HELP_VISIBLE_ROWS,
    KILL_CONFIRM_LABELS,
    KILL_CONFIRM_OPTIONS,
    MODE_FILTER,
    MODE_HELP,
    MODE_KILL_CONFIRM,
    MODE_PROCESS_INFO,
    MODE_SEARCH,
    MODE_SORT_MENU,
    PROCESS_INFO_VISIBLE_ROWS,
    SORT_DEFAULT,
    SORT_LABELS,
    SORT_OPTIONS,
    ProcessViewState,
    elapsed_seconds,
    process_selection_key,
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
PROCESS_TABLE_CHROME_ROWS = 5
PROCESS_TABLE_COLUMN_COUNT = 9
PROCESS_TABLE_CELL_PADDING_WIDTH = 2
PROCESS_TABLE_MIN_COMMAND_WIDTH = 12
# Braille cells have two horizontal dot columns. Packing two one-second
# buckets per terminal cell keeps the dotted graph visually continuous.
GRAPH_COLUMNS_PER_CELL = 2
TIME_AXIS_MARKERS_SECONDS = (1080, 720, 360, 240, 120, 60, 30)
BRAILLE_DOTS_BY_COLUMN = (
    (0x01, 0x02, 0x04, 0x40),
    (0x08, 0x10, 0x20, 0x80),
)


@dataclass(slots=True)
class ProcessRenderRow:
    process: ProcessInfo
    command: str
    visual_height: int


@dataclass(frozen=True, slots=True)
class ProcessTableWidths:
    gpu: int
    pid: int
    user: int
    gpu_memory: int
    gpu_memory_percent: int
    cpu: int
    mem: int
    time: int
    command: int


@dataclass(frozen=True, slots=True)
class ProgressText:
    percent: float
    style: str

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = max(1, options.max_width)
        completed = round(width * self.percent / 100.0)
        text = Text(no_wrap=True, overflow="crop")
        if completed:
            text.append("━" * completed, style=self.style)
        if completed < width:
            text.append("━" * (width - completed), style=DRACULA_TRACK)
        yield text

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(1, options.max_width)


@dataclass(frozen=True, slots=True)
class PopupOverlay:
    base: object
    popup_factory: Callable[[int], object]
    terminal_height: int | None
    terminal_width: int | None

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        options_width = max(1, options.max_width or console.size.width)
        requested_width = max(1, self.terminal_width or options_width)
        width = min(requested_width, options_width)
        target_height = max(1, self.terminal_height or options.height or console.size.height)
        render_options = options.update(width=width, max_width=width, height=None)

        base_lines = [
            Segment.adjust_line_length(line, width, pad=True)
            for line in console.render_lines(self.base, render_options, pad=True)
        ]
        popup_lines = console.render_lines(self.popup_factory(width), render_options, pad=False)
        if not popup_lines:
            yield from self._render_lines(base_lines)
            return

        popup_width = min(width, max(Segment.get_line_length(line) for line in popup_lines))
        popup_lines = [Segment.adjust_line_length(line, popup_width, pad=True) for line in popup_lines]

        canvas_height = max(len(base_lines), target_height)
        blank_line = [Segment(" " * width)]
        canvas = base_lines + [blank_line] * (canvas_height - len(base_lines))
        top = max(0, (target_height - len(popup_lines)) // 2)
        left = max(0, (width - popup_width) // 2)
        right = min(width, left + popup_width)

        if top + len(popup_lines) > len(canvas):
            canvas.extend([blank_line] * (top + len(popup_lines) - len(canvas)))

        for row_index, popup_line in enumerate(popup_lines):
            base_line = Segment.adjust_line_length(canvas[top + row_index], width, pad=True)
            before, _covered, after = list(Segment.divide(base_line, [left, right, width]))
            canvas[top + row_index] = before + popup_line + after

        yield from self._render_lines(canvas)

    @staticmethod
    def _render_lines(lines: list[list[Segment]]) -> RenderResult:
        for index, line in enumerate(lines):
            yield from line
            if index < len(lines) - 1:
                yield Segment.line()


def HelpOverlay(
    base: object,
    process_state: ProcessViewState,
    terminal_height: int | None,
    terminal_width: int | None,
    gpus: Sequence[GpuInfo] | None = None,
) -> PopupOverlay:
    return PopupOverlay(base, lambda width: render_help_popup(process_state, width, gpus), terminal_height, terminal_width)


def render_snapshot(
    snapshot: Snapshot,
    history: MetricsHistory | None = None,
    process_state: ProcessViewState | None = None,
    terminal_height: int | None = None,
    terminal_width: int | None = None,
    display_processes: list[ProcessInfo] | None = None,
    display_time: datetime | None = None,
    show_subsecond_time: bool = False,
    history_samples: Sequence[MetricSample] | None = None,
    graph_time: datetime | None = None,
    graph_time_offset_seconds: int = 0,
) -> Group | PopupOverlay:
    with profile_span("render"):
        process_rows = estimate_process_view_rows(snapshot, history, terminal_height, process_state)
        process_table = render_process_table(
            display_processes if display_processes is not None else snapshot.processes,
            process_state=process_state,
            max_rows=process_rows,
            terminal_width=terminal_width,
            processes_sorted=display_processes is not None,
            process_ancestors=snapshot.process_ancestors,
            elapsed_offset_seconds=process_elapsed_offset_seconds(snapshot.timestamp, display_time),
        )
        if process_state is not None and process_state.process_zoomed:
            parts = [process_table]
        else:
            header = render_header(
                snapshot,
                process_state=process_state,
                process_count=len(snapshot.processes),
                display_time=display_time,
                show_subsecond_time=show_subsecond_time,
                terminal_width=terminal_width,
            )
            parts = [header, render_gpu_table(snapshot.gpus)]
            if history is not None:
                parts.append(
                    render_metrics_graphs(
                        history,
                        end_time=graph_time or display_time,
                        samples=history_samples,
                        time_offset_seconds=graph_time_offset_seconds,
                    )
                )
            parts.append(process_table)
            visible_warnings = ui_warnings(snapshot.warnings)
            if visible_warnings:
                parts.append(render_warnings(visible_warnings))
        base = Group(*parts)
        if process_state is not None and process_state.mode == MODE_HELP:
            return HelpOverlay(base, process_state, terminal_height, terminal_width, snapshot.gpus)
        if process_state is not None and process_state.mode == MODE_PROCESS_INFO:
            return PopupOverlay(
                base,
                lambda width: render_process_info_popup(snapshot, process_state, width),
                terminal_height,
                terminal_width,
            )
        return base


def estimate_process_view_rows(
    snapshot: Snapshot,
    history: MetricsHistory | None,
    terminal_height: int | None,
    process_state: ProcessViewState | None = None,
) -> int | None:
    if terminal_height is None:
        return None
    if process_state is not None and process_state.process_zoomed:
        used_rows = PROCESS_TABLE_CHROME_ROWS + process_inline_menu_rows(process_state)
        return max(1, terminal_height - used_rows)
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


def process_inline_menu_rows(process_state: ProcessViewState | None) -> int:
    if process_state is None:
        return 0
    if process_state.mode in (MODE_SORT_MENU, MODE_KILL_CONFIRM, MODE_SEARCH, MODE_FILTER):
        return 1
    return 0


def process_elapsed_offset_seconds(snapshot_time: datetime, display_time: datetime | None) -> int:
    if display_time is None:
        return 0
    return max(0, int((display_time - snapshot_time).total_seconds()))


def render_header(
    snapshot: Snapshot,
    process_state: ProcessViewState | None = None,
    process_count: int = 0,
    display_time: datetime | None = None,
    show_subsecond_time: bool = False,
    terminal_width: int | None = None,
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
    guide = Text(no_wrap=True, overflow="crop")
    if process_state is not None:
        append_process_help(guide, snapshot.gpus, process_help_separator(terminal_width))
    else:
        guide.append("Ctrl-C: ", style=f"bold {DRACULA_ORANGE}")
        guide.append("quit", style=DRACULA_DIM)
    return Panel(
        Group(details, guide),
        title=title,
        border_style=DRACULA_DIM,
        box=box.SQUARE,
        padding=(0, 0),
    )


def format_header_timestamp(timestamp: datetime, show_subsecond_time: bool = False) -> str:
    if not show_subsecond_time:
        return timestamp.strftime("%a %b %d %H:%M:%S %Y")
    tenth = timestamp.microsecond // 100000
    return f"{timestamp.strftime('%a %b %d %H:%M:%S')}.{tenth} {timestamp.strftime('%Y')}"


def process_help_separator(terminal_width: int | None) -> str:
    return "  " if terminal_width is not None and terminal_width >= 90 else " "


def append_process_help(
    details: Text,
    gpus: Sequence[GpuInfo] | None = None,
    separator: str = " ",
) -> None:
    gpu_keys = gpu_filter_key_label(gpus)
    leading_space = False
    if gpu_keys:
        append_keybinding(details, gpu_keys, "gpu", leading_space=False, separator=separator)
        leading_space = True
    append_keybinding(details, "s", "sort", leading_space=leading_space, separator=separator)
    append_keybinding(details, "t", "tree", separator=separator)
    append_keybinding(details, "/", "search", separator=separator)
    append_keybinding(details, "f", "filter", separator=separator)
    append_keybinding(details, "z", "zoom", separator=separator)
    append_keybinding(details, "i", "inspect", separator=separator)
    append_keybinding(details, "x", "kill", separator=separator)
    append_keybinding(details, "?", "help", separator=separator)
    append_keybinding(details, "q", "quit", separator=separator)


def append_keybinding(
    details: Text,
    key: str,
    action: str,
    leading_space: bool = True,
    separator: str = " ",
) -> None:
    if leading_space:
        details.append(separator)
    details.append(f"{key}: ", style=f"bold {DRACULA_ORANGE}")
    details.append(action, style=DRACULA_DIM)


def gpu_filter_key_label(gpus: Sequence[GpuInfo] | None) -> str:
    indices = sorted({gpu.index for gpu in gpus or () if 0 <= gpu.index <= 9})
    if not indices:
        return ""
    if indices == list(range(indices[0], indices[-1] + 1)):
        if indices[0] == indices[-1]:
            return f"<{indices[0]}>"
        return f"<{indices[0]}-{indices[-1]}>"
    return f"<{','.join(str(index) for index in indices)}>"


def render_help_popup(
    process_state: ProcessViewState,
    terminal_width: int | None = None,
    gpus: Sequence[GpuInfo] | None = None,
) -> Panel:
    max_offset = max(0, len(HELP_ENTRIES) - HELP_VISIBLE_ROWS)
    start = min(max(0, process_state.help_scroll_offset), max_offset)
    end = min(len(HELP_ENTRIES), start + HELP_VISIBLE_ROWS)
    available_width = terminal_width or 88
    panel_width = min(128, max(1, available_width - 4))
    help_rows = help_popup_rows(gpus)
    key_width, action_width, mode_width = help_popup_column_widths(help_rows, panel_width)

    table = Table(box=box.SIMPLE, expand=True, show_lines=False, padding=(0, 1))
    table.add_column("KEY", style=f"bold {DRACULA_ORANGE}", no_wrap=True, width=key_width)
    table.add_column("ACTION", style=DRACULA_FG, width=action_width)
    table.add_column("MODE", style=DRACULA_DIM, no_wrap=True, width=mode_width)
    for key, action, mode in help_rows[start:end]:
        table.add_row(key, action, mode)
    table.caption = "j/k or Up/Down: scroll   h/l or Left/Right: page   ?/Esc: close"
    table.caption_style = DRACULA_DIM

    return Panel(
        table,
        title=Text(f"Help  {start + 1}-{end}/{len(HELP_ENTRIES)}", style=f"bold {DRACULA_CYAN}"),
        border_style=DRACULA_CYAN,
        box=box.SQUARE,
        width=panel_width,
    )


def render_process_info_popup(
    snapshot: Snapshot,
    process_state: ProcessViewState,
    terminal_width: int | None = None,
) -> Panel:
    panel_width = process_info_panel_width(terminal_width)
    info_rows = process_info_rows(snapshot, process_state)
    label_width, value_width = process_info_column_widths(info_rows, panel_width)
    rows = process_info_visual_rows(
        info_rows,
        value_width,
    )
    process_state.process_info_render_row_count = len(rows)
    max_offset = max(0, len(rows) - PROCESS_INFO_VISIBLE_ROWS)
    start = min(max(0, process_state.process_info_scroll_offset), max_offset)
    process_state.process_info_scroll_offset = start
    end = min(len(rows), start + PROCESS_INFO_VISIBLE_ROWS)

    table = Table(box=box.SIMPLE, expand=True, show_header=False, show_lines=False, padding=(0, 1))
    table.add_column("FIELD", style=DRACULA_DIM, no_wrap=True, width=label_width)
    table.add_column("VALUE", style=DRACULA_FG, no_wrap=True, overflow="crop", width=value_width)
    visible_rows = rows[start:end]
    for label, value in visible_rows:
        table.add_row(label, value or "-")
    for _ in range(PROCESS_INFO_VISIBLE_ROWS - len(visible_rows)):
        table.add_row("", "")
    table.caption = "j/k or Up/Down: scroll   h/l or Left/Right: page   i/Esc: close"
    table.caption_style = DRACULA_DIM

    proc = process_state.process_info_process
    title = f"Process  {proc.pid}" if proc is not None else "Process"
    return Panel(
        table,
        title=Text(title, style=f"bold {DRACULA_CYAN}"),
        border_style=DRACULA_CYAN,
        box=box.SQUARE,
        width=panel_width,
    )


def process_info_panel_width(terminal_width: int | None = None) -> int:
    available_width = terminal_width or 88
    return min(128, max(1, available_width - 4))


def help_popup_rows(gpus: Sequence[GpuInfo] | None = None) -> list[tuple[str, str, str]]:
    rows = []
    for key, action, mode in HELP_ENTRIES:
        display_key = gpu_filter_key_label(gpus) if key == "0-9" else key
        rows.append((display_key or key, action, mode))
    return rows


def help_popup_column_widths(rows: list[tuple[str, str, str]], panel_width: int) -> tuple[int, int, int]:
    key_width = max((len(key) for key, _action, _mode in rows), default=0)
    key_width = max(key_width, len("KEY"))
    mode_width = max((len(mode) for _key, _action, mode in rows), default=0)
    mode_width = max(mode_width, len("MODE"))
    max_action_width = max((len(action) for _key, action, _mode in rows), default=0)
    max_action_width = max(max_action_width, len("ACTION"))
    action_width = min(max_action_width, max(1, panel_width - key_width - mode_width - 16))
    return key_width, action_width, mode_width


def process_info_column_widths(rows: list[tuple[str, str]], panel_width: int) -> tuple[int, int]:
    label_width = max((len(label) for label, _value in rows), default=0)
    label_width = min(max(label_width, 8), 18)
    value_width = max(8, panel_width - label_width - 12)
    return label_width, value_width


def process_info_visual_rows(rows: list[tuple[str, str]], value_width: int) -> list[tuple[str, str]]:
    visual_rows: list[tuple[str, str]] = []
    for label, value in rows:
        wrapped_lines = wrap_process_info_value(value or "-", value_width)
        visual_rows.append((label, wrapped_lines[0]))
        for wrapped_line in wrapped_lines[1:]:
            visual_rows.append(("", wrapped_line))
    return visual_rows or [("Status", "-")]


def wrap_process_info_value(value: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in value.splitlines() or ["-"]:
        lines.extend(
            textwrap.wrap(
                raw_line,
                width=max(1, width),
                break_long_words=True,
                break_on_hyphens=False,
                drop_whitespace=True,
            )
            or [""]
        )
    return lines or ["-"]


def process_info_rows(snapshot: Snapshot, process_state: ProcessViewState) -> list[tuple[str, str]]:
    proc = process_state.process_info_process
    if proc is None:
        return [("Status", "No process selected")]

    detail = process_state.process_info_detail
    gpu = next((gpu for gpu in snapshot.gpus if gpu.index == proc.gpu_index), None)
    rows = [
        ("PID", str(proc.pid)),
        ("PPID", "-" if proc.ppid is None else str(proc.ppid)),
        ("User", proc.user or "-"),
        ("Name", proc.name or proc.command or "-"),
        ("Row type", "GPU process" if proc.gpu_index is not None else "Ancestor/context"),
        ("GPU", "-" if proc.gpu_index is None else str(proc.gpu_index)),
        ("GPU model", process_info_gpu_model(gpu)),
        ("GPU GUID", gpu.guid if gpu is not None and gpu.guid else "-"),
        ("GPU memory", process_info_gpu_memory(proc)),
        ("CPU", process_info_percent(proc.cpu_percent)),
        ("Host memory", process_info_percent(proc.host_mem_percent)),
        ("Elapsed", proc.elapsed or "-"),
        ("Parent", process_info_parent(process_state)),
        ("Visible children", str(process_state.process_info_child_count)),
        ("Command", process_command(proc) or "-"),
    ]
    if detail is None:
        rows.append(("Status", "No /proc detail loaded"))
        return rows

    rows.extend(
        [
            ("Cmdline", detail.cmdline or "-"),
            ("State", detail.state or "-"),
            ("Threads", "-" if detail.threads is None else str(detail.threads)),
            ("VmRSS", process_info_kib(detail.vm_rss_kib)),
            ("VmSize", process_info_kib(detail.vm_size_kib)),
            ("VmHWM", process_info_kib(detail.vm_hwm_kib)),
            ("CPU affinity", detail.cpu_allowed_list or "-"),
            ("Cwd", detail.cwd or "-"),
            ("Exe", detail.exe or "-"),
            (
                "Ctx switches",
                f"{process_info_int(detail.voluntary_ctxt_switches)} voluntary / "
                f"{process_info_int(detail.nonvoluntary_ctxt_switches)} nonvoluntary",
            ),
        ]
    )
    if detail.error:
        rows.append(("Status", detail.error))
    return rows


def process_info_gpu_model(gpu: GpuInfo | None) -> str:
    if gpu is None:
        return "-"
    return gpu.gpu_type or gpu.name or "-"


def process_info_gpu_memory(proc: ProcessInfo) -> str:
    return f"{format_bytes_mib(proc.gpu_memory_bytes)} ({percent_text(proc.gpu_memory_percent, digits=1)})"


def process_info_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def process_info_kib(value: int | None) -> str:
    if value is None:
        return "-"
    return format_bytes_mib(value * 1024)


def process_info_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def process_info_parent(process_state: ProcessViewState) -> str:
    parent = process_state.process_info_parent
    if parent is not None:
        return f"{parent.pid} {process_command(parent)}".strip()
    proc = process_state.process_info_process
    if proc is not None and proc.ppid is not None:
        return str(proc.ppid)
    return "-"


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
            bar_with_percent(gpu.utilization_percent, util_style, digits=1),
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


def render_metrics_graphs(
    history: MetricsHistory,
    end_time: datetime | None = None,
    samples: Sequence[MetricSample] | None = None,
    time_offset_seconds: int = 0,
) -> Table:
    metric_samples = tuple(samples) if samples is not None else history.samples
    table = Table(box=box.SQUARE, expand=True, show_header=False, padding=(0, 1))
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_row(
        MetricGraphPair(
            samples=metric_samples,
            end_time=end_time,
            top_metric_name="avg_cpu_percent",
            top_label="Avg %CPU",
            top_style=DRACULA_CYAN,
            bottom_metric_name="avg_mem_percent",
            bottom_label="Avg %MEM",
            bottom_style=DRACULA_PINK,
            time_offset_seconds=time_offset_seconds,
        ),
        MetricGraphPair(
            samples=metric_samples,
            end_time=end_time,
            top_metric_name="avg_gpu_percent",
            top_label="Avg %GPU",
            top_style=DRACULA_ORANGE,
            bottom_metric_name="avg_gpu_mem_percent",
            bottom_label="Avg %GPU MEM",
            bottom_style=DRACULA_YELLOW,
            time_offset_seconds=time_offset_seconds,
        ),
    )
    return table


@dataclass(frozen=True, slots=True)
class MetricGraphPair:
    samples: Sequence[MetricSample]
    end_time: datetime | None
    top_metric_name: str
    top_label: str
    top_style: str
    bottom_metric_name: str
    bottom_label: str
    bottom_style: str
    time_offset_seconds: int = 0
    height: int = 15

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = max(12, options.max_width)
        graph_columns = width * GRAPH_COLUMNS_PER_CELL
        top_values = metric_values_by_time(self.samples, self.top_metric_name, graph_columns, self.end_time)
        bottom_values = metric_values_by_time(self.samples, self.bottom_metric_name, graph_columns, self.end_time)
        yield metric_label(
            self.top_label,
            latest_metric_sample_value(self.samples, self.top_metric_name, self.end_time),
            self.top_style,
        )
        yield from metric_graph_lines(top_values, width=width, height=self.height, style=self.top_style, trim_empty=False)
        yield time_axis_line(width, offset_seconds=self.time_offset_seconds)
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
        yield metric_label(
            self.bottom_label,
            latest_metric_sample_value(self.samples, self.bottom_metric_name, self.end_time),
            self.bottom_style,
        )


def metric_label(label: str, value: float | None, style: str) -> Text:
    text = Text(no_wrap=True, overflow="ellipsis")
    text.append(f"{label}: ", style=DRACULA_FG)
    if value is None:
        text.append("N/A", style=DRACULA_DIM)
    else:
        text.append(percent_text(value, digits=1), style=style)
    return text


def time_axis_line(width: int, offset_seconds: int = 0) -> Text:
    width = max(1, width)
    offset_seconds = max(0, offset_seconds)
    chars = ["─"] * width
    for seconds in TIME_AXIS_MARKERS_SECONDS:
        marker_seconds = seconds - offset_seconds
        if marker_seconds <= 0:
            continue
        label = f"{seconds}s"
        marker = width - 1 - marker_seconds // GRAPH_COLUMNS_PER_CELL
        start = marker - len(label)
        space = start - 1
        if space < 0 or start + len(label) > width:
            continue
        chars[space] = " "
        chars[start:marker] = list(label)
        chars[marker] = "├"
    return Text("".join(chars), style=DRACULA_DIM, no_wrap=True, overflow="crop")


def latest_metric_sample_value(
    samples: Sequence[MetricSample],
    metric_name: str,
    end_time: datetime | None = None,
) -> float | None:
    if not samples:
        return None
    graph_end_time = end_time or samples[-1].timestamp
    graph_end_second = graph_end_time.replace(microsecond=0)
    for sample in reversed(samples):
        if sample.timestamp.replace(microsecond=0) > graph_end_second:
            continue
        value = getattr(sample, metric_name)
        if value is not None:
            return value
    return None


def metric_values_by_time(
    samples: Sequence[MetricSample],
    metric_name: str,
    seconds: int,
    end_time: datetime | None = None,
) -> list[float | None]:
    seconds = max(1, seconds)
    values: list[float | None] = [None] * seconds
    if not samples:
        return values
    graph_end_time = end_time or samples[-1].timestamp
    graph_end_second = graph_end_time.replace(microsecond=0)
    totals = [0.0] * seconds
    counts = [0] * seconds
    for sample in samples:
        sample_second = sample.timestamp.replace(microsecond=0)
        offset = int((graph_end_second - sample_second).total_seconds())
        if offset < 0:
            continue
        if offset >= seconds:
            continue
        value = getattr(sample, metric_name)
        if value is None:
            continue
        index = seconds - 1 - offset
        totals[index] += value
        counts[index] += 1

    for index, count in enumerate(counts):
        if count:
            values[index] = totals[index] / count

    last_value: float | None = None
    for index, value in enumerate(values):
        if value is None:
            values[index] = last_value
        else:
            last_value = value
    return values


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
            cell_values = padded_values[column_start : column_start + GRAPH_COLUMNS_PER_CELL]
            for column, value in enumerate(cell_values):
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
    process_ancestors: list[ProcessInfo] | None = None,
    elapsed_offset_seconds: int = 0,
) -> Table:
    display_processes = list(processes)
    process_count = len(display_processes)
    display_rows: list[ProcessRenderRow]
    title = None
    if process_state is not None:
        if not processes_sorted:
            display_processes = process_state.display_processes(display_processes, process_ancestors)
        process_state.sync(display_processes, viewport_rows=max_rows, adjust_scroll=False)
        title = render_process_title(process_state, len(display_processes))
        tree_prefixes = process_tree_prefixes(display_processes) if process_state.tree_mode else {}
        table_widths = process_table_widths(
            display_processes,
            process_state,
            terminal_width,
            elapsed_offset_seconds,
        )
        display_rows = visible_process_window(
            display_processes,
            process_state,
            max_rows,
            table_widths.command,
            tree_prefixes=tree_prefixes,
        )
    else:
        if max_rows is not None and len(display_processes) > max_rows:
            display_processes = display_processes[: max(1, max_rows)]
            title = render_static_process_title(len(display_processes), process_count)
        table_widths = process_table_widths(
            display_processes,
            process_state,
            terminal_width,
            elapsed_offset_seconds,
        )
        if max_rows is None:
            display_rows = [ProcessRenderRow(proc, process_command(proc), 1) for proc in display_processes]
        else:
            display_rows = [wrapped_process_row(proc, table_widths.command, max_lines=1) for proc in display_processes]

    table = Table(
        box=box.SQUARE,
        expand=True,
        show_lines=False,
        padding=(0, 1),
        title=title,
        title_justify="left",
    )
    table.add_column(
        process_column_header("GPU", "gpu", process_state),
        justify="right",
        style="bold",
        no_wrap=True,
        width=table_widths.gpu,
    )
    table.add_column(
        process_column_header("PID", "pid", process_state),
        justify="right",
        no_wrap=True,
        width=table_widths.pid,
    )
    table.add_column(
        process_column_header("USER", "user", process_state),
        no_wrap=True,
        width=table_widths.user,
    )
    table.add_column(
        process_column_header("GPU-MEM", "gpu_memory", process_state),
        justify="right",
        no_wrap=True,
        width=table_widths.gpu_memory,
    )
    table.add_column(
        process_column_header("%GPU-MEM", "gpu_memory_percent", process_state),
        justify="right",
        no_wrap=True,
        width=table_widths.gpu_memory_percent,
    )
    table.add_column(
        process_column_header("%CPU", "cpu", process_state),
        justify="right",
        no_wrap=True,
        width=table_widths.cpu,
    )
    table.add_column(
        process_column_header("%MEM", "mem", process_state),
        justify="right",
        no_wrap=True,
        width=table_widths.mem,
    )
    table.add_column(
        process_column_header("TIME", "time", process_state),
        justify="right",
        no_wrap=True,
        width=table_widths.time,
    )
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
        command_cell = Text(command, no_wrap=True, overflow="crop") if process_state is not None else command
        gpu_mem_style = percent_style(proc.gpu_memory_percent)
        table.add_row(
            gpu,
            str(proc.pid),
            proc.user or "-",
            Text(format_bytes_mib(proc.gpu_memory_bytes), style=gpu_mem_style),
            Text(percent_text(proc.gpu_memory_percent, digits=1), style=gpu_mem_style),
            metric_text(proc.cpu_percent, digits=1),
            metric_text(proc.host_mem_percent, digits=1),
            process_elapsed_text(proc.elapsed, elapsed_offset_seconds),
            command_cell,
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
    target_pid = process_state.kill_confirm_pid if process_state.kill_confirm_pid is not None else process_state.selected_pid
    if target_pid is None:
        menu.append("Kill process: ", style=f"bold {DRACULA_RED}")
    else:
        menu.append(f"Kill PID {target_pid}: ", style=f"bold {DRACULA_RED}")
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


def process_table_widths(
    processes: Sequence[ProcessInfo],
    process_state: ProcessViewState | None,
    terminal_width: int | None,
    elapsed_offset_seconds: int = 0,
) -> ProcessTableWidths:
    widths = [
        len(process_column_header("GPU", "gpu", process_state).plain),
        len(process_column_header("PID", "pid", process_state).plain),
        len(process_column_header("USER", "user", process_state).plain),
        len(process_column_header("GPU-MEM", "gpu_memory", process_state).plain),
        len(process_column_header("%GPU-MEM", "gpu_memory_percent", process_state).plain),
        len(process_column_header("%CPU", "cpu", process_state).plain),
        len(process_column_header("%MEM", "mem", process_state).plain),
        len(process_column_header("TIME", "time", process_state).plain),
    ]
    for proc in processes:
        for index, value in enumerate(process_metadata_values(proc, elapsed_offset_seconds)):
            widths[index] = max(widths[index], len(value))
    command_width = process_table_command_width(terminal_width, widths)
    return ProcessTableWidths(*widths, command_width)


def process_metadata_values(
    proc: ProcessInfo,
    elapsed_offset_seconds: int = 0,
) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        "-" if proc.gpu_index is None else str(proc.gpu_index),
        str(proc.pid),
        proc.user or "-",
        format_bytes_mib(proc.gpu_memory_bytes),
        percent_text(proc.gpu_memory_percent, digits=1),
        process_metric_value(proc.cpu_percent, digits=1),
        process_metric_value(proc.host_mem_percent, digits=1),
        process_elapsed_text(proc.elapsed, elapsed_offset_seconds),
    )


def process_metric_value(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def process_elapsed_text(value: str, offset_seconds: int = 0) -> str:
    text = str(value or "").strip()
    if not text or text == "-":
        return "-"
    seconds = parse_process_elapsed_seconds(text)
    if seconds is None:
        return text
    return format_process_elapsed_seconds(seconds + max(0, offset_seconds))


def parse_process_elapsed_seconds(value: str) -> int | None:
    text = str(value or "").strip()
    if not text or text == "-":
        return None
    time_text = text
    if "-" in time_text:
        day_text, time_text = time_text.split("-", 1)
        if not day_text.isdigit():
            return None
    parts = time_text.split(":")
    if not parts or len(parts) > 3:
        return None
    if not all(part.isdigit() for part in parts):
        return None
    return elapsed_seconds(text)


def format_process_elapsed_seconds(total_seconds: int) -> str:
    total_seconds = max(0, total_seconds)
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def process_table_command_width(terminal_width: int | None, metadata_widths: Sequence[int]) -> int:
    if terminal_width is None:
        return estimate_process_command_width(terminal_width)
    table_chrome_width = PROCESS_TABLE_COLUMN_COUNT + 1 + PROCESS_TABLE_COLUMN_COUNT * PROCESS_TABLE_CELL_PADDING_WIDTH
    return max(
        PROCESS_TABLE_MIN_COMMAND_WIDTH,
        terminal_width - sum(metadata_widths) - table_chrome_width,
    )


def visible_process_window(
    processes: list[ProcessInfo],
    process_state: ProcessViewState,
    max_visual_rows: int | None,
    command_width: int,
    tree_prefixes: dict[tuple[int, int | None], str] | None = None,
) -> list[ProcessRenderRow]:
    if not processes:
        process_state.scroll_offset = 0
        return []

    with profile_span("process-window"):
        selected_index = max(0, min(process_state.selected_index, len(processes) - 1))
        max_visual_rows = max(1, max_visual_rows or process_state.viewport_rows)
        wrapped_rows: dict[tuple[int, int], ProcessRenderRow] = {}

        def row_at(index: int, max_lines: int | None = None) -> ProcessRenderRow:
            row_max_lines = max(1, max_lines or max_visual_rows)
            key = (index, row_max_lines)
            row = wrapped_rows.get(key)
            if row is None:
                prefix = ""
                if tree_prefixes:
                    prefix = tree_prefixes.get(process_selection_key(processes[index]), "")
                row = wrapped_process_row(
                    processes[index],
                    command_width,
                    max_lines=row_max_lines,
                    tree_prefix=prefix,
                )
                wrapped_rows[key] = row
            return row

        def window_from(start_index: int) -> tuple[list[ProcessRenderRow], bool]:
            used_rows = 0
            rows: list[ProcessRenderRow] = []
            selected_visible = False
            index = start_index
            while index < len(processes):
                row = row_at(index)
                if used_rows > 0 and used_rows + row.visual_height > max_visual_rows:
                    remaining_rows = max_visual_rows - used_rows
                    if remaining_rows > 0 and index != selected_index:
                        rows.append(row_at(index, remaining_rows))
                    break
                used_rows += row.visual_height
                rows.append(row)
                if index == selected_index:
                    selected_visible = True
                index += 1
            return rows, selected_visible

        start = max(0, min(process_state.scroll_offset, len(processes) - 1))
        rows, selected_visible = window_from(start)
        if not selected_visible:
            if selected_index < start:
                start = selected_index
            else:
                start = selected_index
                used_rows = row_at(selected_index).visual_height
                while start > 0:
                    row = row_at(start - 1)
                    if used_rows + row.visual_height > max_visual_rows:
                        break
                    start -= 1
                    used_rows += row.visual_height
            rows, _ = window_from(start)

        process_state.scroll_offset = start
        return rows


def process_command(proc: ProcessInfo) -> str:
    return proc.args or proc.command or proc.name or "N/A"


def wrapped_process_row(
    proc: ProcessInfo,
    command_width: int,
    max_lines: int | None = None,
    tree_prefix: str = "",
) -> ProcessRenderRow:
    command = wrap_prefixed_process_command(process_command(proc), command_width, max_lines, tree_prefix)
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


def wrap_prefixed_process_command(command: str, width: int, max_lines: int | None, prefix: str) -> str:
    if not prefix:
        return wrap_process_command(command, width, max_lines)
    command_width = max(1, width - len(prefix))
    lines = wrap_command_lines(command, command_width)
    if max_lines is not None:
        lines = truncate_command_lines(lines, command_width, max_lines)
    continuation = tree_continuation_prefix(prefix)
    return "\n".join(
        f"{prefix}{line}" if index == 0 else f"{continuation}{line}"
        for index, line in enumerate(lines)
    )


def tree_continuation_prefix(prefix: str) -> str:
    parts = [prefix[index : index + 3] for index in range(0, len(prefix), 3)]
    if not parts:
        return ""
    last = parts[-1]
    if last == "├─ ":
        parts[-1] = "│  "
    elif last == "└─ ":
        parts[-1] = "   "
    return "".join(parts)


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


def process_tree_prefixes(processes: list[ProcessInfo]) -> dict[tuple[int, int | None], str]:
    rows_by_key = {process_selection_key(proc): proc for proc in processes}
    key_by_pid: dict[int, tuple[int, int | None]] = {}
    for proc in processes:
        key_by_pid.setdefault(proc.pid, process_selection_key(proc))

    root_keys: list[tuple[int, int | None]] = []
    children_by_parent: dict[tuple[int, int | None], list[tuple[int, int | None]]] = {}
    for proc in processes:
        key = process_selection_key(proc)
        parent_key = key_by_pid.get(proc.ppid or -1)
        if parent_key is None or parent_key == key:
            root_keys.append(key)
            continue
        children_by_parent.setdefault(parent_key, []).append(key)

    prefixes: dict[tuple[int, int | None], str] = {}
    visited: set[tuple[int, int | None]] = set()

    def assign(
        key: tuple[int, int | None],
        parent_last_flags: list[bool],
        is_root: bool,
        is_last: bool,
    ) -> None:
        if key in visited:
            return
        visited.add(key)
        if is_root:
            prefixes[key] = ""
        else:
            prefix_parts = ["   " if was_last else "│  " for was_last in parent_last_flags]
            prefix_parts.append("└─ " if is_last else "├─ ")
            prefixes[key] = "".join(prefix_parts)
        children = children_by_parent.get(key, [])
        for index, child_key in enumerate(children):
            assign(
                child_key,
                parent_last_flags + ([is_last] if not is_root else []),
                False,
                index == len(children) - 1,
            )

    for index, key in enumerate(root_keys):
        assign(key, [], True, index == len(root_keys) - 1)
    for key in rows_by_key:
        assign(key, [], True, True)
    return prefixes


def metric_text(value: float | int | None, digits: int = 1) -> Text:
    if value is None:
        return Text("-", style=DRACULA_DIM)
    return Text(process_metric_value(value, digits), style=percent_style(value))


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


def progress_text(percent: float, style: str) -> ProgressText:
    return ProgressText(clamp_percent(percent), style)


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

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from rich.console import Console
from rich.live import Live

from . import __version__
from .collectors import CollectionError, CommandInterrupted, CommandTimeout, collect_snapshot, read_process_detail
from .history import GpuMetricSample, MetricSample, MetricsHistory
from .interaction import (
    KEY_DOWN,
    KEY_ENTER,
    KEY_LEFT,
    KEY_PAGE_DOWN,
    KEY_PAGE_UP,
    KEY_RIGHT,
    KEY_UP,
    MODE_FILTER,
    MODE_HELP,
    MODE_KILL_CONFIRM,
    MODE_NORMAL,
    MODE_PROCESS_INFO,
    MODE_SEARCH,
    MODE_SORT_MENU,
    ProcessViewState,
    TerminalKeyboard,
)
from .models import ProcessInfo, Snapshot
from .profiling import profile_span
from .render import GRAPH_COLUMNS_PER_CELL, GRAPH_HISTORY_SECONDS, GPU_GRAPH_WIDE_MIN_WIDTH, render_snapshot


KEY_POLL_SECONDS = 0.05
COLLECTOR_STOP_JOIN_SECONDS = 0.05
GRAPH_FRAME_SECONDS = 1.0
GRAPH_HISTORY_BUCKETS = GRAPH_HISTORY_SECONDS + 1
GRAPH_HISTORY_LABEL_GUTTER_SECONDS = (len(f"{GRAPH_HISTORY_SECONDS}s") + 1) * GRAPH_COLUMNS_PER_CELL
GRAPH_PAN_SECONDS = 10
GRAPH_METRIC_FIELDS = (
    "avg_cpu_percent",
    "avg_mem_percent",
    "avg_gpu_percent",
    "avg_gpu_mem_percent",
)


@dataclass(slots=True)
class SnapshotUpdate:
    sequence: int
    snapshot: Snapshot


@dataclass(frozen=True, slots=True)
class GraphFrame:
    display_time: datetime
    history_samples: tuple[MetricSample, ...]


@dataclass(frozen=True, slots=True)
class GraphSampleValues:
    metric_values: tuple[float | None, ...]
    gpu_metrics: tuple[GpuMetricSample, ...]


class BackgroundSnapshotCollector:
    def __init__(
        self,
        interval: float,
        collect_func: Callable[[], Snapshot] | None = None,
        history: MetricsHistory | None = None,
    ) -> None:
        self.interval = interval
        self.collect_func = collect_func or collect_snapshot
        self.history = history
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._latest: SnapshotUpdate | None = None
        self._error: CollectionError | None = None
        self._sequence = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="roctop-collector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=COLLECTOR_STOP_JOIN_SECONDS)

    def latest_after(self, sequence: int) -> SnapshotUpdate | None:
        with self._lock:
            if self._latest is None or self._latest.sequence <= sequence:
                return None
            return self._latest

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
            self._error = None
        if error is not None:
            raise error

    def _run(self) -> None:
        next_collect_at = time.monotonic() + self.interval
        while not self._stop.wait(max(0.0, next_collect_at - time.monotonic())):
            collect_started_at = time.monotonic()
            snapshot: Snapshot | None = None
            try:
                snapshot = self.collect_func()
            except CommandTimeout:
                pass
            except CollectionError:
                pass
            except Exception:
                pass

            if snapshot is not None:
                if self.history is not None:
                    try:
                        self.history.add_snapshot(snapshot)
                    except Exception:
                        pass
                with self._lock:
                    self._sequence += 1
                    self._latest = SnapshotUpdate(sequence=self._sequence, snapshot=snapshot)
            next_collect_at = collect_started_at + self.interval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor AMD ROCm GPU usage.")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Refresh interval in seconds for the live view. Default: 1.0",
    )
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit.")
    parser.add_argument("--json", action="store_true", help="Print one normalized JSON snapshot and exit.")
    parser.add_argument("--version", action="version", version=f"roctop {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(color_system="truecolor")
    error_console = Console(stderr=True, color_system="truecolor")

    if args.interval <= 0:
        error_console.print("[red]--interval must be greater than 0[/red]")
        return 2

    try:
        if args.json:
            snapshot = collect_snapshot_retry(args.interval)
            print(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))
            return 0

        if args.once:
            snapshot = collect_snapshot_retry(args.interval)
            console.print(
                render_snapshot(
                    snapshot,
                    terminal_height=console.size.height,
                    terminal_width=console.size.width,
                )
            )
            return 0

        return run_live(console, args.interval)
    except (KeyboardInterrupt, CommandInterrupted):
        return 0
    except CollectionError as exc:
        error_console.print(f"[red]{exc}[/red]")
        return 1


def run_live(console: Console, interval: float) -> int:
    history = MetricsHistory(max_samples=graph_history_sample_limit(interval))
    process_state = ProcessViewState()
    history.prime_cpu()
    snapshot = collect_snapshot_retry(interval)
    history.add_snapshot(snapshot)
    collector = BackgroundSnapshotCollector(interval, history=history)
    display_time = datetime.now()
    graph_frame = capture_graph_frame(history, display_time)
    with (
        TerminalKeyboard() as keyboard,
        Live(
            render_live_snapshot(
                snapshot,
                history,
                process_state,
                console,
                interval=interval,
                display_time=display_time,
                graph_frame=graph_frame,
            ),
            console=console,
            screen=True,
            auto_refresh=False,
        ) as live,
    ):
        collector.start()
        try:
            poll_live_until_quit(
                live,
                keyboard,
                snapshot,
                history,
                process_state,
                console,
                collector,
                interval,
                display_time=display_time,
                graph_frame=graph_frame,
            )
        finally:
            collector.stop()
    return 0


def poll_live_until_quit(
    live: Live,
    keyboard: TerminalKeyboard,
    snapshot: Snapshot,
    history: MetricsHistory,
    process_state: ProcessViewState,
    console: Console,
    collector: BackgroundSnapshotCollector,
    interval: float,
    display_time: datetime | None = None,
    graph_frame: GraphFrame | None = None,
) -> None:
    rendered_size = console_dimensions(console)
    latest_sequence = 0
    render_interval = live_render_interval(interval)
    graph_interval = graph_frame_interval(interval)
    next_render_at = time.monotonic() + render_interval
    next_graph_at = time.monotonic() + graph_interval
    display_time = display_time or datetime.now()
    graph_frame = graph_frame or capture_graph_frame(history, display_time)
    while True:
        collector.raise_if_failed()
        update = collector.latest_after(latest_sequence)
        if update is not None:
            snapshot = update.snapshot
            latest_sequence = update.sequence

        pending_key_timeout = KEY_POLL_SECONDS if getattr(keyboard, "pending_input", "") else 0.0
        if handle_live_keyboard_input(
            live,
            keyboard,
            snapshot,
            history,
            process_state,
            console,
            interval,
            display_time,
            graph_frame,
            timeout=pending_key_timeout,
        ):
            return

        now = time.monotonic()
        current_size = console_dimensions(console)
        status_expired = process_state.expire_status_message(now)
        refresh_due = now >= next_render_at
        graph_due = now >= next_graph_at
        if refresh_due or graph_due:
            display_time = datetime.now()
        if graph_due:
            graph_frame = advance_graph_frame(history, graph_frame, display_time)
        if current_size != rendered_size or status_expired or refresh_due or graph_due:
            if not refresh_live(
                live,
                render_live_snapshot(
                    snapshot,
                    history,
                    process_state,
                    console,
                    interval=interval,
                    display_time=display_time,
                    graph_frame=graph_frame,
                ),
            ):
                return
            rendered_size = current_size
            if refresh_due:
                next_render_at = now + render_interval
            if graph_due:
                next_graph_at = now + graph_interval

        next_wakeup_at = min(next_render_at, next_graph_at)
        timeout = min(KEY_POLL_SECONDS, max(0.0, next_wakeup_at - time.monotonic()))
        if handle_live_keyboard_input(
            live,
            keyboard,
            snapshot,
            history,
            process_state,
            console,
            interval,
            display_time,
            graph_frame,
            timeout=timeout,
        ):
            return


def handle_live_keyboard_input(
    live: Live,
    keyboard: TerminalKeyboard,
    snapshot: Snapshot,
    history: MetricsHistory,
    process_state: ProcessViewState,
    console: Console,
    interval: float,
    display_time: datetime,
    graph_frame: GraphFrame,
    timeout: float,
) -> bool:
    keys = keyboard.read_keys(timeout=timeout)
    if not keys:
        return False

    quit_requested, processes = handle_key_batch(
        snapshot,
        process_state,
        keys,
        history=history,
        graph_frame=graph_frame,
        terminal_width=console_dimensions(console)[1],
    )
    if not refresh_live(
        live,
        render_live_snapshot(
            snapshot,
            history,
            process_state,
            console,
            display_processes=processes,
            interval=interval,
            display_time=display_time,
            graph_frame=graph_frame,
        ),
    ):
        return True
    return quit_requested


def poll_input_until_refresh(
    live: Live,
    keyboard: TerminalKeyboard,
    snapshot,
    history: MetricsHistory,
    process_state: ProcessViewState,
    console: Console,
    interval: float,
) -> bool:
    deadline = time.monotonic() + interval
    rendered_size = console_dimensions(console)
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            return False
        current_size = console_dimensions(console)
        status_expired = process_state.expire_status_message(now)
        if current_size != rendered_size or status_expired:
            if not refresh_live(
                live,
                render_snapshot(snapshot, history, process_state, *current_size),
            ):
                return True
            rendered_size = current_size
        keys = keyboard.read_keys(timeout=min(KEY_POLL_SECONDS, remaining))
        if not keys:
            continue
        quit_requested, processes = handle_key_batch(snapshot, process_state, keys, terminal_width=rendered_size[1])
        rendered_size = console_dimensions(console)
        if not refresh_live(
            live,
            render_snapshot(snapshot, history, process_state, *rendered_size, display_processes=processes),
        ):
            return True
        if quit_requested:
            return True


def refresh_live(live: Live, renderable) -> bool:
    try:
        live.update(renderable, refresh=True)
    except (BrokenPipeError, OSError):
        return False
    return True


def handle_key_batch(
    snapshot: Snapshot,
    process_state: ProcessViewState,
    keys: list[str],
    history: MetricsHistory | None = None,
    graph_frame: GraphFrame | None = None,
    terminal_width: int | None = None,
) -> tuple[bool, list[ProcessInfo]]:
    quit_requested = False
    gpu_indices = snapshot_gpu_indices(snapshot)
    processes = process_state.display_processes(snapshot.processes, snapshot.process_ancestors)
    process_state.sync(processes, adjust_scroll=False)
    processes_dirty = False
    with profile_span("key-handling"):
        for key in keys:
            if processes_dirty and key_needs_current_processes(process_state, key):
                processes = process_state.display_processes(snapshot.processes, snapshot.process_ancestors)
                process_state.sync(processes, adjust_scroll=False)
                processes_dirty = False
            if key == "i" and process_state.mode == MODE_NORMAL:
                open_selected_process_info(snapshot, process_state, processes)
                continue
            if handle_graph_pan_key(key, process_state, history, graph_frame, terminal_width):
                continue
            view_before = process_view_key(process_state)
            result = process_state.handle_key(
                key,
                processes,
                processes_synced=True,
                gpu_indices=gpu_indices,
                all_processes=snapshot.processes,
            )
            quit_requested = quit_requested or result.quit
            if process_view_key(process_state) != view_before:
                processes_dirty = True
        if processes_dirty:
            processes = process_state.display_processes(snapshot.processes, snapshot.process_ancestors)
            process_state.sync(processes, adjust_scroll=False)
    return quit_requested, processes


def handle_graph_pan_key(
    key: str,
    process_state: ProcessViewState,
    history: MetricsHistory | None,
    graph_frame: GraphFrame | None,
    terminal_width: int | None = None,
) -> bool:
    if key not in (",", ".", "r"):
        return False
    if process_state.mode != MODE_NORMAL or process_state.process_zoomed:
        return False
    if key == "r":
        process_state.graph_view_offset_seconds = 0
        return True
    if history is None or graph_frame is None:
        return False

    samples = history.samples
    if not samples:
        return True

    max_offset = graph_view_max_offset_seconds(
        samples,
        graph_frame.display_time,
        visible_seconds=graph_view_visible_seconds(terminal_width, process_state),
    )
    current = min(max(0, process_state.graph_view_offset_seconds), max_offset)
    if key == ",":
        process_state.graph_view_offset_seconds = min(max_offset, current + GRAPH_PAN_SECONDS)
    else:
        process_state.graph_view_offset_seconds = max(0, current - GRAPH_PAN_SECONDS)
    return True


def process_view_key(process_state: ProcessViewState) -> tuple[str, bool, str, int | None, bool, bool]:
    return (
        process_state.sort_field,
        process_state.sort_desc,
        process_state.filter_query.strip(),
        process_state.gpu_filter_index,
        process_state.tree_mode,
        process_state.process_zoomed,
    )


def snapshot_gpu_indices(snapshot: Snapshot) -> tuple[int, ...]:
    gpu_indices = tuple(gpu.index for gpu in snapshot.gpus)
    if gpu_indices:
        return gpu_indices
    return tuple(
        sorted(
            {
                proc.gpu_index
                for proc in snapshot.processes
                if proc.gpu_index is not None
            }
        )
    )


def key_needs_current_processes(process_state: ProcessViewState, key: str) -> bool:
    if process_state.mode == MODE_FILTER:
        return False
    if process_state.mode == MODE_HELP:
        return False
    if process_state.mode == MODE_PROCESS_INFO:
        return False
    if process_state.mode == MODE_SORT_MENU:
        return False
    if process_state.mode == MODE_SEARCH:
        return key == KEY_ENTER
    if process_state.mode == MODE_KILL_CONFIRM:
        return key in ("y", "Y", KEY_ENTER)
    if key == "h":
        return process_state.tree_mode
    if key == "i":
        return process_state.mode == MODE_NORMAL
    return key in (
        "j",
        "k",
        "l",
        KEY_LEFT,
        KEY_RIGHT,
        KEY_UP,
        KEY_DOWN,
        KEY_PAGE_UP,
        KEY_PAGE_DOWN,
        "n",
        "N",
        " ",
        "x",
        "p",
    )


def open_selected_process_info(
    snapshot: Snapshot,
    process_state: ProcessViewState,
    processes: list[ProcessInfo],
) -> None:
    selected = process_state.selected_synced_process(processes)
    if selected is None:
        process_state.set_status_message("No process selected")
        return

    process_state.open_process_info(
        selected,
        read_process_detail(selected.pid),
        parent_process_for(selected, processes, snapshot.processes, snapshot.process_ancestors),
        child_count=sum(1 for proc in processes if proc.ppid == selected.pid),
    )


def parent_process_for(
    process: ProcessInfo,
    display_processes: list[ProcessInfo],
    gpu_processes: list[ProcessInfo],
    ancestor_processes: list[ProcessInfo],
) -> ProcessInfo | None:
    if process.ppid is None:
        return None
    for candidate in (*display_processes, *gpu_processes, *ancestor_processes):
        if candidate.pid == process.ppid:
            return candidate
    return None


def render_live_snapshot(
    snapshot,
    history: MetricsHistory,
    process_state: ProcessViewState,
    console: Console,
    display_processes: list[ProcessInfo] | None = None,
    interval: float = 1.0,
    display_time: datetime | None = None,
    graph_frame: GraphFrame | None = None,
):
    display_time = display_time or datetime.now()
    frame = graph_frame or capture_graph_frame(history, display_time)
    terminal_height, terminal_width = console_dimensions(console)
    frame, graph_time_offset_seconds = display_graph_frame(
        history,
        process_state,
        frame,
        terminal_width=terminal_width,
    )
    return render_snapshot(
        snapshot,
        history,
        process_state,
        terminal_height,
        terminal_width,
        display_processes=display_processes,
        display_time=display_time,
        show_subsecond_time=interval < 1.0,
        history_samples=frame.history_samples,
        graph_time=frame.display_time,
        graph_time_offset_seconds=graph_time_offset_seconds,
    )


def display_graph_frame(
    history: MetricsHistory,
    process_state: ProcessViewState,
    live_frame: GraphFrame,
    terminal_width: int | None = None,
) -> tuple[GraphFrame, int]:
    offset_seconds = clamped_graph_view_offset(
        process_state.graph_view_offset_seconds,
        history.samples,
        live_frame.display_time,
        visible_seconds=graph_view_visible_seconds(terminal_width, process_state),
    )
    process_state.graph_view_offset_seconds = offset_seconds
    if offset_seconds <= 0:
        return live_frame, 0
    end_time = live_frame.display_time - timedelta(seconds=offset_seconds)
    return (
        GraphFrame(
            display_time=end_time,
            history_samples=graph_samples_until(history.samples, end_time, GRAPH_HISTORY_BUCKETS),
        ),
        offset_seconds,
    )


def clamped_graph_view_offset(
    offset_seconds: int,
    samples: tuple[MetricSample, ...],
    live_end_time: datetime,
    visible_seconds: int | None = None,
) -> int:
    if offset_seconds <= 0:
        return 0
    max_offset = graph_view_max_offset_seconds(samples, live_end_time, visible_seconds)
    return min(offset_seconds, max_offset)


def graph_view_max_offset_seconds(
    samples: tuple[MetricSample, ...],
    live_end_time: datetime,
    visible_seconds: int | None = None,
) -> int:
    if not samples:
        return 0
    if visible_seconds is None:
        return GRAPH_HISTORY_SECONDS
    visible_span = max(0, visible_seconds - 1)
    return min(
        GRAPH_HISTORY_SECONDS,
        max(0, GRAPH_HISTORY_SECONDS - visible_span + GRAPH_HISTORY_LABEL_GUTTER_SECONDS),
    )


def graph_view_visible_seconds(
    terminal_width: int | None,
    process_state: ProcessViewState | None = None,
) -> int | None:
    if terminal_width is None:
        return None
    graph_width = graph_view_width(terminal_width, process_state)
    return min(GRAPH_HISTORY_BUCKETS, graph_width * GRAPH_COLUMNS_PER_CELL)


def graph_view_width(terminal_width: int, process_state: ProcessViewState | None = None) -> int:
    width = max(1, terminal_width)
    if process_state is not None and process_state.gpu_filter_index is not None and not process_state.gpu_graphs_visible:
        return max(12, width - 4)
    if process_state is not None and process_state.gpu_graphs_visible and width < GPU_GRAPH_WIDE_MIN_WIDTH:
        return max(12, width - 4)
    return max(12, (width - 2) // 2 - 2)


def graph_history_sample_limit(interval: float) -> int:
    interval = max(float(interval), 1e-9)
    return max(1, math.ceil(GRAPH_HISTORY_SECONDS / interval) + 1)


def capture_graph_frame(
    history: MetricsHistory,
    display_time: datetime | None = None,
) -> GraphFrame:
    frame_time = display_time or datetime.now()
    samples = history.samples
    display_second = latest_graph_sample_second(samples)
    if display_second is None:
        display_second = graph_frame_time(frame_time)
    return GraphFrame(
        display_time=display_second,
        history_samples=graph_samples_until(samples, display_second, GRAPH_HISTORY_BUCKETS),
    )


def advance_graph_frame(
    history: MetricsHistory,
    previous_frame: GraphFrame,
    now: datetime | None = None,
) -> GraphFrame:
    closed_second = graph_frame_time(now or datetime.now()) - timedelta(seconds=int(GRAPH_FRAME_SECONDS))
    next_second = previous_frame.display_time + timedelta(seconds=int(GRAPH_FRAME_SECONDS))
    if next_second > closed_second:
        return previous_frame
    raw_samples = history.samples
    if not graph_bucket_can_close(raw_samples, next_second):
        return previous_frame
    previous_values = graph_sample_values(previous_frame.history_samples[-1]) if previous_frame.history_samples else None
    sample = graph_sample_for_second(raw_samples, next_second, previous_values)
    samples = (*previous_frame.history_samples, sample)
    return GraphFrame(display_time=next_second, history_samples=samples[-GRAPH_HISTORY_BUCKETS:])


def graph_samples_until(
    samples: tuple[MetricSample, ...],
    end_time: datetime,
    max_samples: int,
) -> tuple[MetricSample, ...]:
    if not samples:
        return ()
    end_second = graph_frame_time(end_time)
    first_second = min(graph_frame_time(sample.timestamp) for sample in samples)
    start_second = max(first_second, end_second - timedelta(seconds=max(0, max_samples - 1)))
    bucket_samples: list[MetricSample] = []
    previous_values: GraphSampleValues | None = None
    bucket_time = start_second
    while bucket_time <= end_second:
        sample = graph_sample_for_second(samples, bucket_time, previous_values)
        previous_values = graph_sample_values(sample)
        bucket_samples.append(sample)
        bucket_time += timedelta(seconds=1)
    return tuple(bucket_samples)


def graph_sample_for_second(
    samples: tuple[MetricSample, ...],
    bucket_time: datetime,
    previous_values: GraphSampleValues | None = None,
) -> MetricSample:
    totals = [0.0] * len(GRAPH_METRIC_FIELDS)
    counts = [0] * len(GRAPH_METRIC_FIELDS)
    gpu_util_totals: dict[int, float] = {}
    gpu_util_counts: dict[int, int] = {}
    gpu_mem_totals: dict[int, float] = {}
    gpu_mem_counts: dict[int, int] = {}
    for sample in samples:
        if graph_frame_time(sample.timestamp) != bucket_time:
            continue
        for index, field in enumerate(GRAPH_METRIC_FIELDS):
            value = getattr(sample, field)
            if value is None:
                continue
            totals[index] += value
            counts[index] += 1
        for gpu_metric in sample.gpu_metrics:
            gpu_index = gpu_metric.index
            if gpu_metric.utilization_percent is not None:
                gpu_util_totals[gpu_index] = gpu_util_totals.get(gpu_index, 0.0) + gpu_metric.utilization_percent
                gpu_util_counts[gpu_index] = gpu_util_counts.get(gpu_index, 0) + 1
            if gpu_metric.memory_percent is not None:
                gpu_mem_totals[gpu_index] = gpu_mem_totals.get(gpu_index, 0.0) + gpu_metric.memory_percent
                gpu_mem_counts[gpu_index] = gpu_mem_counts.get(gpu_index, 0) + 1

    if previous_values is None:
        previous_values = latest_graph_values_at_or_before(samples, bucket_time)

    previous_metric_values = previous_values.metric_values if previous_values is not None else None
    values: list[float | None] = []
    for index, count in enumerate(counts):
        if count:
            values.append(totals[index] / count)
        else:
            values.append(previous_metric_values[index] if previous_metric_values is not None else None)

    previous_gpu_metrics = {
        gpu_metric.index: gpu_metric for gpu_metric in previous_values.gpu_metrics
    } if previous_values is not None else {}
    gpu_indices = sorted(
        set(previous_gpu_metrics)
        | set(gpu_util_counts)
        | set(gpu_mem_counts)
    )
    gpu_metrics: list[GpuMetricSample] = []
    for gpu_index in gpu_indices:
        previous_gpu_metric = previous_gpu_metrics.get(gpu_index)
        if gpu_index in gpu_util_counts:
            utilization_percent = gpu_util_totals[gpu_index] / gpu_util_counts[gpu_index]
        elif previous_gpu_metric is not None:
            utilization_percent = previous_gpu_metric.utilization_percent
        else:
            utilization_percent = None

        if gpu_index in gpu_mem_counts:
            memory_percent = gpu_mem_totals[gpu_index] / gpu_mem_counts[gpu_index]
        elif previous_gpu_metric is not None:
            memory_percent = previous_gpu_metric.memory_percent
        else:
            memory_percent = None

        gpu_metrics.append(
            GpuMetricSample(
                index=gpu_index,
                utilization_percent=utilization_percent,
                memory_percent=memory_percent,
            )
        )

    return MetricSample(
        timestamp=bucket_time,
        avg_cpu_percent=values[0],
        avg_mem_percent=values[1],
        avg_gpu_percent=values[2],
        avg_gpu_mem_percent=values[3],
        gpu_metrics=tuple(gpu_metrics),
    )


def latest_graph_values_at_or_before(
    samples: tuple[MetricSample, ...],
    bucket_time: datetime,
) -> GraphSampleValues | None:
    values: list[float | None] = [None] * len(GRAPH_METRIC_FIELDS)
    gpu_metrics: dict[int, GpuMetricSample] = {}
    found = False
    for sample in samples:
        if graph_frame_time(sample.timestamp) > bucket_time:
            continue
        for index, field in enumerate(GRAPH_METRIC_FIELDS):
            value = getattr(sample, field)
            if value is None:
                continue
            values[index] = value
            found = True
        for gpu_metric in sample.gpu_metrics:
            gpu_metrics[gpu_metric.index] = gpu_metric
            found = True
    if not found:
        return None
    return GraphSampleValues(
        metric_values=tuple(values),
        gpu_metrics=tuple(gpu_metrics[index] for index in sorted(gpu_metrics)),
    )


def graph_sample_values(sample: MetricSample) -> GraphSampleValues:
    return GraphSampleValues(
        metric_values=tuple(getattr(sample, field) for field in GRAPH_METRIC_FIELDS),
        gpu_metrics=sample.gpu_metrics,
    )


def latest_graph_sample_second(samples: tuple[MetricSample, ...]) -> datetime | None:
    if not samples:
        return None
    return max(graph_frame_time(sample.timestamp) for sample in samples)


def graph_bucket_can_close(samples: tuple[MetricSample, ...], bucket_time: datetime) -> bool:
    return any(graph_frame_time(sample.timestamp) >= bucket_time for sample in samples)


def live_render_interval(interval: float) -> float:
    return max(KEY_POLL_SECONDS, interval)


def graph_frame_interval(interval: float) -> float:
    return GRAPH_FRAME_SECONDS


def graph_frame_time(display_time: datetime) -> datetime:
    return display_time.replace(microsecond=0)


def console_dimensions(console: Console) -> tuple[int, int]:
    size = console.size
    return size.height, size.width


def collect_snapshot_retry(interval: float):
    while True:
        try:
            return collect_snapshot()
        except KeyboardInterrupt:
            raise
        except CommandTimeout:
            time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())

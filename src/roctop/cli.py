from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from rich.console import Console
from rich.live import Live

from . import __version__
from .collectors import CollectionError, CommandInterrupted, CommandTimeout, collect_snapshot, read_process_detail
from .history import MetricSample, MetricsHistory
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
from .render import render_snapshot


KEY_POLL_SECONDS = 0.05
COLLECTOR_STOP_JOIN_SECONDS = 0.05


@dataclass(slots=True)
class SnapshotUpdate:
    sequence: int
    snapshot: Snapshot


@dataclass(frozen=True, slots=True)
class GraphFrame:
    display_time: datetime
    history_samples: tuple[MetricSample, ...]


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
            except CollectionError as exc:
                with self._lock:
                    self._error = exc
                return

            if snapshot is not None:
                if self.history is not None:
                    self.history.add_snapshot(snapshot)
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
    history = MetricsHistory(max_samples=1081)
    process_state = ProcessViewState()
    history.prime_cpu()
    snapshot = collect_snapshot_retry(interval)
    history.add_snapshot(snapshot)
    collector = BackgroundSnapshotCollector(interval, history=history)
    graph_frame = capture_graph_frame(history)
    with (
        TerminalKeyboard() as keyboard,
        Live(
            render_live_snapshot(snapshot, history, process_state, console, interval=interval, graph_frame=graph_frame),
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
    graph_frame: GraphFrame | None = None,
) -> None:
    rendered_size = console_dimensions(console)
    latest_sequence = 0
    render_interval = live_render_interval(interval)
    next_render_at = time.monotonic() + render_interval
    graph_frame = graph_frame or capture_graph_frame(history)
    while True:
        collector.raise_if_failed()
        update = collector.latest_after(latest_sequence)
        if update is not None:
            snapshot = update.snapshot
            latest_sequence = update.sequence

        now = time.monotonic()
        current_size = console_dimensions(console)
        status_expired = process_state.expire_status_message(now)
        refresh_due = now >= next_render_at
        if refresh_due:
            graph_frame = capture_graph_frame(history)
        if current_size != rendered_size or status_expired or refresh_due:
            live.update(
                render_live_snapshot(
                    snapshot,
                    history,
                    process_state,
                    console,
                    interval=interval,
                    graph_frame=graph_frame,
                ),
                refresh=True,
            )
            rendered_size = current_size
            if refresh_due:
                next_render_at = now + render_interval

        timeout = min(KEY_POLL_SECONDS, max(0.0, next_render_at - time.monotonic()))
        keys = keyboard.read_keys(timeout=timeout)
        if not keys:
            continue

        quit_requested, processes = handle_key_batch(snapshot, process_state, keys)
        rendered_size = console_dimensions(console)
        live.update(
            render_live_snapshot(
                snapshot,
                history,
                process_state,
                console,
                display_processes=processes,
                interval=interval,
                graph_frame=graph_frame,
            ),
            refresh=True,
        )
        if quit_requested:
            return


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
            live.update(
                render_snapshot(snapshot, history, process_state, *current_size),
                refresh=True,
            )
            rendered_size = current_size
        keys = keyboard.read_keys(timeout=min(KEY_POLL_SECONDS, remaining))
        if not keys:
            continue
        quit_requested, processes = handle_key_batch(snapshot, process_state, keys)
        rendered_size = console_dimensions(console)
        live.update(
            render_snapshot(snapshot, history, process_state, *rendered_size, display_processes=processes),
            refresh=True,
        )
        if quit_requested:
            return True


def handle_key_batch(
    snapshot: Snapshot,
    process_state: ProcessViewState,
    keys: list[str],
) -> tuple[bool, list[ProcessInfo]]:
    quit_requested = False
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
            view_before = process_view_key(process_state)
            result = process_state.handle_key(key, processes, processes_synced=True)
            quit_requested = quit_requested or result.quit
            if process_view_key(process_state) != view_before:
                processes_dirty = True
        if processes_dirty:
            processes = process_state.display_processes(snapshot.processes, snapshot.process_ancestors)
            process_state.sync(processes, adjust_scroll=False)
    return quit_requested, processes


def process_view_key(process_state: ProcessViewState) -> tuple[str, bool, str, bool]:
    return (
        process_state.sort_field,
        process_state.sort_desc,
        process_state.filter_query.strip(),
        process_state.tree_mode,
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
    graph_frame: GraphFrame | None = None,
):
    frame = graph_frame or capture_graph_frame(history)
    return render_snapshot(
        snapshot,
        history,
        process_state,
        *console_dimensions(console),
        display_processes=display_processes,
        display_time=frame.display_time,
        show_subsecond_time=interval < 1.0,
        history_samples=frame.history_samples,
    )


def capture_graph_frame(history: MetricsHistory) -> GraphFrame:
    return GraphFrame(display_time=datetime.now(), history_samples=history.samples)


def live_render_interval(interval: float) -> float:
    return max(KEY_POLL_SECONDS, interval)


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

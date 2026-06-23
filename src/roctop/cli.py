from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from typing import Callable

from rich.console import Console
from rich.live import Live

from . import __version__
from .collectors import CollectionError, CommandInterrupted, CommandTimeout, collect_snapshot
from .history import MetricsHistory
from .interaction import ProcessViewState, TerminalKeyboard
from .models import ProcessInfo, Snapshot
from .profiling import profile_span
from .render import render_snapshot


KEY_POLL_SECONDS = 0.05
COLLECTOR_STOP_JOIN_SECONDS = 0.05


@dataclass(slots=True)
class SnapshotUpdate:
    sequence: int
    snapshot: Snapshot


class BackgroundSnapshotCollector:
    def __init__(self, interval: float, collect_func: Callable[[], Snapshot] | None = None) -> None:
        self.interval = interval
        self.collect_func = collect_func or collect_snapshot
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
        while not self._stop.wait(self.interval):
            try:
                snapshot = self.collect_func()
            except CommandTimeout:
                continue
            except CollectionError as exc:
                with self._lock:
                    self._error = exc
                return

            with self._lock:
                self._sequence += 1
                self._latest = SnapshotUpdate(sequence=self._sequence, snapshot=snapshot)


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
            console.print(render_snapshot(snapshot))
            return 0

        return run_live(console, args.interval)
    except (KeyboardInterrupt, CommandInterrupted):
        return 0
    except CollectionError as exc:
        error_console.print(f"[red]{exc}[/red]")
        return 1


def run_live(console: Console, interval: float) -> int:
    history = MetricsHistory(max_samples=120)
    process_state = ProcessViewState()
    snapshot = collect_snapshot_retry(interval)
    history.add_snapshot(snapshot)
    collector = BackgroundSnapshotCollector(interval)
    with (
        TerminalKeyboard() as keyboard,
        Live(
            render_live_snapshot(snapshot, history, process_state, console),
            console=console,
            screen=True,
            auto_refresh=False,
        ) as live,
    ):
        collector.start()
        try:
            poll_live_until_quit(live, keyboard, snapshot, history, process_state, console, collector)
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
) -> None:
    rendered_size = console_dimensions(console)
    latest_sequence = 0
    while True:
        collector.raise_if_failed()
        update = collector.latest_after(latest_sequence)
        if update is not None:
            snapshot = update.snapshot
            latest_sequence = update.sequence
            history.add_snapshot(snapshot)
            rendered_size = console_dimensions(console)
            live.update(
                render_snapshot(snapshot, history, process_state, *rendered_size),
                refresh=True,
            )

        now = time.monotonic()
        current_size = console_dimensions(console)
        status_expired = process_state.expire_status_message(now)
        if current_size != rendered_size or status_expired:
            live.update(
                render_snapshot(snapshot, history, process_state, *current_size),
                refresh=True,
            )
            rendered_size = current_size

        keys = keyboard.read_keys(timeout=KEY_POLL_SECONDS)
        if not keys:
            continue

        quit_requested, processes = handle_key_batch(snapshot, process_state, keys)
        rendered_size = console_dimensions(console)
        live.update(
            render_snapshot(snapshot, history, process_state, *rendered_size, display_processes=processes),
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
    processes = process_state.sorted_processes(snapshot.processes)
    process_state.sync(processes)
    with profile_span("key-handling"):
        for key in keys:
            sort_before = (process_state.sort_field, process_state.sort_desc)
            result = process_state.handle_key(key, processes, processes_synced=True)
            quit_requested = quit_requested or result.quit
            if (process_state.sort_field, process_state.sort_desc) != sort_before:
                processes = process_state.sorted_processes(snapshot.processes)
                process_state.sync(processes)
    return quit_requested, processes


def render_live_snapshot(snapshot, history: MetricsHistory, process_state: ProcessViewState, console: Console):
    return render_snapshot(snapshot, history, process_state, *console_dimensions(console))


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

from __future__ import annotations

import argparse
import json
import time

from rich.console import Console
from rich.live import Live

from . import __version__
from .collectors import CollectionError, CommandInterrupted, CommandTimeout, collect_snapshot
from .history import MetricsHistory
from .interaction import ProcessViewState, TerminalKeyboard
from .render import render_snapshot


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
    with (
        TerminalKeyboard() as keyboard,
        Live(
            render_live_snapshot(snapshot, history, process_state, console),
            console=console,
            screen=True,
            auto_refresh=False,
        ) as live,
    ):
        while True:
            if poll_input_until_refresh(live, keyboard, snapshot, history, process_state, console, interval):
                return 0
            try:
                snapshot = collect_snapshot()
            except CommandTimeout:
                continue
            history.add_snapshot(snapshot)
            live.update(
                render_live_snapshot(snapshot, history, process_state, console),
                refresh=True,
            )


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
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        current_size = console_dimensions(console)
        if current_size != rendered_size:
            live.update(
                render_snapshot(snapshot, history, process_state, *current_size),
                refresh=True,
            )
            rendered_size = current_size
        keys = keyboard.read_keys(timeout=min(0.05, remaining))
        if not keys:
            continue
        quit_requested = False
        for key in keys:
            processes = process_state.sorted_processes(snapshot.processes)
            process_state.sync(processes)
            result = process_state.handle_key(key, processes)
            quit_requested = quit_requested or result.quit
        rendered_size = console_dimensions(console)
        live.update(
            render_snapshot(snapshot, history, process_state, *rendered_size),
            refresh=True,
        )
        if quit_requested:
            return True


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

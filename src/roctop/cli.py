from __future__ import annotations

import argparse
import json
import time

from rich.console import Console
from rich.live import Live

from . import __version__
from .collectors import CollectionError, CommandTimeout, collect_snapshot
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
    console = Console()
    error_console = Console(stderr=True)

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
    except CollectionError as exc:
        error_console.print(f"[red]{exc}[/red]")
        return 1


def run_live(console: Console, interval: float) -> int:
    snapshot = collect_snapshot_retry(interval)
    with Live(render_snapshot(snapshot), console=console, screen=True, auto_refresh=False) as live:
        while True:
            try:
                snapshot = collect_snapshot()
            except CommandTimeout:
                time.sleep(interval)
                continue
            live.update(render_snapshot(snapshot), refresh=True)
            time.sleep(interval)


def collect_snapshot_retry(interval: float):
    while True:
        try:
            return collect_snapshot()
        except CommandTimeout:
            time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())

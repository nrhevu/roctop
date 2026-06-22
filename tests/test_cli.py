from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime

from roctop import cli
from roctop.collectors import CommandInterrupted, CommandTimeout
from roctop.models import Snapshot


@dataclass(frozen=True)
class FakeConsoleSize:
    height: int
    width: int


class CliTests(unittest.TestCase):
    def test_main_swallows_keyboard_interrupt(self) -> None:
        original_run_live = cli.run_live

        def fake_run_live(*args, **kwargs) -> int:
            raise KeyboardInterrupt

        try:
            cli.run_live = fake_run_live
            self.assertEqual(cli.main([]), 0)
        finally:
            cli.run_live = original_run_live

    def test_main_swallows_command_interrupted(self) -> None:
        original_run_live = cli.run_live

        def fake_run_live(*args, **kwargs) -> int:
            raise CommandInterrupted("rocm-smi interrupted")

        try:
            cli.run_live = fake_run_live
            self.assertEqual(cli.main([]), 0)
        finally:
            cli.run_live = original_run_live

    def test_collect_snapshot_retry_swallows_timeout(self) -> None:
        calls = 0
        expected = Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0))
        original_collect_snapshot = cli.collect_snapshot

        def fake_collect_snapshot() -> Snapshot:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise CommandTimeout("Command timed out: rocm-smi")
            return expected

        try:
            cli.collect_snapshot = fake_collect_snapshot
            self.assertIs(cli.collect_snapshot_retry(0), expected)
            self.assertEqual(calls, 2)
        finally:
            cli.collect_snapshot = original_collect_snapshot

    def test_poll_input_redraws_when_terminal_size_changes(self) -> None:
        class FakeConsole:
            def __init__(self) -> None:
                self._size = FakeConsoleSize(height=24, width=100)

            @property
            def size(self) -> FakeConsoleSize:
                return self._size

            def resize(self, height: int, width: int) -> None:
                self._size = FakeConsoleSize(height=height, width=width)

        class FakeKeyboard:
            def __init__(self, console: FakeConsole) -> None:
                self.console = console
                self.calls = 0

            def read_keys(self, timeout: float):
                self.calls += 1
                if self.calls == 1:
                    self.console.resize(height=18, width=80)
                    return []
                return ["q"]

        class FakeLive:
            def __init__(self) -> None:
                self.updates = []

            def update(self, renderable, refresh: bool = False) -> None:
                self.updates.append((renderable, refresh))

        console = FakeConsole()
        keyboard = FakeKeyboard(console)
        live = FakeLive()
        snapshot = Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0))

        quit_requested = cli.poll_input_until_refresh(
            live,
            keyboard,
            snapshot,
            cli.MetricsHistory(max_samples=120),
            cli.ProcessViewState(),
            console,
            interval=1.0,
        )

        self.assertTrue(quit_requested)
        self.assertGreaterEqual(len(live.updates), 2)
        self.assertTrue(all(refresh for _renderable, refresh in live.updates))


if __name__ == "__main__":
    unittest.main()

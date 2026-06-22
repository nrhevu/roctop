from __future__ import annotations

import unittest
from datetime import datetime

from roctop import cli
from roctop.collectors import CommandInterrupted, CommandTimeout
from roctop.models import Snapshot


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


if __name__ == "__main__":
    unittest.main()

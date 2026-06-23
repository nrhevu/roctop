from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass
from datetime import datetime
from io import StringIO

from rich.console import Console

from roctop import cli
from roctop.collectors import CommandInterrupted, CommandTimeout
from roctop.interaction import KEY_DOWN, KEY_ENTER, KEY_UP
from roctop.models import ProcessInfo, Snapshot


@dataclass(frozen=True)
class FakeConsoleSize:
    height: int
    width: int


def many_long_processes(count: int) -> list[ProcessInfo]:
    command = (
        "demo_worker --model-path /demo/models/example-checkpoint "
        "--tensor-parallel-size 8 --batch-size 64 --sequence-length 8192 --final-flag"
    )
    return [
        ProcessInfo(gpu_index=index % 8, pid=1000 + index, user="demo", args=f"{command} --rank {index}")
        for index in range(count)
    ]


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

    def test_poll_input_redraws_when_status_message_expires(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=24, width=100)

        class FakeKeyboard:
            def __init__(self) -> None:
                self.calls = 0

            def read_keys(self, timeout: float):
                self.calls += 1
                if self.calls >= 3:
                    return ["q"]
                return []

        class FakeLive:
            def __init__(self) -> None:
                self.updates = []

            def update(self, renderable, refresh: bool = False) -> None:
                self.updates.append((renderable, refresh))

        class FakeClock:
            def __init__(self) -> None:
                self.times = iter([0.0, 1.0, 3.0, 3.1])

            def monotonic(self) -> float:
                return next(self.times)

        console = FakeConsole()
        keyboard = FakeKeyboard()
        live = FakeLive()
        snapshot = Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0))
        process_state = cli.ProcessViewState()
        process_state.set_status_message("Search: demo", now=0.0)
        original_monotonic = cli.time.monotonic

        try:
            cli.time.monotonic = FakeClock().monotonic
            quit_requested = cli.poll_input_until_refresh(
                live,
                keyboard,
                snapshot,
                cli.MetricsHistory(max_samples=120),
                process_state,
                console,
                interval=5.0,
            )
        finally:
            cli.time.monotonic = original_monotonic

        self.assertTrue(quit_requested)
        self.assertEqual(process_state.status_message, "")
        self.assertIsNone(process_state.status_message_expires_at)
        self.assertGreaterEqual(len(live.updates), 2)
        self.assertTrue(all(refresh for _renderable, refresh in live.updates))

    def test_poll_input_batches_movement_keys_into_one_render(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=24, width=100)

        class FakeKeyboard:
            def __init__(self) -> None:
                self.calls = 0

            def read_keys(self, timeout: float):
                self.calls += 1
                if self.calls == 1:
                    return ["j", "j", KEY_DOWN]
                return ["q"]

        class FakeLive:
            def __init__(self) -> None:
                self.updates = []

            def update(self, renderable, refresh: bool = False) -> None:
                self.updates.append((renderable, refresh))

        class CountingProcessViewState(cli.ProcessViewState):
            def __init__(self) -> None:
                super().__init__()
                self.sort_calls = 0

            def sorted_processes(self, processes):
                self.sort_calls += 1
                return super().sorted_processes(processes)

        state = CountingProcessViewState()
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[ProcessInfo(gpu_index=0, pid=100 + index, args=f"cmd-{index}") for index in range(6)],
        )
        live = FakeLive()

        quit_requested = cli.poll_input_until_refresh(
            live,
            FakeKeyboard(),
            snapshot,
            cli.MetricsHistory(max_samples=120),
            state,
            FakeConsole(),
            interval=1.0,
        )

        self.assertTrue(quit_requested)
        self.assertEqual(state.selected_pid, 103)
        self.assertEqual(len(live.updates), 2)
        self.assertEqual(state.sort_calls, 2)
        self.assertTrue(all(refresh for _renderable, refresh in live.updates))

    def test_poll_input_resorts_after_sort_change_within_key_batch(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=24, width=100)

        class FakeKeyboard:
            def read_keys(self, timeout: float):
                return ["s", KEY_DOWN, KEY_DOWN, KEY_DOWN, KEY_ENTER, KEY_UP, "q"]

        class FakeLive:
            def __init__(self) -> None:
                self.updates = []

            def update(self, renderable, refresh: bool = False) -> None:
                self.updates.append((renderable, refresh))

        state = cli.ProcessViewState()
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=1, cpu_percent=1.0, args="cmd-low"),
                ProcessInfo(gpu_index=0, pid=2, cpu_percent=90.0, args="cmd-high"),
                ProcessInfo(gpu_index=0, pid=3, cpu_percent=30.0, args="cmd-mid"),
            ],
        )

        quit_requested = cli.poll_input_until_refresh(
            FakeLive(),
            FakeKeyboard(),
            snapshot,
            cli.MetricsHistory(max_samples=120),
            state,
            FakeConsole(),
            interval=1.0,
        )

        self.assertTrue(quit_requested)
        self.assertEqual(state.sort_field, "cpu")
        self.assertTrue(state.sort_desc)
        self.assertEqual(state.selected_pid, 3)

    def test_handle_key_batch_filters_before_sort_and_selection(self) -> None:
        state = cli.ProcessViewState(selected_pid=2, sort_field="cpu", sort_desc=True)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=1, cpu_percent=1.0, args="train-low"),
                ProcessInfo(gpu_index=0, pid=2, cpu_percent=90.0, args="serve-high"),
                ProcessInfo(gpu_index=0, pid=3, cpu_percent=30.0, args="train-mid"),
            ],
        )

        quit_requested, processes = cli.handle_key_batch(snapshot, state, ["f", *"train"])

        self.assertFalse(quit_requested)
        self.assertEqual(state.filter_query, "train")
        self.assertEqual([row.pid for row in processes], [3, 1])
        self.assertEqual(state.selected_pid, 3)

    def test_handle_key_batch_escape_clears_active_filter(self) -> None:
        state = cli.ProcessViewState(filter_query="train", filter_input="train")
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=1, args="train-low"),
                ProcessInfo(gpu_index=0, pid=2, args="serve-high"),
            ],
        )

        quit_requested, processes = cli.handle_key_batch(snapshot, state, ["esc"])

        self.assertFalse(quit_requested)
        self.assertEqual(state.filter_query, "")
        self.assertEqual(state.filter_input, "")
        self.assertEqual([row.pid for row in processes], [1, 2])

    def test_run_live_responds_under_200ms_while_background_collection_is_blocked(self) -> None:
        background_collect_started = threading.Event()
        release_collect = threading.Event()
        state = {
            "collect_calls": 0,
            "key_ready_time": None,
            "update_time": None,
            "sent_j": False,
            "sent_q": False,
        }
        initial_snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=100, args="cmd-0"),
                ProcessInfo(gpu_index=0, pid=101, args="cmd-1"),
            ],
        )
        updated_snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 1),
            processes=[
                ProcessInfo(gpu_index=0, pid=100, args="cmd-0"),
                ProcessInfo(gpu_index=0, pid=101, args="cmd-1"),
            ],
        )

        def fake_collect_snapshot() -> Snapshot:
            state["collect_calls"] += 1
            if state["collect_calls"] == 1:
                return initial_snapshot
            if state["key_ready_time"] is None:
                state["key_ready_time"] = time.perf_counter()
            background_collect_started.set()
            release_collect.wait(timeout=0.5)
            return updated_snapshot

        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=24, width=100)

        class FakeKeyboard:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def read_keys(self, timeout: float):
                if background_collect_started.is_set() and not state["sent_j"]:
                    state["sent_j"] = True
                    return ["j"]
                if state["sent_j"] and state["update_time"] is not None and not state["sent_q"]:
                    state["sent_q"] = True
                    release_collect.set()
                    return ["q"]
                time.sleep(min(timeout, 0.005))
                return []

        class FakeLive:
            def __init__(self, renderable, console, screen: bool, auto_refresh: bool) -> None:
                self.updates = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def update(self, renderable, refresh: bool = False) -> None:
                if state["sent_j"] and state["update_time"] is None:
                    state["update_time"] = time.perf_counter()
                self.updates.append((renderable, refresh))

        original_collect_snapshot = cli.collect_snapshot
        original_keyboard = cli.TerminalKeyboard
        original_live = cli.Live

        try:
            cli.collect_snapshot = fake_collect_snapshot
            cli.TerminalKeyboard = FakeKeyboard
            cli.Live = FakeLive
            result = cli.run_live(FakeConsole(), interval=0.01)
        finally:
            release_collect.set()
            cli.collect_snapshot = original_collect_snapshot
            cli.TerminalKeyboard = original_keyboard
            cli.Live = original_live

        self.assertEqual(result, 0)
        self.assertTrue(background_collect_started.is_set())
        self.assertTrue(state["sent_j"])
        self.assertIsNotNone(state["key_ready_time"])
        self.assertIsNotNone(state["update_time"])
        self.assertLess(state["update_time"] - state["key_ready_time"], 0.2)

    def test_large_process_keypress_render_completes_under_200ms(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=40, width=160)

        class FakeKeyboard:
            def __init__(self) -> None:
                self.calls = 0

            def read_keys(self, timeout: float):
                self.calls += 1
                if self.calls == 1:
                    key_times["j"] = time.perf_counter()
                    return ["j"]
                return ["q"]

        class RenderingLive:
            def __init__(self) -> None:
                self.updates = []

            def update(self, renderable, refresh: bool = False) -> None:
                render_console = Console(width=160, record=True, file=StringIO())
                render_console.print(renderable)
                if "j" in key_times and "j_update" not in key_times:
                    key_times["j_update"] = time.perf_counter()
                self.updates.append((renderable, refresh))

        key_times: dict[str, float] = {}
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=many_long_processes(1000),
        )
        state = cli.ProcessViewState(viewport_rows=20)

        quit_requested = cli.poll_input_until_refresh(
            RenderingLive(),
            FakeKeyboard(),
            snapshot,
            cli.MetricsHistory(max_samples=120),
            state,
            FakeConsole(),
            interval=1.0,
        )

        self.assertTrue(quit_requested)
        self.assertEqual(state.selected_pid, 1001)
        self.assertLess(key_times["j_update"] - key_times["j"], 0.2)

    def test_large_process_filter_keypress_render_completes_under_200ms(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=40, width=160)

        class FakeKeyboard:
            def __init__(self) -> None:
                self.calls = 0

            def read_keys(self, timeout: float):
                self.calls += 1
                if self.calls == 1:
                    key_times["filter"] = time.perf_counter()
                    return ["f", "9", "9", "9"]
                return [KEY_ENTER, "q"]

        class RenderingLive:
            def __init__(self) -> None:
                self.updates = []

            def update(self, renderable, refresh: bool = False) -> None:
                render_console = Console(width=160, record=True, file=StringIO())
                render_console.print(renderable)
                if "filter" in key_times and "filter_update" not in key_times:
                    key_times["filter_update"] = time.perf_counter()
                self.updates.append((renderable, refresh))

        key_times: dict[str, float] = {}
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=many_long_processes(1000),
        )
        state = cli.ProcessViewState(viewport_rows=20)

        quit_requested = cli.poll_input_until_refresh(
            RenderingLive(),
            FakeKeyboard(),
            snapshot,
            cli.MetricsHistory(max_samples=120),
            state,
            FakeConsole(),
            interval=1.0,
        )

        self.assertTrue(quit_requested)
        self.assertEqual(state.filter_query, "999")
        self.assertLess(key_times["filter_update"] - key_times["filter"], 0.2)


if __name__ == "__main__":
    unittest.main()

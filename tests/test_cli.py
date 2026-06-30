from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import patch

from rich.console import Console

from roctop import cli
from roctop.collectors import CommandInterrupted, CommandTimeout
from roctop.interaction import (
    KEY_CTRL_C,
    KEY_DOWN,
    KEY_ENTER,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
    MODE_GPU_DEBUG,
    MODE_HELP,
    MODE_NORMAL,
    MODE_PROCESS_INFO,
)
from roctop.models import GpuInfo, ProcessDetailInfo, ProcessInfo, Snapshot


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

    def test_main_version_prints_package_version(self) -> None:
        output = StringIO()

        with (
            patch("sys.stdout", output),
            self.assertRaises(SystemExit) as context,
        ):
            cli.main(["--version"])

        self.assertEqual(context.exception.code, 0)
        self.assertIn(f"roctop {cli.__version__}", output.getvalue())

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

    def test_live_loop_redraws_on_requested_interval_without_snapshot_update(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=24, width=100)

        class FakeCollector:
            def raise_if_failed(self) -> None:
                return None

            def latest_after(self, sequence: int):
                return None

        class FakeClock:
            def __init__(self) -> None:
                self.current = 0.0

            def monotonic(self) -> float:
                return self.current

            def advance(self, seconds: float) -> None:
                self.current += seconds

        class FakeLive:
            def __init__(self, clock: FakeClock) -> None:
                self.clock = clock
                self.update_times: list[float] = []

            def update(self, renderable, refresh: bool = False) -> None:
                self.update_times.append(self.clock.current)

        class FakeKeyboard:
            def __init__(self, clock: FakeClock, live: FakeLive) -> None:
                self.clock = clock
                self.live = live

            def read_keys(self, timeout: float):
                self.clock.advance(timeout)
                if len(self.live.update_times) >= 3:
                    return ["q"]
                return []

        clock = FakeClock()
        live = FakeLive(clock)
        original_monotonic = cli.time.monotonic

        try:
            cli.time.monotonic = clock.monotonic
            cli.poll_live_until_quit(
                live,
                FakeKeyboard(clock, live),
                Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0)),
                cli.MetricsHistory(max_samples=120),
                cli.ProcessViewState(),
                FakeConsole(),
                FakeCollector(),
                interval=0.1,
            )
        finally:
            cli.time.monotonic = original_monotonic

        self.assertGreaterEqual(len(live.update_times), 4)
        self.assertEqual([round(update_time, 1) for update_time in live.update_times[:3]], [0.1, 0.2, 0.3])

    def test_live_loop_renders_on_requested_interval_with_fast_collector(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=24, width=100)

        class FakeCollector:
            def __init__(self) -> None:
                self.update_count = 0

            def raise_if_failed(self) -> None:
                return None

            def latest_after(self, sequence: int):
                self.update_count += 1
                next_sequence = sequence + 1
                return cli.SnapshotUpdate(
                    sequence=next_sequence,
                    snapshot=Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0)),
                )

        class FakeClock:
            def __init__(self) -> None:
                self.current = 0.0

            def monotonic(self) -> float:
                return self.current

            def advance(self, seconds: float) -> None:
                self.current += seconds

        class FakeLive:
            def __init__(self, clock: FakeClock) -> None:
                self.clock = clock
                self.update_times: list[float] = []

            def update(self, renderable, refresh: bool = False) -> None:
                self.update_times.append(self.clock.current)

        class FakeKeyboard:
            def __init__(self, clock: FakeClock, live: FakeLive) -> None:
                self.clock = clock
                self.live = live

            def read_keys(self, timeout: float):
                self.clock.advance(timeout)
                if len(self.live.update_times) >= 3:
                    return ["q"]
                return []

        clock = FakeClock()
        collector = FakeCollector()
        live = FakeLive(clock)
        original_monotonic = cli.time.monotonic

        try:
            cli.time.monotonic = clock.monotonic
            cli.poll_live_until_quit(
                live,
                FakeKeyboard(clock, live),
                Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0)),
                cli.MetricsHistory(max_samples=120),
                cli.ProcessViewState(),
                FakeConsole(),
                collector,
                interval=0.1,
            )
        finally:
            cli.time.monotonic = original_monotonic

        self.assertGreater(collector.update_count, len(live.update_times))
        self.assertEqual([round(update_time, 1) for update_time in live.update_times[:3]], [0.1, 0.2, 0.3])

    def test_live_render_interval_uses_key_poll_floor(self) -> None:
        self.assertEqual(cli.live_render_interval(0.01), cli.KEY_POLL_SECONDS)
        self.assertEqual(cli.live_render_interval(0.25), 0.25)

    def test_graph_frame_interval_is_always_one_second(self) -> None:
        self.assertEqual(cli.graph_frame_interval(0.01), 1.0)
        self.assertEqual(cli.graph_frame_interval(0.25), 1.0)
        self.assertEqual(cli.graph_frame_interval(1.0), 1.0)
        self.assertEqual(cli.graph_frame_interval(2.0), 1.0)
        self.assertEqual(cli.graph_frame_interval(3.0), 1.0)

    def test_graph_history_sample_limit_tracks_window_seconds(self) -> None:
        self.assertEqual(cli.graph_history_sample_limit(1.0), cli.GRAPH_HISTORY_BUCKETS)
        self.assertEqual(cli.graph_history_sample_limit(2.0), cli.GRAPH_HISTORY_SECONDS // 2 + 1)
        self.assertEqual(cli.graph_history_sample_limit(0.5), cli.GRAPH_HISTORY_SECONDS * 2 + 1)

    def test_graph_frame_time_uses_whole_second(self) -> None:
        self.assertEqual(
            cli.graph_frame_time(datetime(2026, 6, 22, 12, 0, 1, 500000)),
            datetime(2026, 6, 22, 12, 0, 1),
        )

    def test_advance_graph_frame_moves_one_second_at_a_time(self) -> None:
        history = cli.MetricsHistory(max_samples=120)
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=10.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            )
        )
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 1),
                avg_cpu_percent=20.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            )
        )
        previous = cli.GraphFrame(
            display_time=datetime(2026, 6, 22, 12, 0, 0),
            history_samples=(),
        )

        advanced = cli.advance_graph_frame(
            history,
            previous,
            now=datetime(2026, 6, 22, 12, 0, 5, 900000),
        )
        not_backwards = cli.advance_graph_frame(
            history,
            advanced,
            now=datetime(2026, 6, 22, 12, 0, 0, 900000),
        )

        self.assertEqual(advanced.display_time, datetime(2026, 6, 22, 12, 0, 1))
        self.assertEqual(len(advanced.history_samples), 1)
        self.assertEqual(not_backwards.display_time, advanced.display_time)

    def test_advance_graph_frame_waits_until_bucket_has_data(self) -> None:
        history = cli.MetricsHistory(max_samples=120)
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=10.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            )
        )
        previous = cli.capture_graph_frame(history, datetime(2026, 6, 22, 12, 0, 0, 500000))

        waiting = cli.advance_graph_frame(
            history,
            previous,
            now=datetime(2026, 6, 22, 12, 0, 2, 100000),
        )
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 1, 100000),
                avg_cpu_percent=30.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            )
        )
        advanced = cli.advance_graph_frame(
            history,
            waiting,
            now=datetime(2026, 6, 22, 12, 0, 2, 100000),
        )

        self.assertEqual(waiting.display_time, previous.display_time)
        self.assertEqual(advanced.display_time, datetime(2026, 6, 22, 12, 0, 1))

    def test_advance_graph_frame_does_not_recompute_rendered_buckets(self) -> None:
        history = cli.MetricsHistory(max_samples=120)
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=10.0,
                avg_mem_percent=20.0,
                avg_gpu_percent=30.0,
                avg_gpu_mem_percent=40.0,
            )
        )
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 1, 100000),
                avg_cpu_percent=90.0,
                avg_mem_percent=80.0,
                avg_gpu_percent=70.0,
                avg_gpu_mem_percent=60.0,
            )
        )
        previous = cli.capture_graph_frame(history, datetime(2026, 6, 22, 12, 0, 1, 200000))
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 1, 800000),
                avg_cpu_percent=0.0,
                avg_mem_percent=0.0,
                avg_gpu_percent=0.0,
                avg_gpu_mem_percent=0.0,
            )
        )
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 2, 100000),
                avg_cpu_percent=50.0,
                avg_mem_percent=40.0,
                avg_gpu_percent=30.0,
                avg_gpu_mem_percent=20.0,
            )
        )

        advanced = cli.advance_graph_frame(
            history,
            previous,
            now=datetime(2026, 6, 22, 12, 0, 3, 100000),
        )
        buckets = {sample.timestamp: sample for sample in advanced.history_samples}

        self.assertEqual(buckets[datetime(2026, 6, 22, 12, 0, 1)].avg_cpu_percent, 90.0)
        self.assertEqual(buckets[datetime(2026, 6, 22, 12, 0, 2)].avg_cpu_percent, 50.0)

    def test_graph_frame_buckets_average_and_carry_per_gpu_metrics(self) -> None:
        history = cli.MetricsHistory(max_samples=120)
        start = datetime(2026, 6, 22, 12, 0, 0)
        history.append_sample(
            cli.MetricSample(
                timestamp=start,
                avg_cpu_percent=None,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
                gpu_metrics=(cli.GpuMetricSample(index=0, utilization_percent=10.0, memory_percent=20.0),),
            )
        )
        history.append_sample(
            cli.MetricSample(
                timestamp=start + timedelta(seconds=1, milliseconds=100),
                avg_cpu_percent=None,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
                gpu_metrics=(
                    cli.GpuMetricSample(index=0, utilization_percent=20.0, memory_percent=30.0),
                    cli.GpuMetricSample(index=1, utilization_percent=50.0, memory_percent=60.0),
                ),
            )
        )
        history.append_sample(
            cli.MetricSample(
                timestamp=start + timedelta(seconds=1, milliseconds=800),
                avg_cpu_percent=None,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
                gpu_metrics=(cli.GpuMetricSample(index=0, utilization_percent=40.0, memory_percent=70.0),),
            )
        )
        history.append_sample(
            cli.MetricSample(
                timestamp=start + timedelta(seconds=2),
                avg_cpu_percent=None,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            )
        )

        frame = cli.capture_graph_frame(history, start + timedelta(seconds=2))
        buckets = {sample.timestamp: sample for sample in frame.history_samples}

        second_one = {metric.index: metric for metric in buckets[start + timedelta(seconds=1)].gpu_metrics}
        second_two = {metric.index: metric for metric in buckets[start + timedelta(seconds=2)].gpu_metrics}
        self.assertAlmostEqual(second_one[0].utilization_percent, 30.0)
        self.assertAlmostEqual(second_one[0].memory_percent, 50.0)
        self.assertEqual(second_one[1], cli.GpuMetricSample(index=1, utilization_percent=50.0, memory_percent=60.0))
        self.assertEqual(second_two[0], second_one[0])
        self.assertEqual(second_two[1], second_one[1])

    def test_capture_graph_frame_uses_fixed_graph_second_window(self) -> None:
        history = cli.MetricsHistory(max_samples=2000)
        start = datetime(2026, 6, 22, 12, 0, 0)
        extra_seconds = 120
        end_second = cli.GRAPH_HISTORY_SECONDS + extra_seconds
        for second in range(end_second + 1):
            history.append_sample(
                cli.MetricSample(
                    timestamp=start + timedelta(seconds=second),
                    avg_cpu_percent=float(second),
                    avg_mem_percent=None,
                    avg_gpu_percent=None,
                    avg_gpu_mem_percent=None,
                )
            )

        frame = cli.capture_graph_frame(history, start + timedelta(seconds=end_second))

        self.assertEqual(len(frame.history_samples), cli.GRAPH_HISTORY_BUCKETS)
        self.assertEqual(frame.history_samples[0].timestamp, start + timedelta(seconds=extra_seconds))
        self.assertEqual(frame.history_samples[-1].timestamp, start + timedelta(seconds=end_second))

    def test_graph_pan_clamps_to_graph_second_left_bound_for_viewport(self) -> None:
        start = datetime(2026, 6, 22, 12, 0, 0)
        end_second = cli.GRAPH_HISTORY_SECONDS + 120
        samples = tuple(
            cli.MetricSample(
                timestamp=start + timedelta(seconds=second),
                avg_cpu_percent=float(second),
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            )
            for second in range(end_second + 1)
        )
        live_end = start + timedelta(seconds=end_second)
        visible_seconds = cli.GRAPH_HISTORY_SECONDS // 2 + 1
        expected_offset = (
            cli.GRAPH_HISTORY_SECONDS - (visible_seconds - 1) + cli.GRAPH_HISTORY_LABEL_GUTTER_SECONDS
        )

        self.assertEqual(cli.graph_view_max_offset_seconds(samples, live_end), cli.GRAPH_HISTORY_SECONDS)
        self.assertEqual(
            cli.graph_view_max_offset_seconds(samples, live_end, visible_seconds=visible_seconds),
            expected_offset,
        )
        self.assertEqual(
            cli.clamped_graph_view_offset(2000, samples, live_end, visible_seconds=visible_seconds),
            expected_offset,
        )
        self.assertGreater(
            cli.graph_view_max_offset_seconds(
                samples,
                live_end,
                visible_seconds=cli.GRAPH_HISTORY_BUCKETS + cli.GRAPH_HISTORY_LABEL_GUTTER_SECONDS // 2,
            ),
            0,
        )

    def test_graph_pan_uses_window_limit_even_before_history_fills(self) -> None:
        start = datetime(2026, 6, 22, 12, 0, 0)
        samples = (
            cli.MetricSample(
                timestamp=start,
                avg_cpu_percent=1.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            ),
        )

        self.assertEqual(cli.graph_view_max_offset_seconds(samples, start), cli.GRAPH_HISTORY_SECONDS)
        self.assertGreater(
            cli.graph_view_max_offset_seconds(samples, start, visible_seconds=20),
            0,
        )

    def test_live_loop_redraws_graph_once_per_second_when_table_interval_is_slower(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=24, width=100)

        class FakeCollector:
            def raise_if_failed(self) -> None:
                return None

            def latest_after(self, sequence: int):
                return None

        class FakeClock:
            def __init__(self) -> None:
                self.current = 0.0

            def monotonic(self) -> float:
                return self.current

            def advance(self, seconds: float) -> None:
                self.current += seconds

        class FakeLive:
            def __init__(self, clock: FakeClock) -> None:
                self.clock = clock
                self.update_times: list[float] = []

            def update(self, renderable, refresh: bool = False) -> None:
                self.update_times.append(self.clock.current)

        class FakeKeyboard:
            def __init__(self, clock: FakeClock, live: FakeLive) -> None:
                self.clock = clock
                self.live = live

            def read_keys(self, timeout: float):
                self.clock.advance(timeout)
                if len(self.live.update_times) >= 3:
                    return ["q"]
                return []

        clock = FakeClock()
        live = FakeLive(clock)
        original_monotonic = cli.time.monotonic

        try:
            cli.time.monotonic = clock.monotonic
            cli.poll_live_until_quit(
                live,
                FakeKeyboard(clock, live),
                Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0)),
                cli.MetricsHistory(max_samples=120),
                cli.ProcessViewState(),
                FakeConsole(),
                FakeCollector(),
                interval=2.0,
            )
        finally:
            cli.time.monotonic = original_monotonic

        self.assertEqual([round(update_time, 1) for update_time in live.update_times[:3]], [1.0, 2.0, 3.0])

    def test_live_loop_updates_display_time_on_one_second_redraw(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=24, width=100)

        class FakeCollector:
            def raise_if_failed(self) -> None:
                return None

            def latest_after(self, sequence: int):
                return None

        class FakeClock:
            def __init__(self) -> None:
                self.current = 0.0

            def monotonic(self) -> float:
                return self.current

            def advance(self, seconds: float) -> None:
                self.current += seconds

        class FakeLive:
            def __init__(self, clock: FakeClock) -> None:
                self.clock = clock
                self.update_times: list[float] = []

            def update(self, renderable, refresh: bool = False) -> None:
                self.update_times.append(self.clock.current)

        class FakeKeyboard:
            def __init__(self, clock: FakeClock, live: FakeLive) -> None:
                self.clock = clock
                self.live = live

            def read_keys(self, timeout: float):
                self.clock.advance(timeout)
                if len(self.live.update_times) >= 1:
                    return ["q"]
                return []

        initial_display_time = datetime(2026, 6, 22, 12, 0, 0)
        display_times = []
        clock = FakeClock()
        live = FakeLive(clock)
        original_monotonic = cli.time.monotonic
        original_render_live_snapshot = cli.render_live_snapshot

        def fake_render_live_snapshot(*args, display_time=None, **kwargs):
            display_times.append(display_time)
            return "rendered"

        try:
            cli.time.monotonic = clock.monotonic
            cli.render_live_snapshot = fake_render_live_snapshot
            cli.poll_live_until_quit(
                live,
                FakeKeyboard(clock, live),
                Snapshot(timestamp=initial_display_time),
                cli.MetricsHistory(max_samples=120),
                cli.ProcessViewState(),
                FakeConsole(),
                FakeCollector(),
                interval=2.0,
                display_time=initial_display_time,
            )
        finally:
            cli.time.monotonic = original_monotonic
            cli.render_live_snapshot = original_render_live_snapshot

        self.assertEqual([round(update_time, 1) for update_time in live.update_times[:1]], [1.0])
        self.assertGreaterEqual(len(display_times), 1)
        self.assertNotEqual(display_times[0], initial_display_time)

    def test_live_render_snapshot_uses_cached_graph_frame(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=35, width=160)

        history = cli.MetricsHistory(max_samples=120)
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=10.0,
                avg_mem_percent=20.0,
                avg_gpu_percent=30.0,
                avg_gpu_mem_percent=40.0,
            )
        )
        graph_frame = cli.GraphFrame(
            display_time=datetime(2026, 6, 22, 12, 0, 0),
            history_samples=history.samples,
        )
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 1),
                avg_cpu_percent=90.0,
                avg_mem_percent=80.0,
                avg_gpu_percent=70.0,
                avg_gpu_mem_percent=60.0,
            )
        )

        console = Console(width=160, record=True, file=StringIO())
        console.print(
            cli.render_live_snapshot(
                Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 1)),
                history,
                cli.ProcessViewState(),
                FakeConsole(),
                graph_frame=graph_frame,
            )
        )
        output = console.export_text()

        self.assertIn("Avg %CPU: 90.0%", output)
        self.assertIn("Avg %GPU: 70.0%", output)

    def test_live_render_snapshot_uses_graph_view_offset_from_live(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=35, width=80)

        history = cli.MetricsHistory(max_samples=120)
        start = datetime(2026, 6, 22, 12, 0, 0)
        history.append_sample(
            cli.MetricSample(
                timestamp=start,
                avg_cpu_percent=10.0,
                avg_mem_percent=20.0,
                avg_gpu_percent=30.0,
                avg_gpu_mem_percent=40.0,
            )
        )
        history.append_sample(
            cli.MetricSample(
                timestamp=start + timedelta(seconds=100),
                avg_cpu_percent=90.0,
                avg_mem_percent=80.0,
                avg_gpu_percent=70.0,
                avg_gpu_mem_percent=60.0,
            )
        )
        graph_frame = cli.capture_graph_frame(history, start + timedelta(seconds=100))
        state = cli.ProcessViewState(graph_view_offset_seconds=20)

        console = Console(width=80, record=True, file=StringIO())
        console.print(
            cli.render_live_snapshot(
                Snapshot(timestamp=start + timedelta(seconds=100)),
                history,
                state,
                FakeConsole(),
                graph_frame=graph_frame,
            )
        )
        output = console.export_text()

        self.assertIn("Avg %CPU: 90.0%", output)
        self.assertIn("Avg %GPU: 70.0%", output)

    def test_live_render_snapshot_keeps_live_labels_when_panned_before_history(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=35, width=80)

        start = datetime(2026, 6, 22, 12, 0, 0)
        history = cli.MetricsHistory(max_samples=120)
        history.append_sample(
            cli.MetricSample(
                timestamp=start,
                avg_cpu_percent=12.0,
                avg_mem_percent=34.0,
                avg_gpu_percent=56.0,
                avg_gpu_mem_percent=78.0,
            )
        )
        graph_frame = cli.capture_graph_frame(history, start)
        state = cli.ProcessViewState(graph_view_offset_seconds=120)

        console = Console(width=80, record=True, file=StringIO())
        console.print(
            cli.render_live_snapshot(
                Snapshot(timestamp=start),
                history,
                state,
                FakeConsole(),
                graph_frame=graph_frame,
            )
        )
        output = console.export_text()

        self.assertIn("Avg %CPU: 12.0%", output)
        self.assertIn("Avg %GPU: 56.0%", output)
        self.assertNotIn("Avg %CPU: N/A", output)
        self.assertNotIn("Avg %GPU: N/A", output)

    def test_display_graph_frame_keeps_panned_view_running_with_live_frame(self) -> None:
        history = cli.MetricsHistory(max_samples=120)
        for second in range(41):
            history.append_sample(
                cli.MetricSample(
                    timestamp=datetime(2026, 6, 22, 12, 0, second),
                    avg_cpu_percent=float(second),
                    avg_mem_percent=None,
                    avg_gpu_percent=None,
                    avg_gpu_mem_percent=None,
                )
            )
        state = cli.ProcessViewState(graph_view_offset_seconds=10)
        first_live = cli.GraphFrame(
            display_time=datetime(2026, 6, 22, 12, 0, 30),
            history_samples=cli.graph_samples_until(
                history.samples,
                datetime(2026, 6, 22, 12, 0, 30),
                history.max_samples,
            ),
        )
        next_live = cli.GraphFrame(
            display_time=datetime(2026, 6, 22, 12, 0, 31),
            history_samples=cli.graph_samples_until(
                history.samples,
                datetime(2026, 6, 22, 12, 0, 31),
                history.max_samples,
            ),
        )

        first_display, first_offset = cli.display_graph_frame(history, state, first_live)
        next_display, next_offset = cli.display_graph_frame(history, state, next_live)

        self.assertEqual(first_offset, 10)
        self.assertEqual(next_offset, 10)
        self.assertEqual(first_display.display_time, datetime(2026, 6, 22, 12, 0, 20))
        self.assertEqual(next_display.display_time, datetime(2026, 6, 22, 12, 0, 21))

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

    def test_live_loop_starts_and_stops_debug_collector_only_while_debug_is_open(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=24, width=120)

        class FakeKeyboard:
            def __init__(self) -> None:
                self.keys = iter((["0"], ["d"], ["d"], ["q"]))

            def read_keys(self, timeout: float):
                return next(self.keys)

        class FakeLive:
            def update(self, renderable, refresh: bool = False) -> None:
                return None

        class FakeSnapshotCollector:
            def raise_if_failed(self) -> None:
                return None

            def latest_after(self, sequence: int):
                return None

        class FakeDebugCollector:
            def __init__(self) -> None:
                self.started = 0
                self.stopped = 0
                self.targets: list[int] = []
                created.append(self)

            def start(self) -> None:
                self.started += 1

            def stop(self) -> None:
                self.stopped += 1

            def update_target(self, snapshot: Snapshot, gpu_index: int) -> None:
                self.targets.append(gpu_index)

            def latest_after(self, sequence: int):
                return None

        created: list[FakeDebugCollector] = []
        state = cli.ProcessViewState()
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[GpuInfo(index=0)],
            processes=[ProcessInfo(gpu_index=0, pid=42, args="python train.py")],
        )

        cli.poll_live_until_quit(
            FakeLive(),
            FakeKeyboard(),
            snapshot,
            cli.MetricsHistory(max_samples=120),
            state,
            FakeConsole(),
            FakeSnapshotCollector(),
            interval=1.0,
            debug_collector_factory=FakeDebugCollector,
        )

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].started, 1)
        self.assertEqual(created[0].stopped, 1)
        self.assertIn(0, created[0].targets)
        self.assertEqual(state.mode, MODE_NORMAL)

    def test_live_loop_stops_debug_collector_on_quit_from_debug_mode(self) -> None:
        class FakeConsole:
            @property
            def size(self) -> FakeConsoleSize:
                return FakeConsoleSize(height=24, width=120)

        class FakeKeyboard:
            def __init__(self) -> None:
                self.keys = iter((["0"], ["d"], [KEY_CTRL_C]))

            def read_keys(self, timeout: float):
                return next(self.keys)

        class FakeLive:
            def update(self, renderable, refresh: bool = False) -> None:
                return None

        class FakeSnapshotCollector:
            def raise_if_failed(self) -> None:
                return None

            def latest_after(self, sequence: int):
                return None

        class FakeDebugCollector:
            def __init__(self) -> None:
                self.stopped = 0
                created.append(self)

            def start(self) -> None:
                return None

            def stop(self) -> None:
                self.stopped += 1

            def update_target(self, snapshot: Snapshot, gpu_index: int) -> None:
                return None

            def latest_after(self, sequence: int):
                return None

        created: list[FakeDebugCollector] = []
        state = cli.ProcessViewState()
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[GpuInfo(index=0)],
            processes=[ProcessInfo(gpu_index=0, pid=42, args="python train.py")],
        )

        cli.poll_live_until_quit(
            FakeLive(),
            FakeKeyboard(),
            snapshot,
            cli.MetricsHistory(max_samples=120),
            state,
            FakeConsole(),
            FakeSnapshotCollector(),
            interval=1.0,
            debug_collector_factory=FakeDebugCollector,
        )

        self.assertEqual(state.mode, MODE_GPU_DEBUG)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].stopped, 1)

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
        state = cli.ProcessViewState(filter_query="train", filter_input="train", gpu_filter_index=0)
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
        self.assertIsNone(state.gpu_filter_index)
        self.assertEqual([row.pid for row in processes], [1, 2])

    def test_handle_key_batch_filters_by_gpu_number_from_snapshot_gpus(self) -> None:
        state = cli.ProcessViewState(selected_pid=1)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[GpuInfo(index=0), GpuInfo(index=1), GpuInfo(index=2)],
            processes=[
                ProcessInfo(gpu_index=0, pid=1, args="rank-0"),
                ProcessInfo(gpu_index=1, pid=2, args="rank-1"),
            ],
        )

        quit_requested, processes = cli.handle_key_batch(snapshot, state, ["2"])

        self.assertFalse(quit_requested)
        self.assertEqual(state.gpu_filter_index, 2)
        self.assertEqual(processes, [])
        self.assertIsNone(state.selected_pid)

        quit_requested, processes = cli.handle_key_batch(snapshot, state, ["1"])

        self.assertFalse(quit_requested)
        self.assertEqual(state.gpu_filter_index, 1)
        self.assertEqual([row.pid for row in processes], [2])
        self.assertEqual(state.selected_pid, 2)

    def test_handle_key_batch_toggles_process_zoom_without_reordering_processes(self) -> None:
        state = cli.ProcessViewState(selected_pid=1)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=1, args="rank-0"),
                ProcessInfo(gpu_index=1, pid=2, args="rank-1"),
            ],
        )

        view_before = cli.process_view_key(state)
        quit_requested, processes = cli.handle_key_batch(snapshot, state, ["z"])

        self.assertFalse(quit_requested)
        self.assertTrue(state.process_zoomed)
        self.assertNotEqual(cli.process_view_key(state), view_before)
        self.assertEqual([row.pid for row in processes], [1, 2])
        self.assertEqual(state.selected_pid, 1)

    def test_handle_key_batch_pans_graph_history_and_resets_to_live(self) -> None:
        state = cli.ProcessViewState(selected_pid=1)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 30),
            processes=[ProcessInfo(gpu_index=0, pid=1, args="rank-0")],
        )
        history = cli.MetricsHistory(max_samples=120)
        for second in range(31):
            history.append_sample(
                cli.MetricSample(
                    timestamp=datetime(2026, 6, 22, 12, 0, second),
                    avg_cpu_percent=float(second),
                    avg_mem_percent=None,
                    avg_gpu_percent=None,
                    avg_gpu_mem_percent=None,
                )
            )
        graph_frame = cli.capture_graph_frame(history, datetime(2026, 6, 22, 12, 0, 30))

        cli.handle_key_batch(snapshot, state, [","], history=history, graph_frame=graph_frame)
        self.assertEqual(state.graph_view_offset_seconds, 10)

        cli.handle_key_batch(snapshot, state, [",", ",", ","], history=history, graph_frame=graph_frame)
        self.assertEqual(state.graph_view_offset_seconds, 40)

        cli.handle_key_batch(snapshot, state, ["."], history=history, graph_frame=graph_frame)
        self.assertEqual(state.graph_view_offset_seconds, 30)

        cli.handle_key_batch(snapshot, state, [".", ".", "."], history=history, graph_frame=graph_frame)
        self.assertEqual(state.graph_view_offset_seconds, 0)

        cli.handle_key_batch(snapshot, state, ["r"], history=history, graph_frame=graph_frame)
        self.assertEqual(state.graph_view_offset_seconds, 0)

    def test_handle_key_batch_pans_gpu_graph_view(self) -> None:
        state = cli.ProcessViewState(selected_pid=1, gpu_graphs_visible=True)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 30),
            gpus=[GpuInfo(index=0)],
            processes=[ProcessInfo(gpu_index=0, pid=1, args="rank-0")],
        )
        history = cli.MetricsHistory(max_samples=120)
        for second in range(31):
            history.append_sample(
                cli.MetricSample(
                    timestamp=datetime(2026, 6, 22, 12, 0, second),
                    avg_cpu_percent=None,
                    avg_mem_percent=None,
                    avg_gpu_percent=None,
                    avg_gpu_mem_percent=None,
                    gpu_metrics=(cli.GpuMetricSample(index=0, utilization_percent=float(second), memory_percent=0.0),),
                )
            )
        graph_frame = cli.capture_graph_frame(history, datetime(2026, 6, 22, 12, 0, 30))

        cli.handle_key_batch(snapshot, state, [","], history=history, graph_frame=graph_frame)
        self.assertEqual(state.graph_view_offset_seconds, 10)

        cli.handle_key_batch(snapshot, state, ["r"], history=history, graph_frame=graph_frame)
        self.assertEqual(state.graph_view_offset_seconds, 0)

    def test_handle_key_batch_pans_focused_gpu_graph_view(self) -> None:
        state = cli.ProcessViewState(selected_pid=1, gpu_filter_index=0)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 30),
            gpus=[GpuInfo(index=0)],
            processes=[ProcessInfo(gpu_index=0, pid=1, args="rank-0")],
        )
        history = cli.MetricsHistory(max_samples=120)
        for second in range(31):
            history.append_sample(
                cli.MetricSample(
                    timestamp=datetime(2026, 6, 22, 12, 0, second),
                    avg_cpu_percent=None,
                    avg_mem_percent=None,
                    avg_gpu_percent=None,
                    avg_gpu_mem_percent=None,
                    gpu_metrics=(cli.GpuMetricSample(index=0, utilization_percent=float(second), memory_percent=0.0),),
                )
            )
        graph_frame = cli.capture_graph_frame(history, datetime(2026, 6, 22, 12, 0, 30))

        cli.handle_key_batch(snapshot, state, [","], history=history, graph_frame=graph_frame)
        self.assertEqual(state.graph_view_offset_seconds, 10)

        cli.handle_key_batch(snapshot, state, ["r"], history=history, graph_frame=graph_frame)
        self.assertEqual(state.graph_view_offset_seconds, 0)

    def test_focused_gpu_graph_pan_clamps_to_full_width_left_edge(self) -> None:
        start = datetime(2026, 6, 22, 12, 0, 0)
        history = cli.MetricsHistory(max_samples=cli.GRAPH_HISTORY_SECONDS + 301)
        for second in range(cli.GRAPH_HISTORY_SECONDS + 301):
            history.append_sample(
                cli.MetricSample(
                    timestamp=start + timedelta(seconds=second),
                    avg_cpu_percent=None,
                    avg_mem_percent=None,
                    avg_gpu_percent=None,
                    avg_gpu_mem_percent=None,
                    gpu_metrics=(cli.GpuMetricSample(index=0, utilization_percent=float(second % 100), memory_percent=0.0),),
                )
            )
        graph_frame = cli.capture_graph_frame(history, start + timedelta(seconds=cli.GRAPH_HISTORY_SECONDS + 300))
        snapshot = Snapshot(
            timestamp=graph_frame.display_time,
            gpus=[GpuInfo(index=0)],
            processes=[ProcessInfo(gpu_index=0, pid=1, args="rank-0")],
        )
        focused_state = cli.ProcessViewState(selected_pid=1, gpu_filter_index=0)
        normal_state = cli.ProcessViewState(selected_pid=1)
        terminal_width = 300
        focused_max_offset = cli.graph_view_max_offset_seconds(
            history.samples,
            graph_frame.display_time,
            visible_seconds=cli.graph_view_visible_seconds(terminal_width, focused_state),
        )
        normal_max_offset = cli.graph_view_max_offset_seconds(
            history.samples,
            graph_frame.display_time,
            visible_seconds=cli.graph_view_visible_seconds(terminal_width, normal_state),
        )

        cli.handle_key_batch(
            snapshot,
            focused_state,
            [","] * 200,
            history=history,
            graph_frame=graph_frame,
            terminal_width=terminal_width,
        )

        self.assertLess(focused_max_offset, normal_max_offset)
        self.assertEqual(focused_state.graph_view_offset_seconds, focused_max_offset)

    def test_handle_key_batch_uses_terminal_width_for_graph_pan(self) -> None:
        start = datetime(2026, 6, 22, 12, 0, 0)
        history = cli.MetricsHistory(max_samples=400)
        for second in range(cli.GRAPH_HISTORY_SECONDS + 121):
            history.append_sample(
                cli.MetricSample(
                    timestamp=start + timedelta(seconds=second),
                    avg_cpu_percent=float(second),
                    avg_mem_percent=None,
                    avg_gpu_percent=None,
                    avg_gpu_mem_percent=None,
                )
            )
        graph_frame = cli.capture_graph_frame(
            history,
            start + timedelta(seconds=cli.GRAPH_HISTORY_SECONDS + 120),
        )
        snapshot = Snapshot(
            timestamp=graph_frame.display_time,
            processes=[ProcessInfo(gpu_index=0, pid=1, args="rank-0")],
        )

        narrow_state = cli.ProcessViewState(selected_pid=1)
        wide_state = cli.ProcessViewState(selected_pid=1)
        pan_keys = [","] * 200

        cli.handle_key_batch(
            snapshot,
            narrow_state,
            pan_keys,
            history=history,
            graph_frame=graph_frame,
            terminal_width=20,
        )
        cli.handle_key_batch(
            snapshot,
            wide_state,
            pan_keys,
            history=history,
            graph_frame=graph_frame,
            terminal_width=500,
        )

        self.assertGreater(narrow_state.graph_view_offset_seconds, wide_state.graph_view_offset_seconds)
        self.assertGreater(wide_state.graph_view_offset_seconds, 0)

    def test_handle_key_batch_ignores_graph_pan_keys_when_process_zoomed(self) -> None:
        state = cli.ProcessViewState(selected_pid=1, process_zoomed=True, graph_view_offset_seconds=10)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 30),
            processes=[ProcessInfo(gpu_index=0, pid=1, args="rank-0")],
        )
        history = cli.MetricsHistory(max_samples=120)
        history.append_sample(
            cli.MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 30),
                avg_cpu_percent=30.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            )
        )
        graph_frame = cli.capture_graph_frame(history, datetime(2026, 6, 22, 12, 0, 30))

        cli.handle_key_batch(snapshot, state, [",", ".", "r"], history=history, graph_frame=graph_frame)

        self.assertEqual(state.graph_view_offset_seconds, 10)

    def test_graph_pan_keys_remain_text_input_in_search_and_filter_modes(self) -> None:
        state = cli.ProcessViewState(selected_pid=1)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[ProcessInfo(gpu_index=0, pid=1, args="rank-0")],
        )

        cli.handle_key_batch(snapshot, state, ["/", ",", ".", "r"])
        self.assertEqual(state.search_input, ",.r")
        self.assertEqual(state.graph_view_offset_seconds, 0)

        state.handle_key("esc", snapshot.processes)
        cli.handle_key_batch(snapshot, state, ["f", ",", ".", "r"])
        self.assertEqual(state.filter_query, ",.r")
        self.assertEqual(state.graph_view_offset_seconds, 0)

    def test_handle_key_batch_recomputes_processes_after_tree_toggle(self) -> None:
        state = cli.ProcessViewState(selected_pid=42)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=42, ppid=7, args="python train.py"),
            ],
            process_ancestors=[
                ProcessInfo(gpu_index=None, pid=7, args="bash launcher"),
            ],
        )

        quit_requested, processes = cli.handle_key_batch(snapshot, state, ["t", "k"])

        self.assertFalse(quit_requested)
        self.assertTrue(state.tree_mode)
        self.assertEqual([row.pid for row in processes], [7, 42])
        self.assertEqual(state.selected_pid, 7)

        quit_requested, processes = cli.handle_key_batch(snapshot, state, ["t"])

        self.assertFalse(quit_requested)
        self.assertFalse(state.tree_mode)
        self.assertEqual([row.pid for row in processes], [42])
        self.assertEqual(state.selected_pid, 42)

    def test_handle_key_batch_recomputes_tree_before_parent_jump(self) -> None:
        state = cli.ProcessViewState(selected_pid=42)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=42, ppid=7, args="python train.py"),
            ],
            process_ancestors=[
                ProcessInfo(gpu_index=None, pid=7, args="bash launcher"),
            ],
        )

        quit_requested, processes = cli.handle_key_batch(snapshot, state, ["t", "p"])

        self.assertFalse(quit_requested)
        self.assertTrue(state.tree_mode)
        self.assertEqual([row.pid for row in processes], [7, 42])
        self.assertEqual(state.selected_pid, 7)

    def test_sibling_keys_need_current_process_display(self) -> None:
        state = cli.ProcessViewState(tree_mode=True)

        self.assertFalse(cli.key_needs_current_processes(state, "?"))
        for key in ("h", "l", KEY_LEFT, KEY_RIGHT):
            with self.subTest(key=key):
                self.assertTrue(cli.key_needs_current_processes(state, key))

        state.tree_mode = False
        self.assertFalse(cli.key_needs_current_processes(state, "h"))

    def test_handle_key_batch_opens_and_closes_help_without_changing_selection(self) -> None:
        state = cli.ProcessViewState(selected_pid=42)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=42, args="python train.py"),
            ],
        )

        quit_requested, processes = cli.handle_key_batch(snapshot, state, ["?"])

        self.assertFalse(quit_requested)
        self.assertEqual(state.mode, MODE_HELP)
        self.assertEqual([row.pid for row in processes], [42])
        self.assertEqual(state.selected_pid, 42)

        quit_requested, processes = cli.handle_key_batch(snapshot, state, ["?"])

        self.assertFalse(quit_requested)
        self.assertEqual(state.mode, MODE_NORMAL)
        self.assertEqual([row.pid for row in processes], [42])
        self.assertEqual(state.selected_pid, 42)

    def test_handle_key_batch_opens_process_info_with_proc_detail(self) -> None:
        state = cli.ProcessViewState(selected_pid=42)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=42, ppid=7, args="python train.py"),
            ],
            process_ancestors=[
                ProcessInfo(gpu_index=None, pid=7, args="bash launcher"),
            ],
        )

        with patch("roctop.cli.read_process_detail", return_value=ProcessDetailInfo(pid=42, state="S")) as detail_read:
            quit_requested, processes = cli.handle_key_batch(snapshot, state, ["i"])

        self.assertFalse(quit_requested)
        self.assertEqual(state.mode, MODE_PROCESS_INFO)
        self.assertEqual(state.process_info_process.pid, 42)
        self.assertEqual(state.process_info_detail.state, "S")
        self.assertEqual(state.process_info_parent.pid, 7)
        self.assertEqual([row.pid for row in processes], [42])
        detail_read.assert_called_once_with(42)

    def test_handle_key_batch_does_not_read_process_detail_for_other_keys(self) -> None:
        state = cli.ProcessViewState(selected_pid=42)
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=42, args="python train.py"),
                ProcessInfo(gpu_index=0, pid=43, args="python serve.py"),
            ],
        )

        with patch("roctop.cli.read_process_detail") as detail_read:
            cli.handle_key_batch(snapshot, state, ["j"])

        detail_read.assert_not_called()

    def test_process_info_mode_keys_do_not_need_current_process_display(self) -> None:
        state = cli.ProcessViewState(mode=MODE_PROCESS_INFO)

        for key in ("i", "j", KEY_LEFT, KEY_RIGHT):
            with self.subTest(key=key):
                self.assertFalse(cli.key_needs_current_processes(state, key))

    def test_background_collector_schedules_from_collect_start_time(self) -> None:
        interval = 0.12
        collect_seconds = 0.06
        collect_starts: list[float] = []
        collected_three = threading.Event()

        def fake_collect_snapshot() -> Snapshot:
            collect_starts.append(time.perf_counter())
            if len(collect_starts) >= 3:
                collected_three.set()
            time.sleep(collect_seconds)
            return Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, len(collect_starts)))

        collector = cli.BackgroundSnapshotCollector(interval=interval, collect_func=fake_collect_snapshot)
        collector.start()
        try:
            self.assertTrue(collected_three.wait(timeout=1.0))
        finally:
            collector.stop()

        intervals = [end - start for start, end in zip(collect_starts, collect_starts[1:])]
        self.assertGreaterEqual(len(intervals), 2)
        self.assertLess(max(intervals[:2]), interval + collect_seconds * 0.75)

    def test_background_collector_samples_history_without_live_render(self) -> None:
        sampled_three = threading.Event()

        class CountingHistory(cli.MetricsHistory):
            def __init__(self) -> None:
                super().__init__(max_samples=120, stat_path="/missing/stat", meminfo_path="/missing/meminfo")
                self.added_snapshots: list[Snapshot] = []

            def add_snapshot(self, snapshot: Snapshot):
                sample = super().add_snapshot(snapshot)
                self.added_snapshots.append(snapshot)
                if len(self.added_snapshots) >= 3:
                    sampled_three.set()
                return sample

        calls = 0

        def fake_collect_snapshot() -> Snapshot:
            nonlocal calls
            calls += 1
            return Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, calls))

        history = CountingHistory()
        collector = cli.BackgroundSnapshotCollector(
            interval=0.01,
            collect_func=fake_collect_snapshot,
            history=history,
        )
        collector.start()
        try:
            self.assertTrue(sampled_three.wait(timeout=1.0))
        finally:
            collector.stop()

        self.assertGreaterEqual(len(history.added_snapshots), 3)
        self.assertGreaterEqual(len(history.samples), 3)

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

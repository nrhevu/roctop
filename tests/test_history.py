from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from roctop.history import (
    MetricsHistory,
    average_gpu_mem_percent,
    average_gpu_percent,
    cpu_percent_from_times,
    parse_cpu_times,
    parse_mem_percent,
)
from roctop.models import GpuInfo, Snapshot


class HistoryTests(unittest.TestCase):
    def test_cpu_percent_from_stat_delta(self) -> None:
        previous = parse_cpu_times("cpu  100 0 100 800 0 0 0 0 0 0\n")
        current = parse_cpu_times("cpu  150 0 150 900 0 0 0 0 0 0\n")
        self.assertIsNotNone(previous)
        self.assertIsNotNone(current)
        self.assertAlmostEqual(cpu_percent_from_times(previous, current), 50.0)

    def test_mem_percent_from_meminfo(self) -> None:
        text = "MemTotal:       1000 kB\nMemAvailable:    250 kB\n"
        self.assertAlmostEqual(parse_mem_percent(text), 75.0)

    def test_gpu_averages_from_snapshot(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[
                GpuInfo(
                    index=0,
                    memory_used_bytes=25,
                    memory_total_bytes=100,
                    utilization_percent=20,
                ),
                GpuInfo(
                    index=1,
                    memory_used_bytes=75,
                    memory_total_bytes=100,
                    utilization_percent=80,
                ),
            ],
        )
        self.assertAlmostEqual(average_gpu_percent(snapshot), 50.0)
        self.assertAlmostEqual(average_gpu_mem_percent(snapshot), 50.0)

    def test_history_reads_system_metrics_and_keeps_window(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stat_path = root / "stat"
            meminfo_path = root / "meminfo"
            stat_path.write_text("cpu  100 0 100 800 0 0 0 0 0 0\n")
            meminfo_path.write_text("MemTotal:       1000 kB\nMemAvailable:    400 kB\n")

            history = MetricsHistory(max_samples=1, stat_path=stat_path, meminfo_path=meminfo_path)
            first = history.add_snapshot(Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0)))
            self.assertIsNone(first.avg_cpu_percent)
            self.assertAlmostEqual(first.avg_mem_percent, 60.0)

            stat_path.write_text("cpu  150 0 150 900 0 0 0 0 0 0\n")
            second = history.add_snapshot(Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 1)))
            self.assertAlmostEqual(second.avg_cpu_percent, 50.0)
            self.assertEqual(len(history.samples), 1)
            self.assertIs(history.samples[0], second)

    def test_malformed_system_metrics_return_none(self) -> None:
        self.assertIsNone(parse_cpu_times("not cpu data\n"))
        self.assertIsNone(parse_mem_percent("MemTotal:       0 kB\n"))
        history = MetricsHistory(stat_path="/missing/stat", meminfo_path="/missing/meminfo")
        sample = history.add_snapshot(Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0)))
        self.assertIsNone(sample.avg_cpu_percent)
        self.assertIsNone(sample.avg_mem_percent)


if __name__ == "__main__":
    unittest.main()

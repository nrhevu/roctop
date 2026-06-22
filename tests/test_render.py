from __future__ import annotations

import unittest
from io import StringIO
from datetime import datetime

from rich.console import Console

from roctop.history import MetricSample, MetricsHistory
from roctop.models import GpuInfo, ProcessInfo, Snapshot
from roctop.render import (
    bar_with_percent,
    metric_graph_lines,
    percent_style,
    render_process_table,
    render_snapshot,
)


class RenderTests(unittest.TestCase):
    def test_snapshot_renders_at_narrow_width(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            driver_version="6.14.14",
            gpus=[
                GpuInfo(
                    index=0,
                    name="AMD Instinct",
                    guid="29921",
                    gpu_type="AMD MI350",
                    gfx_version="gfx950",
                    temperature_c=60,
                    fan_percent=42,
                    power_w=266,
                    sclk_mhz=173,
                    mclk_mhz=2000,
                    memory_used_bytes=1024 * 1024 * 1024,
                    memory_total_bytes=4 * 1024 * 1024 * 1024,
                    utilization_percent=42,
                )
            ],
            processes=[
                ProcessInfo(
                    gpu_index=0,
                    pid=123,
                    user="root",
                    cpu_percent=12.3,
                    host_mem_percent=0.4,
                    elapsed="01:02",
                    command="python",
                    args="python train.py --long-argument",
                    gpu_memory_bytes=512 * 1024 * 1024,
                    gpu_memory_percent=12.5,
                )
            ],
        )
        narrow_console = Console(width=80, record=True, file=StringIO())
        narrow_console.print(render_snapshot(snapshot))

        console = Console(width=180, record=True, file=StringIO())
        console.print(render_snapshot(snapshot))
        output = console.export_text()
        self.assertIn("roctop", output)
        self.assertIn("DID", output)
        self.assertIn("GUID", output)
        self.assertNotIn("DIDs", output)
        self.assertNotIn("GUIDs", output)
        self.assertNotIn("IDs (DID, GUID)", output)
        self.assertNotIn("Name", output)
        self.assertIn("AMD", output)
        self.assertIn("AMD Instinct", output)
        self.assertIn("29921", output)
        self.assertNotIn("GUID:", output)
        self.assertIn("Type: AMD MI350", output)
        self.assertIn("GFX: gfx950", output)
        self.assertNotIn("│ Type", output)
        self.assertIn("60°C", output)
        self.assertIn("42%", output)
        self.assertIn("266W", output)
        self.assertIn("173MHz", output)
        self.assertIn("2000MHz", output)
        self.assertIn("25.0%", output)
        self.assertIn("42%", output)
        self.assertNotIn("GPU-Util", output)
        self.assertIn("123", output)
        self.assertIn("%GPU-MEM", output)
        self.assertNotIn("Avg %CPU", output)

    def test_snapshot_renders_history_graphs_between_tables(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[
                GpuInfo(
                    index=0,
                    memory_used_bytes=1024 * 1024 * 1024,
                    memory_total_bytes=4 * 1024 * 1024 * 1024,
                    utilization_percent=42,
                )
            ],
            processes=[
                ProcessInfo(
                    gpu_index=0,
                    pid=123,
                    user="root",
                    cpu_percent=12.3,
                    host_mem_percent=0.4,
                    elapsed="01:02",
                    args="python train.py",
                    gpu_memory_bytes=512 * 1024 * 1024,
                    gpu_memory_percent=12.5,
                )
            ],
        )
        history = MetricsHistory(max_samples=120)
        history.append_sample(
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=37.3,
                avg_mem_percent=77.2,
                avg_gpu_percent=33.2,
                avg_gpu_mem_percent=54.6,
            )
        )
        narrow_console = Console(width=80, record=True, file=StringIO())
        narrow_console.print(render_snapshot(snapshot, history))

        console = Console(width=180, record=True, file=StringIO())
        console.print(render_snapshot(snapshot, history))
        output = console.export_text()
        self.assertIn("Avg %CPU: 37.3%", output)
        self.assertIn("Avg %GPU: 33.2%", output)
        self.assertIn("Avg %MEM: 77.2%", output)
        self.assertIn("Avg %GPU MEM: 54.6%", output)
        self.assertLess(output.index("UTL"), output.index("Avg %CPU"))
        self.assertLess(output.index("Avg %CPU"), output.index("PID"))

    def test_low_history_values_draw_visible_trace_on_right(self) -> None:
        lines = metric_graph_lines([None, 5.0, 12.0], width=6, height=15, style="green")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[-1].plain, "    ⠤⠶")

    def test_fan_column_visible_when_unsupported(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[
                GpuInfo(
                    index=0,
                    name="AMD Instinct",
                    temperature_c=60,
                    power_w=266,
                    memory_used_bytes=1024 * 1024 * 1024,
                    memory_total_bytes=4 * 1024 * 1024 * 1024,
                    utilization_percent=42,
                )
            ],
        )
        console = Console(width=120, record=True, file=StringIO())
        console.print(render_snapshot(snapshot))
        output = console.export_text()
        self.assertIn("Fan", output)
        self.assertIn("N/A", output)

    def test_high_util_bar_uses_red_style(self) -> None:
        console = Console(width=80, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(bar_with_percent(100, percent_style(100)))
        output = console.export_text(styles=True)
        self.assertIn("38;2;255;85;85", output)
        self.assertIn("100%", output)

    def test_process_metric_columns_are_colored(self) -> None:
        console = Console(width=140, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(
            render_process_table(
                [
                    ProcessInfo(
                        gpu_index=0,
                        pid=123,
                        user="root",
                        gpu_memory_bytes=512 * 1024 * 1024,
                        gpu_memory_percent=12.5,
                        cpu_percent=65.2,
                        host_mem_percent=88.4,
                        elapsed="01:02",
                        args="python train.py",
                    )
                ]
            )
        )
        output = console.export_text(styles=True)
        self.assertIn("%CPU", output)
        self.assertIn("%MEM", output)
        self.assertIn("38;2;255;85;85", output)
        self.assertIn("38;2;241;250;140", output)
        self.assertIn("38;2;80;250;123", output)


if __name__ == "__main__":
    unittest.main()

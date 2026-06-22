from __future__ import annotations

import unittest
from io import StringIO
from datetime import datetime

from rich.console import Console

from roctop.models import GpuInfo, ProcessInfo, Snapshot
from roctop.render import bar_with_percent, percent_style, render_process_table, render_snapshot


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
                    cu_occupancy=83,
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
        self.assertIn("IDs (DID, GUID)", output)
        self.assertNotIn("Name", output)
        self.assertIn("AMD", output)
        self.assertIn("AMD Instinct, 29921", output)
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
        self.assertIn("%GPU", output)
        self.assertIn("83", output)

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
                        cu_occupancy=88,
                        cpu_percent=65.2,
                        host_mem_percent=7.4,
                        elapsed="01:02",
                        args="python train.py",
                    )
                ]
            )
        )
        output = console.export_text(styles=True)
        self.assertIn("%GPU", output)
        self.assertIn("38;2;255;85;85", output)
        self.assertIn("38;2;241;250;140", output)
        self.assertIn("38;2;80;250;123", output)


if __name__ == "__main__":
    unittest.main()

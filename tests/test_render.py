from __future__ import annotations

import unittest
from io import StringIO
from datetime import datetime

from rich.console import Console

from roctop.models import GpuInfo, ProcessInfo, Snapshot
from roctop.render import bar_with_percent, render_snapshot


class RenderTests(unittest.TestCase):
    def test_snapshot_renders_at_narrow_width(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            driver_version="6.14.14",
            gpus=[
                GpuInfo(
                    index=0,
                    name="AMD Instinct",
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
        console = Console(width=120, record=True, file=StringIO())
        console.print(render_snapshot(snapshot))
        output = console.export_text()
        self.assertIn("roctop", output)
        self.assertIn("AMD", output)
        self.assertIn("60°C", output)
        self.assertIn("42%", output)
        self.assertIn("266W", output)
        self.assertIn("173MHz", output)
        self.assertIn("2000MHz", output)
        self.assertIn("25.0%", output)
        self.assertIn("42%", output)
        self.assertNotIn("GPU-Util", output)
        self.assertIn("123", output)

    def test_fan_column_hidden_when_unsupported(self) -> None:
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
        self.assertNotIn("Fan", output)

    def test_high_util_bar_uses_red_style(self) -> None:
        console = Console(width=80, force_terminal=True, color_system="standard", record=True, file=StringIO())
        console.print(bar_with_percent(100, "bold red"))
        output = console.export_text(styles=True)
        self.assertIn("\x1b[1;31m", output)
        self.assertIn("100%", output)


if __name__ == "__main__":
    unittest.main()

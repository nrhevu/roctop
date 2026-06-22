from __future__ import annotations

import unittest
from io import StringIO
from datetime import datetime

from rich.console import Console

from roctop.models import GpuInfo, ProcessInfo, Snapshot
from roctop.render import render_snapshot


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
        console = Console(width=60, record=True, file=StringIO())
        console.print(render_snapshot(snapshot))
        output = console.export_text()
        self.assertIn("roctop", output)
        self.assertIn("AMD", output)
        self.assertIn("Instinct", output)
        self.assertIn("60°C", output)
        self.assertIn("123", output)


if __name__ == "__main__":
    unittest.main()

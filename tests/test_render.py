from __future__ import annotations

import unittest
from io import StringIO
from datetime import datetime

from rich.console import Console

from roctop.history import MetricSample, MetricsHistory
from roctop.interaction import ProcessViewState
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

    def test_process_command_wraps_instead_of_truncating(self) -> None:
        long_args = "python -m sglang.launch_server --model-path /models/deepseek --tensor-parallel-size 8 --final-token"
        console = Console(width=90, record=True, file=StringIO())
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
                        host_mem_percent=7.4,
                        elapsed="01:02",
                        command="python",
                        args=long_args,
                    )
                ]
            )
        )
        output = console.export_text()
        self.assertIn("--model-path", output)
        self.assertIn("--final-token", output)

    def test_process_view_state_renders_title_and_selected_row(self) -> None:
        state = ProcessViewState(selected_pid=123, viewport_rows=4)
        console = Console(width=120, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
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
                        host_mem_percent=7.4,
                        elapsed="01:02",
                        args="python train.py",
                    )
                ],
                process_state=state,
                max_rows=4,
            )
        )
        plain = console.export_text(clear=False)
        styled = console.export_text(styles=True)
        self.assertIn("Processes  1/1  sort: default", plain)
        self.assertNotIn("j/k move", plain)
        self.assertIn("48;2;68;71;90", styled)

    def test_process_help_renders_in_header_with_key_labels(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=123, args="python train.py"),
                ProcessInfo(gpu_index=1, pid=456, args="python serve.py"),
            ],
        )
        state = ProcessViewState(selected_pid=456, viewport_rows=4)
        console = Console(width=180, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_snapshot(snapshot, process_state=state, terminal_height=40))
        plain = console.export_text(clear=False)
        styled = console.export_text(styles=True)
        self.assertIn("Processes: 2/2", plain)
        self.assertIn("Sort: default", plain)
        self.assertIn("j/k: move", plain)
        self.assertIn("PgUp/PgDn: scroll", plain)
        self.assertIn("s: sort", plain)
        self.assertIn("x: kill", plain)
        self.assertIn("q: quit", plain)
        self.assertIn("38;2;255;184;108", styled)

    def test_process_view_state_limits_visible_rows(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=101, args="cmd-101"),
            ProcessInfo(gpu_index=0, pid=102, args="cmd-102"),
            ProcessInfo(gpu_index=0, pid=103, args="cmd-103"),
            ProcessInfo(gpu_index=0, pid=104, args="cmd-104"),
            ProcessInfo(gpu_index=0, pid=105, args="cmd-105"),
        ]
        state = ProcessViewState(selected_pid=105, viewport_rows=2)
        console = Console(width=120, record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=2))
        output = console.export_text()
        self.assertNotIn("cmd-101", output)
        self.assertNotIn("cmd-102", output)
        self.assertIn("cmd-104", output)
        self.assertIn("cmd-105", output)

    def test_process_view_wraps_long_commands_within_visual_height(self) -> None:
        long_args = (
            "python3 -m nexus_titan.cli direct --config /tmp/qwen3_8b_hf_config_ft_pretrain.yaml "
            "--data-path /data --torchtitan-path /opt/NexusTitan/thirdparty/torchtitan --nnodes 2 "
            "--nproc-per-node 2 --rdzv-backend c10d --master-addr "
            "q8b-js-2e6cfbb5-0632-4341-b11c-9d7270d66811-replica-0-0."
            "q8b-js-2e6cfbb5-0632-4341-b11c-9d7270d66811.vunguyen13.svc.cluster.local"
        )
        processes = [
            ProcessInfo(gpu_index=index % 8, pid=200 + index, user="root", args=long_args)
            for index in range(20)
        ]
        state = ProcessViewState(selected_pid=219, viewport_rows=20)
        console = Console(width=120, record=True, file=StringIO())
        console.print(
            render_process_table(
                processes,
                process_state=state,
                max_rows=12,
                terminal_width=120,
            )
        )
        output = console.export_text()
        lines = output.splitlines()
        self.assertIn("219", output)
        self.assertNotIn("200", output)
        self.assertLessEqual(max(len(line) for line in lines), 120)
        self.assertLessEqual(len(lines), 18)


if __name__ == "__main__":
    unittest.main()

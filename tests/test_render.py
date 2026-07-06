from __future__ import annotations

import time
import unittest
from io import StringIO
from datetime import datetime, timedelta

from rich.console import Console, Group
from rich.text import Text

import roctop.render as render
from roctop.cli import handle_key_batch
from roctop.history import GpuMetricSample, MetricSample, MetricsHistory
from roctop.interaction import (
    KEY_DOWN,
    KEY_UP,
    MODE_FILTER,
    MODE_HELP,
    MODE_KILL_CONFIRM,
    MODE_PROCESS_INFO,
    MODE_SEARCH,
    ProcessViewState,
    max_process_info_scroll_offset,
)
from roctop.models import GpuInfo, ProcessDetailInfo, ProcessInfo, Snapshot
from roctop.render import (
    bar_with_percent,
    estimate_process_view_rows,
    metric_graph_lines,
    percent_style,
    render_metrics_graphs,
    render_process_table,
    render_snapshot,
)


def has_braille_dots(text: str) -> bool:
    return any("\u2801" <= char <= "\u28ff" for char in text)


def first_braille_index(text: str) -> int:
    for index, char in enumerate(text):
        if "\u2801" <= char <= "\u28ff":
            return index
    return -1


def process_header_index(plain: str, start: int = 0) -> int:
    offset = start
    for line in plain[start:].splitlines(keepends=True):
        if "│" in line and "GPU" in line and "COMMAND" in line:
            return offset
        offset += len(line)
    raise ValueError("process table header not found")


def synthetic_long_processes(count: int) -> list[ProcessInfo]:
    command = (
        "demo_worker --model-path /demo/models/example-checkpoint "
        "--tensor-parallel-size 8 --batch-size 64 --sequence-length 8192 --final-flag"
    )
    return [
        ProcessInfo(gpu_index=index % 8, pid=1000 + index, user="demo", args=f"{command} --rank {index}")
        for index in range(count)
    ]


class RenderTests(unittest.TestCase):
    def test_gpu_table_columns_align_right(self) -> None:
        table = render.render_gpu_table(
            [
                GpuInfo(
                    index=0,
                    guid="29921",
                    memory_used_bytes=1024 * 1024,
                    memory_total_bytes=1024 * 1024 * 1024,
                    utilization_percent=42,
                )
            ]
        )

        self.assertTrue(all(column.justify == "right" for column in table.columns))

    def test_snapshot_renders_at_narrow_width(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            node_name="node-a",
            driver_version="6.14.14",
            gpus=[
                GpuInfo(
                    index=0,
                    name="AMD Instinct",
                    guid="29921",
                    gpu_type="AMD Instinct MI350X",
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
                    user="demo",
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
        self.assertIn("roctop @ node-a", output)
        self.assertIn("GUID", output)
        self.assertNotIn("DID", output)
        self.assertNotIn("DIDs", output)
        self.assertNotIn("GUIDs", output)
        self.assertNotIn("IDs (DID, GUID)", output)
        self.assertNotIn("Name", output)
        self.assertIn("AMD", output)
        self.assertIn("29921", output)
        self.assertNotIn("GUID:", output)
        self.assertIn("Model: AMD Instinct MI350X", output)
        self.assertIn("Architecture: gfx950", output)
        self.assertNotIn("Type:", output)
        self.assertNotIn("GFX:", output)
        self.assertNotIn("│ Type", output)
        self.assertIn("60°C", output)
        self.assertIn("42%", output)
        self.assertIn("266W", output)
        self.assertIn("173MHz", output)
        self.assertIn("2000MHz", output)
        self.assertIn("%Memory-Usage", output)
        self.assertIn("%Utilization", output)
        self.assertIn("25.0%", output)
        self.assertIn("42.0%", output)
        self.assertIn("42%", output)
        self.assertNotIn("UTL", output)
        self.assertNotIn("GPU-Util", output)
        self.assertIn("123", output)
        self.assertIn("%GPU-MEM", output)
        self.assertNotIn("Avg %CPU", output)

    def test_snapshot_renders_history_graphs_between_tables(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            driver_version="6.14.14",
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
                    user="demo",
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

        console = Console(width=300, record=True, file=StringIO())
        console.print(render_snapshot(snapshot, history))
        output = console.export_text()
        self.assertIn("Avg %CPU: 37.3%", output)
        self.assertIn("Avg %GPU: 33.2%", output)
        self.assertIn("Avg %MEM: 77.2%", output)
        self.assertIn("Avg %GPU MEM: 54.6%", output)
        marker = "240s"
        self.assertIn(marker, output)
        self.assertIn("120s", output)
        self.assertIn("60s", output)
        self.assertLess(output.index("Avg %CPU: 37.3%"), output.index(marker))
        self.assertLess(output.index(marker), output.index("Avg %MEM: 77.2%"))
        self.assertLess(output.index("Avg %GPU: 33.2%"), output.index("Avg %GPU MEM: 54.6%"))
        self.assertLess(output.index("%Utilization"), output.index("Avg %CPU"))
        self.assertLess(output.index("Avg %CPU"), output.index("PID"))

    def test_gpu_focus_renders_selected_metrics_graph_and_processes(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            driver_version="6.14.14",
            gpus=[
                GpuInfo(
                    index=0,
                    name="AMD GPU 0",
                    guid="guid-0",
                    memory_used_bytes=1024 * 1024 * 1024,
                    memory_total_bytes=4 * 1024 * 1024 * 1024,
                    utilization_percent=12,
                ),
                GpuInfo(
                    index=1,
                    name="Accelerator 1",
                    guid="guid-1",
                    gpu_type="AMD Instinct MI350X",
                    gfx_version="gfx950",
                    vendor="Advanced Micro Devices, Inc. [AMD/ATI]",
                    vbios_version="113-D7020100-100",
                    pcie_bus="0000:03:00.0",
                    max_power_w=300,
                    performance_level="auto",
                    throttle_status="THERMAL",
                    voltage_mv=1138,
                    unique_id="gpu-unique-1",
                    sku="APM107573",
                    temperature_c=64,
                    fan_percent=50,
                    power_w=270,
                    sclk_mhz=1700,
                    mclk_mhz=2000,
                    memory_used_bytes=3 * 1024 * 1024 * 1024,
                    memory_total_bytes=4 * 1024 * 1024 * 1024,
                    utilization_percent=88,
                ),
            ],
            processes=[
                ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py"),
                ProcessInfo(
                    gpu_index=1,
                    pid=456,
                    user="demo",
                    cpu_percent=21.5,
                    host_mem_percent=3.3,
                    gpu_memory_bytes=512 * 1024 * 1024,
                    gpu_memory_percent=12.5,
                    elapsed="02:03",
                    args="python serve.py",
                ),
                ProcessInfo(
                    gpu_index=1,
                    pid=457,
                    user="worker",
                    cpu_percent=1.5,
                    host_mem_percent=0.7,
                    gpu_memory_bytes=1024 * 1024 * 1024,
                    gpu_memory_percent=25.0,
                    elapsed="04:05",
                    args="worker --rank 1",
                ),
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
                gpu_metrics=(
                    GpuMetricSample(index=0, utilization_percent=12.0, memory_percent=25.0),
                    GpuMetricSample(index=1, utilization_percent=88.0, memory_percent=75.0),
                ),
            )
        )
        state = ProcessViewState(selected_pid=456, gpu_filter_index=1, viewport_rows=4)
        console = Console(width=300, record=True, file=StringIO())
        console.print(render_snapshot(snapshot, history, state, terminal_height=45, terminal_width=300))
        output = console.export_text()

        self.assertIn("Focus: GPU 1", output)
        self.assertNotIn("GPU 1 Metrics", output)
        self.assertNotIn("Metric", output)
        self.assertIn("GPU: 1", output)
        self.assertIn("Name: Accelerator 1", output)
        self.assertIn("Vendor: Advanced Micro Devices, Inc. [AMD/ATI]", output)
        self.assertIn("Model: AMD Instinct MI350X", output)
        self.assertIn("SKU: APM107573", output)
        self.assertIn("Architecture: gfx950", output)
        self.assertIn("GUID: guid-1", output)
        self.assertIn("Unique ID: gpu-unique-1", output)
        self.assertIn("VBIOS: 113-D7020100-100", output)
        self.assertIn("Driver: 6.14.14", output)
        self.assertIn("PCIe: 0000:03:00.0", output)
        self.assertIn("Temperature: 64°C", output)
        self.assertIn("Fan: 50%", output)
        self.assertIn("Power:", output)
        self.assertIn("270W", output)
        self.assertIn("Max Power: 300W", output)
        self.assertIn("Perf: auto", output)
        self.assertNotIn("Throttle:", output)
        self.assertNotIn("THERMAL", output)
        self.assertIn("Voltage: 1138mV", output)
        self.assertIn("SCLK:", output)
        self.assertIn("1700MHz", output)
        self.assertIn("MCLK:", output)
        self.assertIn("2000MHz", output)
        self.assertIn("Memory Used:", output)
        self.assertIn("Memory Free:", output)
        self.assertIn("Memory Total:", output)
        self.assertIn("Memory Usage:", output)
        self.assertIn("Memory Free %:", output)
        self.assertIn("Utilization:", output)
        self.assertIn("88.0%", output)
        self.assertIn("Processes: 2", output)
        self.assertIn("Proc GPU Mem:", output)
        self.assertIn("1.50GiB", output)
        self.assertIn("Proc GPU Mem %: 37.5%", output)
        self.assertIn("Proc CPU: 23.0%", output)
        self.assertIn("Proc Host MEM: 4.0%", output)
        self.assertIn("Top Proc PID: 457", output)
        self.assertIn("Top Proc User: worker", output)
        self.assertIn("Top Proc Mem: 1.00GiB (25.0%)", output)
        self.assertIn("Top Proc Time: 04:05", output)
        self.assertIn("Top Proc Cmd: worker --rank 1", output)
        focused_metric_lines = [
            line
            for line in output.splitlines()
            if any(label in line for label in ("GPU: 1", "Model:", "Utilization:", "Memory Usage:", "Memory Free %:"))
        ]
        self.assertTrue(all(line.count("│") == 2 for line in focused_metric_lines))
        self.assertTrue(
            any(
                "GPU: 1" in line
                and "Processes: 2" in line
                and "Temperature:" in line
                and "Proc CPU:" in line
                and "Architecture:" in line
                for line in focused_metric_lines
            )
        )
        first_metric_line = next(
            line for line in focused_metric_lines if "GPU: 1" in line and "Processes: 2" in line
        )
        self.assertLess(first_metric_line.index("Temperature:"), first_metric_line.index("Processes: 2"))
        model_line = next(
            line
            for line in output.splitlines()
            if "Model: AMD Instinct MI350X" in line and "Power:" in line and "Top Proc Cmd:" in line
        )
        self.assertLess(model_line.index("Power:"), model_line.index("Top Proc Cmd:"))
        self.assertGreaterEqual(
            model_line.index("Top Proc Cmd:") - (model_line.index("Model:") + len("Model: AMD Instinct MI350X")),
            8,
        )
        self.assertNotIn("AMD GPU 0", output)
        self.assertNotIn("guid-0", output)
        self.assertIn("GPU 1", output)
        self.assertIn("%GPU: 88.0%", output)
        self.assertIn("%GPU MEM: 75.0%", output)
        self.assertNotIn("Avg %CPU:", output)
        self.assertNotIn("Avg %MEM:", output)
        self.assertNotIn("Avg %GPU:", output)
        self.assertIn("python serve.py", output)
        self.assertNotIn("python train.py", output)

    def test_snapshot_can_toggle_per_gpu_graphs(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[
                GpuInfo(index=0, memory_used_bytes=25, memory_total_bytes=100, utilization_percent=100),
                GpuInfo(index=1, memory_used_bytes=75, memory_total_bytes=100, utilization_percent=56),
                GpuInfo(index=2, memory_used_bytes=50, memory_total_bytes=100, utilization_percent=42),
                GpuInfo(index=3, memory_used_bytes=10, memory_total_bytes=100, utilization_percent=24),
            ],
            processes=[ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py")],
        )
        history = MetricsHistory(max_samples=120)
        history.append_sample(
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=37.3,
                avg_mem_percent=77.2,
                avg_gpu_percent=33.2,
                avg_gpu_mem_percent=54.6,
                gpu_metrics=(
                    GpuMetricSample(index=0, utilization_percent=100.0, memory_percent=100.0),
                    GpuMetricSample(index=1, utilization_percent=56.0, memory_percent=75.0),
                    GpuMetricSample(index=2, utilization_percent=42.0, memory_percent=50.0),
                    GpuMetricSample(index=3, utilization_percent=24.0, memory_percent=10.0),
                ),
            )
        )
        state = ProcessViewState(gpu_filter_index=1, gpu_graphs_visible=True)
        console = Console(width=180, record=True, file=StringIO())
        console.print(render_snapshot(snapshot, history, state, terminal_height=45, terminal_width=180))
        output = console.export_text()

        self.assertIn("GPU 0", output)
        self.assertIn("GPU 1", output)
        self.assertIn("GPU 2", output)
        self.assertIn("GPU 3", output)
        self.assertIn("%GPU: 100.0%", output)
        self.assertIn("%GPU MEM: 75.0%", output)
        self.assertNotIn("g: avg graph", output)
        self.assertNotIn("PID", output)
        self.assertNotIn("%Utilization", output)
        self.assertNotIn("Avg %GPU:", output)
        self.assertGreaterEqual(sum(1 for line in output.splitlines() if has_braille_dots(line)), 8)
        self.assertTrue(any(line.startswith("├") and "┼" in line for line in output.splitlines()))

    def test_header_can_render_live_subsecond_display_time(self) -> None:
        snapshot = Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0))
        console = Console(width=120, record=True, file=StringIO())
        console.print(
            render_snapshot(
                snapshot,
                display_time=datetime(2026, 6, 22, 12, 0, 1, 234000),
                show_subsecond_time=True,
            )
        )

        output = console.export_text()
        self.assertIn("Mon Jun 22 12:00:01.2 2026", output)
        self.assertNotIn("Mon Jun 22 12:00:00 2026", output)

    def test_low_history_values_draw_visible_trace_on_right(self) -> None:
        lines = metric_graph_lines([None, 5.0, 12.0], width=6, height=15, style="green")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[-1].plain, "     ⠴")

    def test_metric_graph_can_flip_dots_inside_braille_cell(self) -> None:
        normal = metric_graph_lines([25.0], width=1, height=4, style="green", trim_empty=False)
        inverted = metric_graph_lines(
            [25.0],
            width=1,
            height=4,
            style="green",
            trim_empty=False,
            invert_dots=True,
        )
        self.assertEqual(normal[0].plain, "⢀")
        self.assertEqual(inverted[0].plain, "⠈")

    def test_metric_graph_packs_two_time_columns_per_braille_cell(self) -> None:
        lines = metric_graph_lines([25.0, 25.0], width=1, height=4, style="green", trim_empty=False)
        self.assertEqual(lines[0].plain, "⣀")

    def test_gpu_graph_separator_respects_history_cap(self) -> None:
        line = render.gpu_graph_separator_line(700)
        plain = line.plain
        axis_start = render.graph_window_start_index(700)

        self.assertEqual(len(plain), 700)
        self.assertEqual(plain[: axis_start - len("1080s")], " " * (axis_start - len("1080s")))
        self.assertEqual(plain[axis_start], "├")
        self.assertIn("1080s", plain)
        self.assertNotIn("─", plain[: axis_start - len("1080s")])
        self.assertGreater(plain[axis_start:].count("─"), 500)

    def test_metric_values_follow_sample_timestamps(self) -> None:
        samples = [
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=10.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            ),
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 5),
                avg_cpu_percent=50.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            ),
        ]

        values = render.metric_values_by_time(
            samples,
            "avg_cpu_percent",
            seconds=8,
            end_time=datetime(2026, 6, 22, 12, 0, 5),
        )
        self.assertEqual(values[-1], 50.0)
        self.assertEqual(values[-6], 10.0)
        self.assertEqual(values[-2], 10.0)

        shifted_values = render.metric_values_by_time(
            samples,
            "avg_cpu_percent",
            seconds=8,
            end_time=datetime(2026, 6, 22, 12, 0, 7),
        )
        self.assertEqual(shifted_values[-3], 50.0)
        self.assertEqual(shifted_values[-4], 10.0)
        self.assertEqual(shifted_values[-1], 50.0)

    def test_metric_values_average_subsecond_samples_in_same_bucket(self) -> None:
        samples = [
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0, 100000),
                avg_cpu_percent=10.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            ),
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0, 900000),
                avg_cpu_percent=30.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            ),
        ]

        values = render.metric_values_by_time(
            samples,
            "avg_cpu_percent",
            seconds=3,
            end_time=datetime(2026, 6, 22, 12, 0, 0, 900000),
        )

        self.assertEqual(values[-1], 20.0)

    def test_metric_values_keep_last_bucket_when_frame_crosses_second(self) -> None:
        samples = [
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0, 100000),
                avg_cpu_percent=10.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            ),
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0, 900000),
                avg_cpu_percent=30.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            ),
        ]

        current_second = render.metric_values_by_time(
            samples,
            "avg_cpu_percent",
            seconds=3,
            end_time=datetime(2026, 6, 22, 12, 0, 0, 950000),
        )
        next_second = render.metric_values_by_time(
            samples,
            "avg_cpu_percent",
            seconds=3,
            end_time=datetime(2026, 6, 22, 12, 0, 1, 100000),
        )

        self.assertEqual(current_second[-1], 20.0)
        self.assertEqual(next_second[-2], 20.0)
        self.assertEqual(next_second[-1], 20.0)

    def test_metric_values_keep_one_second_scale_for_long_history_window(self) -> None:
        samples = [
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=10.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            ),
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 18, 0),
                avg_cpu_percent=50.0,
                avg_mem_percent=None,
                avg_gpu_percent=None,
                avg_gpu_mem_percent=None,
            ),
        ]

        values = render.metric_values_by_time(
            samples,
            "avg_cpu_percent",
            seconds=130 * render.GRAPH_COLUMNS_PER_CELL,
            end_time=datetime(2026, 6, 22, 12, 18, 0),
        )

        self.assertNotIn(10.0, values)
        self.assertEqual(values[-1], 50.0)

    def test_time_axis_uses_one_second_offsets(self) -> None:
        axis = render.time_axis_line(130).plain
        self.assertEqual(axis.index("120s"), 65)
        self.assertEqual(axis.index("60s"), 96)
        self.assertEqual(axis.index("30s"), 111)
        self.assertEqual(axis[64], " ")
        self.assertEqual(axis[95], " ")
        self.assertEqual(axis[110], " ")
        self.assertEqual(axis[69], "├")
        self.assertEqual(axis[99], "├")
        self.assertEqual(axis[114], "├")

    def test_time_axis_offsets_markers_when_graph_is_panned(self) -> None:
        axis = render.time_axis_line(130, offset_seconds=10).plain

        self.assertEqual(axis.index("120s"), 70)
        self.assertEqual(axis.index("60s"), 101)
        self.assertEqual(axis.index("30s"), 116)
        self.assertEqual(axis[74], "├")
        self.assertEqual(axis[104], "├")
        self.assertEqual(axis[119], "├")

    def test_time_axis_keeps_marker_label_at_left_edge(self) -> None:
        width = 128
        offset_seconds = render.GRAPH_HISTORY_SECONDS - (
            width - 1 - len("1080s")
        ) * render.GRAPH_COLUMNS_PER_CELL

        axis = render.time_axis_line(width, offset_seconds=offset_seconds).plain

        self.assertEqual(axis[:6], "1080s├")

    def test_time_axis_adds_long_window_markers(self) -> None:
        axis = render.time_axis_line(550).plain
        self.assertNotIn("─", axis[:3])
        self.assertEqual(axis[3:10], " 1080s├")
        self.assertEqual(axis[184:190], " 720s├")
        self.assertEqual(axis[364:370], " 360s├")
        self.assertEqual(axis[424:430], " 240s├")
        self.assertEqual(axis[484:490], " 120s├")
        self.assertEqual(axis[515:520], " 60s├")
        self.assertEqual(axis[530:535], " 30s├")
        self.assertEqual(axis[9], "├")
        self.assertEqual(axis[189], "├")
        self.assertEqual(axis[369], "├")
        self.assertEqual(axis[429], "├")
        self.assertEqual(axis[489], "├")
        self.assertEqual(axis[519], "├")
        self.assertEqual(axis[534], "├")

    def test_time_axis_crops_old_labels_on_narrow_width(self) -> None:
        axis = render.time_axis_line(50).plain
        self.assertNotIn("120s", axis)
        self.assertEqual(axis.index("60s"), 16)
        self.assertEqual(axis.index("30s"), 31)
        self.assertEqual(axis[15], " ")
        self.assertEqual(axis[30], " ")
        self.assertEqual(axis[19], "├")
        self.assertEqual(axis[34], "├")

    def test_time_axis_handles_very_narrow_width(self) -> None:
        axis = render.time_axis_line(2).plain
        self.assertEqual(axis, "──")
        self.assertNotIn("30s", axis)

    def test_metric_graph_pair_keeps_axes_and_bottom_labels_aligned(self) -> None:
        history = MetricsHistory(max_samples=120)
        history.append_sample(
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=3.0,
                avg_mem_percent=8.0,
                avg_gpu_percent=4.0,
                avg_gpu_mem_percent=55.0,
            )
        )
        console = Console(width=300, record=True, file=StringIO())
        console.print(render_metrics_graphs(history))
        lines = console.export_text().splitlines()
        axis_lines = [index for index, line in enumerate(lines) if "240s" in line]
        bottom_label_lines = [
            index for index, line in enumerate(lines) if "Avg %MEM:" in line and "Avg %GPU MEM:" in line
        ]
        self.assertEqual(len(axis_lines), 1)
        self.assertEqual(len(bottom_label_lines), 1)
        self.assertLess(axis_lines[0], bottom_label_lines[0])

    def test_metric_graph_pair_keeps_left_edge_time_label_in_both_columns(self) -> None:
        history = MetricsHistory(max_samples=120)
        history.append_sample(
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=3.0,
                avg_mem_percent=8.0,
                avg_gpu_percent=4.0,
                avg_gpu_mem_percent=55.0,
            )
        )
        width = 128
        offset_seconds = render.GRAPH_HISTORY_SECONDS - (
            width - 1 - len("1080s")
        ) * render.GRAPH_COLUMNS_PER_CELL
        console = Console(width=264, record=True, file=StringIO())
        console.print(render_metrics_graphs(history, time_offset_seconds=offset_seconds))

        self.assertEqual(console.export_text().count("1080s"), 2)

    def test_metric_graph_slides_long_history_without_compressing_window(self) -> None:
        total_seconds = render.GRAPH_HISTORY_SECONDS * 2
        history = MetricsHistory(max_samples=total_seconds + 1)
        start_time = datetime(2026, 6, 22, 12, 0, 0)
        for second in range(total_seconds + 1):
            history.append_sample(
                MetricSample(
                    timestamp=start_time + timedelta(seconds=second),
                    avg_cpu_percent=10.0,
                    avg_mem_percent=20.0,
                    avg_gpu_percent=30.0,
                    avg_gpu_mem_percent=40.0,
                )
            )

        console = Console(width=300, record=True, file=StringIO())
        console.print(
            render_metrics_graphs(
                history,
                end_time=start_time + timedelta(seconds=total_seconds),
                samples=history.samples,
            )
        )

        output = console.export_text()
        self.assertIn("240s", output)
        self.assertIn("120s", output)
        self.assertIn("Avg %GPU: 30.0%", output)

    def test_metric_graph_does_not_draw_data_before_history_limit_marker(self) -> None:
        total_seconds = render.GRAPH_HISTORY_SECONDS * 2
        history = MetricsHistory(max_samples=total_seconds + 1)
        start_time = datetime(2026, 6, 22, 12, 0, 0)
        for second in range(total_seconds + 1):
            history.append_sample(
                MetricSample(
                    timestamp=start_time + timedelta(seconds=second),
                    avg_cpu_percent=None,
                    avg_mem_percent=None,
                    avg_gpu_percent=50.0,
                    avg_gpu_mem_percent=50.0,
                )
            )

        console = Console(width=300, record=True, file=StringIO())
        console.print(
            render_metrics_graphs(
                history,
                end_time=start_time + timedelta(seconds=total_seconds),
                samples=history.samples,
            )
        )
        lines = console.export_text().splitlines()
        gpu_axis = next(line for line in lines if "240s" in line)
        marker_index = gpu_axis.index("240s")
        gpu_graph_lines = [
            line
            for line in lines
            if has_braille_dots(line) and "Avg %" not in line
        ]
        first_dot = min(index for line in gpu_graph_lines if (index := first_braille_index(line)) >= 0)

        self.assertGreaterEqual(first_dot, marker_index + render.GRAPH_WINDOW_DATA_PADDING_CELLS)

    def test_metric_graph_bottom_half_sticks_to_time_axis(self) -> None:
        history = MetricsHistory(max_samples=120)
        history.append_sample(
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=0.0,
                avg_mem_percent=8.0,
                avg_gpu_percent=8.0,
                avg_gpu_mem_percent=0.0,
            )
        )
        console = Console(width=120, record=True, file=StringIO())
        console.print(render_metrics_graphs(history))
        lines = console.export_text().splitlines()
        axis_index = next(index for index, line in enumerate(lines) if "30s" in line)
        bottom_label_index = next(index for index, line in enumerate(lines) if "Avg %MEM:" in line)
        first_bottom_graph_line = lines[axis_index + 1]
        last_bottom_graph_line = lines[bottom_label_index - 1]
        self.assertTrue(has_braille_dots(first_bottom_graph_line))
        self.assertFalse(has_braille_dots(last_bottom_graph_line))

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

    def test_percent_bar_expands_to_available_width(self) -> None:
        narrow_console = Console(width=40, record=True, file=StringIO())
        narrow_console.print(bar_with_percent(50, percent_style(50)))
        wide_console = Console(width=100, record=True, file=StringIO())
        wide_console.print(bar_with_percent(50, percent_style(50)))

        narrow_line = narrow_console.export_text().splitlines()[0]
        wide_line = wide_console.export_text().splitlines()[0]
        self.assertEqual(len(narrow_line), 40)
        self.assertEqual(len(wide_line), 100)
        self.assertGreater(wide_line.index("50%"), narrow_line.index("50%"))

    def test_process_metric_columns_are_colored(self) -> None:
        console = Console(width=140, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(
            render_process_table(
                [
                    ProcessInfo(
                        gpu_index=0,
                        pid=123,
                        user="demo",
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
        long_args = "demo_server --model-path /demo/models/deepseek --tensor-parallel-size 8 --final-flag"
        console = Console(width=90, record=True, file=StringIO())
        console.print(
            render_process_table(
                [
                    ProcessInfo(
                        gpu_index=0,
                        pid=123,
                        user="demo",
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
        self.assertIn("--final-flag", output)

    def test_process_view_state_renders_title_and_selected_row(self) -> None:
        state = ProcessViewState(selected_pid=123, viewport_rows=4)
        console = Console(width=120, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(
            render_process_table(
                [
                    ProcessInfo(
                        gpu_index=0,
                        pid=123,
                        user="demo",
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
        self.assertIn("Processes  1/1", plain)
        self.assertNotIn("sort: default", plain)
        self.assertNotIn("j/k move", plain)
        self.assertIn("48;2;68;71;90", styled)

    def test_process_view_highlights_only_selected_duplicate_pid_row(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=123, args="cmd on gpu 0"),
            ProcessInfo(gpu_index=1, pid=123, args="cmd on gpu 1"),
            ProcessInfo(gpu_index=2, pid=456, args="other cmd"),
        ]
        state = ProcessViewState(viewport_rows=4)
        state.sync(processes)
        state.move_selection(processes, 1)

        console = Console(width=120, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4))

        styled = console.export_text(styles=True)
        selected_lines = [line for line in styled.splitlines() if "48;2;68;71;90" in line]
        self.assertEqual(len(selected_lines), 1)
        self.assertIn("cmd on gpu 1", selected_lines[0])
        self.assertNotIn("cmd on gpu 0", selected_lines[0])

    def test_process_table_keeps_command_column_position_when_scrolled(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=1, user="u", args="alpha-command"),
            ProcessInfo(gpu_index=7, pid=222222222, user="verylonguser", args="beta-command"),
        ]
        positions = []
        for pid, command in ((1, "alpha-command"), (222222222, "beta-command")):
            state = ProcessViewState(selected_pid=pid, viewport_rows=1)
            console = Console(width=120, record=True, file=StringIO())
            console.print(
                render_process_table(
                    processes,
                    process_state=state,
                    max_rows=1,
                    terminal_width=120,
                )
            )
            line = next(line for line in console.export_text().splitlines() if command in line)
            positions.append(line.index(command))

        self.assertEqual(positions[0], positions[1])

    def test_process_table_elapsed_time_advances_with_display_offset(self) -> None:
        process = ProcessInfo(gpu_index=0, pid=123, user="demo", elapsed="09:04:23", args="python train.py")
        console = Console(width=120, record=True, file=StringIO())
        console.print(
            render_process_table(
                [process],
                process_state=ProcessViewState(selected_pid=123, viewport_rows=1),
                max_rows=1,
                terminal_width=120,
                elapsed_offset_seconds=1,
            )
        )

        self.assertIn("09:04:24", console.export_text())

    def test_process_window_moves_cursor_up_before_scrolling(self) -> None:
        long_args = (
            "demo_compile_worker --pickler torch_worker_pool --kind fork --workers 32 "
            "--parent 365 --read-fd 72 --write-fd 77 --cache-key demo"
        )
        processes = [
            ProcessInfo(gpu_index=None, pid=9000 + index, user="demo", args=long_args)
            for index in range(23)
        ]
        snapshot = Snapshot(timestamp=datetime(2026, 6, 22, 12, 0, 0), processes=processes)
        state = ProcessViewState(selected_pid=9015, viewport_rows=12)

        def visible_pids(display_processes=None) -> list[int]:
            console = Console(width=150, record=True, file=StringIO())
            console.print(
                render_process_table(
                    display_processes or processes,
                    process_state=state,
                    max_rows=12,
                    terminal_width=150,
                    processes_sorted=display_processes is not None,
                )
            )
            output = console.export_text()
            return [proc.pid for proc in processes if str(proc.pid) in output]

        before_pids = visible_pids()
        self.assertEqual(before_pids[-1], 9015)

        _quit_requested, display_processes = handle_key_batch(snapshot, state, [KEY_UP])
        after_pids = visible_pids(display_processes)

        self.assertEqual(after_pids, before_pids)
        self.assertEqual(state.selected_pid, 9014)
        self.assertEqual(after_pids.index(state.selected_pid), len(after_pids) - 2)

    def test_tree_window_moves_cursor_up_before_scrolling_after_key_batch(self) -> None:
        parent_args = (
            "demo_server --model-path demo-model --host 0.0.0.0 --port 30054 "
            "--tensor-parallel-size 4 --context-length 32768 --watchdog-timeout 1200"
        )
        worker_args = (
            "demo_compile_worker --pickler torch_worker_pool --kind fork --workers 32 "
            "--parent 254 --read-fd 119 --write-fd 122 --cache-key demo"
        )
        ancestors = [
            ProcessInfo(gpu_index=None, pid=7000, user="demo", args="demo_init"),
            ProcessInfo(gpu_index=None, pid=7001, user="demo", ppid=7000, args=parent_args),
        ]
        processes: list[ProcessInfo] = []
        for index in range(8):
            scheduler_pid = 7100 + index
            processes.append(
                ProcessInfo(
                    gpu_index=index if index < 4 else None,
                    pid=scheduler_pid,
                    user="demo",
                    ppid=7001,
                    args=f"demo::scheduler_TP{index}",
                )
            )
            processes.append(
                ProcessInfo(
                    gpu_index=None,
                    pid=7200 + index,
                    user="demo",
                    ppid=scheduler_pid,
                    args=worker_args,
                )
            )
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=processes,
            process_ancestors=ancestors,
        )
        state = ProcessViewState(selected_pid=7103, tree_mode=True, viewport_rows=12)

        def visible_pids(display_processes=None) -> list[int]:
            console = Console(width=150, record=True, file=StringIO())
            console.print(
                render_snapshot(
                    snapshot,
                    process_state=state,
                    terminal_height=24,
                    terminal_width=150,
                    display_processes=display_processes,
                )
            )
            output = console.export_text()
            return [proc.pid for proc in (*ancestors, *processes) if str(proc.pid) in output]

        before_pids = visible_pids()
        self.assertIn(7103, before_pids)
        selected_before_index = before_pids.index(7103)

        _quit_requested, display_processes = handle_key_batch(snapshot, state, [KEY_UP])
        after_pids = visible_pids(display_processes)

        self.assertEqual(after_pids, before_pids)
        self.assertEqual(state.selected_pid, 7202)
        self.assertEqual(after_pids.index(state.selected_pid), selected_before_index - 1)

    def test_process_help_renders_action_keys_without_navigation_hints(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[GpuInfo(index=0), GpuInfo(index=1)],
            processes=[
                ProcessInfo(gpu_index=0, pid=123, args="python train.py"),
                ProcessInfo(gpu_index=1, pid=456, args="python serve.py"),
            ],
        )
        state = ProcessViewState(selected_pid=456, viewport_rows=4)
        console = Console(width=180, color_system="truecolor", record=True, file=StringIO())
        console.print(render_snapshot(snapshot, process_state=state, terminal_height=40, terminal_width=180))
        plain = console.export_text(clear=False)
        styled = console.export_text(styles=True)
        self.assertNotIn("Processes:", plain)
        self.assertNotIn("Sort:", plain)
        self.assertIn("Processes  2/2", plain)
        self.assertNotIn("sort: default", plain)
        self.assertNotIn("j/k: move", plain)
        self.assertNotIn("PgUp/PgDn: scroll", plain)
        self.assertNotIn("n/N: next/prev", plain)
        self.assertIn("?: help", plain)
        self.assertIn("s: sort", plain)
        self.assertIn("t: tree", plain)
        self.assertIn("/: search", plain)
        self.assertIn("f: filter", plain)
        self.assertIn("z: zoom", plain)
        self.assertIn("g: graphs", plain)
        self.assertNotIn(",/. graph", plain)
        self.assertNotIn("r: live", plain)
        self.assertIn("<0-1>: focus", plain)
        self.assertIn("x: kill", plain)
        self.assertIn("i: inspect", plain)
        self.assertNotIn("Space: select", plain)
        self.assertIn("q: quit", plain)
        self.assertLess(plain.index("<0-1>: focus"), plain.index("s: sort"))
        self.assertLess(plain.index("Mon Jun 22"), plain.index("s: sort"))
        self.assertLess(plain.index("z: zoom"), plain.index("g: graphs"))
        self.assertLess(plain.index("g: graphs"), plain.index("i: inspect"))
        self.assertLess(plain.index("i: inspect"), plain.index("x: kill"))
        self.assertLess(plain.index("i: inspect"), plain.index("?: help"))
        self.assertLess(plain.index("x: kill"), plain.index("?: help"))
        self.assertLess(plain.index("?: help"), plain.index("q: quit"))
        self.assertIn("38;2;255;184;108", styled)

        wide_console = Console(width=120, record=True, file=StringIO())
        wide_console.print(render_snapshot(snapshot, process_state=state, terminal_height=40, terminal_width=120))
        wide_plain = wide_console.export_text(clear=False)
        self.assertIn("<0-1>: focus  s: sort", wide_plain)

    def test_process_help_shows_escape_close_when_gpu_graphs_visible(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[GpuInfo(index=0), GpuInfo(index=1)],
            processes=[ProcessInfo(gpu_index=0, pid=123, args="python train.py")],
        )
        state = ProcessViewState(gpu_graphs_visible=True, viewport_rows=4)
        console = Console(width=180, color_system="truecolor", record=True, file=StringIO())

        console.print(render_snapshot(snapshot, process_state=state, terminal_height=40, terminal_width=180))
        plain = console.export_text(clear=False)

        self.assertIn("g: avg graph", plain)
        self.assertIn("Esc: close", plain)
        self.assertLess(plain.index("g: avg graph"), plain.index("Esc: close"))
        self.assertLess(plain.index("Esc: close"), plain.index("i: inspect"))

    def test_help_popup_overlays_process_table_without_reserving_rows(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[GpuInfo(index=index) for index in range(4)],
            processes=[
                ProcessInfo(gpu_index=0, pid=123, args="python train.py"),
            ],
        )
        normal_state = ProcessViewState(selected_pid=123, viewport_rows=4)
        state = ProcessViewState(selected_pid=123, mode=MODE_HELP, viewport_rows=4)
        console = Console(width=140, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_snapshot(snapshot, process_state=state, terminal_height=45, terminal_width=140))
        plain = console.export_text(clear=False)
        styled = console.export_text(styles=True)

        self.assertEqual(
            estimate_process_view_rows(snapshot, None, 45, normal_state),
            estimate_process_view_rows(snapshot, None, 45, state),
        )
        self.assertIn("roctop 0.4.2 - AMD GPU/process monitor for ROCm", plain)
        self.assertIn("Colors:", plain)
        self.assertIn("green  : good headroom, low pressure", plain)
        self.assertIn("yellow : high usage, worth watching", plain)
        self.assertIn("red    : critical usage, hot, full, or high pressure", plain)
        self.assertIn("cyan   : clocks, active selections, focused values", plain)
        self.assertIn("blue   : labels and secondary metric names", plain)
        self.assertIn("gray   : inactive, unavailable, or dimmed values", plain)
        self.assertNotIn("KEY", plain)
        self.assertNotIn("ACTION", plain)
        self.assertNotIn("MODE", plain)
        self.assertIn("<0-3>", plain)
        self.assertIn("<0-3>: focus GPU", plain)
        self.assertIn("Esc: close graphs/menus", plain)
        self.assertNotIn("close graphs, clear selection/filter, or cancel active mode", plain)
        self.assertIn("z: zoom process table", plain)
        self.assertIn("g: toggle GPU graphs", plain)
        self.assertIn("Esc: close GPU graphs", plain)
        self.assertIn(",/.: pan older/newer", plain)
        self.assertIn("r: reset to live", plain)
        self.assertIn("j/k, Up/Down: scroll popup one row", plain)
        self.assertIn("h/l, Left/Right: page popup up/down", plain)
        self.assertIn("Press Esc or ? to return.", plain)
        esc_lines = [line for line in plain.splitlines() if "Esc: close graphs/menus" in line]
        self.assertEqual(len(esc_lines), 1)
        self.assertNotIn("inspect selected process", esc_lines[0])
        self.assertIn("38;2;80;250;123", styled)
        self.assertIn("38;2;241;250;140", styled)
        self.assertIn("38;2;255;85;85", styled)

    def test_help_key_lines_wrap_long_actions(self) -> None:
        lines = render.help_key_lines(
            "Esc",
            "close graphs, clear selection/filter, or cancel active mode",
            key_width=8,
            max_width=34,
        )

        plain_lines = [line.plain for line in lines]
        self.assertGreater(len(plain_lines), 1)
        self.assertTrue(all(len(line) <= 34 for line in plain_lines))
        self.assertTrue(plain_lines[0].lstrip().startswith("Esc:"))
        self.assertTrue(plain_lines[1].startswith(" " * 10))
        self.assertNotIn(":", plain_lines[1])

    def test_help_overlay_only_replaces_popup_rectangle(self) -> None:
        base_line = "L" + "." * 118 + "R"
        base = Group(*(Text(base_line) for _ in range(25)))
        state = ProcessViewState(mode=MODE_HELP)
        console = Console(width=120, record=True, file=StringIO())
        console.print(render.HelpOverlay(base, state, terminal_height=25, terminal_width=120))
        lines = console.export_text(clear=False).splitlines()

        help_row = next(line for line in lines if "roctop 0.4.2" in line)
        self.assertTrue(help_row.startswith("L"))
        self.assertTrue(help_row.rstrip().endswith("R"))
        self.assertIn("AMD GPU/process monitor", help_row)

    def test_help_popup_keeps_key_action_positions_aligned(self) -> None:
        action_positions = []
        for gpus in ([GpuInfo(index=index) for index in range(4)], [GpuInfo(index=index) for index in range(8)]):
            state = ProcessViewState(mode=MODE_HELP)
            console = Console(width=120, record=True, file=StringIO())
            console.print(render.render_help_popup(state, terminal_width=120, gpus=gpus))
            line = next(line for line in console.export_text().splitlines() if "move process cursor" in line)
            action_positions.append(line.index("move process cursor"))

        self.assertEqual(action_positions[0], action_positions[1])

    def test_process_info_popup_renders_selected_process_details(self) -> None:
        process = ProcessInfo(
            gpu_index=0,
            pid=42,
            ppid=7,
            user="demo",
            cpu_percent=12.5,
            host_mem_percent=3.2,
            elapsed="01:02",
            name="python",
            command="python",
            args="python train.py --really-long-flag value",
            gpu_memory_bytes=512 * 1024 * 1024,
            gpu_memory_percent=25.0,
        )
        parent = ProcessInfo(gpu_index=None, pid=7, user="demo", args="bash launcher")
        detail = ProcessDetailInfo(
            pid=42,
            state="S (sleeping)",
            threads=9,
            vm_rss_kib=2048,
            vm_size_kib=4096,
            vm_hwm_kib=8192,
            cpu_allowed_list="0-3",
            voluntary_ctxt_switches=12,
            nonvoluntary_ctxt_switches=3,
            cmdline="python train.py --really-long-flag value",
            cwd="/work/demo",
            exe="/usr/bin/python",
        )
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[
                GpuInfo(index=0, name="AMD GPU", gpu_type="AMD Instinct MI350X", guid="29921"),
            ],
            processes=[process],
            process_ancestors=[parent],
        )
        state = ProcessViewState(selected_pid=42, mode=MODE_PROCESS_INFO, viewport_rows=4)
        state.open_process_info(process, detail, parent=parent, child_count=2)
        console = Console(width=140, record=True, file=StringIO())
        console.print(render_snapshot(snapshot, process_state=state, terminal_height=45, terminal_width=140))
        plain = console.export_text(clear=False)

        self.assertIn("Process  42", plain)
        self.assertIn("PID", plain)
        self.assertIn("PPID", plain)
        self.assertIn("demo", plain)
        self.assertIn("GPU memory", plain)
        self.assertIn("512MiB", plain)
        self.assertIn("25.0%", plain)
        self.assertIn("CPU", plain)
        self.assertIn("12.5%", plain)
        self.assertIn("Parent", plain)
        self.assertIn("7 bash launcher", plain)
        self.assertIn("Visible children", plain)
        self.assertIn("S (sleeping)", plain)
        self.assertIn("Threads", plain)
        self.assertIn("9", plain)
        self.assertIn("VmRSS", plain)
        self.assertIn("2MiB", plain)
        self.assertIn("CPU affinity", plain)
        self.assertIn("0-3", plain)
        self.assertIn("Cwd", plain)
        self.assertIn("/work/demo", plain)
        self.assertIn("Exe", plain)
        self.assertIn("/usr/bin/python", plain)
        self.assertIn("--really-long-flag", plain)
        self.assertIn("j/k or Up/Down: scroll", plain)
        self.assertIn("h/l or Left/Right: page", plain)
        self.assertIn("i/Esc: close", plain)

    def test_process_info_popup_keeps_fixed_height_for_long_values(self) -> None:
        long_command = "bash -lc " + " ".join(f"--flag-{index} value-{index}" for index in range(80))
        process = ProcessInfo(
            gpu_index=None,
            pid=3000674,
            ppid=3000661,
            user="root",
            name="bash",
            command="bash",
            args="bash",
        )
        parent = ProcessInfo(gpu_index=None, pid=3000661, user="root", args=long_command)
        detail = ProcessDetailInfo(
            pid=3000674,
            cmdline=long_command,
            cwd="/work/demo",
            exe="/bin/bash",
        )
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[process],
            process_ancestors=[parent],
        )
        state = ProcessViewState(selected_pid=3000674, mode=MODE_PROCESS_INFO, viewport_rows=4)
        state.open_process_info(process, detail, parent=parent)
        console = Console(width=120, record=True, file=StringIO())
        console.print(render_snapshot(snapshot, process_state=state, terminal_height=35, terminal_width=120))
        plain = console.export_text(clear=False)

        self.assertLessEqual(len(plain.splitlines()), 35)
        self.assertIn("Process  3000674", plain)
        self.assertIn("Parent", plain)
        self.assertIn("--flag-0", plain)
        self.assertGreater(state.process_info_render_row_count, render.PROCESS_INFO_VISIBLE_ROWS)
        self.assertGreater(max_process_info_scroll_offset(state), 0)

        state.handle_key(KEY_DOWN, [process], processes_synced=True)
        self.assertEqual(state.process_info_scroll_offset, 1)

    def test_process_info_popup_keeps_value_column_position_when_scrolled(self) -> None:
        long_command = "/opt/conda/envs/py_3.10/bin/python3 -u /scratch/demo/train.py " + " ".join(
            f"--flag-{index} value-{index}" for index in range(80)
        )
        process = ProcessInfo(
            gpu_index=None,
            pid=3000674,
            ppid=3000661,
            user="root",
            name="python",
            command="python",
            args=long_command,
        )
        detail = ProcessDetailInfo(
            pid=3000674,
            state="S (sleeping)",
            threads=23,
            vm_rss_kib=1024,
            vm_size_kib=2048,
            cmdline=long_command,
        )
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[process],
        )
        value_positions = []
        for offset in (0, 14):
            state = ProcessViewState(selected_pid=3000674, mode=MODE_PROCESS_INFO, viewport_rows=4)
            state.open_process_info(process, detail, child_count=0)
            state.process_info_scroll_offset = offset
            console = Console(width=160, record=True, file=StringIO())
            console.print(render.render_process_info_popup(snapshot, state, terminal_width=160))
            line = next(line for line in console.export_text().splitlines() if "Command" in line)
            value_positions.append(line.index("/opt/conda"))

        self.assertEqual(value_positions[0], value_positions[1])

    def test_process_sort_indicator_renders_on_sorted_column_header(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=123, user="demo", cpu_percent=65.2, args="python train.py"),
            ProcessInfo(gpu_index=1, pid=456, user="demo", cpu_percent=12.3, args="python serve.py"),
        ]
        state = ProcessViewState(selected_pid=123, sort_field="cpu", sort_desc=True, viewport_rows=4)
        console = Console(width=140, record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        output = console.export_text()
        self.assertIn("%CPU ↓", output)
        self.assertNotIn("sort:", output)
        self.assertNotIn("Sorted by", output)

        state.sort_desc = False
        console = Console(width=140, record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        self.assertIn("%CPU ↑", console.export_text())

    def test_sort_menu_highlights_selected_field_without_pointer_text(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py"),
        ]
        state = ProcessViewState(selected_pid=123, mode="sort_menu", sort_menu_index=4, viewport_rows=4)
        console = Console(width=140, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        plain = console.export_text(clear=False)
        styled = console.export_text(styles=True)
        self.assertIn("Sort by:", plain)
        self.assertIn("%MEM", plain)
        self.assertNotIn(">%MEM", plain)
        self.assertLess(plain.index("Sort by:"), process_header_index(plain, plain.index("Sort by:")))
        self.assertIn("38;2;40;42;54", styled)
        self.assertIn("48;2;139;233;253", styled)

    def test_kill_confirm_renders_option_menu_without_yn_caption(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py"),
        ]
        state = ProcessViewState(selected_pid=123, mode=MODE_KILL_CONFIRM, viewport_rows=4)
        console = Console(width=140, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        plain = console.export_text(clear=False)
        styled = console.export_text(styles=True)
        self.assertIn("Kill PID 123:", plain)
        self.assertIn("Cancel", plain)
        self.assertIn("SIGTERM", plain)
        self.assertIn("SIGKILL", plain)
        self.assertNotIn("y/N", plain)
        self.assertNotIn("Kill cancelled", plain)
        self.assertLess(plain.index("Kill PID 123:"), process_header_index(plain, plain.index("Kill PID 123:")))
        self.assertIn("38;2;40;42;54", styled)
        self.assertIn("48;2;139;233;253", styled)

    def test_kill_confirm_renders_selected_pid_count(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py"),
            ProcessInfo(gpu_index=0, pid=456, user="demo", args="python serve.py"),
        ]
        state = ProcessViewState(
            selected_pids={123, 456},
            kill_confirm_pids=(123, 456),
            mode=MODE_KILL_CONFIRM,
            viewport_rows=4,
        )
        console = Console(width=140, record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        plain = console.export_text(clear=False)
        self.assertIn("Kill 2 selected PIDs:", plain)

    def test_process_table_backgrounds_selected_pids(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py"),
            ProcessInfo(gpu_index=0, pid=456, user="demo", args="python serve.py"),
        ]
        state = ProcessViewState(selected_pid=456, selected_pids={123}, viewport_rows=4)
        console = Console(width=140, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        plain = console.export_text(clear=False)
        styled = console.export_text(styles=True)
        train_line = next(line for line in styled.splitlines() if "python train.py" in line)
        serve_line = next(line for line in styled.splitlines() if "python serve.py" in line)
        title_line = next(line for line in plain.splitlines() if "Processes  2/2" in line)
        self.assertIn("48;2;79;152;163", train_line)
        self.assertNotIn("38;2;139;233;253", train_line)
        self.assertNotIn("48;2;79;152;163", serve_line)
        self.assertIn("Selected: 1", title_line)

        state = ProcessViewState(selected_pid=123, selected_pids={123}, viewport_rows=4)
        console = Console(width=140, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        styled = console.export_text(styles=True)
        train_line = next(line for line in styled.splitlines() if "python train.py" in line)
        self.assertIn("48;2;71;121;130", train_line)
        self.assertNotIn("48;2;79;152;163", train_line)

    def test_search_menu_renders_input_above_process_table(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py"),
        ]
        state = ProcessViewState(
            selected_pid=123,
            mode=MODE_SEARCH,
            search_input="train",
            status_message="No matches for: serve",
            viewport_rows=4,
        )
        console = Console(width=140, record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        plain = console.export_text(clear=False)
        title_line = next(line for line in plain.splitlines() if "Processes  1/1" in line)
        self.assertIn("Search: train", plain)
        self.assertIn("No matches for: serve", title_line)
        self.assertLess(plain.index("Search: train"), process_header_index(plain, plain.index("Search: train")))

    def test_filter_menu_renders_input_above_process_table(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py"),
        ]
        state = ProcessViewState(
            selected_pid=123,
            mode=MODE_FILTER,
            filter_input="train",
            filter_query="train",
            viewport_rows=4,
        )
        console = Console(width=140, record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        plain = console.export_text(clear=False)
        title_line = next(line for line in plain.splitlines() if "Processes  1/1" in line)
        self.assertIn("Filter: train", plain)
        self.assertNotIn("Filter: train", title_line)
        self.assertLess(plain.index("Filter: train"), process_header_index(plain, plain.index("Filter: train")))

    def test_active_filter_renders_caption_and_filters_rows(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py"),
            ProcessInfo(gpu_index=1, pid=456, user="demo", args="python serve.py"),
        ]
        state = ProcessViewState(selected_pid=456, filter_query="train", viewport_rows=4)
        console = Console(width=140, record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        plain = console.export_text(clear=False)
        title_line = next(line for line in plain.splitlines() if "Processes  1/1" in line)
        self.assertIn("Filter: train", title_line)
        self.assertIn("python train.py", plain)
        self.assertNotIn("python serve.py", plain)

    def test_active_gpu_focus_renders_caption_and_filters_rows(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py"),
            ProcessInfo(gpu_index=1, pid=456, user="demo", args="python serve.py"),
        ]
        state = ProcessViewState(selected_pid=456, gpu_filter_index=1, viewport_rows=4)
        console = Console(width=140, record=True, file=StringIO())
        console.print(render_process_table(processes, process_state=state, max_rows=4, terminal_width=140))
        plain = console.export_text(clear=False)
        title_line = next(line for line in plain.splitlines() if "Processes  1/1" in line)
        self.assertIn("Focus: GPU 1", title_line)
        self.assertIn("python serve.py", plain)
        self.assertNotIn("python train.py", plain)

    def test_process_zoom_renders_only_process_table_with_full_height(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            node_name="node-a",
            gpus=[
                GpuInfo(index=0, memory_total_bytes=4 * 1024 * 1024 * 1024, utilization_percent=42),
            ],
            processes=[
                ProcessInfo(gpu_index=index % 2, pid=100 + index, user="demo", args=f"python rank-{index}.py")
                for index in range(10)
            ],
            warnings=["demo warning"],
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
        state = ProcessViewState(selected_pid=104, process_zoomed=True, viewport_rows=8)
        console = Console(width=140, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_snapshot(snapshot, history, state, terminal_height=12, terminal_width=140))
        plain = console.export_text(clear=False)

        self.assertLessEqual(len(plain.splitlines()), 12)
        self.assertNotIn("roctop @ node-a", plain)
        self.assertNotIn("GUID", plain)
        self.assertNotIn("Avg %CPU", plain)
        self.assertNotIn("Warnings", plain)
        self.assertIn("Processes  ", plain)
        self.assertIn("python rank-4.p", plain)

    def test_process_zoom_keeps_inline_filter_menu_and_filters_rows(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[GpuInfo(index=0), GpuInfo(index=1)],
            processes=[
                ProcessInfo(gpu_index=0, pid=123, user="demo", args="python train.py"),
                ProcessInfo(gpu_index=1, pid=456, user="demo", args="python serve.py"),
            ],
        )
        state = ProcessViewState(
            selected_pid=123,
            process_zoomed=True,
            mode=MODE_FILTER,
            filter_input="train",
            filter_query="train",
            viewport_rows=4,
        )
        console = Console(width=140, record=True, file=StringIO())
        console.print(render_snapshot(snapshot, process_state=state, terminal_height=8, terminal_width=140))
        plain = console.export_text(clear=False)

        self.assertLessEqual(len(plain.splitlines()), 8)
        self.assertIn("Filter: train", plain)
        self.assertIn("python train.py", plain)
        self.assertNotIn("python serve.py", plain)
        self.assertNotIn("GUID", plain)

    def test_tree_mode_renders_connectors_and_selected_row(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=11, ppid=10, user="demo", args="python train.py"),
            ProcessInfo(gpu_index=1, pid=12, ppid=10, user="demo", args="python serve.py"),
        ]
        ancestors = [ProcessInfo(gpu_index=None, pid=10, user="demo", args="bash launcher")]
        state = ProcessViewState(selected_pid=12, tree_mode=True, viewport_rows=4)
        console = Console(width=140, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(
            render_process_table(
                processes,
                process_state=state,
                max_rows=4,
                terminal_width=140,
                process_ancestors=ancestors,
            )
        )

        plain = console.export_text(clear=False)
        styled = console.export_text(styles=True)
        self.assertIn("Process Tree  3/3", plain)
        self.assertIn("├─ python train", plain)
        self.assertIn("└─ python serve", plain)
        selected_lines = [line for line in styled.splitlines() if "48;2;68;71;90" in line]
        self.assertEqual(len(selected_lines), 1)
        self.assertIn("└─ python serve", selected_lines[0])

    def test_tree_gpu_focus_renders_parent_processes(self) -> None:
        processes = [
            ProcessInfo(gpu_index=7, pid=12, ppid=10, user="demo", args="gpu-7-worker"),
            ProcessInfo(gpu_index=6, pid=13, ppid=10, user="demo", args="gpu-6-worker"),
        ]
        ancestors = [
            ProcessInfo(gpu_index=None, pid=1, user="root", args="init"),
            ProcessInfo(gpu_index=None, pid=10, ppid=1, user="demo", args="launcher"),
        ]
        state = ProcessViewState(selected_pid=12, tree_mode=True, gpu_filter_index=7, viewport_rows=4)
        console = Console(width=140, record=True, file=StringIO())
        console.print(
            render_process_table(
                processes,
                process_state=state,
                max_rows=4,
                terminal_width=140,
                process_ancestors=ancestors,
            )
        )

        plain = console.export_text(clear=False)
        self.assertIn("Process Tree  3/3", plain)
        self.assertIn("Focus: GPU 7", plain)
        self.assertIn("init", plain)
        self.assertIn("launcher", plain)
        self.assertIn("gpu-7-worker", plain)
        self.assertNotIn("gpu-6-worker", plain)

    def test_tree_mode_wraps_continuation_lines_under_prefix(self) -> None:
        root = ProcessInfo(gpu_index=None, pid=1, args="init")
        parent = ProcessInfo(gpu_index=None, pid=10, ppid=1, args="launcher")
        child = ProcessInfo(
            gpu_index=0,
            pid=11,
            ppid=10,
            args=" ".join(f"--very-long-option-{index}=demo-value" for index in range(10)),
        )
        sibling = ProcessInfo(gpu_index=None, pid=20, ppid=1, args="next-service")
        processes = [root, parent, child, sibling]
        state = ProcessViewState(selected_pid=11, tree_mode=True, viewport_rows=3)
        state.sync(processes, viewport_rows=3)

        rows = render.visible_process_window(
            processes,
            state,
            max_visual_rows=3,
            command_width=24,
            tree_prefixes=render.process_tree_prefixes(processes),
        )

        child_row = rows[-1]
        lines = child_row.command.splitlines()
        self.assertTrue(lines[0].startswith("│  └─ "))
        self.assertTrue(lines[1].startswith("│     "))
        self.assertLessEqual(child_row.visual_height, 3)

    def test_sort_menu_stays_above_process_table_when_cropped(self) -> None:
        long_args = (
            "demo_server --model-path /demo/models/huggingface "
            "--served-model-name demo-model --host 0.0.0.0 --port 30000 --tp 1 --context-length 8192"
        )
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[
                GpuInfo(
                    index=index,
                    memory_used_bytes=1024 * 1024 * 1024,
                    memory_total_bytes=4 * 1024 * 1024 * 1024,
                    utilization_percent=42,
                )
                for index in range(8)
            ],
            processes=[
                ProcessInfo(
                    gpu_index=index % 8 if index < 8 else None,
                    pid=3000 + index,
                    user="demo",
                    cpu_percent=12.0,
                    host_mem_percent=0.1,
                    elapsed="01:02",
                    args=long_args,
                )
                for index in range(20)
            ],
        )
        history = MetricsHistory(max_samples=120)
        history.append_sample(
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=6.8,
                avg_mem_percent=8.8,
                avg_gpu_percent=3.8,
                avg_gpu_mem_percent=29.8,
            )
        )
        state = ProcessViewState(selected_pid=3019, mode="sort_menu", sort_menu_index=4, viewport_rows=20)
        console = Console(width=180, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_snapshot(snapshot, history, state, terminal_height=45, terminal_width=180))
        plain = console.export_text(clear=False)
        self.assertLessEqual(len(plain.splitlines()), 45)
        self.assertIn("Sort by:", plain)
        self.assertLess(plain.index("Sort by:"), process_header_index(plain, plain.index("Sort by:")))
        self.assertIn("3019", plain)

    def test_filter_menu_stays_above_process_table_when_cropped(self) -> None:
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[
                GpuInfo(
                    index=index,
                    memory_used_bytes=1024 * 1024 * 1024,
                    memory_total_bytes=4 * 1024 * 1024 * 1024,
                    utilization_percent=42,
                )
                for index in range(8)
            ],
            processes=synthetic_long_processes(20),
        )
        history = MetricsHistory(max_samples=120)
        history.append_sample(
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=6.8,
                avg_mem_percent=8.8,
                avg_gpu_percent=3.8,
                avg_gpu_mem_percent=29.8,
            )
        )
        state = ProcessViewState(
            selected_pid=1019,
            mode=MODE_FILTER,
            filter_input="rank",
            filter_query="rank",
            viewport_rows=20,
        )
        console = Console(width=180, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_snapshot(snapshot, history, state, terminal_height=45, terminal_width=180))
        plain = console.export_text(clear=False)
        self.assertLessEqual(len(plain.splitlines()), 45)
        self.assertIn("Filter: rank", plain)
        self.assertLess(plain.index("Filter: rank"), process_header_index(plain, plain.index("Filter: rank")))
        self.assertIn("1019", plain)

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

    def test_static_snapshot_with_terminal_height_renders_under_100ms(self) -> None:
        process_count = 10000
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=synthetic_long_processes(process_count),
        )
        console = Console(width=180, record=True, file=StringIO())

        start = time.perf_counter()
        console.print(render_snapshot(snapshot, terminal_height=40, terminal_width=180))
        elapsed = time.perf_counter() - start

        output = console.export_text()
        self.assertLess(elapsed, 0.1)
        self.assertIn(f"/{process_count}", output)
        self.assertNotIn("--rank 9999", output)

    def test_snapshot_keeps_selected_process_visible_with_graphs(self) -> None:
        long_args = (
            "demo_server --model-path "
            "/demo/models/glm-fp8/snapshots/demo-checkpoint-0001 "
            "--served-model-name demo/GLM-FP8 --host 0.0.0.0 --port 30052 "
            "--tensor-parallel-size 4 --trust-remote-code --context-length 8192 "
            "--kv-cache-dtype bfloat16 --dsa-prefill-backend aiter --dsa-decode-backend aiter"
        )
        processes = [
            ProcessInfo(
                gpu_index=index % 8 if index < 5 else None,
                pid=2000 + index,
                user="demo",
                cpu_percent=90.0 if index < 5 else 0.4,
                host_mem_percent=0.1,
                gpu_memory_percent=92.5 if index < 5 else 0.0,
                gpu_memory_bytes=1024 * 1024 * 1024 if index < 5 else 0,
                elapsed="01:02",
                args="demo::scheduler" if index < 5 else "demo_server --model-path /demo/models/huggingface",
            )
            for index in range(12)
        ]
        processes.append(
            ProcessInfo(
                gpu_index=None,
                pid=9999,
                user="demo",
                cpu_percent=52.8,
                host_mem_percent=0.0,
                elapsed="00:45",
                args=long_args,
            )
        )
        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            gpus=[
                GpuInfo(
                    index=index,
                    memory_used_bytes=1024 * 1024 * 1024,
                    memory_total_bytes=4 * 1024 * 1024 * 1024,
                    utilization_percent=42,
                )
                for index in range(8)
            ],
            processes=processes,
        )
        history = MetricsHistory(max_samples=120)
        history.append_sample(
            MetricSample(
                timestamp=datetime(2026, 6, 22, 12, 0, 0),
                avg_cpu_percent=6.8,
                avg_mem_percent=8.8,
                avg_gpu_percent=3.8,
                avg_gpu_mem_percent=29.8,
            )
        )
        state = ProcessViewState(selected_pid=9999, viewport_rows=13)
        console = Console(width=240, force_terminal=True, color_system="truecolor", record=True, file=StringIO())
        console.print(render_snapshot(snapshot, history, state, terminal_height=45, terminal_width=240))
        plain = console.export_text(clear=False)
        styled = console.export_text(styles=True)
        self.assertLessEqual(len(plain.splitlines()), 45)
        self.assertIn("9999", plain)
        self.assertIn("48;2;68;71;90", styled)

    def test_process_view_wraps_long_commands_within_visual_height(self) -> None:
        long_args = (
            "demo_trainer direct --config /tmp/demo_qwen_config.yaml "
            "--data-path /demo/data --trainer-path /demo/tools/trainer --nnodes 2 "
            "--nproc-per-node 2 --rdzv-backend c10d --master-addr "
            "trainer-replica-0.demo-job.local"
        )
        processes = [
            ProcessInfo(gpu_index=index % 8, pid=200 + index, user="demo", args=long_args)
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

    def test_process_window_truncates_next_row_to_fill_visual_height(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=500, user="demo", args="short-one"),
            ProcessInfo(gpu_index=1, pid=501, user="demo", args="short-two"),
            ProcessInfo(
                gpu_index=2,
                pid=502,
                user="demo",
                args=" ".join(f"--very-long-option-{index}=demo-value" for index in range(20)),
            ),
            ProcessInfo(gpu_index=3, pid=503, user="demo", args="still-running"),
        ]
        state = ProcessViewState(selected_pid=500, viewport_rows=4)
        state.sync(processes, viewport_rows=4)

        rows = render.visible_process_window(processes, state, max_visual_rows=4, command_width=24)

        self.assertEqual([row.process.pid for row in rows], [500, 501, 502])
        self.assertEqual(sum(row.visual_height for row in rows), 4)
        self.assertEqual(rows[-1].visual_height, 2)
        self.assertTrue(rows[-1].command.endswith("..."))
        self.assertNotIn("still-running", "\n".join(row.command for row in rows))

    def test_selected_process_command_truncates_to_visual_budget(self) -> None:
        process = ProcessInfo(
            gpu_index=0,
            pid=404,
            user="demo",
            args=" ".join(f"--very-long-option-{index}=demo-value" for index in range(30)),
        )
        state = ProcessViewState(selected_pid=404, viewport_rows=3)
        state.sync([process], viewport_rows=3)

        rows = render.visible_process_window([process], state, max_visual_rows=3, command_width=24)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].visual_height, 3)
        self.assertEqual(rows[0].command.count("\n"), 2)
        self.assertTrue(rows[0].command.endswith("..."))

    def test_static_process_command_truncates_when_row_limited(self) -> None:
        process = ProcessInfo(
            gpu_index=0,
            pid=405,
            user="demo",
            args=" ".join(f"--very-long-option-{index}=demo-value" for index in range(30)),
        )
        console = Console(width=120, record=True, file=StringIO())
        console.print(render_process_table([process], max_rows=1, terminal_width=120))

        output = console.export_text()
        self.assertIn("...", output)
        self.assertNotIn("--very-long-option-29", output)

    def test_process_window_keeps_selected_row_visible_near_top_middle_and_bottom(self) -> None:
        processes = synthetic_long_processes(300)
        command_width = render.estimate_process_command_width(120)

        for selected_index in (0, 150, 299):
            with self.subTest(selected_index=selected_index):
                state = ProcessViewState(selected_pid=processes[selected_index].pid, viewport_rows=8)
                state.sync(processes, viewport_rows=8)

                rows = render.visible_process_window(processes, state, max_visual_rows=8, command_width=command_width)

                visible_pids = [row.process.pid for row in rows]
                self.assertIn(processes[selected_index].pid, visible_pids)
                self.assertLessEqual(sum(row.visual_height for row in rows), 8)
                if selected_index == 0:
                    self.assertEqual(visible_pids[0], processes[0].pid)
                if selected_index == len(processes) - 1:
                    self.assertEqual(visible_pids[-1], processes[-1].pid)

    def test_process_render_wraps_visible_window_not_every_process(self) -> None:
        processes = synthetic_long_processes(400)
        state = ProcessViewState(selected_pid=processes[200].pid, viewport_rows=10)
        original_wrap_command_lines = render.wrap_command_lines
        wrap_calls = 0

        def counting_wrap_command_lines(command: str, width: int) -> list[str]:
            nonlocal wrap_calls
            wrap_calls += 1
            return original_wrap_command_lines(command, width)

        try:
            render.wrap_command_lines = counting_wrap_command_lines
            console = Console(width=120, record=True, file=StringIO())
            console.print(
                render_process_table(
                    processes,
                    process_state=state,
                    max_rows=10,
                    terminal_width=120,
                )
            )
        finally:
            render.wrap_command_lines = original_wrap_command_lines

        output = console.export_text()
        self.assertIn(str(processes[200].pid), output)
        self.assertLessEqual(wrap_calls, 10)
        self.assertLess(wrap_calls, len(processes) // 10)


if __name__ == "__main__":
    unittest.main()

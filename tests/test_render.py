from __future__ import annotations

import time
import unittest
from io import StringIO
from datetime import datetime

from rich.console import Console

import roctop.render as render
from roctop.history import MetricSample, MetricsHistory
from roctop.interaction import MODE_FILTER, MODE_KILL_CONFIRM, MODE_SEARCH, ProcessViewState
from roctop.models import GpuInfo, ProcessInfo, Snapshot
from roctop.render import (
    bar_with_percent,
    metric_graph_lines,
    percent_style,
    render_metrics_graphs,
    render_process_table,
    render_snapshot,
)


def has_braille_dots(text: str) -> bool:
    return any("\u2801" <= char <= "\u28ff" for char in text)


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
        self.assertIn("42%", output)
        self.assertNotIn("UTL", output)
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

        console = Console(width=180, record=True, file=StringIO())
        console.print(render_snapshot(snapshot, history))
        output = console.export_text()
        self.assertIn("Avg %CPU: 37.3%", output)
        self.assertIn("Avg %GPU: 33.2%", output)
        self.assertIn("Avg %MEM: 77.2%", output)
        self.assertIn("Avg %GPU MEM: 54.6%", output)
        self.assertIn("120s", output)
        self.assertIn("60s", output)
        self.assertIn("30s", output)
        self.assertLess(output.index("Avg %CPU: 37.3%"), output.index("120s"))
        self.assertLess(output.index("120s"), output.index("Avg %MEM: 77.2%"))
        self.assertLess(output.index("Avg %GPU: 33.2%"), output.index("Avg %GPU MEM: 54.6%"))
        self.assertLess(output.index("%Utilization"), output.index("Avg %CPU"))
        self.assertLess(output.index("Avg %CPU"), output.index("PID"))

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
        self.assertEqual(lines[-1].plain, "    ⠤⠶")

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
        self.assertEqual(normal[0].plain, "⣀")
        self.assertEqual(inverted[0].plain, "⠉")

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
        console = Console(width=180, record=True, file=StringIO())
        console.print(render_metrics_graphs(history))
        lines = console.export_text().splitlines()
        axis_lines = [index for index, line in enumerate(lines) if "120s" in line]
        bottom_label_lines = [
            index for index, line in enumerate(lines) if "Avg %MEM:" in line and "Avg %GPU MEM:" in line
        ]
        self.assertEqual(len(axis_lines), 1)
        self.assertEqual(len(bottom_label_lines), 1)
        self.assertLess(axis_lines[0], bottom_label_lines[0])

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
        axis_index = next(index for index, line in enumerate(lines) if "120s" in line)
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

    def test_process_help_renders_action_keys_without_navigation_hints(self) -> None:
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
        self.assertNotIn("Processes:", plain)
        self.assertNotIn("Sort:", plain)
        self.assertIn("Processes  2/2", plain)
        self.assertNotIn("sort: default", plain)
        self.assertNotIn("j/k: move", plain)
        self.assertNotIn("PgUp/PgDn: scroll", plain)
        self.assertNotIn("n/N: next/prev", plain)
        self.assertIn("s: sort", plain)
        self.assertIn("/: search", plain)
        self.assertIn("f: filter", plain)
        self.assertIn("x: kill", plain)
        self.assertIn("q: quit", plain)
        self.assertLess(plain.index("Mon Jun 22"), plain.index("s: sort"))
        self.assertIn("38;2;255;184;108", styled)

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
        self.assertLess(plain.index("Sort by:"), plain.index("│ GPU", plain.index("Sort by:")))
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
        self.assertLess(plain.index("Kill PID 123:"), plain.index("│ GPU", plain.index("Kill PID 123:")))
        self.assertIn("38;2;40;42;54", styled)
        self.assertIn("48;2;139;233;253", styled)

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
        self.assertLess(plain.index("Search: train"), plain.index("│ GPU", plain.index("Search: train")))

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
        self.assertLess(plain.index("Filter: train"), plain.index("│ GPU", plain.index("Filter: train")))

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
        self.assertLess(plain.index("Sort by:"), plain.index("│ GPU", plain.index("Sort by:")))
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
        self.assertLess(plain.index("Filter: rank"), plain.index("│ GPU", plain.index("Filter: rank")))
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
        self.assertLessEqual(wrap_calls, 8)
        self.assertLess(wrap_calls, len(processes) // 10)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import roctop.debug_counters as debug_counters
from roctop.collectors import CommandResult
from roctop.debug_counters import (
    ResolvedDebugCounters,
    build_rocprofv3_command,
    clear_rocprof_attach_config,
    command_error_text,
    collect_gpu_debug_counters,
    parse_counter_definition_names,
    parse_counter_collection_csv,
    resolve_debug_counters,
    target_attach_library_error,
    target_visible_path,
    sample_process_debug_counters,
    top_gpu_debug_processes,
)
from roctop.models import ProcessInfo, Snapshot


class DebugCounterTests(unittest.TestCase):
    def test_parse_long_counter_csv_aggregates_by_dispatch_and_kernel(self) -> None:
        resolved = ResolvedDebugCounters(
            counters=("SQ_WAVES", "SQ_INSTS", "TCC_HIT_sum", "TCC_MISS_sum", "FETCH_SIZE", "WRITE_SIZE"),
            wave_counter="SQ_WAVES",
            instruction_counters=("SQ_INSTS",),
            cache_hit_counter="TCC_HIT_sum",
            cache_miss_counter="TCC_MISS_sum",
            fetch_counter="FETCH_SIZE",
            write_counter="WRITE_SIZE",
        )
        csv_text = "\n".join(
            [
                "Dispatch_Id,Kernel_Name,Duration_ns,Counter_Name,Counter_Value",
                "1,kernel_a,100,SQ_WAVES,4",
                "1,kernel_a,100,SQ_INSTS,40",
                "1,kernel_a,100,TCC_HIT_sum,90",
                "1,kernel_a,100,TCC_MISS_sum,10",
                "1,kernel_a,100,FETCH_SIZE,1024",
                "1,kernel_a,100,WRITE_SIZE,512",
                "2,kernel_b,200,SQ_WAVES,6",
                "2,kernel_b,200,SQ_INSTS,60",
                "2,kernel_b,200,TCC_HIT_sum,80",
                "2,kernel_b,200,TCC_MISS_sum,20",
                "2,kernel_b,200,FETCH_SIZE,2048",
                "2,kernel_b,200,WRITE_SIZE,1024",
            ]
        )

        sample = parse_counter_collection_csv(
            csv_text,
            ProcessInfo(gpu_index=0, pid=42, args="python train.py"),
            gpu_index=0,
            resolved=resolved,
            sample_seconds=1.0,
        )

        self.assertEqual(sample.pid, 42)
        self.assertEqual(sample.dispatches, 2)
        self.assertEqual(sample.waves, 10)
        self.assertEqual(sample.instructions, 100)
        self.assertAlmostEqual(sample.l2_hit_percent, 85.0)
        self.assertEqual(sample.read_bytes, 3072 * 1024)
        self.assertEqual(sample.write_bytes, 1536 * 1024)
        self.assertEqual([kernel.name for kernel in sample.kernels], ["kernel_b", "kernel_a"])

    def test_parse_wide_counter_csv_reads_derived_l2_hit(self) -> None:
        resolved = ResolvedDebugCounters(
            counters=("SQ_WAVES", "SQ_INSTS", "L2CacheHit", "FETCH_SIZE", "WRITE_SIZE"),
            wave_counter="SQ_WAVES",
            instruction_counters=("SQ_INSTS",),
            l2_hit_percent_counter="L2CacheHit",
            fetch_counter="FETCH_SIZE",
            write_counter="WRITE_SIZE",
        )
        csv_text = "\n".join(
            [
                "Dispatch_Id,Kernel_Name,Duration_ns,SQ_WAVES,SQ_INSTS,L2CacheHit,FETCH_SIZE,WRITE_SIZE",
                "1,kernel_a,100,4,40,75,1024,512",
                "2,kernel_a,200,6,60,85,2048,1024",
            ]
        )

        sample = parse_counter_collection_csv(
            csv_text,
            ProcessInfo(gpu_index=0, pid=42, args="python train.py"),
            gpu_index=0,
            resolved=resolved,
            sample_seconds=2.0,
        )

        self.assertEqual(sample.dispatches, 2)
        self.assertEqual(sample.waves, 10)
        self.assertEqual(sample.instructions, 100)
        self.assertAlmostEqual(sample.l2_hit_percent, 80.0)
        self.assertEqual(len(sample.kernels), 1)

    def test_build_rocprofv3_command_uses_argument_list_and_output_dir(self) -> None:
        command = build_rocprofv3_command(
            "/usr/bin/rocprofv3",
            123,
            ("SQ_WAVES", "SQ_INSTS"),
            "/tmp/roctop-debug-test",
            sample_msec=1000,
        )

        self.assertIsInstance(command, list)
        self.assertEqual(command[0], "/usr/bin/rocprofv3")
        self.assertIn("--attach", command)
        self.assertIn("--output-directory", command)
        self.assertIn("/tmp/roctop-debug-test", command)
        self.assertNotIn(" ".join(command), command)

    def test_sample_process_writes_to_temp_output_and_cleans_up(self) -> None:
        output_dirs: list[Path] = []
        resolved = ResolvedDebugCounters(
            counters=("SQ_WAVES", "SQ_INSTS"),
            wave_counter="SQ_WAVES",
            instruction_counters=("SQ_INSTS",),
        )

        def fake_run(args: list[str], timeout: float) -> CommandResult:
            output_dir = Path(args[args.index("--output-directory") + 1])
            output_dirs.append(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir.joinpath("counter_collection.csv").write_text(
                "\n".join(
                    [
                        "Dispatch_Id,Kernel_Name,SQ_WAVES,SQ_INSTS",
                        "1,kernel_a,4,40",
                    ]
                ),
                encoding="utf-8",
            )
            return CommandResult(args=args, returncode=0, stdout="", stderr="")

        sample = sample_process_debug_counters(
            ProcessInfo(gpu_index=0, pid=42, args="python train.py"),
            gpu_index=0,
            resolved=resolved,
            rocprofv3_path="/usr/bin/rocprofv3",
            run_command_func=fake_run,
        )

        self.assertEqual(sample.dispatches, 1)
        self.assertEqual(sample.waves, 4)
        self.assertEqual(sample.instructions, 40)
        self.assertEqual(len(output_dirs), 1)
        self.assertFalse(output_dirs[0].exists())

    def test_sample_process_reads_modified_rocprof_output_directory(self) -> None:
        resolved = ResolvedDebugCounters(
            counters=("SQ_WAVES", "SQ_INSTS"),
            wave_counter="SQ_WAVES",
            instruction_counters=("SQ_INSTS",),
        )

        def fake_run(args: list[str], timeout: float) -> CommandResult:
            requested_dir = Path(args[args.index("--output-directory") + 1])
            modified_dir = requested_dir.with_name(f"{requested_dir.name}_1")
            modified_dir.mkdir(parents=True, exist_ok=True)
            modified_dir.joinpath("counter_collection.csv").write_text(
                "\n".join(
                    [
                        "Dispatch_Id,Kernel_Name,SQ_WAVES,SQ_INSTS",
                        "1,kernel_a,4,40",
                    ]
                ),
                encoding="utf-8",
            )
            return CommandResult(
                args=args,
                returncode=1,
                stdout="",
                stderr=(
                    "Warning: Option 'output_directory' has been modified.\n"
                    f"output_directory={modified_dir}\n"
                    f"(previously output_directory={requested_dir})"
                ),
            )

        sample = sample_process_debug_counters(
            ProcessInfo(gpu_index=0, pid=42, args="python train.py"),
            gpu_index=0,
            resolved=resolved,
            rocprofv3_path="/usr/bin/rocprofv3",
            run_command_func=fake_run,
        )

        self.assertEqual(sample.status, "ok")
        self.assertEqual(sample.dispatches, 1)
        self.assertEqual(sample.waves, 4)
        self.assertEqual(sample.instructions, 40)

    def test_clear_rocprof_attach_config_removes_stale_pid_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rocprofv3_attach_42.pkl"
            path.write_text("stale", encoding="utf-8")

            error = clear_rocprof_attach_config(42, Path(directory))

            self.assertEqual(error, "")
            self.assertFalse(path.exists())

    def test_target_visible_path_maps_absolute_path_through_proc_root(self) -> None:
        path = target_visible_path(42, Path("/opt/rocm/lib/libx.so"), Path("/fake/proc"))

        self.assertEqual(path, Path("/fake/proc/42/root/opt/rocm/lib/libx.so"))

    def test_target_attach_library_error_detects_missing_container_library(self) -> None:
        with TemporaryDirectory() as directory:
            proc_root = Path(directory)
            target_root = proc_root / "42/root"
            target_root.mkdir(parents=True)

            error = target_attach_library_error(
                42,
                "/host/rocm/bin/rocprofv3",
                proc_root=proc_root,
            )

            self.assertIn("Target process cannot see ROCm attach library", error)

    def test_target_attach_library_error_allows_visible_library(self) -> None:
        with TemporaryDirectory() as directory:
            proc_root = Path(directory)
            library = proc_root / "42/root/host/rocm/lib/rocprofiler-sdk/librocprofv3-attach.so"
            library.parent.mkdir(parents=True)
            library.write_text("", encoding="utf-8")

            error = target_attach_library_error(
                42,
                "/host/rocm/bin/rocprofv3",
                proc_root=proc_root,
            )

        self.assertEqual(error, "")

    def test_resolve_debug_counters_discovers_and_validates_candidates(self) -> None:
        calls: list[list[str]] = []

        def fake_run(args: list[str], timeout: float) -> CommandResult:
            calls.append(args)
            if "list" in args:
                return CommandResult(
                    args=args,
                    returncode=0,
                    stdout="SQ_WAVES SQ_INSTS L2CacheHit FETCH_SIZE WRITE_SIZE",
                    stderr="",
                )
            return CommandResult(args=args, returncode=0, stdout="", stderr="")

        resolved = resolve_debug_counters(0, "/usr/bin/rocprofv3-avail", fake_run)

        self.assertEqual(
            resolved.counters,
            ("SQ_WAVES", "SQ_INSTS", "L2CacheHit", "FETCH_SIZE", "WRITE_SIZE"),
        )
        self.assertIn("pmc-check", calls[-1])

    def test_counter_definition_fallback_reads_rocm_counter_names(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "counter_defs.yaml"
            path.write_text(
                "\n".join(
                    [
                        "- name: SQ_WAVES",
                        "- name: SQ_INSTS",
                        "- name: L2CacheHit",
                        "- name: FETCH_SIZE",
                        "- name: WRITE_SIZE",
                    ]
                ),
                encoding="utf-8",
            )

            names = parse_counter_definition_names((path,))

        self.assertIn("SQ_WAVES", names)
        self.assertIn("SQ_INSTS", names)
        self.assertIn("L2CacheHit", names)

    def test_resolve_debug_counters_falls_back_to_counter_definitions_when_list_is_empty(self) -> None:
        calls: list[list[str]] = []
        original_counter_definition_paths = debug_counters.counter_definition_paths
        with TemporaryDirectory() as directory:
            path = Path(directory) / "counter_defs.yaml"
            path.write_text(
                "\n".join(
                    [
                        "- name: SQ_WAVES",
                        "- name: SQ_INSTS",
                        "- name: L2CacheHit",
                        "- name: FETCH_SIZE",
                        "- name: WRITE_SIZE",
                    ]
                ),
                encoding="utf-8",
            )
            debug_counters.counter_definition_paths = lambda: (path,)

            def fake_run(args: list[str], timeout: float) -> CommandResult:
                calls.append(args)
                if "list" in args:
                    return CommandResult(args=args, returncode=1, stdout="GPU:0\nNAME:\nPMC:\n", stderr="")
                return CommandResult(args=args, returncode=0, stdout="", stderr="")

            try:
                resolved = resolve_debug_counters(0, "/usr/bin/rocprofv3-avail", fake_run)
            finally:
                debug_counters.counter_definition_paths = original_counter_definition_paths

        self.assertEqual(
            resolved.counters,
            ("SQ_WAVES", "SQ_INSTS", "L2CacheHit", "FETCH_SIZE", "WRITE_SIZE"),
        )
        self.assertGreater(sum(1 for call in calls if "pmc-check" in call), 1)

    def test_collect_gpu_debug_counters_lists_processes_when_counters_are_unavailable(self) -> None:
        def fake_run(args: list[str], timeout: float) -> CommandResult:
            if "list" in args:
                return CommandResult(args=args, returncode=0, stdout="SQ_WAVES SQ_INSTS", stderr="")
            return CommandResult(args=args, returncode=1, stdout="", stderr="Fatal error: Invalid counter name")

        snapshot = Snapshot(
            timestamp=datetime(2026, 6, 22, 12, 0, 0),
            processes=[
                ProcessInfo(gpu_index=0, pid=1, args="python low.py", gpu_memory_bytes=100),
                ProcessInfo(gpu_index=0, pid=2, args="python high.py", gpu_memory_bytes=300),
            ],
        )

        sample = collect_gpu_debug_counters(
            snapshot,
            0,
            run_command_func=fake_run,
            rocprofv3_path="/usr/bin/rocprofv3",
            rocprofv3_avail_path="/usr/bin/rocprofv3-avail",
        )

        self.assertEqual([process.pid for process in sample.processes], [2, 1])
        self.assertIn("Counter validation failed", sample.status)
        self.assertIn("Counter validation failed", sample.processes[0].status)

    def test_top_gpu_debug_processes_uses_gpu_memory_descending(self) -> None:
        processes = [
            ProcessInfo(gpu_index=0, pid=1, gpu_memory_bytes=100),
            ProcessInfo(gpu_index=1, pid=2, gpu_memory_bytes=900),
            ProcessInfo(gpu_index=0, pid=3, gpu_memory_bytes=300),
            ProcessInfo(gpu_index=0, pid=4, gpu_memory_bytes=200),
        ]

        self.assertEqual([proc.pid for proc in top_gpu_debug_processes(processes, 0, limit=2)], [3, 4])

    def test_command_error_text_summarizes_unsupported_architecture(self) -> None:
        result = CommandResult(
            args=["rocprofv3-avail"],
            returncode=1,
            stdout="",
            stderr="rocprofiler_iterate_agent_supported_counters failed :: Agent HW architecture is not supported",
        )

        self.assertEqual(
            command_error_text(result),
            "ROCm profiler reports unsupported hardware counters for this GPU architecture",
        )


if __name__ == "__main__":
    unittest.main()

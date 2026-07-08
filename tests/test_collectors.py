from __future__ import annotations

import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roctop import collectors
from roctop.collectors import (
    CommandInterrupted,
    CommandResult,
    CommandTimeout,
    CollectionError,
    collect_process_ancestors,
    collect_snapshot,
    load_json_from_text,
    merge_process_sources,
    parse_memory_bytes_field,
    parse_amd_pci_models,
    parse_amd_smi_gpu_json,
    parse_amd_smi_process_json,
    parse_rocm_smi_json,
    read_ps_rows,
    run_command,
)
from roctop.formatting import format_bytes_mib
from roctop.models import GpuInfo, ProcessInfo


class CollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        collectors._amd_smi_process_backoff_until = 0.0
        collectors._amd_smi_gpu_detail_backoff_until = 0.0
        collectors._ps_row_cache.clear()

    def test_load_json_from_text_skips_warning_prefix(self) -> None:
        data = load_json_from_text('WARNING: noisy\n{"ok": true}')
        self.assertEqual(data, {"ok": True})

    def test_run_command_raises_command_timeout(self) -> None:
        import subprocess
        from unittest.mock import patch

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["rocm-smi"], 1)):
            with self.assertRaises(CommandTimeout):
                run_command(["rocm-smi"], timeout=1)

    def test_run_command_wraps_os_error(self) -> None:
        with patch("subprocess.run", side_effect=PermissionError("permission denied")):
            with self.assertRaises(CollectionError):
                run_command(["rocm-smi"], timeout=1)

    def test_collect_snapshot_wraps_invalid_rocm_json(self) -> None:
        def fake_run_command(args, **kwargs) -> CommandResult:
            if args[0] == "rocm-smi":
                return CommandResult(args=args, returncode=0, stdout="not json", stderr="")
            return CommandResult(args=args, returncode=0, stdout="[]", stderr="")

        with patch("roctop.collectors.run_command", side_effect=fake_run_command):
            with self.assertRaises(CollectionError):
                collect_snapshot()

    def test_collect_snapshot_rejects_non_object_rocm_json(self) -> None:
        def fake_run_command(args, **kwargs) -> CommandResult:
            if args[0] == "rocm-smi":
                return CommandResult(args=args, returncode=0, stdout="[]", stderr="")
            return CommandResult(args=args, returncode=0, stdout="[]", stderr="")

        with patch("roctop.collectors.run_command", side_effect=fake_run_command):
            with self.assertRaises(CollectionError):
                collect_snapshot()

    def test_collect_snapshot_raises_command_interrupted(self) -> None:
        original_run_command = collectors.run_command

        def fake_run_command(*args, **kwargs) -> CommandResult:
            return CommandResult(args=["rocm-smi"], returncode=-11, stdout="", stderr="")

        try:
            collectors.run_command = fake_run_command
            with self.assertRaises(CommandInterrupted):
                collect_snapshot()
        finally:
            collectors.run_command = original_run_command

    def test_collect_snapshot_records_node_name(self) -> None:
        def fake_run_command(args, **kwargs) -> CommandResult:
            if args[0] == "rocm-smi":
                return CommandResult(
                    args=args,
                    returncode=0,
                    stdout='{"system": {"Driver version": "6.14.14"}}',
                    stderr="",
                )
            return CommandResult(args=args, returncode=0, stdout="[]", stderr="")

        with (
            patch("roctop.collectors.run_command", side_effect=fake_run_command),
            patch("roctop.collectors.platform.node", return_value="node-a"),
        ):
            snapshot = collect_snapshot()

        self.assertEqual(snapshot.node_name, "node-a")

    def test_collect_snapshot_runs_smi_commands_in_parallel(self) -> None:
        rocm_started = threading.Event()
        amd_started = threading.Event()

        def fake_run_command(args, **kwargs) -> CommandResult:
            if args[0] == "rocm-smi":
                rocm_started.set()
                self.assertTrue(amd_started.wait(timeout=1.0))
                return CommandResult(
                    args=args,
                    returncode=0,
                    stdout='{"system": {"Driver version": "6.14.14"}}',
                    stderr="",
                )
            if args[0] == "amd-smi":
                amd_started.set()
                self.assertTrue(rocm_started.wait(timeout=1.0))
                return CommandResult(args=args, returncode=0, stdout="[]", stderr="")
            return CommandResult(args=args, returncode=1, stdout="", stderr="")

        with patch("roctop.collectors.run_command", side_effect=fake_run_command):
            snapshot = collect_snapshot()

        self.assertEqual(snapshot.driver_version, "6.14.14")
        self.assertTrue(rocm_started.is_set())
        self.assertTrue(amd_started.is_set())

    def test_amd_smi_process_timeout_falls_back_to_rocm_smi_process_rows(self) -> None:
        calls: list[tuple[list[str], float | None]] = []

        def fake_run_command(args, timeout=None) -> CommandResult:
            calls.append((args, timeout))
            if args[0] == "rocm-smi":
                return CommandResult(
                    args=args,
                    returncode=0,
                    stdout=(
                        '{"card0": {"VRAM Total Memory (B)": "4194304"}, '
                        '"system": {"PID42": "demo-worker, 0, 1048576, 0, 0"}}'
                    ),
                    stderr="",
                )
            if args == collectors.AMD_SMI_PROCESS_ARGS:
                raise CommandTimeout("Command timed out: amd-smi process")
            if args[0] == "amd-smi":
                return CommandResult(args=args, returncode=0, stdout="[]", stderr="")
            return CommandResult(args=args, returncode=1, stdout="", stderr="")

        with patch("roctop.collectors.run_command", side_effect=fake_run_command):
            snapshot = collect_snapshot()

        self.assertEqual([proc.pid for proc in snapshot.processes], [42])
        self.assertEqual(snapshot.processes[0].command, "demo-worker")
        self.assertEqual(snapshot.processes[0].gpu_index, 0)
        self.assertEqual(snapshot.processes[0].gpu_memory_percent, 25.0)
        amd_calls = [(args, timeout) for args, timeout in calls if args == collectors.AMD_SMI_PROCESS_ARGS]
        self.assertEqual(len(amd_calls), 1)
        self.assertEqual(amd_calls[0][1], collectors.AMD_SMI_PROCESS_TIMEOUT_SECONDS)

    def test_amd_smi_process_backoff_skips_optional_command_during_cooldown(self) -> None:
        amd_calls = 0

        def fake_run_command(args, timeout=None) -> CommandResult:
            nonlocal amd_calls
            if args[0] == "rocm-smi":
                return CommandResult(
                    args=args,
                    returncode=0,
                    stdout='{"system": {"PID42": "demo-worker, 0, 1048576, 0, 0"}}',
                    stderr="",
                )
            if args == collectors.AMD_SMI_PROCESS_ARGS:
                amd_calls += 1
                raise CommandTimeout("Command timed out: amd-smi process")
            if args[0] == "amd-smi":
                return CommandResult(args=args, returncode=0, stdout="[]", stderr="")
            return CommandResult(args=args, returncode=1, stdout="", stderr="")

        with patch("roctop.collectors.run_command", side_effect=fake_run_command):
            first = collect_snapshot()
            second = collect_snapshot()

        self.assertEqual([proc.pid for proc in first.processes], [42])
        self.assertEqual([proc.pid for proc in second.processes], [42])
        self.assertEqual(amd_calls, 1)

    def test_ps_enrichment_cache_reuses_rows_within_ttl_and_refreshes_after_expiry(self) -> None:
        calls: list[list[int]] = []

        def fake_read_ps_rows(pids: list[int]) -> dict[int, dict[str, str]]:
            calls.append(list(pids))
            suffix = str(len(calls))
            return {
                pid: {
                    "user": f"demo{suffix}",
                    "cpu": suffix,
                    "mem": suffix,
                    "etime": "00:01",
                    "comm": "python",
                    "args": f"python worker-{suffix}.py",
                }
                for pid in pids
            }

        with (
            patch("roctop.collectors.read_ps_rows", side_effect=fake_read_ps_rows),
            patch("roctop.collectors.time.monotonic", side_effect=[100.0, 101.0, 103.1]),
        ):
            first = collectors.read_ps_rows_cached([42])
            second = collectors.read_ps_rows_cached([42])
            third = collectors.read_ps_rows_cached([42])

        self.assertEqual(calls, [[42], [42]])
        self.assertEqual(first[42]["user"], "demo1")
        self.assertEqual(second[42]["user"], "demo1")
        self.assertEqual(third[42]["user"], "demo2")

    def test_fresh_ps_read_prunes_expired_cache_rows(self) -> None:
        collectors._ps_row_cache[1] = (100.0, {"user": "old"})

        def fake_read_ps_rows(pids: list[int]) -> dict[int, dict[str, str]]:
            return {2: {"user": "new"}}

        with (
            patch("roctop.collectors.read_ps_rows", side_effect=fake_read_ps_rows),
            patch("roctop.collectors.time.monotonic", return_value=100.0 + collectors.PS_CACHE_TTL_SECONDS),
        ):
            rows = collectors.read_ps_rows_fresh([2])

        self.assertEqual(rows[2]["user"], "new")
        self.assertNotIn(1, collectors._ps_row_cache)

    def test_process_enrichment_reads_fresh_ps_rows_without_cache_delay(self) -> None:
        calls: list[list[int]] = []

        def fake_read_ps_rows(pids: list[int]) -> dict[int, dict[str, str]]:
            calls.append(list(pids))
            suffix = str(len(calls))
            return {
                pid: {
                    "user": "demo",
                    "cpu": suffix,
                    "mem": "0.1",
                    "etime": f"00:0{suffix}",
                    "comm": "python",
                    "args": f"python worker-{suffix}.py",
                }
                for pid in pids
            }

        process = ProcessInfo(gpu_index=0, pid=42)
        with patch("roctop.collectors.read_ps_rows", side_effect=fake_read_ps_rows):
            collectors.enrich_processes_with_ps([process])
            first_elapsed = process.elapsed
            collectors.enrich_processes_with_ps([process])

        self.assertEqual(calls, [[42], [42]])
        self.assertEqual(first_elapsed, "00:01")
        self.assertEqual(process.elapsed, "00:02")
        self.assertEqual(process.args, "python worker-2.py")

    def test_read_ps_rows_parses_ppid(self) -> None:
        def fake_run_command(args, timeout=None) -> CommandResult:
            return CommandResult(
                args=args,
                returncode=0,
                stdout="42 7 demo 12.5 0.3 01:02 python python train.py\n",
                stderr="",
            )

        with patch("roctop.collectors.run_command", side_effect=fake_run_command):
            rows = read_ps_rows([42])

        self.assertEqual(rows[42]["ppid"], "7")
        self.assertEqual(rows[42]["args"], "python train.py")

    def test_read_process_detail_reads_proc_fields_without_environ(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proc_root = Path(temp_dir)
            proc_dir = proc_root / "42"
            proc_dir.mkdir()
            (proc_dir / "status").write_text(
                "\n".join(
                    [
                        "Name:\tpython",
                        "State:\tS (sleeping)",
                        "Threads:\t9",
                        "VmRSS:\t2048 kB",
                        "VmSize:\t4096 kB",
                        "VmHWM:\t8192 kB",
                        "Cpus_allowed_list:\t0-3",
                        "voluntary_ctxt_switches:\t12",
                        "nonvoluntary_ctxt_switches:\t3",
                    ]
                ),
                encoding="utf-8",
            )
            (proc_dir / "cmdline").write_bytes(b"python\0train.py\0--batch\0" + b"4\0")
            (proc_dir / "environ").write_text("SECRET_TOKEN=do-not-read", encoding="utf-8")
            cwd_target = proc_root / "work"
            cwd_target.mkdir()
            exe_target = proc_root / "python"
            exe_target.write_text("", encoding="utf-8")
            (proc_dir / "cwd").symlink_to(cwd_target, target_is_directory=True)
            (proc_dir / "exe").symlink_to(exe_target)

            detail = collectors.read_process_detail(42, proc_root)

        self.assertEqual(detail.state, "S (sleeping)")
        self.assertEqual(detail.threads, 9)
        self.assertEqual(detail.vm_rss_kib, 2048)
        self.assertEqual(detail.vm_size_kib, 4096)
        self.assertEqual(detail.vm_hwm_kib, 8192)
        self.assertEqual(detail.cpu_allowed_list, "0-3")
        self.assertEqual(detail.voluntary_ctxt_switches, 12)
        self.assertEqual(detail.nonvoluntary_ctxt_switches, 3)
        self.assertEqual(detail.cmdline, "python train.py --batch 4")
        self.assertTrue(detail.cwd.endswith("/work"))
        self.assertTrue(detail.exe.endswith("/python"))
        self.assertNotIn("SECRET_TOKEN", str(detail))

    def test_read_process_detail_returns_partial_data_on_missing_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proc_root = Path(temp_dir)
            proc_dir = proc_root / "42"
            proc_dir.mkdir()
            (proc_dir / "status").write_text("State:\tR (running)\nThreads:\t1\n", encoding="utf-8")
            detail = collectors.read_process_detail(42, proc_root)

        self.assertEqual(detail.state, "R (running)")
        self.assertEqual(detail.threads, 1)
        self.assertIn("cmdline", detail.error)
        self.assertIn("cwd", detail.error)
        self.assertIn("exe", detail.error)

    def test_read_process_detail_reports_missing_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            detail = collectors.read_process_detail(999, Path(temp_dir))

        self.assertEqual(detail.pid, 999)
        self.assertIn("process exited", detail.error)

    def test_collect_snapshot_collects_process_ancestors_without_moving_gpu_rows(self) -> None:
        def fake_run_command(args, timeout=None) -> CommandResult:
            if args[0] == "rocm-smi":
                return CommandResult(
                    args=args,
                    returncode=0,
                    stdout=(
                        '{"card0": {"VRAM Total Memory (B)": "1000"}, '
                        '"system": {"Driver version": "6.14.14"}}'
                    ),
                    stderr="",
                )
            if args == collectors.AMD_SMI_PROCESS_ARGS:
                return CommandResult(
                    args=args,
                    returncode=0,
                    stdout=(
                        '[{"gpu": 0, "process_list": [{"process_info": '
                        '{"pid": 42, "name": "python", "mem_usage": {"value": 500}}}]}]'
                    ),
                    stderr="",
                )
            if args[0] == "amd-smi":
                return CommandResult(args=args, returncode=0, stdout="[]", stderr="")
            if args[0] == "ps":
                pids = args[-1]
                rows = {
                    "42": "42 7 demo 1.0 0.2 00:10 python python train.py",
                    "7": "7 1 demo 0.1 0.1 01:00 bash bash",
                    "1": "1 0 root 0.0 0.1 02:00 systemd /sbin/init",
                }
                stdout = "\n".join(rows[pid] for pid in pids.split(",") if pid in rows)
                return CommandResult(args=args, returncode=0, stdout=stdout, stderr="")
            return CommandResult(args=args, returncode=1, stdout="", stderr="")

        with (
            patch("roctop.collectors.run_command", side_effect=fake_run_command),
            patch("roctop.collectors.platform.node", return_value="node-a"),
        ):
            snapshot = collect_snapshot()

        self.assertEqual([proc.pid for proc in snapshot.processes], [42])
        self.assertEqual(snapshot.processes[0].ppid, 7)
        self.assertEqual([proc.pid for proc in snapshot.process_ancestors], [7, 1])
        self.assertEqual(snapshot.process_ancestors[0].ppid, 1)
        self.assertEqual(snapshot.process_ancestors[1].ppid, None)

    def test_collect_snapshot_merges_amd_smi_gpu_details(self) -> None:
        def fake_run_command(args, timeout=None) -> CommandResult:
            if args[0] == "rocm-smi":
                return CommandResult(
                    args=args,
                    returncode=0,
                    stdout=(
                        '{"card0": {"Card Model": "0x75b0", "GFX Version": "gfx950", '
                        '"GUID": "29921", "VRAM Total Memory (B)": "308902100992"}, '
                        '"system": {}}'
                    ),
                    stderr="",
                )
            if args == collectors.AMD_SMI_GPU_DETAIL_ARGS[0]:
                return CommandResult(
                    args=args,
                    returncode=0,
                    stdout=(
                        '[{"gpu": 0, "asic": {"market_name": "AMD Radeon Graphics", '
                        '"vendor": {"name": "Advanced Micro Devices, Inc. [AMD/ATI]"}, '
                        '"unique_id": "gpu-unique-0"}, "vbios": {"version": "113-D7020100-100"}, '
                        '"bus": {"bdf": "0000:03:00.0"}, "board": {"sku": "APM107573"}, '
                        '"limit": {"max_power": {"value": "300", "unit": "W"}}, '
                        '"driver": {"version": "6.14.14"}}]'
                    ),
                    stderr="",
                )
            if args == collectors.AMD_SMI_GPU_DETAIL_ARGS[1]:
                return CommandResult(
                    args=args,
                    returncode=0,
                    stdout=(
                        '[{"gpu_id": 0, "perf": {"performance_level": "auto"}, '
                        '"throttle": {"status": "THERMAL"}, "voltage": {"gfx_voltage": "1138mV"}}]'
                    ),
                    stderr="",
                )
            if args == collectors.AMD_SMI_PROCESS_ARGS:
                return CommandResult(args=args, returncode=0, stdout="[]", stderr="")
            return CommandResult(args=args, returncode=1, stdout="", stderr="")

        with patch("roctop.collectors.run_command", side_effect=fake_run_command):
            snapshot = collect_snapshot()

        self.assertEqual(snapshot.driver_version, "6.14.14")
        self.assertEqual(len(snapshot.gpus), 1)
        gpu = snapshot.gpus[0]
        self.assertEqual(gpu.name, "AMD Radeon Graphics")
        self.assertEqual(gpu.gpu_type, "AMD Instinct MI350X")
        self.assertEqual(gpu.vendor, "Advanced Micro Devices, Inc. [AMD/ATI]")
        self.assertEqual(gpu.unique_id, "gpu-unique-0")
        self.assertEqual(gpu.sku, "APM107573")
        self.assertEqual(gpu.vbios_version, "113-D7020100-100")
        self.assertEqual(gpu.pcie_bus, "0000:03:00.0")
        self.assertEqual(gpu.max_power_w, 300.0)
        self.assertEqual(gpu.performance_level, "auto")
        self.assertEqual(gpu.throttle_status, "THERMAL")
        self.assertEqual(gpu.voltage_mv, 1138.0)

    def test_collect_process_ancestors_dedupes_and_skips_missing_parents(self) -> None:
        calls: list[list[int]] = []

        def fake_read_ps_rows_cached(pids: list[int]) -> dict[int, dict[str, str]]:
            calls.append(list(pids))
            if pids == [7]:
                return {
                    7: {
                        "ppid": "999",
                        "user": "demo",
                        "cpu": "0.1",
                        "mem": "0.1",
                        "etime": "01:00",
                        "comm": "bash",
                        "args": "bash",
                    }
                }
            return {}

        processes = [
            ProcessInfo(gpu_index=0, pid=42, ppid=7),
            ProcessInfo(gpu_index=1, pid=43, ppid=7),
        ]
        with patch("roctop.collectors.read_ps_rows_cached", side_effect=fake_read_ps_rows_cached):
            ancestors = collect_process_ancestors(processes)

        self.assertEqual([proc.pid for proc in ancestors], [7])
        self.assertEqual(calls, [[7], [999]])

    def test_parse_rocm_smi_json(self) -> None:
        raw = {
            "card0": {
                "Temperature (Sensor junction) (C)": "60.0",
                "Fan Level": "42%",
                "current_fan_speed (rpm)": "3200",
                "Current Socket Graphics Package Power (W)": "266.0",
                "sclk clock speed:": "(173Mhz)",
                "mclk clock speed:": "(2000Mhz)",
                "GPU use (%)": "99",
                "VRAM Total Memory (B)": "308902100992",
                "VRAM Total Used Memory (B)": "200804560896",
                "Card Series": "AMD MI350X",
                "Card Model": "0x75b0",
                "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
                "Card SKU": "APM107573",
                "GUID": "29921",
                "GFX Version": "gfx950",
                "VBIOS Version": "113-D7020100-100",
                "PCIe Bus": "0000:03:00.0",
                "Max Graphics Package Power (W)": "300",
                "Performance Level": "auto",
                "Throttling Status": "THERMAL",
                "Voltage (mV)": "1138",
                "Unique ID": "gpu-unique-0",
            },
            "system": {
                "Driver version": "6.14.14",
                "PID710898": "demo::schedul, 1, 200145596416, 2735289460695, 88",
                "PID721888": "demo-worker, 0, 0, 0, 0",
            },
        }
        gpus, processes, driver = parse_rocm_smi_json(raw)
        self.assertEqual(driver, "6.14.14")
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].index, 0)
        self.assertEqual(gpus[0].guid, "29921")
        self.assertEqual(gpus[0].gpu_type, "AMD Instinct MI350X")
        self.assertEqual(gpus[0].vendor, "Advanced Micro Devices, Inc. [AMD/ATI]")
        self.assertEqual(gpus[0].sku, "APM107573")
        self.assertEqual(gpus[0].vbios_version, "113-D7020100-100")
        self.assertEqual(gpus[0].pcie_bus, "0000:03:00.0")
        self.assertEqual(gpus[0].max_power_w, 300.0)
        self.assertEqual(gpus[0].performance_level, "auto")
        self.assertEqual(gpus[0].throttle_status, "THERMAL")
        self.assertEqual(gpus[0].voltage_mv, 1138.0)
        self.assertEqual(gpus[0].unique_id, "gpu-unique-0")
        self.assertEqual(gpus[0].utilization_percent, 99)
        self.assertEqual(gpus[0].fan_percent, 42.0)
        self.assertEqual(gpus[0].fan_rpm, 3200)
        self.assertEqual(gpus[0].power_w, 266.0)
        self.assertEqual(gpus[0].sclk_mhz, 173)
        self.assertEqual(gpus[0].mclk_mhz, 2000)
        self.assertEqual(gpus[0].memory_used_bytes, 200804560896)
        self.assertEqual(len(processes), 2)
        self.assertEqual(processes[0].pid, 710898)
        self.assertEqual(processes[0].gpu_index, 1)
        self.assertEqual(processes[1].pid, 721888)
        self.assertEqual(processes[1].gpu_index, 0)
        self.assertEqual(processes[1].gpu_memory_bytes, 0)

    def test_parse_amd_smi_gpu_json_reads_nested_gpu_details(self) -> None:
        gpus = parse_amd_smi_gpu_json(
            {
                "gpu_data": [
                    {
                        "GPU ID": "GPU 0",
                        "asic": {
                            "market_name": "AMD Radeon Graphics",
                            "vendor": {"name": "Advanced Micro Devices, Inc. [AMD/ATI]"},
                            "gfx": "gfx1201",
                            "device_id": "0x75b0",
                            "unique_id": "gpu-unique-0",
                        },
                        "vbios": {"version": "113-D7020100-100"},
                        "bus": {"bdf": "0000:03:00.0"},
                        "pcie": {
                            "width": 16,
                            "speed": {"value": 32, "unit": "GT/s"},
                            "bandwidth": {"value": 120, "unit": "Mb/s"},
                            "current_bandwidth_sent": {"value": 40, "unit": "Mb/s"},
                            "current_bandwidth_received": {"value": 80, "unit": "Mb/s"},
                        },
                        "board": {"sku": "APM107573"},
                        "limit": {"max_power": {"value": "300", "unit": "W"}},
                        "perf": {"performance_level": "auto"},
                        "throttle": {"status": "THERMAL"},
                        "voltage": {"gfx_voltage": "1138 mV"},
                        "metric": {
                            "current_gfxclk": "159MHz",
                            "current_uclk": "2000MHz",
                            "gpu_utilization": "12.5%",
                            "vram_used": {"value": "296", "unit": "MiB"},
                            "vram_total": {"value": "287.7", "unit": "GiB"},
                        },
                    }
                ]
            }
        )

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].index, 0)
        self.assertEqual(gpus[0].name, "AMD Radeon Graphics")
        self.assertEqual(gpus[0].gpu_type, "AMD Radeon Graphics")
        self.assertEqual(gpus[0].gfx_version, "gfx1201")
        self.assertEqual(gpus[0].vendor, "Advanced Micro Devices, Inc. [AMD/ATI]")
        self.assertEqual(gpus[0].unique_id, "gpu-unique-0")
        self.assertEqual(gpus[0].sku, "APM107573")
        self.assertEqual(gpus[0].vbios_version, "113-D7020100-100")
        self.assertEqual(gpus[0].pcie_bus, "0000:03:00.0")
        self.assertEqual(gpus[0].pcie_current_link_speed, "32 GT/s")
        self.assertEqual(gpus[0].pcie_current_link_width, "x16")
        self.assertEqual(gpus[0].pcie_throughput, "120 Mb/s")
        self.assertEqual(gpus[0].pcie_tx_throughput, "40 Mb/s")
        self.assertEqual(gpus[0].pcie_rx_throughput, "80 Mb/s")
        self.assertEqual(gpus[0].max_power_w, 300.0)
        self.assertEqual(gpus[0].performance_level, "auto")
        self.assertEqual(gpus[0].throttle_status, "THERMAL")
        self.assertEqual(gpus[0].voltage_mv, 1138.0)
        self.assertEqual(gpus[0].sclk_mhz, 159)
        self.assertEqual(gpus[0].mclk_mhz, 2000)
        self.assertEqual(gpus[0].memory_used_bytes, 296 * 1024**2)
        self.assertEqual(gpus[0].memory_total_bytes, int(287.7 * 1024**3))
        self.assertEqual(gpus[0].utilization_percent, 12.5)

    def test_enrich_gpus_with_sysfs_pcie_reads_link_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            devices_path = Path(temp_dir)
            device_path = devices_path / "0000:03:00.0"
            device_path.mkdir()
            (device_path / "current_link_speed").write_text("16.0 GT/s PCIe\n", encoding="utf-8")
            (device_path / "current_link_width").write_text("16\n", encoding="utf-8")
            (device_path / "max_link_speed").write_text("32.0 GT/s PCIe\n", encoding="utf-8")
            (device_path / "max_link_width").write_text("16\n", encoding="utf-8")

            gpu = GpuInfo(index=0, pcie_bus="03:00.0")
            collectors.enrich_gpus_with_sysfs_pcie([gpu], devices_path)

        self.assertEqual(gpu.pcie_current_link_speed, "16 GT/s")
        self.assertEqual(gpu.pcie_current_link_width, "x16")
        self.assertEqual(gpu.pcie_max_link_speed, "32 GT/s")
        self.assertEqual(gpu.pcie_max_link_width, "x16")

    def test_parse_amd_smi_gpu_json_reads_pcie_only_metric_entry(self) -> None:
        gpus = parse_amd_smi_gpu_json(
            [
                {
                    "gpu": 0,
                    "pcie": {
                        "width": 8,
                        "speed": {"value": 16, "unit": "GT/s"},
                        "current_bandwidth_received": {"value": 12, "unit": "Mb/s"},
                    },
                }
            ]
        )

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].pcie_current_link_speed, "16 GT/s")
        self.assertEqual(gpus[0].pcie_current_link_width, "x8")
        self.assertEqual(gpus[0].pcie_rx_throughput, "12 Mb/s")

    def test_parse_rocm_smi_json_sets_fallback_process_memory_percent(self) -> None:
        gpus, processes, _driver = parse_rocm_smi_json(
            {
                "card0": {"VRAM Total Memory (B)": str(4 * 1024 * 1024)},
                "system": {"PID42": "demo-worker, 0, 1048576, 0, 0"},
            }
        )

        self.assertEqual(gpus[0].index, 0)
        self.assertEqual(processes[0].gpu_index, 0)
        self.assertEqual(processes[0].gpu_memory_percent, 25.0)

    def test_parse_rocm_smi_json_reads_metric_detail_aliases(self) -> None:
        gpus, _processes, _driver = parse_rocm_smi_json(
            {
                "card0": {
                    "indep_throttle_status": "THERMAL",
                    "voltage_gfx (mV)": "1138",
                }
            }
        )

        self.assertEqual(gpus[0].throttle_status, "THERMAL")
        self.assertEqual(gpus[0].voltage_mv, 1138.0)

    def test_parse_rocm_smi_json_reads_pcie_metrics(self) -> None:
        gpus, _processes, _driver = parse_rocm_smi_json(
            {
                "card0": {
                    "PCIe Link Speed": {"value": 160, "unit": "0.1 GT/s"},
                    "PCIe Link Width": "16",
                    "Estimated maximum PCIe bandwidth over the last second (MB/s)": "123.456",
                }
            }
        )

        self.assertEqual(gpus[0].pcie_current_link_speed, "16 GT/s")
        self.assertEqual(gpus[0].pcie_current_link_width, "x16")
        self.assertEqual(gpus[0].pcie_throughput, "123.456 MB/s")

    def test_parse_rocm_smi_json_treats_unsupported_optional_floats_as_missing(self) -> None:
        gpus, _, _ = parse_rocm_smi_json(
            {
                "card0": {
                    "Temperature (Sensor junction) (C)": "not supported",
                    "Current Socket Graphics Package Power (W)": "N/A",
                }
            }
        )

        self.assertIsNone(gpus[0].temperature_c)
        self.assertIsNone(gpus[0].power_w)

    def test_parse_rocm_smi_json_normalizes_reported_model_name(self) -> None:
        gpus, _, _ = parse_rocm_smi_json(
            {
                "card0": {
                    "Card Series": "AMD MI350X",
                    "GFX Version": "gfx950",
                },
                "card1": {
                    "Card Series": "AMD Instinct\u2122 MI355X",
                },
                "card2": {
                    "Card Series": "Navi 31 [Radeon Pro W7900]",
                },
            }
        )

        self.assertEqual(gpus[0].gpu_type, "AMD Instinct MI350X")
        self.assertEqual(gpus[1].gpu_type, "AMD Instinct MI355X")
        self.assertEqual(gpus[2].gpu_type, "AMD Radeon PRO W7900")

    def test_parse_rocm_smi_json_maps_instinct_device_ids(self) -> None:
        gpus, _, _ = parse_rocm_smi_json(
            {
                "card0": {"Card Model": "0x738c"},
                "card1": {"Card Model": "7408"},
                "card2": {"Card Model": "0x740c"},
                "card3": {"Card Model": "0x740f"},
                "card4": {"Card Model": "0x74a0"},
                "card5": {"Card Model": "0x74a1"},
            }
        )

        self.assertEqual(
            [gpu.gpu_type for gpu in gpus],
            [
                "AMD Instinct MI100",
                "AMD Instinct MI250X",
                "AMD Instinct MI200 Series",
                "AMD Instinct MI210",
                "AMD Instinct MI300A",
                "AMD Instinct MI300X",
            ],
        )

    def test_parse_amd_pci_models_normalizes_product_names(self) -> None:
        models = parse_amd_pci_models(
            """
1002  Advanced Micro Devices, Inc. [AMD/ATI]
\t7448  Navi 31 [Radeon Pro W7900]
\t74a1  Aqua Vanjaram [Instinct MI300X]
\t\t1002 0e3a  Subdevice entry
10de  NVIDIA Corporation
\t2684  AD102 [GeForce RTX 4090]
"""
        )

        self.assertEqual(models["0x7448"], "AMD Radeon PRO W7900")
        self.assertEqual(models["0x74a1"], "AMD Instinct MI300X")
        self.assertNotIn("0x2684", models)

    def test_parse_rocm_smi_json_uses_architecture_series_fallback(self) -> None:
        gpus, _, _ = parse_rocm_smi_json(
            {
                "card0": {
                    "GFX Version": "gfx950",
                }
            }
        )

        self.assertEqual(gpus[0].gpu_type, "AMD Instinct MI350 Series")

    def test_parse_amd_smi_process_json_filters_zero_memory(self) -> None:
        gpus, _, _ = parse_rocm_smi_json(
            {
                "card4": {
                    "GPU use (%)": "99",
                    "VRAM Total Memory (B)": "308902100992",
                    "VRAM Total Used Memory (B)": "200804560896",
                    "Card Model": "0x75b0",
                }
            }
        )
        raw = [
            {
                "gpu": 4,
                "process_list": [
                    {"process_info": {"pid": 1, "name": "idle", "mem_usage": {"value": 0, "unit": "B"}}},
                    {
                        "process_info": {
                            "pid": 710898,
                            "name": "N/A",
                            "mem_usage": {"value": 200145596416, "unit": "B"},
                        }
                    },
                ],
            }
        ]
        processes = parse_amd_smi_process_json(raw, gpus)
        self.assertEqual(len(processes), 1)
        self.assertEqual(processes[0].gpu_index, 4)
        self.assertEqual(processes[0].pid, 710898)
        self.assertGreater(processes[0].gpu_memory_percent, 60)

    def test_parse_amd_smi_process_json_converts_memory_units_to_bytes(self) -> None:
        gpus, _, _ = parse_rocm_smi_json(
            {
                "card0": {
                    "VRAM Total Memory (B)": str(2 * 1024 * 1024 * 1024),
                }
            }
        )
        raw = [
            {
                "gpu": 0,
                "process_list": [
                    {"process_info": {"pid": 42, "name": "worker", "mem_usage": {"value": 1, "unit": "GiB"}}},
                ],
            }
        ]

        processes = parse_amd_smi_process_json(raw, gpus)

        self.assertEqual(processes[0].gpu_memory_bytes, 1024 * 1024 * 1024)
        self.assertEqual(processes[0].gpu_memory_percent, 50.0)

    def test_parse_memory_bytes_field_rejects_non_finite_values(self) -> None:
        self.assertEqual(parse_memory_bytes_field({"value": "NaN", "unit": "B"}), 0)
        self.assertEqual(parse_memory_bytes_field({"value": "Infinity", "unit": "B"}), 0)

    def test_parse_amd_smi_process_json_skips_non_finite_memory(self) -> None:
        raw = [
            {
                "gpu": 0,
                "process_list": [
                    {"process_info": {"pid": 42, "name": "worker", "mem_usage": {"value": "Infinity"}}},
                ],
            }
        ]

        processes = parse_amd_smi_process_json(raw, [GpuInfo(index=0, memory_total_bytes=1024)])

        self.assertEqual(processes, [])

    def test_parse_amd_smi_process_json_skips_non_object_entries(self) -> None:
        gpus, _, _ = parse_rocm_smi_json(
            {
                "card0": {
                    "VRAM Total Memory (B)": "10485760",
                    "VRAM Total Used Memory (B)": "1048576",
                }
            }
        )
        raw = [
            "not a gpu entry",
            {"gpu": 0, "process_list": "N/A"},
            {
                "gpu": 0,
                "process_list": [
                    "not a process entry",
                    {"process_info": "N/A"},
                    {"process_info": {"pid": 123, "memory_usage": "N/A"}},
                    {"process_info": {"pid": 456, "memory_usage": {"vram_mem": {"value": 1048576}}}},
                ],
            },
        ]

        processes = parse_amd_smi_process_json(raw, gpus)

        self.assertEqual(len(processes), 1)
        self.assertEqual(processes[0].pid, 456)

    def test_merge_process_sources_fills_name(self) -> None:
        primary = [ProcessInfo(gpu_index=4, pid=42, gpu_memory_bytes=100)]
        fallback = [
            ProcessInfo(gpu_index=None, pid=42, name="python", command="python", gpu_memory_bytes=200),
            ProcessInfo(gpu_index=None, pid=43, name="server", command="server", gpu_memory_bytes=0),
        ]
        merged = merge_process_sources(primary, fallback)
        self.assertEqual(merged[0].name, "python")
        self.assertEqual(merged[0].gpu_memory_bytes, 100)
        self.assertEqual(merged[1].pid, 43)
        self.assertEqual(merged[1].command, "server")

    def test_format_bytes_mib(self) -> None:
        self.assertEqual(format_bytes_mib(0), "0MiB")
        self.assertEqual(format_bytes_mib(1024 * 1024), "1MiB")
        self.assertEqual(format_bytes_mib(2 * 1024 * 1024 * 1024), "2.00GiB")


if __name__ == "__main__":
    unittest.main()

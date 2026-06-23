from __future__ import annotations

import unittest
from unittest.mock import patch

from roctop import collectors
from roctop.collectors import (
    CommandInterrupted,
    CommandResult,
    CommandTimeout,
    collect_snapshot,
    load_json_from_text,
    merge_process_sources,
    parse_amd_pci_models,
    parse_amd_smi_process_json,
    parse_rocm_smi_json,
    run_command,
)
from roctop.formatting import format_bytes_mib
from roctop.models import ProcessInfo


class CollectorTests(unittest.TestCase):
    def test_load_json_from_text_skips_warning_prefix(self) -> None:
        data = load_json_from_text('WARNING: noisy\n{"ok": true}')
        self.assertEqual(data, {"ok": True})

    def test_run_command_raises_command_timeout(self) -> None:
        import subprocess
        from unittest.mock import patch

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["rocm-smi"], 1)):
            with self.assertRaises(CommandTimeout):
                run_command(["rocm-smi"], timeout=1)

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
                "Card Model": "0x75b0",
                "GUID": "29921",
                "GFX Version": "gfx950",
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
        self.assertEqual(gpus[0].utilization_percent, 99)
        self.assertEqual(gpus[0].fan_percent, 42.0)
        self.assertEqual(gpus[0].fan_rpm, 3200)
        self.assertEqual(gpus[0].power_w, 266.0)
        self.assertEqual(gpus[0].sclk_mhz, 173)
        self.assertEqual(gpus[0].mclk_mhz, 2000)
        self.assertEqual(gpus[0].memory_used_bytes, 200804560896)
        self.assertEqual(len(processes), 2)
        self.assertEqual(processes[0].pid, 710898)
        self.assertEqual(processes[1].pid, 721888)
        self.assertEqual(processes[1].gpu_memory_bytes, 0)

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

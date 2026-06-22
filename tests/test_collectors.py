from __future__ import annotations

import unittest

from roctop.collectors import (
    CommandTimeout,
    load_json_from_text,
    merge_process_sources,
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
                "PID710898": "sglang::schedul, 1, 200145596416, 2735289460695, 88",
                "PID721888": "python3, 0, 0, 0, 0",
            },
        }
        gpus, processes, driver = parse_rocm_smi_json(raw)
        self.assertEqual(driver, "6.14.14")
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].index, 0)
        self.assertEqual(gpus[0].guid, "29921")
        self.assertEqual(gpus[0].gpu_type, "AMD MI350")
        self.assertEqual(gpus[0].utilization_percent, 99)
        self.assertEqual(gpus[0].fan_percent, 42.0)
        self.assertEqual(gpus[0].fan_rpm, 3200)
        self.assertEqual(gpus[0].power_w, 266.0)
        self.assertEqual(gpus[0].sclk_mhz, 173)
        self.assertEqual(gpus[0].mclk_mhz, 2000)
        self.assertEqual(gpus[0].memory_used_bytes, 200804560896)
        self.assertEqual(len(processes), 1)
        self.assertEqual(processes[0].pid, 710898)

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

    def test_merge_process_sources_fills_name(self) -> None:
        primary = [ProcessInfo(gpu_index=4, pid=42, gpu_memory_bytes=100)]
        fallback = [ProcessInfo(gpu_index=None, pid=42, name="python", command="python", gpu_memory_bytes=200)]
        merged = merge_process_sources(primary, fallback)
        self.assertEqual(merged[0].name, "python")
        self.assertEqual(merged[0].gpu_memory_bytes, 100)

    def test_format_bytes_mib(self) -> None:
        self.assertEqual(format_bytes_mib(0), "0MiB")
        self.assertEqual(format_bytes_mib(1024 * 1024), "1MiB")
        self.assertEqual(format_bytes_mib(2 * 1024 * 1024 * 1024), "2.00GiB")


if __name__ == "__main__":
    unittest.main()

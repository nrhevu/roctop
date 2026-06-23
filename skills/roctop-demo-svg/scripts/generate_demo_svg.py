#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import random
import re
import sys
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

FORBIDDEN_STRINGS = (
    "root",
    "python3",
    "sglang",
    "scratch",
    "vunguyen",
    "/home",
    "/proc",
    "/opt",
    ".venv",
    "amd-smi",
    "rocm-smi",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render roctop docs/demo.svg from synthetic data.")
    parser.add_argument("--output", default="docs/demo.svg", help="SVG output path. Default: docs/demo.svg")
    parser.add_argument("--seed", type=int, default=20260622, help="Seed for random-walk graph data.")
    parser.add_argument("--width", type=int, default=180, help="Rich console width.")
    parser.add_argument("--height", type=int, default=54, help="Terminal height used for layout.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    from rich.console import Console
    from roctop.history import MetricSample, MetricsHistory
    from roctop.interaction import ProcessViewState
    from roctop.models import GpuInfo, ProcessInfo, Snapshot
    from roctop.render import render_snapshot

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot()
    history = build_history(snapshot.timestamp, args.seed)
    state = ProcessViewState(
        selected_pid=420100,
        sort_field="gpu_memory_percent",
        sort_desc=True,
        mode="sort_menu",
        sort_menu_index=2,
        viewport_rows=9,
    )

    console = Console(
        width=args.width,
        record=True,
        force_terminal=True,
        color_system="truecolor",
        file=StringIO(),
    )
    console.print(render_snapshot(snapshot, history, state, terminal_height=args.height, terminal_width=args.width))
    console.save_svg(str(output), title="roctop", unique_id="roctop-demo")
    vectorize_braille_graphs(output)
    verify_svg(output)
    print(f"Wrote {output}")
    return 0


def build_snapshot() -> object:
    from roctop.models import GpuInfo, ProcessInfo, Snapshot

    gib = 1024**3
    memory_total = int(288 * gib)
    guid_values = ["38421", "59107", "62043", "71896", "84217", "93504", "104682", "118295"]
    mem_gib = [266.4, 258.8, 241.5, 221.2, 188.4, 176.9, 96.2, 0.3]
    util = [98, 96, 91, 88, 76, 71, 34, 3]
    temps = [60, 59, 58, 62, 57, 56, 54, 53]
    powers = [648, 621, 594, 602, 271, 268, 286, 259]
    sclk = [2187, 2148, 2096, 2160, 194, 205, 822, 206]

    return Snapshot(
        timestamp=datetime(2026, 6, 22, 15, 45, 0),
        node_name="node-a",
        driver_version="6.14.14",
        gpus=[
            GpuInfo(
                index=index,
                name="0x75b0",
                guid=guid_values[index],
                gpu_type="AMD Instinct MI350X",
                gfx_version="gfx950",
                temperature_c=temps[index],
                power_w=powers[index],
                sclk_mhz=sclk[index],
                mclk_mhz=2000,
                memory_used_bytes=int(mem_gib[index] * gib),
                memory_total_bytes=memory_total,
                utilization_percent=util[index],
            )
            for index in range(8)
        ],
        processes=[
            proc(0, 420100, "demo::trainer_rank0", "01:26:03", 96.8, 0.3, 266.4, 92.5),
            proc(1, 420101, "demo::trainer_rank1", "01:26:03", 98.4, 0.3, 258.8, 89.9),
            proc(2, 420102, "demo::trainer_rank2", "01:25:58", 92.6, 0.2, 241.5, 83.9),
            proc(3, 420103, "demo::trainer_rank3", "01:25:57", 89.1, 0.2, 221.2, 76.8),
            proc(4, 420104, "demo::eval_worker", "00:42:17", 74.2, 0.2, 188.4, 65.4),
            proc(5, 420105, "demo::batch_sampler", "00:39:12", 68.7, 0.2, 176.9, 61.4),
            proc(6, 420106, "demo::metrics_agent", "00:18:44", 18.5, 0.1, 96.2, 33.4),
            proc(None, 420107, "demo::preprocess_worker", "01:31:22", 0.4, 0.0, 0.0, 0.0),
        ],
    )


def proc(
    gpu_index: int | None,
    pid: int,
    name: str,
    elapsed: str,
    cpu: float,
    mem: float,
    gpu_mem_gib: float,
    gpu_mem_percent: float,
) -> object:
    from roctop.models import ProcessInfo

    args_by_name = {
        "demo::trainer_rank0": "--model demo-mi350-llm --dataset synthetic-text --tp 8 --batch-size 64",
        "demo::trainer_rank1": "--model demo-mi350-llm --dataset synthetic-text --tp 8 --batch-size 64",
        "demo::trainer_rank2": "--model demo-mi350-llm --dataset synthetic-text --tp 8 --batch-size 64",
        "demo::trainer_rank3": "--model demo-mi350-llm --dataset synthetic-text --tp 8 --batch-size 64",
        "demo::eval_worker": "--suite synthetic-eval --shards 8 --report /demo/results",
        "demo::batch_sampler": "--queue synthetic-batches --prefetch 12 --workers 16",
        "demo::metrics_agent": "--target synthetic-cluster --interval 1s",
        "demo::preprocess_worker": "--input synthetic-corpus --queue demo-preprocess-queue",
    }
    return ProcessInfo(
        gpu_index=gpu_index,
        pid=pid,
        user="demo",
        cpu_percent=cpu,
        host_mem_percent=mem,
        elapsed=elapsed,
        name=name,
        command=name,
        args=f"{name} {args_by_name[name]}",
        gpu_memory_bytes=int(gpu_mem_gib * 1024**3),
        gpu_memory_percent=gpu_mem_percent,
    )


def build_history(timestamp: datetime, seed: int) -> object:
    from roctop.history import MetricSample, MetricsHistory

    rng = random.Random(seed)
    sample_count = 180
    history = MetricsHistory(max_samples=1081)
    start = timestamp - timedelta(seconds=sample_count - 1)
    values = {"cpu": 53.0, "mem": 61.0, "gpu": 79.0, "gpu_mem": 68.0}
    for index in range(sample_count):
        values["cpu"] = clamp(values["cpu"] + rng.uniform(-8.5, 7.0), 22.0, 88.0)
        values["mem"] = clamp(values["mem"] + rng.uniform(-2.8, 3.2), 45.0, 75.0)
        values["gpu"] = clamp(values["gpu"] + rng.uniform(-15.0, 13.0), 35.0, 99.0)
        values["gpu_mem"] = clamp(values["gpu_mem"] + rng.uniform(-6.0, 5.0), 38.0, 92.0)
        if index in {13, 29, 47, 61}:
            values["gpu"] = max(35.0, values["gpu"] - rng.uniform(18.0, 26.0))
        if index in {20, 42, 58}:
            values["cpu"] = min(88.0, values["cpu"] + rng.uniform(14.0, 22.0))
        history.append_sample(
            MetricSample(
                timestamp=start + timedelta(seconds=index),
                avg_cpu_percent=values["cpu"],
                avg_mem_percent=values["mem"],
                avg_gpu_percent=values["gpu"],
                avg_gpu_mem_percent=values["gpu_mem"],
            )
        )
    return history


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


TEXT_ELEMENT_RE = re.compile(r"<text (?P<attrs>[^>]*)>(?P<content>.*?)</text>", re.DOTALL)
SVG_ATTR_RE = re.compile(r'([A-Za-z_:][-A-Za-z0-9_:.]*)="([^"]*)"')
BRAILLE_DOTS = (
    (0x01, 0, 0),
    (0x02, 0, 1),
    (0x04, 0, 2),
    (0x40, 0, 3),
    (0x08, 1, 0),
    (0x10, 1, 1),
    (0x20, 1, 2),
    (0x80, 1, 3),
)


def vectorize_braille_graphs(path: Path) -> None:
    svg = path.read_text()
    font_size = css_px(svg, "font-size", 20.0)

    def replace_text(match: re.Match[str]) -> str:
        content = html.unescape(match.group("content"))
        if not any(is_braille(char) for char in content):
            return match.group(0)
        attrs = dict(SVG_ATTR_RE.findall(match.group("attrs")))
        x = float(attrs["x"])
        y = float(attrs["y"])
        text_length = float(attrs.get("textLength", "0") or "0")
        cell_width = text_length / len(content) if text_length and content else font_size * 0.61
        col_width = cell_width / 2.0
        dot_size = min(col_width * 0.44, font_size * 0.14)
        row_pitch = font_size * 0.22
        row_origin = y - font_size * 0.76
        rects: list[str] = []
        for index, char in enumerate(content):
            if not is_braille(char):
                continue
            mask = ord(char) - 0x2800
            cell_x = x + index * cell_width
            for bit, column, row in BRAILLE_DOTS:
                if not mask & bit:
                    continue
                center_x = cell_x + (column + 0.5) * col_width
                center_y = row_origin + row * row_pitch
                rects.append(
                    '<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="{rx}"/>'.format(
                        x=svg_number(center_x - dot_size / 2.0),
                        y=svg_number(center_y - dot_size / 2.0),
                        size=svg_number(dot_size),
                        rx=svg_number(dot_size * 0.24),
                    )
                )
        group_attrs = []
        if "class" in attrs:
            group_attrs.append(f'class="{attrs["class"]}"')
        if "clip-path" in attrs:
            group_attrs.append(f'clip-path="{attrs["clip-path"]}"')
        return f"<g {' '.join(group_attrs)}>{''.join(rects)}</g>"

    path.write_text(TEXT_ELEMENT_RE.sub(replace_text, svg))


def css_px(svg: str, property_name: str, default: float) -> float:
    match = re.search(rf"{re.escape(property_name)}:\s*([0-9.]+)px", svg)
    return float(match.group(1)) if match else default


def is_braille(char: str) -> bool:
    return "\u2800" <= char <= "\u28ff"


def svg_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def verify_svg(path: Path) -> None:
    text = html.unescape(path.read_text()).replace("\xa0", " ")
    missing = []
    guid_values = ("38421", "59107", "62043", "71896", "84217", "93504", "104682", "118295")
    for expected in guid_values:
        if expected not in text:
            missing.append(f"GUID {expected}")
    for expected in (
        "AMD Instinct MI350X",
        "demo::trainer_rank0",
        "demo::eval_worker",
        "demo::metrics_agent",
        "Sort",
        "%GPU-MEM",
        "node-a",
    ):
        if expected not in text:
            missing.append(expected)
    forbidden = [value for value in FORBIDDEN_STRINGS if value in text]
    if forbidden:
        missing.append("forbidden real-machine strings: " + ", ".join(forbidden))
    if any(is_braille(char) for char in text):
        missing.append("raw Braille graph glyphs")
    if missing:
        raise SystemExit("SVG validation failed: " + "; ".join(missing))


if __name__ == "__main__":
    raise SystemExit(main())

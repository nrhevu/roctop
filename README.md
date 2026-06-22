# roctop

`roctop` is a small AMD ROCm GPU monitor. It shows GPU VRAM usage,
utilization, type, temperature, fan, power, clocks, live system/GPU graphs,
and GPU processes in a refreshing terminal UI.

## Demo

![roctop demo](docs/demo.svg)

## Features

- GPU table with DID/GUID, temperature, fan, power, SCLK/MCLK, VRAM usage, and utilization bars.
- Live graph panel for average host CPU, host memory, GPU utilization, and GPU memory usage.
- Process table with GPU memory, host CPU/memory, elapsed time, full wrapped commands, and a movable cursor.
- Interactive process controls for scrolling, sorting, and SIGTERM kill confirmation.
- `--once` and `--json` modes for logs, scripts, and quick snapshots.

## Install

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .
export PATH="$PWD/.venv/bin:$PATH"
```

## Usage

```bash
roctop
roctop --interval 0.5
roctop --once
roctop --json
```

## Live Controls

```text
j/k or Up/Down    move process cursor
PgUp/PgDn         scroll processes
s                 open sort menu
x                 kill selected process with confirmation
q or Ctrl-C       quit
```

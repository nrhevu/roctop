# roctop

`roctop` is a small AMD ROCm GPU monitor. It shows GPU VRAM usage,
utilization, type, temperature, fan, power, clocks, live system/GPU graphs,
and GPU processes in a refreshing terminal UI.

## Demo

![roctop demo](docs/demo.svg)

The demo is rendered from synthetic data; process names, PIDs, users, and metrics are placeholders.

## Features

- **GPU overview:** compact table for every ROCm GPU with GUID, standardized model, architecture, temperature, fan status, power draw, SCLK/MCLK clocks, VRAM usage, and utilization.
- **Readable utilization bars:** memory and GPU utilization are shown as inline bars with percentage labels and threshold colors, making idle, busy, and saturated devices easy to scan.
- **Live history graphs:** split graph panel tracks average host CPU, host memory, GPU utilization, and GPU memory usage over the recent refresh window.
- **Process visibility:** process table shows GPU index, PID, user, GPU memory, GPU memory percent, host CPU/memory percent, elapsed runtime, and full wrapped command lines.
- **Interactive process navigation:** move through processes with `j/k` or arrow keys, page through long lists, toggle process tree view, keep the selected process visible across refreshes, and sort by GPU, memory, CPU, PID, user, time, or command.
- **Safe process actions:** `x` opens a high-contrast confirmation menu with Cancel, SIGTERM, and SIGKILL choices, plus status messages for missing processes, permission errors, and other failures.
- **Robust data collection:** combines `rocm-smi` GPU snapshots, `amd-smi process` process memory data, and `ps` process metadata, with fallbacks when process-specific data is missing or malformed.
- **Script-friendly modes:** `--once` renders a single terminal snapshot, `--json` prints normalized snapshot data, and `--interval` controls live refresh cadence.

## Install

From GitHub:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install "git+https://github.com/nrhevu/roctop.git"
export PATH="$PWD/.venv/bin:$PATH"
```

From a local checkout:

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
t                 toggle process tree
/                 search processes
n/N               next/previous search match
x                 kill selected process with confirmation
q or Ctrl-C       quit
```

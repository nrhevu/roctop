# roctop

`roctop` is a lightweight terminal monitor for AMD ROCm GPUs. It gives you a
live, keyboard-driven view of GPU health, memory pressure, utilization, history
graphs, and the processes currently using your accelerators.

It is built for busy ROCm machines where you want the quick feel of `top`,
but with GPU details that are easier to scan than raw `rocm-smi` output.

## Demo

![roctop demo](https://raw.githubusercontent.com/nrhevu/roctop/v1.0.0/docs/demo.svg)

The demo uses synthetic data. Process names, users, PIDs, paths, and metrics are
placeholders.

## Highlights

- Live Rich terminal UI for AMD GPUs.
- Compact per-GPU table with model, architecture, GUID, temperature, fan,
  power, clocks, VRAM usage, and utilization.
- Host and GPU history graphs for CPU, system memory, GPU utilization, and GPU
  memory pressure.
- Process table with GPU index, PID, user, GPU memory, host CPU/memory, elapsed
  time, and wrapped command lines.
- Interactive search, filtering, sorting, GPU focus, table zoom, process tree
  navigation, and process inspection.
- Confirmation flow for SIGTERM/SIGKILL actions.
- Scriptable `--once` and `--json` modes.

## Requirements

- Python 3.10 or newer.
- ROCm command-line tools on `PATH`.
- `rocm-smi` is required for GPU snapshots.
- `amd-smi` is optional and used when available for richer GPU/process details.

`roctop` is designed for Linux ROCm systems. On machines without ROCm tools it
will fail fast with a clear command error.

## Install

```bash
pip install roctop
```

From source:

```bash
git clone https://github.com/nrhevu/roctop.git
cd roctop
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/roctop
```

## Usage

Start the live monitor:

```bash
roctop
```

Common commands:

```bash
roctop --interval 0.5
roctop --once
roctop --json
roctop --version
python -m roctop --once
```

Options:

```text
--interval SECONDS   refresh interval for the live view, minimum: 0.05, default: 1.0
--once               render one terminal snapshot and exit
--json               print one normalized JSON snapshot and exit
--version            print the package version
```

## Live Controls

```text
j/k or Up/Down       move process cursor
PgUp/PgDn            move process cursor by page
0-9                  focus a GPU, when that GPU index exists
s                    open sort menu
t                    toggle process tree
z                    zoom process table
g                    toggle GPU graphs
,/.                  pan graphs older/newer
r                    reset graphs to live
/                    search processes
n/N                  next/previous search match
f                    filter processes
i                    inspect selected process
Space                select or deselect process
x                    open kill confirmation
Esc                  close graphs/menus or clear active filters
?                    open or close help
q or Ctrl-C          quit
```

Tree mode:

```text
p                    jump to parent process
h or Left            jump to previous sibling
l or Right           jump to next sibling
```

Menus and popups:

```text
h/l or arrows        move sort/kill menu selection
Enter                apply selected sort or kill option
y                    send SIGTERM in kill confirmation
j/k or Up/Down       scroll help or process inspection
h/l or Left/Right    page help or process inspection
Esc or q             cancel menus
```

## Data Sources

`roctop` combines several local data sources:

- `rocm-smi --json` for core GPU state and ROCm driver information.
- `amd-smi static --json` and `amd-smi metric --json` for extra model and metric
  details when available.
- `amd-smi process -G --json` for GPU process memory when available.
- `/proc`, `ps`, and container metadata for process command, user, runtime, CPU,
  host memory, ancestor, and container context.

Collection is intentionally tolerant. If optional process or detail commands are
missing, slow, or malformed, the live monitor keeps rendering the best available
snapshot.

## Development

Create an editable install:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .
```

Run tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

Run local checks against the current machine:

```bash
.venv/bin/roctop --once
.venv/bin/roctop --json
```

Refresh the synthetic demo asset:

```bash
skills/roctop-demo-svg/scripts/generate_demo_svg.py
```

## License

MIT. See [LICENSE](LICENSE).

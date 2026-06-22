# roctop

`roctop` is a small AMD ROCm GPU monitor inspired by `nvitop`. It shows GPU
VRAM usage, utilization, temperature, and GPU processes in a refreshing terminal
UI.

## Demo

![roctop demo](docs/demo.png)

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

Press `Ctrl-C` to quit the live view.

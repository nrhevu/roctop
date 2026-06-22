---
name: roctop-demo-svg
description: Generate or refresh roctop docs/demo.svg with deterministic synthetic Rich terminal data. Use when updating the roctop demo image, README demo asset, or screenshots that must show 8 AMD GPUs, dummy process names, random-looking metric graphs, and the process sort UI without leaking real host processes, users, or paths.
---

# roctop Demo SVG

## Purpose

Use this skill to regenerate `docs/demo.svg` for the `roctop` repository. Always render from synthetic data; never run `roctop`, `collect_snapshot()`, `rocm-smi`, `amd-smi`, or `ps` to populate the image.

## Required Output

The generated SVG must show:

- 8 GPU rows with realistic numeric GUIDs.
- Green, yellow, and red GPU/memory states by using utilization and VRAM percentages across low, medium, and high thresholds.
- Dummy users, PIDs, process names, and commands, for example `demo::trainer_rank0`.
- Metric graphs that move up and down with a seeded random-walk, not a simple repeating wave.
- The process sort UI open, with `%GPU-MEM` selected and the process table sorted descending.

## Quick Start

From the repository root, run:

```bash
python3 skills/roctop-demo-svg/scripts/generate_demo_svg.py
```

Optional flags:

```bash
python3 skills/roctop-demo-svg/scripts/generate_demo_svg.py --output docs/demo.svg --seed 20260622
```

The script imports `roctop` from `src/` when available, so an editable install is not required if dependencies such as `rich` are already available.

## Workflow

1. Run the script from the repo root.
2. Inspect `docs/demo.svg` or search it for forbidden real-machine strings.
3. Run the project tests if code changed:

```bash
.venv/bin/python -m unittest discover -s tests
```

4. Commit `docs/demo.svg` with a focused message such as `Update demo SVG`.

## Safety Checks

After rendering, verify that the SVG does not contain real host data:

```bash
rg -n "root|python3|sglang|scratch|/home|/proc|/opt|\\.venv" docs/demo.svg
```

No matches should be returned. The script also performs built-in checks and exits nonzero if key requirements are missing.

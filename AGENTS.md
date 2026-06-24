# Repository Guidelines

## Project Structure & Module Organization

`roctop` is a Python 3.10+ package using a `src/` layout. Application code lives in `src/roctop/`: `cli.py` handles command-line entrypoints, `collectors.py` gathers ROCm/AMD SMI data, `render.py` builds the Rich UI, `interaction.py` handles keyboard state, and `models.py` defines shared dataclasses. Tests live in `tests/` and mirror the main behavior areas: collectors, rendering, history, CLI, and interaction. Documentation assets live in `docs/`, including `docs/demo.svg`.

## Build, Test, and Development Commands

Create a local editable install:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .
```

Run the full test suite:

```bash
.venv/bin/python -m unittest discover -s tests
```

Run the app locally:

```bash
.venv/bin/roctop
.venv/bin/roctop --once
.venv/bin/roctop --json
```

Use `--once` or `--json` for quick verification when a live terminal UI is not needed.

## Codex Skills

On first entry to this repo, install bundled Codex skills before task work:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/roctop-demo-svg "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Re-run this when `skills/` changes so local skill behavior matches the repo. Use `skills/roctop-demo-svg/scripts/generate_demo_svg.py` to refresh demo assets.

## Coding Style & Naming Conventions

Use 4-space indentation, type hints, and small functions with explicit dataclasses for structured data. Match the existing straightforward Python style; avoid broad refactors while fixing targeted behavior. Use `snake_case` for functions and variables, `PascalCase` for dataclasses and exceptions, and uppercase constants for command arguments or color constants. There is no configured formatter or linter in `pyproject.toml`, so keep formatting consistent with nearby code.

## Live Metrics Graph Rules

The live metric graphs are intentionally decoupled from table refresh. `--interval` controls collector/table cadence only; graph rendering should advance on a stable 1-second clock. Keep graph samples bucketed by whole seconds so subsecond live redraws do not shift historical columns.

Graph columns use braille cells with two horizontal dot columns packed into one terminal cell. Preserve this dotted style instead of replacing it with solid block characters. When changing graph timing, ensure each tick advances at most one time bucket and does not skip forward multiple visible columns.

Once a graph bucket has been rendered, it must remain stable. Do not recompute already-rendered buckets from raw history on later frames, because late or subsecond samples can make existing dots disappear. Also avoid appending placeholder buckets before collector data exists for that bucket; wait for real history data, then append and freeze the bucket.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Add focused regression tests when fixing parser, rendering, interaction, or CLI behavior. Test files are named `tests/test_*.py`, and test methods use `test_*`. Prefer synthetic fixtures over host-dependent ROCm state so tests remain deterministic.

## Commit & Pull Request Guidelines

Recent commits use short imperative messages, for example `Update demo for eight GPUs` and `Handle malformed amd-smi process entries`. Keep commits scoped to one logical change and include tests or generated assets in the same commit when they prove the change.

Pull requests should include a concise summary, verification commands run, and screenshots or an updated `docs/demo.svg` for UI changes. Do not include real host process names, user names, paths, or machine-specific data in demo assets.

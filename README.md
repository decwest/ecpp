# ECPP

ECPP is a focused Python simulator for Error-Compensated Pure Pursuit path
tracking experiments. The repository targets fixed-speed evaluation:

```text
v = v_max
omega = clip(kappa * v_max, -omega_max, omega_max)
```

The implemented comparison methods are:

- `PP`
- `DPP`
- `ECPP without gate`
- `ECPP`

This repository is intentionally limited to curvature-law evaluation for the
IEEE Access manuscript. Velocity scheduling, acceleration-window search, and
non-differential platform controllers are outside this package.

## Setup

```bash
cd /home/decwest/decwest_workspace/ecpp
uv sync --dev
```

## Run Experiments

```bash
uv run python experiments/run_access_fixed_speed.py --config configs/experiments/access_test1_fixed_speed.yaml
uv run python experiments/run_lookahead_sweep.py
uv run python experiments/export_paper_assets.py
```

Default outputs are written under `results/`.

## Development Checks

```bash
uv run pytest -q
uv run python -m py_compile $(find src experiments -name '*.py')
```

## Layout

```text
configs/       YAML experiment settings
docs/          algorithm and experiment notes
experiments/   runnable experiment entrypoints
src/ecpp/      package source
tests/         regression tests
results/       generated outputs
```

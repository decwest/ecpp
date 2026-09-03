# ECPP

ECPP is a focused Python simulator for Error-Compensated Pure Pursuit path
tracking experiments. The repository targets fixed-speed evaluation:

```text
v = v_max
omega_raw = kappa * v_max
omega = clip(omega_raw, -1.5 rad/s, +1.5 rad/s)
```

The implemented comparison methods are:

- `PP`
- `DPP`
- `ECPP without gate`
- `ECPP`

The simulator runs at 30 Hz. The clip is instantaneous and is used only for
the simulated state update.
Curvature, saturation, and command-rate metrics use the raw request. A Nav2
velocity smoother and its acceleration dynamics are intentionally not modeled.

## Setup

```bash
cd /home/ytpc2022e/decwest_workspace/ecpp_ws/ecpp
uv sync --dev
```

## Run Experiments

```bash
uv run python experiments/run_access_fixed_speed.py --config configs/experiments/access_test1_fixed_speed.yaml
uv run python experiments/run_lookahead_sweep.py
uv run python experiments/export_paper_assets.py
```

Default outputs are written under `results/`.

The IEEE Access artifact generators are also canonical package commands:

```bash
uv run ecpp-paper figure1 --out-root /path/to/ECPP_ACCESS/manuscript
uv run ecpp-paper ieee-access --out-root /path/to/ECPP_ACCESS/manuscript
```

They write to `generated_preview/` by default; pass `--apply` to write the
frozen experiment outputs below `generated/`. Scripts with the historical
names under the paper project are thin compatibility wrappers around these
commands; controller and simulation implementations live only in this repo.

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

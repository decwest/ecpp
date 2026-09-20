# ECPP: Error-Compensated Pure Pursuit

Simulator and reproduction package for the paper

> F. Ohnishi and M. Takahashi, "Error-Compensated Pure Pursuit for Transient
> Response Shaping and Convergence from Large Tracking Errors," submitted to
> *IEEE Access*, 2026.

ECPP keeps the geometric curvature generation of Pure Pursuit (PP) and adds an
error-feedback compensation whose gains are the differences between the
intrinsic PP gains and target gains set by a natural frequency ω_n and a
damping ratio ζ. A gate attenuates the compensation far from the path, so the
controller behaves like PP there. This repository contains

- the `ecpp` Python package: PP, DPP (dual preview points), ungated ECPP, and
  ECPP curvature laws, the fixed-speed unicycle simulator, and the metrics;
- `ecpp-paper`, a command-line tool that regenerates every simulation table
  and figure of the paper's simulation section (Tests 1 to 4 of Section VI).

The ROS 2 / Nav2 controller plugin used in the real-robot experiments is a
separate repository:
[nav2_error_compensated_pure_pursuit_controller](https://github.com/decwest/nav2_error_compensated_pure_pursuit_controller).
The real-robot data and the figures of Section VII are not part of this
repository.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/decwest/ecpp.git
cd ecpp
python -m pip install -e .          # or: uv sync
```

Test 3 compiles a small C++ kernel on first use and needs a C++17 compiler,
preferably with OpenMP (`sudo apt install g++` on Ubuntu; on macOS
`brew install gcc` and `export CXX=g++-14`). Without OpenMP the sweep still
runs, single-threaded.

## Reproducing the paper

Each simulation test of the paper is one sub-command. Outputs go to
`generated_preview/` below the current directory (or `--out-root DIR`); pass
`--apply` to write to `generated/` instead. File names, sub-commands, and
the paper's test numbers agree.

| Command | Paper (Section VI) | Main outputs (`tables/`, `figures/`) | Time* |
|---|---|---|---|
| `ecpp-paper figure1` | Simulated PP/ECPP overview at two lookahead distances (the manuscript's Fig. 1 shows the real-robot runs of this scenario) | `figure1_conditions.tex`, `figure1_pp_ecpp_matrix.pdf` | 2 s |
| `ecpp-paper test1` | Test 1: (ω_n, ζ) sweep at two lookahead distances (Table 1, Fig. 4) | `sim_test1_grid_results_ld100.tex`, `..._ld050.tex`, `sim_test1_metrics.json`, `sim_test1_grid_*.pdf` | 3 s |
| `ecpp-paper test2` | Test 2: preview range versus local response on a straight–arc–straight path (Table 2, Fig. 5) | `sim_test2_results_display.tex`, `sim_test2_metrics.json`, `sim_test2_{trajectory,lateral_error,curvature,legend}.pdf` | 2 s |
| `ecpp-paper test3` | Test 3: convergence from large tracking errors, phase-plane sweep of initial conditions (Table 3, Fig. 6) | `sim_test3_results.tex`, `sim_test3_summary.json`, `sim_test3_phase_planes_row.pdf`, raw sweep in `sim_test3_sweep/` | 50 s |
| `ecpp-paper test4` | Test 4: PP, DPP, ungated ECPP, and ECPP at representative initial conditions (Table 4, Figs. 7–9) | `sim_test4_results_{straight,arc}_display.tex`, `sim_test4_metrics.json`, `sim_test4_by_condition/*.pdf` | 40 s |
| `ecpp-paper all` | everything above | | about 1.5 min |

\* wall time on a 12-thread desktop CPU (Test 3 with `--threads 12`). Table
and figure numbers refer to the submitted manuscript.

```bash
ecpp-paper all --out-root ~/ecpp_repro          # every test, into ~/ecpp_repro/generated_preview/
ecpp-paper test3 --threads 8                    # one test, with its own options (-h lists them)
```

The tables (`.tex`) and metric files (`.json`) are deterministic and
reproduce the submitted manuscript byte for byte; the figures differ only in
PDF metadata. Test 3 integrates 22 × 72,360 trajectories in double
precision without fast-math; its settled counts reproduce exactly with GCC on
x86-64, and other compilers may move a handful of points on the settling
boundary. A complete `sim_test3_sweep/` directory is reused on later runs, so
the tables and figures can be re-rendered without recomputing.

`docs/paper_tests.md` lists the frozen conditions of every test, and
`ecpp-paper hw-reference` writes the ideal simulated references for the arms
of real-robot Experiment 1.

## What is simulated

All tests share the idealized conditions of the paper: true pose, constant
linear velocity v₀ = 0.5 m/s, a 30 Hz control rate, and an instantaneous
angular-velocity limit |ω| ≤ 1.5 rad/s applied to the state update only.
The curvature command κ is converted to the raw angular-velocity command
ω_raw = v₀ κ; curvature, saturation, and rate metrics use ω_raw before
clipping. There is no acceleration limit, velocity smoother, actuation delay,
or noise.

Curvature laws (see `docs/algorithm.md`):

- **PP**: curvature of the circular arc to the lookahead point at distance L_d.
- **DPP**: two preview points on the vehicle axis with gains set by (ω_n, ζ),
  following Wang and Mouri (Trans. JSME, 2025).
- **ECPP w/o gate**: κ_PP − (ΔK_y e_y + ΔK_θ sin e_θ) with ΔK_y = (ω_n/v)² − 2/L_d²
  and ΔK_θ = 2ζω_n/v − 2/L_d.
- **ECPP**: the same compensation multiplied by the gate σ(e_y), a descending
  sigmoid of (e_y/L_d)² between 0.1 and 0.5.

## Using the package

```python
import numpy as np
from ecpp.paper.tracking import simulate_fixed_speed

path = np.array([[0.0, 0.0], [8.0, 0.0]])            # waypoints (x, y)
trace = simulate_fixed_speed(path=path, method="ECPP", lookahead_m=1.0,
                             omega_n=1.03, zeta=1.0, e_y0=0.3, e_psi0=0.0,
                             goal_arc_length=6.0)
print(trace.time[-1], trace.e_y[-1])                # time and lateral error at the goal
```

`method` is one of `"PP"`, `"DPP"`, `"ECPP w/o gate"`, `"ECPP"`. The
controllers themselves live in `ecpp.controllers`
(`calc_pp_curvature_to_point`, `calc_dpp_curvature`, `calc_ecpp_terms`) and
their parameters in `ecpp.config.EcppConfig`; `docs/api.md` lists the
modules.

## Development

```bash
python -m pip install -e .[dev]     # or: uv sync --dev
pytest -q
python -m py_compile $(git ls-files '*.py')
```

The tests cover the controllers, the metrics, the frozen experiment
conditions, and a reduced end-to-end run of every generator (the Test 3
kernel tests are skipped when no C++ compiler is found).

## Layout

```text
src/ecpp/            package: config, controllers/, simulation/, evaluation/, visualization/
src/ecpp/paper/      paper generators: figure1, ieee_access (engine, Test 1, Test 4),
                     test2_preview, test3_phase_plane (+ phase_plane_kernel.cpp), tracking
tests/               pytest suite
docs/                algorithm notes, API notes, frozen test conditions
```

## Citation

```bibtex
@article{ohnishi2026ecpp,
  author  = {Ohnishi, Fumiya and Takahashi, Masaki},
  title   = {Error-Compensated Pure Pursuit for Transient Response Shaping and
             Convergence from Large Tracking Errors},
  journal = {IEEE Access},
  year    = {2026},
  note    = {submitted}
}
```

## License

MIT, see `LICENSE`.

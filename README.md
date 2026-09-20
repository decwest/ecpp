# ECPP: Error-Compensated Pure Pursuit

Simulation code of the paper

> F. Ohnishi and M. Takahashi, "Error-Compensated Pure Pursuit for Transient
> Response Shaping and Convergence from Large Tracking Errors," 2026.

The `ecpp` package implements the PP, DPP, ungated ECPP, and ECPP curvature
laws with the fixed-speed unicycle simulator and metrics used in the paper,
and `ecpp-paper` regenerates every table and figure of the paper's four
simulation tests. The ROS 2 / Nav2 plugin used in the real-robot experiments
is a separate repository:
[nav2_error_compensated_pure_pursuit_controller](https://github.com/decwest/nav2_error_compensated_pure_pursuit_controller).
The real-robot data are not included.

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

## Reproducing the simulation tests

```bash
ecpp-paper all                  # all four tests, into ./generated_preview/
ecpp-paper test3 --threads 8    # one test; -h lists its options
```

Outputs go to `generated_preview/` below the current directory (or
`--out-root DIR`); `--apply` writes to `generated/` instead. File names
follow the paper's test numbers.

| Command | Test | Main outputs (`tables/`, `figures/`) | Time* |
|---|---|---|---|
| `ecpp-paper test1` | Test 1: (ω_n, ζ) sweep at two lookahead distances | `sim_test1_grid_results_ld100.tex`, `..._ld050.tex`, `sim_test1_metrics.json`, `sim_test1_grid_*.pdf` | 3 s |
| `ecpp-paper test2` | Test 2: preview range versus local response on a straight–arc–straight path | `sim_test2_results_display.tex`, `sim_test2_metrics.json`, `sim_test2_{trajectory,lateral_error,curvature,legend}.pdf` | 2 s |
| `ecpp-paper test3` | Test 3: convergence from large tracking errors (phase-plane sweep of initial conditions) | `sim_test3_results.tex`, `sim_test3_summary.json`, `sim_test3_phase_planes_row.pdf`, raw sweep in `sim_test3_sweep/` | 50 s |
| `ecpp-paper test4` | Test 4: PP, DPP, ungated ECPP, and ECPP at representative initial conditions | `sim_test4_results_{straight,arc}_display.tex`, `sim_test4_metrics.json`, `sim_test4_by_condition/*.pdf` | 40 s |
| `ecpp-paper all` | all of the above (plus `figure1`, a simulated PP/ECPP overview) | | about 1.5 min |

\* wall time on a 12-thread desktop CPU (Test 3 with `--threads 12`).

The tables (`.tex`) and metric files (`.json`) are deterministic and
reproduce the paper's numbers exactly; the figures differ only in PDF
metadata. Test 3 integrates 22 × 72,360 trajectories in double precision
without fast-math; its settled counts reproduce exactly with GCC on x86-64,
and other compilers may move a handful of points on the settling boundary.
A complete `sim_test3_sweep/` directory is reused on later runs.
`docs/paper_tests.md` lists the frozen conditions of every test.

## Development

```bash
python -m pip install -e .[dev]     # or: uv sync --dev
pytest -q
```

## Layout

```text
src/ecpp/            controllers/, simulation/, evaluation/, visualization/, config
src/ecpp/paper/      the test generators behind ecpp-paper (Test 3 kernel: phase_plane_kernel.cpp)
tests/               pytest suite
docs/paper_tests.md  frozen test conditions
```

## License

MIT, see `LICENSE`.

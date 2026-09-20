# API Notes

Key modules of the `ecpp` package:

- `ecpp.config`: `EcppConfig` (controller parameters) and YAML loading.
- `ecpp.controllers`: PP (`pure_pursuit`), DPP (`dpp`), and ECPP (`ecpp`)
  curvature laws, and the gate functions (`gates`).
- `ecpp.lookahead`, `ecpp.paths`, `ecpp.geometry`: path projection, lookahead
  point selection, and reference-path generators.
- `ecpp.simulation`: fixed-speed simulation loop.
- `ecpp.evaluation`: path-tracking metrics and table writers.
- `ecpp.visualization`: plots and animation helpers.
- `ecpp.paper`: the paper generators behind `ecpp-paper` (`figure1`,
  `ieee_access` for the shared engine and Tests 1 and 4, `test2_preview`,
  `test3_phase_plane` with `phase_plane_kernel.cpp`, and `tracking` with
  `simulate_fixed_speed`).

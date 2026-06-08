# API Notes

Public code should use the `ecpp` package, not the legacy source repository.

Key modules:

- `ecpp.config`: YAML loading and controller configuration.
- `ecpp.controllers`: PP, DPP, ECPP curvature laws.
- `ecpp.simulation`: fixed-speed simulation loop.
- `ecpp.evaluation`: path-tracking metrics and table writers.
- `ecpp.visualization`: plots and animation helpers.

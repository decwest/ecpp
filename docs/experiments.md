# Experiments

The main manuscript experiment is:

```bash
uv run python experiments/run_access_fixed_speed.py --config configs/experiments/access_test1_fixed_speed.yaml
```

Default conditions:

- `v_max = 0.50 m/s`
- `omega_max = 1.00 rad/s`
- `Lshort = 0.50 m`
- `Llong = 1.20 m`
- `e_y(0) = 0.10 m`
- `e_psi(0) = 15 deg`
- `rho = {1.0, 1.5, 2.0}`
- `zeta = {0.707, 1.0, 1.4}`

Outputs include CSV metrics, time-series CSV, NPZ trajectories, TeX tables,
PDF/PNG plots, and optional animations.

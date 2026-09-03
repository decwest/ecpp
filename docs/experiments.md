# Paper experiments

The canonical IEEE Access simulations are generated with:

```bash
uv run python -m ecpp.paper ieee-access --out-root /path/to/paper
```

Pass `--apply` only when the generated assets should be written below
`generated/`; otherwise they are written below `generated_preview/`.

Common model conditions:

- control rate: `30 Hz`
- forward speed: `0.50 m/s`
- additive gain regularizer: `v_epsilon = 0.05 m/s`
- instantaneous state-update clip: `|omega| <= 1.50 rad/s`
- no acceleration or velocity-smoother model
- raw requested angular velocity is used for curvature, saturation, and rate
  metrics
- continuous arc-length carrot and nominal `L_d` in the intrinsic gains

Test 1 (full design-parameter grid, frozen 2026-07-16):

- single fixed initial condition `(e_y(0), e_psi(0)) = (0.15 m, 0 deg)`
  (gate fully on for both lookaheads: `(0.15/0.5)^2 = 0.09 < eps_on = 0.10`)
- full factorial `omega_n x zeta x L_d = 3 x 3 x 2 = 18` cells on the
  straight path:
  - `L_d = {1.0, 0.5} m`
  - `zeta = {1/sqrt(2), 1, sqrt(2)}`
  - common absolute axis `omega_n = {0.777817, 1.132872, 1.555635} rad/s`;
    every value is named: PP-equivalent `omega_n_PP_cfg(1.0)`, the rate-bound
    design limit `omega_n_max(1.0)`, and PP-equivalent `omega_n_PP_cfg(0.5)`
    (`omega_n_PP_cfg = sqrt(2) (v + v_epsilon) / L_d`)
- out-of-bound cells are kept and labeled (only `L_d=1.0`, `1.556 rad/s`);
  the `(lambda, zeta) = (1, 1/sqrt(2))` cells are exact PP negative controls
- pre-declared selection feeding Test 2 and the hardware ECPP: among in-bound
  cells minimize `T_s^2%`, ties broken by `M_os`, then IAE
  -> `(omega_n, zeta, L_d) = (1.555635 rad/s, 1, 0.5 m)`

Test 2 (method and gate comparison):

- `L_d = 0.5 m`, `omega_n = 1.555635 rad/s`, `zeta = 1` (the selected cell)
- methods: PP, DPP (`L_1=L_d`, `L_2=2L_d`), ECPP without gate, ECPP
- paths: straight and a smooth radius-`3 m` arc
- one-sided full initial-condition grid
  `{0, 0.15, 1.0 m} x {0, -30, -90 deg} minus (0, 0)` = 8 conditions;
  `e_y(0)=0` rows leave the initial-error-normalized transient metrics
  (`T_r`, `T_s`, `M_os`) undefined and are reported as `--`
- 4 methods x 8 conditions x 2 paths = 64 runs
- no discontinuous-corner path

Unified metric vocabulary (chapters 5 and 6): MAE lateral / heading error,
`T_r` (90->10%), `T_s` (2% band in simulation, 10% band on hardware),
`M_os`, `T_m` (Test 2 / hardware only), and the raw curvature-command
maximum `kappa_max` against the budget line `omega_max / v = 3.0 1/m`.
Saturation ratios and angular-rate percentiles stay in the JSON and prose.

The general-purpose config-driven experiments under `experiments/` remain
development tools; they are not the source of the manuscript's frozen Test 1
and Test 2 conditions.

# Paper experiments

The canonical IEEE Access simulations are generated with:

```bash
uv run python -m ecpp.paper ieee-access --out-root /path/to/ECPP_ACCESS/manuscript
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

Test 3 (preview range versus local response, straight-arc-straight path):

- generated with `uv run python -m ecpp.paper test3-preview --out-root ...`
- path: 8 m straight (the Test-1 straight) -> left arc `R = 3 m` over 90 deg
  without a transition curve (the Test-2 radius) -> 6 m straight; the
  evaluation ends 4 m after the arc exit (`s_entry = 8.0 m`,
  `s_exit = 8 + 3 pi / 2 = 12.712 m`, `goal = 16.712 m`)
- initial condition `(e_y(0), e_psi(0)) = (0.15 m, 0 deg)` (the Test-1
  condition), so the recovery metrics over `s in [0, 6 m]` reproduce the
  Test-1 grid cells and are asserted to match
- arms: PP and ECPP at `L_d = {0.5, 1.0} m`; ECPP uses the hardware design
  point `omega_n = omega_n_max(1.0) = 1.132872 rad/s`, `zeta = 1` for both
  lookaheads (inside the rate bound for both; `lambda = 0.73` at 0.5 m)
- metrics: recovery `T_r`, `T_s^2%`, `M_os`; steering lead `d_lead`
  (first `|kappa_des| >= 0.1 / R` before the entry, searched from 6.5 m);
  entry in-cut `e_y,in = max e_y` on `[entry - 1.5, entry + 2.0] m`
  (left turn: inside is `+e_y`); exit out-flow `e_y,out = min e_y` on
  `[exit - 1.0, exit + 3.0] m`; `max |d omega_raw / dt|` over the transition;
  saturation ratio; the zero-error feedforward `kappa_prev(s)` per `L_d`
- 2 methods x 2 lookaheads = 4 runs

Unified metric vocabulary (chapters 5 and 6): MAE lateral / heading error,
`T_r` (90->10%), `T_s` (2% band in simulation, 10% band on hardware),
`M_os`, `T_m` (Test 2 / hardware only), and the raw curvature-command
maximum `kappa_max` against the budget line `omega_max / v = 3.0 1/m`.
Saturation ratios and angular-rate percentiles stay in the JSON and prose.

The general-purpose config-driven experiments under `experiments/` remain
development tools; they are not the source of the manuscript's frozen Test 1
and Test 2 conditions.

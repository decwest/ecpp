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
- Test 1 is a parameter study only; it does not select the Test-2 operating
  point (the earlier pre-declared selection rule was retired on 2026-09-08)

Test 2 (method and gate comparison, redesigned 2026-09-08):

- operating point shared with the hardware experiments and Test 3:
  `L_d = 1.0 m`, `omega_n = omega_n_max(1.0 m) = 1.132872 rad/s`, `zeta = 1`
  (the rate-bound design limit with critical damping), so
  `dK_y = 2.24 1/m^2` and `dK_theta = 2.12 1/m`
- methods: PP, DPP, ECPP without gate, ECPP
- DPP follows Wang and Mouri (Trans. JSME 2025): preview points on the
  vehicle axis at `L_1 = 2 L_d = 2.0 m` and `L_2`, lateral deviations
  `e_p,i` of those points from the path, `kappa = -(a1 e_p1 + a2 e_p2)`;
  `L_2 = 1.429 m`, `a1 = -3.404`, `a2 = 7.647 1/m^2` follow from the three
  design conditions (natural frequency, damping, and the constant-curvature
  condition `a1 L1^2 + a2 L2^2 = 2`, paper eq. (19)).  On a straight path
  the law is exactly `-K_y e_y - K_theta sin(e_psi)`.
- paths: straight 8 m (eval 6 m) and a right-turning arc `R = 3 m`
  (positive `e_y` is the outside), control path 180 deg, evaluation 135 deg
- one-sided initial-condition grid
  `{0, 0.30, 2.0, 3.0 m} x {0, -90 deg} minus (0, 0)` = 7 conditions:
  0.30 m keeps the gate open (`sigma = 0.99`, the hardware local amplitude),
  2.0 m starts with the gate closed (`|e_y|/L_d = 2`), and 3.0 m lies beyond
  the distance at which the ungated linear laws stay saturated
  (`max_theta omega_raw < -omega_max` for `e_y > 1.68 m` (DPP) and
  `> 2.45 m` (ECPP without gate)) and circle with radius `v / omega_max`
- 4 methods x 7 conditions x 2 paths = 56 runs
- table cells: `--` = undefined by construction (`e_y(0) = 0` rows leave
  `T_r`, `T_s`, `M_os` undefined), `n/r` = not reached inside the evaluation
  interval; the saturation ratio is a table column
- response panels are clipped to the evaluation interval (time axes end at
  the largest completion time among the methods that completed it)

Test 3 (preview range versus local response, straight-arc-straight path,
redesigned 2026-09-08):

- generated with `uv run python -m ecpp.paper test3-preview --out-root ...`
- path: 3 m straight -> left arc `R = 3 m` over 90 deg without a transition
  curve (the Test-2 radius) -> 6 m straight; the evaluation ends 4 m after
  the arc exit (`s_entry = 3.0 m`, `s_exit = 3 + 3 pi / 2 = 7.712 m`,
  `goal = 11.712 m`)
- initial condition `(e_y(0), e_psi(0)) = (0, 0)`: every deviation is forced
  by the curvature transitions
- arms: PP and ECPP at `L_d = {0.5, 1.0} m`; ECPP uses the Test-2 design
  point `omega_n = omega_n_max(1.0) = 1.132872 rad/s`, `zeta = 1` for both
  lookaheads (inside the rate bound for both; `lambda = 0.73` at 0.5 m)
- metrics: steering lead `d_lead` (first `|kappa_des| >= 0.1 / R` before the
  entry, searched from 1.0 m); entry in-cut `e_y,in = max e_y` on
  `[entry - 0.5 m, exit]` (left turn: inside is `+e_y`) and the recovery
  time `T_rec,in` from that peak until `|e_y|` falls to 10 % of it (before
  the exit); exit out-flow `e_y,out = min e_y` on `[exit - 0.5 m, goal]` and
  `T_rec,out` likewise; `kappa_max` and the saturation ratio stay in the
  JSON; the zero-error feedforward `kappa_prev(s)` per `L_d` is stored for
  the figure
- 2 methods x 2 lookaheads = 4 runs

Unified metric vocabulary (chapters 5 and 6): MAE lateral / heading error,
`T_r` (90->10%), `T_s` (2% band in simulation, 10% band on hardware),
`M_os`, `T_m` (hardware only), the raw curvature-command maximum
`kappa_max` against the budget line `omega_max / v = 3.0 1/m`, and the
saturation ratio (Test-2 table column).  Angular-rate percentiles stay in
the JSON and prose.

The general-purpose config-driven experiments under `experiments/` remain
development tools; they are not the source of the manuscript's frozen Test 1
and Test 2 conditions.

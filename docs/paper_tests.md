# Paper tests

Frozen conditions of the simulation tests of the ECPP paper (Section VI),
numbered as in the paper.  Each test is one `ecpp-paper` sub-command:

```bash
ecpp-paper test1   # (omega_n, zeta, L_d) sweep
ecpp-paper test2   # preview range versus local response
ecpp-paper test3   # phase-plane sweep of initial conditions
ecpp-paper test4   # method comparison at representative initial conditions
ecpp-paper all     # all four tests and the simulated PP/ECPP overview (figure1)
```

Outputs go below `generated_preview/` (or `generated/` with `--apply`) of the
current directory or of `--out-root`.

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
  - common absolute axis `omega_n = {0.707107, 1.029884, 1.414214} rad/s`
    (paper basis since 2026-09-12: the natural frequency of the local
    second-order model at the nominal speed `v0`); every value is named:
    PP-equivalent `omega_n_PP(1.0)`, the reference design value
    `omega_n_ref(1.0)`, and PP-equivalent `omega_n_PP(0.5)`
    (`omega_n_PP = sqrt(2) v0 / L_d`)
  - the controller/plugin computes its gains with `v_g = v0 + v_epsilon`, so
    the generator hands it `configured_omega_n = omega_n (v0 + v_epsilon)/v0`
    (`{0.778, 1.133, 1.556} rad/s`); the realized `K_y`, `K_theta` are the
    paper's
- out-of-bound cells are kept and labeled (only `L_d=1.0`, `1.414 rad/s`);
  the `(lambda, zeta) = (1, 1/sqrt(2))` cells are exact PP negative controls
- Test 1 is a parameter study only; it does not select the Test-4 operating
  point (the earlier pre-declared selection rule was retired on 2026-09-08)

Test 4 (method and gate comparison, redesigned 2026-09-08):

- operating point shared with the hardware experiments and Test 2:
  `L_d = 1.0 m`, `omega_n = omega_n_ref(1.0 m) = 1.029884 rad/s`, `zeta = 1`
  (the reference design value with critical damping), so
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

Test 2 (preview range versus local response, straight-arc-straight path,
redesigned 2026-09-12 around the hardware-route corner):

- generated with `ecpp-paper test2`
- path: 2 m straight -> left arc `R = 1.0 m` over 90 deg without a transition
  curve (the corner radius of the hardware route) -> 8 m straight; the
  evaluation ends 6 m after the arc exit (`s_entry = 2.0 m`,
  `s_exit = 2 + pi / 2 = 3.571 m`, `goal = 9.571 m`)
- initial condition `(e_y(0), e_psi(0)) = (0, 0)`: every deviation is forced
  by the curvature transitions
- arms: PP and ECPP at `L_d = {0.5, 1.0} m`; ECPP uses the Test-4 design
  point `omega_n = omega_n_ref(1.0) = 1.029884 rad/s`, `zeta = 1` for both
  lookaheads
- metrics (the common chapter-5 set, reference amplitude = the peak after the
  curvature change): entry in-cut `e_y,in = max e_y` on `[entry - 0.5 m, exit]`
  (left turn: inside is `+e_y`); exit out-flow `e_y,out = min e_y` on
  `[exit - 0.5 m, goal]`; `T_s^2%` from the exit peak until `|e_y|` enters and
  stays within 2 % of it up to the goal; `bar_e_y`, `bar_e_theta`; `N_zc`
  after the exit peak (1 mm dead band); `kappa_max`; the saturation ratio
  and the steering lead `d_lead` (first `|kappa_des| >= 0.1 / R`) stay in the
  JSON; the zero-error feedforward `kappa_prev(s)` per `L_d` is stored for
  the figure
- 2 methods x 2 lookaheads = 4 runs; with `L_d = 1.0 m` the robot and the
  carrot are both on the 1.57 m arc for only 0.57 m, so the corner is one
  blended transient (the entry recovery is therefore not reported)

Test 3 (convergence from large tracking errors; phase-plane sweep on an
infinite straight path, frozen 2026-09-14):

- generated with `ecpp-paper test3` (C++ kernel `phase_plane_kernel.cpp`,
  compiled on first use; `--threads N`)
- methods at `L_d = 1.0 m`, `omega_n = omega_n_ref(1.0) = 1.029884 rad/s`,
  `zeta = 1`: PP, DPP, ECPP without gate, ECPP; ECPP additionally at every
  one of the 18 Test-1 settings; PP additionally at `L_d = 0.5 m`
  (22 configurations)
- grid: `e_y(0) in [-10, 10] m` in 0.1 m steps (201 values) x
  `e_theta(0) in [-180, 179] deg` in 1 deg steps (360 values) = 72,360
  initial conditions per configuration; horizon 120 s
- settled: `|e_y| <= 0.02 m` and `|e_theta| <= 1 deg` held from some time on
  until the end of the horizon and for at least 5 s; the settling time is the
  first time of that interval; observation times 30, 60, 120 s
- the points unsettled at 120 s are re-simulated for 600 s (all methods) and,
  for PP and ECPP, the points still unsettled at 600 s for 3600 s
  (`sim_test3_sweep/horizon_extension.csv`)
- agreement with PP: percentage of grid points where ECPP and PP at the same
  `L_d` share the settled/unsettled classification
- the raw sweep (`sim_test3_sweep/data/*.npz`, one float32 record of 14
  fields per initial condition) is reused when it is complete

Unified metric vocabulary (Sections VI and VII): MAE lateral / heading error,
`T_r` (90->10%), `T_s` (2% band in simulation, 10% band on hardware),
`M_os`, `T_m` (hardware only), the raw curvature-command maximum
`kappa_max` against the budget line `omega_max / v = 3.0 1/m`, and the
saturation ratio (Test-4 table column).  Angular-rate percentiles stay in
the JSON and prose.

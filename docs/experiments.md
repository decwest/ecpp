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

Test 1a (speed shaping):

- `L_d = 1.0 m`, `(e_y(0), e_psi(0)) = (0.30 m, 0 deg)`
- `zeta = 1/sqrt(2)`
- `lambda = omega_n / omega_n_PP_cfg = {0.75, 1, 1.25, lambda_max}`
- `omega_n_PP_cfg = sqrt(2) (v + v_epsilon) / L_d`
- `lambda_max` follows from the configured-omega rate bound at the gate
  envelope and the `1.5 rad/s` rate limit

Test 1b (damping shaping):

- `L_d = 0.5 m`, `(e_y(0), e_psi(0)) = (0.15 m, -30 deg)`
- `omega_n = sqrt(2) (v + v_epsilon) / L_d`
- `zeta = {1/sqrt(2), 1, sqrt(2)}`

Test 2 (method and gate comparison):

- `L_d = 0.5 m`, `omega_n = sqrt(2) (v + v_epsilon) / L_d`,
  `zeta = 1`
- methods: PP, DPP (`L_1=L_d`, `L_2=2L_d`), ECPP without gate, ECPP
- paths: straight and a smooth radius-`3 m` arc
- one-sided conditions: `(0.15 m, 0 deg)`, `(0.15 m, -30 deg)`,
  `(1.0 m, 0 deg)`, `(1.0 m, -90 deg)`
- no discontinuous-corner path

The general-purpose config-driven experiments under `experiments/` remain
development tools; they are not the source of the manuscript's frozen Test 1
and Test 2 conditions.

# Algorithm Notes

This package implements fixed-speed curvature controllers for path tracking.

The command interface is intentionally simple:

```text
v_cmd = v_max
omega_raw = kappa * v_max
omega_cmd = clip(omega_raw, -omega_max, omega_max)
```

Implemented curvature laws:

- PP: single lookahead point geometry.
- DPP: two preview points with desired natural frequency and damping ratio.
- ECPP without gate: ECPP compensation with sigma fixed to one.
- ECPP: ECPP compensation with a state-dependent gate.

For DPP, the first preview distance is matched to the PP/ECPP lookahead:

```text
v_gain = v_max
T_p1 = L_d / v_max
```

If the derived second preview time is not positive or the gain equations are
singular, that DPP design point is omitted from the experiment results.

For ECPP, the state-dependent gate follows the relative linearization error
rates used in the paper:

```text
epsilon_y = (e_y / L_d)^2
epsilon_psi = |e_psi - sin(e_psi)| / |sin(e_psi)|
```

Both components use the same on/off error-rate thresholds, `error_on = 0.10`
and `error_off = 0.50` by default. The final gate is
`sigma = sigma_y(epsilon_y) * sigma_psi(epsilon_psi)`.

The package does not implement velocity scheduling, acceleration-window search,
or non-differential platform control.

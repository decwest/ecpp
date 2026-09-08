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

DPP follows Wang and Mouri (Trans. JSME, 2025).  The two preview points lie
on the vehicle axis, `L_1 = dpp_far_factor * L_d` (default `2 L_d`) and `L_2`
ahead of the robot; `e_p,i` is the left-positive lateral deviation of each
point from the reference path (its own closest path location), and

```text
kappa = -(a_1 e_p1 + a_2 e_p2)
a_1 + a_2             = K_y        K_y     = (omega_n / v_gain)^2
a_1 L_1 + a_2 L_2     = K_theta    K_theta = 2 zeta omega_n / v_gain
a_1 L_1^2 + a_2 L_2^2 = 2          (constant-curvature condition, paper eq. 19)
v_gain = v_max + v_epsilon
```

The third condition fixes `L_2 = (2 - L_1 K_theta) / (K_theta - L_1 K_y)`.
On a straight path `e_p,i = e_y + L_i sin(e_psi)` exactly, so the law reduces
to `-K_y e_y - K_theta sin(e_psi)`.  If the derived near preview distance is
not positive or the equations are singular, that DPP design point is omitted
from the experiment results.

For ECPP, the state-dependent gate follows the relative linearization error
rates used in the paper:

```text
epsilon_y = (e_y / L_d)^2
epsilon_psi = |e_psi - sin(e_psi)| / |sin(e_psi)|
```

Both components use the same on/off error-rate thresholds, `error_on = 0.10`
and `error_off = 0.50` by default. Gate modes (`gate.mode` in the config):

- `ey_only` (default, adopted design): `sigma = sigma_y(epsilon_y)`. The
  heading channel needs no gate because the compensation already uses the
  bounded `sin(e_psi)` form, which matches the exact PP curvature's heading
  dependence; the model error that must be gated grows with `e_y` only.
- `sigmoid` (legacy product gate): `sigma = sigma_y(epsilon_y) * sigma_psi(epsilon_psi)`.
  Kept for ablation; it disables heading damping in the small-`e_y` /
  large-`e_psi` regime where damping helps most.
- `always_on` / `off`: ablation modes (`sigma = 1` / `sigma = 0`).

The package does not implement velocity scheduling, acceleration-window search,
or non-differential platform control.

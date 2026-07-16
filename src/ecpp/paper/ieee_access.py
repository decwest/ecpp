"""Chapter-5 simulation generator for the IEEE Access ECPP paper.

Rebuilt from the ancestor engine ``generate_fixed_speed_pp_dpp_ecpp.py`` and
the chapter-5 metric definitions in ``generate_hw_exp1_outputs.py``.

Engine (frozen experiment values)
---------------------------------
* Pure-python unicycle, 30 Hz exact constant-twist update, no LPF (tau = 0).
* v0 = 0.5 m/s and the instantaneous state-update clip is fixed at
  +/-1.5 rad/s.  There is no acceleration or velocity-smoother model.
* Test 1a fixes L_d=1.0 m and sweeps configured omega_n relative to the exact
  PP-equivalent value.  Test 1b fixes L_d=0.5 m and sweeps zeta about the exact
  PP-equivalent gains.  Test 2 uses L_d=0.5 m, zeta=1, and the configured
  PP-equivalent omega_n.
* Controllers: PP, DPP, ECPP w/o gate (sigma == 1), ECPP (gated).
* ECPP law  kappa_des = kappa_PP - sigma(e_y) * (dK_y e_y + dK_theta sin e_theta)
      dK_y     = K_y - 2/L_d^2 ,  K_y     = (omega_n / (|v|+0.05))^2
      dK_theta = K_theta - 2/L_d,  K_theta = 2 zeta omega_n / (|v|+0.05)
* Gate: descending sigmoid of eps_y = (e_y / L_d)^2 ONLY (heading independent),
  with eps_on = 0.10, eps_off = 0.50 (residual p = 0.01). This matches the
  current paper (the gate does not depend on heading error).
* DPP: two-preview curvature a1 yb1 + a2 yb2 with L1 = L_d, L2 = 2 L_d,
  coefficients matched to the target gains (K_y, K_theta).
* Paths: straight 8 m (eval goal s = 6.0 m); arc R = 3.0 m over 180 deg with the
  first 90 deg (eval goal s = R*pi/2) evaluated so the lookahead never pins at
  the path end.
* PP, DPP, and ECPP all use the package's shared continuous ``PathProjection``
  and arc-length-interpolated carrots.  Stored path-pose orientation is not
  used by the controller.

Metrics (chapter-5, noise-free -> no smoothing, deterministic -> no +/- std)
---------------------------------------------------------------------------
Over the eval interval (motion start .. s reaches the eval goal, s in [0, goal]):
  bar_e_y, bar_e_theta [deg], T_r (10->90% progress), T_s (2% of E0 band,
  enters-and-stays), M_os (opposite-side excursion after first zero crossing),
  T_m (motion start -> within 0.02 m of the eval goal),
  kappa_max (max |kappa_des|, unclipped),
  sat_ratio (fraction of steps with |omega_raw| >= omega_max before clip).
Failure/robustness fields: goal_reached (goal position reached within TMAX),
  t_end (last simulated time), max_abs_ey / final_abs_ey (divergence probes).

Deliverables (see the two drivers below). Writes to ``generated_preview/`` by
default; ``--apply`` switches to ``generated/``. Does not edit sections/*.tex.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..controllers.gates import gate_abs
from .tracking import simulate_fixed_speed


ROOT = Path.cwd()

# ---------------------------------------------------------------------------
# Engine parameters (current paper values)
# ---------------------------------------------------------------------------
V0 = 0.50            # m/s forward speed
V_EPSILON = 0.05     # m/s additive low-speed regularization
OMEGA_MAX = 1.5      # rad/s instantaneous state-update clip
DT = 1.0 / 30.0      # s control period (30 Hz)
LD = 1.0             # m mutable lookahead used by one simulation run
LD_DPP2 = 2.0        # m second DPP preview distance (= 2 L_d)
EPS_ON = 0.10
EPS_OFF = 0.50
GATE_P = 0.01        # sigmoid residual
TMAX = 60.0          # s hard sim cap
CARROT_RULE = "arc"  # fixed: nominal arc-length lookahead

STRAIGHT_LEN = 8.0
STRAIGHT_GOAL = 6.0
ARC_R = 3.0
ARC_ANGLE = math.pi          # 180 deg control path
ARC_GOAL = ARC_R * math.pi / 2.0   # evaluate first 90 deg

# secondary acceleration bound (report only): assumed platform angular
# acceleration budget; illustrative, plays no role in the design grid.
ALPHA_MAX = 3.0      # rad/s^2

PP_OMEGA_N = math.sqrt(2.0) * (V0 + V_EPSILON) / LD
# reproduction/legacy test-1 operating point
REPRO_OMEGA_N = 2.12
REPRO_ZETA = 1.0


# ---------------------------------------------------------------------------
# Theory constants (documented functions)
# ---------------------------------------------------------------------------
def gate_envelope_error_bound(eps_off=None, ld=None):
    """Gate-envelope steady error bound  ebar_g = sqrt(eps_off) * L_d."""
    eps_off = EPS_OFF if eps_off is None else eps_off
    ld = LD if ld is None else ld
    return math.sqrt(eps_off) * ld


def omega_n_max(zeta, ebar, sbar=0.0, kappa_ref=0.0,
                v=None, omega_max=None, v_epsilon=V_EPSILON):
    """Rate-based design upper limit

        omega_n_max(zeta; ebar, sbar)
            = (v_g/ebar) * (-zeta sbar + sqrt(zeta^2 sbar^2 + ebar w_budget / v))

    with v_g = |v| + v_epsilon and
    w_budget = omega_max - v |kappa_ref| (straight: kappa_ref = 0).
    This is a bound on the configured omega_n parameter."""
    v = V0 if v is None else v
    omega_max = OMEGA_MAX if omega_max is None else omega_max
    v_gain = abs(v) + v_epsilon
    w_budget = omega_max - v * abs(kappa_ref)
    disc = zeta * zeta * sbar * sbar + ebar * w_budget / v
    return (v_gain / ebar) * (-zeta * sbar + math.sqrt(disc))


def omega_n_accel_bound(
    zeta, ebar, alpha_max=ALPHA_MAX, v=V0, v_epsilon=V_EPSILON
):
    """Secondary configured-omega_n acceleration bound.

    omega_n <= (alpha_max v_g^3 / (2 zeta v^2 ebar))^(1/3).
    Report only; no role in the design grid."""
    v_gain = abs(v) + v_epsilon
    return (
        alpha_max * v_gain**3 / (2.0 * zeta * v * v * ebar)
    ) ** (1.0 / 3.0)


EBAR_G = gate_envelope_error_bound()                 # 0.7071 m
OMEGA_N_MAX = omega_n_max(1.0, EBAR_G, sbar=0.0)


def configure(omega_max=None, ld=None, tmax=None, carrot_rule=None):
    """Reconfigure L_d, horizon, and representative conditions.

    ``omega_max`` remains as a compatibility argument but accepts only the
    fixed 1.5 rad/s state-update clip. ``LD_DPP2`` tracks ``2*L_d`` per the DPP
    definition.
    """
    global OMEGA_MAX, LD, LD_DPP2, TMAX, CARROT_RULE
    global PP_OMEGA_N, EBAR_G, OMEGA_N_MAX
    if omega_max is not None:
        if not math.isclose(float(omega_max), 1.5, abs_tol=1e-12):
            raise ValueError(
                "paper simulation state updates use a fixed +/-1.5 rad/s clip"
            )
        OMEGA_MAX = 1.5
    if ld is not None:
        LD = float(ld)
    if tmax is not None:
        TMAX = float(tmax)
    if carrot_rule is not None:
        if carrot_rule != "arc":
            raise ValueError(f"unknown carrot rule: {carrot_rule!r}")
        CARROT_RULE = carrot_rule
    LD_DPP2 = 2.0 * LD
    PP_OMEGA_N = math.sqrt(2.0) * (V0 + V_EPSILON) / LD
    EBAR_G = gate_envelope_error_bound()
    OMEGA_N_MAX = omega_n_max(1.0, EBAR_G, sbar=0.0)


def worst_effective_error(ld=None, ebar=None):
    """Lateral error inside the gate envelope that maximizes the effective
    (gated) compensation magnitude sigma(e) * e. This is the closed-loop
    worst case the rate bound protects: at the mild local step (e=0.3 m) the
    error is too small and at large e the gate is closed, so this intermediate
    error is where an over-designed omega_n actually saturates the command."""
    ld = LD if ld is None else ld
    ebar = EBAR_G if ebar is None else ebar
    es = np.linspace(1e-3, ebar, 2000)
    prod = np.array([_gate_scalar((e / ld) ** 2) * e for e in es])
    return float(es[int(np.argmax(prod))])


def _gate_scalar(eps):
    return sigmoid_desc(eps)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
class StraightPath:
    key = "straight"
    name = "Straight path"

    def __init__(self, length=STRAIGHT_LEN, goal=STRAIGHT_GOAL):
        self.length = length
        self.goal = goal

    def point(self, s):
        s = float(np.clip(s, 0, self.length))
        return np.array([s, 0.0])

    def theta(self, s):
        return 0.0

    def curvature(self, s):
        return 0.0

    def closest_s(self, p):
        return float(np.clip(p[0], 0, self.length))

    def sample(self, n=400):
        ss = np.linspace(0, self.goal, n)
        return np.array([self.point(s) for s in ss])

    def control_polyline(self):
        return np.array([[0.0, 0.0], [self.length, 0.0]], dtype=float)


class ArcPath:
    key = "arc"
    name = "Constant-curvature arc"

    def __init__(self, R=ARC_R, angle=ARC_ANGLE, goal=ARC_GOAL):
        self.R = R
        self.angle = angle
        self.length = R * angle
        self.goal = goal

    def point(self, s):
        s = float(np.clip(s, 0, self.length))
        a = s / self.R
        return np.array([self.R * math.sin(a), self.R * (1.0 - math.cos(a))])

    def theta(self, s):
        s = float(np.clip(s, 0, self.length))
        return s / self.R

    def curvature(self, s):
        return 1.0 / self.R

    def closest_s(self, p):
        v = np.array([p[0], p[1] - self.R])
        ang = math.atan2(v[1], v[0])
        a = ang + math.pi / 2.0
        return float(np.clip(self.R * a, 0, self.length))

    def sample(self, n=400):
        ss = np.linspace(0, self.goal, n)
        return np.array([self.point(s) for s in ss])

    def control_polyline(self):
        # The controller itself is sampling-density invariant; this resolution
        # only approximates the analytic circle supplied by this paper driver.
        count = max(2, int(math.ceil(self.length / 0.005)) + 1)
        ss = np.linspace(0.0, self.length, count)
        return np.array([self.point(s) for s in ss], dtype=float)


# ---------------------------------------------------------------------------
# Shared controller adapter
# ---------------------------------------------------------------------------
class Gains:
    """Configured gains recorded with a trace for analysis overlays."""

    def __init__(self, omega_n, zeta, v=None, ld=None, ld2=None):
        v = V0 if v is None else v
        ld = LD if ld is None else ld
        ld2 = LD_DPP2 if ld2 is None else ld2
        v_gain = abs(v) + V_EPSILON
        self.omega_n = omega_n
        self.zeta = zeta
        self.v_gain = v_gain
        self.Ky = (omega_n / v_gain) ** 2
        self.Kth = 2.0 * zeta * omega_n / v_gain
        self.Ky_pp = 2.0 / ld**2
        self.Kth_pp = 2.0 / ld
        self.dKy = self.Ky - self.Ky_pp
        self.dKth = self.Kth - self.Kth_pp
        self.L1 = ld
        self.L2 = ld2
        self.a1 = (self.Kth - self.Ky * self.L2) / (self.L1 - self.L2)
        self.a2 = self.Ky - self.a1


def sigmoid_desc(eps, eps_on=EPS_ON, eps_off=EPS_OFF, p=GATE_P):
    return gate_abs(eps, eps_on, eps_off, p)


def gate(ey, ld=None):
    """e_y-only gate (heading independent)."""

    ld = LD if ld is None else ld
    return sigmoid_desc((ey / ld) ** 2)


def simulate(path, method, omega_n, zeta, ey0, eth0):
    """Return the legacy table layout using the canonical package controller."""

    gains = Gains(omega_n, zeta)
    trace = simulate_fixed_speed(
        path=path.control_polyline(),
        method=method,
        lookahead_m=LD,
        omega_n=omega_n,
        zeta=zeta,
        e_y0=ey0,
        e_psi0=eth0,
        speed=V0,
        v_epsilon=V_EPSILON,
        omega_limit=OMEGA_MAX,
        dt=DT,
        t_max=TMAX,
        goal_arc_length=path.goal,
        goal_position=path.point(path.goal),
        goal_tolerance=0.02,
    )
    rows = np.column_stack(
        [
            trace.time,
            trace.pose,
            trace.path_s,
            trace.e_y,
            trace.e_psi,
            trace.curvature,
            trace.omega_cmd,
            trace.omega_raw,
            trace.sigma,
        ]
    )
    return rows, gains


# ---------------------------------------------------------------------------
# Metrics (chapter-5 definitions, no smoothing)
# ---------------------------------------------------------------------------
def rise_time(t_rel, e, e0):
    a = np.abs(e)
    i90 = np.where(a <= 0.9 * abs(e0))[0]
    i10 = np.where(a <= 0.1 * abs(e0))[0]
    if len(i90) == 0 or len(i10) == 0:
        return None
    return float(t_rel[i10[0]] - t_rel[i90[0]])


def settling_time(t_rel, e, band):
    if len(e) == 0:
        return None
    viol = np.where(np.abs(e) > band)[0]
    if len(viol) == 0:
        return float(t_rel[0])
    if viol[-1] == len(e) - 1:
        return None
    return float(t_rel[viol[-1] + 1])


def overshoot(e, e0):
    sgn = math.copysign(1.0, e0)
    crossed = np.where(sgn * e < 0.0)[0]
    if len(crossed) == 0:
        return 0.0
    return float(max(0.0, np.max(-sgn * e[crossed[0]:])))


def second_order_response(t, e0, wn, z):
    """Homogeneous response from gains produced by configured ``wn``/``z``.

    The legend continues to report the configured ``wn``.  The curve itself
    uses the actual ``K_y`` and ``K_theta`` after additive speed
    regularization, as required by the manuscript model.
    """
    return second_order_response_with_initial_rate(t, e0, 0.0, wn, z)


def second_order_response_with_initial_rate(t, e0, e_dot0, wn, z):
    """Linear response computed from the actual regularized gains.

    ``wn`` remains the configured value shown in legends.  The poles are
    computed from ``K_y`` and ``K_psi`` after applying ``|v|+epsilon``.
    """

    t = np.asarray(t, dtype=float)
    gains = Gains(wn, z)
    natural = abs(V0) * math.sqrt(gains.Ky)
    damping = abs(V0) * gains.Kth / (2.0 * natural)
    if damping < 1.0 - 1e-9:
        wd = natural * math.sqrt(1.0 - damping * damping)
        sine_coefficient = (e_dot0 + damping * natural * e0) / wd
        return np.exp(-damping * natural * t) * (
            e0 * np.cos(wd * t) + sine_coefficient * np.sin(wd * t)
        )
    if abs(damping - 1.0) <= 1e-9:
        return np.exp(-natural * t) * (
            e0 + (e_dot0 + natural * e0) * t
        )
    rt = natural * math.sqrt(damping * damping - 1.0)
    r1, r2 = -damping * natural + rt, -damping * natural - rt
    c1 = (e_dot0 - r2 * e0) / (r1 - r2)
    c2 = e0 - c1
    return c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)


def run_metrics(path, method, omega_n, zeta, ey0, eth0):
    """Simulate one run and return metrics + eval-sliced arrays."""
    arr, g = simulate(path, method, omega_n, zeta, ey0, eth0)
    t = arr[:, 0]
    s = arr[:, 4]
    ey = arr[:, 5]
    eth = arr[:, 6]
    kappa = arr[:, 7]
    omega_raw = arr[:, 9]

    e0 = ey0 if abs(ey0) > 1e-9 else 0.0
    emask = (s >= 0.0) & (s <= path.goal)
    te, eye, ethe = t[emask], ey[emask], eth[emask]
    ke, ore = kappa[emask], omega_raw[emask]
    oce = arr[emask, 8]
    sigmae = arr[emask, 10]
    t_rel = te - te[0] if len(te) else te

    m = {"omega_n": omega_n, "zeta": zeta, "ey0": ey0,
         "eth0_deg": round(math.degrees(eth0))}
    if len(te) == 0:
        return m, arr, g

    m["bar_e_y"] = float(np.mean(np.abs(eye)))
    m["bar_e_theta_deg"] = float(np.degrees(np.mean(np.abs(ethe))))
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0 on the paper-generation host
        integrate = np.trapz
    m["iae_e_y"] = float(integrate(np.abs(eye), te))
    m["kappa_max"] = float(np.max(np.abs(ke)))
    m["sat_ratio"] = float(np.mean(np.abs(ore) >= OMEGA_MAX))
    m["peak_omega_ratio"] = float(np.max(np.abs(ore)) / OMEGA_MAX)
    m["omega_raw_max"] = float(np.max(np.abs(ore)))
    m["omega_cmd_max"] = float(np.max(np.abs(oce)))
    m["omega_raw_rate_max"] = (
        float(np.max(np.abs(np.diff(ore)) / DT)) if len(ore) > 1 else 0.0
    )
    m["sigma_min"] = float(np.min(sigmae))
    m["sigma_max"] = float(np.max(sigmae))

    goal_xy = path.point(path.goal)
    goal_distances = np.linalg.norm(arr[:, 1:3] - goal_xy, axis=1)
    goal_idx = np.where(goal_distances <= 0.02)[0]
    m["T_m"] = float(t[goal_idx[0]] - t[0]) if len(goal_idx) else None
    m["goal_reached"] = bool(len(goal_idx))
    # The straight and arc control paths deliberately extend beyond the
    # reported interval so the carrot never pins at its endpoint.  Therefore
    # completing the evaluation interval means crossing its arc-length
    # section, not necessarily passing within 2 cm of that virtual point.
    eval_idx = np.where(s >= path.goal - 0.02)[0]
    m["evaluation_completed"] = bool(len(eval_idx) or len(goal_idx))
    if len(eval_idx):
        m["T_eval"] = float(t[eval_idx[0]] - t[0])
    elif len(goal_idx):
        m["T_eval"] = float(t[goal_idx[0]] - t[0])
    else:
        m["T_eval"] = None
    m["t_end"] = float(t[-1])
    m["max_abs_ey"] = float(np.max(np.abs(ey)))
    m["final_abs_ey"] = float(abs(ey[-1]))

    if abs(e0) > 1e-9:
        m["T_r"] = rise_time(t_rel, eye, e0)
        m["T_s"] = settling_time(t_rel, eye, 0.02 * abs(e0))
        m["T_s_10"] = settling_time(t_rel, eye, 0.10 * abs(e0))
        m["T_s_05"] = settling_time(t_rel, eye, 0.05 * abs(e0))
        m["T_s_02"] = m["T_s"]
        m["M_os"] = overshoot(eye, e0)
    else:
        m["T_r"] = None
        m["T_s"] = None
        m["T_s_10"] = None
        m["T_s_05"] = None
        m["T_s_02"] = None
        m["M_os"] = None
    return m, arr, g


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------
def f(x, d, none="--"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return none
    return f"{x:.{d}f}"


def cond_encode(ey0, eth0_deg):
    eyt = f"ey{ey0:.2f}".replace(".", "p")
    if eth0_deg == 0:
        et = "epsi0"
    else:
        et = f"epsim{abs(int(eth0_deg))}"
    return f"{eyt}_{et}"


# ---------------------------------------------------------------------------
# Legacy development preview (not called by the paper CLI)
#
# This preserves the pre-freeze bound-probing utility for diagnostics only.
# The manuscript source of truth is the frozen TEST 1 block below.
# ---------------------------------------------------------------------------
LEGACY_TEST1_LDS = (0.5, 1.0)
LEGACY_OMEGA_RATIOS = [0.4, 0.7, 1.0]
LEGACY_ZETAS = [0.707, 1.0, 1.5]
LEGACY_OOB_RATIO = 1.5
LEGACY_OOB_ZETA = 1.0
# initial conditions COMMON across L_d: local is gate-ON for every L_d
# (e_y <= sqrt(eps_on)*min L_d = 0.158 m), far is gate-OFF for every L_d
# (e_y >= sqrt(eps_off)*max L_d = 0.707 m).
LEGACY_COND_LOCAL = (0.15, 0.0)
LEGACY_COND_FAR = (1.0, math.radians(-90))


def legacy_test1_grid():
    """List of (omega_n, zeta, ratio, in_bound)."""
    combos = []
    for r in LEGACY_OMEGA_RATIOS:
        for z in LEGACY_ZETAS:
            combos.append((r * OMEGA_N_MAX, z, r, True))
    combos.append((
        LEGACY_OOB_RATIO * OMEGA_N_MAX,
        LEGACY_OOB_ZETA,
        LEGACY_OOB_RATIO,
        False,
    ))
    return combos


def legacy_select_from_rows(rows, e0):
    """Chapter-5 selection rule over metric rows measured at the local step:
    in-bound candidates with M_os <= 0.05 E0, minimum T_s, ties by kappa_max.
    Falls back to min T_s over all in-bound rows if none are feasible.
    Returns (selected_row, relaxed_flag)."""
    big = 1e9

    def ts_key(r):
        return (r["T_s"] if r.get("T_s") is not None else big, r["kappa_max"])

    inb = [r for r in rows if r["in_bound"]]
    feasible = [r for r in inb
                if r.get("M_os") is not None and r["M_os"] <= 0.05 * e0]
    if feasible:
        return min(feasible, key=ts_key), False
    return min(inb, key=ts_key), True


def legacy_write_test1_sweep_table(path_out, rows_by_ld, caption_cond):
    """rows_by_ld: {ld: (rows, omega_n_max)}. One merged booktabs table with a
    subheader block per L_d (mirrors the test-2 per-condition blocks)."""
    lines = [
        r"\begin{tabular}{@{}lllllllllll@{}}",
        r"\toprule",
        r"$\omega_n$ & $\zeta$ & $\omega_n/\omega_n^{\max}$ & $\bar e_y$ & "
        r"$\bar e_\theta$ & $T_r$ & $T_s$ & $M_{\rm os}$ & $T_m$ & "
        r"$\kappa_{\max}$ & sat.[\%] \\",
    ]

    def row_cells(r, dagger=False):
        return [
            f(r["omega_n"], 3) + (r"$^{\dagger}$" if dagger else ""),
            f(r["zeta"], 3), f(r["ratio"], 2),
            f(r["bar_e_y"], 3), f(r["bar_e_theta_deg"], 2),
            f(r.get("T_r"), 2), f(r.get("T_s"), 2), f(r.get("M_os"), 3),
            f(r.get("T_m"), 2), f(r["kappa_max"], 2),
            f(100 * r["sat_ratio"], 1),
        ]

    def row_marked(r, marker):
        cells = row_cells(r)
        cells[0] += marker
        return cells

    oob_ratios = set()
    for ld, (rows, wn_max) in sorted(rows_by_ld.items()):
        lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{11}{@{}l@{}}{$L_d=" + f"{ld:.2f}"
            + r"\,\mathrm{m}\ (\omega_n^{\max}=" + f"{wn_max:.3f}"
            + r"\,\mathrm{rad/s})$} \\")
        lines.append(r"\midrule")
        for r in [x for x in rows if x.get("is_pp")]:
            lines.append(" & ".join(row_marked(r, r"$^{\ddagger}$")) + r"\\")
        for r in [x for x in rows if x["in_bound"] and not x.get("is_pp")]:
            lines.append(" & ".join(row_cells(r)) + r"\\")
        for r in [x for x in rows if not x["in_bound"] and not x.get("is_pp")]:
            oob_ratios.add(r["ratio"])
            lines.append(" & ".join(row_marked(r, r"$^{\dagger}$")) + r"\\")
    oob_txt = ", ".join(f"{x:.3g}" for x in sorted(oob_ratios)) or "1.5"
    lines += [
        r"\bottomrule",
        r"\multicolumn{11}{@{}l@{}}{\footnotesize $^{\ddagger}$"
        r" PP reference (implicit local gains: "
        r"$\omega_{n,\rm PP}=\sqrt{2}v_0/L_d$, $\zeta=0.707$).}\\",
        r"\multicolumn{11}{@{}l@{}}{\footnotesize $^{\dagger}$"
        r" out-of-bound reference ($\omega_n=" + oob_txt
        + r"\,\omega_n^{\max}$).}\\",
        r"\end{tabular}",
        "",
    ]
    path_out.write_text("\n".join(lines), encoding="utf-8")


def legacy_write_test1_selection(path_out, sel):
    lines = [
        r"\begin{tabular}{@{}llllllll@{}}",
        r"\toprule",
        r"Criterion & $L_d$ & $\omega_n$ & $\zeta$ & "
        r"$\omega_n/\omega_n^{\max}$ & $T_s$ & $M_{\rm os}$ & "
        r"$\kappa_{\max}$ \\",
        r"\midrule",
    ]
    lines.append(" & ".join([
        r"selected ($M_{\rm os}\!\le\!0.05E_0$, min $T_s$)",
        f(sel["ld"], 2),
        f(sel["omega_n"], 3), f(sel["zeta"], 3), f(sel["ratio"], 2),
        f(sel.get("T_s"), 2), f(sel.get("M_os"), 3), f(sel["kappa_max"], 2),
    ]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path_out.write_text("\n".join(lines), encoding="utf-8")


def legacy_fig_test1_step_response(fig_dir, straight, tag=""):
    """(a) e_y across omega_n at zeta=1.0 (+OOB, analytic overlay),
    (b) e_y across zeta at the middle omega_n ratio, (c) commanded omega for
    the omega_n family with +/- omega_max lines. Local condition."""
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 6.0))
    ey0, eth0 = LEGACY_COND_LOCAL
    cmap = plt.get_cmap("viridis")

    # (a) omega_n family at zeta = 1.0
    ax = axes[0]
    for i, r in enumerate(LEGACY_OMEGA_RATIOS):
        wn = r * OMEGA_N_MAX
        _, arr, _ = run_metrics(straight, "ECPP", wn, 1.0, ey0, eth0)
        t, ey = arr[:, 0], arr[:, 5]
        col = cmap(i / (len(LEGACY_OMEGA_RATIOS) - 1) * 0.85)
        ax.plot(t, ey, color=col, lw=1.1,
                label=fr"${r:.1f}\,\omega_n^{{\max}}$")
        tt = np.linspace(0, t[-1], 400)
        ax.plot(tt, second_order_response(tt, ey0, wn, 1.0),
                color=col, lw=0.8, ls="--", alpha=0.75)
    # OOB
    wn = LEGACY_OOB_RATIO * OMEGA_N_MAX
    _, arr, _ = run_metrics(
        straight, "ECPP", wn, LEGACY_OOB_ZETA, ey0, eth0
    )
    ax.plot(arr[:, 0], arr[:, 5], color="crimson", lw=1.2,
            label=fr"${LEGACY_OOB_RATIO:.3g}\,\omega_n^{{\max}}\,^{{\dagger}}$")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_ylabel(r"$e_y$ [m]")
    ax.set_title(r"(a) $\omega_n$ sweep ($\zeta{=}1.0$)", fontsize=8)
    ax.legend(fontsize=5.5, loc="best", ncol=2, framealpha=0.85)

    # (b) zeta family at the middle omega_n ratio
    ax = axes[1]
    wn = LEGACY_OMEGA_RATIOS[-2] * OMEGA_N_MAX
    for i, z in enumerate(LEGACY_ZETAS):
        _, arr, _ = run_metrics(straight, "ECPP", wn, z, ey0, eth0)
        col = cmap(i / (len(LEGACY_ZETAS) - 1) * 0.85)
        ax.plot(arr[:, 0], arr[:, 5], color=col, lw=1.1,
                label=fr"$\zeta={z:.3g}$")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_ylabel(r"$e_y$ [m]")
    ax.set_title(fr"(b) $\zeta$ sweep ($\omega_n{{=}}{0.8 * OMEGA_N_MAX:.3f}$)",
                 fontsize=8)
    ax.legend(fontsize=6, loc="best", framealpha=0.85)

    # (c) commanded omega for the omega_n family (zeta = 1.0)
    ax = axes[2]
    for i, r in enumerate(LEGACY_OMEGA_RATIOS):
        wn = r * OMEGA_N_MAX
        _, arr, _ = run_metrics(straight, "ECPP", wn, 1.0, ey0, eth0)
        col = cmap(i / (len(LEGACY_OMEGA_RATIOS) - 1) * 0.85)
        ax.plot(arr[:, 0], arr[:, 8], color=col, lw=1.0,
                label=fr"${r:.1f}\,\omega_n^{{\max}}$")
    wn = LEGACY_OOB_RATIO * OMEGA_N_MAX
    _, arr, _ = run_metrics(
        straight, "ECPP", wn, LEGACY_OOB_ZETA, ey0, eth0
    )
    ax.plot(arr[:, 0], arr[:, 8], color="crimson", lw=1.1,
            label=fr"${LEGACY_OOB_RATIO:.3g}\,\omega_n^{{\max}}\,^{{\dagger}}$")
    ax.axhline(OMEGA_MAX, color="0.5", lw=0.7, ls=":")
    ax.axhline(-OMEGA_MAX, color="0.5", lw=0.7, ls=":")
    ax.set_ylabel(r"$\omega_{\rm cmd}$ [rad/s]")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_title(r"(c) commanded $\omega$ ($\zeta{=}1.0$)", fontsize=8)
    ax.legend(fontsize=5.5, loc="best", ncol=2, framealpha=0.85)

    for ax in axes:
        ax.grid(True, color="0.9", lw=0.5)
        ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"sim_test1_step_response{tag}.{ext}", dpi=300)
    plt.close(fig)


def legacy_fig_test1_bound_validation(
    fig_dir, rows_local, rows_far, rows_probe, e_star, tag=""
):
    """omega_n-zeta plane: grid points colored by measured sat_ratio; theory
    curves omega_n_max(zeta; ebar_g, sbar) for sbar in {0, 0.5}; the OOB point
    is circled. Three panels: (a) local step and (b) far capture show that the
    descending gate keeps even the OOB command unsaturated in these mild
    conditions, while (c) the gate-envelope worst-case step e=e_star exercises
    the compensation and makes the OOB point visibly saturate -- the situation
    the rate bound governs."""
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    zz = np.linspace(0.6, 1.6, 120)
    loc_lbl = (r"(a) local step (%.4g m, $%d^\circ$)"
               % (LEGACY_COND_LOCAL[0],
                  round(math.degrees(LEGACY_COND_LOCAL[1]))))
    far_lbl = (r"(b) far (%.4g m, $%d^\circ$)"
               % (LEGACY_COND_FAR[0],
                  round(math.degrees(LEGACY_COND_FAR[1]))))
    panels = [
        (rows_local, loc_lbl),
        (rows_far, far_lbl),
        (rows_probe, r"(c) envelope worst case ($e{=}%.2f$ m)" % e_star),
    ]
    all_sat = [100 * r["sat_ratio"] for rows, _ in panels for r in rows]
    vmax = max(1.0, max(all_sat))
    sc = None
    for ax, (rows, title) in zip(axes, panels):
        zs = [r["zeta"] for r in rows]
        ws = [r["omega_n"] for r in rows]
        sat = [100 * r["sat_ratio"] for r in rows]
        sc = ax.scatter(zs, ws, c=sat, cmap="viridis", vmin=0, vmax=vmax,
                        s=55, edgecolors="k", linewidths=0.5, zorder=3)
        for r in rows:
            # star = command actually reaches the clip (peak demand >= omega_max)
            if r["peak_omega_ratio"] >= 1.0:
                ax.scatter([r["zeta"]], [r["omega_n"]], marker="*", s=95,
                           color="crimson", edgecolors="k", linewidths=0.4,
                           zorder=5)
            if not r["in_bound"]:
                ax.scatter([r["zeta"]], [r["omega_n"]], s=170,
                           facecolors="none", edgecolors="crimson",
                           linewidths=1.6, zorder=4)
        ax.plot(zz, [omega_n_max(z, EBAR_G, 0.0) for z in zz],
                "b-", lw=1.1, label=r"$\omega_n^{\max}(\bar s{=}0)$")
        ax.plot(zz, [omega_n_max(z, EBAR_G, 0.5) for z in zz],
                "g--", lw=1.1, label=r"$\omega_n^{\max}(\bar s{=}0.5)$")
        ax.set_xlabel(r"$\zeta$")
        ax.set_title(title, fontsize=7.5)
        ax.grid(True, color="0.9", lw=0.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5.5, loc="upper right")
    axes[0].set_ylabel(r"$\omega_n$ [rad/s]")
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("sat. ratio [%]", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    # legend note for the saturation star
    axes[-1].scatter([], [], marker="*", s=95, color="crimson",
                     edgecolors="k", linewidths=0.4,
                     label=r"peak $|\omega|\geq\omega_{\max}$")
    axes[-1].legend(fontsize=5.0, loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"sim_test1_bound_validation{tag}.{ext}",
                    dpi=300)
    plt.close(fig)


def run_legacy_test1_preview(table_dir, fig_dir):
    """Development-only legacy bound probe; never called by the paper CLI.

    Sweep (omega_n, zeta) for every L_d in LEGACY_TEST1_LDS with initial
    conditions COMMON across L_d, write the merged tables, per-L_d figures,
    and the cross-L_d selection."""
    local_by_ld, far_by_ld = {}, {}
    per_ld_meta, all_local = {}, []
    for ld in LEGACY_TEST1_LDS:
        configure(ld=ld)
        tag = f"_ld{ld:g}".replace(".", "p")
        straight = StraightPath()
        e_star = worst_effective_error()
        rows_local, rows_far, rows_probe = [], [], []
        for wn, z, ratio, inb in legacy_test1_grid():
            ml, _, _ = run_metrics(
                straight, "ECPP", wn, z, *LEGACY_COND_LOCAL
            )
            mf, _, _ = run_metrics(
                straight, "ECPP", wn, z, *LEGACY_COND_FAR
            )
            mp, _, _ = run_metrics(straight, "ECPP", wn, z, e_star, 0.0)
            for m in (ml, mf, mp):
                m.update({"ratio": ratio, "in_bound": inb, "ld": ld})
            rows_local.append(ml)
            rows_far.append(mf)
            rows_probe.append(mp)
        # PP reference rows (baseline of each L_d block; implicit local gains
        # omega_n_PP = sqrt(2) v0 / L_d, zeta_PP = 0.707)
        zeta_pp = 1.0 / math.sqrt(2.0)
        ppl, _, _ = run_metrics(straight, "PP", PP_OMEGA_N, zeta_pp,
                                *LEGACY_COND_LOCAL)
        ppf, _, _ = run_metrics(straight, "PP", PP_OMEGA_N, zeta_pp,
                                *LEGACY_COND_FAR)
        for m in (ppl, ppf):
            m.update({"ratio": PP_OMEGA_N / OMEGA_N_MAX, "in_bound": False,
                      "is_pp": True, "ld": ld})
        local_by_ld[ld] = ([ppl] + rows_local, OMEGA_N_MAX)
        far_by_ld[ld] = ([ppf] + rows_far, OMEGA_N_MAX)
        all_local += rows_local
        legacy_fig_test1_step_response(fig_dir, straight, tag)
        legacy_fig_test1_bound_validation(
            fig_dir, rows_local, rows_far, rows_probe, e_star, tag
        )
        per_ld_meta[str(ld)] = {
            "ebar_g": EBAR_G, "omega_n_max": OMEGA_N_MAX,
            "omega_n_pp": PP_OMEGA_N,
            "guaranteed_headroom": OMEGA_N_MAX >= PP_OMEGA_N,
            "accel_bound_zeta1": omega_n_accel_bound(1.0, EBAR_G),
            "worst_effective_error_e_star": e_star,
            "local": rows_local, "far": rows_far,
            "envelope_probe": rows_probe,
            "pp_reference_local": ppl, "pp_reference_far": ppf,
        }

    legacy_write_test1_sweep_table(
        table_dir / "legacy_sim_test1_sweep_local.tex", local_by_ld, "local"
    )
    legacy_write_test1_sweep_table(
        table_dir / "legacy_sim_test1_sweep_far.tex", far_by_ld, "far"
    )

    # cross-L_d selection: in-bound, M_os <= 0.05 E0 (local), min T_s,
    # tie min kappa_max — common conditions make T_s comparable across L_d
    sel, relaxed = legacy_select_from_rows(all_local, LEGACY_COND_LOCAL[0])
    legacy_write_test1_selection(
        table_dir / "legacy_sim_test1_selection.tex", sel
    )

    meta = {
        "v0": V0, "omega_max": OMEGA_MAX,
        "test1_lds": list(LEGACY_TEST1_LDS),
        "dt": DT, "v_epsilon": V_EPSILON,
        "eps_on": EPS_ON, "eps_off": EPS_OFF,
        "omega_n_max_formula": (
            "v_gain*sqrt(omega_budget/(v*ebar_g)); "
            "v_gain=abs(v)+v_epsilon"
        ),
        "alpha_max_assumed": ALPHA_MAX,
        "omega_ratios": LEGACY_OMEGA_RATIOS, "zetas": LEGACY_ZETAS,
        "oob_ratio": LEGACY_OOB_RATIO, "oob_zeta": LEGACY_OOB_ZETA,
        "cond_local": [
            LEGACY_COND_LOCAL[0], math.degrees(LEGACY_COND_LOCAL[1])
        ],
        "cond_far": [
            LEGACY_COND_FAR[0], math.degrees(LEGACY_COND_FAR[1])
        ],
        "selection_relaxed": relaxed,
        "notes": (
            "sat_ratio = fraction of eval steps with |omega_raw| >= "
            "omega_max before clipping. Initial conditions are common across "
            "L_d (local gate-ON for every L_d, far gate-OFF for every L_d) "
            "so T_s is directly comparable and the selection ranges over "
            "(omega_n, zeta, L_d). The gate-envelope worst-case step e=e_star "
            "(argmax sigma(e) e) exercises the compensation and makes the "
            "OOB command saturate, validating the rate bound."
        ),
    }
    meta["carrot_rule"] = "continuous_arc_length"
    out = {"meta": meta, "selected": sel, "per_ld": per_ld_meta}
    (table_dir / "legacy_sim_test1_metrics.json").write_text(
        json.dumps(_clean(out), indent=2), encoding="utf-8")
    return sel, relaxed


# ---------------------------------------------------------------------------
# Frozen TEST 1 design: separate speed and damping studies
# ---------------------------------------------------------------------------
SPEED_LD = 1.0
SPEED_COND = (0.30, 0.0)
SPEED_ZETA = 1.0 / math.sqrt(2.0)
SPEED_PP_OMEGA_N = math.sqrt(2.0) * (V0 + V_EPSILON) / SPEED_LD
SPEED_EBAR_G = math.sqrt(EPS_OFF) * SPEED_LD
SPEED_OMEGA_N_MAX = omega_n_max(SPEED_ZETA, SPEED_EBAR_G, sbar=0.0)
SPEED_LAMBDAS = (
    0.75,
    1.00,
    1.25,
    SPEED_OMEGA_N_MAX / SPEED_PP_OMEGA_N,
)

DAMPING_LD = 0.5
DAMPING_COND = (0.15, math.radians(-30.0))
DAMPING_OMEGA_N = math.sqrt(2.0) * (V0 + V_EPSILON) / DAMPING_LD
DAMPING_ZETAS = (1.0 / math.sqrt(2.0), 1.0, math.sqrt(2.0))


def _speed_arms():
    arms = [{
        "key": "PP",
        "label": "PP",
        "method": "PP",
        "omega_n": SPEED_PP_OMEGA_N,
        "zeta": SPEED_ZETA,
        "lambda": 1.0,
    }]
    for index, lam in enumerate(SPEED_LAMBDAS):
        suffix = "max" if index == len(SPEED_LAMBDAS) - 1 else f"{lam:.2f}"
        arms.append({
            "key": f"ECPP_lambda_{suffix}",
            "label": (
                r"ECPP ($\lambda=\lambda_{\max}$)"
                if suffix == "max"
                else fr"ECPP ($\lambda={lam:.2f}$)"
            ),
            "method": "ECPP",
            "omega_n": lam * SPEED_PP_OMEGA_N,
            "zeta": SPEED_ZETA,
            "lambda": lam,
        })
    return arms


def _damping_arms():
    arms = [{
        "key": "PP",
        "label": "PP",
        "method": "PP",
        "omega_n": DAMPING_OMEGA_N,
        "zeta": 1.0 / math.sqrt(2.0),
    }]
    for zeta, suffix in zip(DAMPING_ZETAS, ("inv_sqrt2", "1", "sqrt2")):
        arms.append({
            "key": f"ECPP_zeta_{suffix}",
            "label": (
                r"ECPP ($\zeta=1/\sqrt{2}$)" if suffix == "inv_sqrt2"
                else r"ECPP ($\zeta=\sqrt{2}$)" if suffix == "sqrt2"
                else r"ECPP ($\zeta=1$)"
            ),
            "method": "ECPP",
            "omega_n": DAMPING_OMEGA_N,
            "zeta": zeta,
        })
    return arms


def _run_test1_family(ld, condition, arms, trace_dir, family):
    configure(ld=ld)
    path = StraightPath()
    results = []
    traces = {}
    trace_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        metric, arr, _ = run_metrics(
            path,
            arm["method"],
            arm["omega_n"],
            arm["zeta"],
            condition[0],
            condition[1],
        )
        row = dict(arm)
        row.update(metric)
        analytic = second_order_response_with_initial_rate(
            arr[:, 0],
            condition[0],
            V0 * math.sin(condition[1]),
            arm["omega_n"],
            arm["zeta"],
        )
        row["analytic_rmse_e_y"] = float(
            np.sqrt(np.mean(np.square(arr[:, 5] - analytic)))
        )
        results.append(row)
        traces[arm["key"]] = arr
        np.savetxt(
            trace_dir / f"sim_test1_{family}_{arm['key']}.csv",
            arr,
            delimiter=",",
            header=(
                "t,x,y,psi,path_s,e_y,e_psi,kappa,omega_cmd,omega_raw,sigma"
            ),
            comments="",
            fmt="%.9g",
        )
    return results, traces


def _write_test1_family_table(path_out, rows, varied):
    lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        (
            r"Method & $\lambda$ & $\omega_n$ & $T_s^{10\%}$ & "
            r"$T_s^{5\%}$ & $T_s^{2\%}$ & $M_{\rm os}$ & IAE & "
            r"$|\omega_{\rm raw}|_{\max}$ \\"
            if varied == "lambda"
            else
            r"Method & $\zeta$ & $\omega_n$ & $T_s^{10\%}$ & "
            r"$T_s^{5\%}$ & $T_s^{2\%}$ & $M_{\rm os}$ & IAE & "
            r"$|\omega_{\rm raw}|_{\max}$ \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        varied_value = row.get("lambda") if varied == "lambda" else row["zeta"]
        lines.append(" & ".join([
            row["label"],
            f(varied_value, 3),
            f(row["omega_n"], 3),
            f(row.get("T_s_10"), 3),
            f(row.get("T_s_05"), 3),
            f(row.get("T_s_02"), 3),
            f(row.get("M_os"), 4),
            f(row.get("iae_e_y"), 4),
            f(row.get("omega_raw_max"), 3),
        ]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path_out.write_text("\n".join(lines), encoding="utf-8")


def _fig_test1_family(fig_dir, family, rows, traces, title):
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(rows)))
    for row, color in zip(rows, colors):
        arr = traces[row["key"]]
        linestyle = "--" if row["method"] == "PP" else "-"
        axes[0].plot(arr[:, 0], arr[:, 5], color=color, ls=linestyle,
                     lw=1.1, label=row["label"])
        analytic = second_order_response_with_initial_rate(
            arr[:, 0],
            arr[0, 5],
            V0 * math.sin(arr[0, 6]),
            row["omega_n"],
            row["zeta"],
        )
        axes[0].plot(
            arr[:, 0], analytic, color=color, ls=":", lw=0.7, alpha=0.7
        )
        axes[1].plot(arr[:, 0], arr[:, 9], color=color, ls=linestyle,
                     lw=1.0, label=row["label"])
    axes[0].axhline(0.0, color="0.65", lw=0.6)
    axes[0].set_ylabel(r"$e_y$ [m]")
    axes[0].set_title(title, fontsize=8)
    axes[0].plot([], [], color="0.35", ls=":", lw=0.8,
                 label="linear model")
    axes[0].legend(fontsize=5.3, ncol=2, framealpha=0.85)
    axes[1].axhline(OMEGA_MAX, color="0.5", lw=0.7, ls=":")
    axes[1].axhline(-OMEGA_MAX, color="0.5", lw=0.7, ls=":")
    axes[1].set_ylabel(r"$\omega_{\rm raw}$ [rad/s]")
    axes[1].set_xlabel(r"$t$ [s]")
    for ax in axes:
        ax.grid(True, color="0.9", lw=0.5)
        ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"sim_test1_{family}_response.{ext}", dpi=300)
    plt.close(fig)


def run_test1(table_dir, fig_dir, trace_dir=None):
    """Run the preregistered speed and damping parameter studies."""

    trace_dir = table_dir.parent / "traces" if trace_dir is None else trace_dir
    speed_rows, speed_traces = _run_test1_family(
        SPEED_LD, SPEED_COND, _speed_arms(), trace_dir, "speed"
    )
    damping_rows, damping_traces = _run_test1_family(
        DAMPING_LD, DAMPING_COND, _damping_arms(), trace_dir, "damping"
    )
    _write_test1_family_table(
        table_dir / "sim_test1_speed_results.tex", speed_rows, "lambda"
    )
    _write_test1_family_table(
        table_dir / "sim_test1_damping_results.tex", damping_rows, "zeta"
    )
    _fig_test1_family(
        fig_dir,
        "speed",
        speed_rows,
        speed_traces,
        r"(a) Speed shaping: $L_d=1.0$ m, $e_y(0)=0.30$ m",
    )
    _fig_test1_family(
        fig_dir,
        "damping",
        damping_rows,
        damping_traces,
        r"(b) Damping shaping: $L_d=0.5$ m, $e_\psi(0)=-30^\circ$",
    )

    out = {
        "meta": {
            "v0": V0,
            "dt": DT,
            "control_rate_hz": 1.0 / DT,
            "omega_max": OMEGA_MAX,
            "v_epsilon": V_EPSILON,
            "acceleration_model": None,
            "clip_semantics": "instantaneous_state_update_only",
            "carrot_rule": "continuous_arc_length",
            "normalized_initial_error": 0.30,
            "normalized_initial_error_rule": (
                "largest 0.1-grid value inside sqrt(eps_on)=0.316..."
            ),
        },
        "speed": {
            "lookahead_m": SPEED_LD,
            "initial_condition": [SPEED_COND[0], math.degrees(SPEED_COND[1])],
            "zeta": SPEED_ZETA,
            "omega_n_pp_configured": SPEED_PP_OMEGA_N,
            "omega_n_max": SPEED_OMEGA_N_MAX,
            "lambdas": list(SPEED_LAMBDAS),
            "selection_rule": (
                "lambda={0.75,1.00,1.25,lambda_max}; lambda_max follows "
                "from the configured-omega rate bound at the gate envelope"
            ),
            "rows": speed_rows,
        },
        "damping": {
            "lookahead_m": DAMPING_LD,
            "initial_condition": [
                DAMPING_COND[0], math.degrees(DAMPING_COND[1])
            ],
            "omega_n": DAMPING_OMEGA_N,
            "zetas": list(DAMPING_ZETAS),
            "selection_rule": (
                "zeta={1/sqrt(2),1,sqrt(2)} about critical damping"
            ),
            "rows": damping_rows,
        },
    }
    (table_dir / "sim_test1_metrics.json").write_text(
        json.dumps(_clean(out), indent=2), encoding="utf-8"
    )
    return out


# ---------------------------------------------------------------------------
# TEST 2 : method comparison on straight and smooth R=3 m arc
# ---------------------------------------------------------------------------
TEST2_METHODS = ["PP", "DPP", "ECPP w/o gate", "ECPP"]
TEST2_CONDS = [
    (0.15, math.radians(0.0)),
    (0.15, math.radians(-30.0)),
    (1.0, math.radians(0.0)),
    (1.0, math.radians(-90.0)),
]
REP_CONDS = TEST2_CONDS
METHOD_COLORS = {"PP": "#d62728", "DPP": "#1f77b4",
                 "ECPP w/o gate": "#2ca02c", "ECPP": "#9467bd"}


def write_test2_table(path_out, results):
    """Write the compact method-comparison table for one path."""
    lines = [
        r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
        r"\toprule",
        r"Method & Eval. & $\bar e_y$ & $\bar e_\theta$ & $T_s$ & "
        r"$M_{\rm os}$ & $T_{\rm eval}$ & $\kappa_{\max}$ & "
        r"$|\omega_{\rm raw}|_{\max}$ & sat. [\%] \\",
        r"\midrule",
    ]
    first = True
    for (ey0, eth0) in TEST2_CONDS:
        if not first:
            lines.append(r"\addlinespace[1pt]")
        first = False
        deg = round(math.degrees(eth0))
        lines.append(
            r"\multicolumn{10}{@{}l}{$e_y(0)=" + f"{ey0:.2f}" +
            r"\,\mathrm{m},\ e_\theta(0)=" + f"{deg}" + r"^\circ$} \\")
        for method in TEST2_METHODS:
            m = results[(method, ey0, deg)]
            lines.append(" & ".join([
                method,
                "yes" if m.get("evaluation_completed") else "no",
                f(m["bar_e_y"], 3),
                f(m["bar_e_theta_deg"], 2),
                f(m.get("T_s"), 2),
                f(m.get("M_os"), 3),
                f(m.get("T_eval"), 2),
                f(m["kappa_max"], 2),
                f(m["omega_raw_max"], 2),
                f(100.0 * m["sat_ratio"], 1),
            ]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path_out.write_text("\n".join(lines), encoding="utf-8")


def make_by_condition_panels(fig_dir, path, omega_n, zeta, ey0, eth0):
    runs = {}
    for method in TEST2_METHODS:
        _, arr, _ = run_metrics(path, method, omega_n, zeta, ey0, eth0)
        runs[method] = arr
    stem = f"{path.key}_{cond_encode(ey0, round(math.degrees(eth0)))}"
    ref = path.sample(400)

    def newfig():
        plt.rcParams.update({"font.size": 9})
        return plt.subplots(figsize=(4.2, 2.5))

    def finish(fig, ax, name, legend=False):
        ax.grid(True, color="0.9", lw=0.6, ls=":")
        ax.tick_params(labelsize=8)
        if legend:
            ax.legend(fontsize=6.5, loc="best", framealpha=0.85)
        fig.tight_layout(pad=0.3)
        for ext in ("pdf", "png"):
            fig.savefig(fig_dir / f"{stem}_{name}.{ext}", dpi=150)
        plt.close(fig)

    # trajectory
    fig, ax = newfig()
    ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.3, label="Reference")
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 1], arr[:, 2], color=METHOD_COLORS[method], lw=1.2,
                label=method)
    ax.set_xlabel(r"$x$ [m]"); ax.set_ylabel(r"$y$ [m]")
    ax.set_aspect("equal", adjustable="datalim")
    finish(fig, ax, "trajectory", legend=True)

    # sigma
    fig, ax = newfig()
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 0], arr[:, 10], color=METHOD_COLORS[method], lw=1.2,
                label=method)
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$\sigma$")
    ax.set_ylim(-0.05, 1.05)
    finish(fig, ax, "sigma")

    # kappa
    fig, ax = newfig()
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 0], arr[:, 7], color=METHOD_COLORS[method], lw=1.2)
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$\kappa$ [1/m]")
    finish(fig, ax, "kappa")

    # error_y
    fig, ax = newfig()
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 0], arr[:, 5], color=METHOD_COLORS[method], lw=1.2)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$e_y$ [m]")
    finish(fig, ax, "error_y")

    # error_psi
    fig, ax = newfig()
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 0], np.degrees(arr[:, 6]),
                color=METHOD_COLORS[method], lw=1.2)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$e_\theta$ [deg]")
    finish(fig, ax, "error_psi")

    # raw requested omega (the state update uses the separately stored clip)
    fig, ax = newfig()
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 0], arr[:, 9], color=METHOD_COLORS[method], lw=1.2)
    ax.axhline(OMEGA_MAX, color="0.5", lw=0.7, ls=":")
    ax.axhline(-OMEGA_MAX, color="0.5", lw=0.7, ls=":")
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$\omega_{\rm raw}$ [rad/s]")
    finish(fig, ax, "omega")


def run_test2(table_dir, fig_dir, omega_n, zeta, ld=None):
    if ld is not None:
        configure(ld=ld)
    bycond_dir = fig_dir / "by_condition"
    bycond_dir.mkdir(parents=True, exist_ok=True)
    paths = {"straight": StraightPath(), "arc": ArcPath()}
    json_out = {"meta": {"omega_n": omega_n, "zeta": zeta, "ld": LD,
                         "omega_max": OMEGA_MAX, "methods": TEST2_METHODS,
                         "L2_dpp": LD_DPP2, "dt": DT,
                         "control_rate_hz": 1.0 / DT,
                         "v0": V0, "v_epsilon": V_EPSILON,
                         "acceleration_model": None,
                         "conditions": [
                             [ey, math.degrees(epsi)]
                             for ey, epsi in TEST2_CONDS
                         ],
                         "condition_count_per_path": len(TEST2_CONDS),
                         "method_count": len(TEST2_METHODS),
                         "path_count": 2,
                         "total_run_count": (
                             len(TEST2_CONDS) * len(TEST2_METHODS) * 2
                         )}, "paths": {}}
    json_out["meta"]["carrot_rule"] = "continuous_arc_length"
    for key, path in paths.items():
        results = {}
        table_rows = {}
        for (ey0, eth0) in TEST2_CONDS:
            deg = round(math.degrees(eth0))
            for method in TEST2_METHODS:
                m, _, _ = run_metrics(path, method, omega_n, zeta, ey0, eth0)
                results[(method, ey0, deg)] = m
                table_rows[f"{method}|{ey0}|{deg}"] = m
        write_test2_table(
            table_dir / f"sim_test2_results_{key}.tex", results)
        json_out["paths"][key] = table_rows
        for (ey0, eth0) in REP_CONDS:
            make_by_condition_panels(bycond_dir, path, omega_n, zeta, ey0, eth0)
    (table_dir / "sim_test2_metrics.json").write_text(
        json.dumps(_clean(json_out), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Hardware-experiment-1 sim reference (chapter-6 direct comparison)
# ---------------------------------------------------------------------------
HW_TS_ABS_BAND = 0.03    # m, absolute settling band (AMCL noise floor)
HW_REACH_BAND = 0.10     # m, far-condition capture threshold (T_10)


def run_hw_reference(table_dir, trace_root):
    """Write ideal references for the frozen hardware experiment 1 suites."""

    straight = StraightPath()
    trace_root.mkdir(parents=True, exist_ok=True)
    out = {"meta": {
        "v0": V0, "v_epsilon": V_EPSILON,
        "omega_max": OMEGA_MAX, "dt": DT, "control_rate_hz": 1.0 / DT,
        "ts_abs_band_m": HW_TS_ABS_BAND, "reach_band_m": HW_REACH_BAND,
        "notes": ("Ideal unicycle, 30 Hz, instantaneous +/-1.5 rad/s "
                  "state-update clip, no acceleration model, no localization "
                  "noise, and no LPF (tau=0)."),
    }, "suites": {}}
    out["meta"]["carrot_rule"] = "continuous_arc_length"

    far_arms = [
        {"key": "PP", "label": "PP", "method": "PP",
         "omega_n": DAMPING_OMEGA_N, "zeta": 1.0 / math.sqrt(2.0)},
        {"key": "ECPP_representative", "label": "ECPP",
         "method": "ECPP", "omega_n": DAMPING_OMEGA_N, "zeta": 1.0},
    ]
    suites = {
        "speed": (SPEED_LD, SPEED_COND, _speed_arms()),
        "damping": (DAMPING_LD, DAMPING_COND, _damping_arms()),
        "far_capture": (DAMPING_LD, (1.0, math.radians(-90.0)), far_arms),
    }
    run_count = 0
    for suite_key, (ld, condition, arms) in suites.items():
        configure(omega_max=1.5, ld=ld)
        ey0, eth0 = condition
        suite_out = {
            "lookahead_m": ld,
            "initial_condition": [ey0, math.degrees(eth0)],
            "arms": {},
        }
        for arm in arms:
            name = arm["key"]
            method = arm["method"]
            wn = arm["omega_n"]
            z = arm["zeta"]
            m, arr, _ = run_metrics(straight, method, wn, z, ey0, eth0)
            t, s, ey = arr[:, 0], arr[:, 4], arr[:, 5]
            emask = (s >= 0.0) & (s <= straight.goal)
            te, eye = t[emask], ey[emask]
            t_rel = te - te[0] if len(te) else te
            e0 = abs(ey0)
            m["T_s_10"] = settling_time(t_rel, eye, 0.10 * e0)
            m["T_s_abs"] = settling_time(t_rel, eye, HW_TS_ABS_BAND)
            cap = np.where(np.abs(eye) <= HW_REACH_BAND)[0]
            m["T_10"] = float(t_rel[cap[0]]) if len(cap) else None
            suite_out["arms"][name] = {k: m.get(k) for k in (
                "omega_n", "zeta", "bar_e_y", "bar_e_theta_deg", "T_r",
                "T_s", "T_s_10", "T_s_05", "T_s_abs", "M_os", "T_m",
                "iae_e_y", "kappa_max", "omega_raw_max",
                "omega_raw_rate_max", "sat_ratio", "T_10", "goal_reached")}
            hdr = "t,e_y,e_theta,omega_cmd,omega_raw,sigma"
            rows = arr[:, [0, 5, 6, 8, 9, 10]]
            np.savetxt(trace_root / f"{suite_key}_{name}.csv", rows,
                       delimiter=",", header=hdr, comments="",
                       fmt="%.6f")
            run_count += 1
        out["suites"][suite_key] = suite_out

    (table_dir / "hw_exp1_sim_reference.json").write_text(
        json.dumps(_clean(out), indent=2), encoding="utf-8")
    print(f"hw reference: {run_count} runs -> "
          f"{table_dir / 'hw_exp1_sim_reference.json'}")


# ---------------------------------------------------------------------------
# Validation: engine reproduction against the current test-1 straight table
# ---------------------------------------------------------------------------
def run_reproduction():
    # Diagnostic only; this legacy point is not part of the frozen experiments.
    saved = (OMEGA_MAX, LD)
    configure(omega_max=1.5, ld=1.0)
    straight = StraightPath()
    conds = [(0.0, -30), (0.0, -90), (0.3, 0), (0.3, -30), (0.3, -90),
             (1.0, 0), (1.0, -30), (1.0, -90)]
    methods = ["PP", "DPP", "ECPP w/o gate", "ECPP"]
    print(f"\n=== Engine reproduction (omega_n={REPRO_OMEGA_N}, "
          f"zeta={REPRO_ZETA}, Ld={LD}, omega_max={OMEGA_MAX}) ===")
    print(f"{'cond':>12} {'method':>14} "
          f"{'e_y':>7}{'e_th':>7}{'Ts':>7}{'Tr':>7}{'Mos':>7}"
          f"{'Tm':>7}{'kmax':>7}")
    rep = {}
    for (ey0, deg) in conds:
        for method in methods:
            m, _, _ = run_metrics(straight, method, REPRO_OMEGA_N, REPRO_ZETA,
                                  ey0, math.radians(deg))
            rep[(ey0, deg, method)] = m
            print(f"{ey0:5.2f},{deg:>4}  {method:>14} "
                  f"{f(m['bar_e_y'],3):>7}{f(m['bar_e_theta_deg'],2):>7}"
                  f"{f(m.get('T_s'),2):>7}{f(m.get('T_r'),2):>7}"
                  f"{f(m.get('M_os'),3):>7}{f(m.get('T_m'),2):>7}"
                  f"{f(m['kappa_max'],2):>7}")
    configure(omega_max=saved[0], ld=saved[1])
    return rep


# ---------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main(
    argv: list[str] | None = None, *, default_output_root: Path | None = None
):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write frozen outputs to generated/ instead of preview")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=default_output_root or Path.cwd(),
        help="paper project root (default: current directory)",
    )
    ap.add_argument("--omega-max", type=float, default=None,
                    help="compatibility option; only 1.5 rad/s is accepted")
    ap.add_argument("--ld", type=float, default=None,
                    help="deprecated compatibility option; frozen suites set L_d")
    ap.add_argument(
        "--carrot-rule",
        choices=("arc",),
        default=None,
        help="compatibility option; continuous arc-length lookahead is fixed",
    )
    ap.add_argument("--tag", type=str, default=None,
                    help="write into a tagged subdirectory of "
                         "generated_preview/ (preview only)")
    ap.add_argument("--hw-reference", action="store_true",
                    help="generate ONLY the chapter-6 experiment-1 sim "
                         "reference (metrics JSON + trace CSVs) and exit")
    args = ap.parse_args(argv)

    if args.omega_max is not None and not math.isclose(
        args.omega_max, 1.5, abs_tol=1e-12
    ):
        ap.error("--omega-max is fixed at 1.5 rad/s for paper simulations")
    if args.ld is not None:
        ap.error("--ld cannot override the frozen per-suite lookahead distances")

    configure(omega_max=args.omega_max, ld=args.ld,
              carrot_rule=args.carrot_rule)
    if args.tag and args.apply:
        ap.error("--tag is preview-only; do not combine with --apply")

    sub = "generated" if args.apply else "generated_preview"
    table_dir = args.out_root / sub / "tables"
    fig_dir = args.out_root / sub / "figures"
    if args.tag:
        table_dir = table_dir / args.tag
        fig_dir = fig_dir / args.tag
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.hw_reference:
        print(f"Out root: {args.out_root / sub}  (apply={args.apply})")
        trace_root = args.out_root / sub / "hw_reference_traces"
        if args.tag:
            trace_root = trace_root / args.tag
        run_hw_reference(table_dir, trace_root)
        return

    print(f"Out root: {args.out_root / sub}  (apply={args.apply})")
    print(f"Config: omega_max = {OMEGA_MAX} rad/s, "
          f"control_rate = {1.0 / DT:.1f} Hz, "
          f"test-1 lookaheads = ({SPEED_LD}, {DAMPING_LD}) m, "
          f"carrot_rule = {CARROT_RULE}")

    test1 = run_test1(table_dir, fig_dir)
    speed = test1["speed"]
    damping = test1["damping"]
    print("\n*** TEST 1 frozen design ***")
    print(f"speed: Ld={speed['lookahead_m']:.2f}, "
          f"omega_pp={speed['omega_n_pp_configured']:.6f}, "
          f"omega_max_design={speed['omega_n_max']:.6f}")
    print(f"damping: Ld={damping['lookahead_m']:.2f}, "
          f"omega_n={damping['omega_n']:.6f}, "
          f"zetas={damping['zetas']}")

    run_test2(
        table_dir,
        fig_dir,
        DAMPING_OMEGA_N,
        1.0,
        ld=DAMPING_LD,
    )
    trace_root = args.out_root / sub / "hw_reference_traces"
    if args.tag:
        trace_root = trace_root / args.tag
    run_hw_reference(table_dir, trace_root)
    print("\nDone.")


if __name__ == "__main__":
    main()

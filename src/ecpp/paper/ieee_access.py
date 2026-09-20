"""Chapter-5 simulation generator for the IEEE Access ECPP paper.

Rebuilt from the ancestor fixed-speed engine and the chapter-5 metric
definitions in ``generate_hw_exp1_outputs.py``.

Engine (frozen experiment values)
---------------------------------
* Pure-python unicycle, 30 Hz exact constant-twist update, no LPF (tau = 0).
* v0 = 0.5 m/s and the instantaneous state-update clip is fixed at
  +/-1.5 rad/s.  There is no acceleration or velocity-smoother model.
* Test 1 runs the full frozen grid: omega_n in {0.707107, 1.029884,
  1.414214} rad/s x zeta in {1/sqrt2, 1, sqrt2} x L_d in
  {1.0, 0.5} m at the single fixed initial condition (0.15 m, 0 deg).
  Every omega_n on the axis is a named quantity (PP-equivalent at L_d=1.0,
  the reference design value at L_d=1.0, PP-equivalent at L_d=0.5).
* Test 2 uses the design point of the hardware experiments: L_d = 1.0 m,
  omega_n = omega_n_ref(1.0 m) = 1.029884 rad/s, zeta = 1 (nominal critical
  damping). Its initial conditions are the one-sided
  grid {0, 0.30, 2.0, 3.0 m} x {0, -90 deg} minus the origin (7 conditions):
  0.30 m keeps the gate open (sigma = 0.99), 2.0 m starts with the gate
  nearly closed, and 3.0 m probes approach from a larger tracking error.
* Controllers: PP, DPP, ECPP w/o gate (sigma == 1), ECPP (gated).
* ECPP law  kappa_des = kappa_PP - sigma(e_y) * (dK_y e_y + dK_theta sin e_theta)
      dK_y     = K_y - 2/L_d^2 ,  K_y     = (omega_n / v0)^2
      dK_theta = K_theta - 2/L_d,  K_theta = 2 zeta omega_n / v0
  omega_n is the paper quantity (natural frequency of the local second-order
  error dynamics at v = v0).  The controller/plugin computes its gains with
  the regularized speed v_g = |v| + 0.05 instead, so the value handed to the
  controller is the configured omega_n^cfg = omega_n * v_g / v0
  (configured_omega_n()).  Both give the same K_y and K_theta.
* Gate: descending sigmoid of eps_y = (e_y / L_d)^2 ONLY (heading independent),
  with eps_on = 0.10, eps_off = 0.50 (residual p = 0.01). This matches the
  current paper (the gate does not depend on heading error).
* DPP (Wang and Mouri, Trans. JSME 2025): preview points on the vehicle axis
  at L_1 = 2 L_d and L_2, lateral deviations e_p,i measured from the path,
  kappa = -(a1 e_p1 + a2 e_p2).  L_2, a1, a2 follow from the three design
  conditions (natural frequency, damping, and the constant-curvature
  condition a1 L1^2 + a2 L2^2 = 2); see ``controllers.dpp``.
* Paths: straight 8 m (eval goal s = 6.0 m); right-turning arc R = 3.0 m over
  180 deg (positive e_y is the outside) with the first 135 deg
  (eval goal s = R*3pi/4) evaluated so the lookahead never pins at the path
  end.
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

# Figure style shared by every paper figure: Times New Roman text and
# STIX (Times-like) math, matching the manuscript body font.
PAPER_RCPARAMS = {
    "pdf.fonttype": 42,  # embed TrueType (Type 42), not Type 3, for IEEE PDF checks
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
}
plt.rcParams.update(PAPER_RCPARAMS)

from ..config import EcppConfig
from ..controllers.dpp import calc_dpp_parameters
from ..controllers.gates import gate_abs
from .tracking import simulate_fixed_speed


ROOT = Path.cwd()

# ---------------------------------------------------------------------------
# Engine parameters (current paper values)
# ---------------------------------------------------------------------------
V0 = 0.50            # m/s forward speed
V_EPSILON = 0.0      # m/s additive low-speed regularization; the paper
                     # simulations run at constant v0 > 0, so none is needed
                     # and omega_n is handed to the controller as is (the
                     # Nav2 plugin uses 0.05 m/s; see configured_omega_n)
OMEGA_MAX = 1.5      # rad/s instantaneous state-update clip
DT = 1.0 / 30.0      # s control period (30 Hz)
LD = 1.0             # m mutable lookahead used by one simulation run
DPP_FAR_FACTOR = 2.0  # DPP far preview distance L_1 = DPP_FAR_FACTOR * L_d
EPS_ON = 0.10
EPS_OFF = 0.50
GATE_P = 0.01        # sigmoid residual
TMAX = 60.0          # s hard sim cap
CARROT_RULE = "arc"  # fixed: nominal arc-length lookahead

STRAIGHT_LEN = 8.0
STRAIGHT_GOAL = 6.0
ARC_R = 3.0
ARC_TURN = -1.0              # right turn: positive e_y is the outside of the arc
ARC_ANGLE = math.pi          # 180 deg control path
ARC_GOAL = ARC_R * 3.0 * math.pi / 4.0   # evaluate first 135 deg

# secondary acceleration bound (report only): assumed platform angular
# acceleration budget; illustrative, plays no role in the design grid.
ALPHA_MAX = 3.0      # rad/s^2

PP_OMEGA_N = math.sqrt(2.0) * V0 / LD
# reproduction/legacy test-1 operating point
REPRO_OMEGA_N = 2.12
REPRO_ZETA = 1.0


# ---------------------------------------------------------------------------
# Theory constants (documented functions)
# ---------------------------------------------------------------------------
def gate_envelope_error_bound(eps_off=None, ld=None):
    """Representative error scale at eps_off; not a zero-compensation boundary."""
    eps_off = EPS_OFF if eps_off is None else eps_off
    ld = LD if ld is None else ld
    return math.sqrt(eps_off) * ld


def omega_n_max(zeta, ebar, sbar=0.0, kappa_ref=0.0,
                v=None, omega_max=None):
    """Nominal-model rate criterion (legacy function name retained).

        omega_n_max(zeta; ebar, sbar)
            = (v/ebar) * (-zeta sbar + sqrt(zeta^2 sbar^2 + ebar w_budget / v))

    with w_budget = omega_max - v |kappa_ref| (straight: kappa_ref = 0).
    With sbar=0 and the representative gate error scale this gives the
    paper's omega_n,ref. It is a reference for tuning, not a non-saturation
    guarantee for the nonlinear closed loop."""
    v = V0 if v is None else v
    omega_max = OMEGA_MAX if omega_max is None else omega_max
    w_budget = omega_max - v * abs(kappa_ref)
    disc = zeta * zeta * sbar * sbar + ebar * w_budget / v
    return (abs(v) / ebar) * (-zeta * sbar + math.sqrt(disc))


def configured_omega_n(omega_n, v=None, v_epsilon=V_EPSILON):
    """Plugin/controller parameter that realizes the paper's ``omega_n``.

    The manuscript defines K_y = (omega_n / v)^2 and K_theta = 2 zeta
    omega_n / v with the nominal speed v; the controller evaluates the same
    formulas with the regularized speed v_g = |v| + v_epsilon.  Passing
    ``omega_n * v_g / |v|`` therefore reproduces the paper's gains exactly."""
    v = V0 if v is None else v
    return float(omega_n) * (abs(v) + v_epsilon) / abs(v)


def omega_n_accel_bound(
    zeta, ebar, alpha_max=ALPHA_MAX, v=V0, v_epsilon=V_EPSILON
):
    """Secondary acceleration bound on the paper's omega_n.

    omega_n <= (alpha_max v / (2 zeta ebar))^(1/3)  (v_epsilon-free once the
    natural frequency is defined at the nominal speed).
    Report only; no role in the design grid."""
    del v_epsilon  # kept for signature compatibility
    return (alpha_max * abs(v) / (2.0 * zeta * ebar)) ** (1.0 / 3.0)


EBAR_G = gate_envelope_error_bound()                 # 0.7071 m
OMEGA_N_MAX = omega_n_max(1.0, EBAR_G, sbar=0.0)


def configure(omega_max=None, ld=None, tmax=None, carrot_rule=None):
    """Reconfigure L_d, horizon, and representative conditions.

    ``omega_max`` remains as a compatibility argument but accepts only the
    fixed 1.5 rad/s state-update clip.
    """
    global OMEGA_MAX, LD, TMAX, CARROT_RULE
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
    PP_OMEGA_N = math.sqrt(2.0) * V0 / LD
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

    def __init__(self, R=ARC_R, angle=ARC_ANGLE, goal=ARC_GOAL, turn=ARC_TURN):
        self.R = R
        self.angle = angle
        self.length = R * angle
        self.goal = goal
        # +1: left turn (positive e_y is the inside); -1: right turn
        # (positive e_y is the outside).
        self.turn = 1.0 if float(turn) >= 0.0 else -1.0

    def point(self, s):
        s = float(np.clip(s, 0, self.length))
        a = s / self.R
        return np.array([
            self.R * math.sin(a),
            self.turn * self.R * (1.0 - math.cos(a)),
        ])

    def theta(self, s):
        s = float(np.clip(s, 0, self.length))
        return self.turn * s / self.R

    def curvature(self, s):
        return self.turn / self.R

    def closest_s(self, p):
        v = np.array([p[0], p[1] - self.turn * self.R])
        ang = math.atan2(v[1], v[0])
        a = self.turn * ang + math.pi / 2.0
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
def dpp_parameters(omega_n, zeta, v=None, ld=None, far_factor=None):
    """DPP preview distances and coefficients (Wang-Mouri design pattern 3).

    ``omega_n`` is the paper quantity (defined at the nominal speed); it is
    converted with :func:`configured_omega_n` before reaching the controller.
    Returns ``(L1, L2, a1, a2, v_gain)`` from the same controller function the
    simulation uses, so the paper tables and the closed loop cannot drift.
    Raises ``ValueError`` when the design point has no admissible near
    preview distance.
    """

    v = V0 if v is None else v
    ld = LD if ld is None else ld
    far_factor = DPP_FAR_FACTOR if far_factor is None else far_factor
    omega_cfg = configured_omega_n(omega_n, v=v)
    config = EcppConfig(
        lookahead_m=float(ld),
        v_max=abs(float(v)),
        ecpp_omega_n=omega_cfg,
        ecpp_zeta=float(zeta),
        ecpp_v_epsilon=V_EPSILON,
        dpp_omega_n=omega_cfg,
        dpp_zeta=float(zeta),
        dpp_gain_speed=abs(float(v)),
        dpp_far_factor=float(far_factor),
    )
    return calc_dpp_parameters(config)


class Gains:
    """Design gains recorded with a trace for analysis overlays.

    ``omega_n`` is the paper quantity (natural frequency at the nominal
    speed).  ``omega_n_cfg`` is the value the controller is configured with;
    both yield the same ``Ky`` and ``Kth``."""

    def __init__(self, omega_n, zeta, v=None, ld=None):
        v = V0 if v is None else v
        ld = LD if ld is None else ld
        v_gain = abs(v) + V_EPSILON
        self.omega_n = omega_n
        self.omega_n_cfg = configured_omega_n(omega_n, v=v)
        self.zeta = zeta
        self.v_gain = v_gain
        self.Ky = (omega_n / abs(v)) ** 2
        self.Kth = 2.0 * zeta * omega_n / abs(v)
        self.Ky_pp = 2.0 / ld**2
        self.Kth_pp = 2.0 / ld
        self.dKy = self.Ky - self.Ky_pp
        self.dKth = self.Kth - self.Kth_pp
        # DPP geometry exists only where the paper's three conditions admit a
        # positive near preview distance; other cells simply record None.
        self.L1 = self.L2 = self.a1 = self.a2 = None
        try:
            self.L1, self.L2, self.a1, self.a2, _ = dpp_parameters(
                omega_n, zeta, v=v, ld=ld
            )
        except ValueError:
            pass


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
        omega_n=gains.omega_n_cfg,
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
    """Homogeneous response of the designed second-order model.

    ``wn`` and ``z`` are the nominal design values, not fitted to a trace.
    """
    return second_order_response_with_initial_rate(t, e0, 0.0, wn, z)


def second_order_response_with_initial_rate(t, e0, e_dot0, wn, z):
    """Linear response of the designed model with an initial rate.

    ``wn`` is the paper's natural frequency (defined at v0).  The poles are
    computed from the target gains at nominal speed. Nonlinear geometry,
    the actual sigmoid, sampling, and saturation are absent from this model.
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
        r"$\bar e_\theta$ & $T_r$ & $T_s$ & $M_\mathrm{os}$ & $T_m$ & "
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
        r"$\omega_{n,\mathrm{PP}}=\sqrt{2}v_0/L_d$, $\zeta=0.707$).}\\",
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
        r"$\omega_n/\omega_n^{\max}$ & $T_s$ & $M_\mathrm{os}$ & "
        r"$\kappa_{\max}$ \\",
        r"\midrule",
    ]
    lines.append(" & ".join([
        r"selected ($M_\mathrm{os}\!\le\!0.05E_0$, min $T_s$)",
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
    ax.set_ylabel(r"$\omega_\mathrm{cmd}$ [rad/s]")
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
            "v*sqrt(omega_budget/(v*ebar_g)) (paper basis, omega_n at v0); "
            "plugin value = omega_n*(abs(v)+v_epsilon)/abs(v)"
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
# Frozen chapter-6 hardware-experiment-1 arm definitions.  These constants
# drive run_hw_reference() (the sim reference traces for the hardware arms)
# and also name the omega_n axis of the test-1 grid below.
# ---------------------------------------------------------------------------
SPEED_LD = 1.0
SPEED_COND = (0.30, 0.0)
SPEED_ZETA = 1.0 / math.sqrt(2.0)
SPEED_PP_OMEGA_N = math.sqrt(2.0) * V0 / SPEED_LD
SPEED_EBAR_G = math.sqrt(EPS_OFF) * SPEED_LD
SPEED_OMEGA_N_MAX = omega_n_max(SPEED_ZETA, SPEED_EBAR_G, sbar=0.0)

GRID_LD_SHORT = 0.5
GRID_PP_OMEGA_N_LD05 = math.sqrt(2.0) * V0 / GRID_LD_SHORT

# ---------------------------------------------------------------------------
# Frozen TEST 1 design: full (omega_n, zeta, L_d) grid at one fixed initial
# condition.  Every omega_n column is a named quantity, not a tuned value
# (paper basis, natural frequency at v0; the plugin is configured with
# configured_omega_n(), i.e. 1.1x these values: 0.778, 1.133, 1.556 rad/s):
#   0.707107 rad/s = PP-equivalent omega_n = sqrt(2) v0 / L_d at L_d = 1.0 m
#   1.029884 rad/s = reference design value omega_n,ref at L_d = 1.0 m
#   1.414214 rad/s = PP-equivalent omega_n at L_d = 0.5 m
# The fixed initial condition keeps the gate nearly fully on for BOTH lookaheads
# ((0.15/0.5)^2 = 0.09 < eps_on = 0.10), so every cell starts in the linear
# nominal-design region; saturation phenomenology is exercised by the far
# test-2 conditions instead.  Test 1 is a parameter study only; it does not
# select the test-2 operating point.
# ---------------------------------------------------------------------------
GRID_LDS = (1.0, GRID_LD_SHORT)
GRID_COND = (0.15, 0.0)
GRID_ZETAS = (1.0 / math.sqrt(2.0), 1.0, math.sqrt(2.0))
GRID_OMEGAS = (SPEED_PP_OMEGA_N, SPEED_OMEGA_N_MAX, GRID_PP_OMEGA_N_LD05)
GRID_OMEGA_NAMES = (
    "pp_equivalent_ld_1p0",
    "omega_n_max_ld_1p0",
    "pp_equivalent_ld_0p5",
)


_HW_ZETAS = (
    (1.0 / math.sqrt(2.0), "z0707", r"\zeta=1/\sqrt{2}"),
    (1.0, "z1000", r"\zeta=1"),
    (math.sqrt(2.0), "z1414", r"\zeta=\sqrt{2}"),
)


def _hw_local_arms():
    """Hardware experiment-1 local group: PP + {0.707, 1.030} x zeta grid.

    The arm keys keep the plugin's configured values (w0778 = 0.778 rad/s,
    w1133 = 1.133 rad/s) because they name the recorded hardware runs; the
    labels report the paper's omega_n (defined at v0)."""

    arms = [{
        "key": "PP",
        "label": "PP",
        "method": "PP",
        "omega_n": SPEED_PP_OMEGA_N,
        "zeta": 1.0 / math.sqrt(2.0),
    }]
    for omega_n, wtag, wmath in (
        (SPEED_PP_OMEGA_N, "w0778", r"\omega_n=0.707"),
        (SPEED_OMEGA_N_MAX, "w1133", r"\omega_n=1.030"),
    ):
        for zeta, ztag, zmath in _HW_ZETAS:
            arms.append({
                "key": f"ECPP_{wtag}_{ztag}",
                "label": fr"ECPP (${wmath}$, ${zmath}$)",
                "method": "ECPP",
                "omega_n": omega_n,
                "zeta": zeta,
            })
    return arms


def _hw_far_arms():
    """Hardware experiment-1 far group: PP + zeta sweep at the reference value."""

    arms = [{
        "key": "PP",
        "label": "PP",
        "method": "PP",
        "omega_n": SPEED_PP_OMEGA_N,
        "zeta": 1.0 / math.sqrt(2.0),
    }]
    for zeta, ztag, zmath in _HW_ZETAS:
        arms.append({
            "key": f"ECPP_w1133_{ztag}",
            "label": fr"ECPP (${zmath}$)",
            "method": "ECPP",
            "omega_n": SPEED_OMEGA_N_MAX,
            "zeta": zeta,
        })
    return arms


def _ld_tag(ld):
    return f"ld{round(ld * 100):03d}"


def _omega_tag(omega_n):
    return f"w{round(omega_n * 1000):04d}"


def _zeta_tag(zeta):
    return f"z{round(zeta * 1000):04d}"


def _zeta_math(zeta):
    if math.isclose(zeta, 1.0 / math.sqrt(2.0), abs_tol=1e-9):
        return r"1/\sqrt{2}"
    if math.isclose(zeta, math.sqrt(2.0), abs_tol=1e-9):
        return r"\sqrt{2}"
    return f"{zeta:g}"


def _run_test1_grid_block(ld, trace_dir):
    """Simulate the 3x3 (omega_n, zeta) block of the test-1 grid at one L_d."""

    configure(ld=ld)
    path = StraightPath()
    ey0, eth0 = GRID_COND
    pp_cfg = PP_OMEGA_N
    bound = OMEGA_N_MAX
    pp_metric, pp_arr, _ = run_metrics(
        path, "PP", pp_cfg, 1.0 / math.sqrt(2.0), ey0, eth0
    )
    rows = []
    traces = {}
    control_dev = None
    for omega_n, omega_name in zip(GRID_OMEGAS, GRID_OMEGA_NAMES):
        for zeta in GRID_ZETAS:
            metric, arr, _ = run_metrics(
                path, "ECPP", omega_n, zeta, ey0, eth0
            )
            key = f"{_ld_tag(ld)}_{_omega_tag(omega_n)}_{_zeta_tag(zeta)}"
            row = dict(metric)
            row["key"] = key
            row["ld"] = ld
            row["omega_name"] = omega_name
            row["lambda"] = omega_n / pp_cfg
            row["in_bound"] = bool(omega_n <= bound + 1e-9)
            row["pp_equivalent"] = bool(
                math.isclose(omega_n, pp_cfg, rel_tol=0.0, abs_tol=1e-9)
                and math.isclose(
                    zeta, 1.0 / math.sqrt(2.0), rel_tol=0.0, abs_tol=1e-9
                )
            )
            if row["pp_equivalent"]:
                steps = min(len(arr), len(pp_arr))
                control_dev = float(
                    np.max(np.abs(arr[:steps, 1:4] - pp_arr[:steps, 1:4]))
                )
                if len(arr) != len(pp_arr) or control_dev > 1e-9:
                    raise RuntimeError(
                        "negative control failed: the PP-equivalent ECPP "
                        f"cell {key} deviates from PP by {control_dev:.3e}"
                    )
            rows.append(row)
            traces[key] = arr
            np.savetxt(
                trace_dir / f"sim_test1_grid_{key}.csv",
                arr,
                delimiter=",",
                header=(
                    "t,x,y,psi,path_s,e_y,e_psi,kappa,omega_cmd,"
                    "omega_raw,sigma"
                ),
                comments="",
                fmt="%.9g",
            )
    block = {
        "lookahead_m": ld,
        "omega_n_pp": pp_cfg,
        "omega_n_pp_configured": configured_omega_n(pp_cfg),
        "omega_n_max": bound,
        "omega_n_max_configured": configured_omega_n(bound),
        "lambda_max": bound / pp_cfg,
        "pp_negative_control_max_dev": control_dev,
        "pp_row": dict(pp_metric),
        "rows": rows,
    }
    return block, traces


def _write_test1_grid_table(path_out, blocks):
    """Write the combined table and compact per-lookahead companion tables."""

    lines = [
        r"\begin{tabular}{@{}rlrrrrrrr@{}}",
        r"\toprule",
        (
            r"$\omega_n$ [rad/s] & $\zeta$ & $\lambda$ & $\bar e_y$ [m] & "
            r"$\bar e_\theta$ [$^\circ$] & $T_r$ [s] & $T_s^{2\%}$ [s] & "
            r"$M_\mathrm{os}$ [m] & $\kappa_{\max}$ [1/m] \\"
        ),
        r"\midrule",
    ]
    first_block = True
    for ld in GRID_LDS:
        block = blocks[ld]
        if not first_block:
            lines.append(r"\midrule")
        first_block = False
        lines.append(
            r"\multicolumn{9}{@{}l}{$L_d=" + f"{ld:.1f}"
            + r"\,\mathrm{m}$:\ $\omega_{n,\mathrm{PP}}="
            + f"{block['omega_n_pp']:.3f}"
            + r"$, $\omega_{n,\mathrm{ref}}=" + f"{block['omega_n_max']:.3f}"
            + r"\,\mathrm{rad/s}$} \\"
        )
        block_start = len(lines)
        pp = block.get("pp_row")
        if pp is not None:
            lines.append(" & ".join([
                r"\multicolumn{3}{@{}l}{PP}",
                f(pp.get("bar_e_y"), 3),
                f(pp.get("bar_e_theta_deg"), 2),
                f(pp.get("T_r"), 2, none="n/r"),
                f(pp.get("T_s_02"), 2, none="n/r"),
                f(pp.get("M_os"), 4),
                f(pp.get("kappa_max"), 2),
            ]) + r"\\")
            lines.append(r"\addlinespace[1pt]")
        previous_omega = None
        for row in block["rows"]:
            if previous_omega is not None and row["omega_n"] != previous_omega:
                lines.append(r"\addlinespace[1pt]")
            previous_omega = row["omega_n"]
            omega_cell = f"{row['omega_n']:.3f}"
            if not row["in_bound"]:
                omega_cell += r"$^{\dagger}$"
            zeta_cell = "$" + _zeta_math(row["zeta"]) + "$"
            if row["pp_equivalent"]:
                zeta_cell += r" (=PP)"
            lines.append(" & ".join([
                omega_cell,
                zeta_cell,
                f(row["lambda"], 2),
                f(row.get("bar_e_y"), 3),
                f(row.get("bar_e_theta_deg"), 2),
                f(row.get("T_r"), 2, none="n/r"),
                f(row.get("T_s_02"), 2, none="n/r"),
                f(row.get("M_os"), 4),
                f(row.get("kappa_max"), 2),
            ]) + r"\\")
        split_lines = [
            r"\begin{tabular}{@{}rlrrrrrrr@{}}",
            r"\toprule",
            r"\shortstack{$\omega_n$\\{[rad/s]}} & $\zeta$ & $\lambda$ & "
            r"\shortstack{$\bar e_y$\\{[m]}} & "
            r"\shortstack{$\bar e_\theta$\\{[$^\circ$]}} & "
            r"\shortstack{$T_r$\\{[s]}} & "
            r"\shortstack{$T_s^{2\%}$\\{[s]}} & "
            r"\shortstack{$M_\mathrm{os}$\\{[m]}} & "
            r"\shortstack{$\kappa_{\max}$\\{[1/m]}} \\",
            r"\midrule",
            *lines[block_start:],
            r"\bottomrule", r"\end{tabular}", "",
        ]
        split_path = path_out.with_name(f"{path_out.stem}_{_ld_tag(ld)}{path_out.suffix}")
        split_path.write_text("\n".join(split_lines), encoding="utf-8")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path_out.write_text("\n".join(lines), encoding="utf-8")


def _plot_test1_response(ax, row, arr, color):
    """Overlay the unfitted nominal response on the unchanged evaluation trace."""
    time = arr[:, 0]
    reference = second_order_response_with_initial_rate(
        time, GRID_COND[0], V0 * math.sin(GRID_COND[1]),
        row["omega_n"], row["zeta"],
    )
    from matplotlib import patheffects
    actual_line, = ax.plot(time, arr[:, 5], color=color, ls="-", lw=1.4)
    reference_line, = ax.plot(
        time, reference, color="0.15", ls=(0, (4, 3)), lw=0.9, zorder=3,
        path_effects=[patheffects.Stroke(linewidth=1.8, foreground="white"),
                      patheffects.Normal()],
    )
    if row["pp_equivalent"]:
        ax.text(0.98, 0.95, rf"$\omega_n={row['omega_n']:.3f}$ (=PP)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="0.25")
    return actual_line, reference_line


def _fig_test1_grid(fig_dir, blocks, traces):
    """2x3 panel figure: rows = L_d, columns = zeta, curves = omega_n."""

    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(
        2, 3, figsize=(7.1, 4.2), sharex=True, sharey="row"
    )
    colors = plt.get_cmap("viridis")(
        np.linspace(0.15, 0.85, len(GRID_OMEGAS))
    )
    for i, ld in enumerate(GRID_LDS):
        rows_by_cell = {
            (round(r["omega_n"], 6), round(r["zeta"], 6)): r
            for r in blocks[ld]["rows"]
        }
        for j, zeta in enumerate(GRID_ZETAS):
            ax = axes[i][j]
            for omega_n, color in zip(GRID_OMEGAS, colors):
                row = rows_by_cell[(round(omega_n, 6), round(zeta, 6))]
                arr = traces[row["key"]]
                _plot_test1_response(ax, row, arr, color)
            ax.axhline(0.0, color="0.65", lw=0.6)
            ax.grid(True, color="0.9", lw=0.5)
            ax.tick_params(labelsize=7)
            ax.set_title(
                f"$L_d={ld:.1f}$ m, $\\zeta={_zeta_math(zeta)}$", fontsize=8
            )
            if i == len(GRID_LDS) - 1:
                ax.set_xlabel(r"$t$ [s]")
            if j == 0:
                ax.set_ylabel(r"$e_y$ [m]")
    from matplotlib.lines import Line2D
    frequency_handles = [
        Line2D([], [], color=color, lw=1.4,
               label=rf"$\omega_n={omega_n:.3f}$ rad/s")
        for omega_n, color in zip(GRID_OMEGAS, colors)
    ]
    style_handles = [
        Line2D([], [], color="0.3", lw=1.4, ls="-", label="Simulation"),
        Line2D([], [], color="0.15", lw=0.9, ls=(0, (4, 3)), label="Nominal model"),
    ]
    legend_handles = frequency_handles + style_handles
    fig.legend(handles=legend_handles, loc="upper center", ncol=5,
               fontsize=7, frameon=False, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"sim_test1_grid_response.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    # The manuscript assembles the grid from one file per cell with LaTeX
    # sub-captions "(a) L_d = ..., zeta = ..." (no parameter text inside the
    # panels) and a legend strip above the panels.
    legend_fig = plt.figure(figsize=(7.1, 0.22))
    legend_fig.legend(handles=legend_handles, loc="center", ncol=5,
                      fontsize=7, frameon=False, columnspacing=1.2)
    for ext in ("pdf", "png"):
        legend_fig.savefig(fig_dir / f"sim_test1_grid_legend.{ext}", dpi=300,
                           bbox_inches="tight", pad_inches=0.02)
    plt.close(legend_fig)
    for ld in GRID_LDS:
        rows_by_cell = {
            (round(r["omega_n"], 6), round(r["zeta"], 6)): r
            for r in blocks[ld]["rows"]
        }
        for zeta in GRID_ZETAS:
            pfig, pax = plt.subplots(figsize=(2.35, 1.75))
            for omega_n, color in zip(GRID_OMEGAS, colors):
                row = rows_by_cell[(round(omega_n, 6), round(zeta, 6))]
                arr = traces[row["key"]]
                _plot_test1_response(pax, row, arr, color)
            pax.axhline(0.0, color="0.65", lw=0.6)
            pax.grid(True, color="0.9", lw=0.5)
            pax.tick_params(labelsize=7)
            pax.set_xlim(0.0, 12.0)
            pax.set_ylim(-0.02, 0.16)
            pax.set_xlabel(r"$t$ [s]", fontsize=8)
            pax.set_ylabel(r"$e_y$ [m]", fontsize=8)
            pfig.tight_layout(pad=0.3)
            stem = f"sim_test1_grid_{_ld_tag(ld)}_{_zeta_tag(zeta)}"
            for ext in ("pdf", "png"):
                pfig.savefig(fig_dir / f"{stem}.{ext}", dpi=300,
                             bbox_inches="tight", pad_inches=0.02)
            plt.close(pfig)


def run_test1(table_dir, fig_dir, trace_dir=None):
    """Run the frozen full-grid (omega_n, zeta, L_d) parameter study."""

    trace_dir = table_dir.parent / "traces" if trace_dir is None else trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    blocks = {}
    traces = {}
    for ld in GRID_LDS:
        block, block_traces = _run_test1_grid_block(ld, trace_dir)
        blocks[ld] = block
        traces.update(block_traces)

    _write_test1_grid_table(table_dir / "sim_test1_grid_results.tex", blocks)
    _fig_test1_grid(fig_dir, blocks, traces)

    out = {
        "meta": {
            "v0": V0,
            "dt": DT,
            "control_rate_hz": 1.0 / DT,
            "omega_max": OMEGA_MAX,
            "v_epsilon": V_EPSILON,
            "omega_n_basis": ("natural frequency at v0; plugin configured value = omega_n*(v0+v_epsilon)/v0"),
            "acceleration_model": None,
            "clip_semantics": "instantaneous_state_update_only",
            "carrot_rule": "continuous_arc_length",
            "design": "full_grid_single_initial_condition",
            "initial_condition": [GRID_COND[0], math.degrees(GRID_COND[1])],
            "initial_condition_rule": (
                "fixed absolute (e_y, e_theta); e_y=0.15 m keeps the gate "
                "nearly fully on for both lookaheads ((0.15/0.5)^2 = 0.09 < "
                "eps_on = 0.10)"
            ),
            "omega_axis": {
                name: omega
                for name, omega in zip(GRID_OMEGA_NAMES, GRID_OMEGAS)
            },
            "reference_model": {
                "equation": "e_y'' + 2*zeta*omega_n*e_y' + omega_n**2*e_y = 0",
                "parameters": "the configured nominal omega_n and zeta of each cell",
                "initial_error_m": GRID_COND[0],
                "initial_error_rate_mps": V0 * math.sin(GRID_COND[1]),
                "time_samples": "same evaluation times as each simulation trace",
                "fitted_to_observations": False,
                "line_styles": {
                    "simulation": "frequency-colored solid",
                    "nominal": "black dashed with white outline",
                },
            },
            "reference_design_value": (
                "legacy keys omega_n_max, lambda_max, and in_bound refer to "
                "the approximate reference design value, not a guaranteed limit"
            ),
            "zetas": list(GRID_ZETAS),
            "lookaheads_m": list(GRID_LDS),
            "cell_count": (
                len(GRID_LDS) * len(GRID_OMEGAS) * len(GRID_ZETAS)
            ),
        },
        "blocks": {_ld_tag(ld): blocks[ld] for ld in GRID_LDS},
    }
    (table_dir / "sim_test1_metrics.json").write_text(
        json.dumps(_clean(out), indent=2), encoding="utf-8"
    )
    return out


# ---------------------------------------------------------------------------
# TEST 2 : method comparison on straight and smooth R=3 m arc
# ---------------------------------------------------------------------------
TEST2_METHODS = ["PP", "DPP", "ECPP w/o gate", "ECPP"]
# Operating point shared with the hardware experiments and test 3: the
# reference design value at L_d = 1.0 m with nominal critical damping.
TEST2_LD = SPEED_LD
TEST2_OMEGA_N = SPEED_OMEGA_N_MAX
TEST2_ZETA = 1.0
# One-sided initial-condition grid minus the trivial origin.  Mirror-symmetric
# negative offsets / positive headings are omitted.  e_y = 0 rows exercise the
# pure-heading response, where the e_y-driven gate starts fully open and the
# initial-error-normalized transient metrics (T_r, T_s, M_os) are undefined.
# 0.30 m keeps the gate open (sigma = 0.99, the hardware local amplitude),
# 2.0 m starts with negligible compensation (|e_y|/L_d = 2), and 3.0 m
# probes approach from a larger tracking error.
TEST2_EY0S = (0.0, 0.30, 2.0, 3.0)
TEST2_ETH0_DEGS = (0.0, -90.0)
TEST2_CONDS = [
    (ey0, math.radians(deg))
    for ey0 in TEST2_EY0S
    for deg in TEST2_ETH0_DEGS
    if not (ey0 == 0.0 and deg == 0.0)
]
REP_CONDS = TEST2_CONDS
METHOD_COLORS = {"PP": "#d62728", "DPP": "#1f77b4",
                 "ECPP w/o gate": "#2ca02c", "ECPP": "#9467bd"}


def write_test2_table(path_out, results):
    """Write the compact method-comparison table for one path.

    ``--`` marks metrics that are undefined by construction (the
    initial-error-normalized transients when e_y(0) = 0); ``n/r`` marks
    metrics whose event was not reached inside the evaluation interval.
    """
    lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Method & $\bar e_y$ [m] & $\bar e_\theta$ [$^\circ$] & "
        r"$T_r$ [s] & $T_s^{2\%}$ [s] & $M_\mathrm{os}$ [m] & "
        r"$\kappa_{\max}$ [1/m] & Sat. [\%] \\",
        r"\midrule",
    ]
    first = True
    for (ey0, eth0) in TEST2_CONDS:
        if not first:
            lines.append(r"\addlinespace[1pt]")
        first = False
        deg = round(math.degrees(eth0))
        undefined = abs(ey0) <= 1e-9
        none = "--" if undefined else "n/r"
        lines.append(
            r"\multicolumn{8}{@{}l}{$e_y(0)=" + f"{ey0:.2f}" +
            r"\,\mathrm{m},\ e_\theta(0)=" + f"{deg}" + r"^\circ$} \\")
        for method in TEST2_METHODS:
            m = results[(method, ey0, deg)]
            lines.append(" & ".join([
                method,
                f(m["bar_e_y"], 3),
                f(m["bar_e_theta_deg"], 2),
                f(m.get("T_r"), 2, none=none),
                f(m.get("T_s"), 2, none=none),
                f(m.get("M_os"), 3, none=none),
                f(m["kappa_max"], 2),
                f(100.0 * m["sat_ratio"], 1),
            ]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path_out.write_text("\n".join(lines), encoding="utf-8")


def make_by_condition_panels(fig_dir, path, omega_n, zeta, ey0, eth0):
    """Response panels clipped to the evaluation interval.

    The time axes end at the largest evaluation-completion time among the
    methods that completed the interval (runs that never complete it, e.g. a
    permanently saturated circle, are shown over that same window).  The
    trajectory panel covers the evaluation section of the reference path and
    the initial position.
    """
    runs = {}
    t_evals = []
    for method in TEST2_METHODS:
        m, arr, _ = run_metrics(path, method, omega_n, zeta, ey0, eth0)
        runs[method] = arr
        if m.get("T_eval") is not None:
            t_evals.append(float(m["T_eval"]) + arr[0, 0])
    t_plot = max(t_evals) if t_evals else TMAX
    runs = {
        method: arr[arr[:, 0] <= t_plot + 1e-9] for method, arr in runs.items()
    }
    stem = f"{path.key}_{cond_encode(ey0, round(math.degrees(eth0)))}"
    ref = path.sample(400)
    starts = np.array([arr[0, 1:3] for arr in runs.values()])
    margin = 0.35
    x_lo = min(ref[:, 0].min(), starts[:, 0].min()) - margin
    x_hi = max(ref[:, 0].max(), starts[:, 0].max()) + margin
    y_lo = min(ref[:, 1].min(), starts[:, 1].min()) - margin
    y_hi = max(ref[:, 1].max(), starts[:, 1].max()) + margin
    local_straight = path.key == "straight" and abs(ey0) < 1.5 and abs(eth0) < 1e-9
    trajectory_aspect = "auto" if local_straight else "equal"
    if local_straight:
        y_values = np.concatenate([ref[:, 1], *(arr[:, 2] for arr in runs.values())])
        y_margin = 0.05 * max(float(np.ptp(y_values)), 0.01)
        y_lo, y_hi = float(y_values.min()) - y_margin, float(y_values.max()) + y_margin

    # One small panel per quantity, sized for a 0.24-textwidth minipage so
    # that the manuscript can put a LaTeX sub-caption under each and a shared
    # legend strip (test2_legend.pdf) above the row.
    def newfig(height=1.35):
        plt.rcParams.update({"font.size": 7})
        return plt.subplots(figsize=(1.8, height))

    def finish(fig, ax, name):
        ax.grid(True, color="0.9", lw=0.6, ls=":")
        ax.tick_params(labelsize=6, pad=1.5)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        fig.tight_layout(pad=0.25)
        for ext in ("pdf", "png"):
            fig.savefig(fig_dir / f"{stem}_{name}.{ext}", dpi=300,
                        bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    _write_condition_row(fig_dir, stem, runs, ref, (x_lo, x_hi), (y_lo, y_hi),
                         t_plot, trajectory_aspect=trajectory_aspect)
    _write_test2_legend(fig_dir)

    # trajectory; far starts get an inset of the approach to the path
    fig, ax = newfig()
    ax.plot(ref[:, 0], ref[:, 1], "k--", lw=0.9)
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 1], arr[:, 2], color=METHOD_COLORS[method], lw=0.9)
    ax.set_xlabel(r"$x$ [m]"); ax.set_ylabel(r"$y$ [m]")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect(trajectory_aspect, adjustable="datalim")
    if abs(ey0) >= 1.5:
        _add_near_path_inset(ax, runs, ref)
    finish(fig, ax, "trajectory")

    # sigma
    fig, ax = newfig()
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 0], arr[:, 10], color=METHOD_COLORS[method], lw=0.9)
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$\sigma$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.0, t_plot)
    finish(fig, ax, "sigma")

    # kappa (unclipped command; dotted lines = the rate limit kappa_bar = omega_max / v0)
    fig, ax = newfig()
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 0], arr[:, 7], color=METHOD_COLORS[method], lw=0.9)
    ax.axhline(OMEGA_MAX / V0, color="0.5", lw=0.7, ls=":")
    ax.axhline(-OMEGA_MAX / V0, color="0.5", lw=0.7, ls=":")
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$\kappa$ [1/m]")
    ax.set_xlim(0.0, t_plot)
    finish(fig, ax, "kappa")

    # error_y
    fig, ax = newfig()
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 0], arr[:, 5], color=METHOD_COLORS[method], lw=0.9)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$e_y$ [m]")
    ax.set_xlim(0.0, t_plot)
    finish(fig, ax, "error_y")

    # error_psi
    fig, ax = newfig()
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 0], np.degrees(arr[:, 6]),
                color=METHOD_COLORS[method], lw=0.9)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$e_\theta$ [deg]")
    ax.set_xlim(0.0, t_plot)
    finish(fig, ax, "error_psi")

    # raw requested omega (the state update uses the separately stored clip)
    fig, ax = newfig()
    for method in TEST2_METHODS:
        arr = runs[method]
        ax.plot(arr[:, 0], arr[:, 9], color=METHOD_COLORS[method], lw=0.9)
    ax.axhline(OMEGA_MAX, color="0.5", lw=0.7, ls=":")
    ax.axhline(-OMEGA_MAX, color="0.5", lw=0.7, ls=":")
    ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$\omega_\mathrm{raw}$ [rad/s]")
    ax.set_xlim(0.0, t_plot)
    finish(fig, ax, "omega")


def _write_test2_legend(fig_dir):
    """Legend strip shared by the per-condition panels of test 2."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="k", ls="--", lw=0.9, label="Reference")]
    handles += [Line2D([], [], color=METHOD_COLORS[m], lw=1.2, label=m)
                for m in TEST2_METHODS]
    fig = plt.figure(figsize=(5.0, 0.3))
    fig.legend(handles=handles, loc="center", ncol=len(handles), fontsize=7,
               frameon=False, columnspacing=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"test2_legend.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _add_near_path_inset(ax, runs, ref, near_m=0.1, along_m=1.0):
    """Inset magnifying where PP and ECPP reach the path from a far start.

    The window covers the PP/ECPP trajectories from the first sample with
    |e_y| < ``near_m`` until ``along_m`` further along the reference path
    (the near-path approach and overshoot differ between PP and ECPP there);
    the inset sits in the emptiest corner of the panel."""
    segs = []
    for method in ("PP", "ECPP"):
        arr = runs.get(method)
        if arr is None:
            continue
        idx = np.flatnonzero(np.abs(arr[:, 5]) < near_m)
        if len(idx) == 0:
            continue
        s0 = arr[idx[0], 4]
        seg = arr[(arr[:, 4] >= s0) & (arr[:, 4] <= s0 + along_m)]
        if len(seg):
            segs.append(seg[:, 1:3])
    if not segs:
        return
    pts = np.vstack(segs)
    margin = 0.03
    wx0, wx1 = pts[:, 0].min() - margin, pts[:, 0].max() + margin
    wy0, wy1 = pts[:, 1].min() - margin, pts[:, 1].max() + margin
    # Keep a little context without expanding away the tracking differences.
    if wx1 - wx0 < 0.2:
        c = 0.5 * (wx0 + wx1); wx0, wx1 = c - 0.1, c + 0.1
    if wy1 - wy0 < 0.2:
        c = 0.5 * (wy0 + wy1); wy0, wy1 = c - 0.1, c + 0.1
    # emptiest candidate box: count plotted samples inside each box (the
    # reference path counts ten times, it must stay visible) and reject boxes
    # that overlap the zoom window
    x_lo, x_hi = ax.get_xlim(); y_lo, y_hi = ax.get_ylim()
    ax.figure.canvas.draw()  # datalim aspect may have widened the limits
    x_lo, x_hi = ax.get_xlim(); y_lo, y_hi = ax.get_ylim()
    traces = np.vstack([runs[m][:, 1:3] for m in TEST2_METHODS])
    candidates = [
        (0.50, 0.46, 0.48, 0.50), (0.08, 0.46, 0.48, 0.50),
        (0.50, 0.02, 0.48, 0.50), (0.08, 0.02, 0.48, 0.50),
        (0.08, 0.20, 0.36, 0.60), (0.62, 0.20, 0.36, 0.60),
    ]
    def cost(box):
        fx0, fy0, fw, fh = box
        bx0 = x_lo + fx0 * (x_hi - x_lo); bx1 = bx0 + fw * (x_hi - x_lo)
        by0 = y_lo + fy0 * (y_hi - y_lo); by1 = by0 + fh * (y_hi - y_lo)
        def inside(pts):
            return ((pts[:, 0] >= bx0) & (pts[:, 0] <= bx1)
                    & (pts[:, 1] >= by0) & (pts[:, 1] <= by1)).sum()
        overlap = not (bx1 < wx0 or bx0 > wx1 or by1 < wy0 or by0 > wy1)
        return inside(traces) + 10 * inside(ref) + (10 ** 6 if overlap else 0)
    box = min(candidates, key=cost)
    axins = ax.inset_axes(list(box))
    axins.plot(ref[:, 0], ref[:, 1], "k--", lw=0.8)
    for method in TEST2_METHODS:
        arr = runs[method]
        axins.plot(arr[:, 1], arr[:, 2], color=METHOD_COLORS[method], lw=0.9)
    axins.set_xlim(wx0, wx1); axins.set_ylim(wy0, wy1)
    # Use the selected coordinate ranges; equal aspect would widen them again.
    axins.set_aspect("auto")
    axins.tick_params(labelsize=4.5, length=1.5, pad=1)
    axins.grid(True, color="0.9", lw=0.4, ls=":")
    ax.indicate_inset_zoom(axins, edgecolor="0.4", lw=0.6)


def _write_condition_row(fig_dir, stem, runs, ref, xlim, ylim, t_plot,
                         trajectory_aspect="equal"):
    """One 1x4 row (trajectory, gate, lateral error, curvature command) with a
    single method legend above the panels; this is the figure the manuscript
    embeds for each test-2 condition."""
    plt.rcParams.update({"font.size": 8})
    (x_lo, x_hi), (y_lo, y_hi) = xlim, ylim
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
    ax_xy, ax_sig, ax_ey, ax_k = axes
    ax_xy.plot(ref[:, 0], ref[:, 1], "k--", lw=0.9, label="Reference")
    for method in TEST2_METHODS:
        arr = runs[method]
        ax_xy.plot(arr[:, 1], arr[:, 2], color=METHOD_COLORS[method], lw=1.0,
                   label=method)
        ax_sig.plot(arr[:, 0], arr[:, 10], color=METHOD_COLORS[method], lw=1.0)
        ax_ey.plot(arr[:, 0], arr[:, 5], color=METHOD_COLORS[method], lw=1.0)
        ax_k.plot(arr[:, 0], arr[:, 7], color=METHOD_COLORS[method], lw=1.0)
    ax_xy.set_xlim(x_lo, x_hi)
    ax_xy.set_ylim(y_lo, y_hi)
    # keep the panel box the same height in every condition (a tall bounding
    # box, e.g. the far start beside the arc, widens the data limits instead)
    ax_xy.set_aspect(trajectory_aspect, adjustable="datalim")
    ax_xy.set_xlabel(r"$x$ [m]"); ax_xy.set_ylabel(r"$y$ [m]")
    ax_xy.set_title("(a) Trajectory", fontsize=8, loc="left")
    ax_sig.set_ylim(-0.05, 1.05)
    ax_sig.set_xlabel(r"$t$ [s]"); ax_sig.set_ylabel(r"$\sigma$")
    ax_sig.set_title("(b) Gate value", fontsize=8, loc="left")
    ax_ey.axhline(0.0, color="0.7", lw=0.6)
    ax_ey.set_xlabel(r"$t$ [s]"); ax_ey.set_ylabel(r"$e_y$ [m]")
    ax_ey.set_title("(c) Lateral error", fontsize=8, loc="left")
    ax_k.axhline(OMEGA_MAX / V0, color="0.5", lw=0.7, ls=":")
    ax_k.axhline(-OMEGA_MAX / V0, color="0.5", lw=0.7, ls=":")
    ax_k.set_xlabel(r"$t$ [s]"); ax_k.set_ylabel(r"$\kappa$ [1/m]")
    ax_k.set_title("(d) Curvature command", fontsize=8, loc="left")
    for ax in (ax_sig, ax_ey, ax_k):
        ax.set_xlim(0.0, t_plot)
    for ax in axes:
        ax.grid(True, color="0.9", lw=0.6, ls=":")
        ax.tick_params(labelsize=7)
    handles, labels = ax_xy.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0.0, 0.0, 1.0, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{stem}_row.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)


def run_test2(table_dir, fig_dir, omega_n, zeta, ld=None):
    if ld is not None:
        configure(ld=ld)
    bycond_dir = fig_dir / "by_condition"
    bycond_dir.mkdir(parents=True, exist_ok=True)
    paths = {"straight": StraightPath(), "arc": ArcPath()}
    dpp_l1, dpp_l2, dpp_a1, dpp_a2, _ = dpp_parameters(omega_n, zeta, ld=LD)
    json_out = {"meta": {"omega_n": omega_n, "zeta": zeta, "ld": LD,
                         "omega_max": OMEGA_MAX, "methods": TEST2_METHODS,
                         "dpp": {"definition": (
                             "vehicle-axis preview points; e_p,i = lateral "
                             "deviation of the point from the path; "
                             "kappa = -(a1 e_p1 + a2 e_p2); Wang-Mouri "
                             "pattern 3 with L1 = far_factor * L_d"),
                                 "far_factor": DPP_FAR_FACTOR,
                                 "L1": dpp_l1, "L2": dpp_l2,
                                 "a1": dpp_a1, "a2": dpp_a2,
                                 "a1L1sq_plus_a2L2sq": (
                                     dpp_a1 * dpp_l1**2 + dpp_a2 * dpp_l2**2)},
                         "arc": {"radius_m": ARC_R, "turn": ARC_TURN,
                                 "control_angle_deg": math.degrees(ARC_ANGLE),
                                 "eval_angle_deg": math.degrees(
                                     ARC_GOAL / ARC_R)},
                         "dt": DT,
                         "control_rate_hz": 1.0 / DT,
                         "v0": V0, "v_epsilon": V_EPSILON,
                         "omega_n_basis": ("natural frequency at v0; plugin configured value = omega_n*(v0+v_epsilon)/v0"),
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
        "omega_n_basis": ("natural frequency at v0; plugin configured value = omega_n*(v0+v_epsilon)/v0"),
        "omega_max": OMEGA_MAX, "dt": DT, "control_rate_hz": 1.0 / DT,
        "ts_abs_band_m": HW_TS_ABS_BAND, "reach_band_m": HW_REACH_BAND,
        "notes": ("Ideal unicycle, 30 Hz, instantaneous +/-1.5 rad/s "
                  "state-update clip, no acceleration model, no localization "
                  "noise, and no LPF (tau=0)."),
    }, "suites": {}}
    out["meta"]["carrot_rule"] = "continuous_arc_length"

    suites = {
        "local": (SPEED_LD, SPEED_COND, _hw_local_arms()),
        "far_capture": (SPEED_LD, (1.0, math.radians(-90.0)), _hw_far_arms()),
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
          f"test-1 grid lookaheads = {GRID_LDS} m, "
          f"carrot_rule = {CARROT_RULE}")

    test1 = run_test1(table_dir, fig_dir)
    print("\n*** TEST 1 frozen grid ***")
    for ld in GRID_LDS:
        block = test1["blocks"][_ld_tag(ld)]
        print(f"Ld={ld:.1f}: omega_pp={block['omega_n_pp']:.6f}, "
              f"omega_n_ref={block['omega_n_max']:.6f}, "
              f"negative_control_dev="
              f"{block['pp_negative_control_max_dev']:.2e}")

    print("\n*** TEST 2 frozen operating point ***")
    dpp_l1, dpp_l2, dpp_a1, dpp_a2, _ = dpp_parameters(
        TEST2_OMEGA_N, TEST2_ZETA, ld=TEST2_LD
    )
    print(f"Ld={TEST2_LD:.2f}, omega_n={TEST2_OMEGA_N:.6f}, "
          f"zeta={TEST2_ZETA:.1f}; DPP L1={dpp_l1:.3f} m, L2={dpp_l2:.3f} m, "
          f"a1={dpp_a1:.3f}, a2={dpp_a2:.3f} 1/m^2 "
          f"(a1L1^2+a2L2^2={dpp_a1 * dpp_l1**2 + dpp_a2 * dpp_l2**2:.6f})")
    run_test2(table_dir, fig_dir, TEST2_OMEGA_N, TEST2_ZETA, ld=TEST2_LD)
    trace_root = args.out_root / sub / "hw_reference_traces"
    if args.tag:
        trace_root = trace_root / args.tag
    run_hw_reference(table_dir, trace_root)
    print("\nDone.")


if __name__ == "__main__":
    main()

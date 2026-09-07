"""Test-3 generator: preview range versus local response on a straight-arc-straight path.

Pure Pursuit couples two roles of the lookahead distance L_d: how far ahead a
curvature change is detected (the carrot reaches the change L_d before the
robot) and how fast a lateral offset is recovered (omega_n,PP = sqrt(2) v /
L_d).  This test separates the two roles on the path shape used by Wang and
Mouri (Dual Preview Points, Trans. JSME 2025, Figs. 22/28): a straight lead-in,
a left-turning constant-curvature arc without a transition curve, and a
straight exit.

Frozen conditions
-----------------
* Engine: the chapter-5 fixed-speed unicycle (``tracking.simulate_fixed_speed``)
  with v0 = 0.5 m/s, 30 Hz, +/-1.5 rad/s state-update clip, v_epsilon = 0.05.
* Path: 8 m straight (identical to the Test-1 straight) -> R = 3.0 m left arc
  over 90 deg (identical radius to the Test-2 arc) -> 6 m straight; the
  evaluation ends 4 m after the arc exit so the carrot never pins at the end.
* Initial condition (e_y(0), e_psi(0)) = (0.15 m, 0 deg) = the Test-1 grid
  condition, so the recovery metrics over the first 6 m reproduce Table 1.
* Arms: PP and ECPP at L_d in {0.5, 1.0} m.  ECPP uses the hardware design
  point (omega_n, zeta) = (omega_n_max(L_d = 1.0 m), 1) for both lookaheads,
  which is inside the rate bound for both.

Metrics
-------
Recovery window s in [0, 6 m]: T_r, T_s (2 % band, enters and stays), M_os.
Transition: d_lead (distance before the arc entry at which |kappa_des| first
exceeds 10 % of the arc curvature, searched from s = 6.5 m), e_y_in (maximum
signed lateral error in [entry-1.5, entry+2.0] m; the inside of a left turn is
+e_y), e_y_out (minimum signed lateral error in [exit-1.0, exit+3.0] m; the
outside is -e_y), omega_rate_max (maximum |d omega_raw / dt| over the
transition window), kappa_max, sat_ratio.  The feedforward curvature
kappa_prev(s) of a zero-error carrot L_d ahead is stored for the figure.

Writes to ``generated_preview/`` by default; ``--apply`` switches to
``generated/``.  Does not edit sections/*.tex.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import ieee_access as ia
from .tracking import simulate_fixed_speed


# ---------------------------------------------------------------------------
# Frozen conditions
# ---------------------------------------------------------------------------
V0 = ia.V0
V_EPSILON = ia.V_EPSILON
OMEGA_MAX = ia.OMEGA_MAX
DT = ia.DT
TMAX = ia.TMAX

OMEGA_N = ia.SPEED_OMEGA_N_MAX     # 1.132872 rad/s, hardware design point
ZETA = 1.0
COND = ia.GRID_COND                # (0.15 m, 0 rad), the Test-1 condition
LOOKAHEADS = (0.5, 1.0)
METHODS = ("PP", "ECPP")
ARMS = tuple((method, ld) for method in METHODS for ld in LOOKAHEADS)

LEAD_IN = ia.STRAIGHT_LEN          # 8.0 m, identical to the Test-1 straight
RECOVERY_GOAL = ia.STRAIGHT_GOAL   # 6.0 m recovery window
ARC_R = ia.ARC_R                   # 3.0 m, identical to the Test-2 arc
ARC_ANGLE = math.pi / 2.0          # left 90 deg turn, no transition curve
EXIT_CONTROL = 6.0                 # m of straight after the arc (control path)
EXIT_EVAL = 4.0                    # m of straight after the arc (evaluation)
ARC_VERTEX_SPACING = 0.005         # m, as in ieee_access.ArcPath

S_ENTRY = LEAD_IN
S_EXIT = LEAD_IN + ARC_R * ARC_ANGLE
GOAL = S_EXIT + EXIT_EVAL
ENTRY_WINDOW = (S_ENTRY - 1.5, S_ENTRY + 2.0)
EXIT_WINDOW = (S_EXIT - 1.0, S_EXIT + 3.0)
TRANSITION_WINDOW = (ENTRY_WINDOW[0], EXIT_WINDOW[1])
LEAD_SEARCH_START = 6.5            # m, after every arm has settled
LEAD_THRESHOLD_FRACTION = 0.1      # of the arc curvature
FIGURE_STEM = "sim_test3_preview_decoupling"
TABLE_STEM = "sim_test3_results"
METRICS_STEM = "sim_test3_metrics"
KAPPA_PANEL_WINDOW = (S_ENTRY - 2.0, S_ENTRY + 1.5)


class StraightArcStraightPath:
    """Straight lead-in, left constant-curvature arc, straight exit."""

    key = "straight_arc_straight"
    name = "Straight-arc-straight path"

    def __init__(self, lead_in=LEAD_IN, radius=ARC_R, angle=ARC_ANGLE,
                 exit_control=EXIT_CONTROL, exit_eval=EXIT_EVAL):
        self.lead_in = float(lead_in)
        self.radius = float(radius)
        self.angle = float(angle)
        self.s_entry = self.lead_in
        self.s_exit = self.lead_in + self.radius * self.angle
        self.length = self.s_exit + float(exit_control)
        self.goal = self.s_exit + float(exit_eval)

    def heading(self, s):
        if s <= self.s_entry:
            return 0.0
        if s <= self.s_exit:
            return (s - self.s_entry) / self.radius
        return self.angle

    def curvature(self, s):
        return 1.0 / self.radius if self.s_entry < s <= self.s_exit else 0.0

    def point(self, s):
        s = float(s)
        if s <= self.s_entry:
            return np.array([s, 0.0])
        if s <= self.s_exit:
            a = (s - self.s_entry) / self.radius
            return np.array([
                self.s_entry + self.radius * math.sin(a),
                self.radius * (1.0 - math.cos(a)),
            ])
        end = np.array([
            self.s_entry + self.radius * math.sin(self.angle),
            self.radius * (1.0 - math.cos(self.angle)),
        ])
        d = s - self.s_exit
        return end + d * np.array([math.cos(self.angle), math.sin(self.angle)])

    def sample(self, n=400):
        return np.array([self.point(s) for s in np.linspace(0.0, self.goal, n)])

    def control_polyline(self):
        count = max(2, int(math.ceil(self.radius * self.angle
                                     / ARC_VERTEX_SPACING)) + 1)
        arc_s = np.linspace(self.s_entry, self.s_exit, count)
        arc = [self.point(s) for s in arc_s[1:]]
        return np.vstack([[0.0, 0.0], [self.s_entry, 0.0], *arc,
                          self.point(self.length)])

    def feedforward_curvature(self, s, ld):
        """PP curvature of a zero-error robot at s with the carrot L_d ahead."""
        origin = self.point(s)
        th = self.heading(s)
        carrot = self.point(min(s + ld, self.length))
        d = carrot - origin
        x_l = math.cos(th) * d[0] + math.sin(th) * d[1]
        y_l = -math.sin(th) * d[0] + math.cos(th) * d[1]
        return 2.0 * y_l / (x_l * x_l + y_l * y_l)


def arm_key(method, ld):
    return f"{method.replace(' ', '_')}_ld{ld:.2f}".replace(".", "p")


def _run(path_polyline, path, method, ld):
    return simulate_fixed_speed(
        path=path_polyline,
        method=method,
        lookahead_m=ld,
        omega_n=OMEGA_N,
        zeta=ZETA,
        e_y0=COND[0],
        e_psi0=COND[1],
        speed=V0,
        v_epsilon=V_EPSILON,
        omega_limit=OMEGA_MAX,
        dt=DT,
        t_max=TMAX,
        goal_arc_length=path.goal,
        goal_position=path.point(path.goal),
        goal_tolerance=0.02,
    )


def recovery_metrics(trace, goal=RECOVERY_GOAL):
    """Test-1 transient metrics over the lead-in straight [0, goal]."""
    e0 = COND[0]
    mask = (trace.path_s >= 0.0) & (trace.path_s <= goal)
    t_rel = trace.time[mask] - trace.time[mask][0]
    e = trace.e_y[mask]
    return {
        "T_r": ia.rise_time(t_rel, e, e0),
        "T_s": ia.settling_time(t_rel, e, 0.02 * abs(e0)),
        "M_os": ia.overshoot(e, e0),
        "kappa_max_recovery": float(np.max(np.abs(trace.curvature[mask]))),
    }


def transition_metrics(trace, path):
    s = trace.path_s
    kappa = trace.curvature
    m = {}
    search = (s >= LEAD_SEARCH_START) & (s <= path.s_entry)
    onset = np.where(search & (np.abs(kappa) >= LEAD_THRESHOLD_FRACTION
                               / path.radius))[0]
    m["d_lead"] = float(path.s_entry - s[onset[0]]) if len(onset) else None
    entry = (s >= ENTRY_WINDOW[0]) & (s <= ENTRY_WINDOW[1])
    exit_ = (s >= EXIT_WINDOW[0]) & (s <= EXIT_WINDOW[1])
    trans = (s >= TRANSITION_WINDOW[0]) & (s <= TRANSITION_WINDOW[1])
    m["e_y_in"] = float(np.max(trace.e_y[entry]))
    m["e_y_out"] = float(np.min(trace.e_y[exit_]))
    m["kappa_max_transition"] = float(np.max(np.abs(kappa[trans])))
    rate = np.abs(np.diff(trace.omega_raw)) / DT
    m["omega_rate_max_transition"] = float(np.max(rate[trans[:-1]]))
    m["bar_e_y_transition"] = float(np.mean(np.abs(trace.e_y[trans])))
    return m


def arm_metrics(trace, path, method, ld):
    s = trace.path_s
    evalmask = (s >= 0.0) & (s <= path.goal)
    m = {
        "method": method,
        "ld": ld,
        "omega_n": OMEGA_N if method != "PP" else None,
        "zeta": ZETA if method != "PP" else None,
        "ey0": COND[0],
        "eth0_deg": round(math.degrees(COND[1])),
    }
    m.update(recovery_metrics(trace))
    m.update(transition_metrics(trace, path))
    m["bar_e_y"] = float(np.mean(np.abs(trace.e_y[evalmask])))
    m["kappa_max"] = float(np.max(np.abs(trace.curvature[evalmask])))
    m["sat_ratio"] = float(np.mean(np.abs(trace.omega_raw) >= OMEGA_MAX))
    m["max_abs_ey"] = float(np.max(np.abs(trace.e_y)))
    goal_xy = path.point(path.goal)
    reached = np.where(np.linalg.norm(trace.pose[:, :2] - goal_xy, axis=1)
                       <= 0.02)[0]
    m["goal_reached"] = bool(len(reached))
    m["T_m"] = float(trace.time[reached[0]] - trace.time[0]) if len(reached) else None
    m["t_end"] = float(trace.time[-1])
    return m


@functools.lru_cache(maxsize=1)
def simulate_arms():
    """Run the four frozen arms once per process (traces, metrics, checks)."""
    path = StraightArcStraightPath()
    polyline = path.control_polyline()
    straight = np.array([[0.0, 0.0], [LEAD_IN, 0.0]])
    traces, metrics, consistency = {}, [], {}
    for method, ld in ARMS:
        key = arm_key(method, ld)
        trace = _run(polyline, path, method, ld)
        traces[key] = trace
        metrics.append(dict(arm_metrics(trace, path, method, ld), key=key))
        # The lead-in equals the Test-1 straight, so the recovery metrics must
        # coincide with the Test-1 grid cell run on the bare straight.
        reference = simulate_fixed_speed(
            path=straight, method=method, lookahead_m=ld, omega_n=OMEGA_N,
            zeta=ZETA, e_y0=COND[0], e_psi0=COND[1], speed=V0,
            v_epsilon=V_EPSILON, omega_limit=OMEGA_MAX, dt=DT, t_max=TMAX,
            goal_arc_length=RECOVERY_GOAL,
            goal_position=np.array([RECOVERY_GOAL, 0.0]),
            goal_tolerance=0.02,
        )
        ref = recovery_metrics(reference)
        test3 = {k: metrics[-1][k] for k in ("T_r", "T_s", "M_os")}
        consistency[key] = {
            "test1_straight": {k: ref[k] for k in ("T_r", "T_s", "M_os")},
            "test3_lead_in": test3,
            "max_abs_dev": max(
                abs((test3[k] or 0.0) - (ref[k] or 0.0))
                for k in ("T_r", "T_s", "M_os")
            ),
        }
    return path, traces, metrics, consistency


def feedforward_profiles(path, window=KAPPA_PANEL_WINDOW, spacing=0.01):
    s = np.arange(window[0], window[1] + 1e-9, spacing)
    return s, {ld: np.array([path.feedforward_curvature(v, ld) for v in s])
               for ld in LOOKAHEADS}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_table(path_out, metrics):
    lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Method & $L_d$ [m] & $T_r$ [s] & $T_s^{2\%}$ [s] & $M_{\rm os}$ [m] & "
        r"$d_{\rm lead}$ [m] & $e_{y,\rm in}$ [m] & $e_{y,\rm out}$ [m] & "
        r"$\dot\omega_{\max}$ [rad/s$^2$] \\",
        r"\midrule",
    ]
    for m in metrics:
        lines.append(" & ".join([
            m["method"],
            f"{m['ld']:.1f}",
            ia.f(m["T_r"], 2),
            ia.f(m["T_s"], 2),
            ia.f(m["M_os"], 4),
            ia.f(m["d_lead"], 2),
            ia.f(m["e_y_in"], 4),
            ia.f(m["e_y_out"], 4),
            ia.f(m["omega_rate_max_transition"], 3),
        ]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path_out.write_text("\n".join(lines), encoding="utf-8")


def _arm_style(method, ld):
    color = ia.METHOD_COLORS[method]
    if math.isclose(ld, max(LOOKAHEADS)):
        return dict(color=color, lw=1.4, ls="-")
    return dict(color=color, lw=1.2, ls=(0, (4, 1.5)))


def make_figure(fig_dir, path, traces):
    plt.rcParams.update({"font.size": 9})
    fig, (ax_e, ax_k) = plt.subplots(2, 1, figsize=(4.2, 4.6))
    for method, ld in ARMS:
        trace = traces[arm_key(method, ld)]
        ax_e.plot(trace.path_s, trace.e_y, label=f"{method}, $L_d={ld:.1f}$ m",
                  **_arm_style(method, ld))
    for s_mark in (path.s_entry, path.s_exit):
        ax_e.axvline(s_mark, color="0.5", lw=0.7, ls=":")
    ax_e.axhline(0.0, color="0.5", lw=0.7, ls=":")
    ax_e.set_xlim(0.0, path.goal)
    ax_e.set_xlabel(r"Reference arc length $s$ [m]")
    ax_e.set_ylabel(r"$e_y$ [m]")
    ax_e.set_title("(a) Lateral error (arc entry and exit dotted)",
                   fontsize=8, loc="left")
    ax_e.legend(fontsize=6.5, loc="upper right", framealpha=0.85, ncol=2)

    s_ff, ff = feedforward_profiles(path)
    ref = np.array([path.curvature(v) for v in s_ff])
    ax_k.plot(s_ff, ref, "k--", lw=1.0, label=r"Reference $\kappa_r$")
    for ld, lw in zip(LOOKAHEADS, (0.9, 1.3)):
        ax_k.plot(s_ff, ff[ld], color="0.45", lw=lw, ls=":",
                  label=rf"$\kappa_{{\rm prev}}$, $L_d={ld:.1f}$ m")
    for method, ld in ARMS:
        trace = traces[arm_key(method, ld)]
        win = ((trace.path_s >= KAPPA_PANEL_WINDOW[0])
               & (trace.path_s <= KAPPA_PANEL_WINDOW[1]))
        ax_k.plot(trace.path_s[win], trace.curvature[win],
                  **_arm_style(method, ld))
    ax_k.axvline(path.s_entry, color="0.5", lw=0.7, ls=":")
    ax_k.set_xlim(*KAPPA_PANEL_WINDOW)
    ax_k.set_xlabel(r"Reference arc length $s$ [m]")
    ax_k.set_ylabel(r"$\kappa$ [1/m]")
    ax_k.set_title("(b) Curvature command around the arc entry",
                   fontsize=8, loc="left")
    ax_k.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
    for ax in (ax_e, ax_k):
        ax.grid(True, color="0.9", lw=0.6, ls=":")
        ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.3)
    written = []
    for ext in ("pdf", "png"):
        out = fig_dir / f"{FIGURE_STEM}.{ext}"
        fig.savefig(out, dpi=300)
        written.append(out)
    plt.close(fig)
    return written


def write_traces(trace_dir, traces):
    trace_dir.mkdir(parents=True, exist_ok=True)
    for key, trace in traces.items():
        rows = np.column_stack([
            trace.time, trace.pose, trace.path_s, trace.e_y, trace.e_psi,
            trace.curvature, trace.omega_cmd, trace.omega_raw, trace.sigma,
        ])
        np.savetxt(
            trace_dir / f"sim_test3_{key}.csv",
            rows,
            delimiter=",",
            header="t,x,y,psi,path_s,e_y,e_psi,kappa,omega_cmd,omega_raw,sigma",
            comments="",
            fmt="%.9g",
        )


def run_test3(table_dir, fig_dir, trace_dir):
    path, traces, metrics, consistency = simulate_arms()
    write_table(table_dir / f"{TABLE_STEM}.tex", metrics)
    figures = make_figure(fig_dir, path, traces)
    write_traces(trace_dir, traces)
    s_ff, ff = feedforward_profiles(path, spacing=0.02)
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
            "omega_n": OMEGA_N,
            "omega_n_name": "omega_n_max_ld_1p0 (hardware design point)",
            "zeta": ZETA,
            "initial_condition": list(COND),
            "lookaheads_m": list(LOOKAHEADS),
            "methods": list(METHODS),
            "design": "preview range (L_d) versus local response (omega_n, zeta)",
        },
        "path": {
            "shape": "straight -> left arc (no transition curve) -> straight",
            "lead_in_m": LEAD_IN,
            "arc_radius_m": ARC_R,
            "arc_angle_deg": math.degrees(ARC_ANGLE),
            "exit_control_m": EXIT_CONTROL,
            "exit_eval_m": EXIT_EVAL,
            "s_entry_m": path.s_entry,
            "s_exit_m": path.s_exit,
            "goal_m": path.goal,
            "control_length_m": path.length,
            "arc_vertex_spacing_m": ARC_VERTEX_SPACING,
        },
        "windows": {
            "recovery_m": [0.0, RECOVERY_GOAL],
            "entry_m": list(ENTRY_WINDOW),
            "exit_m": list(EXIT_WINDOW),
            "transition_m": list(TRANSITION_WINDOW),
            "lead_search_start_m": LEAD_SEARCH_START,
            "lead_threshold_fraction_of_arc_curvature": LEAD_THRESHOLD_FRACTION,
            "sign_convention": "e_y > 0 is left of the path = inside of the left turn",
        },
        "arms": metrics,
        "consistency": consistency,
        "feedforward": {
            "s_m": s_ff.tolist(),
            "kappa_prev": {f"ld{ld:.2f}".replace(".", "p"): ff[ld].tolist()
                           for ld in LOOKAHEADS},
        },
    }
    (table_dir / f"{METRICS_STEM}.json").write_text(
        json.dumps(ia._clean(out), indent=2), encoding="utf-8"
    )
    return out, figures


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
        help="paper project root (default: current directory); outputs go to "
             "<out-root>/generated_preview/ unless --apply is given",
    )
    ap.add_argument("--tag", type=str, default=None,
                    help="write into a tagged subdirectory of "
                         "generated_preview/ (preview only)")
    args = ap.parse_args(argv)
    if args.tag and args.apply:
        ap.error("--tag is preview-only; do not combine with --apply")

    sub = "generated" if args.apply else "generated_preview"
    table_dir = args.out_root / sub / "tables"
    fig_dir = args.out_root / sub / "figures"
    trace_dir = args.out_root / sub / "traces"
    if args.tag:
        table_dir, fig_dir, trace_dir = (
            table_dir / args.tag, fig_dir / args.tag, trace_dir / args.tag
        )
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Out root: {args.out_root / sub}  (apply={args.apply})")
    print(f"Config: omega_n = {OMEGA_N:.6f} rad/s, zeta = {ZETA}, "
          f"lookaheads = {LOOKAHEADS} m, R = {ARC_R} m, "
          f"entry s = {S_ENTRY:.3f} m, exit s = {S_EXIT:.3f} m")
    out, figures = run_test3(table_dir, fig_dir, trace_dir)
    print("\n*** TEST 3 preview decoupling ***")
    print(f"{'arm':14s} {'T_r':>5s} {'T_s':>5s} {'M_os':>6s} {'d_lead':>6s} "
          f"{'e_y,in':>7s} {'e_y,out':>7s} {'wdot':>6s} {'sat':>4s}")
    for m in out["arms"]:
        print(f"{m['method'] + ' Ld=' + str(m['ld']):14s} "
              f"{ia.f(m['T_r'], 2):>5s} {ia.f(m['T_s'], 2):>5s} "
              f"{ia.f(m['M_os'], 4):>6s} {ia.f(m['d_lead'], 2):>6s} "
              f"{ia.f(m['e_y_in'], 4):>7s} {ia.f(m['e_y_out'], 4):>7s} "
              f"{ia.f(m['omega_rate_max_transition'], 3):>6s} "
              f"{m['sat_ratio']:4.2f}")
    worst = max(c["max_abs_dev"] for c in out["consistency"].values())
    print(f"recovery vs Test-1 straight: max |dev| = {worst:.2e} "
          f"({'ok' if worst <= DT + 1e-9 else 'MISMATCH'})")
    for path_out in figures:
        print(f"Generated {path_out}")
    print(f"Generated {table_dir / (TABLE_STEM + '.tex')}")
    print(f"Generated {table_dir / (METRICS_STEM + '.json')}")


if __name__ == "__main__":
    main()

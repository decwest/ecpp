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
* Path: 2 m straight -> R = 1.0 m left arc over 90 deg (the corner radius of
  the hardware route) -> 8 m straight; the evaluation ends 6 m after the arc
  exit so the carrot never pins at the end.
* Initial condition (e_y(0), e_psi(0)) = (0, 0): the robot starts on the
  path, so every deviation is forced by the curvature transitions.
* Arms: PP and ECPP at L_d in {0.5, 1.0} m.  ECPP uses the Test-2 / hardware
  design point (omega_n, zeta) = (omega_n_max(L_d = 1.0 m), 1) for both
  lookaheads, which is inside the rate bound for both.

Metrics
-------
The common chapter-5 metrics, with the reference amplitude taken as the peak
deviation after the curvature change: e_y_in is the maximum signed lateral
error from 0.5 m before the entry to the exit (the inside of the left turn is
+e_y); e_y_out is the minimum signed lateral error from 0.5 m before the exit
to the evaluation end (the outside is -e_y); T_s_02_out is the settling time
from the exit peak until |e_y| enters and stays within 2 % of that peak up to
the evaluation end; bar_e_y and bar_e_theta_deg are the mean absolute errors
over the evaluation interval; N_zc_exit counts the path crossings of e_y after
the exit peak (1 mm dead band); kappa_max and sat_ratio are the unclipped
curvature maximum and the saturation ratio.  d_lead (first |kappa_des| >= 10 %
of the arc curvature before the entry) stays in the JSON for the text, and the
feedforward curvature kappa_prev(s) of a zero-error carrot L_d ahead is stored
for the figure.

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

OMEGA_N = ia.SPEED_OMEGA_N_MAX     # 1.029884 rad/s (at v0), the Test-2 / hardware design point
ZETA = 1.0
COND = (0.0, 0.0)                  # start on the path
LOOKAHEADS = (0.5, 1.0)
METHODS = ("PP", "ECPP")
ARMS = tuple((method, ld) for method in METHODS for ld in LOOKAHEADS)

LEAD_IN = 2.0                      # m straight before the arc
ARC_R = 1.0                        # m, the corner radius of the hardware route (exp. 2)
ARC_ANGLE = math.pi / 2.0          # left 90 deg turn, no transition curve
EXIT_CONTROL = 8.0                 # m of straight after the arc (control path)
EXIT_EVAL = 6.0                    # m of straight after the arc (evaluation)
ARC_VERTEX_SPACING = 0.005         # m, as in ieee_access.ArcPath

S_ENTRY = LEAD_IN
S_EXIT = LEAD_IN + ARC_R * ARC_ANGLE
GOAL = S_EXIT + EXIT_EVAL
ENTRY_MARGIN = 0.5                 # m before the entry where the in-cut may start
EXIT_MARGIN = 0.5                  # m before the exit where the out-flow may start
SETTLING_FRACTION = 0.02           # T_s band, of the exit peak (chapter-5 convention)
ZC_DEADBAND = 0.001                # m, dead band of the path-crossing count
LEAD_SEARCH_START = S_ENTRY - 1.5  # m
LEAD_THRESHOLD_FRACTION = 0.1      # of the arc curvature (JSON only)
FIGURE_STEM = "sim_test3_preview_decoupling"
TABLE_STEM = "sim_test3_results"
METRICS_STEM = "sim_test3_metrics"
KAPPA_PANEL_WINDOW = (S_ENTRY - 1.5, S_EXIT + 1.0)
XY_PANEL_LIMITS = ((0.8, 3.6), (-0.4, 3.4))
XY_INSET_LIMITS = ((2.35, 3.35), (0.2, 1.7))
XY_INSET_POSITION = [0.13, 0.46, 0.50, 0.50]   # axes fraction, upper-left void


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
        omega_n=ia.configured_omega_n(OMEGA_N),
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


def _settling_time(trace, peak_index, s_limit, fraction=SETTLING_FRACTION):
    """Time from the peak until |e_y| enters the band and stays there to s_limit."""
    e = trace.e_y
    s = trace.path_s
    band = fraction * abs(e[peak_index])
    idx = np.where((np.arange(len(e)) > peak_index) & (s <= s_limit))[0]
    if not len(idx):
        return None
    inside = np.abs(e[idx]) <= band
    if not inside[-1]:
        return None
    outside = np.where(~inside)[0]
    k = idx[outside[-1] + 1] if len(outside) else idx[0]
    return float(trace.time[k] - trace.time[peak_index])


def _path_crossings(trace, start_index, deadband=ZC_DEADBAND):
    """Sign reversals of e_y after start_index, ignoring |e_y| <= deadband."""
    sign = 0
    count = 0
    for value in trace.e_y[start_index:]:
        if abs(value) <= deadband:
            continue
        current = 1 if value > 0.0 else -1
        if sign and current != sign:
            count += 1
        sign = current
    return count


def transition_metrics(trace, path):
    s = trace.path_s
    kappa = trace.curvature
    e = trace.e_y
    m = {}
    search = (s >= LEAD_SEARCH_START) & (s <= path.s_entry)
    onset = np.where(search & (np.abs(kappa) >= LEAD_THRESHOLD_FRACTION
                               / path.radius))[0]
    m["d_lead"] = float(path.s_entry - s[onset[0]]) if len(onset) else None

    entry = np.where((s >= path.s_entry - ENTRY_MARGIN) & (s <= path.s_exit))[0]
    i_in = int(entry[np.argmax(e[entry])])
    m["e_y_in"] = float(e[i_in])
    m["s_in"] = float(s[i_in])

    exit_ = np.where((s >= path.s_exit - EXIT_MARGIN) & (s <= path.goal))[0]
    i_out = int(exit_[np.argmin(e[exit_])])
    m["e_y_out"] = float(e[i_out])
    m["s_out"] = float(s[i_out])
    m["T_s_02_out"] = _settling_time(trace, i_out, path.goal)
    m["N_zc_exit"] = int(_path_crossings(trace, i_out))
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
    m.update(transition_metrics(trace, path))
    m["bar_e_y"] = float(np.mean(np.abs(trace.e_y[evalmask])))
    m["bar_e_theta_deg"] = float(np.degrees(np.mean(np.abs(trace.e_psi[evalmask]))))
    m["kappa_max"] = float(np.max(np.abs(trace.curvature[evalmask])))
    m["sat_ratio"] = float(np.mean(np.abs(trace.omega_raw[evalmask]) >= OMEGA_MAX))
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
    """Run the four frozen arms once per process (traces and metrics)."""
    path = StraightArcStraightPath()
    polyline = path.control_polyline()
    traces, metrics = {}, []
    for method, ld in ARMS:
        key = arm_key(method, ld)
        trace = _run(polyline, path, method, ld)
        traces[key] = trace
        metrics.append(dict(arm_metrics(trace, path, method, ld), key=key))
    return path, traces, metrics


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
        r"Method & $L_d$ [m] & $e_{y,\mathrm{in}}$ [m] & $e_{y,\mathrm{out}}$ [m] & "
        r"$T_s^{2\%}$ [s] & $\bar e_y$ [m] & $\bar e_\theta$ [$^\circ$] & "
        r"$N_{\mathrm{zc}}$ & $\kappa_{\max}$ [1/m] \\",
        r"\midrule",
    ]
    for m in metrics:
        lines.append(" & ".join([
            m["method"],
            f"{m['ld']:.1f}",
            ia.f(m["e_y_in"], 3),
            ia.f(m["e_y_out"], 3),
            ia.f(m["T_s_02_out"], 2, none="n/r"),
            ia.f(m["bar_e_y"], 3),
            ia.f(m["bar_e_theta_deg"], 2),
            str(m["N_zc_exit"]),
            ia.f(m["kappa_max"], 2),
        ]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path_out.write_text("\n".join(lines), encoding="utf-8")


def _arm_style(method, ld):
    color = ia.METHOD_COLORS[method]
    if math.isclose(ld, max(LOOKAHEADS)):
        return dict(color=color, lw=1.0, ls="-")
    return dict(color=color, lw=0.9, ls=(0, (4, 1.5)))


def _draw_trajectory(ax, path, traces, ref, inset_ticksize=5.5):
    """(a) trajectories in the corner region with an inset at the arc entry."""
    ax.plot(ref[:, 0], ref[:, 1], "k--", lw=0.8, label="Reference")
    for method, ld in ARMS:
        trace = traces[arm_key(method, ld)]
        ax.plot(trace.pose[:, 0], trace.pose[:, 1],
                label=f"{method}, $L_d={ld:.1f}$ m", **_arm_style(method, ld))
    (x_lo, x_hi), (y_lo, y_hi) = XY_PANEL_LIMITS
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    axins = ax.inset_axes(XY_INSET_POSITION)
    axins.plot(ref[:, 0], ref[:, 1], "k--", lw=0.8)
    for method, ld in ARMS:
        trace = traces[arm_key(method, ld)]
        axins.plot(trace.pose[:, 0], trace.pose[:, 1], **_arm_style(method, ld))
    (ix_lo, ix_hi), (iy_lo, iy_hi) = XY_INSET_LIMITS
    axins.set_xlim(ix_lo, ix_hi)
    axins.set_ylim(iy_lo, iy_hi)
    axins.set_aspect("equal", adjustable="box")
    axins.tick_params(labelsize=inset_ticksize, length=2, pad=1)
    axins.grid(True, color="0.9", lw=0.5, ls=":")
    ax.indicate_inset_zoom(axins, edgecolor="0.4", lw=0.7)


def _draw_lateral_error(ax, path, traces):
    """(b) lateral error versus arc length (entry and exit dotted)."""
    for method, ld in ARMS:
        trace = traces[arm_key(method, ld)]
        ax.plot(trace.path_s, trace.e_y, label=f"{method}, $L_d={ld:.1f}$ m",
                **_arm_style(method, ld))
    for s_mark in (path.s_entry, path.s_exit):
        ax.axvline(s_mark, color="0.5", lw=0.7, ls=":")
    ax.axhline(0.0, color="0.5", lw=0.7, ls=":")
    ax.set_xlim(0.0, path.goal)
    ax.set_xlabel(r"Reference arc length $s$ [m]")
    ax.set_ylabel(r"$e_y$ [m]")


def _draw_curvature(ax, path, traces):
    """(c) curvature command around the arc entry with the feedforward previews."""
    s_ff, ff = feedforward_profiles(path)
    ref_k = np.array([path.curvature(v) for v in s_ff])
    ax.plot(s_ff, ref_k, "k--", lw=1.0, label=r"Reference $\kappa_r$")
    for ld, lw in zip(LOOKAHEADS, (0.8, 1.1)):
        ax.plot(s_ff, ff[ld], color="0.45", lw=lw, ls=":",
                label=rf"$\kappa_{{\mathrm{{prev}}}}$, $L_d={ld:.1f}$ m")
    for method, ld in ARMS:
        trace = traces[arm_key(method, ld)]
        win = ((trace.path_s >= KAPPA_PANEL_WINDOW[0])
               & (trace.path_s <= KAPPA_PANEL_WINDOW[1]))
        ax.plot(trace.path_s[win], trace.curvature[win],
                **_arm_style(method, ld))
    ax.axvline(path.s_entry, color="0.5", lw=0.7, ls=":")
    ax.set_xlim(*KAPPA_PANEL_WINDOW)
    ax.set_xlabel(r"Reference arc length $s$ [m]")
    ax.set_ylabel(r"$\kappa$ [1/m]")


def _legend_entries(ax_k, ax_e):
    k_handles, k_labels = ax_k.get_legend_handles_labels()
    e_handles, e_labels = ax_e.get_legend_handles_labels()
    handles = k_handles + e_handles
    labels = ["Reference" if lab.startswith("Reference") else lab
              for lab in k_labels] + e_labels
    return handles, labels


def make_figure(fig_dir, path, traces):
    """Write the combined three-panel figure and, for the manuscript, one file
    per panel (no titles or legends) plus a legend strip so that LaTeX can
    attach sub-captions."""
    ref = path.sample(800)
    written = []

    # combined figure (deck / preview)
    plt.rcParams.update({"font.size": 8})
    fig, (ax_xy, ax_e, ax_k) = plt.subplots(
        1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]}
    )
    _draw_trajectory(ax_xy, path, traces, ref)
    _draw_lateral_error(ax_e, path, traces)
    _draw_curvature(ax_k, path, traces)
    ax_xy.set_title("(a) Trajectories (inset: corner)", fontsize=8, loc="left")
    ax_e.set_title("(b) Lateral error (entry and exit dotted)", fontsize=8,
                   loc="left")
    ax_k.set_title("(c) Curvature command (corner)", fontsize=8, loc="left")
    for ax in (ax_xy, ax_e, ax_k):
        ax.grid(True, color="0.9", lw=0.6, ls=":")
        ax.tick_params(labelsize=7)
    handles, labels = _legend_entries(ax_k, ax_e)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0.0, 0.0, 1.0, 0.92))
    for ext in ("pdf", "png"):
        out = fig_dir / f"{FIGURE_STEM}.{ext}"
        fig.savefig(out, dpi=300)
        written.append(out)
    plt.close(fig)

    # legend strip
    lfig = plt.figure(figsize=(7.0, 0.3))
    lfig.legend(handles, labels, loc="center", ncol=len(labels), fontsize=6.5,
                frameon=False, handlelength=2.2, columnspacing=1.2)
    for ext in ("pdf", "png"):
        out = fig_dir / f"sim_test3_legend.{ext}"
        lfig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        written.append(out)
    plt.close(lfig)

    # per-panel files for LaTeX sub-captions
    plt.rcParams.update({"font.size": 7})
    panels = (
        ("trajectory", (2.0, 1.9), lambda ax: _draw_trajectory(ax, path, traces, ref, inset_ticksize=4.5)),
        ("lateral_error", (2.35, 1.8), lambda ax: _draw_lateral_error(ax, path, traces)),
        ("curvature", (2.35, 1.8), lambda ax: _draw_curvature(ax, path, traces)),
    )
    for name, size, draw in panels:
        pfig, pax = plt.subplots(figsize=size)
        draw(pax)
        pax.grid(True, color="0.9", lw=0.6, ls=":")
        pax.tick_params(labelsize=6, pad=1.5)
        pax.xaxis.label.set_size(7)
        pax.yaxis.label.set_size(7)
        pfig.tight_layout(pad=0.25)
        for ext in ("pdf", "png"):
            out = fig_dir / f"sim_test3_{name}.{ext}"
            pfig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
            written.append(out)
        plt.close(pfig)
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
    path, traces, metrics = simulate_arms()
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
            "omega_n_configured": ia.configured_omega_n(OMEGA_N),
            "omega_n_basis": "natural frequency at v0; plugin configured value = omega_n*(v0+v_epsilon)/v0",
            "omega_n_name": "omega_n_max_ld_1p0 (Test-2 / hardware design point)",
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
        "metric_definitions": {
            "e_y_in": "max e_y on [s_entry - entry_margin, s_exit] (inside of the left turn is +)",
            "e_y_out": "min e_y on [s_exit - exit_margin, goal] (outside is -)",
            "T_s_02_out": ("time from the e_y_out peak until |e_y| enters and stays "
                           "within settling_fraction * |e_y_out| up to the goal"),
            "bar_e_y": "mean |e_y| over [0, goal]",
            "bar_e_theta_deg": "mean |e_theta| over [0, goal], degrees",
            "N_zc_exit": "sign reversals of e_y after the e_y_out peak, |e_y| <= zc_deadband ignored",
            "d_lead": ("s_entry minus the first s (from lead_search_start) at "
                       "which |kappa_des| >= lead_threshold_fraction / R; JSON only"),
            "lead_search_start_m": LEAD_SEARCH_START,
            "lead_threshold_fraction_of_arc_curvature": LEAD_THRESHOLD_FRACTION,
            "entry_margin_m": ENTRY_MARGIN,
            "exit_margin_m": EXIT_MARGIN,
            "settling_fraction": SETTLING_FRACTION,
            "zc_deadband_m": ZC_DEADBAND,
            "sign_convention": "e_y > 0 is left of the path = inside of the left turn",
        },
        "arms": metrics,
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
          f"entry s = {S_ENTRY:.3f} m, exit s = {S_EXIT:.3f} m, "
          f"initial condition = {COND}")
    out, figures = run_test3(table_dir, fig_dir, trace_dir)
    print("\n*** TEST 3 preview decoupling ***")
    print(f"{'arm':14s} {'d_lead':>6s} {'e_y,in':>7s} {'e_y,out':>7s} "
          f"{'Ts2%out':>7s} {'bar_ey':>7s} {'bar_eth':>7s} {'Nzc':>3s} "
          f"{'kmax':>5s} {'sat':>4s}")
    for m in out["arms"]:
        print(f"{m['method'] + ' Ld=' + str(m['ld']):14s} "
              f"{ia.f(m['d_lead'], 2, none='n/r'):>6s} "
              f"{ia.f(m['e_y_in'], 4):>7s} "
              f"{ia.f(m['e_y_out'], 4):>7s} "
              f"{ia.f(m['T_s_02_out'], 2, none='n/r'):>7s} "
              f"{ia.f(m['bar_e_y'], 4):>7s} "
              f"{ia.f(m['bar_e_theta_deg'], 2):>7s} "
              f"{m['N_zc_exit']:3d} "
              f"{m['kappa_max']:5.2f} {m['sat_ratio']:4.2f}")
    for path_out in figures:
        print(f"Generated {path_out}")
    print(f"Generated {table_dir / (TABLE_STEM + '.tex')}")
    print(f"Generated {table_dir / (METRICS_STEM + '.json')}")


if __name__ == "__main__":
    main()

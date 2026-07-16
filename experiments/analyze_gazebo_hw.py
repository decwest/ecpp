"""Analyze Gazebo hardware-plan experiments (Exp1 L-path, Exp2 comparison).

Reads the CSV trials recorded by ytlab2_whill's ecpp_scenario_runner /
nav_comparison_runner and produces IEEE-styled figures and LaTeX tables in
the paper's generated/ directories.

Usage:
  uv run python experiments/analyze_gazebo_hw.py \
      --data-root ../dwpp_test_simulation/data \
      --tex-project ../tex_docker_environment/projects/fumiya_ieee_access
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from ecpp.visualization.plots import save_figure  # also applies IEEE rcParams

import matplotlib.pyplot as plt


# ----------------------------------------------------------------- loading

def load_trial(path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for name in fields:
            cols[name] = []
        for row in reader:
            for name in fields:
                value = row.get(name, "")
                cols[name].append(float(value) if value not in ("", None) else math.nan)
    return {name: np.asarray(values) for name, values in cols.items()}


def load_polyline(path: Path) -> np.ndarray:
    pts = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row["x"]), float(row["y"])))
    return np.asarray(pts)


def project_onto_path(xy: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed lateral error (positive = left of path direction) and arc-length
    position of the nearest-point projection for each xy sample."""
    seg_a = ref[:-1]
    seg_b = ref[1:]
    seg_v = seg_b - seg_a
    seg_len2 = np.maximum((seg_v ** 2).sum(axis=1), 1e-12)
    seg_len = np.sqrt(seg_len2)
    seg_s0 = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]
    e_out = np.empty(len(xy))
    s_out = np.empty(len(xy))
    for i, p in enumerate(xy):
        t = np.clip(((p - seg_a) * seg_v).sum(axis=1) / seg_len2, 0.0, 1.0)
        proj = seg_a + seg_v * t[:, None]
        d2 = ((p - proj) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        tangent = seg_v[j] / seg_len[j]
        delta = p - proj[j]
        e_out[i] = tangent[0] * delta[1] - tangent[1] * delta[0]
        s_out[i] = seg_s0[j] + t[j] * seg_len[j]
    return e_out, s_out


def signed_lateral_errors(xy: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return project_onto_path(xy, ref)[0]


# ------------------------------------------------------------ exp1 metrics

CORNER = np.array([-31.0, 3.0])
POST_CORNER_HEADING = 0.0  # +x
EVAL_GOAL_X = -26.0        # 5 m after the corner (control path extends to -24)

ECPP_RE = re.compile(r"ECPP_W(\d{3})_Z(\d{4}|\d{3})")


def controller_params(controller: str) -> tuple[float | None, float | None]:
    m = ECPP_RE.match(controller)
    if not m:
        return None, None
    omega_n = int(m.group(1)) / 100.0
    zeta_raw = m.group(2)
    zeta = int(zeta_raw) / (1000.0 if len(zeta_raw) == 4 else 100.0)
    return omega_n, zeta


def exp1_trial_metrics(trial: dict[str, np.ndarray], ref: np.ndarray,
                       status: str) -> dict[str, float | str]:
    t = trial["t"]
    xy = np.stack([trial["x"], trial["y"]], axis=1)
    # evaluate up to the eval goal (5 m past the corner)
    reached = np.where(xy[:, 0] >= EVAL_GOAL_X)[0]
    end = int(reached[0]) + 1 if len(reached) else len(t)
    xy_eval = xy[:end]
    t_eval = t[:end] - t[0]
    e_y, s_proj = project_onto_path(xy_eval, ref)

    # post-corner response: samples after the projection passes the corner
    # (corner is at arc length 2.0 m on the reference path)
    post_idx = np.where(s_proj > 2.05)[0]
    overshoot = math.nan
    settle = math.nan
    if len(post_idx):
        i0 = int(post_idx[0])
        e_c = xy_eval[i0:, 1] - CORNER[1]
        t_c = t_eval[i0:]
        # overshoot beyond the post-corner line, relative to the approach side
        sign0 = 1.0 if len(e_c) == 0 or e_c[0] >= 0.0 else -1.0
        overshoot = max(0.0, float(np.max(-sign0 * e_c))) if len(e_c) else math.nan
        ok = np.abs(e_c) <= 0.05
        hold = np.flip(np.cumprod(np.flip(ok).astype(int))).astype(bool)
        idx = np.where(hold)[0]
        if len(idx):
            settle = float(t_c[idx[0]] - t_c[0])

    # capture time: |e_y| first below 0.1 m (far conditions)
    capture = math.nan
    below = np.where(np.abs(e_y) <= 0.10)[0]
    if len(below):
        capture = float(t_eval[below[0]])

    # Saturation demand is defined on the raw controller request.  w_nav and
    # /cmd_vel_smoothed remain diagnostic signals only.
    w_cmd = trial.get("w_cmd", np.array([]))
    w_valid = w_cmd[np.isfinite(w_cmd)] if len(w_cmd) else np.array([])
    sat = float(np.mean(np.abs(w_valid) >= 1.0)) if len(w_valid) else math.nan

    sigma = trial.get("sigma", np.array([]))
    sigma_valid = sigma[np.isfinite(sigma)] if len(sigma) else np.array([])

    return {
        "success": 1.0 if status == "succeeded" else 0.0,
        "mean_abs_e_y": float(np.mean(np.abs(e_y))),
        "max_abs_e_y": float(np.max(np.abs(e_y))),
        "overshoot_c": overshoot,
        "settle_c": settle,
        "capture_time": capture,
        "travel_time": float(t_eval[-1]),
        "omega_sat": sat,
        "min_sigma": float(np.min(sigma_valid)) if len(sigma_valid) else math.nan,
    }


def fmt(value, digits=3, dash="--"):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return dash
    return f"{value:.{digits}f}"


def analyze_exp1(data_dir: Path, fig_dir: Path, table_dir: Path) -> None:
    summary = json.loads((data_dir / "summary.json").read_text()) \
        if (data_dir / "summary.json").exists() else {"trials": []}
    status_by_label = {row["label"]: row["status"] for row in summary.get("trials", [])}
    # orchestrated runs write one summary per invocation; rebuild from files
    ref = load_polyline(data_dir / "reference_path.csv")

    trials = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path.name == "reference_path.csv":
            continue
        label = csv_path.stem
        m = re.match(r"(.+)_ey([m0-9p]+)_epsi([m0-9p]+)_t(\d+)$", label)
        if not m:
            continue
        controller = m.group(1)
        decode = lambda s: float(s.replace("m", "-").replace("p", "."))
        e_y0 = decode(m.group(2))
        e_psi0 = decode(m.group(3))
        data = load_trial(csv_path)
        if len(data.get("t", [])) < 10:
            continue
        status = status_by_label.get(label, "succeeded")
        metrics = exp1_trial_metrics(data, ref, status)
        trials[(controller, e_y0, e_psi0)] = (data, metrics)

    if not trials:
        print("exp1: no trials found")
        return

    controllers = sorted({key[0] for key in trials},
                         key=lambda c: (c != "PP", c != "DWPP", c))

    # ------------------------------------------------ nominal results table
    lines = [
        r"\begin{tabular}{llllllll}",
        r"\toprule",
        r"Method & $\omega_n$ & $\zeta$ & $\bar e_y$ & $M_{\rm os}^{\rm c}$ & "
        r"$T_s^{\rm c}$ & $T_m$ & sat.\,[\%] \\",
        r"\midrule",
    ]
    for controller in controllers:
        entry = trials.get((controller, 0.0, 0.0))
        if entry is None:
            continue
        _, met = entry
        omega_n, zeta = controller_params(controller)
        name = "ECPP" if controller.startswith("ECPP") else controller
        lines.append(" & ".join([
            name,
            fmt(omega_n, 2), fmt(zeta, 3),
            fmt(met["mean_abs_e_y"], 3),
            fmt(met["overshoot_c"], 3),
            fmt(met["settle_c"], 2),
            fmt(met["travel_time"], 2),
            fmt(100 * met["omega_sat"], 1),
        ]) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (table_dir / "gazebo_exp1_results.tex").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------- far-capture table
    lines = [
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Method & $e_y(0)$ & $e_\theta(0)$ & reach & $T_{10}$ & "
        r"$M_{\rm os}^{\rm c}$ & $\bar e_y$ \\",
        r"\midrule",
    ]
    for controller in controllers:
        omega_n_c, zeta_c = controller_params(controller)
        if zeta_c is not None and abs(zeta_c - 1.0) > 1e-9:
            continue  # keep the table compact: ECPP zeta = 1.0 only
        for (ctrl, e_y0, e_psi0), (_, met) in sorted(trials.items()):
            if ctrl != controller or (e_y0 == 0.0 and e_psi0 == 0.0):
                continue
            name = "ECPP" if controller.startswith("ECPP") else controller
            omega_n, zeta = controller_params(controller)
            if omega_n is not None:
                name = f"ECPP ({omega_n:.2f}, {zeta:.3g})"
            lines.append(" & ".join([
                name,
                fmt(e_y0, 1), fmt(e_psi0, 0),
                "yes" if met["success"] else "no",
                fmt(met["capture_time"], 2),
                fmt(met["overshoot_c"], 3),
                fmt(met["mean_abs_e_y"], 3),
            ]) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (table_dir / "gazebo_exp1_far_results.tex").write_text("\n".join(lines), encoding="utf-8")

    # ----------------------------------------------------------- figures
    # (a) nominal: PP, DWPP, ECPP zeta=1.0 sweep of omega_n
    def style_for(controller, palette):
        if controller == "PP":
            return {"color": "black", "linestyle": "-", "linewidth": 1.2}
        if controller == "DWPP":
            return {"color": "0.45", "linestyle": "--", "linewidth": 1.2}
        omega_n, zeta = controller_params(controller)
        idx = palette.setdefault(controller, len(palette))
        return {"color": plt.get_cmap("tab10")(idx % 10),
                "linestyle": ["-", "--", "-."][idx % 3], "linewidth": 1.0}

    def label_for(controller):
        omega_n, zeta = controller_params(controller)
        if omega_n is None:
            return controller
        return fr"ECPP $\omega_n$={omega_n:.2f} $\zeta$={zeta:.3g}"

    for cond, cond_key, xlim in (
        ((0.0, 0.0), "nominal", (-31.8, -25.5)),
        ((1.0, 0.0), "far_ey", (-31.8, -25.5)),
        ((1.0, -90.0), "far_ey_epsi", (-31.8, -25.5)),
    ):
        selected = [c for c in controllers
                    if (c, cond[0], cond[1]) in trials]
        if not selected:
            continue
        zeta_filter = {None, 1.0}
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6))
        ax_traj, ax_ey, ax_w, ax_sigma = axes.ravel()
        palette: dict[str, int] = {}
        for controller in selected:
            omega_n, zeta = controller_params(controller)
            if zeta_filter is not None and zeta not in zeta_filter:
                continue
            data, _ = trials[(controller, cond[0], cond[1])]
            style = style_for(controller, palette)
            tt = data["t"] - data["t"][0]
            ax_traj.plot(data["x"], data["y"], label=label_for(controller), **style)
            e_y = signed_lateral_errors(
                np.stack([data["x"], data["y"]], axis=1), ref)
            ax_ey.plot(tt, e_y, **style)
            ax_w.plot(tt, data.get("w_nav", np.full_like(tt, np.nan)), **style)
            sigma = data.get("sigma")
            if sigma is not None and np.isfinite(sigma).any():
                ax_sigma.plot(tt, sigma, **style)
        ax_traj.plot(ref[:, 0], ref[:, 1], "k--", linewidth=1.0, label="Reference")
        ax_traj.set_xlabel(r"$x$ [m]"); ax_traj.set_ylabel(r"$y$ [m]")
        ax_traj.set_xlim(*xlim); ax_traj.set_aspect("equal", adjustable="box")
        ax_ey.set_xlabel(r"$t$ [s]"); ax_ey.set_ylabel(r"$e_y$ [m]")
        ax_w.set_xlabel(r"$t$ [s]"); ax_w.set_ylabel(r"$\omega$ [rad/s]")
        ax_w.axhline(1.0, color="0.6", linewidth=0.7, linestyle=":")
        ax_w.axhline(-1.0, color="0.6", linewidth=0.7, linestyle=":")
        ax_sigma.set_xlabel(r"$t$ [s]"); ax_sigma.set_ylabel(r"$\sigma$ [-]")
        ax_sigma.set_ylim(-0.05, 1.05)
        for ax in axes.ravel():
            ax.grid(True, linestyle=":", linewidth=0.5)
        handles, labels = ax_traj.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7,
                   frameon=True, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        save_figure(fig, fig_dir / f"gazebo_exp1_{cond_key}")
    print(f"exp1: {len(trials)} trials analyzed")


# ------------------------------------------------------------ exp2 metrics

def analyze_exp2(data_dir: Path, fig_dir: Path, table_dir: Path) -> None:
    scenarios = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    if not scenarios:
        print("exp2: no scenario directories found")
        return
    summary_rows = []
    for scen_dir in scenarios:
        for csv_path in sorted(scen_dir.glob("*.csv")):
            m = re.match(r"(?P<scen>.+)_(?P<ctrl>ECPP_RAW|ECPP|DWPP|MPPI|DWB)_t(?P<trial>\d+)$",
                         csv_path.stem)
            if not m:
                continue
            plan_path = scen_dir / f"plan_{csv_path.stem}.csv"
            if not plan_path.exists():
                continue
            data = load_trial(csv_path)
            if len(data.get("t", [])) < 10:
                continue
            plan = load_polyline(plan_path)
            xy = np.stack([data["x"], data["y"]], axis=1)
            e = signed_lateral_errors(xy, plan)
            w_cmd = data.get("w_cmd", np.array([]))
            t = data["t"]
            dw = (np.abs(np.diff(w_cmd)) / np.maximum(np.diff(t), 1e-3)
                  if len(w_cmd) > 1 else np.array([]))
            summary_rows.append({
                "scenario": m.group("scen"),
                "controller": m.group("ctrl"),
                "trial": int(m.group("trial")),
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "max_err": float(np.max(np.abs(e))),
                "travel_time": float(t[-1] - t[0]),
                "smooth": float(np.nanmean(dw)) if len(dw) else math.nan,
                "data": data,
                "plan": plan,
            })

    if not summary_rows:
        print("exp2: no trials parsed")
        return

    # results table: mean +- std over trials
    controllers = ["ECPP", "ECPP_RAW", "DWPP", "MPPI", "DWB"]
    display = {"ECPP_RAW": "ECPP (no LPF)"}
    scen_names = sorted({r["scenario"] for r in summary_rows})
    trial_status = {}
    orch = data_dir / "summary.json"
    lines = [
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Scenario & Method & succ. & RMSE [m] & $\max|e|$ [m] & $T_m$ [s] & "
        r"$\overline{|\Delta\omega|}$ [rad/s$^2$] \\",
        r"\midrule",
    ]
    for scen in scen_names:
        first = True
        for controller in controllers:
            rows = [r for r in summary_rows
                    if r["scenario"] == scen and r["controller"] == controller]
            if not rows:
                continue
            n = len(rows)
            rmse = [r["rmse"] for r in rows]
            maxe = [r["max_err"] for r in rows]
            tm = [r["travel_time"] for r in rows]
            sm = [r["smooth"] for r in rows]
            lines.append(" & ".join([
                scen if first else "",
                display.get(controller, controller),
                f"{n}/{n}",
                f"{np.mean(rmse):.3f}$\\pm${np.std(rmse):.3f}",
                f"{np.mean(maxe):.3f}",
                f"{np.mean(tm):.1f}$\\pm${np.std(tm):.1f}",
                f"{np.nanmean(sm):.3f}",
            ]) + r"\\")
            first = False
        lines.append(r"\addlinespace[1pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (table_dir / "gazebo_exp2_results.tex").write_text("\n".join(lines), encoding="utf-8")

    # trajectory figure per scenario (trial 0 of each controller)
    styles = {
        "ECPP": {"color": plt.get_cmap("tab10")(0), "linestyle": "-"},
        "ECPP_RAW": {"color": plt.get_cmap("tab10")(9), "linestyle": "-"},
        "DWPP": {"color": plt.get_cmap("tab10")(1), "linestyle": "--"},
        "MPPI": {"color": plt.get_cmap("tab10")(2), "linestyle": "-."},
        "DWB": {"color": plt.get_cmap("tab10")(3), "linestyle": ":"},
    }
    display = {"ECPP_RAW": "ECPP (no LPF)"}
    for scen in scen_names:
        fig, (ax_traj, ax_e) = plt.subplots(1, 2, figsize=(9.2, 3.4))
        plan_drawn = False
        for controller in controllers:
            rows = [r for r in summary_rows
                    if r["scenario"] == scen and r["controller"] == controller
                    and r["trial"] == 0]
            if not rows:
                continue
            row = rows[0]
            if not plan_drawn:
                ax_traj.plot(row["plan"][:, 0], row["plan"][:, 1], "k--",
                             linewidth=1.0, label="Global plan")
                plan_drawn = True
            data = row["data"]
            tt = data["t"] - data["t"][0]
            e = signed_lateral_errors(
                np.stack([data["x"], data["y"]], axis=1), row["plan"])
            ax_traj.plot(data["x"], data["y"], linewidth=1.1,
                         label=display.get(controller, controller),
                         **styles[controller])
            ax_e.plot(tt, e, linewidth=1.0, **styles[controller])
        ax_traj.set_xlabel(r"$x$ [m]"); ax_traj.set_ylabel(r"$y$ [m]")
        ax_traj.set_aspect("equal", adjustable="box")
        ax_traj.legend(fontsize=7, ncol=2)
        ax_e.set_xlabel(r"$t$ [s]"); ax_e.set_ylabel(r"$e_{\rm path}$ [m]")
        for ax in (ax_traj, ax_e):
            ax.grid(True, linestyle=":", linewidth=0.5)
        fig.tight_layout()
        save_figure(fig, fig_dir / f"gazebo_exp2_{scen}")
    print(f"exp2: {len(summary_rows)} trials analyzed "
          f"({', '.join(scen_names)})")


def load_map(map_yaml: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Occupancy image and its (xmin, xmax, ymin, ymax) extent in map coords."""
    import yaml as _yaml
    meta = _yaml.safe_load(map_yaml.read_text())
    pgm_path = map_yaml.parent / meta["image"]
    data = pgm_path.read_bytes()
    header, idx = [], 0
    while len(header) < 4:
        end = data.index(b"\n", idx)
        line = data[idx:end]
        idx = end + 1
        if line.startswith(b"#"):
            continue
        header.extend(line.split())
    w, h = int(header[1]), int(header[2])
    img = np.frombuffer(data[idx:idx + w * h], dtype=np.uint8).reshape(h, w)
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    return img, (ox, ox + w * res, oy, oy + h * res)


def plot_exp2_map(data_dir: Path, map_yaml: Path, fig_dir: Path) -> None:
    """Map of the experiment environment with the shared fixed global plan."""
    plan_files = sorted(data_dir.glob("*/fixed_plan.csv"))
    if not plan_files or not map_yaml.exists():
        print("exp2 map: fixed plan or map yaml not found, skipping")
        return
    plan = load_polyline(plan_files[0])
    img, extent = load_map(map_yaml)
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255,
              extent=(extent[0], extent[1], extent[2], extent[3]),
              origin="upper", interpolation="nearest")
    ax.plot(plan[:, 0], plan[:, 1], color=plt.get_cmap("tab10")(3),
            linewidth=1.6, label="Global plan")
    ax.plot(plan[0, 0], plan[0, 1], "o", color=plt.get_cmap("tab10")(0),
            markersize=7, label="Start")
    ax.plot(plan[-1, 0], plan[-1, 1], "*", color=plt.get_cmap("tab10")(2),
            markersize=12, label="Goal")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, fig_dir / "gazebo_exp2_map")
    seg = np.diff(plan, axis=0)
    length = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    print(f"exp2 map: plan length = {length:.1f} m, {len(plan)} poses")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("../dwpp_test_simulation/data"))
    parser.add_argument("--tex-project", type=Path,
                        default=Path("../tex_docker_environment/projects/fumiya_ieee_access"))
    parser.add_argument("--exp1-name", default="ecpp_lpath")
    parser.add_argument("--exp1-tau0-name", default="ecpp_lpath_tau0")
    parser.add_argument("--exp2-name", default="comparison_14_5F")
    parser.add_argument("--exp2-map-yaml", type=Path,
                        default=Path("../ytlab2_whill/ytlab2_whill_modules/worlds/14_5F/map/map2d.yaml"))
    args = parser.parse_args()

    fig_dir = args.tex_project / "generated" / "figures"
    table_dir = args.tex_project / "generated" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    exp1_dir = args.data_root / args.exp1_name
    if exp1_dir.exists():
        analyze_exp1(exp1_dir, fig_dir, table_dir)
    exp2_dir = args.data_root / args.exp2_name
    if exp2_dir.exists():
        analyze_exp2(exp2_dir, fig_dir, table_dir)
        plot_exp2_map(exp2_dir, args.exp2_map_yaml, fig_dir)

    # console-only: filter on/off comparison for Exp1 (same environment)
    tau0_dir = args.data_root / args.exp1_tau0_name
    if exp1_dir.exists() and tau0_dir.exists():
        ref = load_polyline(exp1_dir / "reference_path.csv")
        print("--- exp1 filter on/off (nominal condition) ---")
        for ctrl in ("ECPP_W141_Z100", "ECPP_W212_Z100"):
            row = []
            for name, d in (("tau*", exp1_dir), ("tau0", tau0_dir)):
                f = d / f"{ctrl}_ey0_epsi0_t0.csv"
                if not f.exists():
                    continue
                met = exp1_trial_metrics(load_trial(f), ref, "succeeded")
                row.append(f"{name}: e_y={met['mean_abs_e_y']:.3f} "
                           f"Mos={met['overshoot_c']:.3f} sat={100*met['omega_sat']:.1f}%")
            print(f"  {ctrl}: " + " | ".join(row))


if __name__ == "__main__":
    main()

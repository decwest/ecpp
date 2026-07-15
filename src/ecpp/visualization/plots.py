from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..config import ExperimentConfig
from ..evaluation.metrics import calc_signed_heading_errors, calc_signed_lateral_errors
from ..geometry import calc_path_distances
from ..simulation.path_tracking import TrackingResult


METHOD_ORDER = {"PP": 0, "DPP": 1, "ECPP without gate": 2, "ECPP": 3, "ECPP ey": 4}
LINESTYLES = ("-", "--", "-.", ":")
RIGHT_ANGLE_PROFILE_XLIM = (2.0, 6.0)
RIGHT_ANGLE_PAPER_FIGSIZE = (8.0, 4.0)
RIGHT_ANGLE_TRAJECTORY_XMAX = 2.0
RIGHT_ANGLE_TRAJECTORY_YMAX = 1.0
RIGHT_ANGLE_ECPP_COLOR_INDICES = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11)

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.grid"] = True
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.handlelength"] = 1


def plot_trajectory_sweep(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    scenarios = ordered_scenarios(results)
    for scenario in scenarios:
        for lookahead in experiment.lookahead_values_m:
            figsize = RIGHT_ANGLE_PAPER_FIGSIZE if scenario.key == "right_angle" else (8, 4)
            fig, ax = plt.subplots(figsize=figsize)
            reference_path = scenario.evaluation_path
            reference_color = "0.55" if scenario.key == "right_angle" else "k"
            ax.plot(reference_path[:, 0], reference_path[:, 1], color=reference_color, linestyle="--", linewidth=1.1, label="Reference")
            selected = sorted_results(results, scenario.key, lookahead)
            pp_present = contains_pp(selected)
            for index, result in enumerate(selected):
                label = sweep_label(result)
                style = sweep_variant_style(result, index, len(selected), pp_present)
                ax.plot(
                    result.poses[:, 0],
                    result.poses[:, 1],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    label=label,
                )
            if scenario.key != "right_angle":
                ax.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax.set_xlabel(r"$x$ [m]")
            ax.set_ylabel(r"$y$ [m]")
            ax.grid(True, linestyle=":", linewidth=0.6)
            if scenario.key != "straight":
                ax.set_aspect("equal", adjustable="box")
            if scenario.key == "right_angle":
                apply_right_angle_trajectory_limits(ax, selected, reference_path)
            else:
                place_legend_outside(ax)
            fig.tight_layout()
            save_figure(
                fig,
                output_stem_for_case(output_stem, scenario.key, lookahead),
                tight=scenario.key != "right_angle",
            )


def plot_corner_zoom(
    output_stem: Path,
    experiment: ExperimentConfig,
    results: dict[str, TrackingResult],
    margin_m: float = 1.0,
) -> None:
    scenarios = ordered_scenarios(results)
    for scenario in scenarios:
        corner_index = first_corner_index(scenario.evaluation_path)
        if corner_index is None:
            continue
        reference_path = scenario.evaluation_path
        corner = reference_path[corner_index, :2]
        xlim, ylim = corner_window_limits(reference_path, corner_index, margin_m)
        for lookahead in experiment.lookahead_values_m:
            fig, ax = plt.subplots(figsize=(6.0, 4.8))
            ax.plot(reference_path[:, 0], reference_path[:, 1], "k--", linewidth=1.1, label="Reference")
            ax.plot(corner[0], corner[1], "ko", markersize=3.5, label="Corner")
            selected = sorted_results(results, scenario.key, lookahead)
            pp_present = contains_pp(selected)
            for index, result in enumerate(selected):
                style = sweep_variant_style(result, index, len(selected), pp_present)
                ax.plot(
                    result.poses[:, 0],
                    result.poses[:, 1],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    label=sweep_label(result),
                )
            ax.set_title(f"{scenario.label} corner zoom, Ld={lookahead:.2f} m")
            ax.set_xlabel(r"$x$ [m]")
            ax.set_ylabel(r"$y$ [m]")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle=":", linewidth=0.6)
            place_legend_outside(ax)
            fig.tight_layout()
            save_figure(fig, output_stem_for_case(output_stem, scenario.key, lookahead))


def plot_curvature_sweep(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    scenarios = ordered_scenarios(results)
    for scenario in scenarios:
        for lookahead in experiment.lookahead_values_m:
            figsize = RIGHT_ANGLE_PAPER_FIGSIZE if scenario.key == "right_angle" else (11.5, 4.0)
            fig, ax = plt.subplots(figsize=figsize)
            selected = sorted_results(results, scenario.key, lookahead)
            pp_present = contains_pp(selected)
            for index, result in enumerate(selected):
                label = sweep_label(result)
                style = sweep_variant_style(result, index, len(selected), pp_present)
                ax.plot(
                    result.times,
                    result.curvatures,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    label=label,
                )
            ax.axhline(experiment.control.omega_max / experiment.control.v_max, color="0.55", linewidth=0.8, linestyle=":")
            ax.axhline(-experiment.control.omega_max / experiment.control.v_max, color="0.55", linewidth=0.8, linestyle=":")
            if scenario.key != "right_angle":
                ax.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax.set_xlabel(r"$t$ [s]")
            ax.set_ylabel(r"$\kappa$ [1/m]")
            ax.grid(True, linestyle=":", linewidth=0.6)
            if scenario.key == "right_angle":
                ax.set_xlim(*RIGHT_ANGLE_PROFILE_XLIM)
            else:
                place_legend_outside(ax)
            fig.tight_layout()
            save_figure(
                fig,
                output_stem_for_case(output_stem, scenario.key, lookahead),
                tight=scenario.key != "right_angle",
            )


def plot_omega_profiles(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    scenarios = ordered_scenarios(results)
    for scenario in scenarios:
        for lookahead in experiment.lookahead_values_m:
            selected = sorted_results(results, scenario.key, lookahead)
            if not selected:
                continue
            figsize = RIGHT_ANGLE_PAPER_FIGSIZE if scenario.key == "right_angle" else (11.5, 4.0)
            fig, ax = plt.subplots(figsize=figsize)
            pp_present = contains_pp(selected)
            for index, result in enumerate(selected):
                style = sweep_variant_style(result, index, len(selected), pp_present)
                ax.plot(
                    result.times,
                    result.omega_cmd,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    label=sweep_label(result),
                )
            ax.axhline(experiment.control.omega_max, color="0.55", linewidth=0.8, linestyle=":")
            ax.axhline(-experiment.control.omega_max, color="0.55", linewidth=0.8, linestyle=":")
            if scenario.key != "right_angle":
                ax.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax.set_xlabel(r"$t$ [s]")
            ax.set_ylabel(r"$\omega$ [rad/s]")
            ax.grid(True, linestyle=":", linewidth=0.6)
            if scenario.key == "right_angle":
                ax.set_xlim(*RIGHT_ANGLE_PROFILE_XLIM)
            else:
                place_legend_outside(ax)
            fig.tight_layout()
            save_figure(
                fig,
                output_stem_for_case(output_stem, scenario.key, lookahead),
                tight=scenario.key != "right_angle",
            )


def plot_error_profiles(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    scenarios = ordered_scenarios(results)
    for scenario in scenarios:
        for lookahead in experiment.lookahead_values_m:
            selected = sorted_results(results, scenario.key, lookahead)
            if not selected:
                continue
            figsize = RIGHT_ANGLE_PAPER_FIGSIZE if scenario.key == "right_angle" else (11.5, 4.0)
            fig_y, ax_y = plt.subplots(figsize=figsize)
            fig_psi, ax_psi = plt.subplots(figsize=figsize)
            pp_present = contains_pp(selected)
            for index, result in enumerate(selected):
                style = sweep_variant_style(result, index, len(selected), pp_present)
                ey = calc_signed_lateral_errors(result.poses, result.scenario.evaluation_path)
                epsi = calc_signed_heading_errors(result.poses, result.scenario.evaluation_path)
                for ax, values in ((ax_y, ey), (ax_psi, epsi)):
                    ax.plot(
                        result.times,
                        values,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        alpha=style["alpha"],
                        label=sweep_label(result),
                    )
            if scenario.key != "right_angle":
                ax_y.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax_y.set_xlabel(r"$t$ [s]")
            ax_y.set_ylabel(r"$e_y$ [m]")
            if scenario.key != "right_angle":
                ax_psi.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax_psi.set_xlabel(r"$t$ [s]")
            ax_psi.set_ylabel(r"$e_\theta$ [rad]")
            for ax in (ax_y, ax_psi):
                ax.grid(True, linestyle=":", linewidth=0.6)
                if scenario.key == "right_angle":
                    ax.set_xlim(*RIGHT_ANGLE_PROFILE_XLIM)
                else:
                    place_legend_outside(ax)
            fig_y.tight_layout()
            fig_psi.tight_layout()
            save_figure(
                fig_y,
                output_stem_for_case(output_stem.with_name(f"{output_stem.name}_y"), scenario.key, lookahead),
                tight=scenario.key != "right_angle",
            )
            save_figure(
                fig_psi,
                output_stem_for_case(output_stem.with_name(f"{output_stem.name}_psi"), scenario.key, lookahead),
                tight=scenario.key != "right_angle",
            )


def plot_gate_profiles(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    scenarios = ordered_scenarios(results)
    for scenario in scenarios:
        for lookahead in experiment.lookahead_values_m:
            selected = [
                result for result in sorted_results(results, scenario.key, lookahead)
                if result.variant.method == "ecpp"
            ]
            if not selected:
                continue
            figsize = RIGHT_ANGLE_PAPER_FIGSIZE if scenario.key == "right_angle" else (11.5, 4.0)
            fig, ax = plt.subplots(figsize=figsize)
            pp_present = contains_pp(selected)
            for index, result in enumerate(selected):
                style = sweep_variant_style(result, index, len(selected), pp_present)
                ax.plot(
                    result.times,
                    result.sigma,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    label=sweep_label(result),
                )
            if scenario.key != "right_angle":
                ax.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax.set_xlabel(r"$t$ [s]")
            ax.set_ylabel(r"$\sigma$ [-]")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle=":", linewidth=0.6)
            if scenario.key == "right_angle":
                ax.set_xlim(*RIGHT_ANGLE_PROFILE_XLIM)
            else:
                place_legend_outside(ax)
            fig.tight_layout()
            save_figure(
                fig,
                output_stem_for_case(output_stem, scenario.key, lookahead),
                tight=scenario.key != "right_angle",
            )


def plot_right_angle_common_legends(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    for lookahead in experiment.lookahead_values_m:
        selected = sorted_results(results, "right_angle", lookahead)
        if not selected:
            continue
        handles = []
        labels = []
        pp_present = contains_pp(selected)
        for index, result in enumerate(selected):
            style = sweep_variant_style(result, index, len(selected), pp_present)
            handles.append(Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=float(style["linewidth"]),
                alpha=float(style["alpha"]),
            ))
            labels.append(right_angle_common_legend_label(result))
        fig = plt.figure(figsize=(11.5, 1.15))
        fig.legend(handles, labels, loc="center", ncol=4, frameon=True, fontsize=8)
        save_figure(fig, output_stem_for_case(output_stem, "right_angle", lookahead))


def plot_condition_profiles(output_dir: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    for scenario in ordered_scenarios(results):
        for lookahead in experiment.lookahead_values_m:
            for condition in ordered_conditions(results, scenario.key, lookahead):
                selected = sorted_condition_results(experiment, results, scenario.key, lookahead, condition)
                if not selected:
                    continue
                stem = output_dir / condition_stem(scenario.key, condition)
                plot_condition_trajectory(stem.with_name(f"{stem.name}_trajectory"), selected, lookahead)
                plot_condition_time_profile(
                    stem.with_name(f"{stem.name}_kappa"),
                    selected,
                    lookahead,
                    r"$\kappa$ [1/m]",
                    lambda result: result.curvatures,
                    limit=experiment.control.omega_max / experiment.control.v_max,
                )
                plot_condition_time_profile(
                    stem.with_name(f"{stem.name}_omega"),
                    selected,
                    lookahead,
                    r"$\omega$ [rad/s]",
                    lambda result: result.omega_cmd,
                    limit=experiment.control.omega_max,
                )
                plot_condition_time_profile(
                    stem.with_name(f"{stem.name}_error_y"),
                    selected,
                    lookahead,
                    r"$e_y$ [m]",
                    lambda result: calc_signed_lateral_errors(result.poses, result.scenario.evaluation_path),
                )
                plot_condition_time_profile(
                    stem.with_name(f"{stem.name}_error_psi"),
                    selected,
                    lookahead,
                    r"$e_\theta$ [rad]",
                    lambda result: calc_signed_heading_errors(result.poses, result.scenario.evaluation_path),
                )
                gate_results = [result for result in selected if result.variant.method == "ecpp"]
                if gate_results:
                    plot_condition_time_profile(
                        stem.with_name(f"{stem.name}_sigma"),
                        gate_results,
                        lookahead,
                        r"$\sigma$ [-]",
                        lambda result: result.sigma,
                        ylim=(-0.05, 1.05),
                    )


def plot_condition_trajectory(output_stem: Path, results: list[TrackingResult], lookahead: float) -> None:
    scenario = results[0].scenario
    reference_path = scenario.evaluation_path
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(reference_path[:, 0], reference_path[:, 1], "k--", linewidth=1.1, label="Reference")
    for index, result in enumerate(results):
        style = condition_variant_style(index, len(results))
        ax.plot(
            result.poses[:, 0],
            result.poses[:, 1],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            label=method_label(result),
        )
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.grid(True, linestyle=":", linewidth=0.6)
    if scenario.key != "straight":
        ax.set_aspect("equal", adjustable="box")
    place_legend_inside(ax)
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_condition_time_profile(
    output_stem: Path,
    results: list[TrackingResult],
    lookahead: float,
    ylabel: str,
    values_fn,
    limit: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for index, result in enumerate(results):
        style = condition_variant_style(index, len(results))
        ax.plot(
            result.times,
            values_fn(result),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            label=method_label(result),
        )
    if limit is not None:
        ax.axhline(limit, color="0.55", linewidth=0.8, linestyle=":")
        ax.axhline(-limit, color="0.55", linewidth=0.8, linestyle=":")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.6)
    place_legend_inside(ax)
    fig.tight_layout()
    save_figure(fig, output_stem)


def representative_results(experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> list[TrackingResult]:
    out = []
    for result in results.values():
        if result.scenario.key != "straight":
            continue
        if result.variant.label != "PP" and not is_representative(experiment, result):
            continue
        out.append(result)
    return sorted(out, key=lambda result: (result.variant.lookahead_m, METHOD_ORDER.get(result.variant.label, 99)))


def sorted_results(results: dict[str, TrackingResult], scenario_key: str, lookahead: float) -> list[TrackingResult]:
    selected = [
        result for result in results.values()
        if result.scenario.key == scenario_key and abs(result.variant.lookahead_m - lookahead) <= 1e-9
    ]
    return sorted(selected, key=lambda result: (
        METHOD_ORDER.get(result.variant.label, 99),
        -1.0 if result.variant.rho is None else float(result.variant.rho),
        -1.0 if result.variant.zeta is None else float(result.variant.zeta),
    ))


def ordered_conditions(results: dict[str, TrackingResult], scenario_key: str, lookahead: float) -> list[tuple[float, float]]:
    conditions = {
        (float(result.condition.e_y_m), float(result.condition.e_psi_deg))
        for result in results.values()
        if result.scenario.key == scenario_key and abs(result.variant.lookahead_m - lookahead) <= 1e-9
    }
    return sorted(conditions, key=lambda item: (item[0], item[1]))


def sorted_condition_results(
    experiment: ExperimentConfig,
    results: dict[str, TrackingResult],
    scenario_key: str,
    lookahead: float,
    condition: tuple[float, float],
) -> list[TrackingResult]:
    selected = [
        result for result in results.values()
        if result.scenario.key == scenario_key
        and abs(result.variant.lookahead_m - lookahead) <= 1e-9
        and math.isclose(result.condition.e_y_m, condition[0], abs_tol=1e-12)
        and math.isclose(result.condition.e_psi_deg, condition[1], abs_tol=1e-12)
        and (result.variant.label == "PP" or is_representative(experiment, result))
    ]
    return sorted(selected, key=lambda result: METHOD_ORDER.get(result.variant.label, 99))


def ordered_scenarios(results: dict[str, TrackingResult]) -> list:
    order = {"straight": 0, "arc": 1, "corner_90": 2}
    scenarios = {result.scenario.key: result.scenario for result in results.values()}
    return [item[1] for item in sorted(scenarios.items(), key=lambda kv: order.get(kv[0], 99))]


def is_representative(experiment: ExperimentConfig, result: TrackingResult) -> bool:
    if experiment.representative_omega_n is not None:
        return (
            result.variant.omega_n is not None
            and result.variant.zeta is not None
            and math.isclose(result.variant.omega_n, experiment.representative_omega_n)
            and math.isclose(result.variant.zeta, experiment.representative_zeta)
        )
    return (
        result.variant.rho is not None
        and result.variant.zeta is not None
        and math.isclose(result.variant.rho, experiment.representative_rho)
        and math.isclose(result.variant.zeta, experiment.representative_zeta)
    )


def sweep_label(result: TrackingResult) -> str:
    omega_n, zeta = natural_frequency_and_zeta(result)
    return (
        f"{method_label(result)} "
        fr"$\omega_n={omega_n:.2f}$ "
        fr"$\zeta={zeta:.3g}$"
    )


def right_angle_common_legend_label(result: TrackingResult) -> str:
    if result.variant.label == "PP":
        return "PP"
    omega_n, zeta = natural_frequency_and_zeta(result)
    return (
        f"{method_label(result)} "
        fr"$\omega_n={omega_n:.2f}$ "
        fr"$\zeta={zeta:.3g}$"
    )


def profile_label(result: TrackingResult) -> str:
    length_label = f"Ld={result.variant.lookahead_m:.2f}"
    omega_n, zeta = natural_frequency_and_zeta(result)
    return (
        f"{method_label(result)} {length_label} "
        fr"$\omega_n={omega_n:.2f}$ "
        fr"$\zeta={zeta:.3g}$"
    )


def natural_frequency_and_zeta(result: TrackingResult) -> tuple[float, float]:
    if result.variant.label == "PP":
        v_cmd = float(result.v_cmd[0]) if len(result.v_cmd) else 0.0
        omega_n = math.sqrt(2.0) * v_cmd / result.variant.lookahead_m
        zeta = 1.0 / math.sqrt(2.0)
        return omega_n, zeta
    if result.variant.omega_n is None or result.variant.zeta is None:
        return float("nan"), float("nan")
    return float(result.variant.omega_n), float(result.variant.zeta)


def method_label(result: TrackingResult) -> str:
    if result.variant.label == "ECPP without gate":
        return "ECPP w/o gate"
    if result.variant.label == "ECPP ey":
        return r"ECPP ($e_y$ gate)"
    return result.variant.label


def sweep_variant_style(
    result: TrackingResult,
    index: int,
    total: int,
    pp_present: bool = True,
) -> dict[str, object]:
    if result.scenario.key != "right_angle":
        return variant_style(index, total)
    if result.variant.label == "PP":
        return {
            "color": "black",
            "linestyle": "-",
            "linewidth": 1.15,
            "alpha": 1.0,
        }
    ecpp_index = max(index - 1, 0) if pp_present else index
    color_index = RIGHT_ANGLE_ECPP_COLOR_INDICES[ecpp_index % len(RIGHT_ANGLE_ECPP_COLOR_INDICES)]
    return {
        "color": plt.get_cmap("tab20")(color_index),
        "linestyle": LINESTYLES[(ecpp_index // len(RIGHT_ANGLE_ECPP_COLOR_INDICES)) % len(LINESTYLES)],
        "linewidth": 0.72,
        "alpha": 1.0,
    }


def contains_pp(results: list[TrackingResult]) -> bool:
    return any(result.variant.label == "PP" for result in results)


def variant_style(index: int, total: int) -> dict[str, object]:
    if total <= 10:
        color = plt.get_cmap("tab10")(index)
    elif total <= 20:
        color = plt.get_cmap("tab20")(index)
    else:
        color = plt.get_cmap("turbo")(index / max(total - 1, 1))
    return {
        "color": color,
        "linestyle": LINESTYLES[(index // max(1, min(total, 10))) % len(LINESTYLES)],
        "linewidth": 0.5,
        "alpha": 1.00,
    }


def condition_variant_style(index: int, total: int) -> dict[str, object]:
    style = variant_style(index, total)
    style["linewidth"] = 1.2
    return style


def apply_right_angle_trajectory_limits(
    ax: plt.Axes,
    results: list[TrackingResult],
    reference_path: np.ndarray,
) -> None:
    x_values = [reference_path[:, 0]]
    x_values.extend(result.poses[:, 0] for result in results)
    y_values = [reference_path[:, 1]]
    y_values.extend(result.poses[:, 1] for result in results)
    x_min = float(np.min(np.concatenate(x_values)))
    y_min = float(np.min(np.concatenate(y_values)))
    ax.set_xlim(x_min - 0.08, RIGHT_ANGLE_TRAJECTORY_XMAX)
    ax.set_ylim(y_min - 0.08, RIGHT_ANGLE_TRAJECTORY_YMAX)


def first_corner_index(path: np.ndarray, min_heading_change_rad: float = math.radians(20.0)) -> int | None:
    if len(path) < 3 or path.shape[1] < 3:
        return None
    headings = np.unwrap(path[:, 2])
    heading_changes = np.abs(np.diff(headings))
    corner_indices = np.flatnonzero(heading_changes >= min_heading_change_rad)
    if len(corner_indices) == 0:
        return None
    return min(int(corner_indices[0]) + 1, len(path) - 1)


def corner_window_limits(path: np.ndarray, corner_index: int, window_m: float) -> tuple[tuple[float, float], tuple[float, float]]:
    distances = calc_path_distances(path)
    corner_distance = float(distances[corner_index])
    window_mask = (distances >= corner_distance - window_m) & (distances <= corner_distance + window_m)
    window_points = path[window_mask, :2]
    if len(window_points) == 0:
        window_points = path[[corner_index], :2]
    xmin, ymin = np.min(window_points, axis=0)
    xmax, ymax = np.max(window_points, axis=0)
    lower_padding = 0.06
    upper_padding = 0.08
    outside_turn_padding = 0.30
    return (
        (float(xmin - upper_padding), float(xmax + outside_turn_padding)),
        (float(ymin - lower_padding), float(ymax + upper_padding)),
    )


def place_legend_outside(ax: plt.Axes) -> None:
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        ncol=1,
    )


def place_legend_inside(ax: plt.Axes) -> None:
    ax.legend(loc="best", frameon=True, ncol=1)


def output_stem_for_case(output_stem: Path, scenario_key: str, lookahead: float) -> Path:
    return output_stem.with_name(f"{scenario_key}_{output_stem.name}_{lookahead_slug(lookahead)}")


def condition_stem(scenario_key: str, condition: tuple[float, float]) -> str:
    e_y, e_psi = condition
    return f"{scenario_key}_{signed_slug('ey', e_y, 2)}_{signed_slug('epsi', e_psi, 0)}"


def condition_title(result: TrackingResult, lookahead: float) -> str:
    return (
        f"{result.scenario.label}, Ld={lookahead:.2f} m, "
        f"ey0={result.condition.e_y_m:.2f} m, "
        f"epsi0={result.condition.e_psi_deg:.0f} deg"
    )


def signed_slug(prefix: str, value: float, digits: int) -> str:
    sign = "m" if value < -0.5 * 10 ** (-digits) else ""
    abs_value = abs(value)
    text = f"{abs_value:.{digits}f}".replace(".", "p")
    return f"{prefix}{sign}{text}"


def lookahead_slug(lookahead: float) -> str:
    return f"Ld{lookahead:.2f}".replace(".", "p")


def save_figure(fig: plt.Figure, output_stem: Path, tight: bool = True) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    bbox_inches = "tight" if tight else None
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches=bbox_inches)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches=bbox_inches)
    plt.close(fig)

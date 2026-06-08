from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..config import ExperimentConfig
from ..evaluation.metrics import calc_signed_heading_errors, calc_signed_lateral_errors
from ..simulation.path_tracking import TrackingResult


METHOD_ORDER = {"PP": 0, "DPP": 1, "ECPP without gate": 2, "ECPP": 3}
METHOD_STYLE = {
    "PP": {"color": "#222222", "linestyle": "-"},
    "DPP": {"color": "#2b6cb0", "linestyle": "-"},
    "ECPP without gate": {"color": "#c05621", "linestyle": "--"},
    "ECPP": {"color": "#2f855a", "linestyle": "-"},
}


def plot_trajectory_sweep(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    scenarios = ordered_scenarios(results)
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(7.2, 2.9 * len(scenarios)), squeeze=False)
    for row, scenario in enumerate(scenarios):
        for col, lookahead in enumerate((experiment.lookahead_short_m, experiment.lookahead_long_m)):
            ax = axes[row, col]
            ax.plot(scenario.path[:, 0], scenario.path[:, 1], "k--", linewidth=1.1, label="Reference")
            plotted_labels = {"Reference"}
            for result in sorted_results(results, scenario.key, lookahead):
                rep = is_representative(experiment, result)
                label = sweep_label(result, rep)
                style = METHOD_STYLE.get(result.variant.label, {"color": "0.4", "linestyle": "-"})
                ax.plot(
                    result.poses[:, 0],
                    result.poses[:, 1],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.2 if rep or result.variant.label == "PP" else 1.0,
                    alpha=0.95 if rep or result.variant.label == "PP" else 0.24,
                    label=label if label not in plotted_labels else None,
                )
                plotted_labels.add(label)
            ax.set_title(f"{scenario.label}, Ld={lookahead:.2f} m", fontsize=9)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True, linestyle=":", linewidth=0.6)
            if scenario.key != "straight":
                ax.set_aspect("equal", adjustable="box")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_figure(fig, output_stem)


def plot_curvature_sweep(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    scenarios = ordered_scenarios(results)
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(7.2, 2.8 * len(scenarios)), squeeze=False, sharey=True)
    for row, scenario in enumerate(scenarios):
        for col, lookahead in enumerate((experiment.lookahead_short_m, experiment.lookahead_long_m)):
            ax = axes[row, col]
            plotted_labels: set[str] = set()
            for result in sorted_results(results, scenario.key, lookahead):
                rep = is_representative(experiment, result)
                label = sweep_label(result, rep)
                style = METHOD_STYLE.get(result.variant.label, {"color": "0.4", "linestyle": "-"})
                ax.plot(
                    result.times,
                    result.curvatures,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.2 if rep or result.variant.label == "PP" else 1.0,
                    alpha=0.95 if rep or result.variant.label == "PP" else 0.24,
                    label=label if label not in plotted_labels else None,
                )
                plotted_labels.add(label)
            ax.axhline(experiment.control.omega_max / experiment.control.v_max, color="0.55", linewidth=0.8, linestyle=":")
            ax.axhline(-experiment.control.omega_max / experiment.control.v_max, color="0.55", linewidth=0.8, linestyle=":")
            ax.set_title(f"{scenario.label}, Ld={lookahead:.2f} m", fontsize=9)
            ax.set_xlabel("t [s]")
            ax.set_ylabel("kappa [1/m]")
            ax.grid(True, linestyle=":", linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, output_stem)


def plot_omega_profiles(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    selected = representative_results(experiment, results)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for label, result in selected:
        ax.plot(result.times, result.omega_cmd, linewidth=1.4, label=label)
    ax.axhline(experiment.control.omega_max, color="0.55", linewidth=0.8, linestyle=":")
    ax.axhline(-experiment.control.omega_max, color="0.55", linewidth=0.8, linestyle=":")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("omega [rad/s]")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_error_profiles(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    selected = representative_results(experiment, results)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True)
    for label, result in selected:
        ey = calc_signed_lateral_errors(result.poses, result.scenario.path)
        epsi = calc_signed_heading_errors(result.poses, result.scenario.path)
        axes[0].plot(result.times, ey, linewidth=1.3, label=label)
        axes[1].plot(result.times, epsi, linewidth=1.3, label=label)
    axes[0].set_ylabel("e_y [m]")
    axes[1].set_ylabel("e_psi [rad]")
    axes[1].set_xlabel("t [s]")
    for ax in axes:
        ax.grid(True, linestyle=":", linewidth=0.6)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_gate_profiles(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    selected = [(label, result) for label, result in representative_results(experiment, results) if result.variant.label == "ECPP"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for label, result in selected:
        ax.plot(result.times, result.sigma, linewidth=1.4, label=label)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("sigma [-]")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output_stem)


def representative_results(experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> list[tuple[str, TrackingResult]]:
    out = []
    for result in results.values():
        if result.scenario.key != "straight":
            continue
        if result.variant.label != "PP" and not is_representative(experiment, result):
            continue
        length_label = "short" if abs(result.variant.lookahead_m - experiment.lookahead_short_m) < 1e-9 else "long"
        out.append((f"{result.variant.label} {length_label}", result))
    return sorted(out, key=lambda item: (item[1].variant.lookahead_m, METHOD_ORDER.get(item[1].variant.label, 99)))


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


def ordered_scenarios(results: dict[str, TrackingResult]) -> list:
    order = {"straight": 0, "arc": 1, "corner_90": 2}
    scenarios = {result.scenario.key: result.scenario for result in results.values()}
    return [item[1] for item in sorted(scenarios.items(), key=lambda kv: order.get(kv[0], 99))]


def is_representative(experiment: ExperimentConfig, result: TrackingResult) -> bool:
    return (
        result.variant.rho is not None
        and result.variant.zeta is not None
        and math.isclose(result.variant.rho, experiment.representative_rho)
        and math.isclose(result.variant.zeta, experiment.representative_zeta)
    )


def sweep_label(result: TrackingResult, representative: bool) -> str:
    if result.variant.label == "PP":
        return "PP"
    return f"{result.variant.label} {'rep.' if representative else 'sweep'}"


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

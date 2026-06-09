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
            fig, ax = plt.subplots(figsize=(8, 4))
            reference_path = scenario.evaluation_path
            ax.plot(reference_path[:, 0], reference_path[:, 1], "k--", linewidth=1.1, label="Reference")
            for result in sorted_results(results, scenario.key, lookahead):
                rep = is_representative(experiment, result)
                label = sweep_label(result)
                style = METHOD_STYLE.get(result.variant.label, {"color": "0.4", "linestyle": "-"})
                ax.plot(
                    result.poses[:, 0],
                    result.poses[:, 1],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.2 if rep or result.variant.label == "PP" else 1.0,
                    alpha=0.95 if rep or result.variant.label == "PP" else 0.24,
                    label=label,
                )
            ax.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax.set_xlabel(r"$x$ [m]")
            ax.set_ylabel(r"$y$ [m]")
            ax.grid(True, linestyle=":", linewidth=0.6)
            if scenario.key != "straight":
                ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="best", ncol=2, frameon=True)
            fig.tight_layout()
            save_figure(fig, output_stem_for_case(output_stem, scenario.key, lookahead))


def plot_curvature_sweep(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    scenarios = ordered_scenarios(results)
    for scenario in scenarios:
        for lookahead in experiment.lookahead_values_m:
            fig, ax = plt.subplots(figsize=(11.5, 4.0))
            for result in sorted_results(results, scenario.key, lookahead):
                rep = is_representative(experiment, result)
                label = sweep_label(result)
                style = METHOD_STYLE.get(result.variant.label, {"color": "0.4", "linestyle": "-"})
                ax.plot(
                    result.times,
                    result.curvatures,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.2 if rep or result.variant.label == "PP" else 1.0,
                    alpha=0.95 if rep or result.variant.label == "PP" else 0.24,
                    label=label,
                )
            ax.axhline(experiment.control.omega_max / experiment.control.v_max, color="0.55", linewidth=0.8, linestyle=":")
            ax.axhline(-experiment.control.omega_max / experiment.control.v_max, color="0.55", linewidth=0.8, linestyle=":")
            ax.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax.set_xlabel("t [s]")
            ax.set_ylabel("kappa [1/m]")
            ax.grid(True, linestyle=":", linewidth=0.6)
            ax.legend(loc="best", ncol=2, frameon=True)
            fig.tight_layout()
            save_figure(fig, output_stem_for_case(output_stem, scenario.key, lookahead))


def plot_omega_profiles(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    scenarios = ordered_scenarios(results)
    for scenario in scenarios:
        for lookahead in experiment.lookahead_values_m:
            selected = sorted_results(results, scenario.key, lookahead)
            if not selected:
                continue
            fig, ax = plt.subplots(figsize=(11.5, 4.0))
            for result in selected:
                rep = is_representative(experiment, result)
                style = METHOD_STYLE.get(result.variant.label, {"color": "0.4", "linestyle": "-"})
                ax.plot(
                    result.times,
                    result.omega_cmd,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.2 if rep or result.variant.label == "PP" else 1.0,
                    alpha=0.95 if rep or result.variant.label == "PP" else 0.24,
                    label=sweep_label(result),
                )
            ax.axhline(experiment.control.omega_max, color="0.55", linewidth=0.8, linestyle=":")
            ax.axhline(-experiment.control.omega_max, color="0.55", linewidth=0.8, linestyle=":")
            ax.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax.set_xlabel("t [s]")
            ax.set_ylabel("omega [rad/s]")
            ax.grid(True, linestyle=":", linewidth=0.6)
            ax.legend(loc="best", ncol=2, frameon=True)
            fig.tight_layout()
            save_figure(fig, output_stem_for_case(output_stem, scenario.key, lookahead))


def plot_error_profiles(output_stem: Path, experiment: ExperimentConfig, results: dict[str, TrackingResult]) -> None:
    scenarios = ordered_scenarios(results)
    for scenario in scenarios:
        for lookahead in experiment.lookahead_values_m:
            selected = sorted_results(results, scenario.key, lookahead)
            if not selected:
                continue
            fig_y, ax_y = plt.subplots(figsize=(11.5, 4.0))
            fig_psi, ax_psi = plt.subplots(figsize=(11.5, 4.0))
            for result in selected:
                rep = is_representative(experiment, result)
                style = METHOD_STYLE.get(result.variant.label, {"color": "0.4", "linestyle": "-"})
                ey = calc_signed_lateral_errors(result.poses, result.scenario.evaluation_path)
                epsi = calc_signed_heading_errors(result.poses, result.scenario.evaluation_path)
                for ax, values in ((ax_y, ey), (ax_psi, epsi)):
                    ax.plot(
                        result.times,
                        values,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=2.2 if rep or result.variant.label == "PP" else 1.0,
                        alpha=0.95 if rep or result.variant.label == "PP" else 0.24,
                        label=sweep_label(result),
                    )
            ax_y.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax_y.set_xlabel("t [s]")
            ax_y.set_ylabel("e_y [m]")
            ax_psi.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax_psi.set_xlabel("t [s]")
            ax_psi.set_ylabel("e_psi [rad]")
            for ax in (ax_y, ax_psi):
                ax.grid(True, linestyle=":", linewidth=0.6)
                ax.legend(loc="best", ncol=2, frameon=True)
            fig_y.tight_layout()
            fig_psi.tight_layout()
            save_figure(fig_y, output_stem_for_case(output_stem.with_name(f"{output_stem.name}_y"), scenario.key, lookahead))
            save_figure(fig_psi, output_stem_for_case(output_stem.with_name(f"{output_stem.name}_psi"), scenario.key, lookahead))


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
            fig, ax = plt.subplots(figsize=(11.5, 4.0))
            for result in selected:
                rep = is_representative(experiment, result)
                style = METHOD_STYLE.get(result.variant.label, {"color": "0.4", "linestyle": "-"})
                ax.plot(
                    result.times,
                    result.sigma,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.2 if rep else 1.0,
                    alpha=0.95 if rep else 0.24,
                    label=sweep_label(result),
                )
            ax.set_title(f"{scenario.label}, Ld={lookahead:.2f} m")
            ax.set_xlabel("t [s]")
            ax.set_ylabel("sigma [-]")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle=":", linewidth=0.6)
            ax.legend(loc="best", ncol=2, frameon=True)
            fig.tight_layout()
            save_figure(fig, output_stem_for_case(output_stem, scenario.key, lookahead))


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
        return "ECPP-no-gate"
    return result.variant.label


def output_stem_for_case(output_stem: Path, scenario_key: str, lookahead: float) -> Path:
    return output_stem.with_name(f"{scenario_key}_{output_stem.name}_{lookahead_slug(lookahead)}")


def lookahead_slug(lookahead: float) -> str:
    return f"Ld{lookahead:.2f}".replace(".", "p")


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

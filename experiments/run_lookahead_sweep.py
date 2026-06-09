from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ecpp.config import config_for_variant, load_experiment_config
from ecpp.evaluation.metrics import calc_signed_lateral_errors, summarize_result
from ecpp.geometry import pose_from_path_error
from ecpp.paths import build_path
from ecpp.simulation.path_tracking import InitialCondition, MethodVariant, PathScenario, run_path_tracking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pure Pursuit lookahead sweep")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/lookahead_sweep.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/lookahead_sweep"))
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    if output_dir.exists() and not args.keep_output:
        shutil.rmtree(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    path = build_path("straight", {"length": 8.0, "num_points": 800})
    reference_path = build_path("straight", {"length": 6.0, "num_points": 600})
    scenario = PathScenario("straight", "Straight", path, experiment.max_steps, reference_path=reference_path)
    condition = InitialCondition(
        "nominal",
        experiment.initial_e_y_m,
        experiment.initial_e_psi_deg,
        pose_from_path_error(path, experiment.initial_e_y_m, experiment.initial_e_psi_deg),
    )
    results = {}
    rows = []
    for lookahead in experiment.lookahead_sweep_values_m:
        variant = MethodVariant(f"pp_L{lookahead:.2f}".replace(".", "p"), "PP", "pp", "none", lookahead)
        config = config_for_variant(
            experiment.control,
            lookahead_m=lookahead,
            omega_n=experiment.representative_rho * (2.0 ** 0.5) * experiment.control.v_max / lookahead,
            zeta=experiment.representative_zeta,
            gate_mode="off",
        )
        result = run_path_tracking(scenario, condition, variant, config)
        results[lookahead] = result
        rows.append({"lookahead_m": lookahead, **summarize_result(result, experiment.control)})

    write_rows(output_dir / "lookahead_sweep_metrics.csv", rows)
    plot_trajectories(plots_dir / "pp_lookahead_trajectories", scenario, results)
    plot_lateral_errors(plots_dir / "pp_lookahead_lateral_errors", scenario, results)
    print(f"wrote: {output_dir}")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_trajectories(output_stem: Path, scenario: PathScenario, results: dict[float, object]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(scenario.evaluation_path[:, 0], scenario.evaluation_path[:, 1], "k--", linewidth=1.2, label="Reference")
    for lookahead, result in results.items():
        ax.plot(result.poses[:, 0], result.poses[:, 1], linewidth=1.2, label=f"Ld={lookahead:.2f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, output_stem)


def plot_lateral_errors(output_stem: Path, scenario: PathScenario, results: dict[float, object]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for lookahead, result in results.items():
        ey = calc_signed_lateral_errors(result.poses, scenario.evaluation_path)
        ax.plot(result.times, ey, linewidth=1.2, label=f"Ld={lookahead:.2f} m")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e_y [m]")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, output_stem)


def save(fig, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

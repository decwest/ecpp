from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ecpp.config import load_experiment_config
from ecpp.evaluation.tables import (
    rows_from_results,
    write_command_check_csv,
    write_latex_tables,
    write_metrics_csv,
    write_npz,
    write_timeseries,
)
from ecpp.simulation.runner import run_access_experiment
from ecpp.visualization.animation import animate_result
from ecpp.visualization.plots import (
    plot_curvature_sweep,
    plot_error_profiles,
    plot_gate_profiles,
    plot_omega_profiles,
    plot_trajectory_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-speed ECPP access-paper experiments")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/access_test1_fixed_speed.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--no-tex-export", action="store_true")
    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--animation-format", choices=["gif", "mp4"], default="gif")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else experiment.output_dir
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    series_dir = output_dir / "series"
    animations_dir = output_dir / "animations"

    if output_dir.exists() and not args.keep_output:
        shutil.rmtree(output_dir)
    for directory in (output_dir, plots_dir, tables_dir, series_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output = run_access_experiment(experiment)
    rows = rows_from_results(output.results, experiment.control)
    write_metrics_csv(output_dir / "fixed_speed_metrics.csv", rows)
    write_command_check_csv(output_dir / "fixed_speed_command_check.csv", rows)
    write_timeseries(series_dir, output.results)
    write_npz(output_dir / "fixed_speed_trajectories.npz", output.results)
    write_latex_tables(tables_dir, experiment, rows)

    plot_trajectory_sweep(plots_dir / "trajectory_sweep", experiment, output.results)
    plot_curvature_sweep(plots_dir / "curvature_sweep", experiment, output.results)
    plot_omega_profiles(plots_dir / "omega_profiles", experiment, output.results)
    plot_error_profiles(plots_dir / "error_profiles", experiment, output.results)
    plot_gate_profiles(plots_dir / "gate_profiles", experiment, output.results)

    if args.save_animation:
        animations_dir.mkdir(parents=True, exist_ok=True)
        for key, result in representative_animation_targets(output.results).items():
            animate_result(result, animations_dir / f"{key}.{args.animation_format}")

    if not args.no_tex_export and experiment.export_tex_project_dir is not None:
        export_paper_assets(output_dir, experiment.export_tex_project_dir)

    print(f"wrote: {output_dir}")


def representative_animation_targets(results):
    targets = {}
    for key, result in results.items():
        if result.scenario.key != "straight":
            continue
        if result.variant.label == "PP" or (result.variant.label == "ECPP" and result.variant.rho == 1.5 and result.variant.zeta == 1.0):
            targets[key] = result
    return targets


def export_paper_assets(output_dir: Path, tex_project_dir: Path) -> None:
    figures_dest = tex_project_dir / "generated" / "figures"
    tables_dest = tex_project_dir / "generated" / "tables"
    figures_dest.mkdir(parents=True, exist_ok=True)
    tables_dest.mkdir(parents=True, exist_ok=True)
    for src in (output_dir / "plots").glob("*.*"):
        if src.suffix.lower() in {".pdf", ".png"}:
            shutil.copy2(src, figures_dest / src.name)
    for src in (output_dir / "tables").glob("fixed_speed_*.*"):
        if src.suffix.lower() in {".tex", ".csv"}:
            shutil.copy2(src, tables_dest / src.name)


if __name__ == "__main__":
    main()

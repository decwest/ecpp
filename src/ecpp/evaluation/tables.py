from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ..config import EcppConfig, ExperimentConfig
from ..simulation.path_tracking import TrackingResult
from .metrics import calc_signed_heading_errors, calc_signed_lateral_errors, summarize_result


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no metric rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_command_check_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["max_v_cmd_error", "max_omega_relation_error", "max_clip_ratio", "num_rows"])
        writer.writeheader()
        writer.writerow({
            "max_v_cmd_error": max(float(row["max_v_cmd_error"]) for row in rows),
            "max_omega_relation_error": max(float(row["max_omega_relation_error"]) for row in rows),
            "max_clip_ratio": max(float(row["clip_ratio"]) for row in rows),
            "num_rows": len(rows),
        })


def write_timeseries(series_dir: Path, results: dict[str, TrackingResult]) -> None:
    series_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_s",
        "x_m",
        "y_m",
        "theta_rad",
        "v_cmd_mps",
        "kappa_cmd_inv_m",
        "omega_raw_radps",
        "omega_cmd_radps",
        "e_y_m",
        "e_psi_rad",
        "sigma",
    ]
    for key, result in results.items():
        ey = calc_signed_lateral_errors(result.poses, result.scenario.evaluation_path)
        epsi = calc_signed_heading_errors(result.poses, result.scenario.evaluation_path)
        with (series_dir / f"{key}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for idx in range(len(result.times)):
                writer.writerow({
                    "time_s": result.times[idx],
                    "x_m": result.poses[idx, 0],
                    "y_m": result.poses[idx, 1],
                    "theta_rad": result.poses[idx, 2],
                    "v_cmd_mps": result.v_cmd[idx],
                    "kappa_cmd_inv_m": result.curvatures[idx],
                    "omega_raw_radps": result.omega_raw[idx],
                    "omega_cmd_radps": result.omega_cmd[idx],
                    "e_y_m": ey[idx],
                    "e_psi_rad": epsi[idx],
                    "sigma": result.sigma[idx],
                })


def write_npz(path: Path, results: dict[str, TrackingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, result in results.items():
        prefix = key.replace("-", "_")
        arrays[f"{prefix}_times"] = result.times
        arrays[f"{prefix}_poses"] = result.poses
        arrays[f"{prefix}_curvatures"] = result.curvatures
        arrays[f"{prefix}_omega_raw"] = result.omega_raw
        arrays[f"{prefix}_omega_cmd"] = result.omega_cmd
        arrays[f"{prefix}_sigma"] = result.sigma
    np.savez(path, **arrays)


def write_latex_tables(tables_dir: Path, experiment: ExperimentConfig, rows: list[dict[str, object]]) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "fixed_speed_common_conditions.tex").write_text(common_conditions_table(experiment), encoding="utf-8")
    (tables_dir / "fixed_speed_test1_results.tex").write_text(test1_results_table(experiment, rows), encoding="utf-8")
    write_common_conditions_csv(tables_dir / "fixed_speed_common_conditions.csv", experiment)
    write_test1_results_csv(tables_dir / "fixed_speed_test1_results.csv", experiment, rows)


def rows_from_results(results: dict[str, TrackingResult], config: EcppConfig) -> list[dict[str, object]]:
    rows = []
    for key, result in results.items():
        row = {"result_key": key, **summarize_result(result, config)}
        rows.append(row)
    return rows


def common_conditions_table(experiment: ExperimentConfig) -> str:
    return "\n".join([
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Item & Value \\",
        r"\midrule",
        rf"Forward speed & $v_0={experiment.control.v_max:.2f}\,\mathrm{{m/s}}$ \\",
        rf"Yaw-rate limit & $\omega_{{\max}}={experiment.control.omega_max:.2f}\,\mathrm{{rad/s}}$ \\",
        rf"{lookahead_condition_label(experiment)} & {lookahead_values_tex(experiment)} \\",
        rf"Initial condition & $e_y(0)={experiment.initial_e_y_m:.2f}\,\mathrm{{m}},\ e_\psi(0)={experiment.initial_e_psi_deg:.0f}^\circ$ \\",
        rf"Representative design & {representative_design_tex(experiment)} \\",
        rf"Sampling period & $\Delta t={experiment.control.dt:.3f}\,\mathrm{{s}}$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])


def write_common_conditions_csv(path: Path, experiment: ExperimentConfig) -> None:
    rows = [
        {"item": "forward_speed_mps", "value": experiment.control.v_max},
        {"item": "yaw_rate_limit_radps", "value": experiment.control.omega_max},
        {"item": "lookahead_m", "value": ";".join(f"{value:.6g}" for value in experiment.lookahead_values_m)},
        {"item": "initial_e_y_m", "value": experiment.initial_e_y_m},
        {"item": "initial_e_psi_deg", "value": experiment.initial_e_psi_deg},
        {"item": "representative_rho", "value": "" if experiment.representative_omega_n is not None else experiment.representative_rho},
        {"item": "representative_omega_n", "value": "" if experiment.representative_omega_n is None else experiment.representative_omega_n},
        {"item": "representative_zeta", "value": experiment.representative_zeta},
        {"item": "dt_s", "value": experiment.control.dt},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "value"])
        writer.writeheader()
        writer.writerows(rows)


def lookahead_condition_label(experiment: ExperimentConfig) -> str:
    if len(experiment.lookahead_values_m) == 1:
        return "Lookahead distance"
    return "Lookahead distances"


def lookahead_values_tex(experiment: ExperimentConfig) -> str:
    values = experiment.lookahead_values_m
    if len(values) == 1:
        return rf"$L_d={values[0]:.2f}\,\mathrm{{m}}$"
    joined = ", ".join(f"{value:.2f}" for value in values)
    return rf"$L_d\in\{{{joined}\}}\,\mathrm{{m}}$"


def representative_design_tex(experiment: ExperimentConfig) -> str:
    if experiment.representative_omega_n is not None:
        return rf"$(\omega_n^\ast,\zeta^\ast)=({experiment.representative_omega_n:.2f},{experiment.representative_zeta:.1f})$"
    return rf"$(\rho^\ast,\zeta^\ast)=({experiment.representative_rho:.1f},{experiment.representative_zeta:.1f})$"


def test1_results_table(experiment: ExperimentConfig, rows: list[dict[str, object]]) -> str:
    selected = select_representative_rows(experiment, rows)
    lines = [
        r"\begin{tabular}{llllllllllllll}",
        r"\toprule",
        r"Path & Method & $L_d$ & $\rho$ & $\omega_n$ & $\zeta$ & $T_{10,y}$ & $T_{10,\psi}$ & $N_{\rm zc}$ & $\overline{|e_y|}$ & $\overline{|e_\psi|}$ & $T_s$ & $T_{\rm move}$ & $r_{\rm clip}$ \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(" & ".join([
            str(row["path_label"]),
            str(row["method_label"]),
            f"{float(row['lookahead_m']):.2f}",
            fmt_optional(row["rho"], 1),
            fmt_optional(row["omega_n"], 2),
            fmt_optional(row["zeta"], 3).rstrip("0").rstrip("."),
            fmt_float(row["T10_y_s"], 2),
            fmt_float(row["T10_psi_s"], 2),
            str(int(row["signed_lateral_zero_crossings"])),
            fmt_float(row["mean_abs_e_y_m"], 3),
            fmt_float(row["mean_abs_heading_error_deg"], 2),
            fmt_float(row["settling_time_s"], 2),
            fmt_float(row["travel_time_s"], 2),
            fmt_pct(row["clip_ratio"]),
        ]) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_test1_results_csv(path: Path, experiment: ExperimentConfig, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "path",
        "method",
        "lookahead_m",
        "rho",
        "omega_n",
        "zeta",
        "T10_y_s",
        "T10_psi_s",
        "N_zc",
        "mean_abs_e_y_m",
        "mean_abs_e_psi_deg",
        "settling_time_s",
        "travel_time_s",
        "clip_ratio",
    ]
    selected = select_representative_rows(experiment, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "path": row["path_label"],
                "method": row["method_label"],
                "lookahead_m": row["lookahead_m"],
                "rho": row["rho"],
                "omega_n": row["omega_n"],
                "zeta": row["zeta"],
                "T10_y_s": row["T10_y_s"],
                "T10_psi_s": row["T10_psi_s"],
                "N_zc": row["signed_lateral_zero_crossings"],
                "mean_abs_e_y_m": row["mean_abs_e_y_m"],
                "mean_abs_e_psi_deg": row["mean_abs_heading_error_deg"],
                "settling_time_s": row["settling_time_s"],
                "travel_time_s": row["travel_time_s"],
                "clip_ratio": row["clip_ratio"],
            })


def select_representative_rows(experiment: ExperimentConfig, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    method_order = {"PP": 0, "DPP": 1, "ECPP without gate": 2, "ECPP": 3}
    for row in rows:
        if row["method_label"] != "PP":
            if not is_representative_row(experiment, row):
                continue
        selected.append(row)
    return sorted(selected, key=lambda row: (str(row["path_name"]), float(row["lookahead_m"]), method_order.get(str(row["method_label"]), 99)))


def is_representative_row(experiment: ExperimentConfig, row: dict[str, object]) -> bool:
    if abs(float(row["zeta"]) - experiment.representative_zeta) > 1e-9:
        return False
    if experiment.representative_omega_n is not None:
        return abs(float(row["omega_n"]) - experiment.representative_omega_n) <= 1e-9
    return abs(float(row["rho"]) - experiment.representative_rho) <= 1e-9


def fmt_float(value: object, digits: int = 2, nan: str = "--") -> str:
    try:
        f_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f_value):
        return nan
    return f"{f_value:.{digits}f}"


def fmt_optional(value: object, digits: int = 2) -> str:
    if value in {"", None}:
        return "--"
    return fmt_float(value, digits)


def fmt_pct(value: object, digits: int = 1) -> str:
    try:
        f_value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(f_value):
        return "--"
    return f"{100.0 * f_value:.{digits}f}"

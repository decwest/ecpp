from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ..config import EcppConfig, ExperimentConfig
from ..simulation.path_tracking import TrackingResult
from .metrics import calc_signed_heading_errors, calc_signed_lateral_errors, summarize_result


METHOD_ORDER = {"PP": 0, "DPP": 1, "ECPP without gate": 2, "ECPP": 3, "ECPP ey": 4}


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
    for path_name, table in test1_results_path_tables(experiment, rows).items():
        suffix = safe_filename_token(path_name)
        (tables_dir / f"fixed_speed_test1_results_{suffix}.tex").write_text(table, encoding="utf-8")
    write_common_conditions_csv(tables_dir / "fixed_speed_common_conditions.csv", experiment)
    write_test1_results_csv(tables_dir / "fixed_speed_test1_results.csv", experiment, rows)
    for path_name in path_names_in_experiment_order(experiment, rows):
        suffix = safe_filename_token(path_name)
        write_test1_results_csv(
            tables_dir / f"fixed_speed_test1_results_{suffix}.csv",
            experiment,
            rows,
            path_name=path_name,
            exclude_zero_zero=True,
        )
    if any(str(row["path_name"]) == "right_angle" for row in rows):
        (tables_dir / "fixed_speed_test2_right_angle_results.tex").write_text(
            test2_right_angle_results_table(rows),
            encoding="utf-8",
        )
        write_test2_right_angle_results_csv(tables_dir / "fixed_speed_test2_right_angle_results.csv", rows)


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
        rf"Initial condition & {initial_conditions_tex(experiment)} \\",
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
        {"item": "initial_conditions", "value": ";".join(
            f"e_y={e_y:.6g},e_psi={e_psi:.6g}" for e_y, e_psi in experiment.initial_conditions
        )},
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


def initial_conditions_tex(experiment: ExperimentConfig) -> str:
    conditions = experiment.initial_conditions
    if len(conditions) == 1:
        e_y, e_psi = conditions[0]
        return rf"$e_y(0)={e_y:.2f}\,\mathrm{{m}},\ e_\theta(0)={e_psi:.0f}^\circ$"
    e_y_values = sorted({e_y for e_y, _ in conditions})
    e_psi_values = sorted({e_psi for _, e_psi in conditions}, reverse=True)
    e_y_joined = ", ".join(f"{value:.2f}" for value in e_y_values)
    e_psi_joined = ", ".join(f"{value:.0f}" for value in e_psi_values)
    return rf"$e_y(0)\in\{{{e_y_joined}\}}\,\mathrm{{m}},\ e_\theta(0)\in\{{{e_psi_joined}\}}^\circ$"


def representative_design_tex(experiment: ExperimentConfig) -> str:
    if experiment.representative_omega_n is not None:
        return rf"$(\omega_n^\ast,\zeta^\ast)=({experiment.representative_omega_n:.2f},{experiment.representative_zeta:.1f})$"
    return rf"$(\rho^\ast,\zeta^\ast)=({experiment.representative_rho:.1f},{experiment.representative_zeta:.1f})$"


def test1_results_table(experiment: ExperimentConfig, rows: list[dict[str, object]]) -> str:
    selected = select_representative_rows(experiment, rows)
    lines = [
        r"\begin{tabular}{llllllllllll}",
        r"\toprule",
        r"Path & $e_y(0)$ & $e_\theta(0)$ & Method & $L_d$ & $\bar e_y$ & $\bar e_\theta$ & $T_s$ & $T_r$ & $M_{\rm os}$ & $T_m$ & $\kappa_{\max}$ \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(" & ".join([
            str(row["path_label"]),
            fmt_float(row["initial_e_y_m"], 2),
            fmt_float(row["initial_e_psi_deg"], 0),
            display_method_label(row["method_label"]),
            f"{float(row['lookahead_m']):.2f}",
            fmt_float(row["mean_abs_e_y_m"], 3),
            fmt_float(row["mean_abs_heading_error_deg"], 2),
            fmt_float(row["settling_time_s"], 2),
            fmt_float(row["rise_time_s"], 2),
            fmt_float(row["max_overshoot_m"], 3),
            fmt_float(row["travel_time_s"], 2),
            fmt_float(row["max_abs_kappa_inv_m"], 2),
        ]) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def test1_results_path_tables(experiment: ExperimentConfig, rows: list[dict[str, object]]) -> dict[str, str]:
    return {
        path_name: test1_results_path_table(experiment, rows, path_name)
        for path_name in path_names_in_experiment_order(experiment, rows)
    }


def test1_results_path_table(experiment: ExperimentConfig, rows: list[dict[str, object]], path_name: str) -> str:
    selected = select_representative_rows(
        experiment,
        rows,
        path_name=path_name,
        exclude_zero_zero=True,
    )
    condition_order = initial_condition_order(experiment)
    method_order = METHOD_ORDER
    grouped: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in selected:
        condition = condition_key(row)
        grouped.setdefault(condition, []).append(row)

    lines = [
        r"\begin{tabular}{llllllll}",
        r"\toprule",
        r"Method & $\bar e_y$ & $\bar e_\theta$ & $T_s$ & $T_r$ & $M_{\rm os}$ & $T_m$ & $\kappa_{\max}$ \\",
        r"\midrule",
    ]
    first_condition = True
    for condition in sorted(grouped, key=lambda item: condition_order.get(item, len(condition_order))):
        if not first_condition:
            lines.append(r"\addlinespace[1pt]")
        first_condition = False
        lines.append(rf"\multicolumn{{8}}{{l}}{{$e_y(0)={condition[0]:.2f}\,\mathrm{{m}},\ e_\theta(0)={condition[1]:.0f}^\circ$}} \\")
        for row in sorted(grouped[condition], key=lambda item: method_order.get(str(item["method_label"]), 99)):
            lines.append(" & ".join([
                display_method_label(row["method_label"]),
                fmt_float(row["mean_abs_e_y_m"], 3),
                fmt_float(row["mean_abs_heading_error_deg"], 2),
                fmt_float(row["settling_time_s"], 2),
                fmt_float(row["rise_time_s"], 2),
                fmt_float(row["max_overshoot_m"], 3),
                fmt_float(row["travel_time_s"], 2),
                fmt_float(row["max_abs_kappa_inv_m"], 2),
            ]) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_test1_results_csv(
    path: Path,
    experiment: ExperimentConfig,
    rows: list[dict[str, object]],
    path_name: str | None = None,
    exclude_zero_zero: bool = False,
) -> None:
    fieldnames = [
        "path",
        "initial_e_y_m",
        "initial_e_psi_deg",
        "method",
        "lookahead_m",
        "mean_abs_e_y_m",
        "mean_abs_e_psi_deg",
        "settling_time_s",
        "rise_time_s",
        "max_overshoot_m",
        "travel_time_s",
        "max_abs_kappa_inv_m",
    ]
    selected = select_representative_rows(
        experiment,
        rows,
        path_name=path_name,
        exclude_zero_zero=exclude_zero_zero,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "path": row["path_label"],
                "initial_e_y_m": row["initial_e_y_m"],
                "initial_e_psi_deg": row["initial_e_psi_deg"],
                "method": display_method_label(row["method_label"]),
                "lookahead_m": row["lookahead_m"],
                "mean_abs_e_y_m": row["mean_abs_e_y_m"],
                "mean_abs_e_psi_deg": row["mean_abs_heading_error_deg"],
                "settling_time_s": row["settling_time_s"],
                "rise_time_s": row["rise_time_s"],
                "max_overshoot_m": row["max_overshoot_m"],
                "travel_time_s": row["travel_time_s"],
                "max_abs_kappa_inv_m": row["max_abs_kappa_inv_m"],
            })


def test2_right_angle_results_table(rows: list[dict[str, object]]) -> str:
    selected = select_test2_right_angle_rows(rows)
    lines = [
        r"\begin{tabular}{@{}lllllllllll@{}}",
        r"\toprule",
        r"Method & $L_d$ & $\omega_n$ & $\zeta$ & $\bar e_y$ & $\bar e_\theta$ & $T_s$ & $T_r$ & $M_{\rm os}$ & $T_m$ & $\kappa_{\max}$ \\",
        r"\midrule",
    ]
    current_lookahead: float | None = None
    for row in selected:
        lookahead = float(row["lookahead_m"])
        if current_lookahead is not None and abs(current_lookahead - lookahead) > 1e-9:
            lines.append(r"\addlinespace[1pt]")
        current_lookahead = lookahead
        lines.append(" & ".join([
            display_method_label(row["method_label"]),
            f"{lookahead:.2f}",
            fmt_corner_omega_n(row),
            fmt_optional(row["zeta"], 3),
            fmt_float(row["mean_abs_e_y_m"], 3),
            fmt_float(row["mean_abs_heading_error_deg"], 2),
            fmt_float(row["corner_settling_time_s"], 2),
            fmt_float(row["corner_rise_time_s"], 2),
            fmt_float(row["corner_max_overshoot_m"], 3),
            fmt_float(row["travel_time_s"], 2),
            fmt_float(row["max_abs_kappa_inv_m"], 2),
        ]) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_test2_right_angle_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "lookahead_m",
        "rho",
        "omega_n",
        "zeta",
        "mean_abs_e_y_m",
        "mean_abs_e_psi_deg",
        "corner_detection_time_s",
        "corner_settling_time_s",
        "corner_rise_time_s",
        "corner_max_overshoot_m",
        "travel_time_s",
        "max_abs_kappa_inv_m",
        "clip_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in select_test2_right_angle_rows(rows):
            writer.writerow({
                "method": display_method_label(row["method_label"]),
                "lookahead_m": row["lookahead_m"],
                "rho": row["rho"],
                "omega_n": corner_omega_n(row),
                "zeta": row["zeta"],
                "mean_abs_e_y_m": row["mean_abs_e_y_m"],
                "mean_abs_e_psi_deg": row["mean_abs_heading_error_deg"],
                "corner_detection_time_s": row["corner_detection_time_s"],
                "corner_settling_time_s": row["corner_settling_time_s"],
                "corner_rise_time_s": row["corner_rise_time_s"],
                "corner_max_overshoot_m": row["corner_max_overshoot_m"],
                "travel_time_s": row["travel_time_s"],
                "max_abs_kappa_inv_m": row["max_abs_kappa_inv_m"],
                "clip_ratio": row["clip_ratio"],
            })


def select_test2_right_angle_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [row for row in rows if str(row["path_name"]) == "right_angle"]
    return sorted(selected, key=lambda row: (
        float(row["lookahead_m"]),
        METHOD_ORDER.get(str(row["method_label"]), 99),
        -1.0 if corner_omega_n(row) == "" else float(corner_omega_n(row)),
        -1.0 if row["zeta"] == "" else float(row["zeta"]),
    ))


def select_representative_rows(
    experiment: ExperimentConfig,
    rows: list[dict[str, object]],
    path_name: str | None = None,
    exclude_zero_zero: bool = False,
) -> list[dict[str, object]]:
    selected = []
    method_order = METHOD_ORDER
    condition_order = initial_condition_order(experiment)
    for row in rows:
        if path_name is not None and str(row["path_name"]) != path_name:
            continue
        if exclude_zero_zero and is_zero_zero_condition(row):
            continue
        if row["method_label"] != "PP":
            if not is_representative_row(experiment, row):
                continue
        selected.append(row)
    return sorted(selected, key=lambda row: (
        path_order_index(experiment, str(row["path_name"])),
        float(row["lookahead_m"]),
        condition_order.get(condition_key(row), len(condition_order)),
        method_order.get(str(row["method_label"]), 99),
    ))


def path_names_in_experiment_order(experiment: ExperimentConfig, rows: list[dict[str, object]]) -> list[str]:
    available = {str(row["path_name"]) for row in rows}
    ordered = [spec.key for spec in experiment.paths if spec.key in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def path_order_index(experiment: ExperimentConfig, path_name: str) -> int:
    for idx, spec in enumerate(experiment.paths):
        if spec.key == path_name:
            return idx
    return len(experiment.paths)


def initial_condition_order(experiment: ExperimentConfig) -> dict[tuple[float, float], int]:
    return {
        (round(float(e_y), 12), round(float(e_psi), 12)): idx
        for idx, (e_y, e_psi) in enumerate(experiment.initial_conditions)
    }


def condition_key(row: dict[str, object]) -> tuple[float, float]:
    return (round(float(row["initial_e_y_m"]), 12), round(float(row["initial_e_psi_deg"]), 12))


def is_zero_zero_condition(row: dict[str, object]) -> bool:
    return abs(float(row["initial_e_y_m"])) <= 1e-12 and abs(float(row["initial_e_psi_deg"])) <= 1e-12


def safe_filename_token(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    token = "".join(chars).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "path"


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


def display_method_label(label: object) -> str:
    if str(label) == "ECPP without gate":
        return "ECPP w/o gate"
    if str(label) == "ECPP ey":
        return "ECPP-ey"
    return str(label)


def corner_omega_n(row: dict[str, object]) -> float | str:
    if row["omega_n"] not in {"", None}:
        return float(row["omega_n"])
    if str(row["method_label"]) == "PP":
        return math.sqrt(2.0) * float(row["v_cmd_mps"]) / float(row["lookahead_m"])
    return ""


def fmt_corner_omega_n(row: dict[str, object]) -> str:
    return fmt_optional(corner_omega_n(row), 2)


def fmt_optional(value: object, digits: int = 2) -> str:
    if value in {"", None}:
        return "--"
    return fmt_float(value, digits)


def fmt_pct(value: object, digits: int = 1, scale: float = 100.0) -> str:
    try:
        f_value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(f_value):
        return "--"
    return f"{scale * f_value:.{digits}f}"

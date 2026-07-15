from __future__ import annotations

import math

import numpy as np

from ..config import EcppConfig
from ..geometry import calc_path_distances, calc_path_headings, normalize_angle
from ..simulation.path_tracking import TrackingResult


def nearest_path_indices(robot_poses: np.ndarray, path: np.ndarray) -> np.ndarray:
    robot_xy = robot_poses[:, :2]
    path_xy = path[:, :2]
    distances = np.linalg.norm(robot_xy[:, None, :] - path_xy[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def calc_signed_lateral_errors(robot_poses: np.ndarray, path: np.ndarray) -> np.ndarray:
    indices = nearest_path_indices(robot_poses, path)
    ref_headings = calc_path_headings(path)[indices]
    deltas = robot_poses[:, :2] - path[indices, :2]
    return -np.sin(ref_headings) * deltas[:, 0] + np.cos(ref_headings) * deltas[:, 1]


def calc_signed_heading_errors(robot_poses: np.ndarray, path: np.ndarray) -> np.ndarray:
    indices = nearest_path_indices(robot_poses, path)
    robot_headings = robot_poses[:, 2] if robot_poses.shape[1] >= 3 else np.zeros(len(robot_poses))
    ref_headings = calc_path_headings(path)[indices]
    return normalize_angle(robot_headings - ref_headings)


def count_zero_crossings(values: np.ndarray, deadband: float = 1e-3) -> int:
    if deadband < 0.0:
        raise ValueError("deadband must be non-negative")
    values = np.asarray(values, dtype=float)
    signs = np.sign(np.where(np.abs(values) <= deadband, 0.0, values))
    nonzero = signs[signs != 0.0]
    if len(nonzero) <= 1:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def summarize_result(result: TrackingResult, config: EcppConfig) -> dict[str, object]:
    ey = calc_signed_lateral_errors(result.poses, result.scenario.evaluation_path)
    epsi = calc_signed_heading_errors(result.poses, result.scenario.evaluation_path)
    omega_expected = np.clip(result.curvatures * config.v_max, -config.omega_max, config.omega_max)
    corner_metrics = summarize_corner_response(result)
    return {
        "path_name": result.scenario.key,
        "path_label": result.scenario.label,
        "initial_e_y_m": result.condition.e_y_m,
        "initial_e_psi_deg": result.condition.e_psi_deg,
        "method_label": result.variant.label,
        "method_key": result.variant.key,
        "method_internal": result.variant.method,
        "gate_mode": result.variant.gate_mode,
        "lookahead_m": result.variant.lookahead_m,
        "rho": "" if result.variant.rho is None else result.variant.rho,
        "zeta": "" if result.variant.zeta is None else result.variant.zeta,
        "omega_n": "" if result.variant.omega_n is None else result.variant.omega_n,
        "v_cmd_mps": config.v_max,
        "omega_max_radps": config.omega_max,
        "dt_s": config.dt,
        "goal_reached": result.goal_reached,
        "max_steps_reached": result.max_steps_reached,
        "travel_time_s": float(result.times[-1]) if result.goal_reached else float("nan"),
        "elapsed_time_s": float(result.times[-1]),
        "T10_y_s": first_threshold_time(np.abs(ey), result.times, 0.1),
        "T10_psi_s": first_threshold_time(np.abs(epsi), result.times, 0.1),
        "settling_time_s": settling_time_percent_y(ey, result.times, 0.02),
        "settling_time_abs_s": settling_time_absolute(ey, epsi, result.times, config),
        "rise_time_s": rise_time_percent_y(ey, result.times, 0.10, 0.90),
        "max_overshoot_m": max_overshoot_y_m(ey),
        "max_overshoot_pct": max_overshoot_percent_y(ey),
        "signed_lateral_zero_crossings": count_zero_crossings(ey, config.zero_crossing_deadband),
        "mean_abs_e_y_m": float(np.mean(np.abs(ey))),
        "mean_abs_heading_error_rad": float(np.mean(np.abs(epsi))),
        "mean_abs_heading_error_deg": float(math.degrees(np.mean(np.abs(epsi)))),
        "max_abs_e_y_m": float(np.max(np.abs(ey))),
        "max_abs_heading_error_rad": float(np.max(np.abs(epsi))),
        "max_abs_heading_error_deg": float(math.degrees(np.max(np.abs(epsi)))),
        "max_abs_kappa_inv_m": float(np.max(np.abs(result.curvatures))) if len(result.curvatures) else float("nan"),
        "clip_ratio": float(np.mean(np.abs(result.omega_raw) > config.omega_max + 1e-12)),
        "clip_time_s": float(np.sum(np.abs(result.omega_raw) > config.omega_max + 1e-12) * config.dt),
        "initial_sigma": finite_at(result.sigma, 1),
        "min_sigma": finite_min(result.sigma),
        "min_sigma_y": finite_min(result.sigma_y),
        "min_sigma_psi": finite_min(result.sigma_psi),
        "max_v_cmd_error": float(np.max(np.abs(result.v_cmd - config.v_max))) if len(result.v_cmd) else float("nan"),
        "max_omega_relation_error": float(np.max(np.abs(result.omega_cmd - omega_expected))) if len(result.omega_cmd) else float("nan"),
        **corner_metrics,
    }


def first_threshold_time(values: np.ndarray, times: np.ndarray, ratio: float) -> float:
    if len(values) == 0 or values[0] <= 1e-12:
        return 0.0
    indices = np.where(values <= ratio * values[0])[0]
    return float(times[indices[0]]) if len(indices) else float("nan")


def settling_time_percent_y(ey: np.ndarray, times: np.ndarray, ratio: float = 0.02) -> float:
    ey = np.asarray(ey, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(ey) == 0 or len(times) == 0:
        return float("nan")
    initial_abs = abs(float(ey[0]))
    if initial_abs <= 1e-12:
        return float("nan")
    threshold = ratio * initial_abs
    ok = np.abs(ey) <= threshold
    stays_in_band = np.flip(np.cumprod(np.flip(ok).astype(int))).astype(bool)
    indices = np.where(stays_in_band)[0]
    return float(times[indices[0]]) if len(indices) else float("nan")


def rise_time_percent_y(
    ey: np.ndarray,
    times: np.ndarray,
    lower_ratio: float = 0.10,
    upper_ratio: float = 0.90,
) -> float:
    ey = np.asarray(ey, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(ey) == 0 or len(times) == 0:
        return float("nan")
    initial_abs = abs(float(ey[0]))
    if initial_abs <= 1e-12:
        return float("nan")
    progress = 1.0 - np.abs(ey) / initial_abs
    tol = 1e-12
    lower_indices = np.where(progress >= lower_ratio - tol)[0]
    upper_indices = np.where(progress >= upper_ratio - tol)[0]
    if len(lower_indices) == 0 or len(upper_indices) == 0:
        return float("nan")
    lower_time = float(times[lower_indices[0]])
    upper_time = float(times[upper_indices[0]])
    if upper_time < lower_time:
        return float("nan")
    return upper_time - lower_time


def max_overshoot_y_m(ey: np.ndarray) -> float:
    ey = np.asarray(ey, dtype=float)
    if len(ey) == 0:
        return float("nan")
    initial = float(ey[0])
    initial_abs = abs(initial)
    if initial_abs <= 1e-12:
        return float("nan")
    initial_sign = math.copysign(1.0, initial)
    return max(0.0, float(np.max(-initial_sign * ey)))


def max_overshoot_percent_y(ey: np.ndarray) -> float:
    ey = np.asarray(ey, dtype=float)
    if len(ey) == 0:
        return float("nan")
    initial_abs = abs(float(ey[0]))
    if initial_abs <= 1e-12:
        return float("nan")
    overshoot = max_overshoot_y_m(ey)
    return 100.0 * overshoot / initial_abs


def settling_time_absolute(ey: np.ndarray, epsi: np.ndarray, times: np.ndarray, config: EcppConfig) -> float:
    ok = (np.abs(ey) <= config.settling_e_y) & (np.abs(epsi) <= config.settling_e_psi)
    hold_steps = max(1, int(math.ceil(config.settling_hold_time / config.dt)))
    if len(ok) < hold_steps:
        return float("nan")
    for idx in range(0, len(ok) - hold_steps + 1):
        if bool(np.all(ok[idx : idx + hold_steps])):
            return float(times[idx])
    return float("nan")


def summarize_corner_response(result: TrackingResult) -> dict[str, float]:
    corner_idx = first_corner_index(result.scenario.path)
    eval_corner_idx = first_corner_index(result.scenario.evaluation_path)
    if corner_idx is None or eval_corner_idx is None:
        return empty_corner_response()

    detection = lookahead_corner_detection(result, corner_idx)
    if detection is None:
        return empty_corner_response()

    detect_idx, detect_time = detection
    eval_path = result.scenario.evaluation_path
    corner_point = eval_path[eval_corner_idx, :2]
    target_heading = float(eval_path[eval_corner_idx, 2])
    times = result.times[detect_idx:] - float(result.times[detect_idx])
    corner_error = signed_lateral_error_to_line(result.poses[detect_idx:, :2], corner_point, target_heading)
    return {
        "corner_detection_time_s": float(detect_time),
        "corner_initial_e_y_m": float(corner_error[0]) if len(corner_error) else float("nan"),
        "corner_settling_time_s": settling_time_percent_y(corner_error, times, 0.02),
        "corner_rise_time_s": rise_time_percent_y(corner_error, times, 0.10, 0.90),
        "corner_max_overshoot_m": max_overshoot_y_m(corner_error),
    }


def empty_corner_response() -> dict[str, float]:
    return {
        "corner_detection_time_s": float("nan"),
        "corner_initial_e_y_m": float("nan"),
        "corner_settling_time_s": float("nan"),
        "corner_rise_time_s": float("nan"),
        "corner_max_overshoot_m": float("nan"),
    }


def first_corner_index(path: np.ndarray, min_heading_change_rad: float = math.radians(20.0)) -> int | None:
    if len(path) < 3 or path.shape[1] < 3:
        return None
    headings = np.unwrap(path[:, 2])
    heading_changes = np.abs(np.diff(headings))
    corner_indices = np.flatnonzero(heading_changes >= min_heading_change_rad)
    if len(corner_indices) == 0:
        return None
    return min(int(corner_indices[0]) + 1, len(path) - 1)


def lookahead_corner_detection(result: TrackingResult, corner_idx: int) -> tuple[int, float] | None:
    if len(result.lookahead) == 0:
        return None
    path_distances = calc_path_distances(result.scenario.path)
    corner_distance = float(path_distances[corner_idx])
    lookahead_indices = nearest_indices_for_points(result.lookahead, result.scenario.path)
    lookahead_distances = path_distances[lookahead_indices]
    crossed = np.where(lookahead_distances > corner_distance + 1e-9)[0]
    if len(crossed) == 0:
        return None
    idx = int(crossed[0])
    return min(idx, len(result.times) - 1), float(result.times[min(idx, len(result.times) - 1)])


def nearest_indices_for_points(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    point_xy = points[:, :2]
    path_xy = path[:, :2]
    distances = np.linalg.norm(point_xy[:, None, :] - path_xy[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def signed_lateral_error_to_line(points_xy: np.ndarray, line_point: np.ndarray, line_heading: float) -> np.ndarray:
    deltas = np.asarray(points_xy, dtype=float)[:, :2] - np.asarray(line_point, dtype=float)[:2]
    return -math.sin(line_heading) * deltas[:, 0] + math.cos(line_heading) * deltas[:, 1]


def finite_at(values: np.ndarray, idx: int) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(finite[min(idx, len(finite) - 1)])


def finite_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.min(finite))

from __future__ import annotations

import math

import numpy as np

from ..config import EcppConfig
from ..geometry import calc_path_headings, normalize_angle
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
    ey = calc_signed_lateral_errors(result.poses, result.scenario.path)
    epsi = calc_signed_heading_errors(result.poses, result.scenario.path)
    norm = error_norm(ey, epsi, config.error_norm_heading_scale)
    omega_expected = np.clip(result.curvatures * config.v_max, -config.omega_max, config.omega_max)
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
        "travel_time_s": result.times[-1],
        "T10_s": first_threshold_time(norm, result.times, 0.1),
        "settling_time_s": settling_time(ey, epsi, result.times, config),
        "signed_lateral_zero_crossings": count_zero_crossings(ey, config.zero_crossing_deadband),
        "mean_abs_e_y_m": float(np.mean(np.abs(ey))),
        "mean_abs_heading_error_rad": float(np.mean(np.abs(epsi))),
        "mean_abs_heading_error_deg": float(math.degrees(np.mean(np.abs(epsi)))),
        "max_abs_e_y_m": float(np.max(np.abs(ey))),
        "max_abs_heading_error_rad": float(np.max(np.abs(epsi))),
        "max_abs_heading_error_deg": float(math.degrees(np.max(np.abs(epsi)))),
        "clip_ratio": float(np.mean(np.abs(result.omega_raw) > config.omega_max + 1e-12)),
        "clip_time_s": float(np.sum(np.abs(result.omega_raw) > config.omega_max + 1e-12) * config.dt),
        "initial_sigma": finite_at(result.sigma, 1),
        "min_sigma": finite_min(result.sigma),
        "min_sigma_y": finite_min(result.sigma_y),
        "min_sigma_psi": finite_min(result.sigma_psi),
        "max_v_cmd_error": float(np.max(np.abs(result.v_cmd - config.v_max))) if len(result.v_cmd) else float("nan"),
        "max_omega_relation_error": float(np.max(np.abs(result.omega_cmd - omega_expected))) if len(result.omega_cmd) else float("nan"),
    }


def error_norm(ey: np.ndarray, epsi: np.ndarray, heading_scale: float) -> np.ndarray:
    return np.sqrt(ey * ey + (heading_scale * epsi) ** 2)


def first_threshold_time(values: np.ndarray, times: np.ndarray, ratio: float) -> float:
    if len(values) == 0 or values[0] <= 1e-12:
        return 0.0
    indices = np.where(values <= ratio * values[0])[0]
    return float(times[indices[0]]) if len(indices) else float("nan")


def settling_time(ey: np.ndarray, epsi: np.ndarray, times: np.ndarray, config: EcppConfig) -> float:
    ok = (np.abs(ey) <= config.settling_e_y) & (np.abs(epsi) <= config.settling_e_psi)
    hold_steps = max(1, int(math.ceil(config.settling_hold_time / config.dt)))
    if len(ok) < hold_steps:
        return float("nan")
    for idx in range(0, len(ok) - hold_steps + 1):
        if bool(np.all(ok[idx : idx + hold_steps])):
            return float(times[idx])
    return float("nan")


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

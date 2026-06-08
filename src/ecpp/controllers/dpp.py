from __future__ import annotations

import math

import numpy as np

from ..config import EcppConfig
from ..geometry import calc_path_theta
from ..lookahead import calc_lookahead_index


def calc_dpp_parameters(config: EcppConfig) -> tuple[float, float, float, float, float]:
    eps = 1e-12
    v_gain = max(float(config.dpp_gain_speed), float(config.dpp_v_min), eps)
    omega_n = float(config.dpp_omega_n)
    zeta = float(config.dpp_zeta)
    t_p1 = float(config.dpp_preview_time)
    gain_denominator = t_p1 * t_p1 * omega_n * omega_n - 4.0 * zeta * omega_n * t_p1 + 2.0
    t_p2_denominator = -omega_n * omega_n * t_p1 + 2.0 * zeta * omega_n
    if abs(gain_denominator) <= eps or abs(t_p2_denominator) <= eps:
        raise ValueError("DPP parameters are singular")
    g1 = 2.0 * omega_n * omega_n * (1.0 - 2.0 * zeta * zeta) / (v_gain * gain_denominator)
    g2 = omega_n * omega_n * (omega_n * t_p1 - 2.0 * zeta) ** 2 / (v_gain * gain_denominator)
    t_p2 = 2.0 * (1.0 - zeta * omega_n * t_p1) / t_p2_denominator
    if not (math.isfinite(g1) and math.isfinite(g2) and math.isfinite(t_p2)) or t_p2 <= 0.0:
        raise ValueError("DPP computed a non-forward second preview point")
    if abs(t_p2 - t_p1) <= eps:
        raise ValueError("DPP preview times must be distinct")
    return t_p1, t_p2, g1, g2, v_gain


def calc_dpp_preview_lateral_error(
    current_pose: np.ndarray,
    current_idx: np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    preview_distance: float,
) -> tuple[float, np.ndarray, np.intp]:
    preview_idx = calc_lookahead_index(current_idx, path_distances, preview_distance)
    path_pos = path[int(preview_idx), :2].astype(float, copy=False)
    path_yaw = calc_path_theta(path, preview_idx)
    psi = float(current_pose[2])
    preview_pos = np.array([
        float(current_pose[0]) + preview_distance * math.cos(psi),
        float(current_pose[1]) + preview_distance * math.sin(psi),
    ])
    dx = float(preview_pos[0] - path_pos[0])
    dy = float(preview_pos[1] - path_pos[1])
    c_r = math.cos(path_yaw)
    s_r = math.sin(path_yaw)
    y_p = -s_r * dx + c_r * dy
    return y_p, path_pos.copy(), preview_idx


def calc_dpp_curvature(
    current_pose: np.ndarray,
    current_idx: np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    config: EcppConfig,
) -> tuple[float, np.ndarray]:
    t_p1, t_p2, g1, g2, v_gain = calc_dpp_parameters(config)
    preview_distance_1 = t_p1 * v_gain
    preview_distance_2 = t_p2 * v_gain
    y_p1, path_pos_1, _ = calc_dpp_preview_lateral_error(
        current_pose, current_idx, path, path_distances, preview_distance_1
    )
    y_p2, path_pos_2, _ = calc_dpp_preview_lateral_error(
        current_pose, current_idx, path, path_distances, preview_distance_2
    )
    gamma_des = -g1 * y_p1 - g2 * y_p2
    lookahead_pos = path_pos_1 if preview_distance_1 >= preview_distance_2 else path_pos_2
    return float(gamma_des / v_gain), lookahead_pos

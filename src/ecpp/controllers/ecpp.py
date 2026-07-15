from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import EcppConfig
from ..geometry import calc_path_theta, normalize_angle
from ..lookahead import calc_lookahead_position
from .gates import gate_abs_by_mode
from .pure_pursuit import calc_pp_curvature_to_point


@dataclass(frozen=True)
class EcppTerms:
    curvature: float
    lookahead_pos: np.ndarray
    kappa_pp: float
    compensation: float
    compensation_raw: float
    lateral_raw: float
    heading_raw: float
    sigma: float
    sigma_y: float
    sigma_psi: float
    e_y: float
    e_psi: float
    sin_e_psi: float
    k_y: float
    k_psi: float
    k_y_pp: float
    k_psi_pp: float
    dK_y: float
    dK_psi: float
    lookahead_distance: float
    v_gain: float


def calc_ecpp_terms(
    current_pose: np.ndarray,
    current_velocity: np.ndarray,
    current_idx: np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    lookahead_distance: float,
    config: EcppConfig,
) -> EcppTerms:
    eps = 1e-12
    path_pos = path[int(current_idx), :2]
    lookahead_pos, _ = calc_lookahead_position(current_idx, path, path_distances, lookahead_distance)
    path_yaw = calc_path_theta(path, current_idx)

    c = math.cos(path_yaw)
    s = math.sin(path_yaw)
    dx = float(current_pose[0] - path_pos[0])
    dy = float(current_pose[1] - path_pos[1])
    e_y = -s * dx + c * dy
    e_psi = float(normalize_angle(current_pose[2] - path_yaw))

    kappa_pp = calc_pp_curvature_to_point(current_pose, lookahead_pos)
    L_d = max(float(lookahead_distance), eps)
    omega_n = float(config.ecpp_omega_n)
    zeta = float(config.ecpp_zeta)
    current_speed = abs(float(current_velocity[0]))
    v_gain = current_speed + max(float(config.ecpp_v_epsilon), eps)

    k_y = (omega_n / v_gain) ** 2
    k_psi = 2.0 * zeta * omega_n / v_gain
    k_y_pp = 2.0 / (L_d * L_d)
    k_psi_pp = 2.0 / L_d
    dK_y = k_y - k_y_pp
    dK_psi = k_psi - k_psi_pp

    gate_mode = config.ecpp_gate_mode
    sin_e_psi = math.sin(e_psi)
    epsilon_y = (e_y / L_d) ** 2
    if abs(e_psi) <= eps:
        epsilon_psi = 0.0
    else:
        epsilon_psi = abs(e_psi - sin_e_psi) / max(abs(sin_e_psi), eps)

    sub_gate_mode = "sigmoid" if gate_mode == "ey_only" else gate_mode
    sigma_y = gate_abs_by_mode(
        epsilon_y,
        config.ecpp_gate_error_on,
        config.ecpp_gate_error_off,
        config.ecpp_gate_sigmoid_endpoint_value,
        sub_gate_mode,
    )
    sigma_psi = gate_abs_by_mode(
        epsilon_psi,
        config.ecpp_gate_error_on,
        config.ecpp_gate_error_off,
        config.ecpp_gate_sigmoid_endpoint_value,
        sub_gate_mode,
    )
    if gate_mode == "always_on":
        sigma = 1.0
    elif gate_mode == "off":
        sigma = 0.0
    elif gate_mode == "ey_only":
        sigma = sigma_y
    else:
        sigma = sigma_y * sigma_psi

    lateral_raw = dK_y * e_y
    heading_raw = dK_psi * sin_e_psi
    compensation_raw = lateral_raw + heading_raw
    compensation = -sigma * compensation_raw
    curvature = kappa_pp + compensation
    return EcppTerms(
        curvature=float(curvature),
        lookahead_pos=lookahead_pos,
        kappa_pp=float(kappa_pp),
        compensation=float(compensation),
        compensation_raw=float(compensation_raw),
        lateral_raw=float(lateral_raw),
        heading_raw=float(heading_raw),
        sigma=float(sigma),
        sigma_y=float(sigma_y),
        sigma_psi=float(sigma_psi),
        e_y=float(e_y),
        e_psi=float(e_psi),
        sin_e_psi=float(sin_e_psi),
        k_y=float(k_y),
        k_psi=float(k_psi),
        k_y_pp=float(k_y_pp),
        k_psi_pp=float(k_psi_pp),
        dK_y=float(dK_y),
        dK_psi=float(dK_psi),
        lookahead_distance=float(L_d),
        v_gain=float(v_gain),
    )

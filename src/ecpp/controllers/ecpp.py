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
    if config.ecpp_gain_speed_regularization == "epsilon":
        v_gain = current_speed + max(float(config.ecpp_v_epsilon), eps)
    else:
        v_gain = max(current_speed, float(config.ecpp_v_min), eps)

    k_y = (omega_n / v_gain) ** 2
    k_psi = 2.0 * zeta * omega_n / v_gain
    k_y_pp = 2.0 / (L_d * L_d)
    k_psi_pp = 2.0 / L_d
    dK_y = k_y - k_y_pp
    dK_psi = k_psi - k_psi_pp

    gate_mode = config.ecpp_gate_mode
    sigma_y = gate_abs_by_mode(
        abs(e_y),
        config.ecpp_lateral_gate_on_ratio * L_d,
        config.ecpp_lateral_gate_off_ratio * L_d,
        config.ecpp_gate_sigmoid_endpoint_value,
        gate_mode,
    )
    sigma_psi = gate_abs_by_mode(
        abs(float(normalize_angle(e_psi))),
        config.ecpp_heading_gate_on,
        config.ecpp_heading_gate_off,
        config.ecpp_gate_sigmoid_endpoint_value,
        gate_mode,
    )
    if gate_mode == "always_on":
        sigma = 1.0
    elif gate_mode == "off":
        sigma = 0.0
    else:
        sigma = sigma_y * sigma_psi

    sin_e_psi = math.sin(e_psi)
    lateral_raw = dK_y * _saturate_lateral_error(e_y, config.ecpp_lateral_saturation_ratio * L_d)
    heading_raw = dK_psi * sin_e_psi
    compensation_raw = lateral_raw + heading_raw
    compensation = -config.ecpp_blend * sigma * compensation_raw
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


def _saturate_lateral_error(e_y: float, scale: float) -> float:
    if scale <= 0.0:
        return e_y
    return scale * math.tanh(e_y / scale)

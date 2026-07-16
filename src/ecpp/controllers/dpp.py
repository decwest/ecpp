from __future__ import annotations

import math

import numpy as np

from ..config import EcppConfig
from ..lookahead import (
    PathLocation,
    PathProjection,
    calc_lookahead_location,
    resolve_path_projection,
)


def calc_dpp_parameters(config: EcppConfig) -> tuple[float, float, float, float, float]:
    """Return the paper's two-preview distances and curvature coefficients.

    The coefficients satisfy a1 + a2 = K_y and
    a1 L1 + a2 L2 = K_psi, with L1 = L_d and L2 = 2 L_d.
    This is the DPP definition used in the manuscript and makes its local
    straight-path model equal to ungated ECPP for the same configured gains.
    """

    eps = 1e-12
    l1 = float(config.lookahead_m)
    l2 = 2.0 * l1
    if not math.isfinite(l1) or l1 <= eps:
        raise ValueError("DPP lookahead distance must be positive and finite")

    speed = abs(float(config.dpp_gain_speed))
    v_gain = speed + max(float(config.ecpp_v_epsilon), eps)
    omega_n = float(config.dpp_omega_n)
    zeta = float(config.dpp_zeta)
    k_y = (omega_n / v_gain) ** 2
    k_psi = 2.0 * zeta * omega_n / v_gain
    a1 = (k_psi - k_y * l2) / (l1 - l2)
    a2 = k_y - a1
    if not all(math.isfinite(value) for value in (a1, a2, v_gain)):
        raise ValueError("DPP parameters must be finite")
    return l1, l2, a1, a2, v_gain


def _body_lateral_coordinate(current_pose: np.ndarray, point: np.ndarray) -> float:
    dx = float(point[0] - current_pose[0])
    dy = float(point[1] - current_pose[1])
    heading = float(current_pose[2])
    return -math.sin(heading) * dx + math.cos(heading) * dy


def calc_dpp_preview_lateral_error(
    current_pose: np.ndarray,
    current_location: PathProjection | PathLocation | np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    preview_distance: float,
) -> tuple[float, np.ndarray, PathLocation]:
    projection = resolve_path_projection(
        current_pose, current_location, path, path_distances
    )
    preview = calc_lookahead_location(
        projection, path, path_distances, preview_distance
    )
    y_b = _body_lateral_coordinate(current_pose, preview.position)
    return y_b, preview.position.copy(), preview


def calc_dpp_curvature(
    current_pose: np.ndarray,
    current_location: PathProjection | PathLocation | np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    config: EcppConfig,
) -> tuple[float, np.ndarray]:
    l1, l2, a1, a2, _ = calc_dpp_parameters(config)
    projection = resolve_path_projection(
        current_pose, current_location, path, path_distances
    )
    y_b1, path_pos_1, _ = calc_dpp_preview_lateral_error(
        current_pose, projection, path, path_distances, l1
    )
    y_b2, _, _ = calc_dpp_preview_lateral_error(
        current_pose, projection, path, path_distances, l2
    )
    curvature = a1 * y_b1 + a2 * y_b2
    return float(curvature), path_pos_1

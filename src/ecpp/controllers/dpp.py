from __future__ import annotations

import math

import numpy as np

from ..config import EcppConfig
from ..lookahead import (
    PathLocation,
    PathProjection,
    calc_path_frame_error,
    project_to_path,
    resolve_path_projection,
)


def calc_dpp_parameters(config: EcppConfig) -> tuple[float, float, float, float, float]:
    """Return the Dual Preview Points distances and curvature coefficients.

    The construction follows Wang and Mouri (Trans. JSME, 2025), design
    pattern 3: the far preview distance ``L_1 = dpp_far_factor * L_d`` and the
    target ``(omega_n, zeta)`` are given; the near preview distance ``L_2`` and
    the coefficients ``a_1``, ``a_2`` of the curvature law
    ``kappa = -(a_1 e_p1 + a_2 e_p2)`` follow from three conditions

        a_1 + a_2             = K_y      (natural frequency)
        a_1 L_1 + a_2 L_2     = K_theta  (damping)
        a_1 L_1^2 + a_2 L_2^2 = 2        (a constant-curvature path is followed
                                          without steady-state lateral error in
                                          the linearized model; paper eq. (19))

    with ``K_y = (omega_n / v_g)^2`` and ``K_theta = 2 zeta omega_n / v_g`` in
    curvature units and ``v_g = v + epsilon_v``.  The third condition is what
    distinguishes the paper's DPP from an arbitrary two-point combination; it
    is equivalent to the two Pure Pursuit gains ``2/L_i^2`` being combined with
    weights that sum to one.
    """

    eps = 1e-12
    ld = float(config.lookahead_m)
    far_factor = float(config.dpp_far_factor)
    l1 = far_factor * ld
    if not math.isfinite(l1) or l1 <= eps:
        raise ValueError("DPP far preview distance must be positive and finite")

    speed = abs(float(config.dpp_gain_speed))
    v_gain = speed + max(float(config.ecpp_v_epsilon), eps)
    omega_n = float(config.dpp_omega_n)
    zeta = float(config.dpp_zeta)
    k_y = (omega_n / v_gain) ** 2
    k_psi = 2.0 * zeta * omega_n / v_gain
    denominator = k_psi - l1 * k_y
    if abs(denominator) <= eps:
        raise ValueError("DPP near preview distance is singular at this design point")
    l2 = (2.0 - l1 * k_psi) / denominator
    if not math.isfinite(l2) or l2 <= eps:
        raise ValueError("DPP near preview distance is not positive at this design point")
    if abs(l1 - l2) <= eps:
        raise ValueError("DPP preview distances coincide at this design point")
    a1 = (k_psi - k_y * l2) / (l1 - l2)
    a2 = k_y - a1
    if not all(math.isfinite(value) for value in (a1, a2, v_gain)):
        raise ValueError("DPP parameters must be finite")
    return l1, l2, a1, a2, v_gain


def calc_dpp_preview_lateral_error(
    current_pose: np.ndarray,
    current_location: PathProjection | PathLocation | np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    preview_distance: float,
) -> tuple[float, np.ndarray, PathProjection]:
    """Lateral deviation of a vehicle-axis preview point from the path.

    The preview point lies ``preview_distance`` ahead of the robot along its
    heading (Wang and Mouri, Fig. 8).  Its deviation ``e_p`` is the
    left-positive lateral distance between that point and the reference path,
    measured in the path frame of the point's own closest path location.  On a
    straight path this equals ``e_y + L sin(e_psi)`` exactly, so the curvature
    law reduces to ``-K_y e_y - K_theta sin(e_psi)`` there.
    """

    resolve_path_projection(current_pose, current_location, path, path_distances)
    heading = float(current_pose[2])
    preview_pose = np.array(
        [
            float(current_pose[0]) + float(preview_distance) * math.cos(heading),
            float(current_pose[1]) + float(preview_distance) * math.sin(heading),
            heading,
        ],
        dtype=float,
    )
    preview_projection = project_to_path(preview_pose, path, path_distances)
    e_p, _ = calc_path_frame_error(preview_pose, preview_projection)
    return float(e_p), preview_pose[:2].copy(), preview_projection


def calc_dpp_curvature(
    current_pose: np.ndarray,
    current_location: PathProjection | PathLocation | np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    config: EcppConfig,
) -> tuple[float, np.ndarray]:
    """Curvature ``-(a_1 e_p1 + a_2 e_p2)`` and the far preview point."""

    l1, l2, a1, a2, _ = calc_dpp_parameters(config)
    projection = resolve_path_projection(
        current_pose, current_location, path, path_distances
    )
    e_p1, preview_1, _ = calc_dpp_preview_lateral_error(
        current_pose, projection, path, path_distances, l1
    )
    e_p2, _, _ = calc_dpp_preview_lateral_error(
        current_pose, projection, path, path_distances, l2
    )
    curvature = -(a1 * e_p1 + a2 * e_p2)
    return float(curvature), preview_1

from __future__ import annotations

import math

import numpy as np

from ..config import EcppConfig
from ..lookahead import calc_lookahead_position


def calc_pp_curvature_to_point(current_pose: np.ndarray, target_xy: np.ndarray) -> float:
    dx = float(target_xy[0] - current_pose[0])
    dy = float(target_xy[1] - current_pose[1])
    psi = float(current_pose[2])
    c = math.cos(psi)
    s = math.sin(psi)
    x_body = c * dx + s * dy
    y_body = -s * dx + c * dy
    ell2 = x_body * x_body + y_body * y_body
    if ell2 <= 1e-12:
        return 0.0
    return float(2.0 * y_body / ell2)


def calc_pp_curvature(
    current_pose: np.ndarray,
    current_idx: np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    lookahead_distance: float,
    config: EcppConfig | None = None,
) -> tuple[float, np.ndarray]:
    del config
    lookahead_pos, _ = calc_lookahead_position(current_idx, path, path_distances, lookahead_distance)
    return calc_pp_curvature_to_point(current_pose, lookahead_pos), lookahead_pos

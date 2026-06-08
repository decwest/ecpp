from __future__ import annotations

import numpy as np

from .geometry import calc_path_theta


def calc_index(current_pose: np.ndarray, path: np.ndarray) -> np.intp:
    distances = np.linalg.norm(path[:, :2] - current_pose[:2], axis=1)
    return np.argmin(distances)


def calc_lookahead_index(
    current_idx: np.intp | int,
    path_distances: np.ndarray,
    lookahead_distance: float,
) -> np.intp:
    current_distance = path_distances[int(current_idx)]
    target_distance = current_distance + float(lookahead_distance)
    return np.intp(min(np.searchsorted(path_distances, target_distance), len(path_distances) - 1))


def calc_lookahead_position(
    current_idx: np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    lookahead_distance: float,
) -> tuple[np.ndarray, np.intp]:
    lookahead_idx = calc_lookahead_index(current_idx, path_distances, lookahead_distance)
    return path[lookahead_idx, :2].astype(float, copy=True), lookahead_idx


def calc_lookahead_pose(
    current_idx: np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    lookahead_distance: float,
) -> np.ndarray:
    pos, idx = calc_lookahead_position(current_idx, path, path_distances, lookahead_distance)
    return np.array([float(pos[0]), float(pos[1]), calc_path_theta(path, idx)])

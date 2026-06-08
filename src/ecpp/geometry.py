from __future__ import annotations

import math

import numpy as np


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def calc_path_distances(path: np.ndarray) -> np.ndarray:
    differences = np.diff(path[:, :2], axis=0)
    distances = np.linalg.norm(differences, axis=1)
    return np.concatenate(([0.0], np.cumsum(distances)))


def calc_path_headings(path: np.ndarray) -> np.ndarray:
    if path.shape[1] >= 3:
        return path[:, 2]
    if len(path) == 1:
        return np.array([0.0])
    diffs = np.diff(path[:, :2], axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.concatenate([headings, [headings[-1]]])


def calc_path_theta(path: np.ndarray, idx: np.intp | int) -> float:
    if path.shape[1] >= 3:
        return float(path[int(idx), 2])
    if len(path) == 1:
        return 0.0
    prev_idx = max(int(idx) - 1, 0)
    next_idx = min(int(idx) + 1, len(path) - 1)
    delta = path[next_idx, :2] - path[prev_idx, :2]
    if float(np.linalg.norm(delta)) <= 1e-12:
        return 0.0
    return float(math.atan2(delta[1], delta[0]))


def append_heading_to_path(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) == 0:
        return np.empty((0, 3))
    if len(path_xy) == 1:
        return np.array([[path_xy[0, 0], path_xy[0, 1], 0.0]])
    diffs = np.diff(path_xy[:, :2], axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.concatenate([headings, [headings[-1]]])
    return np.c_[path_xy[:, :2], headings]


def as_pose_path(path: np.ndarray) -> np.ndarray:
    if path.ndim != 2 or path.shape[1] < 2:
        raise ValueError("path must be a two-dimensional array with x/y columns")
    if path.shape[1] >= 3:
        return path[:, :3].astype(float, copy=False)
    return append_heading_to_path(path[:, :2])


def pose_from_path_error(path: np.ndarray, e_y_m: float, e_psi_deg: float) -> np.ndarray:
    start = as_pose_path(path)[0]
    heading = float(start[2])
    left_normal = np.array([-math.sin(heading), math.cos(heading)], dtype=float)
    return np.array([
        float(start[0] + e_y_m * left_normal[0]),
        float(start[1] + e_y_m * left_normal[1]),
        float(normalize_angle(heading + math.radians(e_psi_deg))),
    ])

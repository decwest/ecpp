from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry import append_heading_to_path


def straight_line_path(length: float = 6.0, num_points: int = 600, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    if length <= 0.0:
        raise ValueError("length must be > 0")
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    x = np.linspace(float(x0), float(x0) + float(length), int(num_points), dtype=float)
    y = np.full_like(x, float(y0))
    theta = np.zeros_like(x)
    return np.c_[x, y, theta]


def constant_curvature_arc_path(radius: float = 1.5, angle_deg: float = 90.0, num_points: int = 500) -> np.ndarray:
    if radius <= 0.0:
        raise ValueError("radius must be > 0")
    if angle_deg <= 0.0:
        raise ValueError("angle_deg must be > 0")
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    angle_rad = math.radians(float(angle_deg))
    arc_angle = np.linspace(0.0, angle_rad, int(num_points), dtype=float)
    x = float(radius) * np.sin(arc_angle)
    y = float(radius) * (1.0 - np.cos(arc_angle))
    return np.c_[x, y, arc_angle]


def right_angle_path(segment_length: float = 3.0, points_per_segment: int = 300) -> np.ndarray:
    if segment_length <= 0.0:
        raise ValueError("segment_length must be > 0")
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be > 0")
    x1 = np.linspace(0.0, float(segment_length), int(points_per_segment) + 1)
    y1 = np.zeros_like(x1)
    theta1 = np.zeros_like(x1)
    x2 = np.full(int(points_per_segment) + 1, float(segment_length))
    y2 = np.linspace(0.0, float(segment_length), int(points_per_segment) + 1)
    theta2 = np.full_like(x2, math.pi / 2.0)
    return np.c_[np.concatenate([x1, x2[1:]]), np.concatenate([y1, y2[1:]]), np.concatenate([theta1, theta2[1:]])]


def corner_stress_path(
    segment_lengths: tuple[float, ...] | list[float] = (0.8, 0.45, 0.8, 0.45, 0.8, 0.45, 0.8),
    headings_deg: tuple[float, ...] | list[float] = (0.0, 90.0, -45.0, 45.0, -90.0, 0.0, 90.0),
    points_per_meter: float = 160.0,
) -> np.ndarray:
    if len(segment_lengths) != len(headings_deg):
        raise ValueError("segment_lengths and headings_deg must have the same length")
    points: list[np.ndarray] = []
    current = np.array([0.0, 0.0], dtype=float)
    for index, (length, heading_deg) in enumerate(zip(segment_lengths, headings_deg)):
        if float(length) <= 0.0:
            raise ValueError("segment lengths must be > 0")
        heading = math.radians(float(heading_deg))
        end = current + float(length) * np.array([math.cos(heading), math.sin(heading)], dtype=float)
        num_points = max(int(math.ceil(float(length) * points_per_meter)) + 1, 2)
        segment = np.column_stack([
            np.linspace(current[0], end[0], num_points),
            np.linspace(current[1], end[1], num_points),
        ])
        if index > 0:
            segment = segment[1:]
        points.append(segment)
        current = end
    return append_heading_to_path(np.vstack(points))


def build_path(path_type: str, params: dict[str, Any] | None = None) -> np.ndarray:
    params = {} if params is None else dict(params)
    normalized = path_type.strip().lower().replace("_", "-")
    if normalized in {"straight", "straight-line"}:
        return straight_line_path(**params)
    if normalized in {"arc", "constant-curvature-arc", "constant-curvature"}:
        return constant_curvature_arc_path(**params)
    if normalized in {"right-angle", "corner-90", "l-corner"}:
        return right_angle_path(**params)
    if normalized in {"corner-stress", "practical"}:
        return corner_stress_path(**params)
    raise ValueError(f"unsupported path type: {path_type}")

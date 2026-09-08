from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class PathLocation:
    """A continuous location on a polyline path.

    ``segment_index`` identifies the segment from vertex ``i`` to ``i + 1``.
    Path pose yaw values are deliberately ignored; ``tangent`` is derived from
    the non-zero polyline segment that contains the location.
    """

    segment_index: int
    interpolation: float
    arc_length: float
    position: np.ndarray
    tangent: np.ndarray
    remaining_length: float

    @property
    def heading(self) -> float:
        return float(math.atan2(float(self.tangent[1]), float(self.tangent[0])))


@dataclass(frozen=True)
class PathProjection(PathLocation):
    """Closest continuous path location used by all controllers in one cycle."""

    squared_distance: float


def calc_path_frame_error(
    current_pose: np.ndarray,
    projection: PathProjection | PathLocation,
) -> tuple[float, float]:
    """Return left-positive lateral and wrapped heading errors."""

    offset = np.asarray(current_pose, dtype=float)[:2] - projection.position
    tangent_x = float(projection.tangent[0])
    tangent_y = float(projection.tangent[1])
    e_y = -tangent_y * float(offset[0]) + tangent_x * float(offset[1])
    e_psi = (float(current_pose[2]) - projection.heading + math.pi) % (
        2.0 * math.pi
    ) - math.pi
    return float(e_y), float(e_psi)


def _validate_path(path: np.ndarray, path_distances: np.ndarray | None) -> np.ndarray:
    if path.ndim != 2 or path.shape[1] < 2 or len(path) < 2:
        raise ValueError("path must contain at least two x/y vertices")
    if path_distances is None:
        differences = np.diff(path[:, :2], axis=0)
        path_distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(differences, axis=1))))
    distances = np.asarray(path_distances, dtype=float)
    if distances.shape != (len(path),):
        raise ValueError("path_distances must contain one value per path vertex")
    if np.any(np.diff(distances) < -_EPS):
        raise ValueError("path_distances must be non-decreasing")
    if not np.any(np.diff(distances) > _EPS):
        raise ValueError("path must contain at least one non-zero segment")
    return distances


def project_to_path(
    current_pose: np.ndarray,
    path: np.ndarray,
    path_distances: np.ndarray | None = None,
    minimum_arc_length: float = 0.0,
) -> PathProjection:
    """Project a pose continuously onto a polyline.

    Zero-length segments are ignored. Equal-distance candidates are resolved
    toward the smaller arc length, which makes a crossing deterministic and
    preserves the current branch when ``minimum_arc_length`` is carried from
    the preceding control cycle.  The search is vectorized over all segments;
    the selection rule is the same as the original per-segment scan.
    """

    distances = _validate_path(path, path_distances)
    query = np.asarray(current_pose, dtype=float)[:2]
    total_length = float(distances[-1])
    min_s = float(np.clip(float(minimum_arc_length), 0.0, total_length))

    points = np.asarray(path[:, :2], dtype=float)
    p0 = points[:-1]
    delta = points[1:] - p0
    length = np.linalg.norm(delta, axis=1)
    s_start = distances[:-1]
    s_end = distances[1:]
    valid = (length > _EPS) & ~(s_end < min_s - _EPS)
    indices = np.flatnonzero(valid)

    if len(indices) == 0:
        # This is only reachable when min_s is at the end and all trailing
        # segments have zero length. Sample the final non-zero segment instead.
        location = sample_path_at_arc_length(path, distances, total_length)
        offset = query - location.position
        return PathProjection(
            **location.__dict__,
            squared_distance=float(np.dot(offset, offset)),
        )

    seg_p0 = p0[indices]
    seg_delta = delta[indices]
    seg_length = length[indices]
    seg_s_start = s_start[indices]
    t_min = np.clip((min_s - seg_s_start) / seg_length, 0.0, 1.0)
    t_unclamped = np.einsum("ij,ij->i", query - seg_p0, seg_delta) / (
        seg_length * seg_length
    )
    interpolation = np.minimum(np.maximum(t_unclamped, t_min), 1.0)
    positions = seg_p0 + interpolation[:, None] * seg_delta
    residual = query - positions
    squared = np.einsum("ij,ij->i", residual, residual)
    arc_lengths = seg_s_start + interpolation * seg_length

    best_squared = float(np.min(squared))
    tolerance = _EPS * max(1.0, best_squared)
    tied = np.flatnonzero(squared <= best_squared + tolerance)
    chosen = int(tied[np.argmin(arc_lengths[tied])])

    segment_index = int(indices[chosen])
    return PathProjection(
        segment_index=segment_index,
        interpolation=float(interpolation[chosen]),
        arc_length=float(arc_lengths[chosen]),
        position=positions[chosen].copy(),
        tangent=seg_delta[chosen] / seg_length[chosen],
        remaining_length=max(total_length - float(arc_lengths[chosen]), 0.0),
        squared_distance=float(squared[chosen]),
    )


def resolve_path_projection(
    current_pose: np.ndarray,
    current_location: PathProjection | PathLocation | np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
) -> PathProjection:
    """Return the shared projection, or compute it for a legacy index caller."""

    if isinstance(current_location, PathProjection):
        return current_location
    if isinstance(current_location, PathLocation):
        offset = np.asarray(current_pose, dtype=float)[:2] - current_location.position
        return PathProjection(
            **current_location.__dict__,
            squared_distance=float(np.dot(offset, offset)),
        )
    # A legacy vertex index still carries forward-progress information. Use
    # its arc length as the lower bound so callers cannot jump to an earlier
    # branch at a self-intersection.
    index = int(current_location)
    if index < 0 or index >= len(path_distances):
        raise IndexError("path vertex index is out of range")
    return project_to_path(
        current_pose,
        path,
        path_distances,
        minimum_arc_length=float(path_distances[index]),
    )


def sample_path_at_arc_length(
    path: np.ndarray,
    path_distances: np.ndarray,
    arc_length: float,
) -> PathLocation:
    """Interpolate a continuous polyline location, clamped to both endpoints."""

    distances = _validate_path(path, path_distances)
    total_length = float(distances[-1])
    target_s = float(np.clip(float(arc_length), 0.0, total_length))

    nonzero_segments = np.flatnonzero(np.diff(distances) > _EPS)
    # First non-zero segment whose end lies at or beyond the target arc length
    # (the original linear scan), found with a binary search.
    segment_ends = distances[nonzero_segments + 1]
    position_in_list = int(np.searchsorted(segment_ends, target_s - _EPS, side="left"))
    if position_in_list >= len(nonzero_segments):
        position_in_list = len(nonzero_segments) - 1
    selected_index = int(nonzero_segments[position_in_list])

    p0 = np.asarray(path[selected_index, :2], dtype=float)
    p1 = np.asarray(path[selected_index + 1, :2], dtype=float)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    interpolation = float(np.clip((target_s - float(distances[selected_index])) / length, 0.0, 1.0))
    return PathLocation(
        segment_index=selected_index,
        interpolation=interpolation,
        arc_length=target_s,
        position=p0 + interpolation * delta,
        tangent=delta / length,
        remaining_length=max(total_length - target_s, 0.0),
    )


def calc_index(current_pose: np.ndarray, path: np.ndarray) -> np.intp:
    """Return the closest vertex index (legacy helper).

    Controllers use :func:`project_to_path`; this helper remains for API
    compatibility with analysis code that explicitly needs a vertex index.
    """

    distances = np.linalg.norm(path[:, :2] - current_pose[:2], axis=1)
    return np.argmin(distances)


def calc_lookahead_index(
    current_idx: PathProjection | PathLocation | np.intp | int,
    path_distances: np.ndarray,
    lookahead_distance: float,
) -> np.intp:
    current_distance = (
        float(current_idx.arc_length)
        if isinstance(current_idx, PathLocation)
        else float(path_distances[int(current_idx)])
    )
    target_distance = current_distance + float(lookahead_distance)
    return np.intp(min(np.searchsorted(path_distances, target_distance), len(path_distances) - 1))


def calc_lookahead_location(
    projection: PathProjection | PathLocation,
    path: np.ndarray,
    path_distances: np.ndarray,
    lookahead_distance: float,
) -> PathLocation:
    return sample_path_at_arc_length(
        path,
        path_distances,
        float(projection.arc_length) + max(float(lookahead_distance), 0.0),
    )


def calc_lookahead_position(
    current_idx: PathProjection | PathLocation | np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    lookahead_distance: float,
) -> tuple[np.ndarray, np.intp]:
    if isinstance(current_idx, PathLocation):
        current_distance = float(current_idx.arc_length)
    else:
        current_distance = float(path_distances[int(current_idx)])
    location = sample_path_at_arc_length(
        path,
        path_distances,
        current_distance + max(float(lookahead_distance), 0.0),
    )
    lookahead_vertex_index = min(
        int(np.searchsorted(path_distances, location.arc_length)),
        len(path_distances) - 1,
    )
    return location.position.copy(), np.intp(lookahead_vertex_index)


def calc_lookahead_pose(
    current_idx: PathProjection | PathLocation | np.intp | int,
    path: np.ndarray,
    path_distances: np.ndarray,
    lookahead_distance: float,
) -> np.ndarray:
    if isinstance(current_idx, PathLocation):
        current_distance = float(current_idx.arc_length)
    else:
        current_distance = float(path_distances[int(current_idx)])
    location = sample_path_at_arc_length(
        path,
        path_distances,
        current_distance + max(float(lookahead_distance), 0.0),
    )
    return np.array([float(location.position[0]), float(location.position[1]), location.heading])

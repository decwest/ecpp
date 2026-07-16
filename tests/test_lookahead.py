import math

import numpy as np
import pytest

from ecpp.geometry import calc_path_distances
from ecpp.lookahead import (
    calc_lookahead_position,
    calc_lookahead_location,
    calc_path_frame_error,
    project_to_path,
    resolve_path_projection,
)


def _path(points):
    return np.asarray(points, dtype=float)


def test_sparse_and_dense_straight_paths_give_same_projection_and_carrot():
    sparse = _path([[-1.0, 0.2], [2.0, 0.2]])
    dense = _path([[-1.0, 0.2], [0.0, 0.2], [1.0, 0.2], [2.0, 0.2]])
    pose = np.array([0.0, 0.0, 0.0])

    results = []
    for path in (sparse, dense):
        distances = calc_path_distances(path)
        projection = project_to_path(pose, path, distances)
        carrot = calc_lookahead_location(projection, path, distances, 0.5)
        results.append((projection, carrot))

    for projection, carrot in results:
        np.testing.assert_allclose(projection.position, [0.0, 0.2])
        np.testing.assert_allclose(carrot.position, [0.5, 0.2])
        assert projection.arc_length == pytest.approx(1.0)


def test_lateral_offset_larger_than_lookahead_still_advances_by_path_arc():
    path = _path([[-1.0, 2.0], [2.0, 2.0]])
    distances = calc_path_distances(path)
    projection = project_to_path(np.array([0.0, 0.0, 0.0]), path, distances)
    carrot = calc_lookahead_location(projection, path, distances, 0.5)

    np.testing.assert_allclose(carrot.position, [0.5, 2.0])


def test_exact_corner_uses_incoming_tangent_and_interpolates_outgoing_carrot():
    path = _path([[-1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    distances = calc_path_distances(path)
    pose = np.array([0.0, 0.0, 0.0])
    projection = project_to_path(pose, path, distances)
    carrot = calc_lookahead_location(projection, path, distances, 0.5)

    assert projection.segment_index == 0
    assert projection.heading == pytest.approx(0.0)
    np.testing.assert_allclose(carrot.position, [0.0, 0.5])


def test_projection_on_second_half_of_segment_keeps_continuous_progress():
    path = _path([[0.0, 0.0], [2.0, 0.0]])
    distances = calc_path_distances(path)
    projection = project_to_path(np.array([1.5, 0.2, 0.0]), path, distances)
    carrot = calc_lookahead_location(projection, path, distances, 0.25)

    assert projection.interpolation == pytest.approx(0.75)
    assert projection.arc_length == pytest.approx(1.5)
    np.testing.assert_allclose(carrot.position, [1.75, 0.0])


def test_endpoint_is_clamped_but_nominal_lookahead_is_not_changed():
    path = _path([[0.0, 0.0], [1.0, 0.0]])
    distances = calc_path_distances(path)
    projection = project_to_path(np.array([0.8, 0.1, 0.0]), path, distances)
    carrot = calc_lookahead_location(projection, path, distances, 0.5)

    np.testing.assert_allclose(carrot.position, [1.0, 0.0])
    assert carrot.arc_length == pytest.approx(1.0)
    assert projection.remaining_length == pytest.approx(0.2)


def test_zero_length_segments_are_skipped_and_pose_yaw_is_ignored():
    path = _path([
        [0.0, 0.0, math.pi],
        [0.0, 0.0, -math.pi / 2.0],
        [1.0, 0.0, math.pi / 2.0],
    ])
    distances = calc_path_distances(path)
    pose = np.array([0.25, 0.2, 0.1])
    projection = project_to_path(pose, path, distances)
    e_y, e_psi = calc_path_frame_error(pose, projection)

    assert projection.segment_index == 1
    assert projection.heading == pytest.approx(0.0)
    assert e_y == pytest.approx(0.2)
    assert e_psi == pytest.approx(0.1)


def test_minimum_arc_length_keeps_progress_on_a_later_crossing_branch():
    path = _path([
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
    ])
    distances = calc_path_distances(path)
    pose = np.array([0.0, 0.0, 0.0])

    first = project_to_path(pose, path, distances)
    later = project_to_path(
        pose,
        path,
        distances,
        minimum_arc_length=float(distances[2]),
    )

    assert first.segment_index == 0
    assert later.segment_index == 2


def test_previous_projection_keeps_later_branch_during_sequential_crossing_update():
    path = _path([
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
    ])
    distances = calc_path_distances(path)
    later_branch = project_to_path(
        np.array([0.0, -0.5, math.pi / 2.0]),
        path,
        distances,
        minimum_arc_length=float(distances[2]),
    )
    at_crossing = project_to_path(
        np.array([0.0, 0.0, math.pi / 2.0]),
        path,
        distances,
        minimum_arc_length=later_branch.arc_length,
    )

    assert later_branch.segment_index == 2
    assert at_crossing.segment_index == 2
    assert at_crossing.arc_length >= later_branch.arc_length


def test_legacy_vertex_index_preserves_branch_progress_and_vertex_index_semantics():
    path = _path([
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
    ])
    distances = calc_path_distances(path)
    pose = np.array([0.0, 0.0, math.pi / 2.0])

    projection = resolve_path_projection(pose, 2, path, distances)
    carrot, vertex_index = calc_lookahead_position(
        projection, path, distances, 0.2
    )

    assert projection.segment_index == 2
    np.testing.assert_allclose(carrot, [0.0, 0.2])
    assert vertex_index == 3

import math

import numpy as np
import pytest

from ecpp.config import EcppConfig
from ecpp.controllers.ecpp import calc_ecpp_terms
from ecpp.controllers.gates import gate_abs
from ecpp.controllers.pure_pursuit import calc_pp_curvature
from ecpp.geometry import calc_path_distances
from ecpp.lookahead import project_to_path
from ecpp.paths import straight_line_path


def test_ecpp_gate_off_matches_pp():
    path = straight_line_path(length=2.0, num_points=80)
    distances = calc_path_distances(path)
    pose = np.array([0.0, 0.1, 0.1])
    config = EcppConfig(ecpp_gate_mode="off", ecpp_omega_n=2.0, ecpp_zeta=1.0)
    terms = calc_ecpp_terms(pose, np.array([0.5, 0.0]), 0, path, distances, 0.5, config)
    pp, _ = calc_pp_curvature(pose, 0, path, distances, 0.5)
    assert abs(terms.curvature - pp) < 1e-12
    assert terms.sigma == 0.0


def test_ecpp_always_on_applies_compensation():
    path = straight_line_path(length=2.0, num_points=80)
    distances = calc_path_distances(path)
    pose = np.array([0.0, 0.1, 0.1])
    config = EcppConfig(ecpp_gate_mode="always_on", ecpp_omega_n=3.0, ecpp_zeta=1.2)
    terms = calc_ecpp_terms(pose, np.array([0.5, 0.0]), 0, path, distances, 0.5, config)
    assert terms.sigma == 1.0
    assert abs(terms.compensation) > 1e-6


def test_sigmoid_gate_decreases_between_thresholds():
    assert gate_abs(0.1, 0.1, 0.5) > gate_abs(0.3, 0.1, 0.5) > gate_abs(0.5, 0.1, 0.5)


def test_ecpp_y_gate_uses_relative_error_rate():
    path = straight_line_path(length=2.0, num_points=80)
    distances = calc_path_distances(path)
    lookahead = 0.5
    config = EcppConfig(
        ecpp_gate_mode="sigmoid",
        ecpp_gate_error_on=0.10,
        ecpp_gate_error_off=0.50,
        ecpp_gate_sigmoid_endpoint_value=0.01,
    )

    pose_on = np.array([0.0, math.sqrt(config.ecpp_gate_error_on) * lookahead, 0.0])
    terms_on = calc_ecpp_terms(pose_on, np.array([0.5, 0.0]), 0, path, distances, lookahead, config)
    assert terms_on.sigma_y == pytest.approx(0.99)

    pose_off = np.array([0.0, math.sqrt(config.ecpp_gate_error_off) * lookahead, 0.0])
    terms_off = calc_ecpp_terms(pose_off, np.array([0.5, 0.0]), 0, path, distances, lookahead, config)
    assert terms_off.sigma_y == pytest.approx(0.01)


def test_ecpp_psi_gate_uses_sine_linearization_error_rate():
    path = straight_line_path(length=2.0, num_points=80)
    distances = calc_path_distances(path)
    config = EcppConfig(ecpp_gate_mode="sigmoid", ecpp_gate_error_on=0.10, ecpp_gate_error_off=0.50)

    terms_near = calc_ecpp_terms(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0]), 0, path, distances, 0.5, config)
    assert terms_near.sigma_psi > 0.99

    terms_far = calc_ecpp_terms(np.array([0.0, 0.0, math.pi / 2.0]), np.array([0.5, 0.0]), 0, path, distances, 0.5, config)
    assert terms_far.sigma_psi < config.ecpp_gate_sigmoid_endpoint_value


def test_ecpp_ey_only_gate_ignores_heading_error():
    path = straight_line_path(length=2.0, num_points=80)
    distances = calc_path_distances(path)
    pose_on_path_large_heading = np.array([0.0, 0.0, math.pi / 2.0])

    config_ey = EcppConfig(ecpp_gate_mode="ey_only")
    terms_ey = calc_ecpp_terms(pose_on_path_large_heading, np.array([0.5, 0.0]), 0, path, distances, 0.5, config_ey)
    assert terms_ey.sigma == pytest.approx(terms_ey.sigma_y)
    assert terms_ey.sigma > 0.98

    config_product = EcppConfig(ecpp_gate_mode="sigmoid")
    terms_product = calc_ecpp_terms(pose_on_path_large_heading, np.array([0.5, 0.0]), 0, path, distances, 0.5, config_product)
    assert terms_product.sigma < 0.02


def test_ecpp_ey_only_gate_closes_far_from_path():
    path = straight_line_path(length=2.0, num_points=80)
    distances = calc_path_distances(path)
    lookahead = 0.5
    config = EcppConfig(ecpp_gate_mode="ey_only", ecpp_gate_error_on=0.10, ecpp_gate_error_off=0.50)

    pose_far = np.array([0.0, 2.0 * lookahead, 0.0])
    terms_far = calc_ecpp_terms(pose_far, np.array([0.5, 0.0]), 0, path, distances, lookahead, config)
    assert terms_far.sigma < config.ecpp_gate_sigmoid_endpoint_value + 1e-9


def test_ecpp_golden_vector_uses_additive_regularization_and_nominal_arc_length():
    path = np.array([[-1.0, 0.0, math.pi], [2.0, 0.0, math.pi]])
    distances = calc_path_distances(path)
    pose = np.array([0.0, 0.15, 0.0])
    projection = project_to_path(pose, path, distances)
    config = EcppConfig(
        ecpp_gate_mode="always_on",
        ecpp_omega_n=1.1892,
        ecpp_zeta=1.0,
        ecpp_v_epsilon=0.05,
    )

    terms = calc_ecpp_terms(
        pose, np.array([0.5, 0.0]), projection,
        path, distances, 0.5, config,
    )

    assert terms.v_gain == pytest.approx(0.55)
    assert terms.lookahead_distance == pytest.approx(0.5)
    assert terms.projection.segment_index == 0
    assert terms.projection.interpolation == pytest.approx(1.0 / 3.0)
    assert terms.projection.arc_length == pytest.approx(1.0)
    assert terms.projection.remaining_length == pytest.approx(2.0)
    np.testing.assert_allclose(terms.projection.position, [0.0, 0.0])
    np.testing.assert_allclose(terms.lookahead_pos, [0.5, 0.0])
    assert terms.lookahead_location.arc_length == pytest.approx(1.5)
    assert terms.e_y == pytest.approx(0.15)
    assert terms.e_psi == pytest.approx(0.0)
    assert terms.k_y == pytest.approx((1.1892 / 0.55) ** 2)
    assert terms.k_y_pp == pytest.approx(8.0)
    assert terms.k_psi == pytest.approx(2.0 * 1.1892 / 0.55)
    assert terms.k_psi_pp == pytest.approx(4.0)
    assert terms.dK_y == pytest.approx(terms.k_y - terms.k_y_pp)
    assert terms.dK_psi == pytest.approx(terms.k_psi - terms.k_psi_pp)
    assert terms.sigma == pytest.approx(1.0)
    expected_kappa_pp = -0.3 / (0.5 ** 2 + 0.15 ** 2)
    expected = expected_kappa_pp - (
        ((1.1892 / 0.55) ** 2 - 8.0) * 0.15
    )
    assert terms.kappa_pp == pytest.approx(expected_kappa_pp)
    assert terms.curvature == pytest.approx(expected)


def test_ecpp_gated_heading_golden_vector_covers_heading_compensation_sign():
    path = np.array([[-1.0, 0.0], [2.0, 0.0]])
    distances = calc_path_distances(path)
    pose = np.array([0.0, 0.15, 0.2])
    config = EcppConfig(
        ecpp_gate_mode="ey_only",
        ecpp_omega_n=1.1892,
        ecpp_zeta=1.0,
        ecpp_v_epsilon=0.05,
    )
    terms = calc_ecpp_terms(
        pose,
        np.array([0.5, 0.0]),
        project_to_path(pose, path, distances),
        path,
        distances,
        0.5,
        config,
    )

    target_delta = np.array([0.5, -0.15])
    target_y_body = (
        -math.sin(pose[2]) * target_delta[0]
        + math.cos(pose[2]) * target_delta[1]
    )
    expected_kappa_pp = 2.0 * target_y_body / float(
        np.dot(target_delta, target_delta)
    )
    expected_sigma = gate_abs(
        (0.15 / 0.5) ** 2,
        config.ecpp_gate_error_on,
        config.ecpp_gate_error_off,
        config.ecpp_gate_sigmoid_endpoint_value,
    )
    expected_curvature = expected_kappa_pp - expected_sigma * (
        ((1.1892 / 0.55) ** 2 - 8.0) * 0.15
        + (2.0 * 1.1892 / 0.55 - 4.0) * math.sin(0.2)
    )

    assert terms.e_y == pytest.approx(0.15)
    assert terms.e_psi == pytest.approx(0.2)
    assert terms.sigma == pytest.approx(expected_sigma)
    assert terms.kappa_pp == pytest.approx(expected_kappa_pp)
    assert terms.curvature == pytest.approx(expected_curvature)


def test_ecpp_endpoint_clamp_keeps_nominal_arc_length_gains():
    path = np.array([[0.0, 0.0], [1.0, 0.0]])
    distances = calc_path_distances(path)
    pose = np.array([0.9, 0.1, 0.0])
    projection = project_to_path(pose, path, distances)
    terms = calc_ecpp_terms(
        pose,
        np.array([0.5, 0.0]),
        projection,
        path,
        distances,
        0.5,
        EcppConfig(ecpp_gate_mode="always_on"),
    )

    np.testing.assert_allclose(terms.lookahead_pos, [1.0, 0.0])
    assert terms.projection.remaining_length == pytest.approx(0.1)
    assert terms.lookahead_distance == pytest.approx(0.5)
    assert terms.k_y_pp == pytest.approx(2.0 / 0.5**2)
    assert terms.k_psi_pp == pytest.approx(2.0 / 0.5)


def test_ecpp_ignores_stored_path_orientations():
    path_a = np.array([[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    path_b = np.array([
        [-1.0, 0.0, math.pi / 2.0],
        [2.0, 0.0, -math.pi],
    ])
    pose = np.array([0.0, 0.15, 0.2])
    config = EcppConfig(ecpp_gate_mode="always_on")

    results = []
    for path in (path_a, path_b):
        distances = calc_path_distances(path)
        projection = project_to_path(pose, path, distances)
        results.append(
            calc_ecpp_terms(
                pose, np.array([0.5, 0.0]), projection,
                path, distances, 0.5, config,
            )
        )

    assert results[0].e_y == pytest.approx(results[1].e_y)
    assert results[0].e_psi == pytest.approx(results[1].e_psi)
    assert results[0].curvature == pytest.approx(results[1].curvature)


def test_ecpp_is_invariant_to_equivalent_sparse_and_dense_path_sampling():
    paths = (
        np.array([[-1.0, 0.0], [2.0, 0.0]]),
        np.array([[-1.0, 0.0], [-0.2, 0.0], [0.7, 0.0], [2.0, 0.0]]),
    )
    pose = np.array([0.35, 0.15, 0.2])
    config = EcppConfig(
        ecpp_gate_mode="sigmoid",
        ecpp_omega_n=1.1892,
        ecpp_zeta=1.0,
        ecpp_v_epsilon=0.05,
    )

    results = []
    for path in paths:
        distances = calc_path_distances(path)
        projection = project_to_path(pose, path, distances)
        results.append(
            calc_ecpp_terms(
                pose,
                np.array([0.5, 0.0]),
                projection,
                path,
                distances,
                0.5,
                config,
            )
        )

    for attribute in (
        "e_y",
        "e_psi",
        "kappa_pp",
        "sigma",
        "dK_y",
        "dK_psi",
        "curvature",
    ):
        assert getattr(results[0], attribute) == pytest.approx(
            getattr(results[1], attribute)
        )
    assert results[0].projection.arc_length == pytest.approx(
        results[1].projection.arc_length
    )
    np.testing.assert_allclose(results[0].lookahead_pos, results[1].lookahead_pos)

import numpy as np
import pytest

from ecpp.config import EcppConfig
from ecpp.controllers.dpp import calc_dpp_curvature, calc_dpp_parameters
from ecpp.geometry import calc_path_distances
from ecpp.lookahead import project_to_path
from ecpp.paths import straight_line_path


def test_dpp_coefficients_match_paper_gain_equations():
    config = EcppConfig(
        lookahead_m=0.5,
        dpp_omega_n=1.189,
        dpp_zeta=1.0,
        dpp_gain_speed=0.5,
        ecpp_v_epsilon=0.05,
    )

    l1, l2, a1, a2, v_gain = calc_dpp_parameters(config)
    k_y = (config.dpp_omega_n / v_gain) ** 2
    k_psi = 2.0 * config.dpp_zeta * config.dpp_omega_n / v_gain

    assert l1 == pytest.approx(0.5)
    assert l2 == pytest.approx(1.0)
    assert v_gain == pytest.approx(0.55)
    assert a1 + a2 == pytest.approx(k_y)
    assert a1 * l1 + a2 * l2 == pytest.approx(k_psi)


def test_dpp_uses_shared_continuous_projection_and_interpolated_previews():
    path = straight_line_path(length=2.0, num_points=3)
    distances = calc_path_distances(path)
    pose = np.array([0.25, 0.1, 0.0])
    projection = project_to_path(pose, path, distances)
    config = EcppConfig(
        lookahead_m=0.5,
        dpp_omega_n=1.189,
        dpp_zeta=1.0,
        dpp_gain_speed=0.5,
        ecpp_v_epsilon=0.05,
    )

    curvature, lookahead = calc_dpp_curvature(
        pose, projection, path, distances, config
    )
    _, _, _, _, v_gain = calc_dpp_parameters(config)
    expected_k_y = (config.dpp_omega_n / v_gain) ** 2

    np.testing.assert_allclose(lookahead, [0.75, 0.0])
    assert curvature == pytest.approx(-expected_k_y * 0.1)


def test_dpp_nonzero_heading_uses_body_frame_preview_coordinates():
    path = straight_line_path(length=2.0, num_points=3)
    distances = calc_path_distances(path)
    pose = np.array([0.25, 0.1, 0.2])
    projection = project_to_path(pose, path, distances)
    config = EcppConfig(
        lookahead_m=0.5,
        dpp_omega_n=1.1892,
        dpp_zeta=1.0,
        dpp_gain_speed=0.5,
        ecpp_v_epsilon=0.05,
    )

    curvature, _ = calc_dpp_curvature(
        pose, projection, path, distances, config
    )
    _, _, a1, a2, _ = calc_dpp_parameters(config)
    y_b1 = -np.sin(pose[2]) * 0.5 + np.cos(pose[2]) * -0.1
    y_b2 = -np.sin(pose[2]) * 1.0 + np.cos(pose[2]) * -0.1

    assert curvature == pytest.approx(a1 * y_b1 + a2 * y_b2)

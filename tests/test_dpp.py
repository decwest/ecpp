import numpy as np
import pytest

from ecpp.config import EcppConfig
from ecpp.controllers.dpp import calc_dpp_curvature, calc_dpp_parameters
from ecpp.geometry import calc_path_distances
from ecpp.lookahead import project_to_path
from ecpp.paths import straight_line_path


def _config(lookahead_m=0.5, omega_n=1.189, zeta=1.0):
    return EcppConfig(
        lookahead_m=lookahead_m,
        dpp_omega_n=omega_n,
        dpp_zeta=zeta,
        dpp_gain_speed=0.5,
        ecpp_v_epsilon=0.05,
    )


def test_dpp_coefficients_satisfy_the_paper_three_conditions():
    config = _config()
    l1, l2, a1, a2, v_gain = calc_dpp_parameters(config)
    k_y = (config.dpp_omega_n / v_gain) ** 2
    k_psi = 2.0 * config.dpp_zeta * config.dpp_omega_n / v_gain

    assert v_gain == pytest.approx(0.55)
    assert l1 == pytest.approx(2.0 * config.lookahead_m)
    assert l2 > 0.0
    assert a1 + a2 == pytest.approx(k_y)
    assert a1 * l1 + a2 * l2 == pytest.approx(k_psi)
    # Wang-Mouri eq. (19): the pure-pursuit gains 2/L_i^2 are combined with
    # weights summing to one, so a constant-curvature path is reproduced.
    assert a1 * l1**2 + a2 * l2**2 == pytest.approx(2.0)
    assert a1 * l1**2 / 2.0 + a2 * l2**2 / 2.0 == pytest.approx(1.0)


def test_dpp_paper_design_point_geometry():
    # The frozen test-2 design point: omega_n = omega_n_max(1.0 m), zeta = 1.
    config = _config(lookahead_m=1.0, omega_n=1.1328719291, zeta=1.0)
    l1, l2, a1, a2, _ = calc_dpp_parameters(config)
    assert l1 == pytest.approx(2.0)
    assert l2 == pytest.approx(1.4286, abs=2e-3)
    assert a1 == pytest.approx(-3.404, abs=2e-3)
    assert a2 == pytest.approx(7.647, abs=2e-3)


def test_dpp_rejects_singular_or_negative_near_preview():
    # K_theta = L_1 K_y makes the near preview distance singular.
    with pytest.raises(ValueError):
        calc_dpp_parameters(_config(lookahead_m=0.5, omega_n=1.5556349186, zeta=np.sqrt(2.0)))


def test_dpp_preview_points_lie_on_the_vehicle_axis():
    path = straight_line_path(length=4.0, num_points=3)
    distances = calc_path_distances(path)
    pose = np.array([0.25, 0.1, 0.0])
    projection = project_to_path(pose, path, distances)
    config = _config()

    curvature, preview = calc_dpp_curvature(pose, projection, path, distances, config)
    l1, _, _, _, v_gain = calc_dpp_parameters(config)
    expected_k_y = (config.dpp_omega_n / v_gain) ** 2

    np.testing.assert_allclose(preview, [0.25 + l1, 0.1])
    assert curvature == pytest.approx(-expected_k_y * 0.1)


def test_dpp_straight_path_law_is_minus_ky_ey_minus_ktheta_sin_epsi():
    path = straight_line_path(length=12.0, num_points=3)
    distances = calc_path_distances(path)
    config = _config()
    _, _, a1, a2, v_gain = calc_dpp_parameters(config)
    k_y = a1 + a2
    l1, l2, _, _, _ = calc_dpp_parameters(config)
    k_psi = a1 * l1 + a2 * l2
    for e_y, e_psi in ((0.1, 0.2), (2.0, -1.2), (-0.7, 0.9)):
        pose = np.array([3.0, e_y, e_psi])
        projection = project_to_path(pose, path, distances)
        curvature, _ = calc_dpp_curvature(pose, projection, path, distances, config)
        assert curvature == pytest.approx(-k_y * e_y - k_psi * np.sin(e_psi), abs=1e-9)

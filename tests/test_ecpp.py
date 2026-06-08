import math

import numpy as np
import pytest

from ecpp.config import EcppConfig
from ecpp.controllers.ecpp import calc_ecpp_terms
from ecpp.controllers.gates import gate_abs
from ecpp.controllers.pure_pursuit import calc_pp_curvature
from ecpp.geometry import calc_path_distances
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

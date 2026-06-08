import numpy as np

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

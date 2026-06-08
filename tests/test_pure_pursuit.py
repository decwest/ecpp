import numpy as np

from ecpp.controllers.pure_pursuit import calc_pp_curvature
from ecpp.geometry import calc_path_distances
from ecpp.paths import straight_line_path


def test_pp_zero_error_curvature_is_zero():
    path = straight_line_path(length=2.0, num_points=50)
    kappa, _ = calc_pp_curvature(np.array([0.0, 0.0, 0.0]), 0, path, calc_path_distances(path), 0.5)
    assert abs(kappa) < 1e-12


def test_pp_positive_lateral_error_commands_right_turn_on_straight_path():
    path = straight_line_path(length=2.0, num_points=50)
    kappa, _ = calc_pp_curvature(np.array([0.0, 0.1, 0.0]), 0, path, calc_path_distances(path), 0.5)
    assert kappa < 0.0

import pytest
import numpy as np

from ecpp.config import EcppConfig
from ecpp.controllers.dpp import calc_dpp_curvature, calc_dpp_parameters
from ecpp.geometry import calc_path_distances
from ecpp.paths import straight_line_path


def test_dpp_returns_finite_curvature_for_valid_parameters():
    path = straight_line_path(length=2.0, num_points=80)
    config = EcppConfig(dpp_omega_n=2.0, dpp_zeta=1.0, dpp_preview_time=0.2, dpp_gain_speed=0.5)
    kappa, lookahead = calc_dpp_curvature(np.array([0.0, 0.1, 0.0]), 0, path, calc_path_distances(path), config)
    assert np.isfinite(kappa)
    assert lookahead.shape == (2,)


def test_dpp_rejects_singular_parameters():
    config = EcppConfig(dpp_omega_n=1.0, dpp_zeta=1.0, dpp_preview_time=1.0, dpp_gain_speed=0.5)
    with pytest.raises(ValueError):
        calc_dpp_parameters(config)

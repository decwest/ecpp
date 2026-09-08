import math

import numpy as np
import pytest

from ecpp.config import EcppConfig
from ecpp.geometry import pose_from_path_error
from ecpp.paths import straight_line_path
from ecpp.simulation.path_tracking import InitialCondition, MethodVariant, PathScenario, run_path_tracking


def test_path_tracking_keeps_raw_request_and_uses_instantaneous_clip():
    path = straight_line_path(length=1.0, num_points=80)
    config = EcppConfig(
        v_max=0.5,
        omega_max=1.5,
        dt=1.0 / 30.0,
        goal_tolerance_dist=0.03,
    )
    scenario = PathScenario("straight", "Straight", path, max_steps=20)
    condition = InitialCondition("test", 0.2, 15.0, pose_from_path_error(path, 0.2, 15.0))
    variant = MethodVariant("pp", "PP", "pp", "none", 0.05)
    result = run_path_tracking(scenario, condition, variant, config)
    assert np.allclose(result.v_cmd, config.v_max)
    assert np.allclose(result.omega_cmd, np.clip(result.curvatures * config.v_max, -config.omega_max, config.omega_max))
    assert np.any(np.abs(result.omega_raw) > 1.5)
    assert np.max(np.abs(result.omega_cmd)) <= 1.5
    assert len(result.times) == len(result.poses) == len(result.omega_raw)
    np.testing.assert_allclose(result.poses[0], condition.pose)
    assert result.omega_raw[0] == result.curvatures[0] * config.v_max
    assert result.omega_raw[0] != 0.0


def test_direct_dpp_tracking_uses_variant_lookahead_not_config_metadata():
    path = straight_line_path(length=4.0, num_points=3)
    # dpp_omega_n = 1.5 keeps the paper's near preview distance positive.
    config = EcppConfig(lookahead_m=1.2, v_max=0.5, dpp_omega_n=1.5)
    scenario = PathScenario("straight", "Straight", path, max_steps=0)
    pose = np.array([0.0, 0.1, 0.0])
    condition = InitialCondition("test", 0.1, 0.0, pose)
    variant = MethodVariant("dpp", "DPP", "dpp", "none", 0.5)

    result = run_path_tracking(scenario, condition, variant, config)

    # The far preview point lies 2 L_d ahead on the vehicle axis, with the
    # variant's L_d = 0.5 m rather than the config's 1.2 m.
    np.testing.assert_allclose(result.lookahead[0], [1.0, 0.1])


def test_initial_condition_ignores_incorrect_stored_path_orientation():
    path = np.array([[0.0, 0.0, math.pi], [1.0, 0.0, math.pi]])

    pose = pose_from_path_error(path, 0.2, 10.0)

    np.testing.assert_allclose(pose[:2], [0.0, 0.2])
    assert pose[2] == pytest.approx(math.radians(10.0))

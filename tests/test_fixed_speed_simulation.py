import numpy as np

from ecpp.config import EcppConfig
from ecpp.geometry import pose_from_path_error
from ecpp.paths import straight_line_path
from ecpp.simulation.fixed_speed import InitialCondition, MethodVariant, PathScenario, run_fixed_speed


def test_fixed_speed_commands_use_vmax_and_clipped_kappa_relation():
    path = straight_line_path(length=1.0, num_points=80)
    config = EcppConfig(v_max=0.5, omega_max=0.4, dt=0.02, goal_tolerance_dist=0.03)
    scenario = PathScenario("straight", "Straight", path, max_steps=20)
    condition = InitialCondition("test", 0.1, 15.0, pose_from_path_error(path, 0.1, 15.0))
    variant = MethodVariant("pp", "PP", "pp", "none", 0.2)
    result = run_fixed_speed(scenario, condition, variant, config)
    assert np.allclose(result.v_cmd, config.v_max)
    assert np.allclose(result.omega_cmd, np.clip(result.curvatures * config.v_max, -config.omega_max, config.omega_max))

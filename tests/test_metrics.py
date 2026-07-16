import numpy as np
import math

from ecpp.config import EcppConfig, experiment_config_from_mapping
from ecpp.evaluation.metrics import (
    count_zero_crossings,
    first_corner_index,
    first_threshold_time,
    max_overshoot_percent_y,
    max_overshoot_y_m,
    rise_time_percent_y,
    settling_time_percent_y,
    signed_lateral_error_to_line,
    summarize_result,
    summarize_corner_response,
)
from ecpp.evaluation.tables import (
    test1_results_path_table as render_test1_results_path_table,
    test1_results_table as render_test1_results_table,
)
from ecpp.paths import corner_stress_path
from ecpp.simulation.path_tracking import InitialCondition, MethodVariant, PathScenario, TrackingResult


def test_zero_crossing_count_uses_deadband():
    values = np.array([0.1, 0.0, -0.1, 0.1, 0.001])
    assert count_zero_crossings(values, deadband=0.01) == 2


def test_first_threshold_time_returns_first_ratio_hit():
    values = np.array([10.0, 5.0, 1.0, 0.5])
    times = np.array([0.0, 1.0, 2.0, 3.0])
    assert first_threshold_time(values, times, 0.1) == 2.0


def test_percent_settling_and_rise_time_for_monotonic_lateral_error():
    ey = np.array([1.0, 0.9, 0.5, 0.1, 0.02, 0.01])
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    assert settling_time_percent_y(ey, times, 0.02) == 4.0
    assert rise_time_percent_y(ey, times, 0.10, 0.90) == 2.0


def test_max_overshoot_amount_uses_initial_lateral_error_sign():
    ey = np.array([1.0, 0.2, -0.25, -0.1, 0.0])

    assert max_overshoot_y_m(ey) == 0.25


def test_max_overshoot_percent_uses_initial_lateral_error():
    ey = np.array([1.0, 0.2, -0.25, -0.1, 0.0])

    assert max_overshoot_percent_y(ey) == 25.0


def test_lateral_response_metrics_are_nan_for_zero_initial_error():
    ey = np.array([0.0, 0.1, 0.0])
    times = np.array([0.0, 1.0, 2.0])

    assert math.isnan(settling_time_percent_y(ey, times, 0.02))
    assert math.isnan(rise_time_percent_y(ey, times, 0.10, 0.90))
    assert math.isnan(max_overshoot_y_m(ey))
    assert math.isnan(max_overshoot_percent_y(ey))


def test_corner_response_metrics_use_lookahead_detection_and_post_corner_line():
    path = corner_stress_path(
        start_xy=(0.0, 2.0),
        segment_lengths=(2.0, 3.0),
        headings_deg=(-90.0, 0.0),
        points_per_meter=10.0,
    )
    corner_idx = first_corner_index(path)
    assert corner_idx is not None
    assert np.allclose(path[corner_idx, :2], [0.0, 0.0])

    result = TrackingResult(
        scenario=PathScenario("right_angle", "Right-angle", path, 10, path),
        condition=InitialCondition("ey0_epsi0", 0.0, 0.0, path[0]),
        variant=MethodVariant("pp_L1", "PP", "pp", "none", 1.0),
        times=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        poses=np.array([
            [0.0, 1.0, -math.pi / 2.0],
            [0.0, 0.5, -math.pi / 2.0],
            [0.1, 0.2, 0.0],
            [1.0, -0.05, 0.0],
            [2.0, 0.0, 0.0],
        ]),
        lookahead=np.array([
            [0.0, 0.5],
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
        ]),
        curvatures=np.zeros(5),
        omega_raw=np.array([0.0, 2.0, -2.0, 0.0, 0.0]),
        omega_cmd=np.array([0.0, 1.0, -1.0, 0.0, 0.0]),
        v_cmd=np.full(5, 0.5),
        sigma=np.full(5, np.nan),
        sigma_y=np.full(5, np.nan),
        sigma_psi=np.full(5, np.nan),
        path_s=np.zeros(5),
        e_y=np.zeros(5),
        e_psi=np.zeros(5),
        goal_reached=True,
        max_steps_reached=False,
    )

    metrics = summarize_corner_response(result)

    assert metrics["corner_detection_time_s"] == 2.0
    assert metrics["corner_initial_e_y_m"] == 0.2
    assert metrics["corner_max_overshoot_m"] == 0.05

    summary = summarize_result(result, EcppConfig(omega_max=1.5))
    assert summary["clip_ratio"] == 0.4
    assert summary["mean_abs_omega_raw_rate_radps2"] == 2.0
    assert summary["max_abs_omega_raw_rate_radps2"] == 4.0


def test_signed_lateral_error_to_post_corner_line_y_zero():
    values = signed_lateral_error_to_line(
        np.array([[0.0, 0.3], [1.0, -0.1]]),
        np.array([0.0, 0.0]),
        0.0,
    )

    assert np.allclose(values, [0.3, -0.1])


def test_result_table_formats_unreached_travel_time_as_dash():
    experiment = experiment_config_from_mapping({"lookahead": {"values_m": [1.0]}})
    rows = [{
        "path_name": "straight",
        "path_label": "Straight",
        "initial_e_y_m": 0.3,
        "initial_e_psi_deg": -30.0,
        "method_label": "PP",
        "lookahead_m": 1.0,
        "mean_abs_e_y_m": 0.1,
        "mean_abs_heading_error_deg": 5.0,
        "settling_time_s": 2.0,
        "rise_time_s": 1.0,
        "max_overshoot_m": 0.125,
        "travel_time_s": float("nan"),
        "max_abs_kappa_inv_m": 2.5,
    }]

    table = render_test1_results_table(experiment, rows)
    assert "$\\kappa_{\\max}$" in table
    assert "$T_m$" in table
    assert "0.125 & -- & 2.50" in table


def test_path_result_table_excludes_zero_zero_condition():
    experiment = experiment_config_from_mapping({
        "lookahead": {"values_m": [1.0]},
        "initial_conditions": {
            "e_y_m": [0.0, 0.3],
            "e_psi_deg": [0.0, -30.0],
        },
    })
    base = {
        "path_name": "straight",
        "path_label": "Straight",
        "method_label": "PP",
        "lookahead_m": 1.0,
        "mean_abs_e_y_m": 0.1,
        "mean_abs_heading_error_deg": 5.0,
        "settling_time_s": 2.0,
        "rise_time_s": 1.0,
        "max_overshoot_m": 0.125,
        "travel_time_s": float("nan"),
        "max_abs_kappa_inv_m": 2.5,
    }
    rows = [
        {**base, "initial_e_y_m": 0.0, "initial_e_psi_deg": 0.0},
        {**base, "initial_e_y_m": 0.0, "initial_e_psi_deg": -30.0},
        {**base, "initial_e_y_m": 0.3, "initial_e_psi_deg": 0.0},
    ]

    table = render_test1_results_path_table(experiment, rows, "straight")

    assert "e_y(0)=0.00" in table
    assert "e_\\theta(0)=-30" in table
    assert "e_y(0)=0.30" in table
    assert "$\\kappa_{\\max}$" in table
    assert "e_y(0)=0.00\\,\\mathrm{m},\\ e_\\theta(0)=0^\\circ" not in table

import math

import pytest

from ecpp.config import EcppConfig, config_for_variant, experiment_config_from_mapping
from ecpp.controllers.dpp import calc_dpp_parameters
from ecpp.simulation.runner import build_scenarios, nominal_variants


def test_dpp_far_preview_is_twice_ld_and_near_preview_follows_the_paper():
    config = EcppConfig(v_max=0.5)
    variant_config = config_for_variant(
        config,
        lookahead_m=1.2,
        omega_n=1.0,
        zeta=1.0,
        gate_mode="off",
    )
    assert math.isclose(variant_config.dpp_gain_speed, config.v_max)
    l1, l2, a1, a2, v_gain = calc_dpp_parameters(variant_config)
    k_y = (1.0 / v_gain) ** 2
    k_psi = 2.0 / v_gain
    assert math.isclose(l1, 2.4)
    assert math.isclose(l2, (2.0 - l1 * k_psi) / (k_psi - l1 * k_y))
    assert math.isclose(a1 * l1**2 + a2 * l2**2, 2.0)
    assert math.isclose(v_gain, 0.55)


def test_simulation_angular_clip_is_fixed_at_one_point_five_radps():
    with pytest.raises(ValueError, match="fixed at 1.5"):
        EcppConfig(omega_max=0.5)


def test_experiment_config_reads_additive_velocity_regularization():
    experiment = experiment_config_from_mapping({"ecpp": {"v_epsilon": 0.05}})

    assert math.isclose(experiment.control.ecpp_v_epsilon, 0.05)


def test_experiment_config_accepts_single_lookahead_value():
    experiment = experiment_config_from_mapping({"lookahead": {"values_m": [1.2]}})

    assert experiment.lookahead_values_m == (1.2,)
    assert math.isclose(experiment.lookahead_short_m, 1.2)
    assert math.isclose(experiment.lookahead_long_m, 1.2)


def test_experiment_config_keeps_legacy_short_long_lookaheads():
    experiment = experiment_config_from_mapping({"lookahead": {"short_m": 0.5, "long_m": 1.2}})

    assert experiment.lookahead_values_m == (0.5, 1.2)
    assert math.isclose(experiment.lookahead_short_m, 0.5)
    assert math.isclose(experiment.lookahead_long_m, 1.2)


def test_experiment_config_accepts_direct_omega_n_sweep():
    experiment = experiment_config_from_mapping({
        "gain_sweep": {
            "rho": [1.0, 2.0],
            "omega_n": [1.2, 1.8],
            "zeta": [1.0],
            "representative": {"omega_n": 1.8, "zeta": 1.0},
        }
    })

    assert experiment.rho_values == ()
    assert experiment.omega_n_values == (1.2, 1.8)
    assert math.isclose(experiment.representative_omega_n, 1.8)


def test_experiment_config_accepts_scalar_direct_omega_n():
    experiment = experiment_config_from_mapping({
        "gain_sweep": {
            "omega_n": 1.2,
            "zeta": [1.0],
        }
    })

    assert experiment.rho_values == ()
    assert experiment.omega_n_values == (1.2,)
    assert math.isclose(experiment.representative_omega_n, 1.2)


def test_nominal_variants_use_direct_omega_n_when_configured():
    experiment = experiment_config_from_mapping({
        "lookahead": {"values_m": [1.2]},
        "gain_sweep": {
            "omega_n": [1.2],
            "zeta": [1.0],
        },
    })

    variants = nominal_variants(experiment)
    controlled = [variant for variant in variants if variant.label != "PP"]
    assert {variant.label for variant in controlled} == {"DPP", "ECPP without gate", "ECPP"}
    assert all(variant.rho is None for variant in controlled)
    assert all(math.isclose(variant.omega_n, 1.2) for variant in controlled if variant.omega_n is not None)
    assert all("wn1p2" in variant.key for variant in controlled)


def test_nominal_variants_respect_enabled_methods():
    experiment = experiment_config_from_mapping({
        "methods": ["PP", "ECPP"],
        "lookahead": {"values_m": [1.2]},
        "gain_sweep": {"rho": [2.0], "zeta": [1.0]},
    })

    variants = nominal_variants(experiment)
    assert experiment.method_labels == ("PP", "ECPP")
    assert [variant.label for variant in variants] == ["PP", "ECPP"]


def test_experiment_config_expands_initial_condition_grid():
    experiment = experiment_config_from_mapping({
        "initial_conditions": {
            "e_y_m": [0.0, 0.3, 0.6],
            "e_psi_deg": [0.0, -30.0, -60.0],
        }
    })

    assert len(experiment.initial_conditions) == 9
    assert experiment.initial_conditions[0] == (0.0, 0.0)
    assert experiment.initial_conditions[-1] == (0.6, -60.0)
    assert experiment.initial_e_y_m == 0.0
    assert experiment.initial_e_psi_deg == 0.0


def test_experiment_config_keeps_single_legacy_initial_condition():
    experiment = experiment_config_from_mapping({
        "initial_condition": {"e_y_m": 0.1, "e_psi_deg": -15.0}
    })

    assert experiment.initial_conditions == ((0.1, -15.0),)


def test_arc_path_can_use_extended_control_path_and_shorter_evaluation_path():
    experiment = experiment_config_from_mapping({
        "paths": [
            {
                "key": "arc",
                "label": "Arc",
                "type": "arc",
                "params": {"radius": 1.5, "angle_deg": 180.0, "num_points": 120},
                "evaluation": {"angle_deg": 90.0, "num_points": 60},
            }
        ]
    })

    scenario = build_scenarios(experiment)[0]
    assert math.isclose(scenario.path[-1, 2], math.pi)
    assert math.isclose(scenario.evaluation_path[-1, 2], math.pi / 2.0)
    assert math.isclose(scenario.goal_pose[2], math.pi / 2.0)


def test_straight_path_can_use_extended_control_path_and_shorter_evaluation_path():
    experiment = experiment_config_from_mapping({
        "paths": [
            {
                "key": "straight",
                "label": "Straight",
                "type": "straight",
                "params": {"length": 8.0, "num_points": 80},
                "evaluation": {"length": 6.0, "num_points": 60},
            }
        ]
    })

    scenario = build_scenarios(experiment)[0]
    assert math.isclose(scenario.path[-1, 0], 8.0)
    assert math.isclose(scenario.evaluation_path[-1, 0], 6.0)
    assert math.isclose(scenario.goal_pose[0], 6.0)

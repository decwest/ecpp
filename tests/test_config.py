import math

from ecpp.config import EcppConfig, config_for_variant, experiment_config_from_mapping
from ecpp.simulation.runner import build_scenarios, nominal_variants


def test_default_dpp_first_preview_distance_matches_lookahead_target():
    config = EcppConfig(v_max=0.5)
    variant_config = config_for_variant(
        config,
        lookahead_m=1.2,
        omega_n=1.0,
        zeta=1.0,
        gate_mode="off",
    )
    assert math.isclose(variant_config.dpp_gain_speed, config.v_max)
    assert math.isclose(variant_config.dpp_preview_time * variant_config.dpp_gain_speed, 1.2)


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

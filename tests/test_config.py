import math

from ecpp.config import EcppConfig, config_for_variant


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

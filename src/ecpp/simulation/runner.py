from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import ExperimentConfig, config_for_variant
from ..geometry import pose_from_path_error
from ..paths import build_path
from .fixed_speed import FixedSpeedResult, InitialCondition, MethodVariant, PathScenario, run_fixed_speed


@dataclass(frozen=True)
class AccessExperimentOutput:
    results: dict[str, FixedSpeedResult]


def run_access_experiment(experiment: ExperimentConfig) -> AccessExperimentOutput:
    results: dict[str, FixedSpeedResult] = {}
    for scenario in build_scenarios(experiment):
        condition = initial_condition(scenario, experiment.initial_e_y_m, experiment.initial_e_psi_deg)
        for variant in nominal_variants(experiment):
            omega_n = variant.omega_n
            zeta = variant.zeta if variant.zeta is not None else experiment.representative_zeta
            config = config_for_variant(
                experiment.control,
                lookahead_m=variant.lookahead_m,
                omega_n=omega_n if omega_n is not None else representative_omega_n(experiment, variant.lookahead_m),
                zeta=zeta,
                gate_mode=variant.gate_mode if variant.method == "ecpp" else "off",
            )
            key = result_key("test1", scenario.key, condition.key, variant)
            results[key] = run_fixed_speed(scenario, condition, variant, config)
    return AccessExperimentOutput(results=results)


def build_scenarios(experiment: ExperimentConfig) -> tuple[PathScenario, ...]:
    scenarios = []
    for spec in experiment.paths:
        scenarios.append(PathScenario(
            key=spec.key,
            label=spec.label,
            path=build_path(spec.type, spec.params),
            max_steps=experiment.max_steps,
        ))
    return tuple(scenarios)


def initial_condition(scenario: PathScenario, e_y_m: float, e_psi_deg: float) -> InitialCondition:
    pose = pose_from_path_error(scenario.path, e_y_m, e_psi_deg)
    return InitialCondition(condition_key(e_y_m, e_psi_deg), e_y_m, e_psi_deg, pose)


def nominal_variants(experiment: ExperimentConfig) -> tuple[MethodVariant, ...]:
    variants: list[MethodVariant] = []
    for lookahead in (experiment.lookahead_short_m, experiment.lookahead_long_m):
        variants.append(MethodVariant(variant_key("PP", lookahead, None, None), "PP", "pp", "none", lookahead))
        for rho in experiment.rho_values:
            for zeta in experiment.zeta_values:
                omega_n = rho * math.sqrt(2.0) * experiment.control.v_max / lookahead
                variants.extend([
                    MethodVariant(variant_key("DPP", lookahead, rho, zeta), "DPP", "dpp", "none", lookahead, rho, zeta, omega_n),
                    MethodVariant(
                        variant_key("ECPP without gate", lookahead, rho, zeta),
                        "ECPP without gate",
                        "ecpp",
                        "always_on",
                        lookahead,
                        rho,
                        zeta,
                        omega_n,
                    ),
                    MethodVariant(variant_key("ECPP", lookahead, rho, zeta), "ECPP", "ecpp", "sigmoid", lookahead, rho, zeta, omega_n),
                ])
    return tuple(variants)


def representative_omega_n(experiment: ExperimentConfig, lookahead_m: float) -> float:
    return experiment.representative_rho * math.sqrt(2.0) * experiment.control.v_max / lookahead_m


def is_representative(experiment: ExperimentConfig, variant: MethodVariant) -> bool:
    return (
        variant.rho is not None
        and variant.zeta is not None
        and abs(variant.rho - experiment.representative_rho) < 1e-9
        and abs(variant.zeta - experiment.representative_zeta) < 1e-9
    )


def result_key(phase: str, scenario_key: str, condition_key_value: str, variant: MethodVariant) -> str:
    return f"{phase}_{scenario_key}_{condition_key_value}_{variant.key}"


def condition_key(e_y_m: float, e_psi_deg: float) -> str:
    return f"ey{slug_float(e_y_m)}_epsi{slug_float(e_psi_deg)}"


def variant_key(label: str, lookahead: float, rho: float | None, zeta: float | None) -> str:
    parts = [label.lower().replace(" ", "_"), f"L{slug_float(lookahead)}"]
    if rho is not None:
        parts.append(f"rho{slug_float(rho)}")
    if zeta is not None:
        parts.append(f"zeta{slug_float(zeta)}")
    return "_".join(parts)


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")

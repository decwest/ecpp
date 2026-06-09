from __future__ import annotations

import math
from dataclasses import dataclass
import warnings

from ..config import ExperimentConfig, config_for_variant
from ..geometry import pose_from_path_error
from ..paths import build_path
from .path_tracking import TrackingResult, InitialCondition, MethodVariant, PathScenario, run_path_tracking


@dataclass(frozen=True)
class AccessExperimentOutput:
    results: dict[str, TrackingResult]


def run_access_experiment(experiment: ExperimentConfig) -> AccessExperimentOutput:
    results: dict[str, TrackingResult] = {}
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
            try:
                results[key] = run_path_tracking(scenario, condition, variant, config)
            except ValueError as exc:
                if variant.method != "dpp":
                    raise
                warnings.warn(f"skip invalid DPP variant {variant.key}: {exc}", RuntimeWarning, stacklevel=2)
    return AccessExperimentOutput(results=results)


def build_scenarios(experiment: ExperimentConfig) -> tuple[PathScenario, ...]:
    scenarios = []
    for spec in experiment.paths:
        path = build_path(spec.type, spec.params)
        reference_path = path
        if spec.evaluation_type is not None or spec.evaluation_params:
            evaluation_params = {**spec.params, **spec.evaluation_params}
            reference_path = build_path(spec.evaluation_type or spec.type, evaluation_params)
        scenarios.append(PathScenario(
            key=spec.key,
            label=spec.label,
            path=path,
            max_steps=experiment.max_steps,
            reference_path=reference_path,
        ))
    return tuple(scenarios)


def initial_condition(scenario: PathScenario, e_y_m: float, e_psi_deg: float) -> InitialCondition:
    pose = pose_from_path_error(scenario.path, e_y_m, e_psi_deg)
    return InitialCondition(condition_key(e_y_m, e_psi_deg), e_y_m, e_psi_deg, pose)


def nominal_variants(experiment: ExperimentConfig) -> tuple[MethodVariant, ...]:
    variants: list[MethodVariant] = []
    for lookahead in experiment.lookahead_values_m:
        variants.append(MethodVariant(variant_key("PP", lookahead, None, None, None), "PP", "pp", "none", lookahead))
        if experiment.omega_n_values:
            for omega_n in experiment.omega_n_values:
                for zeta in experiment.zeta_values:
                    variants.extend(method_variants_for_design(lookahead, None, zeta, omega_n))
        else:
            for rho in experiment.rho_values:
                for zeta in experiment.zeta_values:
                    omega_n = rho * math.sqrt(2.0) * experiment.control.v_max / lookahead
                    variants.extend(method_variants_for_design(lookahead, rho, zeta, omega_n))
    return tuple(variants)


def method_variants_for_design(
    lookahead: float,
    rho: float | None,
    zeta: float,
    omega_n: float,
) -> list[MethodVariant]:
    return [
        MethodVariant(variant_key("DPP", lookahead, rho, zeta, omega_n), "DPP", "dpp", "none", lookahead, rho, zeta, omega_n),
        MethodVariant(
            variant_key("ECPP without gate", lookahead, rho, zeta, omega_n),
            "ECPP without gate",
            "ecpp",
            "always_on",
            lookahead,
            rho,
            zeta,
            omega_n,
        ),
        MethodVariant(variant_key("ECPP", lookahead, rho, zeta, omega_n), "ECPP", "ecpp", "sigmoid", lookahead, rho, zeta, omega_n),
    ]


def representative_omega_n(experiment: ExperimentConfig, lookahead_m: float) -> float:
    if experiment.representative_omega_n is not None:
        return experiment.representative_omega_n
    return experiment.representative_rho * math.sqrt(2.0) * experiment.control.v_max / lookahead_m


def is_representative(experiment: ExperimentConfig, variant: MethodVariant) -> bool:
    if experiment.representative_omega_n is not None:
        return (
            variant.omega_n is not None
            and variant.zeta is not None
            and abs(variant.omega_n - experiment.representative_omega_n) < 1e-9
            and abs(variant.zeta - experiment.representative_zeta) < 1e-9
        )
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


def variant_key(label: str, lookahead: float, rho: float | None, zeta: float | None, omega_n: float | None) -> str:
    parts = [label.lower().replace(" ", "_"), f"L{slug_float(lookahead)}"]
    if rho is not None:
        parts.append(f"rho{slug_float(rho)}")
    elif omega_n is not None:
        parts.append(f"wn{slug_float(omega_n)}")
    if zeta is not None:
        parts.append(f"zeta{slug_float(zeta)}")
    return "_".join(parts)


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


DEFAULT_METHOD_LABELS = ("PP", "DPP", "ECPP without gate", "ECPP")
METHOD_ALIASES = {
    "pp": "PP",
    "pure-pursuit": "PP",
    "pure_pursuit": "PP",
    "pure pursuit": "PP",
    "dpp": "DPP",
    "ecpp-without-gate": "ECPP without gate",
    "ecpp_without_gate": "ECPP without gate",
    "ecpp without gate": "ECPP without gate",
    "ecpp-nogate": "ECPP without gate",
    "ecpp_nogate": "ECPP without gate",
    "ecpp-no-gate": "ECPP without gate",
    "ecpp_no_gate": "ECPP without gate",
    "ecpp": "ECPP",
    "ecpp-ey": "ECPP ey",
    "ecpp_ey": "ECPP ey",
    "ecpp ey": "ECPP ey",
    "ecpp-ey-gate": "ECPP ey",
    "ecpp_ey_gate": "ECPP ey",
    "ecpp ey gate": "ECPP ey",
}


@dataclass(frozen=True)
class EcppConfig:
    lookahead_m: float = 1.20
    v_max: float = 0.50
    omega_max: float = 1.00
    dt: float = 0.020
    goal_tolerance_dist: float = 0.02
    goal_tolerance_heading: float = math.radians(2.0)
    settling_e_y: float = 0.02
    settling_e_psi: float = math.radians(2.0)
    settling_hold_time: float = 0.5
    zero_crossing_deadband: float = 0.01
    ecpp_omega_n: float = 1.0
    ecpp_zeta: float = 1.0
    ecpp_v_epsilon: float = 0.05
    ecpp_gate_error_on: float = 0.10
    ecpp_gate_error_off: float = 0.50
    ecpp_gate_sigmoid_endpoint_value: float = 0.01
    ecpp_gate_mode: str = "ey_only"
    dpp_omega_n: float = 1.0
    dpp_zeta: float = 1.0
    dpp_preview_time: float = 0.5
    dpp_gain_speed: float = 0.50
    dpp_v_min: float = 0.05

    def __post_init__(self) -> None:
        if self.lookahead_m <= 0.0:
            raise ValueError("lookahead_m must be > 0")
        if self.v_max <= 0.0:
            raise ValueError("v_max must be > 0")
        if self.omega_max <= 0.0:
            raise ValueError("omega_max must be > 0")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.ecpp_omega_n <= 0.0 or self.dpp_omega_n <= 0.0:
            raise ValueError("omega_n values must be > 0")
        if self.ecpp_zeta <= 0.0 or self.dpp_zeta <= 0.0:
            raise ValueError("zeta values must be > 0")
        if self.dpp_preview_time <= 0.0:
            raise ValueError("dpp_preview_time must be > 0")
        if self.ecpp_v_epsilon <= 0.0:
            raise ValueError("ecpp_v_epsilon must be > 0")
        if self.ecpp_gate_error_on < 0.0:
            raise ValueError("gate error on threshold must be non-negative")
        if self.ecpp_gate_error_off <= self.ecpp_gate_error_on:
            raise ValueError("gate error off threshold must be greater than on threshold")
        if not 0.0 < self.ecpp_gate_sigmoid_endpoint_value < 0.5:
            raise ValueError("sigmoid endpoint value must be in (0, 0.5)")
        if self.ecpp_gate_mode not in {"sigmoid", "ey_only", "always_on", "off"}:
            raise ValueError("ecpp_gate_mode must be sigmoid, ey_only, always_on, or off")


@dataclass(frozen=True)
class PathSpec:
    key: str
    label: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    evaluation_type: str | None = None
    evaluation_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    control: EcppConfig = field(default_factory=EcppConfig)
    method_labels: tuple[str, ...] = DEFAULT_METHOD_LABELS
    lookahead_short_m: float = 0.50
    lookahead_long_m: float = 1.20
    lookahead_values_m: tuple[float, ...] = ()
    initial_e_y_m: float = 0.10
    initial_e_psi_deg: float = 15.0
    initial_conditions: tuple[tuple[float, float], ...] = ()
    rho_values: tuple[float, ...] = (1.0, 1.5, 2.0)
    omega_n_values: tuple[float, ...] = ()
    zeta_values: tuple[float, ...] = (1.0 / math.sqrt(2.0), 1.0, 1.4)
    representative_rho: float = 1.5
    representative_omega_n: float | None = None
    representative_zeta: float = 1.0
    max_steps: int = 1100
    output_dir: Path = Path("results/access_ecpp_fixed_speed")
    export_tex_project_dir: Path | None = Path("../tex_docker_environment/projects/fumiya_ieee_access")
    paths: tuple[PathSpec, ...] = (
        PathSpec("straight", "Straight", "straight", {"length": 6.0, "num_points": 700}),
        PathSpec("arc", "Arc", "arc", {"radius": 1.5, "angle_deg": 90.0, "num_points": 600}),
    )
    lookahead_sweep_values_m: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00, 1.20)

    def __post_init__(self) -> None:
        values = self.lookahead_values_m
        if not values:
            values = (self.lookahead_short_m, self.lookahead_long_m)
        values = tuple(float(value) for value in values)
        if not values:
            raise ValueError("at least one lookahead distance is required")
        method_labels = tuple(_canonical_method_label(label) for label in self.method_labels)
        if not method_labels:
            raise ValueError("at least one method is required")
        if any(value <= 0.0 for value in values):
            raise ValueError("lookahead distances must be > 0")
        initial_conditions = self.initial_conditions
        if not initial_conditions:
            initial_conditions = ((float(self.initial_e_y_m), float(self.initial_e_psi_deg)),)
        initial_conditions = tuple((float(e_y), float(e_psi)) for e_y, e_psi in initial_conditions)
        if not initial_conditions:
            raise ValueError("at least one initial condition is required")
        if any(value <= 0.0 for value in self.rho_values):
            raise ValueError("rho values must be > 0")
        if any(value <= 0.0 for value in self.omega_n_values):
            raise ValueError("omega_n values must be > 0")
        if any(value <= 0.0 for value in self.zeta_values):
            raise ValueError("zeta values must be > 0")
        if self.representative_rho <= 0.0:
            raise ValueError("representative_rho must be > 0")
        if self.representative_omega_n is not None and self.representative_omega_n <= 0.0:
            raise ValueError("representative_omega_n must be > 0")
        if self.representative_zeta <= 0.0:
            raise ValueError("representative_zeta must be > 0")
        object.__setattr__(self, "lookahead_values_m", values)
        object.__setattr__(self, "lookahead_short_m", values[0])
        object.__setattr__(self, "lookahead_long_m", values[-1])
        object.__setattr__(self, "method_labels", method_labels)
        object.__setattr__(self, "initial_conditions", initial_conditions)
        object.__setattr__(self, "initial_e_y_m", initial_conditions[0][0])
        object.__setattr__(self, "initial_e_psi_deg", initial_conditions[0][1])


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    data = _load_yaml_with_extends(path)
    return experiment_config_from_mapping(data, base_dir=Path.cwd())


def experiment_config_from_mapping(data: Mapping[str, Any], base_dir: Path | None = None) -> ExperimentConfig:
    base_dir = Path(".") if base_dir is None else base_dir
    control = EcppConfig()

    control_data = _mapping(data.get("control"))
    gate_data = _mapping(data.get("gate"))
    control = replace(
        control,
        v_max=_float(control_data, "v_max", control.v_max),
        omega_max=_float(control_data, "omega_max", control.omega_max),
        dt=_float(control_data, "dt", control.dt),
        goal_tolerance_dist=_float(control_data, "goal_tolerance_dist", control.goal_tolerance_dist),
        goal_tolerance_heading=math.radians(_float(control_data, "goal_tolerance_heading_deg", math.degrees(control.goal_tolerance_heading))),
        ecpp_gate_mode=str(gate_data.get("mode", control.ecpp_gate_mode)),
        ecpp_gate_sigmoid_endpoint_value=_float(gate_data, "sigmoid_endpoint_value", control.ecpp_gate_sigmoid_endpoint_value),
        ecpp_gate_error_on=_float(gate_data, "error_on", control.ecpp_gate_error_on),
        ecpp_gate_error_off=_float(gate_data, "error_off", control.ecpp_gate_error_off),
    )

    lookahead = _mapping(data.get("lookahead"))
    initial = _mapping(data.get("initial_condition"))
    initial_conditions_data = data.get("initial_conditions")
    sweep = _mapping(data.get("gain_sweep"))
    representative = _mapping(sweep.get("representative"))
    experiment = _mapping(data.get("experiment"))
    lookahead_sweep = _mapping(data.get("lookahead_sweep"))
    lookahead_values = _lookahead_values(lookahead)
    omega_n_values = _optional_float_tuple(sweep, ("omega_n", "omega_n_radps", "omega_n_values"))
    representative_omega_n = _optional_float(representative, ("omega_n", "omega_n_radps"))
    if omega_n_values and representative_omega_n is None:
        representative_omega_n = omega_n_values[0]

    output_dir = Path(str(experiment.get("output_dir", "results/access_ecpp_fixed_speed")))
    export_raw = experiment.get("export_tex_project_dir", "../tex_docker_environment/projects/fumiya_ieee_access")
    export_dir = None if export_raw in {None, ""} else _resolve_path(base_dir, Path(str(export_raw)))

    return ExperimentConfig(
        control=control,
        method_labels=_method_labels(data.get("methods")),
        lookahead_short_m=lookahead_values[0],
        lookahead_long_m=lookahead_values[-1],
        lookahead_values_m=lookahead_values,
        initial_e_y_m=_float(initial, "e_y_m", 0.10),
        initial_e_psi_deg=_float(initial, "e_psi_deg", 15.0),
        initial_conditions=_initial_conditions(initial_conditions_data, initial),
        rho_values=() if omega_n_values else tuple(float(v) for v in sweep.get("rho", (1.0, 1.5, 2.0))),
        omega_n_values=omega_n_values,
        zeta_values=tuple(float(v) for v in sweep.get("zeta", (1.0 / math.sqrt(2.0), 1.0, 1.4))),
        representative_rho=_float(representative, "rho", 1.5),
        representative_omega_n=representative_omega_n,
        representative_zeta=_float(representative, "zeta", 1.0),
        max_steps=int(experiment.get("max_steps", 1100)),
        output_dir=_resolve_path(base_dir, output_dir),
        export_tex_project_dir=export_dir,
        paths=_path_specs(data.get("paths")),
        lookahead_sweep_values_m=tuple(float(v) for v in lookahead_sweep.get("values_m", (0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00, 1.20))),
    )


def config_for_variant(
    config: EcppConfig,
    lookahead_m: float,
    omega_n: float,
    zeta: float,
    gate_mode: str,
) -> EcppConfig:
    preview_time = lookahead_m / config.v_max
    return replace(
        config,
        lookahead_m=lookahead_m,
        ecpp_omega_n=omega_n,
        ecpp_zeta=zeta,
        ecpp_gate_mode=gate_mode,
        dpp_omega_n=omega_n,
        dpp_zeta=zeta,
        dpp_preview_time=preview_time,
        dpp_gain_speed=config.v_max,
    )


def _load_yaml_with_extends(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError("YAML root must be a mapping")
    extends = data.get("extends")
    if extends is None:
        return dict(data)
    parent_path = (path.parent / str(extends)).resolve()
    parent = _load_yaml_with_extends(parent_path)
    child = dict(data)
    child.pop("extends", None)
    return _deep_merge(parent, child)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _path_specs(raw: Any) -> tuple[PathSpec, ...]:
    if raw is None:
        return ExperimentConfig().paths
    specs = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("path entries must be mappings")
        evaluation = _mapping(item.get("evaluation"))
        specs.append(PathSpec(
            key=str(item["key"]),
            label=str(item.get("label", item["key"])),
            type=str(item.get("type", item["key"])),
            params=dict(_mapping(item.get("params"))),
            evaluation_type=str(evaluation["type"]) if "type" in evaluation else None,
            evaluation_params=_evaluation_params(evaluation),
        ))
    return tuple(specs)


def _initial_conditions(raw: Any, legacy_initial: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        e_y_values = _float_values(raw.get("e_y_m", legacy_initial.get("e_y_m", 0.10)))
        e_psi_values = _float_values(raw.get("e_psi_deg", legacy_initial.get("e_psi_deg", 15.0)))
        return tuple((e_y, e_psi) for e_y in e_y_values for e_psi in e_psi_values)
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise ValueError("initial_conditions must be a mapping or a list of mappings")
    conditions = []
    for item in raw:
        item_mapping = _mapping(item)
        if not item_mapping:
            raise ValueError("initial_conditions list entries must be mappings")
        conditions.append((
            _float(item_mapping, "e_y_m", _float(legacy_initial, "e_y_m", 0.10)),
            _float(item_mapping, "e_psi_deg", _float(legacy_initial, "e_psi_deg", 15.0)),
        ))
    return tuple(conditions)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _method_labels(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_METHOD_LABELS
    if isinstance(raw, Mapping):
        raw = raw.get("enabled", raw.get("include", ()))
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        labels = (raw,)
    else:
        labels = tuple(raw)
    if not labels:
        raise ValueError("methods must contain at least one method")
    return tuple(_canonical_method_label(str(label)) for label in labels)


def _canonical_method_label(label: str) -> str:
    normalized = label.strip().lower().replace("/", "-")
    normalized = " ".join(normalized.split())
    normalized = normalized.replace(" ", "-")
    if normalized in METHOD_ALIASES:
        return METHOD_ALIASES[normalized]
    normalized_spaces = normalized.replace("-", " ")
    if normalized_spaces in METHOD_ALIASES:
        return METHOD_ALIASES[normalized_spaces]
    allowed = ", ".join(dict.fromkeys(METHOD_ALIASES.values()))
    raise ValueError(f"unsupported method: {label}. Allowed methods: {allowed}")


def _evaluation_params(data: Mapping[str, Any]) -> dict[str, Any]:
    if "params" in data:
        return dict(_mapping(data.get("params")))
    return {str(key): value for key, value in data.items() if key not in {"type", "label"}}


def _float(data: Mapping[str, Any], key: str, default: float) -> float:
    return float(data.get(key, default))


def _optional_float(data: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in data and data[key] is not None:
            return float(data[key])
    return None


def _optional_float_tuple(data: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[float, ...]:
    for key in keys:
        if key in data and data[key] is not None:
            raw = data[key]
            if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
                values = (float(raw),)
            else:
                values = tuple(float(value) for value in raw)
            if not values:
                raise ValueError(f"gain_sweep.{key} must contain at least one value")
            return values
    return ()


def _float_values(raw: Any) -> tuple[float, ...]:
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        return (float(raw),)
    values = tuple(float(value) for value in raw)
    if not values:
        raise ValueError("float value lists must contain at least one value")
    return values


def _lookahead_values(data: Mapping[str, Any]) -> tuple[float, ...]:
    if "values_m" in data:
        values = tuple(float(value) for value in data["values_m"])
    elif "value_m" in data:
        values = (float(data["value_m"]),)
    elif "m" in data:
        values = (float(data["m"]),)
    else:
        values = (_float(data, "short_m", 0.50), _float(data, "long_m", 1.20))
    if not values:
        raise ValueError("lookahead.values_m must contain at least one value")
    if any(value <= 0.0 for value in values):
        raise ValueError("lookahead distances must be > 0")
    return values


def _resolve_path(base_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()

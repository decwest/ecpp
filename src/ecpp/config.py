from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EcppConfig:
    lookahead_m: float = 1.20
    v_max: float = 0.50
    omega_max: float = 1.00
    dt: float = 0.020
    goal_tolerance_dist: float = 0.06
    goal_tolerance_heading: float = math.radians(8.0)
    settling_e_y: float = 0.02
    settling_e_psi: float = math.radians(5.0)
    settling_hold_time: float = 0.5
    error_norm_heading_scale: float = 0.50
    zero_crossing_deadband: float = 0.01
    ecpp_omega_n: float = 1.0
    ecpp_zeta: float = 1.0
    ecpp_v_min: float = 0.05
    ecpp_v_epsilon: float = 0.05
    ecpp_gain_speed_regularization: str = "floor"
    ecpp_lateral_gate_on_ratio: float = 0.316
    ecpp_lateral_gate_off_ratio: float = 0.707
    ecpp_heading_gate_on: float = 0.749
    ecpp_heading_gate_off: float = 1.496
    ecpp_gate_sigmoid_endpoint_value: float = 0.01
    ecpp_gate_mode: str = "sigmoid"
    ecpp_lateral_saturation_ratio: float = 0.0
    ecpp_blend: float = 1.0
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
        if self.ecpp_lateral_gate_off_ratio <= self.ecpp_lateral_gate_on_ratio:
            raise ValueError("lateral gate off ratio must be greater than on ratio")
        if self.ecpp_heading_gate_off <= self.ecpp_heading_gate_on:
            raise ValueError("heading gate off must be greater than heading gate on")
        if not 0.0 < self.ecpp_gate_sigmoid_endpoint_value < 0.5:
            raise ValueError("sigmoid endpoint value must be in (0, 0.5)")
        if self.ecpp_gate_mode not in {"sigmoid", "smoothstep", "always_on", "off"}:
            raise ValueError("ecpp_gate_mode must be sigmoid, smoothstep, always_on, or off")
        if self.ecpp_gain_speed_regularization not in {"floor", "epsilon"}:
            raise ValueError("ecpp_gain_speed_regularization must be floor or epsilon")


@dataclass(frozen=True)
class PathSpec:
    key: str
    label: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    control: EcppConfig = field(default_factory=EcppConfig)
    lookahead_short_m: float = 0.50
    lookahead_long_m: float = 1.20
    initial_e_y_m: float = 0.10
    initial_e_psi_deg: float = 15.0
    rho_values: tuple[float, ...] = (1.0, 1.5, 2.0)
    zeta_values: tuple[float, ...] = (1.0 / math.sqrt(2.0), 1.0, 1.4)
    representative_rho: float = 1.5
    representative_zeta: float = 1.0
    max_steps: int = 1100
    output_dir: Path = Path("results/access_ecpp_fixed_speed")
    export_tex_project_dir: Path | None = Path("../tex_docker_environment/projects/fumiya_ieee_access")
    paths: tuple[PathSpec, ...] = (
        PathSpec("straight", "Straight", "straight", {"length": 6.0, "num_points": 700}),
        PathSpec("arc", "Arc", "arc", {"radius": 1.5, "angle_deg": 90.0, "num_points": 600}),
    )
    lookahead_sweep_values_m: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00, 1.20)


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
        ecpp_lateral_gate_on_ratio=_float(gate_data, "lateral_on_ratio", control.ecpp_lateral_gate_on_ratio),
        ecpp_lateral_gate_off_ratio=_float(gate_data, "lateral_off_ratio", control.ecpp_lateral_gate_off_ratio),
        ecpp_heading_gate_on=_float(gate_data, "heading_on", control.ecpp_heading_gate_on),
        ecpp_heading_gate_off=_float(gate_data, "heading_off", control.ecpp_heading_gate_off),
    )

    lookahead = _mapping(data.get("lookahead"))
    initial = _mapping(data.get("initial_condition"))
    sweep = _mapping(data.get("gain_sweep"))
    representative = _mapping(sweep.get("representative"))
    experiment = _mapping(data.get("experiment"))
    lookahead_sweep = _mapping(data.get("lookahead_sweep"))

    output_dir = Path(str(experiment.get("output_dir", "results/access_ecpp_fixed_speed")))
    export_raw = experiment.get("export_tex_project_dir", "../tex_docker_environment/projects/fumiya_ieee_access")
    export_dir = None if export_raw in {None, ""} else _resolve_path(base_dir, Path(str(export_raw)))

    return ExperimentConfig(
        control=control,
        lookahead_short_m=_float(lookahead, "short_m", 0.50),
        lookahead_long_m=_float(lookahead, "long_m", 1.20),
        initial_e_y_m=_float(initial, "e_y_m", 0.10),
        initial_e_psi_deg=_float(initial, "e_psi_deg", 15.0),
        rho_values=tuple(float(v) for v in sweep.get("rho", (1.0, 1.5, 2.0))),
        zeta_values=tuple(float(v) for v in sweep.get("zeta", (1.0 / math.sqrt(2.0), 1.0, 1.4))),
        representative_rho=_float(representative, "rho", 1.5),
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
    preview_time = min(lookahead_m / config.v_max, 0.5 / omega_n)
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
        specs.append(PathSpec(
            key=str(item["key"]),
            label=str(item.get("label", item["key"])),
            type=str(item.get("type", item["key"])),
            params=dict(_mapping(item.get("params"))),
        ))
    return tuple(specs)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(data: Mapping[str, Any], key: str, default: float) -> float:
    return float(data.get(key, default))


def _resolve_path(base_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()

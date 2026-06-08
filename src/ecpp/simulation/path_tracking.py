from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import EcppConfig
from ..controllers.dpp import calc_dpp_curvature
from ..controllers.ecpp import EcppTerms, calc_ecpp_terms
from ..controllers.pure_pursuit import calc_pp_curvature
from ..geometry import calc_path_distances, normalize_angle
from ..lookahead import calc_index


@dataclass(frozen=True)
class PathScenario:
    key: str
    label: str
    path: np.ndarray
    max_steps: int


@dataclass(frozen=True)
class InitialCondition:
    key: str
    e_y_m: float
    e_psi_deg: float
    pose: np.ndarray


@dataclass(frozen=True)
class MethodVariant:
    key: str
    label: str
    method: str
    gate_mode: str
    lookahead_m: float
    rho: float | None = None
    zeta: float | None = None
    omega_n: float | None = None


@dataclass(frozen=True)
class TrackingResult:
    scenario: PathScenario
    condition: InitialCondition
    variant: MethodVariant
    times: np.ndarray
    poses: np.ndarray
    lookahead: np.ndarray
    curvatures: np.ndarray
    omega_raw: np.ndarray
    omega_cmd: np.ndarray
    v_cmd: np.ndarray
    sigma: np.ndarray
    sigma_y: np.ndarray
    sigma_psi: np.ndarray
    goal_reached: bool
    max_steps_reached: bool


def run_path_tracking(
    scenario: PathScenario,
    condition: InitialCondition,
    variant: MethodVariant,
    config: EcppConfig,
) -> TrackingResult:
    path = scenario.path
    path_distances = calc_path_distances(path)
    goal_pose = path[-1, :3]
    current_pose = condition.pose.astype(float, copy=True)
    current_omega = 0.0

    times = [0.0]
    poses = [current_pose.copy()]
    lookahead_points: list[np.ndarray] = []
    curvatures = [0.0]
    omega_raw_values = [0.0]
    omega_cmd_values = [0.0]
    v_cmd_values = [config.v_max]
    sigmas = [float("nan")]
    sigma_y_values = [float("nan")]
    sigma_psi_values = [float("nan")]
    goal_reached = False
    max_steps_reached = False

    for _ in range(scenario.max_steps):
        if position_goal_reached(current_pose, goal_pose, config.goal_tolerance_dist):
            goal_reached = True
            break

        current_idx = calc_index(current_pose, path)
        current_velocity = np.array([config.v_max, current_omega], dtype=float)
        curvature, lookahead_pos, terms = compute_curvature(
            current_pose=current_pose,
            current_velocity=current_velocity,
            current_idx=current_idx,
            path=path,
            path_distances=path_distances,
            variant=variant,
            config=config,
        )
        omega_raw = float(curvature * config.v_max)
        omega_cmd = float(np.clip(omega_raw, -config.omega_max, config.omega_max))
        next_pose = step_unicycle(current_pose, config.v_max, omega_cmd, config.dt)

        lookahead_points.append(np.asarray(lookahead_pos, dtype=float)[:2])
        curvatures.append(float(curvature))
        omega_raw_values.append(omega_raw)
        omega_cmd_values.append(omega_cmd)
        v_cmd_values.append(config.v_max)
        if terms is None:
            sigmas.append(float("nan"))
            sigma_y_values.append(float("nan"))
            sigma_psi_values.append(float("nan"))
        else:
            sigmas.append(float(terms.sigma))
            sigma_y_values.append(float(terms.sigma_y))
            sigma_psi_values.append(float(terms.sigma_psi))
        times.append(times[-1] + config.dt)
        poses.append(next_pose)
        current_pose = next_pose
        current_omega = omega_cmd
    else:
        max_steps_reached = True

    if not goal_reached:
        goal_reached = position_goal_reached(current_pose, goal_pose, config.goal_tolerance_dist)

    return TrackingResult(
        scenario=scenario,
        condition=condition,
        variant=variant,
        times=np.asarray(times, dtype=float),
        poses=np.asarray(poses, dtype=float),
        lookahead=np.asarray(lookahead_points, dtype=float),
        curvatures=np.asarray(curvatures, dtype=float),
        omega_raw=np.asarray(omega_raw_values, dtype=float),
        omega_cmd=np.asarray(omega_cmd_values, dtype=float),
        v_cmd=np.asarray(v_cmd_values, dtype=float),
        sigma=np.asarray(sigmas, dtype=float),
        sigma_y=np.asarray(sigma_y_values, dtype=float),
        sigma_psi=np.asarray(sigma_psi_values, dtype=float),
        goal_reached=goal_reached,
        max_steps_reached=max_steps_reached,
    )


def compute_curvature(
    current_pose: np.ndarray,
    current_velocity: np.ndarray,
    current_idx: np.intp,
    path: np.ndarray,
    path_distances: np.ndarray,
    variant: MethodVariant,
    config: EcppConfig,
) -> tuple[float, np.ndarray, EcppTerms | None]:
    if variant.method == "pp":
        curvature, lookahead_pos = calc_pp_curvature(
            current_pose, current_idx, path, path_distances, variant.lookahead_m, config
        )
        return curvature, lookahead_pos, None
    if variant.method == "dpp":
        curvature, lookahead_pos = calc_dpp_curvature(current_pose, current_idx, path, path_distances, config)
        return curvature, lookahead_pos, None
    if variant.method == "ecpp":
        terms = calc_ecpp_terms(
            current_pose=current_pose,
            current_velocity=current_velocity,
            current_idx=current_idx,
            path=path,
            path_distances=path_distances,
            lookahead_distance=variant.lookahead_m,
            config=config,
        )
        return terms.curvature, terms.lookahead_pos, terms
    raise ValueError(f"unsupported method: {variant.method}")


def step_unicycle(pose: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    theta = float(pose[2])
    if abs(omega) <= 1e-12:
        delta = np.array([v * math.cos(theta), v * math.sin(theta), 0.0], dtype=float) * dt
    else:
        radius = v / omega
        delta = np.array([
            radius * (math.sin(theta + omega * dt) - math.sin(theta)),
            radius * (-math.cos(theta + omega * dt) + math.cos(theta)),
            omega * dt,
        ])
    next_pose = pose + delta
    next_pose[2] = float(normalize_angle(next_pose[2]))
    return next_pose


def position_goal_reached(pose: np.ndarray, goal_pose: np.ndarray, tolerance: float) -> bool:
    return float(np.linalg.norm(pose[:2] - goal_pose[:2])) <= tolerance

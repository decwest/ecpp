"""Shared simulation adapter for paper artifact generators.

The paper generators intentionally contain only experiment orchestration and
rendering.  Controller evaluation is delegated to the same continuous path
projection and PP/DPP/ECPP implementations used by the public simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import EcppConfig
from ..controllers.dpp import calc_dpp_curvature
from ..controllers.ecpp import calc_ecpp_terms
from ..controllers.gates import gate_abs
from ..controllers.pure_pursuit import calc_pp_curvature
from ..geometry import calc_path_distances, normalize_angle
from ..lookahead import calc_path_frame_error, project_to_path, sample_path_at_arc_length


@dataclass(frozen=True)
class PaperTrace:
    """One fixed-speed simulation trace.

    ``omega_raw`` is the controller request used by curvature, saturation, and
    smoothness metrics.  ``omega_cmd`` is the instantaneous clipped value used
    only by the unicycle state update, matching the paper simulation model.
    """

    time: np.ndarray
    pose: np.ndarray
    path_s: np.ndarray
    e_y: np.ndarray
    e_psi: np.ndarray
    curvature: np.ndarray
    omega_raw: np.ndarray
    omega_cmd: np.ndarray
    sigma: np.ndarray


def simulate_fixed_speed(
    *,
    path: np.ndarray,
    method: str,
    lookahead_m: float,
    omega_n: float,
    zeta: float,
    e_y0: float,
    e_psi0: float,
    speed: float = 0.5,
    v_epsilon: float = 0.05,
    omega_limit: float = 1.5,
    dt: float = 1.0 / 30.0,
    t_max: float = 60.0,
    goal_arc_length: float | None = None,
    goal_position: np.ndarray | None = None,
    goal_tolerance: float = 0.02,
) -> PaperTrace:
    """Run the canonical paper simulation through the package controllers."""

    path = np.asarray(path, dtype=float)
    path_distances = calc_path_distances(path)
    start = sample_path_at_arc_length(path, path_distances, 0.0)
    left = np.array([-start.tangent[1], start.tangent[0]], dtype=float)
    pose = np.array(
        [
            start.position[0] + e_y0 * left[0],
            start.position[1] + e_y0 * left[1],
            normalize_angle(start.heading + e_psi0),
        ],
        dtype=float,
    )
    projection = project_to_path(pose, path, path_distances)
    if not math.isclose(float(omega_limit), 1.5, abs_tol=1e-12):
        raise ValueError(
            "paper simulation state updates use a fixed +/-1.5 rad/s clip"
        )
    if goal_tolerance <= 0.0:
        raise ValueError("goal_tolerance must be positive")
    goal_s = path_distances[-1] if goal_arc_length is None else goal_arc_length
    goal_s = float(np.clip(goal_s, 0.0, path_distances[-1]))
    if goal_position is None:
        goal_position = sample_path_at_arc_length(
            path, path_distances, goal_s
        ).position
    goal_position = np.asarray(goal_position, dtype=float)[:2]

    gate_mode = "ey_only"
    normalized_method = method.strip().lower().replace("_", " ")
    if normalized_method in {"ecpp w/o gate", "ecpp without gate"}:
        gate_mode = "always_on"
    config = EcppConfig(
        lookahead_m=float(lookahead_m),
        v_max=float(speed),
        omega_max=float(omega_limit),
        dt=float(dt),
        ecpp_omega_n=float(omega_n),
        ecpp_zeta=float(zeta),
        ecpp_v_epsilon=float(v_epsilon),
        ecpp_gate_mode=gate_mode,
        dpp_omega_n=float(omega_n),
        dpp_zeta=float(zeta),
        dpp_gain_speed=float(speed),
    )

    time_values: list[float] = []
    pose_values: list[np.ndarray] = []
    path_s_values: list[float] = []
    e_y_values: list[float] = []
    e_psi_values: list[float] = []
    curvature_values: list[float] = []
    omega_raw_values: list[float] = []
    omega_cmd_values: list[float] = []
    sigma_values: list[float] = []

    max_steps = int(math.floor(float(t_max) / float(dt))) + 1
    for step in range(max_steps):
        e_y, e_psi = calc_path_frame_error(pose, projection)
        if normalized_method == "pp":
            curvature, _ = calc_pp_curvature(
                pose,
                projection,
                path,
                path_distances,
                lookahead_m,
                config,
            )
            sigma = gate_abs(
                (e_y / lookahead_m) ** 2,
                config.ecpp_gate_error_on,
                config.ecpp_gate_error_off,
                config.ecpp_gate_sigmoid_endpoint_value,
            )
        elif normalized_method == "dpp":
            curvature, _ = calc_dpp_curvature(
                pose, projection, path, path_distances, config
            )
            sigma = gate_abs(
                (e_y / lookahead_m) ** 2,
                config.ecpp_gate_error_on,
                config.ecpp_gate_error_off,
                config.ecpp_gate_sigmoid_endpoint_value,
            )
        elif normalized_method in {
            "ecpp",
            "ecpp w/o gate",
            "ecpp without gate",
        }:
            terms = calc_ecpp_terms(
                current_pose=pose,
                current_velocity=np.array([speed, 0.0], dtype=float),
                current_idx=projection,
                path=path,
                path_distances=path_distances,
                lookahead_distance=lookahead_m,
                config=config,
            )
            curvature = terms.curvature
            sigma = terms.sigma
        else:
            raise ValueError(f"unsupported paper simulation method: {method!r}")

        omega_raw = float(speed * curvature)
        omega_cmd = float(np.clip(omega_raw, -omega_limit, omega_limit))
        time_values.append(step * dt)
        pose_values.append(pose.copy())
        path_s_values.append(projection.arc_length)
        e_y_values.append(e_y)
        e_psi_values.append(e_psi)
        curvature_values.append(float(curvature))
        omega_raw_values.append(omega_raw)
        omega_cmd_values.append(omega_cmd)
        sigma_values.append(float(sigma))

        if (
            float(np.linalg.norm(pose[:2] - goal_position)) <= goal_tolerance
            and step > 0
        ):
            break
        if float(np.linalg.norm(pose[:2])) > 50.0:
            break

        heading = float(pose[2])
        if abs(omega_cmd) <= 1e-12:
            pose = pose + np.array(
                [speed * math.cos(heading), speed * math.sin(heading), 0.0],
                dtype=float,
            ) * dt
        else:
            radius = speed / omega_cmd
            pose = pose + np.array(
                [
                    radius
                    * (math.sin(heading + omega_cmd * dt) - math.sin(heading)),
                    radius
                    * (-math.cos(heading + omega_cmd * dt) + math.cos(heading)),
                    omega_cmd * dt,
                ],
                dtype=float,
            )
        pose[2] = float(normalize_angle(pose[2]))
        projection = project_to_path(
            pose,
            path,
            path_distances,
            minimum_arc_length=projection.arc_length,
        )

    return PaperTrace(
        time=np.asarray(time_values, dtype=float),
        pose=np.asarray(pose_values, dtype=float),
        path_s=np.asarray(path_s_values, dtype=float),
        e_y=np.asarray(e_y_values, dtype=float),
        e_psi=np.asarray(e_psi_values, dtype=float),
        curvature=np.asarray(curvature_values, dtype=float),
        omega_raw=np.asarray(omega_raw_values, dtype=float),
        omega_cmd=np.asarray(omega_cmd_values, dtype=float),
        sigma=np.asarray(sigma_values, dtype=float),
    )

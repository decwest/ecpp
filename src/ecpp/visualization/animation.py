from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch

from ..simulation.path_tracking import TrackingResult


def animate_result(
    result: TrackingResult,
    output_path: Path,
    fps: int = 20,
    frame_stride: int = 5,
    max_frames: int = 600,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = sample_frame_indices(len(result.poses), frame_stride, max_frames)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(result.scenario.path[:, 0], result.scenario.path[:, 1], "k--", linewidth=1.2, label="Reference")
    trail, = ax.plot([], [], color="#2f855a", linewidth=1.6, label=result.variant.label)
    point, = ax.plot([], [], "o", color="#2b6cb0", markersize=6)
    lookahead, = ax.plot([], [], "o", color="#c05621", markersize=5, label="Lookahead")
    heading_arrow = FancyArrowPatch((0.0, 0.0), (0.0, 0.0), mutation_scale=14, color="#2b6cb0")
    ax.add_patch(heading_arrow)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(fontsize=8)
    set_equal_limits(ax, result)
    arrow_length = max(0.08, 0.04 * float(np.ptp(result.scenario.path[:, 0]) + np.ptp(result.scenario.path[:, 1])))

    def init() -> tuple[object, ...]:
        trail.set_data([], [])
        point.set_data([], [])
        lookahead.set_data([], [])
        heading_arrow.set_visible(False)
        return trail, point, lookahead, heading_arrow

    def update(frame_number: int) -> tuple[object, ...]:
        pose_index = int(frame_indices[frame_number])
        pose = result.poses[pose_index]
        history = result.poses[: pose_index + 1]
        trail.set_data(history[:, 0], history[:, 1])
        point.set_data([pose[0]], [pose[1]])
        heading_arrow.set_positions(
            (pose[0], pose[1]),
            (pose[0] + arrow_length * np.cos(pose[2]), pose[1] + arrow_length * np.sin(pose[2])),
        )
        heading_arrow.set_visible(True)
        if len(result.lookahead) > 0:
            lookahead_index = min(max(pose_index - 1, 0), len(result.lookahead) - 1)
            target = result.lookahead[lookahead_index]
            lookahead.set_data([target[0]], [target[1]])
        return trail, point, lookahead, heading_arrow

    animation = FuncAnimation(fig, update, frames=len(frame_indices), init_func=init, blit=False, repeat=False)
    suffix = output_path.suffix.lower().lstrip(".") or "gif"
    if suffix == "gif":
        animation.save(output_path, writer=PillowWriter(fps=fps))
    elif suffix == "mp4":
        animation.save(output_path, writer=FFMpegWriter(fps=fps))
    else:
        raise ValueError("animation output suffix must be .gif or .mp4")
    plt.close(fig)
    return output_path


def sample_frame_indices(num_poses: int, frame_stride: int, max_frames: int) -> np.ndarray:
    indices = np.arange(0, num_poses, max(1, frame_stride), dtype=int)
    if len(indices) == 0 or indices[-1] != num_poses - 1:
        indices = np.append(indices, num_poses - 1)
    if len(indices) > max_frames:
        chosen = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = indices[chosen]
    return indices


def set_equal_limits(ax: plt.Axes, result: TrackingResult) -> None:
    xy = np.vstack([result.scenario.path[:, :2], result.poses[:, :2]])
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1e-3)
    pad = 0.08 * span
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")

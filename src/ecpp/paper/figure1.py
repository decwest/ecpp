"""Generate the preregistered PP/ECPP overview figure for the paper.

The figure is descriptive, not a parameter-selection experiment.  Its two
rows reuse exact arms from chapter 5: the fastest admitted speed-shaping arm
and the representative critically damped arm.  No post-hoc gain sweep is
performed here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import ieee_access as study
from .tracking import simulate_fixed_speed


ROOT = Path.cwd()
FIG_OUT = ROOT / "generated_preview" / "figures"
TABLE_OUT = ROOT / "generated_preview" / "tables"

V0 = study.V0
V_EPSILON = study.V_EPSILON
OMEGA_MAX = study.OMEGA_MAX
DT = study.DT
TMAX = 20.0
CONTROL_PATH_LENGTH = 8.0
EVALUATION_GOAL = 6.0

SCENARIOS = (
    {
        "key": "speed",
        "row_label": "Speed shaping",
        "lookahead": study.SPEED_LD,
        "ey0": study.SPEED_COND[0],
        "eth0": study.SPEED_COND[1],
        "omega_n": study.SPEED_OMEGA_N_MAX,
        "zeta": study.SPEED_ZETA,
        "design_label": r"$\omega_n=\omega_{n,\max}$, $\zeta=1/\sqrt{2}$",
    },
    {
        # Same condition and lookahead as the speed row; critically damped.
        # This is the hardware arm ECPP_L100_W1133_Z1000.
        "key": "damping",
        "row_label": "Damping shaping",
        "lookahead": study.SPEED_LD,
        "ey0": study.SPEED_COND[0],
        "eth0": study.SPEED_COND[1],
        "omega_n": study.SPEED_OMEGA_N_MAX,
        "zeta": 1.0,
        "design_label": r"$\omega_n=\omega_{n,\max}$, $\zeta=1$",
    },
)


def target_gains(omega_n: float, zeta: float) -> tuple[float, float]:
    """Return the paper's gains for ``omega_n`` defined at the nominal speed."""

    return (omega_n / abs(V0)) ** 2, 2.0 * zeta * omega_n / abs(V0)


def simulate(method: str, scenario: dict[str, object]) -> np.ndarray:
    """Return ``t,x,y,yaw,kappa,omega_raw,omega_cmd,sigma``."""

    trace = simulate_fixed_speed(
        path=np.array(
            [[0.0, 0.0], [CONTROL_PATH_LENGTH, 0.0]], dtype=float
        ),
        method=method,
        lookahead_m=float(scenario["lookahead"]),
        omega_n=study.configured_omega_n(float(scenario["omega_n"])),
        zeta=float(scenario["zeta"]),
        e_y0=float(scenario["ey0"]),
        e_psi0=float(scenario["eth0"]),
        speed=V0,
        v_epsilon=V_EPSILON,
        omega_limit=OMEGA_MAX,
        dt=DT,
        t_max=TMAX,
        goal_arc_length=EVALUATION_GOAL,
        goal_position=np.array([EVALUATION_GOAL, 0.0]),
    )
    return np.column_stack(
        [
            trace.time,
            trace.pose,
            trace.curvature,
            trace.omega_raw,
            trace.omega_cmd,
            trace.sigma,
        ]
    )


def zero_crossings(y_values: np.ndarray) -> int:
    deadband = 0.003
    last_sign = 0
    count = 0
    for value in y_values:
        sign = 0 if abs(value) < deadband else int(math.copysign(1, value))
        if sign == 0:
            continue
        if last_sign and sign != last_sign:
            count += 1
        last_sign = sign
    return count


def settling_time(time: np.ndarray, error: np.ndarray, fraction: float) -> float | None:
    band = fraction * abs(float(error[0]))
    violations = np.flatnonzero(np.abs(error) > band)
    if not len(violations):
        return 0.0
    if violations[-1] == len(error) - 1:
        return None
    return float(time[violations[-1] + 1] - time[0])


def overshoot(error: np.ndarray) -> float:
    initial_sign = math.copysign(1.0, float(error[0]))
    crossed = np.flatnonzero(initial_sign * error < 0.0)
    if not len(crossed):
        return 0.0
    return float(max(0.0, np.max(-initial_sign * error[crossed[0] :])))


def trace_metrics(trace: np.ndarray) -> dict[str, float | int | None]:
    """Metrics used in the figure annotations.

    Column 5 is the raw request, so clipping and smoothness diagnostics remain
    upstream of the instantaneous state-update clip.
    """

    time = trace[:, 0]
    error = trace[:, 2]
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0 on the paper-generation host
        integrate = np.trapz
    return {
        "settling_time_2pct_s": settling_time(time, error, 0.02),
        "overshoot_m": overshoot(error),
        "zero_crossings": zero_crossings(error),
        "iae_m_s": float(integrate(np.abs(error), time)),
        "omega_raw_max_rad_s": float(np.max(np.abs(trace[:, 5]))),
        "clip_ratio": float(np.mean(np.abs(trace[:, 5]) >= OMEGA_MAX)),
    }


def _format_metric(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _write_metadata(
    traces: dict[tuple[str, str], np.ndarray]
) -> tuple[Path, Path]:
    metadata = {
        "meta": {
            "selection": "none; exact preregistered chapter-5 arms",
            "v0_m_s": V0,
            "v_epsilon_m_s": V_EPSILON,
            "control_rate_hz": 1.0 / DT,
            "omega_state_update_limit_rad_s": OMEGA_MAX,
            "acceleration_model": None,
            "carrot_rule": "continuous_arc_length",
        },
        "scenarios": [],
    }
    for scenario in SCENARIOS:
        record = {
            "key": scenario["key"],
            "lookahead_m": scenario["lookahead"],
            "initial_condition": [
                scenario["ey0"], math.degrees(float(scenario["eth0"]))
            ],
            "ecpp": {
                "omega_n": scenario["omega_n"],
                "omega_n_configured": study.configured_omega_n(
                    float(scenario["omega_n"])
                ),
                "zeta": scenario["zeta"],
            },
            "metrics": {
                method: trace_metrics(traces[(str(scenario["key"]), method)])
                for method in ("PP", "ECPP")
            },
        }
        metadata["scenarios"].append(record)

    json_path = TABLE_OUT / "figure1_conditions.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    table_path = TABLE_OUT / "figure1_conditions.tex"
    lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Case & Method & $L_d$ & $\omega_n$ & $\zeta$ & $M_{\rm os}$ \\",
        r"\midrule",
    ]
    for scenario in SCENARIOS:
        for method in ("PP", "ECPP"):
            metrics = trace_metrics(traces[(str(scenario["key"]), method)])
            omega_text = "--" if method == "PP" else f"{scenario['omega_n']:.3f}"
            zeta_text = "--" if method == "PP" else f"{scenario['zeta']:.3f}"
            lines.append(
                " & ".join(
                    [
                        str(scenario["row_label"]),
                        method,
                        f"{scenario['lookahead']:.2f}",
                        omega_text,
                        zeta_text,
                        _format_metric(metrics["overshoot_m"], 3),
                    ]
                )
                + r"\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    table_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, table_path


def make_figure() -> tuple[Path, Path]:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    traces = {
        (str(scenario["key"]), method): simulate(method, scenario)
        for scenario in SCENARIOS
        for method in ("PP", "ECPP")
    }
    json_path, table_path = _write_metadata(traces)

    methods = (("PP", "Pure Pursuit", "#d62728"), ("ECPP", "ECPP", "#2ca02c"))
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.9), sharex=True, sharey=True)
    for column, (_, title, _) in enumerate(methods):
        axes[0, column].set_title(title, fontsize=12, fontweight="bold", pad=7)

    for row, scenario in enumerate(SCENARIOS):
        for column, (method, _, color) in enumerate(methods):
            ax = axes[row, column]
            trace = traces[(str(scenario["key"]), method)]
            metrics = trace_metrics(trace)
            ax.plot([0.0, EVALUATION_GOAL], [0.0, 0.0], "k--", lw=1.4)
            ax.plot(trace[:, 1], trace[:, 2], color=color, lw=2.3)

            heading = float(trace[0, 3])
            ax.annotate(
                "",
                xy=(
                    trace[0, 1] + 0.32 * math.cos(heading),
                    trace[0, 2] + 0.32 * math.sin(heading),
                ),
                xytext=(trace[0, 1], trace[0, 2]),
                arrowprops={"arrowstyle": "-|>", "color": "#305c8a", "lw": 1.8},
            )
            ax.plot(trace[0, 1], trace[0, 2], "o", ms=3.8, color="#305c8a")
            text = (
                rf"$T_s^{{2\%}}={_format_metric(metrics['settling_time_2pct_s'])}$ s"
                "\n"
                rf"$M_{{\rm os}}={_format_metric(metrics['overshoot_m'], 3)}$ m"
            )
            ax.text(
                0.97,
                0.08,
                text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.9},
            )
            if column == 0:
                condition = (
                    rf"$L_d={scenario['lookahead']:.1f}$ m, "
                    rf"$(e_y(0),e_\theta(0))="
                    rf"({scenario['ey0']:.2f}\,\mathrm{{m}},"
                    rf"{math.degrees(float(scenario['eth0'])):.0f}^\circ)$"
                )
                ax.set_ylabel(
                    f"{scenario['row_label']}\n{condition}\n$y=e_y$ [m]",
                    fontsize=8.0,
                )
            if row == 1:
                ax.set_xlabel(r"$x$ [m]")
            if method == "ECPP":
                ax.text(
                    0.03,
                    0.91,
                    str(scenario["design_label"]),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.5,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.85",
                        "alpha": 0.88,
                    },
                )
            ax.set_xlim(-0.05, EVALUATION_GOAL + 0.05)
            ax.set_ylim(-0.045, 0.33)
            ax.grid(True, color="0.9", lw=0.55)
            ax.tick_params(labelsize=8)

    fig.tight_layout(pad=0.6, w_pad=1.2, h_pad=1.0)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIG_OUT / f"figure1_pp_ecpp_matrix.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    return json_path, table_path


def configure_output_root(out_root: str | Path, *, apply: bool = False) -> None:
    """Set artifact directories without writing anything at import time."""

    global ROOT, FIG_OUT, TABLE_OUT
    ROOT = Path(out_root).resolve()
    subdirectory = "generated" if apply else "generated_preview"
    FIG_OUT = ROOT / subdirectory / "figures"
    TABLE_OUT = ROOT / subdirectory / "tables"


def main(
    argv: list[str] | None = None, *, default_output_root: Path | None = None
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=default_output_root or Path.cwd(),
        help="paper project root (default: current directory; generated_preview by default)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write to generated/ instead of preview"
    )
    args = parser.parse_args(argv)
    configure_output_root(args.out_root, apply=args.apply)
    json_path, table_path = make_figure()
    print("Generated preregistered Figure 1 (no post-hoc parameter selection)")
    print(f"Generated {FIG_OUT / 'figure1_pp_ecpp_matrix.png'}")
    print(f"Generated {FIG_OUT / 'figure1_pp_ecpp_matrix.pdf'}")
    print(f"Generated {json_path}")
    print(f"Generated {table_path}")


if __name__ == "__main__":
    main()

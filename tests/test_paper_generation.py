from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

from ecpp.paper import figure1, ieee_access
from ecpp.paper.tracking import simulate_fixed_speed


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = (
    REPO_ROOT.parent
    / "tex_docker_environment"
    / "projects"
    / "ECPP_ACCESS"
)


@pytest.mark.parametrize(
    "relative_script",
    (
        "generate_sim_evaluation_outputs.py",
        "generate_figure1_pp_ecpp_matrix.py",
        "generate_fixed_speed_pp_dpp_ecpp.py",
    ),
)
def test_paper_local_generators_are_thin_runnable_wrappers(relative_script):
    script = PAPER_ROOT / "scripts" / relative_script
    source_lines = script.read_text(encoding="utf-8").splitlines()
    assert len(source_lines) <= 20
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=PAPER_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "generated_preview" in completed.stdout


def test_canonical_cli_is_importable():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "ecpp.paper", "--help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "figure1" in completed.stdout
    assert "ieee-access" in completed.stdout


def test_canonical_ieee_cli_accepts_explicit_apply(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ecpp.paper",
            "ieee-access",
            "--out-root",
            str(tmp_path),
            "--apply",
            "--hw-reference",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "generated" / "tables" / "hw_exp1_sim_reference.json").is_file()


def test_paper_generators_use_additive_speed_regularization():
    ky, kpsi = figure1.target_gains(1.0, 1.0)
    gains = ieee_access.Gains(1.0, 1.0)
    assert gains.v_gain == pytest.approx(0.55)
    assert ky == pytest.approx(gains.Ky)
    assert kpsi == pytest.approx(gains.Kth)


def test_analytic_overlay_uses_gains_after_speed_regularization():
    configured_omega_n = 1.2
    time = np.array([0.0, 1.0])
    response = ieee_access.second_order_response(
        time, 1.0, configured_omega_n, 1.0
    )
    natural_from_gains = configured_omega_n * 0.5 / 0.55
    expected = np.exp(-natural_from_gains * time) * (
        1.0 + natural_from_gains * time
    )
    assert response == pytest.approx(expected)


def test_preview_is_the_default_output_location(tmp_path):
    figure1.configure_output_root(tmp_path)
    assert figure1.FIG_OUT == tmp_path / "generated_preview" / "figures"
    assert figure1.TABLE_OUT == tmp_path / "generated_preview" / "tables"
    assert not (tmp_path / "generated").exists()


def test_apply_output_location_is_explicit(tmp_path):
    figure1.configure_output_root(tmp_path, apply=True)
    assert figure1.FIG_OUT == tmp_path / "generated" / "figures"
    assert figure1.TABLE_OUT == tmp_path / "generated" / "tables"


def test_paper_tracking_uses_position_goal_tolerance():
    trace = simulate_fixed_speed(
        path=np.array([[0.0, 0.0], [2.0, 0.0]]),
        method="PP",
        lookahead_m=0.5,
        omega_n=1.0,
        zeta=1.0,
        e_y0=0.0,
        e_psi0=0.0,
        goal_arc_length=1.005,
        goal_position=np.array([1.005, 0.0]),
    )
    assert np.linalg.norm(trace.pose[-1, :2] - np.array([1.005, 0.0])) <= 0.02
    assert trace.path_s[-1] < 1.005


def test_paper_simulation_rejects_nonstandard_state_update_clip():
    with pytest.raises(ValueError, match=r"fixed \+/-1\.5"):
        simulate_fixed_speed(
            path=np.array([[0.0, 0.0], [1.0, 0.0]]),
            method="PP",
            lookahead_m=0.5,
            omega_n=1.0,
            zeta=1.0,
            e_y0=0.0,
            e_psi0=0.0,
            omega_limit=2.0,
        )


def test_paper_cli_rejects_nonstandard_state_update_clip(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ecpp.paper",
            "ieee-access",
            "--out-root",
            str(tmp_path),
            "--omega-max",
            "2.0",
            "--hw-reference",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "fixed at 1.5" in completed.stderr
    assert not (tmp_path / "generated_preview").exists()


def test_saturation_includes_raw_request_exactly_at_limit():
    trace = np.zeros((2, 8), dtype=float)
    trace[:, 0] = (0.0, 0.1)
    trace[:, 2] = (0.1, 0.05)
    trace[:, 5] = (1.5, -1.5)
    assert figure1.trace_metrics(trace)["clip_ratio"] == 1.0


def test_frozen_paper_experiment_conditions_are_exact():
    assert ieee_access.DT == pytest.approx(1.0 / 30.0)
    assert ieee_access.OMEGA_MAX == pytest.approx(1.5)
    assert ieee_access.SPEED_LD == pytest.approx(1.0)
    assert ieee_access.SPEED_COND == pytest.approx((0.30, 0.0))
    assert ieee_access.SPEED_ZETA == pytest.approx(1.0 / np.sqrt(2.0))
    assert ieee_access.SPEED_PP_OMEGA_N == pytest.approx(
        np.sqrt(2.0) * 0.55
    )
    assert ieee_access.DAMPING_LD == pytest.approx(0.5)
    assert ieee_access.DAMPING_OMEGA_N == pytest.approx(
        np.sqrt(2.0) * 0.55 / 0.5
    )
    # Frozen hardware experiment-1 arms (redesigned 2026-07-16, L_d=1.0).
    local_arms = ieee_access._hw_local_arms()
    far_arms = ieee_access._hw_far_arms()
    assert [arm["key"] for arm in local_arms] == [
        "PP",
        "ECPP_w0778_z0707", "ECPP_w0778_z1000", "ECPP_w0778_z1414",
        "ECPP_w1133_z0707", "ECPP_w1133_z1000", "ECPP_w1133_z1414",
    ]
    assert [arm["key"] for arm in far_arms] == [
        "PP", "ECPP_w1133_z0707", "ECPP_w1133_z1000", "ECPP_w1133_z1414",
    ]
    assert local_arms[1]["omega_n"] == pytest.approx(
        ieee_access.SPEED_PP_OMEGA_N
    )
    assert local_arms[4]["omega_n"] == pytest.approx(
        ieee_access.SPEED_OMEGA_N_MAX
    )
    assert far_arms[1]["omega_n"] == pytest.approx(
        ieee_access.SPEED_OMEGA_N_MAX
    )
    assert ieee_access.GRID_LDS == pytest.approx((1.0, 0.5))
    assert ieee_access.GRID_COND == pytest.approx((0.15, 0.0))
    assert ieee_access.GRID_ZETAS == pytest.approx(
        (1.0 / np.sqrt(2.0), 1.0, np.sqrt(2.0))
    )
    assert ieee_access.GRID_OMEGAS == pytest.approx((
        ieee_access.SPEED_PP_OMEGA_N,
        ieee_access.SPEED_OMEGA_N_MAX,
        ieee_access.DAMPING_OMEGA_N,
    ))
    assert [(ey, round(np.rad2deg(epsi)))
            for ey, epsi in ieee_access.TEST2_CONDS] == [
        (0.0, -30), (0.0, -90),
        (0.15, 0), (0.15, -30), (0.15, -90),
        (1.0, 0), (1.0, -30), (1.0, -90),
    ]


def test_figure1_reuses_preregistered_chapter5_arms_without_gain_search():
    assert not hasattr(figure1, "OMEGA_N_CANDIDATES")
    assert not hasattr(figure1, "evaluate_candidates")
    speed, damping = figure1.SCENARIOS
    assert speed["key"] == "speed"
    assert speed["lookahead"] == pytest.approx(ieee_access.SPEED_LD)
    assert speed["ey0"] == pytest.approx(ieee_access.SPEED_COND[0])
    assert speed["eth0"] == pytest.approx(ieee_access.SPEED_COND[1])
    assert speed["omega_n"] == pytest.approx(ieee_access.SPEED_OMEGA_N_MAX)
    assert speed["zeta"] == pytest.approx(ieee_access.SPEED_ZETA)
    assert damping["key"] == "damping"
    assert damping["lookahead"] == pytest.approx(ieee_access.SPEED_LD)
    assert damping["ey0"] == pytest.approx(ieee_access.SPEED_COND[0])
    assert damping["eth0"] == pytest.approx(ieee_access.SPEED_COND[1])
    assert damping["omega_n"] == pytest.approx(ieee_access.SPEED_OMEGA_N_MAX)
    assert damping["zeta"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("ld", "ey0", "epsi0"),
    ((1.0, 0.30, 0.0), (0.5, 0.15, np.deg2rad(-30.0))),
)
def test_configured_pp_equivalent_ecpp_matches_pp(ld, ey0, epsi0):
    path = np.array([[0.0, 0.0], [8.0, 0.0]])
    omega_n = np.sqrt(2.0) * 0.55 / ld
    common = dict(
        path=path,
        lookahead_m=ld,
        omega_n=omega_n,
        zeta=1.0 / np.sqrt(2.0),
        e_y0=ey0,
        e_psi0=epsi0,
        speed=0.5,
        v_epsilon=0.05,
        omega_limit=1.5,
        dt=1.0 / 30.0,
        t_max=20.0,
        goal_arc_length=6.0,
        goal_position=np.array([6.0, 0.0]),
    )
    pp = simulate_fixed_speed(method="PP", **common)
    ecpp = simulate_fixed_speed(method="ECPP", **common)
    np.testing.assert_allclose(ecpp.pose, pp.pose, atol=2e-12, rtol=0.0)
    np.testing.assert_allclose(ecpp.curvature, pp.curvature, atol=2e-12, rtol=0.0)

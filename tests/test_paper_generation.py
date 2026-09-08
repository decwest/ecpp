from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

from ecpp.paper import figure1, ieee_access, test3_preview
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
        "generate_sim_test3_outputs.py",
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
    assert "test3-preview" in completed.stdout


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
    assert ieee_access.GRID_LD_SHORT == pytest.approx(0.5)
    assert ieee_access.GRID_PP_OMEGA_N_LD05 == pytest.approx(
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
        ieee_access.GRID_PP_OMEGA_N_LD05,
    ))
    assert not hasattr(ieee_access, "_select_grid_cell")
    # Test 2 (redesigned 2026-09-08): the hardware design point, the paper's
    # DPP construction, a right-turning arc, and the far conditions that
    # expose the reachability limit of the ungated linear laws.
    assert ieee_access.TEST2_LD == pytest.approx(1.0)
    assert ieee_access.TEST2_OMEGA_N == pytest.approx(ieee_access.SPEED_OMEGA_N_MAX)
    assert ieee_access.TEST2_ZETA == pytest.approx(1.0)
    assert ieee_access.DPP_FAR_FACTOR == pytest.approx(2.0)
    assert ieee_access.ARC_TURN == pytest.approx(-1.0)
    assert ieee_access.ARC_GOAL == pytest.approx(3.0 * 3.0 * np.pi / 4.0)
    assert [(ey, round(np.rad2deg(epsi)))
            for ey, epsi in ieee_access.TEST2_CONDS] == [
        (0.0, -90),
        (0.3, 0), (0.3, -90),
        (2.0, 0), (2.0, -90),
        (3.0, 0), (3.0, -90),
    ]
    l1, l2, a1, a2, v_gain = ieee_access.dpp_parameters(
        ieee_access.TEST2_OMEGA_N, ieee_access.TEST2_ZETA, ld=ieee_access.TEST2_LD
    )
    assert v_gain == pytest.approx(0.55)
    assert l1 == pytest.approx(2.0)
    assert l2 == pytest.approx(1.4286, abs=2e-3)
    assert a1 * l1**2 + a2 * l2**2 == pytest.approx(2.0)


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


def test_frozen_test3_conditions_are_exact():
    assert test3_preview.V0 == pytest.approx(0.5)
    assert test3_preview.DT == pytest.approx(1.0 / 30.0)
    assert test3_preview.OMEGA_MAX == pytest.approx(1.5)
    assert test3_preview.OMEGA_N == pytest.approx(ieee_access.SPEED_OMEGA_N_MAX)
    assert test3_preview.OMEGA_N == pytest.approx(ieee_access.TEST2_OMEGA_N)
    assert test3_preview.ZETA == pytest.approx(1.0)
    assert test3_preview.COND == pytest.approx((0.0, 0.0))
    assert test3_preview.LOOKAHEADS == pytest.approx((0.5, 1.0))
    assert test3_preview.METHODS == ("PP", "ECPP")
    assert test3_preview.ARMS == (("PP", 0.5), ("PP", 1.0), ("ECPP", 0.5), ("ECPP", 1.0))
    assert test3_preview.LEAD_IN == pytest.approx(3.0)
    assert test3_preview.ARC_R == pytest.approx(3.0)
    assert test3_preview.ARC_ANGLE == pytest.approx(np.pi / 2.0)
    assert test3_preview.EXIT_CONTROL == pytest.approx(6.0)
    assert test3_preview.EXIT_EVAL == pytest.approx(4.0)
    assert test3_preview.RECOVERY_FRACTION == pytest.approx(0.1)
    assert test3_preview.ENTRY_MARGIN == pytest.approx(0.5)
    assert test3_preview.EXIT_MARGIN == pytest.approx(0.5)
    assert test3_preview.LEAD_SEARCH_START == pytest.approx(1.0)
    path = test3_preview.StraightArcStraightPath()
    assert path.s_entry == pytest.approx(3.0)
    assert path.s_exit == pytest.approx(3.0 + 1.5 * np.pi)
    assert path.goal == pytest.approx(path.s_exit + 4.0)
    assert path.length == pytest.approx(path.s_exit + 6.0)
    np.testing.assert_allclose(path.point(path.s_exit), [6.0, 3.0], atol=1e-12)
    assert path.heading(path.s_exit) == pytest.approx(np.pi / 2.0)
    np.testing.assert_allclose(path.point(path.goal), [6.0, 7.0], atol=1e-12)
    # The zero-error feedforward is exactly the arc curvature once the carrot
    # and the robot are both on the arc, and zero on the lead-in straight.
    assert path.feedforward_curvature(4.5, 1.0) == pytest.approx(1.0 / 3.0)
    assert path.feedforward_curvature(1.0, 1.0) == pytest.approx(0.0)
    assert path.feedforward_curvature(2.5, 1.0) > 0.0


def test_test3_preview_follows_ld_and_recovery_follows_the_gains():
    _, _, metrics = test3_preview.simulate_arms()
    assert [m["key"] for m in metrics] == [
        "PP_ld0p50", "PP_ld1p00", "ECPP_ld0p50", "ECPP_ld1p00",
    ]
    by_key = {m["key"]: m for m in metrics}
    for m in metrics:
        assert m["sat_ratio"] == 0.0
        assert m["goal_reached"]
        assert m["d_lead"] is not None
        assert m["T_rec_in"] is not None and m["T_rec_out"] is not None
        assert m["e_y_in"] > 0.0 > m["e_y_out"]
    # Preview: the steering lead follows L_d for both methods (about 0.6 L_d).
    for method in ("PP", "ECPP"):
        short, long = by_key[f"{method}_ld0p50"], by_key[f"{method}_ld1p00"]
        assert long["d_lead"] > 1.5 * short["d_lead"]
        assert long["e_y_in"] > short["e_y_in"]
        assert long["e_y_out"] < short["e_y_out"]
    # Local response: PP's exit recovery scales with L_d, ECPP's does not.
    pp_ratio = by_key["PP_ld1p00"]["T_rec_out"] / by_key["PP_ld0p50"]["T_rec_out"]
    ecpp_ratio = by_key["ECPP_ld1p00"]["T_rec_out"] / by_key["ECPP_ld0p50"]["T_rec_out"]
    assert pp_ratio > 1.8
    assert abs(ecpp_ratio - 1.0) < 0.15


def test_test3_cli_preview_and_apply(tmp_path):
    test3_preview.main(["--out-root", str(tmp_path)])
    preview = tmp_path / "generated_preview"
    assert (preview / "tables" / "sim_test3_results.tex").is_file()
    assert (preview / "tables" / "sim_test3_metrics.json").is_file()
    assert (preview / "figures" / "sim_test3_preview_decoupling.pdf").is_file()
    assert not (tmp_path / "generated").exists()
    test3_preview.main(["--out-root", str(tmp_path), "--apply"])
    applied = tmp_path / "generated"
    assert (applied / "tables" / "sim_test3_results.tex").is_file()
    assert (applied / "figures" / "sim_test3_preview_decoupling.pdf").is_file()
    assert (applied / "traces" / "sim_test3_PP_ld0p50.csv").is_file()
    table = (applied / "tables" / "sim_test3_results.tex").read_text()
    assert "\\bottomrule" in table and table.count("\nPP &") == 2 and table.count("\nECPP &") == 2
    assert "$T_{\\rm rec,in}$" in table and "$T_{\\rm rec,out}$" in table
    with pytest.raises(SystemExit):
        test3_preview.main(["--out-root", str(tmp_path), "--apply", "--tag", "x"])


def test_ungated_linear_laws_stay_saturated_far_from_the_path():
    """Beyond the reachability limit, DPP and ungated ECPP circle forever.

    At the test-2 design point the ungated lateral term keeps
    max_theta omega_raw below -omega_max once e_y exceeds about 1.7 m (DPP)
    and 2.4 m (ECPP w/o gate); the gated ECPP falls back to PP and converges.
    """
    ieee_access.configure(ld=ieee_access.TEST2_LD)
    path = ieee_access.StraightPath()
    results = {}
    for method in ("DPP", "ECPP w/o gate", "ECPP"):
        m, _, _ = ieee_access.run_metrics(
            path, method, ieee_access.TEST2_OMEGA_N, ieee_access.TEST2_ZETA, 3.0, 0.0
        )
        results[method] = m
    assert results["DPP"]["evaluation_completed"] is False
    assert results["ECPP w/o gate"]["evaluation_completed"] is False
    assert results["DPP"]["sat_ratio"] == pytest.approx(1.0)
    assert results["ECPP w/o gate"]["sat_ratio"] == pytest.approx(1.0)
    assert results["ECPP"]["evaluation_completed"] is True
    assert results["ECPP"]["sat_ratio"] == 0.0
    assert results["ECPP"]["T_s"] is not None

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ecpp.paper import ieee_access
from ecpp.paper import test3_phase_plane as t3

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_CXX = shutil.which("g++") is not None or shutil.which("c++") is not None


def test_frozen_configurations():
    configs = t3.configurations()
    assert len(configs) == 22
    names = [c.name for c in configs]
    assert names[:4] == ["pp_ld1", "dpp_baseline", "ungated_baseline", "ecpp_ld1_w1_z1"]
    assert names[-1] == "pp_ld0.5"
    assert sum(c.method == 3 for c in configs) == 18
    base = configs[3]
    assert base.wn == pytest.approx(ieee_access.SPEED_OMEGA_N_MAX)
    assert base.wn == pytest.approx(ieee_access.TEST4_OMEGA_N)
    assert (base.ny, base.nth, base.horizon, base.dt) == (201, 360, 120.0, pytest.approx(1 / 30))
    assert base.tsv().split("\t")[0] == "ecpp_ld1_w1_z1"


def test_gate_matches_the_paper_gate():
    for y in np.linspace(-3.0, 3.0, 25):
        for ld in (0.5, 1.0):
            assert float(t3.gate(y, ld)) == pytest.approx(ieee_access.sigmoid_desc((y / ld) ** 2), abs=1e-12)


def test_analytic_commands_match_the_package_controllers():
    result = t3.validate_core(states_per_setting=20)
    assert max(result["core_command_max_abs_error_rad_s"].values()) < 1e-8


@pytest.mark.skipif(not HAS_CXX, reason="needs a C++ compiler")
def test_kernel_matches_the_python_trace(tmp_path):
    exe = tmp_path / "kernel"
    t3.compile_kernel(exe)
    configs = t3.reduced(t3.configurations()[:4], horizon=20.0)
    points = np.array([[0.3, 0.0], [3.0, -90.0], [-2.0, 150.0], [0.0, 30.0]])
    pts = tmp_path / "points.tsv"
    np.savetxt(pts, points, fmt="%.17g", delimiter="\t")
    grid = tmp_path / "grid.tsv"
    grid.write_text("\n".join(c.tsv() for c in configs) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    t3.run_kernel(exe, grid, out, threads=2, points_tsv=pts)
    for c in configs:
        rec = np.fromfile(out / f"{c.name}.bin", dtype="<f4").reshape(-1, len(t3.FIELDS))
        for (y0, th0), r in zip(points, rec):
            a = t3.trace(c, y0, th0)
            assert r[5] == pytest.approx(a[-1, 2], abs=2e-4)      # final e_y
            assert np.cos(r[6]) == pytest.approx(np.cos(a[-1, 3]), abs=2e-4)
            assert r[11] == pytest.approx(np.abs(a[:, 4]).max(), rel=1e-4, abs=1e-4)


@pytest.mark.skipif(not HAS_CXX, reason="needs a C++ compiler")
def test_test3_cli_smoke(tmp_path):
    t3.main(["--out-root", str(tmp_path), "--ny", "5", "--nth", "8", "--horizon", "20",
             "--threads", "2"])
    preview = tmp_path / "generated_preview"
    sweep = preview / t3.SWEEP_DIRNAME
    assert len(list((sweep / "data").glob("*.npz"))) == 22
    assert (sweep / "summary.csv").is_file() and (sweep / "core_validation.json").is_file()
    assert (sweep / "horizon_extension.csv").is_file()
    assert (preview / "tables" / "sim_test3_results.tex").is_file()
    assert (preview / "figures" / "sim_test3_phase_planes_row.pdf").is_file()
    assert (preview / "figures" / "sim_test3_phase_plane_ecpp.pdf").is_file()
    summary = json.loads((preview / "tables" / "sim_test3_summary.json").read_text())
    assert summary["per_controller"]["pp_ld1"]["grid_points"] == 40
    assert set(summary["horizon_extension"]) == {c.name for c in t3.configurations()}
    # a complete sweep directory is reused without recomputing
    meta_before = (sweep / "metadata.json").read_text()
    t3.main(["--out-root", str(tmp_path), "--ny", "5", "--nth", "8", "--horizon", "20"])
    assert (sweep / "metadata.json").read_text() == meta_before
    with pytest.raises(SystemExit):
        t3.main(["--out-root", str(tmp_path), "--apply", "--ny", "3"])

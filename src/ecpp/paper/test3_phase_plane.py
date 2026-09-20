"""Test-3 generator (paper numbering): phase-plane sweep of initial conditions.

For PP, DPP, ungated ECPP, and ECPP on an infinite straight path, every
initial condition on the grid e_y(0) in [-10, 10] m (0.1 m steps) x
e_theta(0) in [-180, 179] deg (1 deg steps) is simulated for 120 s, and the
settling time (|e_y| <= 0.02 m and |e_theta| <= 1 deg, held until the end of
the horizon and for at least 5 s) is recorded.  ECPP is swept over the 18
(omega_n, zeta, L_d) settings of Test 1 as well.  The unsettled points are
then re-simulated for 600 s (and, for PP and ECPP, 3600 s).

The sweep runs in a small C++ kernel (``phase_plane_kernel.cpp``, compiled
with ``g++ -O3 -std=c++17 -fopenmp`` on first use) because it integrates
22 x 72,360 trajectories.  The kernel implements the same control laws as the
``ecpp`` package; :func:`validate_core` cross-checks the analytic commands
against the package controllers on random states before every sweep, and
:func:`trace` is an independent pure-Python evaluator used by the tests.

Outputs (paper numbering, below ``generated_preview/`` or ``generated/``):

* ``sim_test3_sweep/``            raw sweep (``data/*.npz``, ``summary.csv``,
                                  ``horizon_extension.csv``, metadata)
* ``tables/sim_test3_results.tex``  settled counts per controller
* ``tables/sim_test3_summary.json`` every number quoted in the text
* ``figures/sim_test3_phase_planes_row.{pdf,png}`` and the per-panel files

A sweep directory that already holds ``data/*.npz`` for every configuration
is reused as is, so the tables and figures can be re-rendered without
recomputing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.resources
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import ieee_access as ia

plt.rcParams.update(ia.PAPER_RCPARAMS)

V = ia.V0                      # 0.5 m/s
LIMIT = ia.OMEGA_MAX           # 1.5 rad/s instantaneous clip
DT = ia.DT                     # 30 Hz
WN_REF = ia.SPEED_OMEGA_N_MAX  # 1.029884 rad/s, the reference design value at L_d = 1 m
# Gate of the paper: sigma = 1 / (1 + exp(k (eps_y - eps_c))) with eps_y = (e_y/L_d)^2,
# eps_on = 0.1, eps_off = 0.5, residual 0.01 -> eps_c = 0.3, k = 2 ln 99 / 0.4.
GATE_CENTER = 0.5 * (ia.EPS_ON + ia.EPS_OFF)
GATE_SLOPE = 2.0 * math.log((1.0 - ia.GATE_P) / ia.GATE_P) / (ia.EPS_OFF - ia.EPS_ON)
# The kernel hard-codes the same constants; keep the two in step.
assert math.isclose(V, 0.5) and math.isclose(LIMIT, 1.5) and math.isclose(DT, 1.0 / 30.0)
assert math.isclose(GATE_CENTER, 0.3) and math.isclose(GATE_SLOPE, 2.0 * math.log(99.0) / 0.4)
assert math.isclose(WN_REF, math.sqrt(V * LIMIT / math.sqrt(0.5)))

FIELDS = ["settle_30_s", "settle_60_s", "settle_end_s", "saturation_fraction",
          "max_abs_ey_m", "final_ey_m", "final_heading_rad", "tail_min_ey_m",
          "tail_max_ey_m", "tail_saturation_fraction", "net_turns", "peak_raw_rad_s",
          "exact_axis_equilibrium", "last_outside_plus_dt_s"]
METHOD_NAMES = {0: "PP", 1: "DPP", 2: "ECPP w/o gate", 3: "ECPP"}
BASELINES = ("pp_ld1", "dpp_baseline", "ungated_baseline", "ecpp_ld1_w1_z1")
OPERATING = {"pp_ld1": ("PP", 1.0, "--"), "dpp_baseline": ("DPP", 1.0, "(1.030, 1)"),
             "ungated_baseline": ("ECPP w/o gate", 1.0, "(1.030, 1)"),
             "ecpp_ld1_w1_z1": ("ECPP", 1.0, "(1.030, 1)")}
ETA_STAR_DEG = math.degrees(math.pi - math.atan(1.0 / (2.0 * math.sqrt(2.0))))
SETTLING = {"abs_ey_m": 0.02, "abs_heading_deg": 1.0,
            "remain_inside_until_evaluation_end": True,
            "minimum_time_inside_s": 5.0, "observation_times_s": [30, 60, 120]}
MODEL = ("Infinite straight path; true closest projection; arc-length lookahead. "
         "Exact ZOH unicycle at 30 Hz. No endpoint, progress clamp, smoothing, "
         "acceleration limit, noise or disturbances.")
UNSETTLED_COLOR = "#d9d9d9"
KERNEL_SOURCE = importlib.resources.files(__package__) / "phase_plane_kernel.cpp"
SWEEP_DIRNAME = "sim_test3_sweep"


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """One sweep configuration; the field order is the kernel's TSV order."""
    name: str
    method: int          # 0 PP, 1 DPP, 2 ungated ECPP, 3 ECPP
    ld: float
    wn: float
    zeta: float
    dt: float = DT
    horizon: float = 120.0
    ymin: float = -10.0
    dy: float = 0.1
    ny: int = 201
    thmin: float = -180.0
    dth: float = 1.0
    nth: int = 360

    def tsv(self):
        return "\t".join(str(v) for v in asdict(self).values())


def configurations():
    """The 22 frozen configurations: four baselines at L_d = 1 m, the other
    17 ECPP settings of the Test-1 grid, and PP at L_d = 0.5 m."""
    result = [Config("pp_ld1", 0, 1, WN_REF, 1),
              Config("dpp_baseline", 1, 1, WN_REF, 1),
              Config("ungated_baseline", 2, 1, WN_REF, 1),
              Config("ecpp_ld1_w1_z1", 3, 1, WN_REF, 1)]
    for ld in (1.0, 0.5):
        for wi, wn in enumerate((math.sqrt(0.5), WN_REF, math.sqrt(2))):
            for zi, zeta in enumerate((1 / math.sqrt(2), 1, math.sqrt(2))):
                name = f"ecpp_ld{ld:g}_w{wi}_z{zi}"
                if name != "ecpp_ld1_w1_z1":
                    result.append(Config(name, 3, ld, wn, zeta))
    result.append(Config("pp_ld0.5", 0, .5, WN_REF, 1))
    return result


def reduced(configs, ny=None, nth=None, horizon=None):
    """Coarser grid / shorter horizon for smoke tests (same extents)."""
    out = []
    for c in configs:
        kw = {}
        if ny is not None:
            kw.update(ny=ny, dy=20.0 / (ny - 1))
        if nth is not None:
            kw.update(nth=nth, dth=360.0 / nth)
        if horizon is not None:
            kw.update(horizon=float(horizon))
        out.append(replace(c, **kw))
    return out


# ---------------------------------------------------------------------------
# Pure-Python evaluator (tests and kernel cross-check)
# ---------------------------------------------------------------------------
def gate(y, ld):
    """The paper's gate as the kernel computes it (overflow-safe sigmoid)."""
    a = GATE_SLOPE * ((np.asarray(y, dtype=float) / ld) ** 2 - GATE_CENTER)
    e = np.exp(-np.abs(a))          # never overflows
    return np.where(a >= 0, e / (1 + e), 1 / (1 + e))


def raw_command(y, th, c):
    """Unclipped angular-velocity command v0 * kappa of configuration ``c``."""
    ky, kth = (c.wn / V) ** 2, 2 * c.zeta * c.wn / V
    if c.method == 1:
        return -V * (ky * y + kth * np.sin(th))
    pp = -2 * (y * np.cos(th) + c.ld * np.sin(th)) / (c.ld * c.ld + y * y)
    sig = gate(y, c.ld) if c.method == 3 else (1 if c.method == 2 else 0)
    return V * (pp - sig * ((ky - 2 / c.ld ** 2) * y + (kth - 2 / c.ld) * np.sin(th)))


def trace(c, y0, theta_deg, horizon=None):
    """Independent Python integration (exact zero-order-hold unicycle update).

    Returns an array of rows ``[t, x, e_y, e_theta, omega_raw, omega]``."""
    horizon = c.horizon if horizon is None else horizon
    n = round(horizon / c.dt)
    a = np.zeros((n + 1, 6))
    a[0, 2:4] = y0, math.radians(theta_deg)
    eq = y0 == 0 and theta_deg in (-180, 0, 180)
    for i in range(n + 1):
        t, x, y, th = a[i, :4]
        raw = 0.0 if eq else float(raw_command(y, th, c))
        omega = float(np.clip(raw, -LIMIT, LIMIT))
        a[i, 4:] = raw, omega
        if i == n:
            break
        h = omega * c.dt / 2
        scale = V * c.dt * np.sinc(h / np.pi)
        yy = y if eq else y + scale * np.sin(th + h)
        a[i + 1, :4] = ((i + 1) * c.dt, x + scale * np.cos(th + h), yy,
                        (th + 2 * h + np.pi) % (2 * np.pi) - np.pi)
    return a


def validate_core(configs=None, states_per_setting=80, seed=20260914):
    """Compare the analytic commands with the ``ecpp`` package controllers."""
    from ..config import EcppConfig
    from ..controllers.dpp import calc_dpp_curvature, calc_dpp_parameters
    from ..controllers.ecpp import calc_ecpp_terms
    from ..controllers.pure_pursuit import calc_pp_curvature_to_point
    from ..lookahead import project_to_path

    configs = configurations() if configs is None else configs
    assert (ia.EPS_ON, ia.EPS_OFF, ia.GATE_P) == (.1, .5, .01)
    path = np.array([[-1000., 0., 0.], [1000., 0., 0.]])
    distances = np.array([0., 2000.])
    rng = np.random.default_rng(seed)
    errors = {}
    dpp_parameters = None
    for c in configs:
        cfg = EcppConfig(lookahead_m=c.ld, ecpp_omega_n=c.wn, ecpp_zeta=c.zeta,
                         ecpp_v_epsilon=0.,
                         ecpp_gate_mode="always_on" if c.method == 2 else "ey_only",
                         ecpp_gate_error_on=ia.EPS_ON, ecpp_gate_error_off=ia.EPS_OFF,
                         ecpp_gate_sigmoid_endpoint_value=ia.GATE_P,
                         dpp_omega_n=c.wn, dpp_zeta=c.zeta, dpp_gain_speed=V,
                         dpp_far_factor=2.)
        peak = 0.
        for y, theta in zip(rng.uniform(-10, 10, states_per_setting),
                            rng.uniform(-np.pi, np.pi, states_per_setting)):
            pose = np.array([0., y, theta])
            projection = project_to_path(pose, path, distances)
            if c.method == 0:
                curvature = calc_pp_curvature_to_point(pose, np.array([c.ld, 0.]))
            elif c.method == 1:
                curvature = calc_dpp_curvature(pose, projection, path, distances, cfg)[0]
                dpp_parameters = calc_dpp_parameters(cfg)
            else:
                curvature = calc_ecpp_terms(pose, np.array([V, 0.]), projection,
                                            path, distances, c.ld, cfg).curvature
            peak = max(peak, abs(V * curvature - float(raw_command(y, theta, c))))
        errors[c.name] = peak
    gate_dev = max(abs(float(gate(y, 1.0)) - ia.sigmoid_desc(y * y)) for y in np.linspace(-3, 3, 61))
    assert max(errors.values()) < 1e-8, errors
    assert gate_dev < 1e-12, gate_dev
    return {"random_seed": seed, "states_per_setting": states_per_setting,
            "core_command_max_abs_error_rad_s": errors,
            "gate_max_abs_error": gate_dev,
            "dpp_L1_L2_a1_a2_vgain": dpp_parameters,
            "note": "Core has a 1e-12 speed floor even with epsilon_v=0; analytic paper gains do not."}


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def compile_kernel(exe):
    """Compile the kernel next to the sweep; fall back to a serial build when
    the compiler has no OpenMP (Apple clang)."""
    cxx = os.environ.get("CXX", "g++")
    if shutil.which(cxx) is None:
        raise RuntimeError(f"Test 3 needs a C++ compiler ({cxx!r} not found); "
                           "install g++ or set CXX")
    src = Path(str(KERNEL_SOURCE))
    base = [cxx, "-O3", "-std=c++17", src, "-o", exe]
    try:
        _run(base[:4] + ["-fopenmp"] + base[4:], capture_output=True, text=True)
        threaded = True
    except subprocess.CalledProcessError:
        _run(base, capture_output=True, text=True)
        threaded = False
        print("warning: compiled without OpenMP; the sweep runs single-threaded", flush=True)
    version = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    return threaded, (version[0] if version else cxx)


def run_kernel(exe, grid_tsv, out_dir, threads, points_tsv=None):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    args = [exe, grid_tsv, out_dir] + ([points_tsv] if points_tsv is not None else [])
    _run(args, env=env)


def _npz(path_bin):
    return path_bin.with_suffix(".npz")


def _package(path_bin, shape):
    """Convert one raw kernel output to a compressed ``.npz`` and delete it."""
    records = np.fromfile(path_bin, dtype="<f4").reshape(shape)
    np.savez_compressed(_npz(path_bin), records=records)
    assert np.array_equal(records, np.load(_npz(path_bin))["records"])
    path_bin.unlink()
    return records


def load_records(sweep_dir, c, folder="data"):
    p = sweep_dir / folder / f"{c.name}.bin"
    if p.exists():
        return np.fromfile(p, dtype="<f4").reshape(c.ny, c.nth, len(FIELDS))
    return np.load(_npz(p))["records"]


def sweep_complete(sweep_dir, configs):
    return all(_npz(sweep_dir / "data" / f"{c.name}.bin").exists() for c in configs)


# ---------------------------------------------------------------------------
# Sweep, summary, horizon extension
# ---------------------------------------------------------------------------
def summarize(configs, arrays):
    wide = np.pi - np.arctan(1 / (2 * np.sqrt(2)))
    rows = []
    for c in configs:
        r = arrays[c.name]
        y = c.ymin + np.arange(c.ny) * c.dy
        th = np.deg2rad(c.thmin + np.arange(c.nth) * c.dth)
        eta = (th[None, :] + np.arctan(y[:, None] / c.ld) + np.pi) % (2 * np.pi) - np.pi
        front = abs(eta) < np.pi / 2 - 1e-12
        star = abs(eta) < wide - 1e-12
        ok = r[:, :, 2] >= 0
        settled = r[:, :, 2][ok]
        rows.append({"name": c.name, "method": METHOD_NAMES[c.method], "Ld_m": c.ld,
                     "omega_n_rad_s": c.wn, "zeta": c.zeta, "grid_points": int(ok.size),
                     "settled_by_30s": int(np.sum(r[:, :, 0] >= 0)),
                     "settled_by_60s": int(np.sum(r[:, :, 1] >= 0)),
                     "settled_by_120s": int(ok.sum()), "unsettled_at_120s": int((~ok).sum()),
                     "settled_percent_120s": float(ok.mean() * 100),
                     "conditional_median_Ts_s": float(np.median(settled)) if ok.any() else None,
                     "conditional_max_Ts_s": float(np.max(settled)) if ok.any() else None,
                     "front_points": int(front.sum()), "front_settled": int((front & ok).sum()),
                     "pp_extended_points": int(star.sum()), "pp_extended_settled": int((star & ok).sum()),
                     "trials_with_saturation": int((r[:, :, 3] > 0).sum()),
                     "unsettled_tail_fully_saturated": int(((~ok) & (r[:, :, 9] >= 1)).sum()),
                     "max_abs_ey_all_trials_m": float(np.max(r[:, :, 4])),
                     "max_raw_demand_all_trials_rad_s": float(np.max(r[:, :, 11])),
                     "dKy_nonnegative": (c.wn / V) ** 2 >= 2 / c.ld ** 2 - 1e-12,
                     "dKtheta_nonnegative": 2 * c.zeta * c.wn / V >= 2 / c.ld - 1e-12})
    return rows


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def horizon_extension(sweep_dir, exe, configs, arrays, threads):
    """Re-simulate the points unsettled at the base horizon for 600 s, and the
    PP / ECPP points still unsettled after 600 s for 3600 s."""
    folder = sweep_dir / "supplemental"
    folder.mkdir(exist_ok=True)
    extension = []
    for c in configs:
        r = arrays[c.name]
        ids = np.argwhere(r[:, :, 2] < 0)
        points = np.column_stack((c.ymin + ids[:, 0] * c.dy, c.thmin + ids[:, 1] * c.dth))
        rec = {"name": c.name, "initial_unsettled_120s": len(ids),
               "additional_settled_600s": 0, "unsettled_600s": 0,
               "tail_fully_saturated_600s": 0,
               "additional_settled_3600s": None, "unsettled_3600s": None,
               "max_abs_ey_600s_m": None, "max_abs_ey_3600s_m": None}
        if len(ids) == 0:
            extension.append(rec)
            continue
        p = folder / f"{c.name}_points.tsv"
        np.savetxt(p, points, fmt="%.17g", delimiter="\t")
        cs = replace(c, name=c.name + "_600s", horizon=600)
        conf = folder / f"{cs.name}.tsv"
        conf.write_text(cs.tsv() + "\n")
        out_bin = folder / f"{cs.name}.bin"
        if not _npz(out_bin).exists():
            run_kernel(exe, conf, folder, threads, p)
            _package(out_bin, (-1, len(FIELDS)))
        rr = np.load(_npz(out_bin))["records"].reshape(-1, len(FIELDS))
        rec.update(additional_settled_600s=int((rr[:, 2] >= 0).sum()),
                   unsettled_600s=int((rr[:, 2] < 0).sum()),
                   tail_fully_saturated_600s=int(((rr[:, 2] < 0) & (rr[:, 9] >= 1)).sum()),
                   max_abs_ey_600s_m=float(rr[:, 4].max()))
        if c.method in (0, 3):
            remaining = rr[:, 2] < 0
            if remaining.any():
                p2 = folder / f"{c.name}_remaining_600s.tsv"
                np.savetxt(p2, points[remaining], fmt="%.17g", delimiter="\t")
                cs2 = replace(cs, name=c.name + "_3600s", horizon=3600)
                conf2 = folder / f"{cs2.name}.tsv"
                conf2.write_text(cs2.tsv() + "\n")
                out2 = folder / f"{cs2.name}.bin"
                if not _npz(out2).exists():
                    run_kernel(exe, conf2, folder, threads, p2)
                    _package(out2, (-1, len(FIELDS)))
                r2 = np.load(_npz(out2))["records"].reshape(-1, len(FIELDS))
                rec.update(additional_settled_3600s=int((r2[:, 2] >= 0).sum()),
                           unsettled_3600s=int((r2[:, 2] < 0).sum()),
                           max_abs_ey_3600s_m=float(r2[:, 4].max()))
            else:
                rec.update(additional_settled_3600s=0, unsettled_3600s=0)
        extension.append(rec)
    _write_json(sweep_dir / "horizon_extension.json", extension)
    _write_csv(sweep_dir / "horizon_extension.csv", extension)
    return extension


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_sweep(sweep_dir, threads, configs=None, extension=True):
    """Run the kernel over every configuration (unless the sweep directory is
    already complete), package the results, and write the summaries."""
    configs = configurations() if configs is None else configs
    sweep_dir.mkdir(parents=True, exist_ok=True)
    data = sweep_dir / "data"
    data.mkdir(exist_ok=True)
    exe = sweep_dir / "phase_plane_kernel"
    if not sweep_complete(sweep_dir, configs):
        validation = validate_core(configs)
        _write_json(sweep_dir / "core_validation.json", validation)
        print("controller cross-check passed: max |dv| = "
              f"{max(validation['core_command_max_abs_error_rad_s'].values()):.2e} rad/s", flush=True)
        threaded, compiler = compile_kernel(exe)
        _write_json(sweep_dir / "configurations.json", [asdict(c) for c in configs])
        (sweep_dir / "grid.tsv").write_text("\n".join(c.tsv() for c in configs) + "\n")
        try:
            from importlib.metadata import version as _version
            ecpp_version = _version("ecpp")
        except Exception:  # pragma: no cover - not installed as a distribution
            ecpp_version = None
        metadata = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "ecpp_version": ecpp_version, "ecpp_git_head": _git_head(),
            "kernel_source_sha256": hashlib.sha256(Path(str(KERNEL_SOURCE)).read_bytes()).hexdigest(),
            "compiler": compiler, "openmp": threaded, "threads": threads,
            "binary_record": {"dtype": "little-endian float32", "fields": FIELDS,
                              "ordering": "y major; heading minor"},
            "velocity_m_s": V, "instantaneous_omega_clip_rad_s": LIMIT,
            "settling": SETTLING, "model": MODEL,
            "PP_parameter_note": ("For method=0 (PP), wn and zeta are unused placeholders in "
                                  "the shared record format; the PP curvature depends on Ld "
                                  "and geometry only."),
        }
        _write_json(sweep_dir / "metadata.json", metadata)
        run_kernel(exe, sweep_dir / "grid.tsv", data, threads)
        for c in configs:
            _package(data / f"{c.name}.bin", (c.ny, c.nth, len(FIELDS)))
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(sweep_dir / "metadata.json", metadata)
    else:
        print(f"reusing the complete sweep in {sweep_dir}", flush=True)
    arrays = {c.name: load_records(sweep_dir, c) for c in configs}
    rows = summarize(configs, arrays)
    _write_csv(sweep_dir / "summary.csv", rows)
    _write_json(sweep_dir / "summary.json", rows)
    if extension and not (sweep_dir / "horizon_extension.csv").exists():
        if not exe.exists():
            compile_kernel(exe)
        horizon_extension(sweep_dir, exe, configs, arrays, threads)
    return configs, arrays


# ---------------------------------------------------------------------------
# Tables and figures of the manuscript
# ---------------------------------------------------------------------------
def load_sweep(sweep_dir):
    meta = json.loads((sweep_dir / "metadata.json").read_text())
    configs = {c["name"]: c for c in json.loads((sweep_dir / "configurations.json").read_text())}
    records = {name: np.load(sweep_dir / "data" / (name + ".npz"))["records"] for name in configs}
    ext = {}
    csv_path = sweep_dir / "horizon_extension.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                ext[row["name"]] = row
    return meta, configs, records, ext


def grid_axes(cfg):
    ey = cfg["ymin"] + cfg["dy"] * np.arange(cfg["ny"])
    th = cfg["thmin"] + cfg["dth"] * np.arange(cfg["nth"])
    return ey, th


def eta_deg(ey, th_deg, ld):
    """Angle between the heading and the direction to the lookahead point."""
    e = th_deg[None, :] + np.degrees(np.arctan(ey[:, None] / ld))
    return (e + 180.0) % 360.0 - 180.0


def analyse(meta, configs, records):
    fields = meta["binary_record"]["fields"]
    i_settle = fields.index("settle_end_s")
    i_tailsat = fields.index("tail_saturation_fraction")
    out = {}
    for name, cfg in configs.items():
        arr = records[name]
        settle = arr[..., i_settle]
        settled = settle >= 0.0
        ey, th = grid_axes(cfg)
        eta = eta_deg(ey, th, cfg["ld"])
        front = np.abs(eta) < 90.0
        star = np.abs(eta) < ETA_STAR_DEG
        abs_ey = np.abs(ey)[:, None] * np.ones_like(settle)
        j0 = int(round((0.0 - cfg["thmin"]) / cfg["dth"]))
        out[name] = {
            "controller": name, "method": cfg["method"], "ld": cfg["ld"],
            "wn": cfg["wn"], "zeta": cfg["zeta"], "grid_points": int(settle.size),
            "settled": int(settled.sum()),
            "settled_percent": 100.0 * settled.mean(),
            "unsettled": int((~settled).sum()),
            "median_settle_s": float(np.median(settle[settled])) if settled.any() else None,
            "front_points": int(front.sum()), "front_settled": int((front & settled).sum()),
            "star_points": int(star.sum()), "star_settled": int((star & settled).sum()),
            "max_abs_ey_settled_m": float(abs_ey[settled].max()) if settled.any() else None,
            "max_abs_ey_settled_at_heading0_m": (
                float(abs_ey[:, j0][settled[:, j0]].max()) if settled[:, j0].any() else None),
            "unsettled_tail_fully_saturated": int(((~settled) & (arr[..., i_tailsat] >= 0.999)).sum()),
        }
    for name, cfg in configs.items():
        if cfg["method"] != 3:
            continue
        ref = "pp_ld1" if abs(cfg["ld"] - 1.0) < 1e-9 else "pp_ld0.5"
        a = records[name][..., i_settle] >= 0.0
        b = records[ref][..., i_settle] >= 0.0
        out[name]["agreement_with_pp_percent"] = 100.0 * float((a == b).mean())
        out[name]["disagreement_points"] = int((a != b).sum())
        out[name]["pp_reference"] = ref
    return out


def fmt_int(n):
    return "{:,}".format(int(n))


def write_table(out, table_dir):
    rows = []
    for key in BASELINES:
        r = out[key]
        label, ld, gains = OPERATING[key]
        rows.append((label, "{:.1f}".format(ld), gains, fmt_int(r["settled"]),
                     "{:.1f}".format(r["settled_percent"])))
    ecpp18 = [r for r in out.values() if r["method"] == 3]
    lo = min(ecpp18, key=lambda r: r["settled"])
    hi = max(ecpp18, key=lambda r: r["settled"])
    medians = [r["median_settle_s"] for r in ecpp18 if r["median_settle_s"] is not None]
    rows.append(("ECPP (18 settings)", "0.5, 1.0", "Table~\\ref{tab:sim_test1_results}",
                 "{}--{}".format(fmt_int(lo["settled"]), fmt_int(hi["settled"])),
                 "{:.1f}--{:.1f}".format(lo["settled_percent"], hi["settled_percent"])))
    r = out["pp_ld0.5"]
    rows.append(("PP", "0.5", "--", fmt_int(r["settled"]), "{:.1f}".format(r["settled_percent"])))
    lines = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
             "Method & $L_d$ [m] & $(\\omega_n,\\zeta)$ & Settled & [\\%] \\\\",
             "\\midrule"]
    for row in rows:
        lines.append(" & ".join(row) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "sim_test3_results.tex").write_text("\n".join(lines) + "\n")
    return {"ecpp18_settled_min": lo["settled"], "ecpp18_settled_max": hi["settled"],
            "ecpp18_agreement_min": min(r["agreement_with_pp_percent"] for r in ecpp18),
            "ecpp18_agreement_max": max(r["agreement_with_pp_percent"] for r in ecpp18),
            "ecpp18_median_settle_min_s": min(medians) if medians else None,
            "ecpp18_median_settle_max_s": max(medians) if medians else None,
            "ecpp18_disagreement_points_max": max(r["disagreement_points"] for r in ecpp18)}


def boundary_curves(ey, ld, limit_deg):
    """e_theta(e_y) on which |eta| = limit_deg, split at wrap-around jumps."""
    curves = []
    for sign in (1.0, -1.0):
        th = sign * limit_deg - np.degrees(np.arctan(ey / ld))
        th = (th + 180.0) % 360.0 - 180.0
        jumps = np.flatnonzero(np.abs(np.diff(th)) > 180.0)
        start = 0
        for j in list(jumps) + [len(th) - 1]:
            curves.append((ey[start:j + 1], th[start:j + 1]))
            start = j + 1
    return curves


def _panel(ax, key, cfg, settle, cmap, horizon):
    ey, th = grid_axes(cfg)
    data = np.ma.masked_less(settle, 0.0).T   # rows = heading, cols = e_y
    extent = [ey[0] - cfg["dy"] / 2, ey[-1] + cfg["dy"] / 2,
              th[0] - cfg["dth"] / 2, th[-1] + cfg["dth"] / 2]
    image = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                      cmap=cmap, vmin=0.0, vmax=horizon, interpolation="nearest")
    if key in ("pp_ld1", "ecpp_ld1_w1_z1"):
        for x, y in boundary_curves(ey, cfg["ld"], 90.0):
            ax.plot(x, y, color="white", linestyle="--", linewidth=0.9)
        for x, y in boundary_curves(ey, cfg["ld"], ETA_STAR_DEG):
            ax.plot(x, y, color="white", linestyle=":", linewidth=1.2)
    return image


def plot_phase_planes_row(meta, configs, records, fig_dir):
    """Four panels at manuscript text width, with shared y axis and colorbar."""
    i_settle = meta["binary_record"]["fields"].index("settle_end_s")
    horizon = configs["pp_ld1"]["horizon"]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(UNSETTLED_COLOR)
    panels = [(key, OPERATING[key][0]) for key in BASELINES]
    fig = plt.figure(figsize=(7.0, 1.9))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06],
                            left=0.085, right=0.91, bottom=0.32, top=0.965,
                            wspace=0.22)
    axes = []
    image = None
    for index, ((key, label), tag) in enumerate(zip(panels, "abcd")):
        ax = fig.add_subplot(grid[0, index], sharey=axes[0] if axes else None)
        axes.append(ax)
        cfg = configs[key]
        image = _panel(ax, key, cfg, records[key][..., i_settle].astype(float), cmap, horizon)
        ax.set_xticks([-10, 0, 10])
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.tick_params(labelsize=8, length=2.5, pad=2, labelleft=index == 0)
        ax.set_xlabel("$e_y(0)$ [m]", fontsize=9, labelpad=2)
        if index == 0:
            ax.set_ylabel("$e_\\theta(0)$ [$^\\circ$]", fontsize=9, labelpad=2)
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.08, "({}) {}".format(tag, label),
                 ha="center", va="bottom", fontsize=9)
    cb = fig.colorbar(image, cax=fig.add_subplot(grid[0, 4]))
    cb.set_ticks(np.linspace(0.0, horizon, 7))
    cb.set_label("Settling time $T_s$ [s]", fontsize=9, labelpad=3)
    cb.ax.tick_params(labelsize=8, length=2.5, pad=2)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "sim_test3_phase_planes_row.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "sim_test3_phase_planes_row.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_planes(meta, configs, records, fig_dir):
    """2 x 2 figure, separate panels without titles, and a standalone colorbar."""
    i_settle = meta["binary_record"]["fields"].index("settle_end_s")
    horizon = configs["pp_ld1"]["horizon"]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(UNSETTLED_COLOR)
    panels = [(key, OPERATING[key][0]) for key in BASELINES]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
    axes = axes.ravel()
    image = None
    for ax, (key, label), tag in zip(axes, panels, "abcd"):
        image = _panel(ax, key, configs[key], records[key][..., i_settle].astype(float), cmap, horizon)
        ax.set_title("({}) {}".format(tag, label), fontsize=9, loc="left")
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.tick_params(labelsize=8)
    for ax in axes[2:]:
        ax.set_xlabel("$e_y(0)$ [m]", fontsize=9)
    for ax in axes[::2]:
        ax.set_ylabel("$e_\\theta(0)$ [$^\\circ$]", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 0.90, 1.0))
    cax = fig.add_axes([0.915, 0.12, 0.018, 0.76])
    cb = fig.colorbar(image, cax=cax)
    cb.set_label("Settling time $T_s$ [s]", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "sim_test3_phase_planes.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "sim_test3_phase_planes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    panel_files = {"pp_ld1": "pp", "dpp_baseline": "dpp",
                   "ungated_baseline": "ungated", "ecpp_ld1_w1_z1": "ecpp"}
    for key, stem in panel_files.items():
        pfig, pax = plt.subplots(figsize=(3.3, 2.45))
        _panel(pax, key, configs[key], records[key][..., i_settle].astype(float), cmap, horizon)
        pax.set_yticks([-180, -90, 0, 90, 180])
        pax.tick_params(labelsize=8)
        pax.set_xlabel("$e_y(0)$ [m]", fontsize=9)
        pax.set_ylabel("$e_\\theta(0)$ [$^\\circ$]", fontsize=9)
        pfig.tight_layout(pad=0.3)
        pfig.savefig(fig_dir / "sim_test3_phase_plane_{}.pdf".format(stem), bbox_inches="tight")
        pfig.savefig(fig_dir / "sim_test3_phase_plane_{}.png".format(stem), dpi=180, bbox_inches="tight")
        plt.close(pfig)
    cfig = plt.figure(figsize=(0.9, 4.6))
    cax = cfig.add_axes([0.05, 0.05, 0.25, 0.90])
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=horizon))
    cb = cfig.colorbar(mappable, cax=cax)
    cb.set_label("Settling time $T_s$ [s]", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cfig.savefig(fig_dir / "sim_test3_colorbar.pdf", bbox_inches="tight")
    cfig.savefig(fig_dir / "sim_test3_colorbar.png", dpi=180, bbox_inches="tight")
    plt.close(cfig)


def render(sweep_dir, table_dir, fig_dir):
    """Write the manuscript table, summary JSON, and figures from a sweep."""
    meta, configs, records, ext = load_sweep(sweep_dir)
    out = analyse(meta, configs, records)
    extra = write_table(out, table_dir)
    plot_phase_planes(meta, configs, records, fig_dir)
    plot_phase_planes_row(meta, configs, records, fig_dir)
    keys = ("initial_unsettled_120s", "unsettled_600s", "unsettled_3600s", "tail_fully_saturated_600s")
    summary = {
        "source": str(sweep_dir),
        "summary_csv_sha256": hashlib.sha256((sweep_dir / "summary.csv").read_bytes()).hexdigest(),
        "eta_star_deg": ETA_STAR_DEG,
        "settling_definition": meta["settling"], "model": meta["model"],
        "horizon_extension": {k: {kk: v[kk] for kk in keys} for k, v in ext.items()},
        "per_controller": out, "ecpp18": extra,
    }
    (table_dir / "sim_test3_summary.json").write_text(json.dumps(summary, indent=1))
    for key in BASELINES + ("pp_ld0.5",):
        r = out[key]
        print("{:18s} settled {:>6d} ({:5.2f}%) medianTs {}  maxAbsEy {}  front {}/{}  star {}/{}  agree {}".format(
            key, r["settled"], r["settled_percent"],
            ia.f(r["median_settle_s"], 2), ia.f(r["max_abs_ey_settled_m"], 2),
            r["front_settled"], r["front_points"], r["star_settled"], r["star_points"],
            r.get("agreement_with_pp_percent")))
    print("ECPP 18 settings:", extra)
    if ext:
        print("600 s extension:", {k: ext[k]["unsettled_600s"] for k in ("pp_ld1", "ecpp_ld1_w1_z1", "pp_ld0.5") if k in ext})
    return summary


# ---------------------------------------------------------------------------
def main(argv=None, *, default_output_root=None, default_sweep_dir=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="write frozen outputs to generated/ instead of preview")
    ap.add_argument("--out-root", type=Path, default=default_output_root or Path.cwd(),
                    help="output root (default: current directory); outputs go to "
                         "<out-root>/generated_preview/ unless --apply is given")
    ap.add_argument("--sweep-dir", type=Path, default=default_sweep_dir,
                    help=f"sweep directory (default: <outputs>/{SWEEP_DIRNAME}); "
                         "a complete one is reused without recomputing")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                    help="OpenMP threads for the kernel (default: all cores)")
    ap.add_argument("--skip-extension", action="store_true",
                    help="skip the 600 s / 3600 s re-simulation of unsettled points")
    ap.add_argument("--ny", type=int, default=None, help="grid points in e_y (smoke tests only)")
    ap.add_argument("--nth", type=int, default=None, help="grid points in e_theta (smoke tests only)")
    ap.add_argument("--horizon", type=float, default=None, help="horizon in s (smoke tests only)")
    args = ap.parse_args(argv)

    sub = "generated" if args.apply else "generated_preview"
    table_dir = args.out_root / sub / "tables"
    fig_dir = args.out_root / sub / "figures"
    sweep_dir = args.sweep_dir if args.sweep_dir is not None else args.out_root / sub / SWEEP_DIRNAME
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    if any(v is not None for v in (args.ny, args.nth, args.horizon)):
        if args.apply:
            ap.error("--ny/--nth/--horizon are smoke-test options; do not combine with --apply")
        configs = reduced(configs, args.ny, args.nth, args.horizon)
    print(f"Out root: {args.out_root / sub}  (apply={args.apply})")
    print(f"Sweep dir: {sweep_dir}  threads = {args.threads}")
    run_sweep(sweep_dir, args.threads, configs, extension=not args.skip_extension)
    render(sweep_dir, table_dir, fig_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()

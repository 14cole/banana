#!/usr/bin/env python3
"""
validate_vs_analytical.py
=========================

Generates matplotlib comparison plots between the 2D BIE/MoM solver and
closed-form Mie-series / physical-optics references.

Each test produces a two-panel PNG:
    - top panel: solver vs analytical sigma_2D (dB)
    - bottom panel: solver - analytical (dB), with +/-0.5 dB guides

A summary table at the end flags PASS / MARGINAL / FAIL per test.

Dependencies
------------
Must be importable from the current Python environment:
    rcs_solver.py        (your solver)
    mie_reference.py     (Mie series oracle)
and of course numpy + scipy + matplotlib.

Usage
-----
    python validate_vs_analytical.py                  # run everything
    python validate_vs_analytical.py --quick          # coarser, much faster
    python validate_vs_analytical.py --outdir plots/  # custom directory
    python validate_vs_analytical.py --test pec_freq  # run one test

Available tests:
    pec_freq, pec_azim, diel_lossless, diel_lossy,
    coated_pec, pec_po, mesh_conv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-friendly backend
import matplotlib.pyplot as plt

from rcs_solver import solve_monostatic_rcs_2d, C0
from mie_reference import (
    sigma_pec_cylinder,
    sigma_dielectric_cylinder,
    sigma_coated_pec_cylinder,
)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry builders  (match validate_mie.py conventions)
# ─────────────────────────────────────────────────────────────────────────────

def _cw_circle_pairs(radius_m: float, n_sides: int):
    """CW polygon endpoints for radius r, so (-ty, tx) normals point OUTWARD.

    For TYPE 2 (PEC) and TYPE 3 (dielectric interface) the user-drawn normal
    must point INTO AIR — drawing CW does exactly that.
    """
    theta = np.linspace(0.0, -2.0 * np.pi, n_sides + 1)
    xs = radius_m * np.cos(theta)
    ys = radius_m * np.sin(theta)
    return [
        {"x1": float(xs[i]), "y1": float(ys[i]),
         "x2": float(xs[i + 1]), "y2": float(ys[i + 1])}
        for i in range(n_sides)
    ]


def build_pec_cylinder(radius_m, n_sides, ppw=20):
    return {
        "segments": [{
            "name": "pec_circle",
            "seg_type": 2,
            "properties": ["2", str(-max(2, ppw)), "0.0", "0", "0", "0"],
            "point_pairs": _cw_circle_pairs(radius_m, n_sides),
        }],
        "ibcs": [],
        "dielectrics": [],
    }


def build_dielectric_cylinder(radius_m, eps_r, mu_r, n_sides, ppw=20):
    eps_r = complex(eps_r)
    mu_r = complex(mu_r)
    return {
        "segments": [{
            "name": "diel_circle",
            "seg_type": 3,
            "properties": ["3", str(-max(2, ppw)), "0.0", "0", "1", "0"],
            "point_pairs": _cw_circle_pairs(radius_m, n_sides),
        }],
        "ibcs": [],
        # NOTE: dielectric table takes POSITIVE imaginary for loss; the solver
        # flips sign internally to e^{-jwt}. Pass physical eps_r, negate imag here.
        "dielectrics": [[
            "1",
            f"{eps_r.real:.6g}", f"{-eps_r.imag:.6g}",
            f"{mu_r.real:.6g}",  f"{-mu_r.imag:.6g}",
        ]],
    }


def build_coated_pec(a_in, a_out, eps_r, mu_r, n_sides_out, n_sides_in, ppw=20):
    eps_r = complex(eps_r)
    mu_r = complex(mu_r)
    return {
        "segments": [
            {
                "name": "outer_air_diel",
                "seg_type": 3,
                "properties": ["3", str(-max(2, ppw)), "0.0", "0", "1", "0"],
                "point_pairs": _cw_circle_pairs(a_out, n_sides_out),
            },
            {
                "name": "inner_diel_pec",
                "seg_type": 4,
                "properties": ["4", str(-max(2, ppw)), "0.0", "0", "1", "0"],
                "point_pairs": _cw_circle_pairs(a_in, n_sides_in),
            },
        ],
        "ibcs": [],
        "dielectrics": [[
            "1",
            f"{eps_r.real:.6g}", f"{-eps_r.imag:.6g}",
            f"{mu_r.real:.6g}",  f"{-mu_r.imag:.6g}",
        ]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

def to_db(sigma):
    sigma = np.asarray(sigma, dtype=float)
    return 10.0 * np.log10(np.maximum(sigma, 1e-30))


def _comparison_plot(
    title, xlabel, sweep, ref_db, solver_db, savename,
    ref_label="Analytical (Mie)",
    solver_label="Solver (2D BIE/MoM)",
):
    """Two-panel comparison figure; returns max |error| in dB."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.plot(sweep, ref_db, "k-", lw=2, label=ref_label)
    ax_top.plot(
        sweep, solver_db, "o-", color="#d62728",
        ms=4, lw=1, alpha=0.85, label=solver_label,
    )
    ax_top.set_ylabel(r"$\sigma_{2\mathrm{D}}$  (dB$\cdot$m)")
    ax_top.set_title(title)
    ax_top.legend(loc="best")
    ax_top.grid(True, alpha=0.3)

    err = np.asarray(solver_db) - np.asarray(ref_db)
    ax_bot.plot(sweep, err, "o-", color="#2ca02c", ms=3, lw=1)
    ax_bot.axhline(0, color="k", lw=0.5)
    for lvl in (0.5, -0.5):
        ax_bot.axhline(lvl, color="k", lw=0.3, ls="--", alpha=0.5)
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("error (dB)")
    ax_bot.grid(True, alpha=0.3)

    finite = np.isfinite(err)
    if finite.any():
        p95 = float(np.nanpercentile(np.abs(err[finite]), 95))
        yabs = min(6.0, max(1.0, 1.3 * p95))
        ax_bot.set_ylim(-yabs, yabs)

    fig.tight_layout()
    fig.savefig(savename, dpi=130)
    plt.close(fig)
    return float(np.nanmax(np.abs(err)))


def _banner(title):
    print()
    print("═" * 74)
    print(title)
    print("═" * 74)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_pec_freq(outdir, quick=False):
    """PEC circular cylinder — monostatic sigma_2D vs ka, TE & TM."""
    _banner("PEC cylinder — frequency sweep vs Mie series")
    radius_m = 0.1
    ka_min, ka_max = 0.3, 15.0
    n_freq = 15 if quick else 30

    results = []
    for pol in ("TE", "TM"):
        ka_vals = np.linspace(ka_min, ka_max, n_freq)
        freqs_ghz = ka_vals * C0 / (2.0 * np.pi * radius_m) / 1.0e9

        mie = np.array([
            sigma_pec_cylinder(radius_m, f * 1e9, pol) for f in freqs_ghz
        ])

        # Enough input polygon sides for Mie-quality accuracy at ka_max.
        n_sides = max(60, int(np.ceil(20 * ka_max * 1.2)))

        snap = build_pec_cylinder(radius_m, n_sides, ppw=20)

        t0 = time.time()
        result = solve_monostatic_rcs_2d(
            geometry_snapshot=snap,
            frequencies_ghz=freqs_ghz.tolist(),
            elevations_deg=[0.0],
            polarization=pol,
            geometry_units="meters",
            material_base_dir=".",
            mesh_reference_ghz=float(freqs_ghz.max()),
            max_panels=200000,
        )
        dt = time.time() - t0

        samples = sorted(result["samples"], key=lambda s: s["frequency_ghz"])
        solver = np.array([s["rcs_linear"] for s in samples])

        fname = os.path.join(outdir, f"validation_pec_freq_{pol}.png")
        title = (f"PEC cylinder frequency sweep — {pol}    "
                 f"radius = {radius_m*1000:.0f} mm, {n_sides} input sides")
        max_err = _comparison_plot(
            title, r"$ka$", ka_vals, to_db(mie), to_db(solver), fname,
        )
        results.append((f"PEC freq  {pol}", max_err, dt, fname))
        print(f"  [{pol}] n_freq={n_freq}  n_sides={n_sides}  "
              f"max|err|={max_err:.3f} dB  ({dt:.1f}s)  →  {os.path.basename(fname)}")
    return results


def test_pec_azim(outdir, quick=False):
    """Monostatic azimuth sweep on a PEC cylinder: sigma should be CONSTANT
    (rotational symmetry). Non-constancy reveals angle-convention or
    discretization asymmetries in the solver."""
    _banner("PEC cylinder — azimuth sweep  (rotational-invariance check)")
    radius_m = 0.1
    freq_ghz = 3.0
    n_angles = 37 if quick else 73

    results = []
    for pol in ("TE", "TM"):
        angles = np.linspace(0.0, 360.0, n_angles)
        ka = 2.0 * np.pi * freq_ghz * 1e9 * radius_m / C0

        mie_const = sigma_pec_cylinder(radius_m, freq_ghz * 1e9, pol)
        mie = np.full(n_angles, mie_const)

        n_sides = max(120, int(np.ceil(25 * ka)))
        snap = build_pec_cylinder(radius_m, n_sides, ppw=25)

        t0 = time.time()
        result = solve_monostatic_rcs_2d(
            geometry_snapshot=snap,
            frequencies_ghz=[freq_ghz],
            elevations_deg=angles.tolist(),
            polarization=pol,
            geometry_units="meters",
            material_base_dir=".",
            max_panels=200000,
        )
        dt = time.time() - t0

        samples = sorted(result["samples"], key=lambda s: s["theta_inc_deg"])
        solver = np.array([s["rcs_linear"] for s in samples])

        fname = os.path.join(outdir, f"validation_pec_azim_{pol}.png")
        title = (f"PEC cylinder azimuth sweep — {pol}    "
                 f"f = {freq_ghz} GHz  (ka = {ka:.2f})    {n_sides} sides")
        max_err = _comparison_plot(
            title, "azimuth (deg)", angles, to_db(mie), to_db(solver), fname,
            ref_label=f"Mie (const at {to_db(np.array([mie_const]))[0]:.3f} dB)",
        )
        results.append((f"PEC azim  {pol}", max_err, dt, fname))
        print(f"  [{pol}] ka={ka:.2f}  n_angles={n_angles}  n_sides={n_sides}  "
              f"max|err|={max_err:.3f} dB  ({dt:.1f}s)  →  {os.path.basename(fname)}")
    return results


def _test_diel_freq(outdir, eps_r, suffix, quick=False,
                    radius_m=0.1, ka_min=0.3, ka_max=8.0):
    """Dielectric cylinder frequency sweep, TE & TM."""
    n_freq = 10 if quick else 20
    n_rel = abs(np.sqrt(complex(eps_r) * 1.0))

    results = []
    for pol in ("TE", "TM"):
        ka_vals = np.linspace(ka_min, ka_max, n_freq)
        freqs_ghz = ka_vals * C0 / (2.0 * np.pi * radius_m) / 1.0e9

        mie = np.array([
            sigma_dielectric_cylinder(radius_m, eps_r, 1.0, f * 1e9, pol)
            for f in freqs_ghz
        ])

        # Mesh must resolve the INTERIOR wavelength: need n_rel * ppw per λ0.
        n_sides = max(80, int(np.ceil(25 * ka_max * n_rel * 1.2)))
        snap = build_dielectric_cylinder(radius_m, eps_r, 1.0, n_sides, ppw=20)

        t0 = time.time()
        result = solve_monostatic_rcs_2d(
            geometry_snapshot=snap,
            frequencies_ghz=freqs_ghz.tolist(),
            elevations_deg=[0.0],
            polarization=pol,
            geometry_units="meters",
            material_base_dir=".",
            mesh_reference_ghz=float(freqs_ghz.max()),
            max_panels=300000,
        )
        dt = time.time() - t0

        samples = sorted(result["samples"], key=lambda s: s["frequency_ghz"])
        solver = np.array([s["rcs_linear"] for s in samples])

        fname = os.path.join(outdir, f"validation_diel_{suffix}_freq_{pol}.png")
        title = (f"Dielectric cylinder ({suffix}) frequency sweep — {pol}    "
                 f"radius = {radius_m*1000:.0f} mm, "
                 rf"$\varepsilon_r$ = {eps_r.real:g} "
                 f"{'- ' if eps_r.imag < 0 else '+ '}{abs(eps_r.imag):g}j")
        max_err = _comparison_plot(
            title, r"$k_0 a$", ka_vals, to_db(mie), to_db(solver), fname,
        )
        results.append((f"Diel {suffix} freq {pol}", max_err, dt, fname))
        print(f"  [{pol}] n_freq={n_freq}  n_sides={n_sides}  "
              f"max|err|={max_err:.3f} dB  ({dt:.1f}s)  →  {os.path.basename(fname)}")
    return results


def test_diel_lossless(outdir, quick=False):
    _banner("Dielectric cylinder (lossless, eps_r = 4) — frequency sweep vs Mie")
    return _test_diel_freq(outdir, 4.0 + 0.0j, "lossless", quick=quick)


def test_diel_lossy(outdir, quick=False):
    _banner("Dielectric cylinder (LOSSY, eps_r = 4 - 0.5j) — frequency sweep vs Mie")
    print("  (This exercises the complex-k Hankel code path.)")
    return _test_diel_freq(outdir, 4.0 - 0.5j, "lossy", quick=quick)


def test_coated_pec(outdir, quick=False):
    """Coated PEC (dielectric coating over PEC core) — TE & TM frequency sweep."""
    _banner("Coated PEC cylinder — frequency sweep vs Mie")
    a_in, a_out = 0.05, 0.1
    eps_r = 4.0 + 0.0j
    ka_min, ka_max = 0.5, 6.0
    n_freq = 10 if quick else 15
    n_rel = abs(np.sqrt(complex(eps_r) * 1.0))

    results = []
    for pol in ("TE", "TM"):
        ka_vals = np.linspace(ka_min, ka_max, n_freq)
        freqs_ghz = ka_vals * C0 / (2.0 * np.pi * a_out) / 1.0e9

        mie = np.array([
            sigma_coated_pec_cylinder(a_in, a_out, eps_r, 1.0, f * 1e9, pol)
            for f in freqs_ghz
        ])

        # Outer and inner radii get their own panel count (so panel sizes match).
        n_sides_out = max(80, int(np.ceil(25 * ka_max * 1.2)))
        n_sides_in = max(60, int(np.ceil(25 * ka_max * n_rel * 1.2
                                          * (a_in / a_out))))

        snap = build_coated_pec(
            a_in, a_out, eps_r, 1.0,
            n_sides_out=n_sides_out, n_sides_in=n_sides_in, ppw=20,
        )

        t0 = time.time()
        result = solve_monostatic_rcs_2d(
            geometry_snapshot=snap,
            frequencies_ghz=freqs_ghz.tolist(),
            elevations_deg=[0.0],
            polarization=pol,
            geometry_units="meters",
            material_base_dir=".",
            mesh_reference_ghz=float(freqs_ghz.max()),
            max_panels=300000,
        )
        dt = time.time() - t0

        samples = sorted(result["samples"], key=lambda s: s["frequency_ghz"])
        solver = np.array([s["rcs_linear"] for s in samples])

        fname = os.path.join(outdir, f"validation_coated_pec_freq_{pol}.png")
        title = (f"Coated PEC cylinder frequency sweep — {pol}    "
                 f"a_in = {a_in*1000:.0f} mm, a_out = {a_out*1000:.0f} mm, "
                 rf"$\varepsilon_r$ = {eps_r.real:g}")
        max_err = _comparison_plot(
            title, r"$k_0 a_{\mathrm{out}}$", ka_vals,
            to_db(mie), to_db(solver), fname,
        )
        results.append((f"Coated PEC {pol}", max_err, dt, fname))
        print(f"  [{pol}] n_freq={n_freq}  n_out/n_in={n_sides_out}/{n_sides_in}  "
              f"max|err|={max_err:.3f} dB  ({dt:.1f}s)  →  {os.path.basename(fname)}")
    return results


def test_pec_po(outdir, quick=False):
    """High-frequency (large ka) PO asymptote: sigma_2D / (pi*a) -> 1 for PEC."""
    _banner("PEC cylinder — high-ka physical-optics asymptote")
    radius_m = 0.1
    ka_min, ka_max = 2.0, 25.0
    n_freq = 12 if quick else 20

    results = []
    for pol in ("TE", "TM"):
        ka_vals = np.linspace(ka_min, ka_max, n_freq)
        freqs_ghz = ka_vals * C0 / (2.0 * np.pi * radius_m) / 1.0e9

        mie = np.array([
            sigma_pec_cylinder(radius_m, f * 1e9, pol) for f in freqs_ghz
        ])
        po_limit = np.pi * radius_m  # specular-optics limit, all ka

        n_sides = max(120, int(np.ceil(25 * ka_max * 1.2)))
        snap = build_pec_cylinder(radius_m, n_sides, ppw=25)

        t0 = time.time()
        result = solve_monostatic_rcs_2d(
            geometry_snapshot=snap,
            frequencies_ghz=freqs_ghz.tolist(),
            elevations_deg=[0.0],
            polarization=pol,
            geometry_units="meters",
            material_base_dir=".",
            mesh_reference_ghz=float(freqs_ghz.max()),
            max_panels=500000,
        )
        dt = time.time() - t0

        samples = sorted(result["samples"], key=lambda s: s["frequency_ghz"])
        solver = np.array([s["rcs_linear"] for s in samples])

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(ka_vals, mie / po_limit, "k-", lw=2, label="Mie series")
        ax.plot(
            ka_vals, solver / po_limit, "o-", color="#d62728",
            ms=5, lw=1, alpha=0.85, label="Solver",
        )
        ax.axhline(1.0, color="#7f7f7f", lw=1.3, ls="--",
                   label=r"PO asymptote  ($\sigma_{2D} = \pi a$)")
        ax.set_xlabel(r"$ka$")
        ax.set_ylabel(r"$\sigma_{2\mathrm{D}} \,/\, (\pi a)$")
        ax.set_title(
            f"PEC cylinder PO asymptote — {pol}    "
            f"radius = {radius_m*1000:.0f} mm, {n_sides} sides"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ymin = float(min(mie.min(), solver.min()) / po_limit)
        ymax = float(max(mie.max(), solver.max()) / po_limit)
        pad = 0.08 * max(1.0, ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
        fig.tight_layout()
        fname = os.path.join(outdir, f"validation_pec_po_{pol}.png")
        fig.savefig(fname, dpi=130)
        plt.close(fig)

        err_db = to_db(solver) - to_db(mie)
        max_err = float(np.nanmax(np.abs(err_db)))
        ratio_range = (float((solver / po_limit).min()),
                       float((solver / po_limit).max()))
        results.append((f"PEC PO    {pol}", max_err, dt, fname))
        print(f"  [{pol}] ka in [{ka_min},{ka_max}]  n_sides={n_sides}  "
              f"sigma/(pi*a) in [{ratio_range[0]:.3f}, {ratio_range[1]:.3f}]  "
              f"max|err vs Mie|={max_err:.3f} dB  ({dt:.1f}s)")
        print(f"          →  {os.path.basename(fname)}")
    return results


def test_mesh_conv(outdir, quick=False):
    """Mesh convergence: |solver - Mie| vs panels-per-wavelength at fixed ka."""
    _banner("PEC cylinder — mesh-convergence study at fixed ka")
    radius_m = 0.1
    ka = 5.0
    freq_ghz = ka * C0 / (2.0 * np.pi * radius_m) / 1.0e9
    n_sides_fixed = 60  # input polygon edges; internal subdivision varies with ppw

    ppw_vals = [6, 10, 15, 20, 30, 50] if not quick else [6, 15, 30]

    results = []
    for pol in ("TE", "TM"):
        mie_val = sigma_pec_cylinder(radius_m, freq_ghz * 1e9, pol)
        mie_db = 10.0 * np.log10(mie_val)

        solver_db_list = []
        times = []
        for ppw in ppw_vals:
            snap = build_pec_cylinder(radius_m, n_sides_fixed, ppw=ppw)
            t0 = time.time()
            result = solve_monostatic_rcs_2d(
                geometry_snapshot=snap,
                frequencies_ghz=[freq_ghz],
                elevations_deg=[0.0],
                polarization=pol,
                geometry_units="meters",
                material_base_dir=".",
                max_panels=200000,
            )
            times.append(time.time() - t0)
            s = result["samples"][0]
            solver_db_list.append(10.0 * np.log10(float(s["rcs_linear"])))
        solver_db = np.array(solver_db_list)
        err_db = np.abs(solver_db - mie_db)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True,
            gridspec_kw={"height_ratios": [1, 1]},
        )
        ax_top.axhline(mie_db, color="k", lw=2, label=f"Mie = {mie_db:.3f} dB")
        ax_top.plot(ppw_vals, solver_db, "o-", color="#d62728",
                    ms=7, lw=1.5, label="Solver")
        ax_top.set_ylabel(r"$\sigma_{2\mathrm{D}}$  (dB$\cdot$m)")
        ax_top.set_title(f"Mesh convergence at ka = {ka} — {pol}   "
                         f"({n_sides_fixed} input sides + internal refinement)")
        ax_top.legend(loc="best")
        ax_top.grid(True, alpha=0.3)

        ax_bot.loglog(ppw_vals, np.maximum(err_db, 1e-6),
                      "o-", color="#2ca02c", ms=7, lw=1.5)
        ax_bot.set_xlabel("panels per wavelength (internal)")
        ax_bot.set_ylabel(r"$|$error$|$  (dB)")
        ax_bot.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        fname = os.path.join(outdir, f"validation_mesh_conv_{pol}.png")
        fig.savefig(fname, dpi=130)
        plt.close(fig)

        best_err = float(err_db[-1])
        results.append((f"Mesh conv {pol}", best_err, sum(times), fname))
        print(f"  [{pol}] ka={ka}  best (ppw={ppw_vals[-1]}): |err|={best_err:.3f} dB  "
              f"({sum(times):.1f}s)  →  {os.path.basename(fname)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = {
    "pec_freq":      test_pec_freq,
    "pec_azim":      test_pec_azim,
    "diel_lossless": test_diel_lossless,
    "diel_lossy":    test_diel_lossy,
    "coated_pec":    test_coated_pec,
    "pec_po":        test_pec_po,
    "mesh_conv":     test_mesh_conv,
}


def _verdict(err_db):
    if err_db < 0.5:
        return "PASS"
    if err_db < 1.5:
        return "MARGINAL"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser(
        description=("Generate solver-vs-analytical validation plots "
                     "for the 2D BIE/MoM RCS solver."),
    )
    ap.add_argument("--outdir", default="validation_plots",
                    help="Directory for output PNG files.")
    ap.add_argument("--quick", action="store_true",
                    help="Coarser sweeps — much faster (~2-3x).")
    ap.add_argument("--test", default="all",
                    help=("Run only one test. "
                          f"Choices: all, {', '.join(ALL_TESTS)}"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.outdir)}")
    print(f"Quick mode:       {args.quick}")

    tests_to_run = list(ALL_TESTS.keys()) if args.test == "all" else [args.test]
    unknown = [t for t in tests_to_run if t not in ALL_TESTS]
    if unknown:
        print(f"Unknown test(s): {unknown}. Valid: {list(ALL_TESTS.keys())}")
        sys.exit(2)

    all_results = []
    t_total = time.time()
    for name in tests_to_run:
        try:
            all_results.extend(ALL_TESTS[name](args.outdir, args.quick))
        except Exception as exc:
            print(f"\n  !! test '{name}' raised: {exc}")
            traceback.print_exc(file=sys.stdout)
            print()
    dt_total = time.time() - t_total

    print()
    print("═" * 74)
    print(f"SUMMARY  (wall time {dt_total:.1f}s)")
    print("═" * 74)
    print(f"  {'Test':22s}  {'max|err|':>11s}  {'time':>8s}  {'verdict':>8s}  plot")
    print("  " + "─" * 72)
    n_pass = n_marg = n_fail = 0
    for label, err, t, fname in all_results:
        v = _verdict(err)
        n_pass += (v == "PASS")
        n_marg += (v == "MARGINAL")
        n_fail += (v == "FAIL")
        print(f"  {label:22s}  {err:>8.3f} dB  {t:>6.1f}s  "
              f"{v:>8s}  {os.path.basename(fname)}")
    print("  " + "─" * 72)
    print(f"  {n_pass} PASS    {n_marg} MARGINAL    {n_fail} FAIL    "
          f"/ {len(all_results)} total")
    print()


if __name__ == "__main__":
    main()

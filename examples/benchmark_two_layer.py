"""
benchmark_two_layer.py — First inhomogeneous test: two-half-space model.

Verifies that the Lebedev scheme with `layered_isotropic` media correctly
handles a sharp conductivity contrast, by comparing to the exact Sommerfeld
integral solution for a VED in a two-layer model.

Physical setup
--------------
Source   : z-directed (VED) unit electric current dipole at (0, 0, 0).
Medium   : Layer 1  z < 4 m,  σ₁ = 0.1 S/m
           Layer 2  z ≥ 4 m,  σ₂ = 1.0 S/m
           Contact at z_c = 4 m (10× conductivity jump)
           f = 2500 Hz
Receivers: on-axis (x = y = 0), z ∈ [0.5, 7.5] m.

Analytic
--------
  • Homogeneous run:  closed-form `electric_dipole_Ez_homogeneous_onaxis`
    (two half-dipoles at z = ±DZ matching the C000 source geometry).
  • Two-layer run:    `electric_dipole_Ez_two_layer_onaxis` (Sommerfeld integral).

Field reading — correct Lebedev average
----------------------------------------
For VED (dipole_comp=2), C000 natively owns E_z at type-(0,0,1) R-nodes
(i_even, j_even, k_odd).  The on-axis R-nodes are exactly these nodes at
(Mx2, My2, k_odd).  We therefore:
  1. Read C000's E_z directly from result["E_c"][C000] at these DOFs.
  2. Interpolate E_z for the other three clusters using interpolate_cluster_E.
  3. Average the four cluster values — this is the true Lebedev average.

The raw E_avg stored by the solver is NOT the correct physical average:
it averages the DOF vectors directly without cluster-aware interpolation,
giving ~0.25× the correct amplitude at native-C000 nodes.

Grid: same hybrid z-grid as `benchmark_logging.py` (k=3, DZ=0.0625 m).

Usage
-----
    python examples/benchmark_two_layer.py

Output: examples/benchmark_two_layer.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em.grid import (
    LebedevGrid3D, symmetric_optimal_grid, hybrid_axial_grid,
    C000, C101, C110, C011,
)
from lebedev_em.media import homogeneous_isotropic, layered_isotropic, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.analytics import (
    electric_dipole_Ez_homogeneous_onaxis,
    electric_dipole_Ez_two_layer_onaxis,
)
from lebedev_em.sources import _native_type_for_cluster_comp
from lebedev_em.postprocess import interpolate_cluster_E

# ── Physical parameters ───────────────────────────────────────────────────────
SIGMA1  = 0.1          # [S/m] layer 1 conductivity (source layer)
SIGMA2  = 1.0          # [S/m] layer 2 conductivity
Z_SRC   = 0.0          # [m]  source location
Z_CONT  = 4.0          # [m]  contact depth
FREQ    = 2500.0        # [Hz]
OMEGA   = 2.0 * np.pi * FREQ
MU      = MU0
EPS     = EPS0

delta1 = np.sqrt(2.0 / (OMEGA * MU * SIGMA1))
delta2 = np.sqrt(2.0 / (OMEGA * MU * SIGMA2))

print("=" * 70)
print("Lebedev FD — Two-half-space benchmark")
print(f"  σ₁ = {SIGMA1} S/m, δ₁ ≈ {delta1:.1f} m")
print(f"  σ₂ = {SIGMA2} S/m, δ₂ ≈ {delta2:.1f} m")
print(f"  f  = {FREQ:.0f} Hz,  contact at z_c = {Z_CONT} m")
print("=" * 70)

# ── Grid ─────────────────────────────────────────────────────────────────────
DZ          = 0.0625
Z_INNER_MIN = -0.25
Z_INNER_MAX =  7.75
N_INNER     = int(round((Z_INNER_MAX - Z_INNER_MIN) / DZ))   # 128
K_OUTER     = 8
GAMMA       = 1.0 / np.sqrt(2.0)
H_MIN       = 0.5
L_TRANS     = 300.0
K_GRID      = 3        # transverse refinement level (Mx = My = 4k = 12)

z_fine = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)
grid   = symmetric_optimal_grid(H_MIN, L_TRANS, z_fine, GAMMA, k=K_GRID)

Mx2  = grid.Mx // 2
My2  = grid.My // 2
x_max = float(grid.x[-1])
print(f"\n  Grid: Mx=My={grid.Mx}, Mz={grid.Mz}, N_R={grid.N_R}")
print(f"  x[{Mx2}]={grid.x[Mx2]:.4f}, y[{My2}]={grid.y[My2]:.4f}")
print(f"  Transverse domain ±{x_max:.2f} m ({x_max/delta1:.1f}δ₁, {x_max/delta2:.1f}δ₂)")
print(f"  z domain [{z_fine[0]:.1f}, {z_fine[-1]:.1f}] m")

# ── Find C000 native Ez on-axis nodes ────────────────────────────────────────
# For VED (dipole_comp=2), C000's native Ez sub-grid is type (0,0,1):
#   i_even, j_even, k_odd.
# On the z-axis x[Mx2]=0, y[My2]=0, so on-axis R-nodes are at (Mx2, My2, k_odd).
nat_c000 = _native_type_for_cluster_comp(C000, 2)   # should be (0, 0, 1)
print(f"\n  C000 native type for Ez: {nat_c000}")

Z_RECV_MIN = 0.5
Z_RECV_MAX = 7.5

z_eval_list, seq_c000_list = [], []
for seq, (i, j, k) in enumerate(grid.R_nodes):
    if (i == Mx2 and j == My2 and
            (i % 2, j % 2, k % 2) == nat_c000):
        zv = float(grid.z[k])
        if Z_RECV_MIN <= zv <= Z_RECV_MAX:
            z_eval_list.append(zv)
            seq_c000_list.append(seq)

z_eval   = np.array(z_eval_list)
seq_c000 = np.array(seq_c000_list, dtype=int)
order    = np.argsort(z_eval)
z_eval   = z_eval[order]
seq_c000 = seq_c000[order]
print(f"  C000 native Ez on-axis nodes in [{Z_RECV_MIN}, {Z_RECV_MAX}] m: "
      f"{len(z_eval)} nodes")
if len(z_eval) > 0:
    print(f"  z values: {z_eval.round(4)}")

# ── Analytic helper ───────────────────────────────────────────────────────────
def analytic_Ez_hom(z_obs: float, moment: float = 1.0) -> complex:
    """
    E_z at (0, 0, z_obs) from two half-dipoles at (0, 0, ±DZ).
    Matches C000's native VED source geometry: the source is split equally
    between the nearest k_odd nodes above and below z=0 (at z=±DZ).
    """
    E_p = electric_dipole_Ez_homogeneous_onaxis(z_obs, +DZ, SIGMA1, OMEGA, MU, EPS,
                                                 moment=0.5 * moment)
    E_m = electric_dipole_Ez_homogeneous_onaxis(z_obs, -DZ, SIGMA1, OMEGA, MU, EPS,
                                                 moment=0.5 * moment)
    return E_p + E_m


def analytic_Ez_two_layer(z_obs: float, moment: float = 1.0) -> complex:
    """
    E_z at (0, 0, z_obs) from two half-dipoles at (0, 0, ±DZ) in the
    two-layer model.  For z_obs > z_s (both half-dipoles), we add the two
    Sommerfeld integrals.  Both half-dipoles are in layer 1 (z_s=±DZ < z_c=4).
    """
    # Both source depths are in layer 1 (z_s = ±DZ << z_c = 4 m).
    E_p = electric_dipole_Ez_two_layer_onaxis(
        z_obs, +DZ, Z_CONT, SIGMA1, SIGMA2, OMEGA, MU, EPS, moment=0.5 * moment)
    E_m = electric_dipole_Ez_two_layer_onaxis(
        z_obs, -DZ, Z_CONT, SIGMA1, SIGMA2, OMEGA, MU, EPS, moment=0.5 * moment)
    return E_p + E_m


def lebedev_Ez_at(result: dict, z_obs_arr: np.ndarray) -> np.ndarray:
    """
    True Lebedev-averaged E_z at on-axis points (x=0, y=0, z_obs).

    C000: direct DOF read at the native on-axis seq indices (seq_c000).
    C101, C110, C011: interpolated via interpolate_cluster_E.
    """
    N_R = grid.N_R
    Ez_c000 = np.array([result["E_c"][C000][2 * N_R + s] for s in seq_c000])

    Ez_per_c = {C000: Ez_c000}
    for c in (C101, C110, C011):
        vals = np.array([
            interpolate_cluster_E(grid, result["E_c"][c], c, 2, 0.0, 0.0, zv)
            for zv in z_obs_arr
        ])
        Ez_per_c[c] = vals

    Ez_leb = np.mean(
        np.stack([Ez_per_c[c] for c in (C000, C101, C110, C011)], axis=0),
        axis=0,
    )
    return Ez_leb, Ez_per_c


def rms_rel_error(fd: np.ndarray, analytic: np.ndarray) -> float:
    """RMS relative error of Re(fd) vs Re(analytic)."""
    fd_re  = np.real(fd)
    an_re  = np.real(analytic)
    mask   = np.abs(an_re) > 0.01 * np.max(np.abs(an_re))
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean(((fd_re[mask] - an_re[mask]) / an_re[mask]) ** 2)))


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 1: Homogeneous sanity check (σ₁ = σ₂ = SIGMA1)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("Run 1: Homogeneous (σ = 0.1 S/m) — sanity check")
print("─" * 60)

media_hom = homogeneous_isotropic(grid, sigma=SIGMA1, mu=MU, eps=EPS)
solver_hom = LebedevMaxwellSolver(grid, media_hom, omega=OMEGA)
result_hom = solver_hom.solve(0.0, 0.0, Z_SRC, dipole_comp=2, moment=1.0)
print("  Solve complete.")

Ez_leb_hom, Ez_per_c_hom = lebedev_Ez_at(result_hom, z_eval)

# C000 analytic: two half-dipoles at ±DZ (matching C000 source geometry)
Ez_an_c000_hom = np.array([analytic_Ez_hom(zv, 1.0) for zv in z_eval])
# Lebedev analytic: single dipole at z=0 (after Lebedev averaging the two halves)
Ez_an_leb_hom  = np.array([
    electric_dipole_Ez_homogeneous_onaxis(zv, Z_SRC, SIGMA1, OMEGA, MU, EPS)
    for zv in z_eval
])

err_c000_hom = rms_rel_error(Ez_per_c_hom[C000], Ez_an_c000_hom)
err_leb_hom  = rms_rel_error(Ez_leb_hom,         Ez_an_leb_hom)

print(f"  C000-only  RMS relative error (Re E_z): {err_c000_hom * 100:.3f}%")
print(f"  Lebedev    RMS relative error (Re E_z): {err_leb_hom  * 100:.3f}%")

print(f"\n  {'z [m]':>6}  {'Ana (leb)':>13}  {'FD leb':>13}  {'ratio':>7}  {'err%':>7}")
for i, zv in enumerate(z_eval):
    a = np.real(Ez_an_leb_hom[i])
    f = np.real(Ez_leb_hom[i])
    ratio = f / a if abs(a) > 1e-30 else float("nan")
    err   = (f - a) / a * 100 if abs(a) > 1e-30 else float("nan")
    print(f"  {zv:>6.3f}  {a:>13.4e}  {f:>13.4e}  {ratio:>7.4f}  {err:>7.3f}%")

print(f"\n  Per-cluster ratios (FD/analytic Re(E_z)):")
print(f"  {'z [m]':>6}  {'C000':>7}  {'C101':>7}  {'C110':>7}  {'C011':>7}  {'Lebedev':>8}")
for i, zv in enumerate(z_eval):
    a = np.real(Ez_an_leb_hom[i])
    if abs(a) < 1e-30:
        continue
    rs = {c: np.real(Ez_per_c_hom[c][i]) / a for c in (C000, C101, C110, C011)}
    lr = np.real(Ez_leb_hom[i]) / a
    print(f"  {zv:>6.3f}  "
          + "  ".join(f"{rs[c]:>7.4f}" for c in (C000, C101, C110, C011))
          + f"  {lr:>8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 2: Two-layer (σ₁ = 0.1, σ₂ = 1.0 S/m, contact at z = 4 m)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print(f"Run 2: Two-layer  σ₁={SIGMA1} S/m, σ₂={SIGMA2} S/m, z_c={Z_CONT} m")
print("─" * 60)

media_lay = layered_isotropic(
    grid,
    layer_boundaries=[Z_CONT],
    sigma_values=[SIGMA1, SIGMA2],
    direction="z",
)
solver_lay = LebedevMaxwellSolver(grid, media_lay, omega=OMEGA)
result_lay = solver_lay.solve(0.0, 0.0, Z_SRC, dipole_comp=2, moment=1.0)
print("  Solve complete.")

Ez_leb_lay, Ez_per_c_lay = lebedev_Ez_at(result_lay, z_eval)

print("  Computing Sommerfeld analytic (this takes a few seconds)...")
# Lebedev analytic: single dipole at z=0 in two-layer model
Ez_an_leb_lay = np.array([
    electric_dipole_Ez_two_layer_onaxis(
        zv, Z_SRC, Z_CONT, SIGMA1, SIGMA2, OMEGA, MU, EPS
    )
    for zv in z_eval
])
# C000 analytic: two half-dipoles at ±DZ in two-layer model
Ez_an_c000_lay = np.array([analytic_Ez_two_layer(zv, 1.0) for zv in z_eval])
print("  Analytic done.")

# Separate into above/below contact
mask_L1 = z_eval < Z_CONT
mask_L2 = z_eval >= Z_CONT
err_leb_L1  = rms_rel_error(Ez_leb_lay[mask_L1], Ez_an_leb_lay[mask_L1])
err_leb_L2  = rms_rel_error(Ez_leb_lay[mask_L2], Ez_an_leb_lay[mask_L2])
err_leb_all = rms_rel_error(Ez_leb_lay, Ez_an_leb_lay)

print(f"\n  Two-layer Lebedev RMS relative error (Re E_z):")
print(f"    Layer 1 (z < {Z_CONT} m):  {err_leb_L1  * 100:.3f}%")
print(f"    Layer 2 (z ≥ {Z_CONT} m):  {err_leb_L2  * 100:.3f}%")
print(f"    Overall:              {err_leb_all * 100:.3f}%")

print(f"\n  {'z [m]':>6}  {'Layer':>6}  {'Ana Re':>13}  {'FD leb Re':>13}  "
      f"{'ratio':>7}  {'err%':>7}")
for i, zv in enumerate(z_eval):
    a    = np.real(Ez_an_leb_lay[i])
    f    = np.real(Ez_leb_lay[i])
    lyr  = "L2" if zv >= Z_CONT else "L1"
    ratio = f / a if abs(a) > 1e-30 else float("nan")
    err   = (f - a) / a * 100 if abs(a) > 1e-30 else float("nan")
    print(f"  {zv:>6.3f}  {lyr:>6}  {a:>13.4e}  {f:>13.4e}  {ratio:>7.4f}  {err:>7.3f}%")

print(f"\n  Per-cluster ratios (FD/analytic Re(E_z)):")
print(f"  {'z [m]':>6}  {'Lyr':>4}  {'C000':>7}  {'C101':>7}  {'C110':>7}  {'C011':>7}  {'Lebedev':>8}")
for i, zv in enumerate(z_eval):
    a = np.real(Ez_an_leb_lay[i])
    if abs(a) < 1e-30:
        continue
    rs = {c: np.real(Ez_per_c_lay[c][i]) / a for c in (C000, C101, C110, C011)}
    lr = np.real(Ez_leb_lay[i]) / a
    lyr = "L2" if zv >= Z_CONT else "L1"
    print(f"  {zv:>6.3f}  {lyr:>4}  "
          + "  ".join(f"{rs[c]:>7.4f}" for c in (C000, C101, C110, C011))
          + f"  {lr:>8.4f}")

# Contrast with homogeneous
print("\n  Effect of contact (analytic two-layer / analytic homo):")
print(f"  {'z [m]':>6}  {'Layer':>6}  {'Ana ratio':>11}  {'FD ratio':>10}")
for i, zv in enumerate(z_eval):
    lyr  = "L2" if zv >= Z_CONT else "L1"
    hom  = np.real(Ez_an_leb_hom[i])
    lay  = np.real(Ez_an_leb_lay[i])
    fd_hom = np.real(Ez_leb_hom[i])
    fd_lay = np.real(Ez_leb_lay[i])
    r_an = lay / hom if abs(hom) > 1e-30 else float("nan")
    r_fd = fd_lay / fd_hom if abs(fd_hom) > 1e-30 else float("nan")
    print(f"  {zv:>6.3f}  {lyr:>6}  {r_an:>11.4f}  {r_fd:>10.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    rf"Two-layer benchmark  (σ₁={SIGMA1}, σ₂={SIGMA2} S/m, z_c={Z_CONT} m, "
    rf"f={FREQ:.0f} Hz, VED source at z=0)"
    "\n"
    rf"Lebedev k={K_GRID} grid, DZ={DZ} m, piecewise-constant σ (no interface averaging)",
    fontsize=9,
)

# Panel 1: |E_z| vs z — homogeneous and two-layer, FD and analytic
ax = axes[0]
ax.semilogy(z_eval, np.abs(Ez_an_leb_hom), "b--",  lw=1.5, label="Homo analytic")
ax.semilogy(z_eval, np.abs(Ez_leb_hom),    "bs",   ms=6,   label="Homo FD (Lebedev)", alpha=0.8)
ax.semilogy(z_eval, np.abs(Ez_an_leb_lay), "r-",   lw=1.5, label="Two-layer analytic (Sommerfeld)")
ax.semilogy(z_eval, np.abs(Ez_leb_lay),    "ro",   ms=6,   label="Two-layer FD (Lebedev)", alpha=0.8)
ax.axvline(Z_CONT, color="gray", ls=":", lw=1.2, label=f"Contact z={Z_CONT} m")
ax.set_xlabel("z  [m]", fontsize=11)
ax.set_ylabel(r"$|E_z|$  [V/m]", fontsize=11)
ax.set_title("|E_z| on z-axis (Lebedev average)", fontsize=11)
ax.legend(fontsize=7.5)
ax.grid(True, which="both", alpha=0.2)

# Panel 2: % deviation from analytic
ax = axes[1]
dev_hom = (np.real(Ez_leb_hom) - np.real(Ez_an_leb_hom)) / np.abs(Ez_an_leb_hom) * 100
dev_lay = (np.real(Ez_leb_lay) - np.real(Ez_an_leb_lay)) / np.abs(Ez_an_leb_lay) * 100
ax.plot(z_eval, dev_hom, "bs-", ms=7, lw=1.5,
        label=f"Homo Lebedev (RMS {err_leb_hom*100:.2f}%)")
ax.plot(z_eval, dev_lay, "ro-", ms=7, lw=1.5,
        label=f"Two-layer Lebedev (RMS {err_leb_all*100:.2f}%)")
ax.axhline(0.0, color="black", lw=1.0, ls="--")
ax.axvline(Z_CONT, color="gray", ls=":", lw=1.2)
ax.set_xlabel("z  [m]", fontsize=11)
ax.set_ylabel(
    r"$(E_z^\mathrm{FD} - E_z^\mathrm{analytic})\,/\,|E_z^\mathrm{analytic}|$  [%]",
    fontsize=9,
)
ax.set_title("% deviation from analytic (Lebedev)", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 3: contact effect — analytic ratio two-layer / homogeneous
ax = axes[2]
ratio_an = np.real(Ez_an_leb_lay) / np.real(Ez_an_leb_hom)
ratio_fd = np.real(Ez_leb_lay) / np.real(Ez_leb_hom)
ax.plot(z_eval, ratio_an, "r-", lw=2, label="Analytic (Sommerfeld / closed-form)")
ax.plot(z_eval, ratio_fd, "ko--", ms=6, lw=1.5, label="FD: two-layer / homo (Lebedev)")
ax.axhline(1.0, color="gray", ls=":", lw=1.0)
ax.axvline(Z_CONT, color="gray", ls=":", lw=1.2, label=f"Contact z={Z_CONT} m")
ax.set_xlabel("z  [m]", fontsize=11)
ax.set_ylabel(r"$E_z^\mathrm{two-layer}\,/\,E_z^\mathrm{homo}$", fontsize=11)
ax.set_title("Contact effect (ratio to homogeneous)", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
outfile = os.path.join(os.path.dirname(__file__), "benchmark_two_layer.png")
plt.savefig(outfile, dpi=150)
print(f"\nPlot saved → {outfile}")

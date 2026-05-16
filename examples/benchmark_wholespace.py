"""
benchmark_wholespace.py — Homogeneous whole-space validation.

Validates the Lebedev 4-cluster FD scheme against the analytic solution
for an x-directed electric current dipole in a homogeneous whole-space.

Physical setup
--------------
Source  : x-directed unit electric current dipole at the grid origin.
Field   : E_x along the z-axis.
Medium  : σ = 1 S/m, f = 2500 Hz  →  skin depth δ ≈ 10 m.

Analytic formula on the z-axis (x=0, y=0) for an x-dipole at origin:
    E_x(0,0,z) = (p/4πσ̇) [ k²/r − 1/r³ + ik/r² ] exp(ikr)
where r=|z|, σ̇=σ−iωε, k=sqrt(iωμσ̇), Im(k)>0 (decaying convention).

Grid design — corrected DDH03 optimal grid
------------------------------------------
The DDH03 optimal grid interleaves primary nodes x₁,…,x_k and dual nodes
x̂₁,…,x̂_k on each half-axis, giving 2k+1 interleaved nodes per half-axis
and Mx = 4k in the full symmetric grid (DDH03 §Implementation):

    Primary spacings:  hᵢ = h_min·αⁱ⁻¹,  α = exp(γπ/√k),  γ = 1/√2
    Dual spacings:     ĥ₁ = h_min/(1+√α),  ĥᵢ = hᵢ/√α  (i ≥ 2)

k=3, h_min=0.5 m → 13 nodes per side (Mx=12), domain ≈ ±8.8 m ≈ 0.9δ.
The optimal grid + mixed (E/M) cluster BCs provide an accurate impedance
condition even at small domain sizes.

Source normalisation
--------------------
The FD source occupies two C101-native z-axis nodes at z = ±x̂₁ = ±h_min/(1+√α)
(each with weight 0.5).  For k=3, x̂₁ ≈ 0.172 m.
p_eff is calibrated empirically from C101's native z-axis DOF vs. the
two-dipole analytic (two half-dipoles at z=±x̂₁).

C101 is the only cluster with native Ex DOF on the z-axis (x=0, y=0 exactly).
The other clusters require transverse interpolation:

  C000 interpolates in x only (Δx ≈ ±x̂₁ ≈ 0.17 m) → small transverse error
  C011 interpolates in x and y (Δ ≈ ±x̂₁ in each)   → small transverse error
  C110 has small transverse y-offset (Δy ≈ ±x̂₁), but its z-grid (primary
       positions: 0, ±0.5, ±2.3 m) must interpolate to dual z-nodes between
       these primaries (e.g. z=1.12 m).  The 1/r³ near-field varies ~8×
       over the interval [0.5, 2.3] m, so C110's z-interpolation error remains
       ~6–8× at z ≈ 1.1 m.  This is a geometry limitation of the cube setup:
       the proper logging geometry uses a fine equidistant z-grid along the
       borehole, eliminating C110's z-interpolation problem.

What this benchmark demonstrates
---------------------------------
  1. C101 single-cluster accuracy is now much better than the old primary-only
     grid: the interleaved optimal grid has kh_eff << 1 at all interior nodes.
  2. C110's z-interpolation error persists for this cube geometry.
  3. The per-cluster DOF behaviour matches the analytic shape qualitatively.

Usage
-----
    python examples/benchmark_wholespace.py

Output: examples/benchmark_Ex_wholespace.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em.grid import (
    LebedevGrid3D, optimal_geometric_1d,
    C000, C101, C110, C011,
)
from lebedev_em.media import homogeneous_isotropic, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.analytics import electric_dipole_E
from lebedev_em.sources import _native_type_for_cluster_comp
from lebedev_em.postprocess import interpolate_cluster_E

# ── Physical parameters ───────────────────────────────────────────────────────
SIGMA = 1.0          # [S/m]
FREQ  = 2500.0       # [Hz]
OMEGA = 2.0 * np.pi * FREQ
MU    = MU0
EPS   = EPS0

print("=" * 65)
print("Lebedev FD — homogeneous whole-space benchmark")
print(f"  σ = {SIGMA} S/m,  f = {FREQ} Hz")
print("=" * 65)

delta = np.sqrt(2.0 / (OMEGA * MU * SIGMA))
print(f"\n  Skin depth δ ≈ {delta:.2f} m")

# ── Optimal geometric grid ─────────────────────────────────────────────────────
k       = 3              # DDH03 standard benchmark size: Mx=4k=12, 13 nodes/side
h_min   = 0.5          # [m] minimum primary step near source
GAMMA   = 1.0 / np.sqrt(2.0)

# optimal_geometric_1d now returns the full interleaved half-axis (2k+1 nodes):
#   [x̂_0=0, x̂_1, x_1, x̂_2, x_2, ..., x̂_k, x_k]
# The full symmetric grid has 4k+1 = 13 nodes, Mx = 4k = 12.
x_half = optimal_geometric_1d(k, h_min, 100.0, GAMMA)
x_full = np.concatenate([-x_half[::-1], x_half[1:]])  # 13 nodes, Mx=12
grid   = LebedevGrid3D(x_full, x_full, x_full)
Mx2    = grid.Mx // 2

h_min_actual = float(np.min(np.diff(x_full)))
x_max        = float(x_full[-1])

print(f"\n  Grid : {grid.Nx}³ nodes, h_min = {h_min_actual:.4f} m, domain = ±{x_max:.2f} m")
print(f"  Domain / δ = {x_max/delta:.2f},  h_min / δ = {h_min_actual/delta:.3f}")
print(f"  DOFs per cluster: {3 * grid.N_R} complex")

# ── Material & solver ─────────────────────────────────────────────────────────
media  = homogeneous_isotropic(grid, sigma=SIGMA, mu=MU, eps=EPS)
solver = LebedevMaxwellSolver(grid, media, omega=OMEGA)
print("\nAssembling and solving …")
result = solver.solve(x0=0.0, y0=0.0, z0=0.0, dipole_comp=0, moment=1.0)
print("  Solve complete.")

# ── C101 native z-axis DOF ────────────────────────────────────────────────────
# Native type for C101/Ex is (0,0,1):
#   x_sub = x[0::2] contains x[Mx//2] = 0  (Mx//2=6, even → tx=0 ✓)
#   y_sub = y[0::2] contains y[Mx//2] = 0  (ty=0 ✓)
#   z_sub = z[1::2] → native z-axis nodes.
#
nat101 = _native_type_for_cluster_comp(C101, 0)   # (0, 0, 1)
E_c101 = result["E_c"][C101]

zax_z, zax_Ex = [], []
for seq, (i, j, k_node) in enumerate(grid.R_nodes):
    if i == Mx2 and j == Mx2 and (i % 2, j % 2, k_node % 2) == nat101:
        z_val = float(grid.z[k_node])
        zax_z.append(z_val)
        zax_Ex.append(E_c101[0 * grid.N_R + seq])

zax_z  = np.array(zax_z)
zax_Ex = np.array(zax_Ex)
order  = np.argsort(zax_z)
zax_z  = zax_z[order]
zax_Ex = zax_Ex[order]

# Exclude source nodes (|z| = h_min) and the far-boundary node (|z| > 30m)
interior_mask = (np.abs(zax_z) > 0.3) & (np.abs(zax_z) < 30.0)
z_int  = zax_z[interior_mask]
Ex_int = zax_Ex[interior_mask]

# ── Calibrate p_eff from C101 native DOF ──────────────────────────────────────
# The FD source at (0,0,0) is split by trilinear interpolation onto two
# C101 native z-axis nodes at z = ±h_min (each with weight 0.5).  The correct
# analytic for this source geometry is TWO half-dipoles at z = ±h_min:
#
#   E_analytic(z_obs) = 0.5 × E_dipole(z_obs − h_min) + 0.5 × E_dipole(z_obs + h_min)
#
# Using a single origin dipole at z=0 introduces a ~10% error at z≈4.8m
# (dominant 1/r³ term makes the average of 1/(z±h_min)³ differ from 1/z³).
# This two-dipole analytic removes that geometric bias and makes p_eff
# interpretable as the true source normalisation constant of the FD scheme.

def two_dipole_Ex(z_obs: float, moment: float) -> complex:
    """E_x at (0,0,z_obs) from two half-dipoles of total moment at z=±h_min."""
    E_plus  = electric_dipole_E(0.0, 0.0, z_obs - h_min_actual,
                                 SIGMA, OMEGA, MU, EPS, 0, 0.5 * moment)[0]
    E_minus = electric_dipole_E(0.0, 0.0, z_obs + h_min_actual,
                                 SIGMA, OMEGA, MU, EPS, 0, 0.5 * moment)[0]
    return E_plus + E_minus

Ex_ana_per_z = np.array([two_dipole_Ex(zv, 1.0) for zv in z_int])

# Best-fit p_eff (real part, positive-z interior nodes only)
pos_mask = z_int > 0
p_eff = float(np.median(
    np.real(Ex_int[pos_mask]) / np.real(Ex_ana_per_z[pos_mask])
))
print(f"\n  Calibrated p_eff = {p_eff:.4f} A·m  "
      f"(two-dipole analytic, median over {pos_mask.sum()} interior z > 0 node(s))")

# ── Full analytic for comparison ──────────────────────────────────────────────
Ex_analytic_int = np.array([two_dipole_Ex(zv, p_eff) for zv in z_int])

# ── Per-cluster contributions at z_int (via interpolation) ───────────────────
#
# NOTE on interpretation:
#   C101  → native in x, y, z → direct DOF read; no interpolation
#   C011  → native in z, but interpolated ±0.5 m in x and y (small transverse)
#   C000  → native in x and y direction but needs z interpolation from coarse grid
#   C110  → needs both y and z interpolation; z spans steep near-field gradient
#           → gives badly overestimated contribution at z ≈ 4.8 m
#
labels  = {C000: "000", C101: "101", C110: "110", C011: "011"}
colors  = {C000: "#E63946", C101: "#F4A261", C110: "#2A9D8F", C011: "#8338EC"}
lstyles = {C000: "--",      C101: ":",        C110: "-.",      C011: (0, (3,1,1,1))}

Ex_per_c = {}
for c in (C000, C101, C110, C011):
    vals = np.array([
        interpolate_cluster_E(grid, result["E_c"][c], c, 0, 0.0, 0.0, zv)
        for zv in z_int
    ])
    Ex_per_c[c] = vals

Ex_leb = np.mean(
    np.stack([Ex_per_c[c] for c in (C000, C101, C110, C011)], axis=0),
    axis=0,
)

# ── Error statistics ──────────────────────────────────────────────────────────
def rms_rel_err(fd_vals, ana_vals, fn=np.real):
    fd, ana = fn(fd_vals), fn(ana_vals)
    mask = np.abs(ana) > 0.01 * np.max(np.abs(ana))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(((fd[mask] - ana[mask]) / ana[mask])**2)))

print(f"\n  RMS relative error of Re(Ex) on z-axis (interior, |z| in (0.3, 30) m):")
for c in (C000, C101, C110, C011):
    e = rms_rel_err(Ex_per_c[c], Ex_analytic_int)
    print(f"    Cluster {labels[c]} (interpolated to z-axis) : {e*100:7.2f}%")
e_leb = rms_rel_err(Ex_leb, Ex_analytic_int)
print(f"    Lebedev average (interpolated)            : {e_leb*100:7.2f}%")
# C110's Ex native type is (0,1,0) → k%2=0 → sits on even-indexed z nodes (primary positions).
# These are grid.z[0::2].  Show the first few positive ones.
c110_z_primary_pos = sorted(z for z in grid.z[0::2] if 0 < z < x_max / 2)[:4]
print(f"\n  ⚠  C110 z-interpolation: C110's Ex sits on primary z-nodes"
      f" {[f'{z:.3f}' for z in c110_z_primary_pos]} m.")
print(f"     Query points (C101 dual z-nodes) fall between these, spanning"
      f" steep 1/r³ gradients → large C110 z-interpolation overestimate.")
print(f"     Fix: use a fine equidistant z-grid along the borehole axis"
      f" (the standard DDH03 logging geometry).")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    rf"Homogeneous whole-space  (σ={SIGMA} S/m, f={FREQ/1e3:.1f} kHz, "
    rf"geometric grid {grid.Nx}³, h_min={h_min_actual:.3f} m, δ≈{delta:.1f} m)"
    "\n"
    rf"C101 calibrated p_eff = {p_eff:.3f} A·m.  "
    r"⚠ C110 z-interp spans steep near-field; Lebedev avg unreliable on z-axis here.",
    fontsize=9,
)

for ax_idx, (part_fn, part_name) in enumerate([(np.real, "Re"), (np.imag, "Im")]):
    ax = axes[ax_idx]

    # Per-cluster contributions
    for c in (C000, C101, C110, C011):
        ax.plot(z_int, part_fn(Ex_per_c[c]),
                linestyle=lstyles[c], color=colors[c],
                lw=1.6, alpha=0.85, label=f"Cluster {labels[c]}")

    # Lebedev average
    ax.plot(z_int, part_fn(Ex_leb), "-", color="navy",
            lw=2.5, zorder=5, label=f"Lebedev avg ({e_leb*100:.0f}% err)")

    # Analytic
    ax.plot(z_int, part_fn(Ex_analytic_int), "o", color="black",
            ms=6, zorder=10, markerfacecolor="white", markeredgewidth=1.5,
            label=f"Analytic (p={p_eff:.2f} A·m)")

    ax.set_xlabel("z  (m)", fontsize=11)
    ax.set_ylabel(rf"{part_name}$(E_x)$  (V/m)", fontsize=11)
    ax.set_title(f"{part_name} part", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)

plt.tight_layout()
outfile = os.path.join(os.path.dirname(__file__), "benchmark_Ex_wholespace.png")
plt.savefig(outfile, dpi=150)
print(f"\nPlot saved → {outfile}")

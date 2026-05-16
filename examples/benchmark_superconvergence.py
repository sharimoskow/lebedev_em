"""
benchmark_superconvergence.py — O(h²) cluster / O(h⁴) Lebedev convergence.

Demonstrates the DDH03 superconvergence algebraically as a classical h-refinement
study.  Key design choice:

  • k = 5  FIXED  (fixes the geometric ratio α = exp(π/√10), fixes relative grid
            shape, fixes domain extent relative to h_min)
  • h_min halved each run  (0.5 → 0.25 → 0.125 m)
  • dz = h_min / 2  PROPORTIONAL  (so transverse and axial FD errors both scale
            as O(h_min²))
  • z-domain ±6 m  (fixed physical extent; Nz grows as h_min shrinks)

Why fixed k and proportional dz?
---------------------------------
With k fixed, the SHAPE of the geometric grid is frozen (all spacings just
scale by h_min).  Every FD stencil spans ≈ 2 local spacings → FD truncation
error ∝ (local spacing)² ∝ h_min².  This gives each individual cluster a clean
O(h_min²) convergence rate.

The Lebedev 4-cluster average cancels the O(h_min²) leading error terms
(this is the W₁V₀=1/λ identity from DDH03 Sec. 4.2), leaving an O(h_min⁴)
residual.  Halving h_min should:
  • Reduce each cluster error by 4×  (second order)
  • Reduce the Lebedev error by 16× (fourth order)

Why fixed k keeps the domain large enough:
  x_max = h_min · Σᵢαⁱ  (α≈2.70 for k=5)
  x_max ≈ 227 · h_min  →  113 m, 56 m, 28 m for h_min = 0.5, 0.25, 0.125 m.
  Skin depth δ ≈ 10 m → all domains are 2.8–11 δ ≫ z_recv_max = 5 m.
  BC reflections are exponentially negligible (≪ 0.1 %) for all three runs,
  so cluster errors are PURELY FD-truncation, not BC artefacts.

Physical setup:
  Source: x-dipole at origin.
  Medium: σ = 1 S/m, f = 2500 Hz  →  δ ≈ 10 m.
  Receivers: z-axis (x=y=0), z ∈ (4, 5) m.
  Analytic: two half-dipoles at z = ±dz  (C101 source geometry for each run).

Usage:
  python examples/benchmark_superconvergence.py

Output: examples/benchmark_superconvergence.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em.grid import (
    LebedevGrid3D, symmetric_optimal_grid,
    C000, C101, C110, C011,
)
from lebedev_em.media import homogeneous_isotropic, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.analytics import electric_dipole_E
from lebedev_em.sources import _native_type_for_cluster_comp
from lebedev_em.postprocess import interpolate_cluster_E

# ── Physical parameters ────────────────────────────────────────────────────────
SIGMA = 1.0
FREQ  = 2500.0
OMEGA = 2.0 * np.pi * FREQ
MU    = MU0
EPS   = EPS0
delta = np.sqrt(2.0 / (OMEGA * MU * SIGMA))

print("=" * 70)
print("Lebedev FD — O(h²)/O(h⁴) superconvergence rate benchmark")
print(f"  σ = {SIGMA} S/m,  f = {FREQ:.0f} Hz,  δ ≈ {delta:.2f} m")
print("=" * 70)

# ── Fixed parameters ───────────────────────────────────────────────────────────
K       = 5           # fixed k; α = exp(π/√10) ≈ 2.699
GAMMA   = 1.0 / np.sqrt(2.0)
L_TRANS = 300.0       # large enough to never interfere

Z_HALF      = 6.0     # z-domain ±6 m (fixed physical extent)
Z_RECV_MIN  = 4.0     # receivers start well away from near-field
Z_RECV_MAX  = 5.0

# ── h_min sweep ────────────────────────────────────────────────────────────────
# Each entry: (h_min, dz = h_min/2)
H_MIN_VALS = [0.5, 0.25, 0.125]

labels = {C000: "C000", C101: "C101", C110: "C110", C011: "C011"}
colors = {C000: "#E63946", C101: "#F4A261", C110: "#2A9D8F", C011: "#8338EC"}


def two_dipole_Ex(z_obs, h_z_src, moment=1.0):
    """E_x at (0,0,z_obs) from two half-dipoles at (0,0,±h_z_src)."""
    E_p = electric_dipole_E(0.0, 0.0, z_obs - h_z_src,
                             SIGMA, OMEGA, MU, EPS, 0, 0.5 * moment)[0]
    E_m = electric_dipole_E(0.0, 0.0, z_obs + h_z_src,
                             SIGMA, OMEGA, MU, EPS, 0, 0.5 * moment)[0]
    return E_p + E_m


def rms_rel(fd_vals, ana_vals, fn=np.real):
    fd, ana = fn(np.asarray(fd_vals)), fn(np.asarray(ana_vals))
    mask = np.abs(ana) > 0.01 * np.max(np.abs(ana))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(((fd[mask] - ana[mask]) / ana[mask]) ** 2)))


results = []

for h_min in H_MIN_VALS:
    dz   = h_min / 2.0
    Nz   = int(round(2.0 * Z_HALF / dz)) + 1
    z_ax = np.linspace(-Z_HALF, Z_HALF, Nz)

    print(f"\n{'─'*65}")
    print(f"  h_min = {h_min:.4f} m,  dz = {dz:.5f} m,  Nz = {Nz}")

    grid = symmetric_optimal_grid(h_min, L_TRANS, z_ax, GAMMA, k=K)
    Mx2  = grid.Mx // 2
    x_hat1 = float(grid.x[Mx2 + 1])
    x_max  = float(grid.x[-1])

    print(f"  x̂₁ = {x_hat1:.5f} m,  x_max = ±{x_max:.2f} m ({x_max/delta:.2f}δ)")
    print(f"  N_R = {grid.N_R}")

    # Solve
    media  = homogeneous_isotropic(grid, sigma=SIGMA, mu=MU, eps=EPS)
    solver = LebedevMaxwellSolver(grid, media, omega=OMEGA)
    result = solver.solve(x0=0.0, y0=0.0, z0=0.0, dipole_comp=0, moment=1.0)
    print("  Solve complete.")

    # ── Locate C101 native Ex DOFs on z-axis in receiver range ────────────────
    nat101 = _native_type_for_cluster_comp(C101, 0)   # (0,0,1) for Ex
    E_c101 = result["E_c"][C101]

    z_list, Ex101_list = [], []
    for seq, (i, j, k_node) in enumerate(grid.R_nodes):
        if (i == Mx2 and j == Mx2 and
                (i % 2, j % 2, k_node % 2) == nat101):
            z_val = float(grid.z[k_node])
            if Z_RECV_MIN < z_val < Z_RECV_MAX:
                z_list.append(z_val)
                Ex101_list.append(E_c101[0 * grid.N_R + seq])

    if len(z_list) == 0:
        print("  ⚠ No C101 native nodes in receiver range — skipping.")
        results.append(None)
        continue

    z_eval  = np.array(z_list)
    Ex_c101 = np.array(Ex101_list)
    order   = np.argsort(z_eval)
    z_eval  = z_eval[order]
    Ex_c101 = Ex_c101[order]
    print(f"  {len(z_eval)} C101 nodes at z = {np.round(z_eval, 4)}")

    # ── Analytic (two half-dipoles at ±dz, matching C101 source geometry) ─────
    ana = np.array([two_dipole_Ex(zv, dz, 1.0) for zv in z_eval])

    # ── All four cluster fields at (0,0,z_eval) ────────────────────────────────
    Ex_per_c = {C101: Ex_c101}
    for c in (C000, C110, C011):
        vals = np.array([
            interpolate_cluster_E(grid, result["E_c"][c], c, 0, 0.0, 0.0, zv)
            for zv in z_eval
        ])
        Ex_per_c[c] = vals

    Ex_leb = np.mean(
        np.stack([Ex_per_c[c] for c in (C000, C101, C110, C011)], axis=0),
        axis=0,
    )

    # ── Errors ─────────────────────────────────────────────────────────────────
    errs     = {c: rms_rel(Ex_per_c[c], ana) for c in (C000, C101, C110, C011)}
    err_avg  = float(np.mean([errs[c] for c in (C000, C101, C110, C011)]))
    err_geom = float(np.exp(np.mean([np.log(errs[c]) for c in (C000, C101, C110, C011)])))
    err_leb  = rms_rel(Ex_leb, ana)
    p_leb    = float(np.median(np.real(Ex_leb) / np.real(ana)))

    print(f"\n  RMS relative error vs analytic:")
    for c in (C000, C101, C110, C011):
        print(f"    {labels[c]}: {errs[c]*100:8.4f}%")
    print(f"    Arith avg:    {err_avg*100:8.4f}%")
    print(f"    Geom  avg:    {err_geom*100:8.4f}%")
    print(f"    Lebedev:      {err_leb*100:8.4f}%")
    print(f"    p_leb:        {p_leb:.5f}")
    if err_geom > 1e-10:
        ratio = err_leb / err_geom**2
        print(f"    Leb / (geom avg)²: {ratio:.3f}  "
              f"(theory ≈ 1/(2√λ) = {1/(2*np.sqrt(np.sqrt(np.exp(np.pi/np.sqrt(10))))):.3f})")

    results.append({
        "h_min":   h_min,
        "dz":      dz,
        "Nz":      Nz,
        "N_R":     grid.N_R,
        "x_max":   x_max,
        "errs":    errs,
        "err_avg": err_avg,
        "err_geom": err_geom,
        "err_leb": err_leb,
        "p_leb":   p_leb,
        "z_eval":  z_eval,
        "Ex_per_c": Ex_per_c,
        "Ex_leb":  Ex_leb,
        "ana":     ana,
    })

# ── Summary ────────────────────────────────────────────────────────────────────
valid = [r for r in results if r is not None]

print("\n" + "=" * 70)
print("CONVERGENCE SUMMARY")
print(f"  k = {K},  α ≈ {np.exp(np.pi/np.sqrt(10)):.4f}")
hdr = (f"  {'h_min':>7}  {'dz':>7}  {'N_R':>7}  "
       f"{'C101%':>8}  {'C000%':>8}  {'C110%':>8}  {'C011%':>8}  "
       f"{'Leb%':>8}  {'Leb/gavg²':>10}")
print(hdr)
for r in valid:
    print(f"  {r['h_min']:>7.4f}  {r['dz']:>7.5f}  {r['N_R']:>7}  "
          f"{r['errs'][C101]*100:>8.4f}  {r['errs'][C000]*100:>8.4f}  "
          f"{r['errs'][C110]*100:>8.4f}  {r['errs'][C011]*100:>8.4f}  "
          f"{r['err_leb']*100:>8.4f}  "
          f"{r['err_leb']/r['err_geom']**2:>10.3f}")

if len(valid) >= 2:
    print("\nConvergence rates (successive halvings of h_min):")
    print(f"  {'h_min pair':>18}  {'cluster rate':>14}  {'Lebedev rate':>14}  "
          f"{'expected cluster':>16}  {'expected Leb':>14}")
    for i in range(1, len(valid)):
        r0, r1 = valid[i-1], valid[i]
        ratio_h = r0["h_min"] / r1["h_min"]
        rate_cl = np.log(r0["err_geom"] / r1["err_geom"]) / np.log(ratio_h)
        rate_lb = np.log(r0["err_leb"]  / r1["err_leb"])  / np.log(ratio_h)
        print(f"  {r0['h_min']:.4f} → {r1['h_min']:.4f}    "
              f"p = {rate_cl:>6.3f}            p = {rate_lb:>6.3f}            "
              f"2.000 (O(h²))          4.000 (O(h⁴))")

# ── Plot ───────────────────────────────────────────────────────────────────────
if len(valid) < 2:
    print("Need at least 2 valid runs for plot.")
    sys.exit(0)

h_arr   = np.array([r["h_min"]    for r in valid])
err_c101 = np.array([r["errs"][C101] for r in valid])
err_c000 = np.array([r["errs"][C000] for r in valid])
err_c110 = np.array([r["errs"][C110] for r in valid])
err_c011 = np.array([r["errs"][C011] for r in valid])
err_gavg = np.array([r["err_geom"]   for r in valid])
err_leb  = np.array([r["err_leb"]    for r in valid])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    rf"DDH03 Lebedev superconvergence: O($h^2$) clusters vs O($h^4$) average  "
    rf"(k={K}, σ={SIGMA} S/m, f={FREQ/1e3:.1f} kHz, δ≈{delta:.1f} m)"
    "\n"
    r"$dz = h_{\min}/2$ (proportional), z-domain $\pm$"
    f"{Z_HALF:.0f} m, receivers z $\in$ ({Z_RECV_MIN},{Z_RECV_MAX}) m",
    fontsize=9,
)

# ── Left: log-log convergence plot ────────────────────────────────────────────
ax = axes[0]

ax.loglog(h_arr, err_c101 * 100, "o-", color=colors[C101], lw=2, ms=9,
          label=f"C101  ({labels[C101]})")
ax.loglog(h_arr, err_c000 * 100, "v:", color=colors[C000], lw=1.5, ms=7, alpha=0.8,
          label=f"C000")
ax.loglog(h_arr, err_c110 * 100, ">:", color=colors[C110], lw=1.5, ms=7, alpha=0.8,
          label=f"C110")
ax.loglog(h_arr, err_c011 * 100, "<:", color=colors[C011], lw=1.5, ms=7, alpha=0.8,
          label=f"C011")
ax.loglog(h_arr, err_gavg * 100, "D-", color="#555555", lw=2, ms=9,
          label="Geom avg (4 clusters)")
ax.loglog(h_arr, err_leb * 100,  "s-", color="navy",    lw=2.5, ms=9, zorder=10,
          label="Lebedev average")

# O(h²) and O(h⁴) reference lines anchored at h_min=0.5
h_ref = np.array([min(h_arr)*0.8, max(h_arr)*1.3])
C2 = err_gavg[0] / h_arr[0]**2
C4 = err_leb[0]  / h_arr[0]**4
ax.loglog(h_ref, C2 * h_ref**2 * 100, "k--", lw=1.2, alpha=0.5, label=r"$O(h^2)$ ref")
ax.loglog(h_ref, C4 * h_ref**4 * 100, "k-.", lw=1.2, alpha=0.5, label=r"$O(h^4)$ ref")

ax.set_xlabel(r"$h_{\min}$  [m]", fontsize=12)
ax.set_ylabel("RMS relative error of Re$(E_x)$  [%]", fontsize=11)
ax.set_title("Convergence rate  (log–log)", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, which="both", alpha=0.25)

# Annotate measured slopes
if len(valid) >= 2:
    for i in range(1, len(valid)):
        r0, r1 = valid[i-1], valid[i]
        ratio_h = r0["h_min"] / r1["h_min"]
        rate_cl = np.log(r0["err_geom"] / r1["err_geom"]) / np.log(ratio_h)
        rate_lb = np.log(r0["err_leb"]  / r1["err_leb"])  / np.log(ratio_h)
        hm = np.sqrt(r0["h_min"] * r1["h_min"])
        em = np.sqrt(r0["err_geom"] * r1["err_geom"]) * 100
        el = np.sqrt(r0["err_leb"]  * r1["err_leb"])  * 100
        ax.annotate(f"p={rate_cl:.2f}", xy=(hm, em), fontsize=8,
                    color="#555555", ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")
        ax.annotate(f"p={rate_lb:.2f}", xy=(hm, el), fontsize=8,
                    color="navy", ha="center", va="top",
                    xytext=(0, -8), textcoords="offset points")

# ── Right: superconvergence ratio Leb / (geom avg)² ──────────────────────────
ax2 = axes[1]
ratio_arr = err_leb / err_gavg**2
lambda_val = np.sqrt(np.exp(np.pi / np.sqrt(10)))   # √α for k=5
theory_val = 1.0 / (2.0 * np.sqrt(lambda_val))

ax2.semilogx(h_arr, ratio_arr, "s-", color="navy", lw=2, ms=10,
             label=r"$\epsilon_\mathrm{Leb} / \bar\epsilon^2$  (measured)")
ax2.axhline(theory_val, color="crimson", lw=1.5, ls="--",
            label=rf"DDH03 theory: $1/(2\sqrt{{\lambda}})$ = {theory_val:.3f}")
ax2.axhline(0.0, color="gray", lw=0.8, ls=":")

ax2.set_xlabel(r"$h_{\min}$  [m]", fontsize=12)
ax2.set_ylabel(r"$\epsilon_\mathrm{Leb}\ /\ \bar\epsilon_\mathrm{cluster}^2$",
               fontsize=12)
ax2.set_title(
    r"Superconvergence ratio: $\epsilon_\mathrm{Leb} \approx \bar\epsilon^2 / (2\sqrt{\lambda})$",
    fontsize=10,
)
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.25)

# Add text box with p_leb values
info_lines = [f"h_min={r['h_min']:.4f}:  p_leb={r['p_leb']:.5f}" for r in valid]
ax2.text(0.02, 0.05, "\n".join(info_lines), transform=ax2.transAxes,
         fontsize=8, va="bottom", family="monospace",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
outfile = os.path.join(os.path.dirname(__file__), "benchmark_superconvergence.png")
plt.savefig(outfile, dpi=150)
print(f"\nPlot saved → {outfile}")

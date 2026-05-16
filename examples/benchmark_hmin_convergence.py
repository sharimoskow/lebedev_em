"""
benchmark_hmin_convergence.py — DDH03 superconvergence rate: O(h²) vs O(h⁴).

Varies h_min = 2.0, 1.0, 0.5, 0.25 m at fixed k = 5 (large transverse domain)
with proportional axial refinement dz = h_min / 2.  All grid spacings refine
together, so the convergence is truly isotropic and we can measure the power-law
exponent of the error vs h_min.

Physical prediction (DDH03 Theorem 1)
--------------------------------------
  Individual clusters : error ∝ h_min²   (standard 2nd-order FD)
  Lebedev average     : error ∝ h_min⁴   (superconvergence, quadratic improvement)

The Lebedev superconvergence arises because the four clusters' leading-order
truncation errors are equal and opposite — they cancel in the average, leaving
only the next-order term.

Grid design
-----------
  Transverse (x, y): optimal geometric grid, k = 5 fixed, Mx = My = 20.
    x_max ∝ h_min × Σ(geometric ratios) — domain shrinks with h_min but always
    stays ≫ z_recv for h_min ≤ 2 m, keeping BC reflections negligible.
  Axial (z): uniform, dz = h_min / 2, domain ±L_Z = ±10 m.
    C101 source nodes land at z = ±dz (nearest dual nodes to z = 0).

Evaluation
----------
  Receivers: C101 native z-axis DOF in z ∈ (3.5, 8.5 m).
    – C101  : direct node read — no interpolation error.
    – C000  : parity midpoint in x (average over ±x̂₁); error O((x̂₁/r)²) ∝ h_min².
    – C110  : parity midpoint in y (average over ±x̂₁); same.
    – C011  : parity midpoint in x and y; error O((x̂₁/r)⁴) ∝ h_min⁴ (tiny).
  The parity midpoint errors scale as h_min² — the same order as individual-
  cluster FD truncation — and therefore do not obscure the convergence slopes.
  For the Lebedev, the FD cancellation reduces the O(h_min²) term; whether we
  see a clear h_min⁴ slope depends on whether the axial FD error (also O(dz²)
  = O(h_min²) but with a small coefficient) creates a floor before h_min = 0.25 m.

Reference
---------
  Davydycheva, Druskin & Habashy (2003), Geophysics 68(5):1525–1536.

Usage
-----
    python examples/benchmark_hmin_convergence.py

Output: examples/benchmark_hmin_convergence.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em.grid import (
    LebedevGrid3D, optimal_geometric_1d, symmetric_optimal_grid,
    C000, C101, C110, C011,
)
from lebedev_em.media import homogeneous_isotropic, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.analytics import electric_dipole_E
from lebedev_em.sources import _native_type_for_cluster_comp
from lebedev_em.postprocess import interpolate_cluster_E

# ── Physical parameters ───────────────────────────────────────────────────────
SIGMA = 1.0
FREQ  = 2500.0
OMEGA = 2.0 * np.pi * FREQ
MU    = MU0
EPS   = EPS0

delta = np.sqrt(2.0 / (OMEGA * MU * SIGMA))
k_abs = np.sqrt(2.0) / delta   # |k_EM| [m⁻¹]

print("=" * 70)
print("Lebedev FD — h_min convergence  (O(h²) vs O(h⁴))")
print(f"  σ = {SIGMA} S/m,  f = {FREQ:.0f} Hz,  δ ≈ {delta:.2f} m,  |k| = {k_abs:.4f} m⁻¹")
print("=" * 70)

# ── Sweep parameters ──────────────────────────────────────────────────────────
H_MIN_VALS = [2.0, 1.0, 0.5, 0.25]   # [m]  halved each step
K          = 5                         # fixed transverse geometric steps (Mx=20)
GAMMA      = 1.0 / np.sqrt(2.0)
L_TRANS    = 1e6                       # effectively unlimited (k clips naturally)
L_Z        = 10.0                      # [m] axial domain half-length

Z_RECV_MIN = 3.5   # [m]  far-field evaluation window
Z_RECV_MAX = 8.5   # [m]

# ── Helper: two-dipole analytic (matches C101 source placement) ───────────────
def two_dipole_Ex(z_obs: float, h_z_src: float, moment: float = 1.0) -> complex:
    """E_x at (0,0,z_obs) from two half-dipoles at (0,0,±h_z_src)."""
    Ep = electric_dipole_E(0.0, 0.0, z_obs - h_z_src,
                            SIGMA, OMEGA, MU, EPS, 0, 0.5 * moment)[0]
    Em = electric_dipole_E(0.0, 0.0, z_obs + h_z_src,
                            SIGMA, OMEGA, MU, EPS, 0, 0.5 * moment)[0]
    return Ep + Em


def rms_rel(fd_vals, ana_vals, fn=np.real):
    fd, ana = fn(np.asarray(fd_vals)), fn(np.asarray(ana_vals))
    mask = np.abs(ana) > 0.01 * np.max(np.abs(ana))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(((fd[mask] - ana[mask]) / ana[mask]) ** 2)))


labels = {C000: "C000", C101: "C101", C110: "C110", C011: "C011"}
colors = {C000: "#E63946", C101: "#F4A261", C110: "#2A9D8F", C011: "#8338EC"}

results = []

for h_min in H_MIN_VALS:
    dz      = h_min / 2.0
    H_Z_SRC = dz            # C101 source nodes split to z = ±dz

    # Build fine axial grid (uniform, symmetric about origin)
    n_half = int(round(L_Z / dz))
    Nz = 2 * n_half + 1       # always odd → Mz = Nz-1 even ✓
    z_fine = np.linspace(-L_Z, L_Z, Nz)

    # Build logging-geometry grid: optimal (x,y) + fine uniform z
    grid  = symmetric_optimal_grid(h_min, L_TRANS, z_fine, GAMMA, k=K)
    Mx2   = grid.Mx // 2
    x_hat1 = float(grid.x[Mx2 + 1])
    x_max  = float(grid.x[-1])

    print(f"\n{'─'*65}")
    print(f"  h_min = {h_min:.3f} m,  dz = {dz:.4f} m,  |k|·dz = {k_abs*dz:.4f}")
    print(f"  Nz = {Nz},  k = {K},  x̂₁ = {x_hat1:.4f} m,  x_max = {x_max:.1f} m "
          f"({x_max/delta:.2f}δ)")
    print(f"  N_R = {grid.N_R}")

    if x_max < Z_RECV_MIN:
        print("  ⚠ Domain too small for far-field evaluation — skipping.")
        results.append(None)
        continue

    # Solve
    media  = homogeneous_isotropic(grid, sigma=SIGMA, mu=MU, eps=EPS)
    solver = LebedevMaxwellSolver(grid, media, omega=OMEGA)
    result = solver.solve(x0=0.0, y0=0.0, z0=0.0, dipole_comp=0, moment=1.0)
    print("  Solve complete.")

    # ── C101 native z-axis DOF ────────────────────────────────────────────────
    nat101  = _native_type_for_cluster_comp(C101, 0)   # (0, 0, 1)
    E_c101  = result["E_c"][C101]

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

    # ── Analytic at each C101 node position ───────────────────────────────────
    ana_vals = np.array([two_dipole_Ex(zv, H_Z_SRC) for zv in z_eval])

    # ── All-cluster evaluation (C101 direct; others via midpoint-parity interp) ─
    Ex_per_c: dict[int, np.ndarray] = {}
    for c in (C000, C101, C110, C011):
        if c == C101:
            Ex_per_c[c] = Ex_c101.copy()
        else:
            vals = np.array([
                interpolate_cluster_E(grid, result["E_c"][c], c, 0, 0.0, 0.0, zv)
                for zv in z_eval
            ])
            Ex_per_c[c] = vals

    Ex_leb = np.mean(np.stack(list(Ex_per_c.values()), axis=0), axis=0)

    # ── p_leb: normalisation check ────────────────────────────────────────────
    pos_mask = z_eval > 0
    p_leb = float(np.median(
        np.real(Ex_leb[pos_mask]) / np.real(ana_vals[pos_mask])
    )) if pos_mask.sum() > 0 else np.nan

    # ── Per-cluster RMS errors ────────────────────────────────────────────────
    err = {c: rms_rel(Ex_per_c[c], ana_vals) for c in (C000, C101, C110, C011)}
    err_leb = rms_rel(Ex_leb, ana_vals)

    print(f"\n  n_recv = {len(z_eval)},  p_leb = {p_leb:.5f}")
    print(f"  RMS relative error vs two-dipole analytic [Re(Ex)]:")
    for c in (C000, C101, C110, C011):
        print(f"    {labels[c]}: {err[c]*100:8.3f}%")
    print(f"    Lebedev: {err_leb*100:8.3f}%")
    if not np.isnan(err[C101]) and err[C101] > 1e-12:
        ratio = err_leb / err[C101]**2
        print(f"    Leb / C101² = {ratio:.3f}  (≈1 expected for true h⁴ behaviour)")

    results.append({
        "h_min":    h_min,
        "dz":       dz,
        "H_Z_SRC":  H_Z_SRC,
        "x_hat1":   x_hat1,
        "x_max":    x_max,
        "N_R":      grid.N_R,
        "n_recv":   len(z_eval),
        "z_eval":   z_eval,
        "p_leb":    p_leb,
        "err":      err,
        "err_leb":  err_leb,
        "Ex_per_c": Ex_per_c,
        "Ex_leb":   Ex_leb,
        "ana_vals": ana_vals,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
valid = [r for r in results if r is not None]

print("\n" + "=" * 80)
print(f"SUMMARY  (k={K} fixed, dz = h_min/2, z-axis receivers)")
hdr = (f"  {'h_min':>6}  {'dz':>6}  {'N_R':>8}  {'p_leb':>8}  "
       f"{'C101%':>8}  {'C000%':>8}  {'C110%':>8}  {'C011%':>8}  {'Leb%':>8}")
print(hdr)
for r in valid:
    print(f"  {r['h_min']:>6.3f}  {r['dz']:>6.4f}  {r['N_R']:>8}  {r['p_leb']:>8.5f}  "
          f"{r['err'][C101]*100:>8.3f}  {r['err'][C000]*100:>8.3f}  "
          f"{r['err'][C110]*100:>8.3f}  {r['err'][C011]*100:>8.3f}  "
          f"{r['err_leb']*100:>8.3f}")

# ── Empirical convergence slopes ──────────────────────────────────────────────
if len(valid) >= 2:
    h_arr  = np.array([r["h_min"] for r in valid])
    print(f"\n  Empirical slopes  d(log error)/d(log h_min):")
    for c in (C000, C101, C110, C011):
        errs = np.array([r["err"][c] for r in valid])
        good = ~np.isnan(errs) & (errs > 0)
        if good.sum() >= 2:
            slopes = np.diff(np.log(errs[good])) / np.diff(np.log(h_arr[good]))
            print(f"    {labels[c]}: " + "  ".join(f"{s:+.2f}" for s in slopes))
    errs_leb = np.array([r["err_leb"] for r in valid])
    good = ~np.isnan(errs_leb) & (errs_leb > 0)
    if good.sum() >= 2:
        slopes = np.diff(np.log(errs_leb[good])) / np.diff(np.log(h_arr[good]))
        print(f"    Lebedev: " + "  ".join(f"{s:+.2f}" for s in slopes))
    print()
    print("  Interpretation:")
    print("    slope ≈ +2.0  → O(h²)  individual-cluster FD truncation")
    print("    slope ≈ +4.0  → O(h⁴)  Lebedev superconvergence (DDH03 Theorem 1)")
    print("    slope ≈ +2.0 for Lebedev → axial FD error (O(dz²)=O(h²)) dominates;")
    print("      refine dz further to see the h⁴ regime.")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    rf"DDH03 Lebedev convergence rate  "
    rf"(σ={SIGMA} S/m, f={FREQ:.0f} Hz, δ≈{delta:.0f} m,  k={K} fixed, dz=h_min/2)"
    "\n"
    rf"Receivers: z-axis ∈ ({Z_RECV_MIN}, {Z_RECV_MAX}) m,  true unit-moment analytic",
    fontsize=9,
)

h_arr    = np.array([r["h_min"] for r in valid])
err_c    = {c: np.array([r["err"][c]  for r in valid]) for c in (C000, C101, C110, C011)}
err_leb  = np.array([r["err_leb"] for r in valid])

# ── Left: log-log convergence ─────────────────────────────────────────────────
ax = axes[0]
for c in (C000, C101, C110, C011):
    ax.loglog(h_arr, err_c[c] * 100, "o-", color=colors[c],
              lw=1.8, ms=8, label=labels[c])
ax.loglog(h_arr, err_leb * 100, "s-", color="navy",
          lw=2.5, ms=10, label="Lebedev avg")

# Reference slopes anchored at the coarsest resolution
h_ref  = h_arr[0]
h_line = np.array([h_arr[-1] * 0.8, h_arr[0] * 1.2])

# O(h²) through median individual-cluster error at coarsest h_min
e2_ref = float(np.median([err_c[c][0] for c in (C000, C101, C110, C011)])) * 100
ax.loglog(h_line, e2_ref * (h_line / h_ref) ** 2,
          "k--", lw=1.2, alpha=0.55, label=r"$O(h^2)$  reference")

# O(h⁴) through Lebedev error at coarsest h_min
e4_ref = err_leb[0] * 100
ax.loglog(h_line, e4_ref * (h_line / h_ref) ** 4,
          "k:", lw=1.2, alpha=0.55, label=r"$O(h^4)$  reference")

ax.set_xlabel(r"$h_{\min}$  [m]", fontsize=11)
ax.set_ylabel(r"RMS relative error of Re$(E_x)$  [%]", fontsize=11)
ax.set_title(
    f"Convergence rate  (k={K}, dz=h_min/2)\n"
    r"Predicted: clusters $O(h^2)$, Lebedev $O(h^4)$",
    fontsize=10,
)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.25)

# ── Right: field profile at finest h_min ─────────────────────────────────────
if valid:
    r  = valid[-1]
    ax = axes[1]
    z_p = r["z_eval"]
    ana = r["ana_vals"]

    for c in (C000, C101, C110, C011):
        ax.plot(z_p, np.real(r["Ex_per_c"][c]),
                "o-", color=colors[c], lw=1.5, ms=7, alpha=0.85, label=labels[c])
    ax.plot(z_p, np.real(r["Ex_leb"]),
            "s-", color="navy", lw=2.5, ms=8, zorder=5, label="Lebedev")
    ax.plot(z_p, np.real(ana),
            "kD", ms=8, mfc="white", mew=1.5, zorder=10,
            label=f"Analytic (two-dipole, dz={r['dz']:.3f} m)")

    ax.set_xlabel("z  [m]", fontsize=11)
    ax.set_ylabel(r"Re$(E_x)$  [V/m]", fontsize=11)
    ax.set_title(
        f"Field profile at h_min = {r['h_min']:.3f} m  (dz = {r['dz']:.3f} m)\n"
        f"Lebedev error = {r['err_leb']*100:.3f}%",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
outfile = os.path.join(os.path.dirname(__file__), "benchmark_hmin_convergence.png")
plt.savefig(outfile, dpi=150)
print(f"\nPlot saved → {outfile}")

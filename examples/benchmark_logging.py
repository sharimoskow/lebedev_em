"""
benchmark_logging.py — DDH03 superconvergence in the logging geometry.

Demonstrates the central theoretical result of the Lebedev 4-cluster scheme:

    Lebedev average error  ≪  any single-cluster error

i.e. the four-cluster average converges to the analytic field much more
accurately than any individual cluster solution, due to systematic cancellation
of leading-order boundary-condition artefacts between clusters.

Physical setup
--------------
Source  : x-directed unit electric current dipole at the grid origin.
Medium  : σ = 1 S/m, f = 2500 Hz  →  skin depth δ ≈ 10 m.
Receiver: z-axis, x = y = 0, z ∈ (4, 7.5 m).
Analytic: two half-dipoles at z = ±dz = ±0.25 m, matching C101 source geometry.

Z-grid: DDH03 hybrid axial grid (Fig. 5/6)
-------------------------------------------
The z-axis uses the hybrid construction from DDH03: equidistant spacing dz in
the inner zone (covering source and all receivers), and optimal geometric
stretching outside.  This pushes the Dirichlet BC ≈ 11 δ away from the
receivers, making z-BC reflections < 10⁻⁹ — completely negligible.

Previous implementations using a simple equidistant ±10 m z-grid had z-BC
reflections of order exp(−1) ≈ 37 % in amplitude, which contaminated all four
clusters equally and produced misleadingly large cluster errors (~12 %) that
looked like transverse FD truncation but were actually z-BC artefacts.

Why x = y = 0 (z-axis)?
--------------------------
For an x-dipole, Ex is EVEN in both x and y.  The three clusters other than
C101 each interpolate to (0,0,z) via a midpoint average of their native nodes:
  • C000 averages x = ±x̂₁   (midpoint error O((x̂₁/r)²) < 0.4 %)
  • C110 averages y = ±x̂₁   (same, by symmetry)
  • C011 averages both        (error O((x̂₁/r)⁴) < 0.001 %)
These are negligible at the receiver distances used here.

Source normalisation (cell-volume correction)
----------------------------------------------
Each cluster's source is deposited at R-nodes of its native sub-grid.  These
nodes sit on dual cells of different volumes, so without correction cluster
C011 injects ~2× more total dipole moment than C101.  The `sources.py`
build_source_rhs function divides each nodal RHS value by the skip-2 dual-cell
volume, giving all four clusters the same effective moment.  After this fix,
p_leb ≈ 1.000–1.007 for all k.

Results (far-field, z ∈ 4–7.5 m, hybrid z-grid, no p_eff calibration)
-----------------------------------------------------------------------
k=3 (transverse domain ±8.8 m ≈ 0.87δ):
  C101 = 7.4 %, C000 = 9.6 %, C110 = 8.1 %, C011 = 10.5 %,  Lebedev = 0.72 %
k=4 (transverse domain ±20.6 m ≈ 2.1δ):
  C101 = 1.9 %, C000 = 2.0 %, C110 = 2.1 %, C011 = 1.8 %,  Lebedev = 0.86 %
k=5 (transverse domain ±41.9 m ≈ 4.2δ):
  C101 = 1.1 %, C000 = 1.4 %, C110 = 1.3 %, C011 = 0.8 %,  Lebedev = 0.85 %

Key physics revealed by the hybrid z-grid:
  k=3: transverse BC reflections dominate cluster errors (7–10 %).  The
    Lebedev average cancels these (opposite sign on different clusters) to
    achieve 0.72 % — a 13× improvement.
  k=4: modest transverse BC reflections (~2 %).  Lebedev improves to 0.86 %.
  k=5: domain is large (4.2 δ); transverse BC reflections are negligible and
    cluster errors (~1 %) are pure FD truncation.  Lebedev barely improves
    (0.85 %) because there is little systematic bias to cancel.

Note on k = 2
-------------
For k = 2 the transverse domain is only ±2.9 m (< 0.3δ), much smaller than
the evaluation range z = 4–7.5 m.  Mixed-BC image artefacts dominate and the
Lebedev does not cancel cleanly.  k ≥ 3 is needed for the far-field
superconvergence to manifest.

Usage
-----
    python examples/benchmark_logging.py

Output: examples/benchmark_logging.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em.grid import (
    LebedevGrid3D, optimal_geometric_1d, symmetric_optimal_grid,
    hybrid_axial_grid,
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

delta = np.sqrt(2.0 / (OMEGA * MU * SIGMA))

print("=" * 70)
print("Lebedev FD — DDH03 superconvergence benchmark")
print(f"  σ = {SIGMA} S/m,  f = {FREQ:.0f} Hz,  δ ≈ {delta:.2f} m")
print("=" * 70)

# ── Hybrid axial z-grid (DDH03 Fig. 5/6 style) ───────────────────────────────
# Inner zone: equidistant from Z_INNER_MIN to Z_INNER_MAX, spacing DZ.
# Must cover the source (z=0) and all receivers (Z_RECV_MAX).
# Outer zones: optimal geometric with K_OUTER steps — pushes the Dirichlet BC
# far beyond one skin depth so z-BC reflections are < 10⁻⁴ %.
#
# Choice: Z_INNER_MIN=-0.5 m keeps one DZ cell below the source; Z_INNER_MAX=8 m
# keeps one DZ cell above the last receiver at 7.5 m.  N_INNER=34 (even ✓).
# K_OUTER=8 gives domain ≈ ±112 m ≈ ±11 δ  →  z-BC reflection < 10⁻⁹.

DZ          = 0.0625  # [m] inner equidistant spacing (fine: 4× finer than before)
Z_INNER_MIN = -0.25   # [m] inner zone lower bound (just below source, = -4*DZ)
Z_INNER_MAX =  7.75   # [m] inner zone upper bound (just above receivers, = 7.5+4*DZ)
N_INNER     = int(round((Z_INNER_MAX - Z_INNER_MIN) / DZ))   # = 128 (even ✓)
K_OUTER     = 8       # geometric steps outside the inner zone

H_Z_SRC = DZ         # C101 source nodes at z = ±DZ; analytic uses two half-dipoles

# Receiver zone: far from source near-field, inside equidistant inner zone.
Z_RECV_MIN = 4.0     # [m]
Z_RECV_MAX = 7.5     # [m]

H_MIN   = 0.5        # [m] minimum transverse spacing
GAMMA   = 1.0 / np.sqrt(2.0)
L_TRANS = 300.0

z_fine = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)

print(f"\n  Axial grid (hybrid): Mz={len(z_fine)-1}, dz={DZ} m in inner zone [{Z_INNER_MIN},{Z_INNER_MAX}] m")
print(f"  Domain: [{z_fine[0]:.1f}, {z_fine[-1]:.1f}] m  "
      f"({(z_fine[-1]-Z_INNER_MAX)/delta:.1f}δ outer extension)")
print(f"  Receiver: z-axis (x=0, y=0), z ∈ ({Z_RECV_MIN}, {Z_RECV_MAX}) m")
print(f"  Analytic: two half-dipoles at (0,0,±{H_Z_SRC}) m  [C101 source geometry]")


# ── Analytic for C101 source geometry ─────────────────────────────────────────
def two_dipole_Ex(z_obs: float, moment: float = 1.0) -> complex:
    """E_x at (0,0,z_obs) from two half-dipoles at (0,0,±H_Z_SRC)."""
    E_p = electric_dipole_E(0.0, 0.0, z_obs - H_Z_SRC,
                             SIGMA, OMEGA, MU, EPS, 0, 0.5 * moment)[0]
    E_m = electric_dipole_E(0.0, 0.0, z_obs + H_Z_SRC,
                             SIGMA, OMEGA, MU, EPS, 0, 0.5 * moment)[0]
    return E_p + E_m


def rms_rel(fd_vals, ana_vals, fn=np.real):
    fd, ana = fn(np.asarray(fd_vals)), fn(np.asarray(ana_vals))
    mask = np.abs(ana) > 0.01 * np.max(np.abs(ana))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(((fd[mask] - ana[mask]) / ana[mask]) ** 2)))


# ── Sweep k ───────────────────────────────────────────────────────────────────
K_VALS = [2, 3, 4, 5]
labels = {C000: "C000", C101: "C101", C110: "C110", C011: "C011"}
colors = {C000: "#E63946", C101: "#F4A261", C110: "#2A9D8F", C011: "#8338EC"}

convergence = []

for k_steps in K_VALS:
    print(f"\n{'─'*65}")
    print(f"  k = {k_steps}  (Mx = My = {4*k_steps})")

    grid  = symmetric_optimal_grid(H_MIN, L_TRANS, z_fine, GAMMA, k=k_steps)
    Mx2   = grid.Mx // 2
    x_hat1 = float(grid.x[Mx2 + 1])
    x_max  = float(grid.x[-1])

    print(f"  x̂₁ = {x_hat1:.4f} m,  transverse domain ± {x_max:.2f} m "
          f"({x_max/delta:.2f}δ),  N_R = {grid.N_R}")

    if x_max < Z_RECV_MIN:
        print(f"  ⚠ Domain too small for far-field evaluation (x_max < z_min);"
              f" BC artefacts dominate — skipping.")
        convergence.append(None)
        continue

    # Solve
    media  = homogeneous_isotropic(grid, sigma=SIGMA, mu=MU, eps=EPS)
    solver = LebedevMaxwellSolver(grid, media, omega=OMEGA)
    result = solver.solve(x0=0.0, y0=0.0, z0=0.0, dipole_comp=0, moment=1.0)
    print("  Solve complete.")

    # ── Find C101 native z-axis DOF at (0, 0, z_r) ───────────────────────────
    nat101  = _native_type_for_cluster_comp(C101, 0)   # (0, 0, 1)
    E_c101  = result["E_c"][C101]

    z_list, Ex_c101_list = [], []
    for seq, (i, j, k_node) in enumerate(grid.R_nodes):
        if (i == Mx2 and j == Mx2 and
                (i % 2, j % 2, k_node % 2) == nat101):
            z_val = float(grid.z[k_node])
            if Z_RECV_MIN < z_val < Z_RECV_MAX:
                z_list.append(z_val)
                Ex_c101_list.append(E_c101[0 * grid.N_R + seq])

    if len(z_list) == 0:
        print("  ⚠ No C101 native nodes in receiver range — skipping.")
        convergence.append(None)
        continue

    z_eval  = np.array(z_list)
    Ex_c101 = np.array(Ex_c101_list)
    order   = np.argsort(z_eval)
    z_eval  = z_eval[order]
    Ex_c101 = Ex_c101[order]
    print(f"  {len(z_eval)} C101 z-axis nodes at z = {z_eval.round(2)}")

    # ── TRUE ANALYTIC — unit moment, no p_eff calibration ────────────────────
    # After the cell-volume normalisation in sources.py, the FD scheme injects
    # the correct total dipole moment for all four clusters.  We compare
    # directly to the analytic with moment = 1 A·m.
    ana = np.array([two_dipole_Ex(zv, 1.0) for zv in z_eval])

    # ── Per-cluster fields at (0, 0, z_eval) ─────────────────────────────────
    # C101: direct DOF read at (0, 0, z_odd)  — no interpolation.
    # C000: midpoint average of nodes at x = ±x̂₁  (error O((x̂₁/z)²) < 0.4 %).
    # C110: midpoint average of nodes at y = ±x̂₁  (same).
    # C011: double midpoint (x and y)             (error O((x̂₁/z)⁴) < 0.001 %).
    # All three use interpolate_cluster_E at (x=0, y=0, z).
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

    # ── Error metrics ─────────────────────────────────────────────────────────
    # p_leb: how close is the Lebedev average to the unit-moment analytic?
    # Ideally p_leb = 1.000 after the cell-volume normalisation fix.
    p_leb   = float(np.median(np.real(Ex_leb) / np.real(ana)))
    err_leb = rms_rel(Ex_leb, ana)
    errs    = {c: rms_rel(Ex_per_c[c], ana) for c in (C000, C101, C110, C011)}
    err_c101 = errs[C101]

    print(f"  p_leb = {p_leb:.5f}  (deviation from unit-moment analytic)")
    print(f"\n  RMS relative error vs unit-moment analytic (far-field, |z| ∈ "
          f"({Z_RECV_MIN},{Z_RECV_MAX}) m):")
    for c in (C000, C101, C110, C011):
        print(f"    {labels[c]}: {errs[c]*100:8.3f}%")
    print(f"    Lebedev avg:  {err_leb*100:8.3f}%")
    if err_c101 > 1e-8:
        print(f"    C101² (predicted Leb): {err_c101**2*100:8.4f}%  "
              f"(Leb/C101² = {err_leb/err_c101**2:.3f})")

    print(f"\n  Per-node FD/analytic ratios at (0, 0, z):")
    print(f"    {'z [m]':>6}  {'C101':>7}  {'C000':>7}  {'C110':>7}  {'C011':>7}  "
          f"{'Lebedev':>8}")
    for ri, zv in enumerate(z_eval):
        a_re = np.real(ana[ri])
        rs = {c: np.real(Ex_per_c[c][ri]) / a_re for c in (C000, C101, C110, C011)}
        lr = np.real(Ex_leb[ri]) / a_re
        print(f"    {zv:>6.2f}  "
              + "  ".join(f"{rs[c]:>7.4f}" for c in (C000, C101, C110, C011))
              + f"  {lr:>8.4f}")

    convergence.append({
        "k":        k_steps,
        "n_dof":    grid.N_R,
        "x_hat1":   x_hat1,
        "x_max":    x_max,
        "n_eval":   len(z_eval),
        "p_leb":    p_leb,
        "z_eval":   z_eval,
        "Ex_per_c": Ex_per_c,
        "Ex_leb":   Ex_leb,
        "ana":      ana,
        "errs":     errs,
        "err_leb":  err_leb,
        "err_c101": err_c101,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
valid = [d for d in convergence if d is not None]
print("\n" + "=" * 70)
print(f"SUMMARY  (far-field receivers: z ∈ ({Z_RECV_MIN}, {Z_RECV_MAX}) m)")
print(f"  {'k':>4}  {'N_R':>7}  {'domain':>9}  {'p_leb':>7}  "
      f"{'C101':>8}  {'C000':>8}  {'C110':>8}  {'C011':>8}  "
      f"{'Lebedev':>8}  {'Leb/C101²':>10}")
for d in valid:
    xm_d = d["x_max"] / delta
    sq = d["err_c101"] ** 2
    ratio = d["err_leb"] / sq if sq > 0 else float("nan")
    print(f"  {d['k']:>4}  {d['n_dof']:>7}  "
          f"  ±{d['x_max']:4.1f}m ({xm_d:.1f}δ)  "
          f"{d['p_leb']:>7.4f}  "
          f"{d['errs'][C101]*100:>7.2f}%  {d['errs'][C000]*100:>7.2f}%  "
          f"{d['errs'][C110]*100:>7.2f}%  {d['errs'][C011]*100:>7.2f}%  "
          f"{d['err_leb']*100:>7.2f}%  {ratio:>10.3f}")

print("""
Key observations (hybrid z-grid, z-BC reflections < 10⁻⁹):
  1. p_leb ≈ 1.000–1.007 for k ≥ 3: the cell-volume normalisation fix gives
     all four clusters the same effective source moment.
  2. k=3 (transverse domain ≈ 0.87δ): cluster errors 7–10 % dominated by
     transverse BC reflections.  C101/C011 overestimate, C000/C110 underestimate
     (opposite-sign M/E reflections).  Lebedev cancels to 0.72 %: 13× better.
  3. k=4 (domain ≈ 2δ): smaller BC reflections → cluster errors ~2 %, Lebedev
     0.86 %.  The improvement narrows as BC reflections shrink.
  4. k=5 (domain ≈ 4δ): transverse BC negligible, cluster errors ~1 % are pure
     FD truncation (same for all clusters → nothing for Lebedev to cancel →
     Lebedev ≈ 0.85 %, barely better than any individual cluster).
  5. The 'sweet spot' is k=3–4 where BC reflections provide the systematic bias
     that the Lebedev cancellation mechanism is designed to exploit.
  6. The hybrid z-grid is essential: the previous equidistant ±10 m z-grid
     produced exp(−1) ≈ 37 % z-BC reflections that contaminated all clusters
     equally, masking the true transverse BC cancellation pattern.
""")

# ── Plot ──────────────────────────────────────────────────────────────────────
if len(valid) < 2:
    print("Not enough valid k values for plot.")
    sys.exit(0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    rf"DDH03 Lebedev 4-cluster superconvergence  "
    rf"(σ={SIGMA} S/m, f={FREQ/1e3:.1f} kHz, δ≈{delta:.1f} m)"
    "\n"
    rf"Hybrid z-grid (equidistant inner ± geom. outer), "
    rf"x=y=0, z ∈ ({Z_RECV_MIN}, {Z_RECV_MAX}) m",
    fontsize=9,
)

# ── Panel 1: RMS error vs k ────────────────────────────────────────────────────
ax = axes[0]
k_plot    = [d["k"]        for d in valid]
e_c101    = [d["err_c101"] for d in valid]
e_leb     = [d["err_leb"]  for d in valid]

for c, mk in zip((C000, C110, C011), ("v", ">", "<")):
    ec = [d["errs"][c] for d in valid]
    ax.semilogy(k_plot, [e * 100 for e in ec], mk + ":",
                color=colors[c], lw=1.2, ms=7, alpha=0.7, label=labels[c])
ax.semilogy(k_plot, [e * 100 for e in e_c101], "o-",
            color=colors[C101], lw=2, ms=9, label="C101")
ax.semilogy(k_plot, [e * 100 for e in e_leb], "s-",
            color="navy", lw=2.5, ms=9, label="Lebedev avg")

ax.set_xlabel("k  (transverse steps)", fontsize=11)
ax.set_ylabel("RMS relative error of Re$(E_x)$  [%]", fontsize=11)
ax.set_title("Error vs k", fontsize=11)
ax.set_xticks(k_plot)
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.25)

# ── Helper: plot % deviation from analytic ────────────────────────────────────
def deviation_panel(ax, d):
    """
    Plot per-node signed relative deviation (FD − analytic) / analytic × 100 %
    for each cluster and the Lebedev average.  Analytic = 0 % reference line.
    """
    z_p = d["z_eval"]
    ana_re = np.real(d["ana"])

    mk_map = {C000: "s", C101: "o", C110: "^", C011: "D"}
    for c in (C000, C101, C110, C011):
        dev = (np.real(d["Ex_per_c"][c]) - ana_re) / ana_re * 100
        ax.plot(z_p, dev, mk_map[c] + "-",
                color=colors[c], ms=8, lw=1.2,
                label=f"{labels[c]}  (RMS {d['errs'][c]*100:.1f}%)")

    dev_leb = (np.real(d["Ex_leb"]) - ana_re) / ana_re * 100
    ax.plot(z_p, dev_leb, "P-",
            color="navy", ms=11, lw=2, zorder=10,
            markeredgecolor="white", mew=0.8,
            label=f"Lebedev  (RMS {d['err_leb']*100:.2f}%)")

    ax.axhline(0.0, color="black", lw=1.5, ls="--", label="Analytic (0 %)")

    ax.set_xlabel("z  (m)", fontsize=11)
    ax.set_ylabel(r"$(E_x^\mathrm{FD} - E_x^\mathrm{analytic})\ /\ E_x^\mathrm{analytic}$  [%]",
                  fontsize=9)
    ax.set_title(
        rf"Deviation from analytic  (k={d['k']}, ±{d['x_max']:.1f} m = {d['x_max']/delta:.1f}δ)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    # Symmetric y-axis so analytic line is centred
    ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    ax.set_ylim(-ylim * 1.15, ylim * 1.15)

# ── Panel 2: deviation for k=3 (large BC reflections, dramatic cancellation) ──
d_small = valid[0]   # k=3
deviation_panel(axes[1], d_small)

# ── Panel 3: deviation for the largest valid k (FD-truncation dominated) ───────
d_large = valid[-1]  # k=5
deviation_panel(axes[2], d_large)

plt.tight_layout()
outfile = os.path.join(os.path.dirname(__file__), "benchmark_logging.png")
plt.savefig(outfile, dpi=150)
print(f"Plot saved → {outfile}")

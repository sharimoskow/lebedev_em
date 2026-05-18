"""
benchmark_figs2_3.py  —  Combined DDH03 Figures 2 and 3 reproduction.

Grid strategy (matching DDH03):
  x, y : optimal geometric grid (k steps on each half-axis → Mx = My = 4k)
  z    : HYBRID — fine equidistant inner zone near source + optimal geometric outer

The hybrid z-grid is described in DDH03: "along the z-axis we set the grid to
be equidistant between the transmitter and the receiver, and optimal geometric
otherwise."  For the whole-space benchmark (no receiver separation), we set
the inner zone to ±Z_INNER centred on the source, with step DZ_INNER =
H_MIN/4 = 0.025 m so that the first P-node on-axis is at z = 2·DZ_INNER =
H_MIN/2 = 0.05 m.

Source centering: the source P-node must land EXACTLY at z = 0.  The full-grid
index of the center inner node is  2·K_OUTER + N_INNER//2.  For this to be
even (i.e. a P-node in the Lebedev checkerboard), N_INNER must be divisible
by 4.  With N_INNER=240 (= 4·60 ✓) and K_OUTER=12:
  center index = 2·12 + 120 = 144  (even ✓)  → source exactly at z = 0.

Physical parameters (DDH03 Figs 2 & 3):
  sigma = 1.0 S/m,  f = 2500 Hz,  x-directed magnetic dipole at origin
  Homogeneous isotropic whole-space

z-grid parameters:
  Z_INNER = 3.0 m,  N_INNER = 240 (equidistant, div-by-4 ✓),  K_OUTER = 12
  → DZ_INNER = 6.0 / 240 = 0.025 m
  → Mz = 4·12 + 240 = 288  (even ✓)
  → source exactly at z = 0  (P-node index 144, even ✓)
  → first on-axis P-node: z = 2·DZ = 0.05 m  (= H_MIN/2)
  → inner zone: 7 positive P-nodes in Fig-2 window (z = 0.05, 0.10, …, 0.35 m)
  → outer zone: K_OUTER geometric steps per side, z_max printed at runtime

x,y transverse domain (varies with k):
  k=4 → x_max ≈ 3.1 m  (0.3δ)
  k=6 → x_max ≈ 7.8 m  (0.8δ)
  k=8 → x_max ≈ 22.4 m (2.2δ)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from lebedev_em.grid import (symmetric_optimal_grid, optimal_geometric_1d,
                              hybrid_axial_grid,
                              LebedevGrid3D, C000, C101, C110, C011)
from lebedev_em.media import homogeneous_isotropic, MU0
from lebedev_em.solver import LebedevMaxwellSolver, _cluster_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import compute_B_from_E, build_rhs_per_cluster
from lebedev_em.analytics import Bxx_homogeneous

# ── Physical parameters ───────────────────────────────────────────────────────
SIGMA = 1.0
FREQ  = 2500.
OMEGA = 2 * np.pi * FREQ
delta = np.sqrt(2.0 / (OMEGA * MU0 * SIGMA))
print(f'sigma={SIGMA} S/m  f={FREQ:.0f} Hz  delta={delta:.2f} m')

# ── Grid parameters ───────────────────────────────────────────────────────────
H_MIN    = 0.1             # minimum transverse (x, y) spacing [m]
GAMMA    = 1.0 / np.sqrt(2)
L        = 300.            # target transverse domain half-length (informational)

# Hybrid z-grid parameters (fixed — independent of k):
Z_INNER  = 3.0             # inner equidistant zone half-width [m]
N_INNER  = 240             # equidistant inner intervals; MUST be divisible by 4
                           # so that source lands exactly at z=0 (even P-node).
                           # N_INNER=240 → DZ=0.025 m, 7 Fig-2 points in [0,0.35] m.
DZ_INNER = 2 * Z_INNER / N_INNER   # = 0.025 m = H_MIN/4
K_OUTER  = 12              # geometric outer steps per side (increased from 8 to
                           # compensate for finer DZ and keep z_max large)
# Mz = 4*K_OUTER + N_INNER = 48 + 240 = 288  (even ✓)
# Source center-index = 2*K_OUTER + N_INNER//2 = 24 + 120 = 144  (even ✓) → z=0 exactly
# z_max printed at runtime from the hybrid grid print below.

# Pre-build the shared z-array (same for all k values)
_Z_FULL = hybrid_axial_grid(-Z_INNER, Z_INNER, N_INNER, K_OUTER, GAMMA)
print(f'Hybrid z-grid: {len(_Z_FULL)} nodes, Mz={len(_Z_FULL)-1}, '
      f'z_max={_Z_FULL[-1]:.3f} m ({_Z_FULL[-1]/delta:.2f}δ), '
      f'first P-node at {2*DZ_INNER:.4f} m')


def build_grid(k):
    """
    Build a 3D Lebedev grid:
      x, y : optimal geometric (k steps → Mx = My = 4k)
      z     : fixed hybrid grid (_Z_FULL, Mz = 4·K_OUTER + N_INNER = 130)

    Inner zone: 49 positive P-nodes at z ≈ 0.12, 0.24, …, 3.0 m (DZ ≈ 0.0612 m).
    Outer zone: K_OUTER = 8 P-nodes per side reaching z_max ≈ 18 m (1.8δ).
    """
    return symmetric_optimal_grid(H_MIN, L, _Z_FULL, GAMMA, k=k)


# ── Per-cluster B extraction from separate per-cluster B-vectors ──────────────
# DDH03 Fig 2 comes from 4 SEPARATE cluster solves (one unit-moment source each).
# Each cluster's B is read at its OWN native P-node positions from its OWN solve.
# This is the correct approach: each cluster sees only its own unit-moment dipole.
#
# Cluster → native P-node type for Hx (from _H_CLUSTER_MAP):
#   C011: type (0,0,0) → node (i0, j0, k)          (single node per receiver)
#   C000: type (0,1,1) → nodes (i0, j0±1, k±1)     (4 nodes, averaged)
#   C101: type (1,1,0) → nodes (i0±1, j0±1, k)     (4 nodes, averaged)
#   C110: type (1,0,1) → nodes (i0±1, j0, k±1)     (4 nodes, averaged)
def extract_Bxx_separate(grid, B_clusters):
    """
    Extract per-cluster Bxx along the z-axis from 4 separate cluster B-vectors.

    B_clusters : dict {cluster → ndarray (3·N_P,)} — one B-field per cluster,
                 each computed from a separate solve with that cluster's own source.
    """
    Mx, My, Mz = grid.Mx, grid.My, grid.Mz
    i0, j0 = Mx // 2, My // 2
    N_P    = grid.N_P
    comp   = 0  # Bx

    def b_at(B_vec, i, j, k):
        if not (0 <= i <= Mx and 0 <= j <= My and 0 <= k <= Mz):
            return 0j
        seq = int(grid.P_idx[i, j, k])
        return 0j if seq < 0 else complex(B_vec[comp * N_P + seq])

    z_list, avg_list = [], []
    per_cl = {C011: [], C000: [], C101: [], C110: []}

    for k in range(0, Mz + 1, 2):
        if grid.P_idx[i0, j0, k] < 0:
            continue
        # Each cluster reads its OWN B_vec at its OWN native Hx P-node positions
        v011 = b_at(B_clusters[C011], i0, j0, k)
        v000 = np.mean([b_at(B_clusters[C000], i0, j0+dj, k+dk) for dj in (+1,-1) for dk in (+1,-1)])
        v101 = np.mean([b_at(B_clusters[C101], i0+di, j0+dj, k) for di in (+1,-1) for dj in (+1,-1)])
        v110 = np.mean([b_at(B_clusters[C110], i0+di, j0,   k+dk) for di in (+1,-1) for dk in (+1,-1)])
        z_list.append(float(grid.z[k]))
        per_cl[C011].append(complex(v011))
        per_cl[C000].append(complex(v000))
        per_cl[C101].append(complex(v101))
        per_cl[C110].append(complex(v110))
        avg_list.append((complex(v011)+complex(v000)+complex(v101)+complex(v110))/4.)

    z_arr = np.array(z_list)
    order = np.argsort(z_arr)
    z_arr = z_arr[order]
    B_per_cl = {c: np.array(per_cl[c], dtype=complex)[order]
                for c in (C000, C101, C110, C011)}
    B_avg = np.array(avg_list, dtype=complex)[order]
    return z_arr, B_per_cl, B_avg


def solve(k):
    t0   = time.time()
    grid = build_grid(k)
    x_max = float(grid.x[-1])
    z_max = float(grid.z[-1])
    z1    = float(grid.z[grid.Mz // 2 + 2])   # first P-node above source (= 2·DZ_INNER)
    print(f'\nk={k}  Mx=My={grid.Mx}  Mz={grid.Mz}  N_R={grid.N_R}  DOFs={3*grid.N_R}')
    print(f'  x_max={x_max:.3f} m ({x_max/delta:.3f}δ)  '
          f'z_max={z_max:.3f} m ({z_max/delta:.3f}δ)  '
          f'first receiver z={z1:.4f} m  t={time.time()-t0:.1f}s', flush=True)

    med    = homogeneous_isotropic(grid, sigma=SIGMA)
    solver = LebedevMaxwellSolver(grid, med, OMEGA)

    # Build 4 separate per-cluster RHS (each with unit moment for its own cluster).
    # This is the DDH03 approach: each cluster is solved independently with its own
    # source, so near the source each cluster sees only its own unit-moment dipole.
    rhs_per_c = build_rhs_per_cluster(grid, solver._C_PR, OMEGA)

    # Solve each cluster separately with its own DDH03 mixed BC.
    B_clusters = {}
    for c in (C000, C101, C110, C011):
        bc_dofs = _cluster_bc_dofs(grid, c)
        A_bc, b_bc = apply_electric_bc(solver._A.copy(), rhs_per_c[c].copy(), bc_dofs)
        d     = A_bc.diagonal()
        d_inv = np.where(np.abs(d) > 1e-30, 1.0/d, 1.0)
        M = spla.LinearOperator(A_bc.shape, matvec=lambda x: d_inv*x, dtype=complex)
        print(f'  Solving cluster {c} (LGMRES)...', flush=True)
        E, info = spla.lgmres(A_bc, b_bc, M=M, rtol=1e-8, atol=0,
                              maxiter=300, inner_m=30, outer_k=10)
        print(f'  LGMRES cluster {c} info={info}  t={time.time()-t0:.1f}s', flush=True)
        B_clusters[c] = compute_B_from_E(grid, E, OMEGA)

    # Extract per-cluster Bxx: each cluster reads from its own B_vec at its own
    # native Hx P-node positions → correct DDH03 per-cluster values.
    z_ax, B_per_cl, B_avg = extract_Bxx_separate(grid, B_clusters)
    return dict(grid=grid, z_ax=z_ax, B_per_cl=B_per_cl,
                B_avg=B_avg, x_max=x_max, z_max=z_max, z1=z1)


# ── Run — k=6 only for this fine-grid test (4 separate LGMRES solves) ────────
# The finer z-grid (Mz=288 vs 130) roughly doubles DOFs; running all three k
# values would take ~3× longer.  Add k=4 or k=8 back once the Fig-2 cluster
# spread looks right.  k=8 is needed for Fig-3 (large transverse domain).
results = {}
for k in [6, 8]:
    results[k] = solve(k)

# ── Print errors ──────────────────────────────────────────────────────────────
cluster_colors = {C000: '#e11d48', C101: '#2563eb', C110: '#d97706', C011: '#16a34a'}
cluster_labels = {C000: 'C000 (Yee)', C101: 'C101', C110: 'C110', C011: 'C011'}

for k, r in results.items():
    z_ax, B_per_cl, B_avg = r['z_ax'], r['B_per_cl'], r['B_avg']
    near = (z_ax >= 0.5) & (z_ax <= 3.0)
    if not np.any(near):
        continue
    ana = np.imag(Bxx_homogeneous(z_ax[near], SIGMA, OMEGA))
    scale = np.max(np.abs(ana))
    print(f'\nk={k} near-source RMS error (0 < z < 1 m):')
    for lbl, arr in [('C000', B_per_cl[C000]), ('C101', B_per_cl[C101]),
                     ('C110', B_per_cl[C110]), ('C011', B_per_cl[C011]),
                     ('Leb avg', B_avg)]:
        rms = np.sqrt(np.mean(((np.imag(arr[near])-ana)/scale)**2))
        print(f'  {lbl:8s}: {rms*100:.1f}%')


# ── Plot: two panels matching DDH03 Figs 2 & 3 exactly ───────────────────────
# Fig 2: z = 0 to 0.35 m  (near source — cluster spread, k=6)
# Fig 3: z = 2 to 12 m    (near outer boundary — error cancellation, k=8)
r2 = results[6]
r3 = results[8] if 8 in results else results[6]
z_ax2, B_per_cl2, B_avg2 = r2['z_ax'], r2['B_per_cl'], r2['B_avg']
z_ax3, B_per_cl3, B_avg3 = r3['z_ax'], r3['B_per_cl'], r3['B_avg']
x_max2, z_max2 = r2['x_max'], r2['z_max']
x_max3, z_max3 = r3['x_max'], r3['z_max']

# Use local aliases for the plot
z_ax, B_per_cl, B_avg = z_ax2, B_per_cl2, B_avg2
x_max, z_max = x_max2, z_max2

fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    rf'DDH03 Figs 2 & 3 reproduction  —  $\sigma$={SIGMA} S/m, f={FREQ:.0f} Hz, '
    rf'$\delta$={delta:.1f} m,  k=6 (Fig2) / k=8 (Fig3)  '
    rf'(x,y: ±{x_max2:.1f}/±{x_max3:.1f} m, z: ±{z_max2:.1f} m)',
    fontsize=10)

# ── Fig 2: near source, z = 0 to 0.35 m ──────────────────────────────────────
m2 = (z_ax > 0) & (z_ax <= 0.35)
for c in (C000, C101, C110, C011):
    ax2.plot(z_ax[m2], np.imag(B_per_cl[c][m2]),
             'o-', color=cluster_colors[c], lw=1.5, ms=5, label=cluster_labels[c])
ax2.plot(z_ax[m2], np.imag(B_avg[m2]), 'k-', lw=2.5, label='Lebedev avg', zorder=5)
ax2.plot(z_ax[m2], np.imag(Bxx_homogeneous(z_ax[m2], SIGMA, OMEGA)),
         'ko', ms=7, zorder=6, markerfacecolor='white', markeredgewidth=1.5,
         label='Analytic')
ax2.set_xlabel('z (m)'); ax2.set_ylabel('Im(Bxx) [T/A·m²]')
ax2.set_title('Fig. 2 — Im(Bxx) near transmitter\n(clusters, Lebedev avg, analytic)')
ax2.set_xlim([0, 0.35]); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# ── Fig 3: near outer boundary, z = 2 to 12 m (k=8 for large transverse domain)
m3 = (z_ax3 >= 2.0) & (z_ax3 <= 12.0)
for c in (C000, C101, C110, C011):
    ax3.plot(z_ax3[m3], np.imag(B_per_cl3[c][m3]),
             'o-', color=cluster_colors[c], lw=1.5, ms=5, label=cluster_labels[c])
ax3.plot(z_ax3[m3], np.imag(B_avg3[m3]), 'k-', lw=2.5, label='Lebedev avg', zorder=5)
ax3.plot(z_ax3[m3], np.imag(Bxx_homogeneous(z_ax3[m3], SIGMA, OMEGA)),
         'ko', ms=7, zorder=6, markerfacecolor='white', markeredgewidth=1.5,
         label='Analytic')
ax3.axvline(z_max3, color='gray', ls=':', lw=1.2, label=f'z_max={z_max3:.1f} m')
ax3.set_xlabel('z (m)'); ax3.set_ylabel('Im(Bxx) [T/A·m²]')
ax3.set_title('Fig. 3 — Im(Bxx) near outer boundary\n(clusters, Lebedev avg, analytic)')
ax3.set_xlim([2, 12]); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plotfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'benchmark_figs2_3.png')
plt.savefig(plotfile, dpi=120, bbox_inches='tight')
print(f'\nPlot saved: {plotfile}')
plt.show()

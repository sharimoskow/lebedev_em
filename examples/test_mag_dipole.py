"""
test_mag_dipole.py — Test magnetic x-dipole source implementation.

Implements b = iω C_PR M_P_vec (where M_P_vec has Hx at origin P-node)
and compares Bxx on the z-axis with the analytic homogeneous-medium formula.
"""
import sys, time
sys.path.insert(0, "src")

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import homogeneous_isotropic
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import compute_B_from_E, extract_B_on_axis
from lebedev_em.analytics import Bxx_homogeneous
from lebedev_em.media import MU0

FREQ  = 52650.0
OMEGA = 2.0 * np.pi * FREQ
SIGMA = 0.1  # homogeneous conductivity for calibration

# ── Grid ─────────────────────────────────────────────────────────────────────
Z_INNER_MIN, Z_INNER_MAX = -3.5, 2.5
N_INNER = 98
K_OUTER = 8
GAMMA   = 1.0 / np.sqrt(2.0)
H_MIN, L_TRANS = 0.5, 300.0
K_GRID = 3  # smallest k for speed

z_fd_grid = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)
grid = symmetric_optimal_grid(H_MIN, L_TRANS, z_fd_grid, GAMMA, k=K_GRID)
print(f"Grid: Mx={grid.Mx}, My={grid.My}, Mz={grid.Mz}, N_R={grid.N_R}, N_P={grid.N_P}")
print(f"x-center: x[Mx//2]={grid.x[grid.Mx//2]:.6f}")
print(f"y-center: y[My//2]={grid.y[grid.My//2]:.6f}")

# ── Build media & solver ──────────────────────────────────────────────────────
med = homogeneous_isotropic(grid, SIGMA)
solver = LebedevMaxwellSolver(grid, med, OMEGA)

# ── Magnetic x-dipole source: b = iω C_PR M_P_vec ────────────────────────────
# Hx at P-nodes with type (0,0,0): all-even (i,j,k)
# i=Mx//2 (center in x) — must be even
# y: My//2 might be odd, so use nearest even j-pairs
# z: find even k with z[k] nearest 0

Mx, My, Mz = grid.Mx, grid.My, grid.Mz

# x-center
i0 = Mx // 2
assert i0 % 2 == 0, f"i0={i0} is odd; grid Mx={Mx}"
print(f"i0={i0}, x[i0]={grid.x[i0]:.4f}")

# y: find even indices symmetric about center
j_center = My // 2
print(f"j_center={j_center} (y[j_center]={grid.y[j_center]:.4f})")
if j_center % 2 == 0:
    j_list = [j_center]
    weights_j = [1.0]
else:
    # Use the two nearest even j indices (symmetric about center)
    j_lo = j_center - 1   # even
    j_hi = j_center + 1   # even
    if j_lo >= 0 and j_hi <= My:
        j_list = [j_lo, j_hi]
        weights_j = [0.5, 0.5]
    elif j_lo >= 0:
        j_list = [j_lo]; weights_j = [1.0]
    else:
        j_list = [j_hi]; weights_j = [1.0]
print(f"j_list={j_list}, y-values={[grid.y[j] for j in j_list]}")

# z: find even k nearest z=0
k_even = np.arange(0, Mz+1, 2)
z_vals = grid.z[k_even]
k0_idx = int(np.argmin(np.abs(z_vals)))
k0 = int(k_even[k0_idx])
print(f"k0={k0}, z[k0]={grid.z[k0]:.4f}")

# Collect (P-seq, weight) pairs
P_sources = []
for j0, wj in zip(j_list, weights_j):
    # Check P-node condition: i+j+k must be even
    assert (i0 + j0 + k0) % 2 == 0, f"Not P-node: ({i0},{j0},{k0})"
    P_seq = int(grid.P_idx[i0, j0, k0])
    assert P_seq >= 0, f"P_idx is -1 for ({i0},{j0},{k0})"
    # Dual cell volume at this P-node (skip-2)
    dx = float(grid.x[min(i0+1,Mx)] - grid.x[max(i0-1,0)])
    dy = float(grid.y[min(j0+1,My)] - grid.y[max(j0-1,0)])
    dz = float(grid.z[min(k0+1,Mz)] - grid.z[max(k0-1,0)])
    vol_P = abs(dx * dy * dz)
    P_sources.append((P_seq, wj, vol_P, j0))
    print(f"  Hx P-node: ({i0},{j0},{k0}), seq={P_seq}, "
          f"pos=({grid.x[i0]:.3f},{grid.y[j0]:.3f},{grid.z[k0]:.3f}), vol={vol_P:.6e}")

# Build M_P_vec and b
m_x = 1.0  # unit magnetic dipole moment [A·m²]
M_P_vec = np.zeros(3 * grid.N_P, dtype=complex)
for (P_seq, wj, vol_P, j0) in P_sources:
    dof_Hx = 0 * grid.N_P + P_seq
    M_P_vec[dof_Hx] += m_x * wj / vol_P

C_PR = solver._C_PR
b_mag = 1j * OMEGA * (C_PR @ M_P_vec)
print(f"\nb_mag nonzeros: {np.count_nonzero(b_mag)}")
print(f"max |b_mag|: {np.abs(b_mag).max():.4e}")

# ── Solve ─────────────────────────────────────────────────────────────────────
bc_dofs = _component_aware_bc_dofs(grid)
A_bc, b_bc = apply_electric_bc(solver._A.copy(), b_mag, bc_dofs)
print("Solving...")
t0 = time.time()
E = spla.spsolve(A_bc, b_bc)
print(f"Solved in {time.time()-t0:.1f}s")

# ── Compute Bx on z-axis ──────────────────────────────────────────────────────
B_vec = compute_B_from_E(grid, E, OMEGA)
z_bx, Bx_axis = extract_B_on_axis(grid, B_vec, comp=0, axis='z')

# Filter to measurement window [-2, 0] m
mask = (z_bx >= -2.0) & (z_bx <= -0.05)
z_meas = z_bx[mask]
Bx_meas = Bx_axis[mask]

# ── Analytic comparison ───────────────────────────────────────────────────────
Bx_analytic = np.array([Bxx_homogeneous(z, SIGMA, OMEGA) for z in z_meas])

print(f"\nComparison at selected z values:")
print(f"{'z':>8}  {'Im(Bx_FD)':>14}  {'Im(Bx_ana)':>14}  {'ratio':>8}")
for i in range(0, len(z_meas), max(1, len(z_meas)//10)):
    z = z_meas[i]
    fd = np.imag(Bx_meas[i])
    ana = np.imag(Bx_analytic[i])
    ratio = fd/ana if abs(ana) > 0 else np.nan
    print(f"{z:8.3f}  {fd:14.4e}  {ana:14.4e}  {ratio:8.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.semilogy(z_meas, np.abs(np.imag(Bx_meas))*1e9,  'b-', lw=2, label='FD |Im(Bxx)|')
ax.semilogy(z_meas, np.abs(np.imag(Bx_analytic))*1e9, 'r--', lw=2, label='Analytic')
ax.set_xlabel('z [m]'); ax.set_ylabel('Im(Bxx) [nT]')
ax.set_title(f'Magnetic x-dipole: FD vs Analytic\nσ={SIGMA} S/m, f={FREQ/1e3:.2f} kHz')
ax.legend(); ax.grid(True, which='both', alpha=0.3)

ax2 = axes[1]
ratio = np.imag(Bx_meas) / np.imag(Bx_analytic)
ax2.plot(z_meas, ratio, 'k-', lw=1.5)
ax2.axhline(1.0, color='r', ls='--')
ax2.set_xlabel('z [m]'); ax2.set_ylabel('FD / Analytic')
ax2.set_title('Ratio FD/Analytic Im(Bxx)')
ax2.set_ylim(-2, 3); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/sessions/wizardly-intelligent-feynman/mnt/outputs/lebedev_em/examples/mag_dipole_test.png', dpi=150)
print("\nSaved mag_dipole_test.png")

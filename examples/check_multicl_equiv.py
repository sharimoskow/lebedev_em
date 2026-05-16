"""
check_multicl_equiv.py  —  Verify that the single combined solve and 4 separate
cluster solves give identical B-field results (to machine precision).

Usage:
    python check_multicl_equiv.py [k]        (default k=3, fast ~16 s total)

Method A (current):  one combined RHS, one solve, multi-cluster extraction.
Method B (separate): 4 per-cluster RHS, 4 solves, lebedev_B_on_z_axis average.

Expected: max |B_A - B_B| / max|B_A|  < 1e-10  (floating-point round-off only).
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import factorized

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid, C000, C101, C110, C011
from lebedev_em.media import EMMedia, MU0, EPS0, _nodal_eff_tensor_general, _volume_frac_layer1_planar, _frac_1d_layer1
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (
    compute_B_from_E,
    build_rhs_multicl,
    extract_B_on_axis_multicl,
    build_rhs_per_cluster,
    lebedev_B_on_z_axis,
)

# ── Parameters (same as run_k6_mc_bg.py) ──────────────────────────────────────
FREQ = 52650.; OMEGA = 2 * 3.14159265358979 * FREQ
SIGMA_BORE = 0.05; SIGMA_INV = 0.10; SIGMA_ISO = 0.50
SIGMA_N = 0.01; SIGMA_T = 0.10; R_BORE = 0.1; R_INV = 0.6
DIP_RAD = 1.04719755119660
N_HAT = np.array([np.sin(DIP_RAD), 0., np.cos(DIP_RAD)])
D_PLANE = N_HAT[0]*0.0 + N_HAT[2]*(-0.5)
SIGMA_ANISO = SIGMA_T * np.eye(3) + (SIGMA_N - SIGMA_T) * np.outer(N_HAT, N_HAT)
z_fd = hybrid_axial_grid(-3.5, 2.5, 98, 8, 1.0/2**0.5)
GAMMA = 1.0/2**0.5; Z_MIN, Z_MAX = -1.7, 0.02; H = 0.10

_S_BORE = SIGMA_BORE * np.eye(3, dtype=complex)
_S_INV  = SIGMA_INV  * np.eye(3, dtype=complex)
_S_ISO  = SIGMA_ISO  * np.eye(3, dtype=complex)
_S_ANI  = SIGMA_ANISO.astype(complex)


def _pointwise_sigma(x, y, z):
    r = (x**2 + y**2)**0.5
    if r < R_BORE:   return _S_BORE.copy()
    elif r < R_INV:  return _S_INV.copy()
    else:
        return _S_ANI.copy() if N_HAT[0]*x + N_HAT[2]*z < D_PLANE else _S_ISO.copy()


def _outer_sigma(x, y, z):
    return _S_ANI.copy() if N_HAT[0]*x + N_HAT[2]*z < D_PLANE else _S_ISO.copy()


def _line_frac(box_min, box_max, n_hat, d_plane, node):
    n_dot_node = float(np.dot(n_hat, node))
    f = np.empty(3, dtype=float)
    for a in range(3):
        d_rest = d_plane - n_dot_node + float(n_hat[a]) * float(node[a])
        f[a] = _frac_1d_layer1(box_min[a], box_max[a], float(n_hat[a]), d_rest)
    return f


def build_media(grid):
    nx, ny, nz = len(grid.x), len(grid.y), len(grid.z)
    sr = np.zeros((grid.N_R, 3, 3), dtype=complex)
    for seq, (i, j, kk) in enumerate(grid.R_nodes):
        x = float(grid.x[i]); y = float(grid.y[j]); z = float(grid.z[kk])
        r = (x**2 + y**2)**0.5; node = np.array([x, y, z])
        bmin = np.array([float(grid.x[max(i-1,0)]),  float(grid.y[max(j-1,0)]),  float(grid.z[max(kk-1,0)])])
        bmax = np.array([float(grid.x[min(i+1,nx-1)]),float(grid.y[min(j+1,ny-1)]),float(grid.z[min(kk+1,nz-1)])])
        sr[seq] = _pointwise_sigma(x, y, z); averaged = False
        if r > 1e-10:
            n_hat_r = np.array([x/r, y/r, 0.0])
            xs = [bmin[0], bmax[0]]; ys = [bmin[1], bmax[1]]
            corner_d = [n_hat_r[0]*cx + n_hat_r[1]*cy for cx in xs for cy in ys]
            if min(corner_d) < R_BORE <= max(corner_d):
                sr[seq] = _nodal_eff_tensor_general(
                    _S_BORE, _S_INV,
                    _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_BORE),
                    _line_frac(bmin, bmax, n_hat_r, R_BORE, node), n_hat_r)
                averaged = True
        if not averaged and r > 1e-10:
            n_hat_r = np.array([x/r, y/r, 0.0])
            xs = [bmin[0], bmax[0]]; ys = [bmin[1], bmax[1]]
            corner_d = [n_hat_r[0]*cx + n_hat_r[1]*cy for cx in xs for cy in ys]
            if min(corner_d) < R_INV <= max(corner_d):
                d_out = R_INV + 0.5*(bmax[0]-bmin[0])
                x_out = x + (d_out - r)*n_hat_r[0]; y_out = y + (d_out - r)*n_hat_r[1]
                sr[seq] = _nodal_eff_tensor_general(
                    _S_INV, _outer_sigma(x_out, y_out, z),
                    _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_INV),
                    _line_frac(bmin, bmax, n_hat_r, R_INV, node), n_hat_r)
                averaged = True
        if not averaged and r >= R_INV:
            xs = [bmin[0], bmax[0]]; zs = [bmin[2], bmax[2]]
            corner_dp = [N_HAT[0]*cx + N_HAT[2]*cz for cx in xs for cz in zs]
            if min(corner_dp) < D_PLANE <= max(corner_dp):
                sr[seq] = _nodal_eff_tensor_general(
                    _S_ANI, _S_ISO,
                    _volume_frac_layer1_planar(bmin, bmax, N_HAT, D_PLANE),
                    _line_frac(bmin, bmax, N_HAT, D_PLANE, node), N_HAT)
    return EMMedia(grid, sr, np.full(grid.N_P, complex(MU0)), np.full(grid.N_R, complex(EPS0)))


# ── Setup ──────────────────────────────────────────────────────────────────────
k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
t0 = time.time()
print(f"=== Multi-cluster equivalence check  k={k} ===", flush=True)

grid  = symmetric_optimal_grid(H, 300., z_fd, GAMMA, k=k)
med   = build_media(grid)
solver = LebedevMaxwellSolver(grid, med, OMEGA)
bc    = _component_aware_bc_dofs(grid)

print(f"Grid built: N_R={grid.N_R}  t={time.time()-t0:.1f}s", flush=True)

# Assemble system (one matrix, shared by all solves)
A = solver._A.copy()

# ── Factor the matrix once (shared by all solves) ─────────────────────────────
print("\nFactoring system matrix...", flush=True)
b_combined = build_rhs_multicl(grid, solver._C_PR, OMEGA)
A_bc, b_bc = apply_electric_bc(A.copy(), b_combined, bc)
tf = time.time()
solve = factorized(A_bc)   # LU once
print(f"  Factor done  t={time.time()-tf:.1f}s", flush=True)

# ── Method A: single combined solve ───────────────────────────────────────────
print("\n--- Method A: single combined RHS ---", flush=True)
tA = time.time()
E_comb = solve(b_bc)
print(f"  Back-sub done  t={time.time()-tA:.3f}s", flush=True)

B_comb = compute_B_from_E(grid, E_comb, OMEGA)
z_x_A, Bxx_A = extract_B_on_axis_multicl(grid, B_comb, comp=0, axis='z')
z_z_A, Bxz_A = extract_B_on_axis_multicl(grid, B_comb, comp=2, axis='z')

# ── Method B: four separate solves (reuse factorization) ──────────────────────
print("\n--- Method B: four separate cluster RHS ---", flush=True)
rhs_per_c = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=0)

B_clusters: dict = {}
for c_label, c in [('C011', C011), ('C000', C000), ('C101', C101), ('C110', C110)]:
    b_c = rhs_per_c[c]
    _, b_bc_c = apply_electric_bc(A_bc.copy(), b_c, bc)
    ts = time.time()
    E_c = solve(b_bc_c)    # back-sub only
    print(f"  Cluster {c_label}: back-sub t={time.time()-ts:.3f}s", flush=True)
    B_clusters[c] = compute_B_from_E(grid, E_c, OMEGA)

z_x_B, Bxx_B = lebedev_B_on_z_axis(grid, B_clusters, comp=0)
z_z_B, Bxz_B = lebedev_B_on_z_axis(grid, B_clusters, comp=2)

# ── Method C: verify linearity — sum separate B_c, apply same multicl extraction ──
print("\n--- Method C: sum of separate B fields, same multicl extraction as Method A ---", flush=True)
B_sum = sum(B_clusters.values())   # should equal B_comb by linearity
z_x_C, Bxx_C = extract_B_on_axis_multicl(grid, B_sum, comp=0, axis='z')
z_z_C, Bxz_C = extract_B_on_axis_multicl(grid, B_sum, comp=2, axis='z')

# ── Compare ────────────────────────────────────────────────────────────────────
print("\n--- Comparison ---", flush=True)

def _rel_err(a, b):
    denom = np.max(np.abs(a))
    return np.max(np.abs(a - b)) / denom if denom > 0 else 0.0

err_xx_AB = _rel_err(Bxx_A, Bxx_B)
err_xz_AB = _rel_err(Bxz_A, Bxz_B)
err_xx_AC = _rel_err(Bxx_A, Bxx_C)
err_xz_AC = _rel_err(Bxz_A, Bxz_C)

print(f"  A vs B  (different extraction functions):")
print(f"    max|Bxx_A - Bxx_B| / max|Bxx_A| = {err_xx_AB:.3e}")
print(f"    max|Bxz_A - Bxz_B| / max|Bxz_A| = {err_xz_AB:.3e}")
print(f"  A vs C  (same extraction, combined vs sum-of-separate — tests pure solve linearity):")
print(f"    max|Bxx_A - Bxx_C| / max|Bxx_A| = {err_xx_AC:.3e}")
print(f"    max|Bxz_A - Bxz_C| / max|Bxz_A| = {err_xz_AC:.3e}")
if err_xx_AC < 1e-8 and err_xz_AC < 1e-8:
    print("  C vs A PASS: superposition holds to machine precision.")

# Side-by-side table in nT
mx  = (z_x_A >= Z_MIN) & (z_x_A <= Z_MAX)
mz  = (z_z_A >= Z_MIN) & (z_z_A <= Z_MAX)
mzB = (z_z_B >= Z_MIN) & (z_z_B <= Z_MAX)

bxx_A = np.imag(Bxx_A[mx]) * 1e9
bxz_A = np.interp(z_x_A[mx], z_z_A[mz], np.imag(Bxz_A[mz]) * 1e9)
bxx_B = np.imag(Bxx_B[mx]) * 1e9
bxz_B = np.interp(z_x_B[mx], z_z_B[mzB], np.imag(Bxz_B[mzB]) * 1e9)
bxx_C = np.imag(Bxx_C[mx]) * 1e9
mzC   = (z_z_C >= Z_MIN) & (z_z_C <= Z_MAX)
bxz_C = np.interp(z_x_A[mx], z_z_C[mzC], np.imag(Bxz_C[mzC]) * 1e9)

print(f"\n  {'z(m)':>8}  {'Bxx_A':>9}  {'Bxx_C':>9}  {'Bxz_A':>9}  {'Bxz_C':>9}  {'Bxz_B':>9}  (nT)")
print(f"  {'':>8}  {'(comb)':>9}  {'(sum)':>9}  {'(comb)':>9}  {'(sum)':>9}  {'(lebB)':>9}")
for i, z in enumerate(z_x_A[mx]):
    print(f"  {z:+8.4f}  {bxx_A[i]:+9.4f}  {bxx_C[i]:+9.4f}  {bxz_A[i]:+9.4f}  {bxz_C[i]:+9.4f}  {bxz_B[i]:+9.4f}")

# Crossing positions
def _crossing(z, diff):
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc):
        zi = sc[0]
        return z[zi] - diff[zi] / (diff[zi+1] - diff[zi]) * (z[zi+1] - z[zi])
    return None

zc_A = _crossing(z_x_A[mx], bxx_A - bxz_A)
zc_C = _crossing(z_x_A[mx], bxx_C - bxz_C)
zc_B = _crossing(z_x_B[mx], bxx_B - bxz_B)

print(f"\n  Crossing (Bxx=Bxz):")
print(f"    Method A (combined solve + multicl extraction): {f'z = {zc_A:.4f} m' if zc_A is not None else 'none in window'}")
print(f"    Method C (sum-of-separate + multicl extraction): {f'z = {zc_C:.4f} m' if zc_C is not None else 'none in window'}")
print(f"    Method B (separate + lebedev_B extraction):     {f'z = {zc_B:.4f} m' if zc_B is not None else 'none in window'}")

print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)

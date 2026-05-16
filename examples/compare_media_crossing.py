"""
compare_media_crossing.py  —  Compare Bxx/Bxz crossing positions using two
media-building strategies for cells that straddle both the invasion cylinder
and the dipping anisotropy plane simultaneously.

Strategy A (original): assign the outer medium from a single probe point.
Strategy B (improved):  use the full 3-region nodal homogenization with n̂ = n̂_r,
    computing D1/D2/D3 and G_TT/G_nn as genuine line/volume averages over all
    three materials (σ_INV, σ_ANI, σ_ISO).

Usage:
    python compare_media_crossing.py [k]    (default k=3)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
from scipy.sparse.linalg import factorized

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import (EMMedia, MU0, EPS0,
                               _nodal_eff_tensor_general,
                               _nodal_eff_tensor_multiregion,
                               _multiregion_line_and_vol_fracs,
                               _volume_frac_layer1_planar, _frac_1d_layer1)
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_multicl,
                                    extract_B_on_axis_multicl)

# ── Physical parameters (same as run_k6_mc_bg.py) ─────────────────────────────
FREQ = 52650.; OMEGA = 2 * np.pi * FREQ
SIGMA_BORE = 0.05; SIGMA_INV = 0.10; SIGMA_ISO = 0.50
SIGMA_N = 0.01; SIGMA_T = 0.10; R_BORE = 0.1; R_INV = 0.6
DIP_RAD = 1.04719755119660
N_HAT = np.array([np.sin(DIP_RAD), 0., np.cos(DIP_RAD)])
D_PLANE = N_HAT[0]*0.0 + N_HAT[2]*(-0.5)
SIGMA_ANISO = SIGMA_T * np.eye(3) + (SIGMA_N - SIGMA_T) * np.outer(N_HAT, N_HAT)
z_fd = hybrid_axial_grid(-3.5, 2.5, 98, 8, 1.0/2**0.5)
GAMMA = 1.0/2**0.5; Z_MIN, Z_MAX = -2.5, 0.02; H = 0.10

_S_BORE = SIGMA_BORE * np.eye(3, dtype=complex)
_S_INV  = SIGMA_INV  * np.eye(3, dtype=complex)
_S_ISO  = SIGMA_ISO  * np.eye(3, dtype=complex)
_S_ANI  = SIGMA_ANISO.astype(complex)


def _pointwise_sigma(x, y, z):
    r = (x**2 + y**2)**0.5
    if r < R_BORE:  return _S_BORE.copy()
    elif r < R_INV: return _S_INV.copy()
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


def _cell_straddles(bmin, bmax, n_hat, d_plane, dims=(0, 1)):
    """True if any pair of corners spans d_plane along the given axes."""
    corners = [float(n_hat[d]) * c
               for d in dims
               for c in ([bmin[d], bmax[d]] if d < len(bmin) else [0, 0])]
    # recompute properly for arbitrary dims
    vals = []
    for combo in _corner_vals(bmin, bmax, n_hat):
        vals.append(combo)
    return min(vals) < d_plane <= max(vals)


def _corner_vals(bmin, bmax, n_hat):
    """Dot product of n_hat with all 8 corners of [bmin,bmax]."""
    vals = []
    for cx in [bmin[0], bmax[0]]:
        for cy in [bmin[1], bmax[1]]:
            for cz in [bmin[2], bmax[2]]:
                vals.append(n_hat[0]*cx + n_hat[1]*cy + n_hat[2]*cz)
    return vals


# ── Backus (laminate) homogenization of two symmetric tensors ─────────────────
def _backus_2region(sig1, sig2, f1, n_hat):
    """
    Exact Backus laminate homogenization of sig1 (volume frac f1) and sig2 (frac 1-f1)
    with layering normal n_hat.
      sigma_nn_eff  = 1 / (f1/nn1 + f2/nn2)           [harmonic mean in normal dir]
      sigma_Tn_eff  = <sigma_Tn/sigma_nn> * sigma_nn_eff [cross terms]
      sigma_TT_eff  = <Schur> + outer(avg_Tn/nn, avg_Tn/nn) * sigma_nn_eff
    where Schur_r = sig_r - outer(sig_r @ n, sig_r.T @ n) / sigma_nn_r.
    """
    f2 = 1.0 - f1
    n = n_hat / np.linalg.norm(n_hat)
    nn1 = float(np.real(n @ sig1 @ n))
    nn2 = float(np.real(n @ sig2 @ n))
    nn_eff = 1.0 / (f1/nn1 + f2/nn2)
    Tn1 = sig1 @ n; Tn2 = sig2 @ n
    avg_Tn_over_nn = f1 * Tn1/nn1 + f2 * Tn2/nn2
    S1 = sig1 - np.outer(Tn1, sig1.conj().T @ n) / nn1
    S2 = sig2 - np.outer(Tn2, sig2.conj().T @ n) / nn2
    return (f1*S1 + f2*S2) + nn_eff * np.outer(avg_Tn_over_nn, avg_Tn_over_nn)


# ── Strategy A: original (single-probe outer medium) ──────────────────────────
def build_media_original(grid):
    nx, ny, nz = len(grid.x), len(grid.y), len(grid.z)
    sr = np.zeros((grid.N_R, 3, 3), dtype=complex)
    for seq, (i, j, kk) in enumerate(grid.R_nodes):
        x = float(grid.x[i]); y = float(grid.y[j]); z = float(grid.z[kk])
        r = (x**2 + y**2)**0.5
        node = np.array([x, y, z])
        bmin = np.array([float(grid.x[max(i-1,0)]),   float(grid.y[max(j-1,0)]),   float(grid.z[max(kk-1,0)])])
        bmax = np.array([float(grid.x[min(i+1,nx-1)]), float(grid.y[min(j+1,ny-1)]), float(grid.z[min(kk+1,nz-1)])])
        sr[seq] = _pointwise_sigma(x, y, z); averaged = False

        if r > 1e-10:
            n_hat_r = np.array([x/r, y/r, 0.0])
            cd = [n_hat_r[0]*cx + n_hat_r[1]*cy
                  for cx in [bmin[0], bmax[0]] for cy in [bmin[1], bmax[1]]]
            if min(cd) < R_BORE <= max(cd):
                sr[seq] = _nodal_eff_tensor_general(
                    _S_BORE, _S_INV,
                    _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_BORE),
                    _line_frac(bmin, bmax, n_hat_r, R_BORE, node), n_hat_r)
                averaged = True
        if not averaged and r > 1e-10:
            n_hat_r = np.array([x/r, y/r, 0.0])
            cd = [n_hat_r[0]*cx + n_hat_r[1]*cy
                  for cx in [bmin[0], bmax[0]] for cy in [bmin[1], bmax[1]]]
            if min(cd) < R_INV <= max(cd):
                d_out = R_INV + 0.5*(bmax[0]-bmin[0])
                x_out = x + (d_out - r)*n_hat_r[0]
                y_out = y + (d_out - r)*n_hat_r[1]
                sr[seq] = _nodal_eff_tensor_general(
                    _S_INV, _outer_sigma(x_out, y_out, z),
                    _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_INV),
                    _line_frac(bmin, bmax, n_hat_r, R_INV, node), n_hat_r)
                averaged = True
        if not averaged and r >= R_INV:
            cdp = _corner_vals(bmin, bmax, N_HAT)
            if min(cdp) < D_PLANE <= max(cdp):
                sr[seq] = _nodal_eff_tensor_general(
                    _S_ANI, _S_ISO,
                    _volume_frac_layer1_planar(bmin, bmax, N_HAT, D_PLANE),
                    _line_frac(bmin, bmax, N_HAT, D_PLANE, node), N_HAT)
    return EMMedia(grid, sr, np.full(grid.N_P, complex(MU0)), np.full(grid.N_R, complex(EPS0)))


# ── Strategy B: improved (3-region nodal homogenization) ──────────────────────
def build_media_improved(grid):
    nx, ny, nz = len(grid.x), len(grid.y), len(grid.z)
    sr = np.zeros((grid.N_R, 3, 3), dtype=complex)
    n_double = 0   # count doubly-straddled cells

    for seq, (i, j, kk) in enumerate(grid.R_nodes):
        x = float(grid.x[i]); y = float(grid.y[j]); z = float(grid.z[kk])
        r = (x**2 + y**2)**0.5
        node = np.array([x, y, z])
        bmin = np.array([float(grid.x[max(i-1,0)]),   float(grid.y[max(j-1,0)]),   float(grid.z[max(kk-1,0)])])
        bmax = np.array([float(grid.x[min(i+1,nx-1)]), float(grid.y[min(j+1,ny-1)]), float(grid.z[min(kk+1,nz-1)])])
        sr[seq] = _pointwise_sigma(x, y, z); averaged = False

        if r > 1e-10:
            n_hat_r = np.array([x/r, y/r, 0.0])
            cd = [n_hat_r[0]*cx + n_hat_r[1]*cy
                  for cx in [bmin[0], bmax[0]] for cy in [bmin[1], bmax[1]]]
            if min(cd) < R_BORE <= max(cd):
                sr[seq] = _nodal_eff_tensor_general(
                    _S_BORE, _S_INV,
                    _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_BORE),
                    _line_frac(bmin, bmax, n_hat_r, R_BORE, node), n_hat_r)
                averaged = True

        if not averaged and r > 1e-10:
            n_hat_r = np.array([x/r, y/r, 0.0])
            cd = [n_hat_r[0]*cx + n_hat_r[1]*cy
                  for cx in [bmin[0], bmax[0]] for cy in [bmin[1], bmax[1]]]
            if min(cd) < R_INV <= max(cd):
                # Check whether the dipping plane also crosses this cell
                cdp = _corner_vals(bmin, bmax, N_HAT)
                dip_also_crosses = (min(cdp) < D_PLANE <= max(cdp))

                if dip_also_crosses:
                    # 3-region nodal homogenization with n̂ = n_hat_r
                    # Planes in order: invasion (inside=INV), then dipping (inside=ANI)
                    planes = [(n_hat_r, R_INV), (N_HAT, D_PLANE)]
                    lf, vf = _multiregion_line_and_vol_fracs(
                        bmin, bmax, planes, node)
                    # regions: 0=INV, 1=ANI, 2=ISO
                    sr[seq] = _nodal_eff_tensor_multiregion(
                        [_S_INV, _S_ANI, _S_ISO], vf, lf, n_hat_r)
                    n_double += 1
                else:
                    d_out = R_INV + 0.5*(bmax[0]-bmin[0])
                    x_out = x + (d_out - r)*n_hat_r[0]
                    y_out = y + (d_out - r)*n_hat_r[1]
                    sr[seq] = _nodal_eff_tensor_general(
                        _S_INV, _outer_sigma(x_out, y_out, z),
                        _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_INV),
                        _line_frac(bmin, bmax, n_hat_r, R_INV, node), n_hat_r)
                averaged = True

        if not averaged and r >= R_INV:
            cdp = _corner_vals(bmin, bmax, N_HAT)
            if min(cdp) < D_PLANE <= max(cdp):
                sr[seq] = _nodal_eff_tensor_general(
                    _S_ANI, _S_ISO,
                    _volume_frac_layer1_planar(bmin, bmax, N_HAT, D_PLANE),
                    _line_frac(bmin, bmax, N_HAT, D_PLANE, node), N_HAT)

    print(f"  Doubly-straddled cells handled with 3-region formula: {n_double}")
    return EMMedia(grid, sr, np.full(grid.N_P, complex(MU0)), np.full(grid.N_R, complex(EPS0)))


# ── Strategy E: Backus(ANI+ISO) outer pre-homogenization + nodal n̂=r̂ ─────────
def build_media_E(grid):
    """
    For doubly-straddled cells only:
      1. Pre-homogenize the outer region (ANI + ISO) using Backus with n̂ = N_HAT.
         Use relative fracs f_ANI/(f_ANI+f_ISO) and f_ISO/(f_ANI+f_ISO).
      2. Treat that Backus-effective outer tensor as a single medium and do
         nodal homogenization against INV with n̂ = n_hat_r (radial).
    All other cells are identical to Strategy A.
    """
    nx, ny, nz = len(grid.x), len(grid.y), len(grid.z)
    sr = np.zeros((grid.N_R, 3, 3), dtype=complex)
    n_double = 0

    for seq, (i, j, kk) in enumerate(grid.R_nodes):
        x = float(grid.x[i]); y = float(grid.y[j]); z = float(grid.z[kk])
        r = (x**2 + y**2)**0.5
        node = np.array([x, y, z])
        bmin = np.array([float(grid.x[max(i-1,0)]),    float(grid.y[max(j-1,0)]),    float(grid.z[max(kk-1,0)])])
        bmax = np.array([float(grid.x[min(i+1,nx-1)]), float(grid.y[min(j+1,ny-1)]), float(grid.z[min(kk+1,nz-1)])])
        sr[seq] = _pointwise_sigma(x, y, z); averaged = False

        if r > 1e-10:
            n_hat_r = np.array([x/r, y/r, 0.0])
            cd = [n_hat_r[0]*cx + n_hat_r[1]*cy
                  for cx in [bmin[0], bmax[0]] for cy in [bmin[1], bmax[1]]]
            if min(cd) < R_BORE <= max(cd):
                sr[seq] = _nodal_eff_tensor_general(
                    _S_BORE, _S_INV,
                    _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_BORE),
                    _line_frac(bmin, bmax, n_hat_r, R_BORE, node), n_hat_r)
                averaged = True

        if not averaged and r > 1e-10:
            n_hat_r = np.array([x/r, y/r, 0.0])
            cd = [n_hat_r[0]*cx + n_hat_r[1]*cy
                  for cx in [bmin[0], bmax[0]] for cy in [bmin[1], bmax[1]]]
            if min(cd) < R_INV <= max(cd):
                cdp = _corner_vals(bmin, bmax, N_HAT)
                dip_also_crosses = (min(cdp) < D_PLANE <= max(cdp))

                if dip_also_crosses:
                    # Step 1: vol/line fracs for all three regions
                    planes = [(n_hat_r, R_INV), (N_HAT, D_PLANE)]
                    lf, vf = _multiregion_line_and_vol_fracs(bmin, bmax, planes, node)
                    f_ani, f_iso = vf[1], vf[2]
                    f_outer = f_ani + f_iso
                    # Step 2: Backus pre-homogenize outer (ANI+ISO) with dipping normal
                    if f_outer > 1e-12:
                        f_ani_rel = f_ani / f_outer
                        sigma_outer = _backus_2region(_S_ANI, _S_ISO, f_ani_rel, N_HAT)
                    else:
                        sigma_outer = _S_ISO.copy()
                    # Step 3: nodal homogenization INV vs sigma_outer with n̂ = n_hat_r
                    vf_inv = _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_INV)
                    lf_inv = _line_frac(bmin, bmax, n_hat_r, R_INV, node)
                    sr[seq] = _nodal_eff_tensor_general(
                        _S_INV, sigma_outer, vf_inv, lf_inv, n_hat_r)
                    n_double += 1
                else:
                    d_out = R_INV + 0.5*(bmax[0]-bmin[0])
                    x_out = x + (d_out - r)*n_hat_r[0]
                    y_out = y + (d_out - r)*n_hat_r[1]
                    sr[seq] = _nodal_eff_tensor_general(
                        _S_INV, _outer_sigma(x_out, y_out, z),
                        _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_INV),
                        _line_frac(bmin, bmax, n_hat_r, R_INV, node), n_hat_r)
                averaged = True

        if not averaged and r >= R_INV:
            cdp = _corner_vals(bmin, bmax, N_HAT)
            if min(cdp) < D_PLANE <= max(cdp):
                sr[seq] = _nodal_eff_tensor_general(
                    _S_ANI, _S_ISO,
                    _volume_frac_layer1_planar(bmin, bmax, N_HAT, D_PLANE),
                    _line_frac(bmin, bmax, N_HAT, D_PLANE, node), N_HAT)

    print(f"  Doubly-straddled cells (Backus outer + nodal n̂=r̂): {n_double}")
    return EMMedia(grid, sr, np.full(grid.N_P, complex(MU0)), np.full(grid.N_R, complex(EPS0)))


# ── Shared solve + extraction ──────────────────────────────────────────────────
def run_crossing(grid, med, label):
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    bc = _component_aware_bc_dofs(grid)
    b_combined = build_rhs_multicl(grid, solver._C_PR, OMEGA)
    A = solver._A.copy()
    A_bc, b_bc = apply_electric_bc(A, b_combined, bc)
    E = factorized(A_bc)(b_bc)
    B = compute_B_from_E(grid, E, OMEGA)
    z_x, Bxx = extract_B_on_axis_multicl(grid, B, comp=0, axis='z')
    z_z, Bxz = extract_B_on_axis_multicl(grid, B, comp=2, axis='z')

    mx = (z_x >= Z_MIN) & (z_x <= Z_MAX)
    mz = (z_z >= Z_MIN) & (z_z <= Z_MAX)
    bxx = np.imag(Bxx[mx]) * 1e9
    bxz = np.interp(z_x[mx], z_z[mz], np.imag(Bxz[mz]) * 1e9)

    def _crossing(z, diff):
        sc = np.where(np.diff(np.sign(diff)))[0]
        if len(sc):
            zi = sc[0]
            return z[zi] - diff[zi]/(diff[zi+1]-diff[zi])*(z[zi+1]-z[zi])
        return None

    zc = _crossing(z_x[mx], bxx - bxz)
    print(f"  {label}: crossing Bxx=Bxz at z = "
          f"{f'{zc:.4f} m' if zc is not None else 'none in window'}")
    return zc, z_x[mx], bxx, bxz


# ── Main ───────────────────────────────────────────────────────────────────────
k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
t0 = time.time()
print(f"=== Media comparison  k={k} ===\n")

grid = symmetric_optimal_grid(H, 300., z_fd, GAMMA, k=k)
print(f"Grid: N_R={grid.N_R}  t={time.time()-t0:.1f}s\n")

print("Building media A (original single-probe)...")
ta = time.time()
med_A = build_media_original(grid)
print(f"  done  t={time.time()-ta:.1f}s")

print("Building media B (3-region nodal, n̂=r̂)...")
tb = time.time()
med_B = build_media_improved(grid)
print(f"  done  t={time.time()-tb:.1f}s")

print("Building media E (Backus outer + nodal n̂=r̂)...")
te = time.time()
med_E = build_media_E(grid)
print(f"  done  t={time.time()-te:.1f}s\n")

print("Solving A...")
zc_A, z, bxx_A, bxz_A = run_crossing(grid, med_A, "A original")

print("Solving B...")
zc_B, _, bxx_B, bxz_B = run_crossing(grid, med_B, "B 3-reg nodal n̂=r̂")

print("Solving E...")
zc_E, _, bxx_E, bxz_E = run_crossing(grid, med_E, "E Backus outer + nodal n̂=r̂")

print(f"\n  Shifts vs A:")
if zc_B is not None: print(f"    B - A: {(zc_B-zc_A)*100:+.1f} cm")
if zc_E is not None: print(f"    E - A: {(zc_E-zc_A)*100:+.1f} cm")

print(f"\n  {'z(m)':>8}  {'Bxx_A':>9}  {'Bxx_B':>9}  {'Bxx_E':>9}  "
      f"{'Bxz_A':>9}  {'Bxz_B':>9}  {'Bxz_E':>9}  (nT)")
for i in range(len(z)):
    print(f"  {z[i]:+8.4f}  {bxx_A[i]:+9.4f}  {bxx_B[i]:+9.4f}  {bxx_E[i]:+9.4f}  "
          f"  {bxz_A[i]:+9.4f}  {bxz_B[i]:+9.4f}  {bxz_E[i]:+9.4f}")

print(f"\nTotal time: {time.time()-t0:.1f}s")

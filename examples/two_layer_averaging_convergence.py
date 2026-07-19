"""
two_layer_averaging_convergence.py — transverse-grid (k) convergence of the
coupled-solve two-half-space benchmark for pointwise / backus / nodal.

Holds the z (axial) grid fixed and refines only the transverse optimal grid
(Mx = My = 4k).  If the ~1-3% bulk residual seen at k=3 is transverse
discretization error, it should shrink as k grows; the near-source and
interface-node errors (z-grid / source approximation / physical E_z
discontinuity) should NOT, since they do not depend on k.

Usage:
    python examples/two_layer_averaging_convergence.py [kmax]     # default kmax=5
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from lebedev_em.grid import (symmetric_optimal_grid, hybrid_axial_grid,
                             C000, C101, C110, C011)
from lebedev_em.media import from_geometry_exact, MU0, EPS0
from lebedev_em.geometry import PlanarBoundary, GeometryStack
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.analytics import electric_dipole_Ez_two_layer_onaxis
from lebedev_em.sources import _native_type_for_cluster_comp
from lebedev_em.postprocess import interpolate_cluster_E

SIGMA1, SIGMA2 = 0.1, 1.0
Z_SRC = 0.0
FREQ = 2500.0
OMEGA = 2.0 * np.pi * FREQ
MU, EPS = MU0, EPS0
GAMMA = 1.0 / np.sqrt(2.0)
Z_INNER_MIN, Z_INNER_MAX = -0.25, 7.75
N_INNER = int(os.environ.get("N_INNER", "128"))   # axial refinement (override to probe z-grid)
K_OUTER = 8
H_MIN, L_TRANS = 0.5, 300.0
Z_RECV_MIN, Z_RECV_MAX = 0.5, 7.5
METHODS = ("pointwise", "backus", "nodal")


def build_grid(k):
    z_fine = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)
    return symmetric_optimal_grid(H_MIN, L_TRANS, z_fine, GAMMA, k=k)


def make_sigma_func(z_cont):
    def sigma_func(X, Y, Z):
        X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
        shape = np.broadcast(X, Y, Z).shape
        out = np.zeros(shape + (3, 3), dtype=complex)
        out[...] = SIGMA1 * np.eye(3)
        out[Z >= z_cont] = SIGMA2 * np.eye(3)
        return out
    return sigma_func


def onaxis_ez_nodes(grid):
    Mx2, My2 = grid.Mx // 2, grid.My // 2
    nat = _native_type_for_cluster_comp(C000, 2)
    zlist, seqlist = [], []
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        if i == Mx2 and j == My2 and (i % 2, j % 2, k % 2) == nat:
            zv = float(grid.z[k])
            if Z_RECV_MIN <= zv <= Z_RECV_MAX:
                zlist.append(zv); seqlist.append(seq)
    z = np.array(zlist); seq = np.array(seqlist, dtype=int)
    o = np.argsort(z)
    return z[o], seq[o]


def lebedev_Ez(grid, result, z_eval, seq_c000):
    N_R = grid.N_R
    Ez = [np.array([result["E_c"][C000][2 * N_R + s] for s in seq_c000])]
    for c in (C101, C110, C011):
        Ez.append(np.array([
            interpolate_cluster_E(grid, result["E_c"][c], c, 2, 0.0, 0.0, zv)
            for zv in z_eval]))
    return np.mean(np.stack(Ez, axis=0), axis=0)


def rms_rel(fd, an):
    fd_re, an_re = np.real(fd), np.real(an)
    m = np.abs(an_re) > 0.01 * np.max(np.abs(an_re))
    return float(np.sqrt(np.mean(((fd_re[m] - an_re[m]) / an_re[m]) ** 2)))


def run_k(k):
    grid = build_grid(k)
    z_eval, seq_c000 = onaxis_ez_nodes(grid)
    k0 = int(np.argmin(np.abs(grid.z - 4.0)))
    Z_CONT = 0.5 * (grid.z[k0] + grid.z[k0 + 1])
    dz_cell = float(grid.z[k0 + 1] - grid.z[k0])
    geo = GeometryStack([PlanarBoundary(n_hat=[0.0, 0.0, 1.0], d=Z_CONT)])
    sf = make_sigma_func(Z_CONT)
    Ez_an = np.array([electric_dipole_Ez_two_layer_onaxis(
        zv, Z_SRC, Z_CONT, SIGMA1, SIGMA2, OMEGA, MU, EPS) for zv in z_eval])

    near_src = z_eval < 1.0
    iface = np.abs(z_eval - Z_CONT) < 1.5 * dz_cell
    bulk_L1 = (~near_src) & (~iface) & (z_eval < Z_CONT)
    bulk_L2 = (~iface) & (z_eval >= Z_CONT)

    out = {}
    for method in METHODS:
        med = from_geometry_exact(grid, sf, geo, method=method, h_svd=0.02)
        solver = LebedevMaxwellSolver(grid, med, omega=OMEGA)
        res = solver.solve(0.0, 0.0, Z_SRC, dipole_comp=2, moment=1.0)
        fd = lebedev_Ez(grid, res, z_eval, seq_c000)
        out[method] = dict(
            src=rms_rel(fd[near_src], Ez_an[near_src]),
            l1=rms_rel(fd[bulk_L1], Ez_an[bulk_L1]),
            l2=rms_rel(fd[bulk_L2], Ez_an[bulk_L2]))
    return grid, out


if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"{'k':>2} {'Mx':>3} {'3N_R':>7}   "
          + "  ".join(f"{m[:5]:>5}L1 {m[:5]:>5}L2" for m in METHODS)
          + "     src(shared)")
    for k in range(3, kmax + 1):
        t0 = time.time()
        grid, out = run_k(k)
        row = f"{k:>2} {grid.Mx:>3} {3*grid.N_R:>7}   "
        for m in METHODS:
            row += f"{out[m]['l1']*100:8.3f} {out[m]['l2']*100:8.3f} "
        row += f"    {out['pointwise']['src']*100:6.2f}%  ({time.time()-t0:.0f}s)"
        print(row, flush=True)

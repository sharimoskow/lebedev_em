"""
two_layer_averaging_probe.py — inspect the straddling-cell tensors for the
axis-aligned two-half-space model, built via from_geometry_exact for
method = pointwise / backus / nodal.

No solve; just confirms
  (a) the z = Z_CONT interface straddles a dual cell,
  (b) the averaged tensor there for each method,
  (c) has_offdiagonal_sigma is False (axis-aligned normal => diagonal).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid, C000
from lebedev_em.media import from_geometry_exact
from lebedev_em.geometry import PlanarBoundary, GeometryStack

SIGMA1 = 0.1
SIGMA2 = 1.0
Z_CONT = 3.9375   # mid-cell (between z-nodes 3.875 and 4.0) so a dual cell straddles

GAMMA = 1.0 / np.sqrt(2.0)


def sigma_func(X, Y, Z):
    X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
    shape = np.broadcast(X, Y, Z).shape
    out = np.zeros(shape + (3, 3), dtype=complex)
    out[...] = SIGMA1 * np.eye(3)
    out[Z >= Z_CONT] = SIGMA2 * np.eye(3)
    return out


def build_grid():
    # modest grid: interface at z=4 lands between nodes
    z_fine = hybrid_axial_grid(-0.25, 7.75, 64, 8, GAMMA)
    return symmetric_optimal_grid(0.5, 300.0, z_fine, GAMMA, k=2)


def main():
    grid = build_grid()
    geo = GeometryStack([PlanarBoundary(n_hat=[0.0, 0.0, 1.0], d=Z_CONT)])
    print(f"grid Mx={grid.Mx} My={grid.My} Mz={grid.Mz} N_R={grid.N_R}")

    # z-node just below and just above the interface
    zk = grid.z
    k_below = int(np.searchsorted(zk, Z_CONT) - 1)
    print(f"interface z={Z_CONT}: between z[{k_below}]={zk[k_below]:.4f} "
          f"and z[{k_below+1}]={zk[k_below+1]:.4f}")

    Mx2, My2 = grid.Mx // 2, grid.My // 2

    for method in ("pointwise", "backus", "nodal"):
        med = from_geometry_exact(grid, sigma_func, geo, method=method, h_svd=0.02)
        print(f"\n=== method={method}  has_offdiagonal_sigma="
              f"{med.has_offdiagonal_sigma} ===")
        # find the on-axis R-nodes near the interface and print their tensors
        for seq, (i, j, k) in enumerate(grid.R_nodes):
            if i == Mx2 and j == My2 and k in (k_below, k_below + 1):
                S = med.sigma_R[seq]
                d = np.real(np.diag(S))
                offmax = np.abs(S - np.diag(np.diag(S))).max()
                print(f"  node (i,j,k)=({i},{j},{k}) z={zk[k]:.4f}  "
                      f"diag={np.round(d,5)}  |offdiag|max={offmax:.2e}")


if __name__ == "__main__":
    main()

"""
test_anisotropic_coupling.py — Cross-component physics in anisotropic media.

Locks in three facts established during the July 2026 investigation:

1. Off-diagonal σ entries enter the assembled system as inter-cluster
   couplings at shared R-nodes, with value exactly −iω σ_offdiag, and the
   coupling correctly ANTI-commutes with the x-mirror (reflecting x flips
   the sign of σ_xz).

2. In a homogeneous tilted transversely-isotropic medium, the COUPLED solve
   (all-cluster sources, component-aware BCs) produces a nonzero
   cross-component Im B_xz on the axis — the physical triaxial coupling.

3. The per-cluster-source procedure with own-sub-grid extraction loses that
   cross-component (it lives on partner clusters' sub-grids) — documenting
   why anisotropic media must use the coupled solve.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lebedev_em.grid import symmetric_uniform_grid, C000, C101, C110, C011
from lebedev_em.media import EMMedia, MU0, EPS0
from lebedev_em.solver import (LebedevMaxwellSolver, _cluster_bc_dofs,
                               _component_aware_bc_dofs)
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_per_cluster,
                                    interpolate_cluster_B)

OMEGA = 2 * np.pi * 52650.0
TH = np.radians(60.0)
NH = np.array([np.sin(TH), 0.0, np.cos(TH)])
SIG_TILTED = 0.10 * np.eye(3) + (0.01 - 0.10) * np.outer(NH, NH)
CLUSTERS = [C000, C101, C110, C011]


def _media(grid, sig):
    sigma_R = np.zeros((grid.N_R, 3, 3), dtype=complex)
    sigma_R[:] = sig
    return EMMedia(grid, sigma_R,
                   np.full(grid.N_P, complex(MU0)),
                   np.full(grid.N_R, complex(EPS0)))


def test_offdiagonal_sigma_couples_clusters_with_correct_value():
    grid = symmetric_uniform_grid(Mx=6, My=6, Mz=6, Lx=3, Ly=3, Lz=3)
    N_R = grid.N_R
    A = LebedevMaxwellSolver(grid, _media(grid, SIG_TILTED), OMEGA)._A

    # constant E = z_hat probe: interior x-rows must return exactly -i w s_xz
    E = np.zeros(3 * N_R, dtype=complex)
    E[2 * N_R:] = 1.0
    v = A @ E
    expect = -1j * OMEGA * SIG_TILTED[0, 2]
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        if 1 < i < grid.Mx - 1 and 1 < j < grid.My - 1 and 1 < k < grid.Mz - 1:
            assert abs(v[seq] - expect) < 1e-6 * abs(expect)

    # mirror anti-commutation: T A T - A has support only on the coupling,
    # with entry magnitude exactly 2 w s_xz
    mirror = np.empty(3 * N_R, dtype=int)
    sign = np.empty(3 * N_R)
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        sm = int(grid.R_idx[grid.Mx - i, j, k])
        for comp in range(3):
            mirror[comp * N_R + seq] = comp * N_R + sm
            sign[comp * N_R + seq] = -1.0 if comp == 0 else 1.0
    P = sp.csr_matrix((sign, (np.arange(3 * N_R), mirror)), shape=A.shape)
    D = abs((P @ A @ P - A)).max()
    assert abs(D - 2 * OMEGA * abs(SIG_TILTED[0, 2])) < 1e-6 * D


def test_coupled_solve_has_cross_component_and_clustered_procedure_loses_it():
    # modest uniform grid; homogeneous tilted TI; x-dipole at center
    grid = symmetric_uniform_grid(Mx=10, My=10, Mz=10, Lx=4, Ly=4, Lz=4)
    med = _media(grid, SIG_TILTED)
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    rhs = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=0)
    zr = 1.2  # receiver on axis

    # --- coupled: all-cluster sources, component-aware BCs, one solve
    b = sum(rhs[c] for c in CLUSTERS)
    bc = _component_aware_bc_dofs(grid)
    A_bc, b_bc = apply_electric_bc(solver._A.copy(), b.copy(), bc)
    E = spla.spsolve(A_bc.tocsc(), b_bc)
    B = compute_B_from_E(grid, E, OMEGA)
    bx = np.mean([interpolate_cluster_B(grid, B, c, 0, 0.0, 0.0, zr)
                  for c in CLUSTERS]).imag
    bz = np.mean([interpolate_cluster_B(grid, B, c, 2, 0.0, 0.0, zr)
                  for c in CLUSTERS]).imag
    # physical cross-coupling: a substantial fraction of the co-component
    assert abs(bx) > 0
    assert abs(bz) > 0.15 * abs(bx), (
        f"coupled solve lost the cross-component: Bz={bz:.3e}, Bx={bx:.3e}")

    # --- per-cluster-source solves, own-sub-grid extraction (the pitfall)
    bz_diag = []
    for c in CLUSTERS:
        bc_c = _cluster_bc_dofs(grid, c)
        A_c, b_c = apply_electric_bc(solver._A.copy(), rhs[c].copy(), bc_c)
        E_c = spla.spsolve(A_c.tocsc(), b_c)
        B_c = compute_B_from_E(grid, E_c, OMEGA)
        bz_diag.append(interpolate_cluster_B(grid, B_c, c, 2, 0.0, 0.0, zr))
    bz_clustered = np.mean(bz_diag).imag
    assert abs(bz_clustered) < 0.05 * abs(bz), (
        "expected the clustered procedure to lose the cross-component "
        f"(got {bz_clustered:.3e} vs coupled {bz:.3e}); if this now passes "
        "with a large value, the extraction has been fixed — update this test")


def test_isotropic_limit_coupled_equals_clustered():
    grid = symmetric_uniform_grid(Mx=8, My=8, Mz=8, Lx=3, Ly=3, Lz=3)
    med = _media(grid, 0.2 * np.eye(3))
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    rhs = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=0)
    zr = 0.9

    b = sum(rhs[c] for c in CLUSTERS)
    bc = _component_aware_bc_dofs(grid)
    A_bc, b_bc = apply_electric_bc(solver._A.copy(), b.copy(), bc)
    B = compute_B_from_E(grid, spla.spsolve(A_bc.tocsc(), b_bc), OMEGA)
    bx_coupled = np.mean([interpolate_cluster_B(grid, B, c, 0, 0.0, 0.0, zr)
                          for c in CLUSTERS]).imag

    vals = []
    for c in CLUSTERS:
        bc_c = _cluster_bc_dofs(grid, c)
        A_c, b_c = apply_electric_bc(solver._A.copy(), rhs[c].copy(), bc_c)
        B_c = compute_B_from_E(grid, spla.spsolve(A_c.tocsc(), b_c), OMEGA)
        vals.append(interpolate_cluster_B(grid, B_c, c, 0, 0.0, 0.0, zr))
    bx_clustered = np.mean(vals).imag

    assert abs(bx_coupled - bx_clustered) < 5e-2 * abs(bx_coupled)

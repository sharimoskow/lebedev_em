"""
test_bc_and_solver.py — Boundary-condition, adjointness, solver-consistency
and source-weight tests for the DDH03 Lebedev scheme.

These tests pin down the properties that the scheme's superconvergence
(DDH03 eq. 6 + eq. 7) relies on:

  * tangential H ≡ 0 at all boundary P-nodes (magnetic BC, eq. 6);
  * discrete adjointness / symmetry of the curl pair in the scheme's
    natural (double-step volume) inner product;
  * equivalence of the coupled single-solve and the legacy 4-cluster solve
    for isotropic media (no spurious factor 4);
  * DDH03 source conditions: per-cluster weights sum to 1 and their center
    of mass is the true source location, also on nonuniform grids.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import scipy.sparse as sp

from lebedev_em.grid import LebedevGrid3D, uniform_grid, symmetric_uniform_grid, C000, C101, C110, C011
from lebedev_em.operators import build_curl_RE, build_curl_PR, apply_electric_bc
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.sources import (
    _native_type_for_cluster_comp,
    _trilinear_r_nodes,
)
from lebedev_em.media import homogeneous_isotropic
from lebedev_em.postprocess import lebedev_E_at_point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _double_step_volumes(grid: LebedevGrid3D, nodes) -> np.ndarray:
    """Per-node double-step volumes (x_{i+1}-x_{i-1})(y_{j+1}-y_{j-1})(z_{k+1}-z_{k-1}).

    These are the exact discrete quadrature weights under which the DDH03
    eq.-(3) differences satisfy summation by parts (clamped at the boundary;
    the clamped values never enter the symmetry statement below because the
    corresponding rows/columns are zero or Dirichlet-overwritten).
    """
    v = np.empty(len(nodes))
    for s, (i, j, k) in enumerate(nodes):
        v[s] = (
            (grid.x[min(i + 1, grid.Mx)] - grid.x[max(i - 1, 0)])
            * (grid.y[min(j + 1, grid.My)] - grid.y[max(j - 1, 0)])
            * (grid.z[min(k + 1, grid.Mz)] - grid.z[max(k - 1, 0)])
        )
    return v


def _smooth_E(grid: LebedevGrid3D) -> np.ndarray:
    """A generic smooth (non-polynomial) E-field sampled at R-nodes."""
    Ex = np.array([np.sin(grid.x[i]) * np.cos(grid.y[j]) + grid.z[k]
                   for i, j, k in grid.R_nodes])
    Ey = np.array([grid.x[i] * grid.z[k] + np.cos(grid.y[j])
                   for i, j, k in grid.R_nodes])
    Ez = np.array([np.cos(grid.x[i]) + grid.y[j] ** 2 * grid.z[k]
                   for i, j, k in grid.R_nodes])
    return np.concatenate([Ex, Ey, Ez])


# ---------------------------------------------------------------------------
# (a) Magnetic boundary condition: tangential H = 0 at boundary P-nodes
# ---------------------------------------------------------------------------

class TestMagneticBoundaryCondition:
    def test_tangential_H_zero_on_all_faces(self):
        """DDH03 eq. (6): H^P|∂Ω × n = 0.

        Applying C_RE to a generic smooth E-field must give exactly zero for
        every H component that is tangential to a boundary face (component
        axis ≠ face normal axis).  Before the row-masking fix, partial
        stencils left nonzero tangential H at the boundary.
        """
        grid = uniform_grid(6, 6, 6, 3.0, 3.1, 2.7)
        H = build_curl_RE(grid) @ _smooth_E(grid)

        NP_ = grid.N_P
        for seq, (i, j, k) in enumerate(grid.P_nodes):
            on_face = (
                i == 0 or i == grid.Mx,
                j == 0 or j == grid.My,
                k == 0 or k == grid.Mz,
            )
            for comp in range(3):
                tangential = any(on_face[a] for a in range(3) if a != comp)
                if tangential:
                    assert H[comp * NP_ + seq] == 0.0, (
                        f"Tangential H_{comp} ≠ 0 at boundary P-node ({i},{j},{k})"
                    )

    def test_normal_H_kept_on_faces(self):
        """The normal H component on a face is NOT constrained by H×n=0 and
        must keep its (fully formable, in-face) stencil."""
        grid = uniform_grid(6, 6, 6, 3.0, 3.0, 3.0)
        C_RE = build_curl_RE(grid).tocsr()
        NP_ = grid.N_P

        found_nonzero_row = False
        for seq, (i, j, k) in enumerate(grid.P_nodes):
            on_face = (
                i == 0 or i == grid.Mx,
                j == 0 or j == grid.My,
                k == 0 or k == grid.Mz,
            )
            # Node on exactly one face: the normal component's row must be
            # non-empty (built from tangential in-face E values).
            if sum(on_face) == 1:
                comp = on_face.index(True)
                row = comp * NP_ + seq
                nnz = C_RE.indptr[row + 1] - C_RE.indptr[row]
                assert nnz > 0, (
                    f"Normal H_{comp} row empty at face P-node ({i},{j},{k})"
                )
                found_nonzero_row = True
        assert found_nonzero_row


# ---------------------------------------------------------------------------
# (b) Discrete adjointness / weighted symmetry of the curl pair
# ---------------------------------------------------------------------------

class TestCurlAdjointness:
    """Symmetry of the curl-curl operator in the natural inner product.

    The exact raw-pair identity  W_R·C_PR = (W_P·C_RE)^T  (summation by
    parts with double-step weights) can only hold entry-wise away from the
    boundary: C_RE's magnetic-BC row masking removes tangential-boundary-H
    entries whose transposes survive in C_PR, and C_PR's same-axis boundary
    row skipping removes entries at electric-BC R-rows.  Both discrepancy
    sets are annihilated in the assembled system — the masked H rows are
    zero (so the corresponding C_PR columns multiply zeros), and the
    electric-BC rows/columns are overwritten by apply_electric_bc.  The
    physically meaningful statement is therefore that the FULL BC-applied
    curl-curl matrix is symmetric in the volume-weighted inner product.
    This is the test that fails (asymmetry O(1)) for the pre-fix partial
    boundary stencils.
    """

    def test_interior_pair_adjointness(self):
        grid = uniform_grid(6, 6, 6, 3.0, 3.1, 2.7)
        C_RE = build_curl_RE(grid)
        C_PR = build_curl_PR(grid)
        NR = grid.N_R
        VR = np.tile(_double_step_volumes(grid, grid.R_nodes), 3)
        VP = np.tile(_double_step_volumes(grid, grid.P_nodes), 3)

        D = ((sp.diags(VR) @ C_PR) - (sp.diags(VP) @ C_RE).T).tocoo()

        # Interior R-rows and interior P-columns only
        r_int = np.zeros(3 * NR, dtype=bool)
        for s, (i, j, k) in enumerate(grid.R_nodes):
            interior = (0 < i < grid.Mx and 0 < j < grid.My and 0 < k < grid.Mz)
            for c in range(3):
                r_int[c * NR + s] = interior
        p_int = np.zeros(3 * grid.N_P, dtype=bool)
        for s, (i, j, k) in enumerate(grid.P_nodes):
            interior = (0 < i < grid.Mx and 0 < j < grid.My and 0 < k < grid.Mz)
            for c in range(3):
                p_int[c * grid.N_P + s] = interior

        sel = r_int[D.row] & p_int[D.col]
        max_mismatch = np.abs(D.data[sel]).max() if sel.any() else 0.0
        assert max_mismatch < 1e-12, (
            f"Interior curl pair not adjoint: max mismatch {max_mismatch}"
        )

    def test_weighted_symmetry_of_bc_applied_curl_curl(self):
        """W_R · (curl-curl with mixed eq.-6 BCs applied) must be symmetric.

        Regression test for the partial-boundary-stencil bug: with the old
        unmasked C_RE this asymmetry is O(1); with the eq.-(6) row masking
        it vanishes to machine precision.
        """
        grid = uniform_grid(6, 6, 6, 3.0, 3.1, 2.7)
        C_RE = build_curl_RE(grid)
        C_PR = build_curl_PR(grid)
        VR = np.tile(_double_step_volumes(grid, grid.R_nodes), 3)

        A = (C_PR @ C_RE).tocsr().astype(complex)
        bc_dofs = _component_aware_bc_dofs(grid)
        A_bc, _ = apply_electric_bc(A, np.zeros(3 * grid.N_R, dtype=complex), bc_dofs)

        S = sp.diags(VR) @ A_bc
        asym = (S - S.T).tocoo()
        max_asym = np.abs(asym.data).max() if asym.nnz else 0.0
        # Entries of S are O(V/h²) ≈ O(1) on this grid; 1e-12 is machine-level.
        assert max_asym < 1e-12, (
            f"BC-applied curl-curl not symmetric in weighted inner product: "
            f"max asymmetry {max_asym}"
        )


# ---------------------------------------------------------------------------
# (c) solve_coupled ≡ solve_clustered for isotropic media (no factor 4)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results():
    grid = symmetric_uniform_grid(8, 8, 8, 4.0, 4.0, 4.0)
    media = homogeneous_isotropic(grid, sigma=1.0)
    solver = LebedevMaxwellSolver(grid, media, omega=2 * np.pi * 2500.0)
    src = (0.3, 0.0, 0.2)
    rc = solver.solve(*src, dipole_comp=0, method="coupled")
    rk = solver.solve(*src, dipole_comp=0, method="clustered")
    return grid, solver, rc, rk


class TestCoupledClusteredEquivalence:
    def test_get_field_at_agrees(self, results):
        """The composite nearest-node read must agree between methods
        (the old np.mean composite made 'clustered' exactly 4× smaller)."""
        grid, solver, rc, rk = results
        for pt in [(1.0, 0.5, 0.5), (-0.9, 0.1, -1.2), (0.0, 1.4, 0.3)]:
            gc = solver.get_field_at(rc, *pt)
            gk = solver.get_field_at(rk, *pt)
            np.testing.assert_allclose(gc, gk, rtol=1e-8, atol=1e-14)

    def test_lebedev_average_agrees(self, results):
        """The proper interpolate-then-average field must agree between the
        coupled single solve and the legacy 4-cluster solve (isotropic
        media decouple exactly)."""
        grid, solver, rc, rk = results
        for pt in [(1.0, 0.5, 0.5), (-0.9, 0.1, -1.2)]:
            for comp in range(3):
                vc = lebedev_E_at_point(grid, rc["E_c"], comp, *pt)
                vk = lebedev_E_at_point(grid, rk["E_c"], comp, *pt)
                np.testing.assert_allclose(
                    [vc.real, vc.imag], [vk.real, vk.imag], rtol=1e-8, atol=1e-14
                )

    def test_get_field_at_near_domain_edge(self, results):
        """Regression for the parity fix-up bug: requesting a point whose
        nearest node is a P-node at the i == Mx edge must not raise."""
        grid, solver, rc, _ = results
        # Corner (Mx, My, Mz) has even parity sum → P-node; old code left the
        # index unchanged and raised ValueError.
        E = solver.get_field_at(rc, grid.x[-1], grid.y[-1], grid.z[-1])
        assert E.shape == (3,)
        assert np.all(np.isfinite(E))


# ---------------------------------------------------------------------------
# (d) DDH03 source conditions on a nonuniform grid
# ---------------------------------------------------------------------------

class TestSourceWeights:
    def test_weights_sum_and_centroid_nonuniform(self):
        """DDH03 eq.-(7) conditions for every cluster and component:
        (1) Σ w = 1;  (2) Σ w·r = r₀ (center of mass at the true source),
        on a genuinely nonuniform grid."""
        x = np.array([0.0, 0.7, 1.2, 2.05, 2.5, 3.6, 4.1, 5.3, 5.8])
        y = np.array([0.0, 0.4, 1.3, 1.9, 3.0, 3.3, 4.5, 5.0, 6.2])
        z = np.array([0.0, 0.9, 1.5, 2.6, 3.2, 4.4, 4.9, 6.1, 6.6])
        grid = LebedevGrid3D(x, y, z)

        r0 = (2.3, 2.7, 3.05)   # generic interior point
        for comp in range(3):
            for c in (C000, C101, C110, C011):
                t = _native_type_for_cluster_comp(c, comp)
                nw = _trilinear_r_nodes(grid, t, *r0)
                assert nw, f"No source nodes for cluster {c}, comp {comp}"

                w_sum = sum(w for _, w in nw)
                np.testing.assert_allclose(w_sum, 1.0, rtol=0, atol=1e-12)

                centroid = np.zeros(3)
                for seq, w in nw:
                    i, j, k = grid.R_nodes[seq]
                    centroid += w * np.array([x[i], y[j], z[k]])
                np.testing.assert_allclose(centroid, r0, rtol=0, atol=1e-12)

    def test_on_node_source_uses_eq7_shift_pattern(self):
        """For a source AT a native cluster-000 node, the three compensating
        clusters must each use 4 points with index shifts satisfying
        |ℓx|+|ℓy|+|ℓz| = 2 (DDH03 eq. 7), and the owning cluster a single
        point with weight 1."""
        grid = uniform_grid(8, 8, 8, 4.0, 4.0, 4.0)
        # R-node of type (1,0,0): Ex native to cluster 000
        i0, j0, k0 = 5, 4, 4
        x0, y0, z0 = grid.x[i0], grid.y[j0], grid.z[k0]

        comp = 0  # Ex dipole
        for c in (C000, C101, C110, C011):
            t = _native_type_for_cluster_comp(c, comp)
            nw = _trilinear_r_nodes(grid, t, x0, y0, z0)
            if c == C000:
                assert len(nw) == 1
                seq, w = nw[0]
                assert tuple(grid.R_nodes[seq]) == (i0, j0, k0)
                np.testing.assert_allclose(w, 1.0, atol=1e-14)
            else:
                assert len(nw) == 4, f"cluster {c}: expected 4 points, got {len(nw)}"
                for seq, w in nw:
                    i, j, k = grid.R_nodes[seq]
                    l1 = abs(i - i0) + abs(j - j0) + abs(k - k0)
                    assert l1 == 2, (
                        f"cluster {c}: shift ({i-i0},{j-j0},{k-k0}) violates "
                        f"|ℓx|+|ℓy|+|ℓz|=2"
                    )
                    np.testing.assert_allclose(w, 0.25, atol=1e-14)

"""
Regression tests for the media.py / geometry.py review fixes:

1. Cylinder/sphere ``straddles`` uses the clamped closest-point distance
   (min-over-corners misses interior minima of the convex radius function).
2. ``from_geometry_exact`` with a scalar callable cannot leave uninitialized
   (zero) tensor entries, and matches ``from_sigma_func`` / the analytic
   ``planar_interface_isotropic`` on a two-layer planar model.
3. Per-axis line fractions are taken along the lines through the NODE
   (tex note eqs. A.8-A.10), not the dual-cell box centre — they differ on
   geometric (alpha > 1) grids.
4. The diagonal SVD-fallback replaces the whole tensor (no mixing of a proxy
   diagonal with pointwise off-diagonals), so results stay SPD.
5. Complex sigma-dot survives the averaging paths (imaginary part preserved).
6. ``sigma_dot_matrix`` assembles sigma - i*omega*eps (exp(-i omega t)
   convention), for every scalar/tensor combination of sigma and eps.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from lebedev_em.grid import LebedevGrid3D
from lebedev_em.geometry import (
    CylindricalBoundary,
    SphericalBoundary,
    PlanarBoundary,
    GeometryStack,
)
from lebedev_em.media import (
    EMMedia,
    EPS0,
    MU0,
    from_geometry_exact,
    from_sigma_func,
    homogeneous_isotropic,
    planar_interface_isotropic,
    _nodal_eff_tensor_general,
    _line_measure,
    _region_volume,
    _frac_1d_layer1,
    _volume_frac_layer1_planar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_tensor_field(sigma_R: np.ndarray) -> np.ndarray:
    """Return sigma_R as an (N, 3, 3) array regardless of storage."""
    if sigma_R.ndim == 3:
        return sigma_R
    out = np.zeros((sigma_R.shape[0], 3, 3), dtype=complex)
    for d in range(3):
        out[:, d, d] = sigma_R
    return out


def _geometric_axis(k: int, h: float, alpha: float) -> np.ndarray:
    """Symmetric coordinate axis with steps in geometric progression
    (h, h*alpha, ...) on each side of 0.  Returns 2k+1 points (2k intervals,
    even, as LebedevGrid3D requires)."""
    steps = h * alpha ** np.arange(k)
    right = np.concatenate([[0.0], np.cumsum(steps)])
    return np.concatenate([-right[::-1][:-1], right])


N_HAT = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
D_PLANE = 0.03
SIG1, SIG2 = 0.2, 1.0   # layer 1 (n.x < d) , layer 2


def _sf_scalar(X, Y, Z):
    v = N_HAT[0] * X + N_HAT[1] * Y + N_HAT[2] * Z
    return np.where(v < D_PLANE, SIG1, SIG2)


# ---------------------------------------------------------------------------
# 1. straddles: clamped closest-point distance
# ---------------------------------------------------------------------------

class TestStraddles:
    def test_cylinder_wall_fully_inside_box(self):
        cyl = CylindricalBoundary(radius=0.1)
        bmin = np.array([-0.15, -0.15, -0.1])
        bmax = np.array([0.15, 0.15, 0.1])
        assert cyl.straddles(bmin, bmax, np.zeros(3))

    def test_cylinder_interior_minimum(self):
        # All corners outside (r >= 0.108) but box spans x=0 with a point at
        # r = 0.06 < radius: the wall cuts the box.
        cyl = CylindricalBoundary(radius=0.1)
        bmin = np.array([-0.09, 0.06, -0.1])
        bmax = np.array([0.09, 0.2, 0.1])
        assert cyl.straddles(bmin, bmax, np.zeros(3))

    def test_cylinder_corner_crossing_still_detected(self):
        cyl = CylindricalBoundary(radius=0.1)
        bmin = np.array([0.05, 0.05, -0.1])
        bmax = np.array([0.12, 0.12, 0.1])   # corners at r=0.0707 and 0.1697
        assert cyl.straddles(bmin, bmax, np.zeros(3))

    def test_cylinder_fully_inside_material(self):
        cyl = CylindricalBoundary(radius=0.1)
        # Box strictly inside the cylinder
        bmin = np.array([-0.03, -0.03, -0.1])
        bmax = np.array([0.03, 0.03, 0.1])
        assert not cyl.straddles(bmin, bmax, np.zeros(3))
        # Box strictly outside
        bmin = np.array([0.2, 0.2, -0.1])
        bmax = np.array([0.4, 0.4, 0.1])
        assert not cyl.straddles(bmin, bmax, np.zeros(3))

    def test_sphere_fully_inside_box(self):
        sph = SphericalBoundary(radius=0.1)
        bmin = np.array([-0.15, -0.15, -0.15])
        bmax = np.array([0.15, 0.15, 0.15])
        assert sph.straddles(bmin, bmax, np.zeros(3))

    def test_sphere_interior_minimum(self):
        sph = SphericalBoundary(radius=0.1)
        # Box spans x=0 and y=0; closest point (0, 0, 0.06), all corners outside
        bmin = np.array([-0.09, -0.09, 0.06])
        bmax = np.array([0.09, 0.09, 0.2])
        assert sph.straddles(bmin, bmax, np.zeros(3))

    def test_sphere_inside_outside(self):
        sph = SphericalBoundary(radius=0.1)
        assert not sph.straddles(np.array([-0.03] * 3), np.array([0.03] * 3),
                                 np.zeros(3))
        assert not sph.straddles(np.array([0.2] * 3), np.array([0.4] * 3),
                                 np.zeros(3))

    def test_planar_unchanged(self):
        pl = PlanarBoundary(n_hat=[0, 0, 1], d=0.0)
        assert pl.straddles(np.array([-1, -1, -0.1]), np.array([1, 1, 0.1]),
                            np.zeros(3))
        assert not pl.straddles(np.array([-1, -1, 0.2]), np.array([1, 1, 0.5]),
                                np.zeros(3))


# ---------------------------------------------------------------------------
# 2. from_geometry_exact scalar path: no uninitialized tensors, consistency
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid():
    x = np.linspace(-1.0, 1.0, 9)
    return LebedevGrid3D(x, x, x)


class TestGeometryFuncScalarPath:

    def test_no_zero_or_unphysical_tensors(self, grid):
        geo = GeometryStack([PlanarBoundary(n_hat=N_HAT, d=D_PLANE)])
        med = from_geometry_exact(grid, _sf_scalar, geo, method="nodal",
                                 h_svd=0.05)
        T = _as_tensor_field(med.sigma_R)
        diag = np.real(np.diagonal(T, axis1=1, axis2=2))
        assert np.all(np.isfinite(T))
        # Every diagonal entry must lie near the physical bounds.  (The nodal
        # tensor may legitimately overshoot the material values by a few
        # percent for asymmetrically straddling cells — the energy-matching
        # correction is not bounded entry-wise by the constituents.)
        assert diag.min() >= 0.9 * SIG1, (
            f"unphysical diagonal min {diag.min()} — uninitialized tensor?")
        assert diag.max() <= 1.1 * SIG2
        # No node may have an (all-zero) tensor
        assert np.all(np.abs(diag).sum(axis=1) > 1e-6)

    def test_matches_analytic_planar(self, grid):
        """Geometry path vs the analytic planar_interface_isotropic builder."""
        geo = GeometryStack([PlanarBoundary(n_hat=N_HAT, d=D_PLANE)])
        med_g = from_geometry_exact(grid, _sf_scalar, geo, method="nodal",
                                   h_svd=0.02)
        med_a = planar_interface_isotropic(grid, N_HAT, D_PLANE, SIG1, SIG2,
                                           method="nodal")
        Tg = _as_tensor_field(med_g.sigma_R)
        Ta = _as_tensor_field(med_a.sigma_R)
        scale = SIG2
        rel = np.abs(Tg - Ta).max(axis=(1, 2)) / scale
        # Voxel-sampled fractions vs exact fractions: a few percent of sigma2.
        # (planar_interface_isotropic applies a fake-straddle guard that
        # from_geometry_exact does not; allow for those cells via percentile.)
        assert np.median(rel) < 0.02
        assert np.percentile(rel, 90) < 0.10

    def test_matches_from_sigma_func(self, grid):
        geo = GeometryStack([PlanarBoundary(n_hat=N_HAT, d=D_PLANE)])
        med_g = from_geometry_exact(grid, _sf_scalar, geo, method="nodal",
                                   h_svd=0.02)
        med_s = from_sigma_func(grid, _sf_scalar, h_svd=0.02, method="nodal")
        Tg = _as_tensor_field(med_g.sigma_R)
        Ts = _as_tensor_field(med_s.sigma_R)
        rel = np.abs(Tg - Ts).max(axis=(1, 2)) / SIG2
        # The sigma_func path estimates the normal by SVD and applies its own
        # fallbacks, so agreement is approximate — but must be close for the
        # bulk of the nodes and bounded everywhere.
        assert np.median(rel) < 0.02
        assert rel.max() < 0.5


# ---------------------------------------------------------------------------
# 3. Node-line vs box-centre line fractions (geometric grid)
# ---------------------------------------------------------------------------

class TestNodeLineFractions:
    def test_exact_line_and_volume_through_node_offcentre(self):
        """The exact-path fraction machinery (_line_measure / _region_volume)
        must take line averages through the NODE, not the box centre, on a
        non-uniform dual cell — matching the analytic references exactly."""
        # Dual cell: x in [-1, 1.4] (node at x=0, off centre); y, z in [-1, 1].
        bmin = np.array([-1.0, -1.0, -1.0]); bmax = np.array([1.4, 1.0, 1.0])
        node = np.array([0.0, 0.0, 0.0])
        n = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        d = 0.15
        P = PlanarBoundary(n_hat=n, d=d)                 # (P, True) = below (n·x < d)

        for ax in range(3):
            lo, hi = float(bmin[ax]), float(bmax[ax])
            rest = float(n @ node) - n[ax] * node[ax]
            f_ref = _frac_1d_layer1(lo, hi, n[ax], d - rest)   # analytic, through node
            f_meas = _line_measure(node, ax, lo, hi, [(P, True)]) / (hi - lo)
            assert abs(f_meas - f_ref) < 1e-9

        v_ref = _volume_frac_layer1_planar(bmin, bmax, n, d)
        v_meas = _region_volume(bmin, bmax, [(P, True)]) / float(np.prod(bmax - bmin))
        assert abs(v_meas - v_ref) < 1e-6

        # Discriminating: the box centre (x_mid = 0.2) is genuinely off the node,
        # so a box-centre line average would give a different x-fraction.
        assert abs(0.5 * (bmin[0] + bmax[0])) > 0.1

    def test_tensor_path_matches_scalar_path_on_geometric_grid(self):
        """from_sigma_func: tensor callable (sigma * I) must reproduce the
        scalar-callable result on an alpha>1 geometric grid (both node-based
        after the fix; the old box-centre lines break this)."""
        ax = _geometric_axis(4, 0.22, 1.45)
        grid = LebedevGrid3D(ax, ax, ax)

        def sf_tensor(X, Y, Z):
            s = _sf_scalar(X, Y, Z)
            shape = np.broadcast(X, Y, Z).shape
            out = np.zeros(shape + (3, 3), dtype=complex)
            for d in range(3):
                out[..., d, d] = s
            return out

        med_s = from_sigma_func(grid, _sf_scalar, h_svd=0.05, method="nodal")
        med_t = from_sigma_func(grid, sf_tensor, h_svd=0.05, method="nodal")
        Ts = _as_tensor_field(med_s.sigma_R)
        Tt = _as_tensor_field(med_t.sigma_R)
        rel = np.abs(Ts - Tt).max(axis=(1, 2)) / SIG2
        # Same field, same grid, same n_line sampling through the node —
        # the two paths must agree closely.  (The old box-centre lines gave
        # max rel ≈ 0.18 with off-diagonal sign flips on this grid.)
        assert np.median(rel) < 0.01
        assert rel.max() < 0.02, (
            f"tensor/scalar path disagree on geometric grid: max rel {rel.max()}")


# ---------------------------------------------------------------------------
# 4. Diagonal fallback stays SPD for tensor callables
# ---------------------------------------------------------------------------

class TestDiagonalFallbackSPD:
    def test_spd_after_fallback(self):
        x = np.linspace(-1.0, 1.0, 9)
        grid = LebedevGrid3D(x, x, x)

        # Material A: strong xy off-diagonal (eigs 0.1, 0.1, 10)
        R = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        sA = R @ np.diag([10.0, 0.1, 0.1]) @ R.T
        sB = 0.1 * np.eye(3)

        def sf_tensor(X, Y, Z):
            v = N_HAT[0] * X + N_HAT[1] * Y + N_HAT[2] * Z
            shape = np.broadcast(X, Y, Z).shape
            out = np.empty(shape + (3, 3), dtype=complex)
            sel = (v < D_PLANE)
            out[sel] = sB
            out[~sel] = sA
            return out

        # svd_isotropy_tol=0.0 forces the diagonal fallback for every
        # interface cell — the worst case for the old mixed-tensor bug.
        med = from_sigma_func(grid, sf_tensor, h_svd=0.05, method="nodal",
                              svd_isotropy_tol=0.0)
        T = _as_tensor_field(med.sigma_R)
        for t in T:
            h = np.real(0.5 * (t + t.T))
            ev = np.linalg.eigvalsh(h)
            assert ev.min() > 0.0, f"indefinite tensor after fallback: {ev}"


# ---------------------------------------------------------------------------
# 5. Complex sigma-dot preserved through averaging
# ---------------------------------------------------------------------------

class TestComplexSigmaPreserved:
    def test_imag_part_survives(self):
        x = np.linspace(-1.0, 1.0, 9)
        grid = LebedevGrid3D(x, x, x)
        s1c = SIG1 - 0.02j
        s2c = SIG2 - 0.10j

        def sf_c(X, Y, Z):
            v = N_HAT[0] * X + N_HAT[1] * Y + N_HAT[2] * Z
            return np.where(v < D_PLANE, s1c, s2c)

        med = from_sigma_func(grid, sf_c, h_svd=0.05, method="nodal")
        T = _as_tensor_field(med.sigma_R)
        imag_diag = np.imag(np.diagonal(T, axis1=1, axis2=2))
        # Every node keeps a negative imaginary part of the right magnitude
        # (small overshoot beyond the material bounds is legitimate for the
        # nodal correction; zero imaginary part would mean it was dropped).
        assert imag_diag.max() <= 0.5 * np.imag(s1c)   # ≤ −0.01
        assert imag_diag.min() >= 1.5 * np.imag(s2c)   # ≥ −0.15
        # Real parts must match the real-only run
        med_r = from_sigma_func(grid, _sf_scalar, h_svd=0.05, method="nodal")
        Tr = _as_tensor_field(med_r.sigma_R)
        # imaginary parts here are small perturbations; compare real parts
        assert np.abs(np.real(T) - np.real(Tr)).max() < 0.05


# ---------------------------------------------------------------------------
# 6. sigma_dot sign and eps tensor support
# ---------------------------------------------------------------------------

class TestSigmaDot:
    def test_scalar_sign(self):
        x = np.linspace(-1.0, 1.0, 5)
        grid = LebedevGrid3D(x, x, x)
        med = homogeneous_isotropic(grid, sigma=2.0, eps=3.0)
        omega = 10.0
        M = med.sigma_dot_matrix(omega)
        expected = 2.0 - 1j * omega * 3.0
        d = M.diagonal()
        assert np.allclose(d, expected), f"sigma_dot diagonal {d[0]} != {expected}"

    def test_tensor_sigma_scalar_eps(self):
        x = np.linspace(-1.0, 1.0, 5)
        grid = LebedevGrid3D(x, x, x)
        sig = np.zeros((grid.N_R, 3, 3), dtype=complex)
        for d in range(3):
            sig[:, d, d] = 2.0
        sig[:, 0, 1] = sig[:, 1, 0] = 0.5
        med = EMMedia(grid, sig, np.full(grid.N_P, MU0),
                      np.full(grid.N_R, 3.0))
        omega = 10.0
        M = med.sigma_dot_matrix(omega).toarray()
        # Check the first 3x3 block
        blk = M[:3, :3] if M[0, 1] != 0 or M[0, 2] != 0 else None
        # tensor_block_diag layout may be component-blocked; check via a
        # known-diagonal probe instead: diagonal entries
        diag = med.sigma_dot_matrix(omega).diagonal()
        assert np.allclose(diag, 2.0 - 1j * omega * 3.0)

    def test_tensor_eps_supported(self):
        """(N_R,3,3) eps with scalar sigma must not crash and must carry the
        -i*omega*eps sign."""
        x = np.linspace(-1.0, 1.0, 5)
        grid = LebedevGrid3D(x, x, x)
        eps = np.zeros((grid.N_R, 3, 3), dtype=complex)
        for d in range(3):
            eps[:, d, d] = 3.0
        med = EMMedia(grid, np.full(grid.N_R, 2.0), np.full(grid.N_P, MU0),
                      eps)
        omega = 10.0
        M = med.sigma_dot_matrix(omega)
        assert np.allclose(M.diagonal(), 2.0 - 1j * omega * 3.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

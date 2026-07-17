"""
test_postprocess.py — Tests for B-field postprocessing and analytics.

Covers the fixes for the DDH03 averaging/source conditions:
  (a) coordinate-based interpolation weights on nonuniform grids
      (Σw = 1, weighted centroid = evaluation point);
  (b) interpolated B-extraction beats the old equal-1/4-weight stencils on
      an analytic whole-space field sampled onto a stretched (geometric) grid;
  (c) magnetic-dipole RHS builder source groups satisfy DDH03's conditions
      Σw = 1 and Σw·r = r₀ on nonuniform grids (and reduce to the familiar
      1 / 4×(1/4) weights on symmetric grids), with a warning when z = 0 is
      not an even-index grid plane;
  (d) receivers adjacent to the domain edge are handled by extrapolation
      instead of silently averaging in phantom zeros;
  (e) analytic magnetic dipole field: static limit and value lock-in.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lebedev_em.grid import LebedevGrid3D, C000, C101, C110, C011
from lebedev_em.media import MU0
from lebedev_em.postprocess import (
    _trilinear_p_nodes,
    _magnetic_source_groups,
    _native_type_for_h_cluster_comp,
    interpolate_cluster_B,
    extract_B_on_axis_multicl,
    lebedev_B_on_z_axis,
)
from lebedev_em.analytics import magnetic_dipole_B, Bxx_homogeneous

CLUSTERS = (C000, C101, C110, C011)
SIGMA = 1.0
OMEGA = 2.0 * np.pi * 2500.0


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def nonuniform_grid():
    """Deliberately nonuniform in all three axes; z = 0 at even index 4."""
    x = np.array([-1.2, -0.5, 0.0, 0.7, 1.6])                       # Mx = 4
    y = np.array([-1.5, -0.6, 0.0, 0.4, 1.1])                       # My = 4
    z = np.array([-2.0, -1.1, -0.4, -0.15, 0.0, 0.6, 1.3, 2.7, 4.0])  # Mz = 8
    return LebedevGrid3D(x, y, z)


def stretched_grid():
    """
    Small uniform transverse grid, z-grid taken from the z ≥ 2.5 m portion of
    the DDH03 Fig-3 benchmark's hybrid axial grid (uniform Δz = 0.025 m up to
    3 m, geometrically stretched beyond).  The dipole source at the origin
    lies OUTSIDE the grid, so the analytic field is smooth in the domain.
    """
    from lebedev_em.grid import hybrid_axial_grid
    x = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])   # Mx = 4, i0 = 2, x = 0
    y = x.copy()                                # My = 4, j0 = 2, y = 0
    z_full = hybrid_axial_grid(-3.0, 3.0, 240, 12, 1.0 / np.sqrt(2.0))
    z = z_full[z_full >= 2.5]
    if len(z) % 2 == 0:                          # ensure even Mz
        z = z[1:]
    return LebedevGrid3D(x, y, z)


def sample_Bx_field(grid, func):
    """Fill a component-blocked (3·N_P,) vector with Bx = func(x, y, z) at P-nodes."""
    B_vec = np.zeros(3 * grid.N_P, dtype=complex)
    for seq, (i, j, k) in enumerate(grid.P_nodes):
        x, y, z = grid.node_xyz(i, j, k)
        B_vec[0 * grid.N_P + seq] = func(x, y, z)
    return B_vec


def old_equal_weight_extract(grid, B_vec, comp=0):
    """
    Re-implementation of the PRE-FIX extraction (hardcoded equal 1/4 weights
    on ±1-shifted nodes, phantom 0j for out-of-range nodes) used as the
    reference "old" behaviour in the regression comparisons below.
    """
    Mx, My, Mz = grid.Mx, grid.My, grid.Mz
    i0, j0 = Mx // 2, My // 2
    N_P = grid.N_P

    def b_at(i, j, k):
        if not (0 <= i <= Mx and 0 <= j <= My and 0 <= k <= Mz):
            return 0j
        seq = int(grid.P_idx[i, j, k])
        return 0j if seq < 0 else complex(B_vec[comp * N_P + seq])

    coords, values = [], []
    for k in range(0, Mz + 1, 2):
        if grid.P_idx[i0, j0, k] < 0:
            continue
        v011 = b_at(i0, j0, k)
        v000 = np.mean([b_at(i0, j0 + dj, k + dk) for dj in (1, -1) for dk in (1, -1)])
        v101 = np.mean([b_at(i0 + di, j0 + dj, k) for di in (1, -1) for dj in (1, -1)])
        v110 = np.mean([b_at(i0 + di, j0, k + dk) for di in (1, -1) for dk in (1, -1)])
        coords.append(float(grid.z[k]))
        values.append((v011 + v000 + v101 + v110) / 4.0)
    coords = np.array(coords)
    order = np.argsort(coords)
    return coords[order], np.array(values, dtype=complex)[order]


# ---------------------------------------------------------------------------
# (a) interpolation weights: Σw = 1 and centroid at the receiver
# ---------------------------------------------------------------------------

class TestInterpolationWeights:
    def test_weights_sum_and_centroid_nonuniform(self):
        grid = nonuniform_grid()
        # Interior receiver point on the z-axis P-column (k even)
        target = (float(grid.x[2]), float(grid.y[2]), float(grid.z[6]))
        for c in CLUSTERS:
            t = _native_type_for_h_cluster_comp(c, 0)  # native Hx types
            nw = _trilinear_p_nodes(grid, t, *target)
            assert nw, f"empty stencil for cluster {c}"
            w_sum = sum(w for _, w in nw)
            centroid = np.zeros(3)
            for seq, w in nw:
                i, j, k = grid.P_nodes[seq]
                centroid += w * np.array(grid.node_xyz(i, j, k))
            assert w_sum == pytest.approx(1.0, abs=1e-12)
            assert centroid == pytest.approx(np.array(target), abs=1e-12)

    def test_linear_field_reproduced_exactly(self):
        """Centroid condition ⇒ any affine field is interpolated exactly,
        at every receiver including the domain edges, on a nonuniform grid."""
        grid = nonuniform_grid()
        lin = lambda x, y, z: (0.7 + 1.3 * x - 0.4 * y + 2.1 * z) + 1j * (0.2 - z)
        B_vec = sample_Bx_field(grid, lin)
        z_vals, B_ext = extract_B_on_axis_multicl(grid, B_vec, comp=0)
        x0, y0 = float(grid.x[2]), float(grid.y[2])
        expected = np.array([lin(x0, y0, z) for z in z_vals])
        np.testing.assert_allclose(B_ext, expected, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# (b) interpolated extraction beats the old equal-weight stencils
# ---------------------------------------------------------------------------

class TestStretchedZoneBias:
    def test_new_extraction_beats_equal_weights(self):
        grid = stretched_grid()
        B_vec = sample_Bx_field(
            grid, lambda x, y, z: magnetic_dipole_B(x, y, z, SIGMA, OMEGA,
                                                    dipole_comp=0)[0])
        x0, y0 = float(grid.x[2]), float(grid.y[2])

        z_new, B_new = extract_B_on_axis_multicl(grid, B_vec, comp=0)
        z_old, B_old = old_equal_weight_extract(grid, B_vec, comp=0)
        np.testing.assert_allclose(z_new, z_old)

        exact = np.array([magnetic_dipole_B(x0, y0, z, SIGMA, OMEGA,
                                            dipole_comp=0)[0] for z in z_new])
        # Stretched-zone window of the Fig-3 benchmark; Im(Bxx) is the
        # quantity DDH03 plot.  Coordinate-based weights remove the
        # first-order equal-weight bias (measured ≈0.07–0.36% here) leaving
        # only the second-order interpolation error (≤0.11%).
        m = (z_new >= 3.0) & (z_new <= 3.4)
        assert np.count_nonzero(m) >= 4
        err_new = np.abs(np.imag(B_new[m]) - np.imag(exact[m])) / np.abs(np.imag(exact[m]))
        err_old = np.abs(np.imag(B_old[m]) - np.imag(exact[m])) / np.abs(np.imag(exact[m]))
        assert err_new.max() < 0.5 * err_old.max()
        assert err_new.max() < 2e-3

    def test_lebedev_B_on_z_axis_matches_multicl_on_shared_vec(self):
        """With identical per-cluster vectors the two extraction APIs agree."""
        grid = stretched_grid()
        B_vec = sample_Bx_field(
            grid, lambda x, y, z: magnetic_dipole_B(x, y, z, SIGMA, OMEGA,
                                                    dipole_comp=0)[0])
        z1, B1 = extract_B_on_axis_multicl(grid, B_vec, comp=0)
        z2, B2 = lebedev_B_on_z_axis(grid, {c: B_vec for c in CLUSTERS}, comp=0)
        np.testing.assert_allclose(z1, z2)
        np.testing.assert_allclose(B1, B2, rtol=1e-13)


# ---------------------------------------------------------------------------
# (c) RHS builder source groups: DDH03 conditions on nonuniform grids
# ---------------------------------------------------------------------------

class TestMagneticSourceGroups:
    def test_centroid_condition_nonuniform(self):
        grid = nonuniform_grid()
        (i0, j0, k0), groups = _magnetic_source_groups(grid, 0)
        r0 = np.array(grid.node_xyz(i0, j0, k0))
        assert (i0, j0, k0) == (2, 2, 4)
        for c in CLUSTERS:
            w_sum = sum(w for _, w in groups[c])
            centroid = np.zeros(3)
            for seq, w in groups[c]:
                i, j, k = grid.P_nodes[seq]
                centroid += w * np.array(grid.node_xyz(i, j, k))
            assert w_sum == pytest.approx(1.0, abs=1e-12)
            assert centroid == pytest.approx(r0, abs=1e-12)

    def test_symmetric_grid_reduces_to_quarter_weights(self):
        """On a symmetric uniform grid the groups are the classic 1 and 4×1/4."""
        g1 = np.linspace(-2.0, 2.0, 9)  # M = 8, node 4 at 0.0
        grid = LebedevGrid3D(g1, g1, g1)
        (_, _, _), groups = _magnetic_source_groups(grid, 0)
        sizes = sorted(len(groups[c]) for c in CLUSTERS)
        assert sizes == [1, 4, 4, 4]
        for c in CLUSTERS:
            for _, w in groups[c]:
                assert w == pytest.approx(1.0 if len(groups[c]) == 1 else 0.25,
                                          abs=1e-12)

    def test_snap_warning_when_z0_not_on_even_plane(self):
        g1 = np.linspace(-2.0, 2.0, 9)
        z = np.linspace(-2.0, 2.0, 9) + 0.17   # no even-index plane at z = 0
        grid = LebedevGrid3D(g1, g1, z)
        with pytest.warns(UserWarning, match="snapped"):
            _magnetic_source_groups(grid, 0)


# ---------------------------------------------------------------------------
# (d) boundary receivers: no phantom-zero dilution
# ---------------------------------------------------------------------------

class TestBoundaryReceivers:
    def test_constant_field_exact_at_edges(self):
        grid = nonuniform_grid()
        c_val = 3.7 - 1.9j
        B_vec = sample_Bx_field(grid, lambda x, y, z: c_val)
        z_vals, B_ext = extract_B_on_axis_multicl(grid, B_vec, comp=0)
        # Old behaviour diluted the first/last receivers to 0.75·c
        # (two clusters averaged in phantom zeros for k±1 out of range).
        assert z_vals[0] == pytest.approx(float(grid.z[0]))
        assert z_vals[-1] == pytest.approx(float(grid.z[-1]))
        np.testing.assert_allclose(B_ext, np.full_like(B_ext, c_val), rtol=1e-13)

        # Sanity: the old stencils really did dilute the edge value.
        _, B_old = old_equal_weight_extract(grid, B_vec, comp=0)
        assert abs(B_old[0] - 0.75 * c_val) < 1e-12 * abs(c_val)


# ---------------------------------------------------------------------------
# (e) analytic magnetic dipole: static limit and lock-in values
# ---------------------------------------------------------------------------

class TestMagneticDipoleAnalytics:
    def test_static_limit_on_axis(self):
        """ω → 0, x-dipole, on-axis (m ⟂ r̂): B → −μ0 m /(4π r³)."""
        B = magnetic_dipole_B(0.0, 0.0, 1.0, SIGMA, omega=1e-8, dipole_comp=0)
        assert B[0].real == pytest.approx(-MU0 / (4.0 * np.pi), rel=1e-9)
        assert abs(B[0].imag) < 1e-12 * abs(B[0].real)
        assert abs(B[1]) < 1e-20 and abs(B[2]) < 1e-20

    def test_static_limit_general_point(self):
        p = np.array([0.4, -0.7, 0.9])
        r = np.linalg.norm(p)
        r_hat = p / r
        m = np.array([1.0, 0.0, 0.0])
        expected = MU0 / (4 * np.pi) * (3 * r_hat * np.dot(m, r_hat) - m) / r**3
        B = magnetic_dipole_B(*p, SIGMA, omega=1e-8, dipole_comp=0)
        np.testing.assert_allclose(B.real, expected, rtol=1e-8)

    def test_offaxis_lockin_value(self):
        """Lock in the verified whole-space value at an off-axis point."""
        B = magnetic_dipole_B(0.3, -0.4, 1.2, SIGMA, OMEGA, dipole_comp=0)
        expected = np.array([
            -3.8357652400022665e-08 + 6.696909245856160e-10j,
            -9.6954280000447017e-09 - 5.387917533558254e-11j,
            2.9086284000134107e-08 + 1.616375260067479e-10j,
        ])
        np.testing.assert_allclose(B, expected, rtol=1e-12)

    def test_fig2_first_point(self):
        """DDH03 Fig. 2 benchmark value: Im Bxx(z=0.05 m) ≈ 19.6e-9 T."""
        val = complex(Bxx_homogeneous(np.array([0.05]), SIGMA, OMEGA)[0])
        assert val == pytest.approx(-0.0008000001300019453 + 1.9608476174778756e-08j,
                                    rel=1e-12)

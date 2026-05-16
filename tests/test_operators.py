"""
test_operators.py — Tests for the sparse FD curl operators.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lebedev_em.grid import uniform_grid
from lebedev_em.operators import (
    build_curl_RE,
    build_curl_PR,
    _build_d_dx,
    _build_d_dy,
    _build_d_dz,
)


class TestDerivativeMatrices:
    def setup_method(self):
        self.grid = uniform_grid(6, 6, 6, 2.0, 2.0, 2.0)

    def test_dx_shape_RP(self):
        Dx = _build_d_dx(self.grid, from_R=True)
        assert Dx.shape == (self.grid.N_P, self.grid.N_R)

    def test_dx_shape_PR(self):
        Dx = _build_d_dx(self.grid, from_R=False)
        assert Dx.shape == (self.grid.N_R, self.grid.N_P)

    def test_constant_field_derivative_zero(self):
        """Derivative of a uniform field should be zero at interior nodes."""
        grid = self.grid
        Dx = _build_d_dx(grid, from_R=True)
        # Set all R-node values to a constant
        f_R = np.ones(grid.N_R)
        result = Dx @ f_R
        # Only interior P-nodes should have valid derivatives; boundary rows are zero
        # All entries should be zero for a constant field
        np.testing.assert_allclose(result, 0.0, atol=1e-12,
            err_msg="d/dx of constant field should be zero.")

    def test_linear_field_derivative_correct(self):
        """d/dx of f(x,y,z) = x should equal 1 at interior nodes."""
        grid = uniform_grid(8, 4, 4, 4.0, 2.0, 2.0)
        Dx = _build_d_dx(grid, from_R=True)
        # Field = x at each R-node
        f_R = np.array([grid.x[i] for i, j, k in grid.R_nodes])
        result = Dx @ f_R
        # At interior P-nodes the result should be ≈ 1.0 (uniform grid, exact)
        for seq, (i, j, k) in enumerate(grid.P_nodes):
            # The build_curl_RE operator skips ALL boundary P-nodes (not just
            # x-faces), so we must skip any P-node on any boundary face.
            if (i == 0 or i == grid.Mx or
                    j == 0 or j == grid.My or
                    k == 0 or k == grid.Mz):
                continue  # boundary — row is zero in matrix
            np.testing.assert_allclose(
                result[seq], 1.0, rtol=1e-10,
                err_msg=f"d/dx(x) ≠ 1 at P-node ({i},{j},{k}), seq={seq}."
            )


class TestCurlOperators:
    def setup_method(self):
        self.grid = uniform_grid(6, 6, 6, 3.0, 3.0, 3.0)

    def test_CRE_shape(self):
        C = build_curl_RE(self.grid)
        assert C.shape == (3 * self.grid.N_P, 3 * self.grid.N_R)

    def test_CPR_shape(self):
        C = build_curl_PR(self.grid)
        assert C.shape == (3 * self.grid.N_R, 3 * self.grid.N_P)

    def test_curl_of_gradient_is_zero(self):
        """curl(∇φ) = 0 for any scalar field φ.

        We construct a potential field E = ∇φ (i.e., only Ex = ∂φ/∂x, etc.)
        on R-nodes and verify C_RE @ E ≈ 0 at interior P-nodes.

        NOTE: This property holds exactly for smooth fields; boundary effects
        mean we only check interior rows.
        """
        grid = uniform_grid(8, 8, 8, 4.0, 4.0, 4.0)
        C_RE = build_curl_RE(grid)

        # φ(x,y,z) = x*y + 2*y*z + 3*x*z  (a smooth scalar potential)
        # E = ∇φ:  Ex = y+3z, Ey = x+2z, Ez = 2y+3x
        Ex = np.array([grid.y[j] + 3*grid.z[k] for i, j, k in grid.R_nodes])
        Ey = np.array([grid.x[i] + 2*grid.z[k] for i, j, k in grid.R_nodes])
        Ez = np.array([2*grid.y[j] + 3*grid.x[i] for i, j, k in grid.R_nodes])

        E_vec = np.concatenate([Ex, Ey, Ez])
        curl_E = C_RE @ E_vec  # shape (3*N_P,)

        # Check interior P-nodes (not on boundary)
        for seq, (i, j, k) in enumerate(grid.P_nodes):
            is_bdy = (i == 0 or i == grid.Mx or
                      j == 0 or j == grid.My or
                      k == 0 or k == grid.Mz)
            if is_bdy:
                continue
            for comp in range(3):
                val = curl_E[comp * grid.N_P + seq]
                np.testing.assert_allclose(
                    val, 0.0, atol=1e-8,
                    err_msg=f"curl(∇φ)_{comp} ≠ 0 at P-node ({i},{j},{k})"
                )

    def test_curl_curl_is_real_nonneg_diagonal(self):
        """The diagonal of C_PR @ C_RE should be real and non-negative (it represents
        the sum of squared gradient stencil coefficients)."""
        grid = uniform_grid(6, 6, 6, 2.0, 2.0, 2.0)
        C_RE = build_curl_RE(grid)
        C_PR = build_curl_PR(grid)
        A = C_PR @ C_RE
        diag = np.array(A.diagonal())
        # Diagonal should be real (imaginary part = 0)
        np.testing.assert_allclose(np.imag(diag), 0.0, atol=1e-12,
            err_msg="Diagonal of curl-curl has unexpected imaginary part.")
        # Diagonal should be >= 0
        assert np.all(np.real(diag) >= -1e-12), \
            "Diagonal of curl-curl has negative entries."

"""
media.py — Electromagnetic medium parameters on the Lebedev grid.

Stores conductivity σ, permeability μ, and permittivity ε at every grid node
as either a scalar field (isotropic) or a full 3×3 tensor field (anisotropic).
Provides convenience builders and the material matrices needed by `operators.py`.

Interface averaging
-------------------
`layered_isotropic` accepts an `interface_averaging` parameter (default False).

With ``interface_averaging=False`` (recommended) material parameters are
assigned **pointwise**: each R-node takes the conductivity of the layer it
falls in, using a ``side='right'`` rule so nodes exactly on a boundary are
assigned to the layer above.  Empirically this gives the best accuracy for
the Lebedev scheme because the 4-cluster averaging already cancels leading-
order interface artefacts; adding explicit interface averaging fights that
mechanism and can produce catastrophic errors (sign reversal, 10–1000× over-
estimation in the opposite layer).

With ``interface_averaging=True`` any dual cell that straddles a layer
boundary is replaced by an anisotropic effective medium using the *standard*
(arithmetic/harmonic) formula:

    σ‖  = f₁ σ₁ + f₂ σ₂        (arithmetic mean — tangential components)
    σ⊥  = (f₁/σ₁ + f₂/σ₂)⁻¹   (harmonic mean  — normal component)

With ``nodal_averaging=True`` the *nodal* homogenization (Moskow et al. 1999,
extended to 3-D) is used instead.  The derivation matches the discrete and
continuous energy inner products over each dual cell.

For a planar interface with unit normal n̂ and two orthonormal tangentials
m̂, q̂ the local solution space is L(H) = {1, m̂·x, q̂·x, ∫₀^{n̂·x} σ⁻¹ dt}.
The discrete (FD) gradient of any φ ∈ L(H) with coefficients (c₁,c₂,c₃) is

    ∇̃φ = c₁ m̂ + c₂ q̂ + c₃ D n̂,   D = diag(σ⁻¹_x, σ⁻¹_y, σ⁻¹_z)

where σ⁻¹_k is the per-axis **line average** of σ⁻¹ along grid axis eₖ.
Energy matching ∫_H ∇φ·σ∇ψ dV = |H| ∇̃φ·ΣD ∇̃ψ gives

    ΣD = L̃⁻ᵀ G L̃⁻¹,   L̃ = [m̂ | q̂ | Dn̂],   G = diag(σ̄, σ̄, ⟨σ⁻¹⟩_vol)

Note G[2,2] = ⟨σ⁻¹⟩_vol (arithmetic mean of resistivity), NOT σ̃.
For axis-aligned n̂ = ê_α this reduces correctly to ΣD = diag(σ̄, σ̄, σ̃).

Angled interfaces
-----------------
`planar_interface_isotropic` handles a single planar interface at arbitrary
angle using the full nodal formula.  The dual-cell line averages of σ⁻¹ along
all three grid-axis edges are computed analytically, the volume fraction is
approximated with Gauss-Legendre quadrature, and the full ΣD is assembled.
For non-axis-aligned normals ΣD has off-diagonal entries that couple field
components across the interface.

References
----------
Davydycheva, Druskin & Habashy (2003), Geophysics 68(5):1525–1536.
Moskow, Druskin, Habashy, Lee & Davydycheva (1999), SIAM J. Numer. Anal.
36(2):442–466.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator

from .grid import LebedevGrid3D
from .operators import scalar_diag, tensor_block_diag

MU0  = 4e-7 * np.pi          # H/m — free-space permeability
EPS0 = 8.854187817e-12        # F/m — free-space permittivity


class EMMedia:
    """
    Electromagnetic medium parameters on a LebedevGrid3D.

    All parameters may be **isotropic** (scalar per node, stored as 1-D array
    of length N) or **anisotropic** (full 3×3 tensor per node, stored as
    ndarray of shape (N, 3, 3)).

    Parameters are stored separately for R-nodes (σ, ε) and P-nodes (μ),
    matching the subgrid structure of the DDH03 scheme.

    Parameters
    ----------
    grid : LebedevGrid3D
    sigma_R : ndarray
        Conductivity at R-nodes. Shape (N_R,) for isotropic or (N_R, 3, 3)
        for anisotropic.
    mu_P : ndarray
        Permeability at P-nodes. Shape (N_P,) or (N_P, 3, 3).
    eps_R : ndarray, optional
        Permittivity at R-nodes. If None defaults to ε₀·I everywhere.
    """

    def __init__(
        self,
        grid: LebedevGrid3D,
        sigma_R: np.ndarray,
        mu_P: np.ndarray,
        eps_R: np.ndarray | None = None,
    ) -> None:
        self.grid   = grid
        self.sigma_R = np.asarray(sigma_R, dtype=complex)
        self.mu_P    = np.asarray(mu_P,    dtype=complex)

        if eps_R is None:
            eps_R = np.full(grid.N_R, EPS0)
        self.eps_R = np.asarray(eps_R, dtype=complex)

        self._validate()

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        ok_shapes_R = (self.grid.N_R,), (self.grid.N_R, 3, 3)
        ok_shapes_P = (self.grid.N_P,), (self.grid.N_P, 3, 3)
        if self.sigma_R.shape not in ok_shapes_R:
            raise ValueError(f"sigma_R shape {self.sigma_R.shape} invalid.")
        if self.mu_P.shape not in ok_shapes_P:
            raise ValueError(f"mu_P shape {self.mu_P.shape} invalid.")
        if self.eps_R.shape not in ok_shapes_R:
            raise ValueError(f"eps_R shape {self.eps_R.shape} invalid.")

    @property
    def is_isotropic(self) -> bool:
        return self.sigma_R.ndim == 1 and self.mu_P.ndim == 1

    # ------------------------------------------------------------------
    # Material matrices for the system assembler
    # ------------------------------------------------------------------

    def sigma_dot_matrix(self, omega: float) -> sp.spmatrix:
        """
        Return σ̇ = σ − iωε as a (3·N_R × 3·N_R) sparse matrix.

        Sign convention: with the exp(−iωt) time dependence used throughout
        (see analytics.py), the complex conductivity is σ̇ = σ − iωε.
        (DDH03's printed "+iωε" in eq. 1 is a typo for this convention.)

        For isotropic media: block-diagonal with scalar entries.
        For anisotropic:     3×3 block-diagonal.

        Handles every combination of scalar/tensor σ and ε.  Note: checks
        sigma_R.ndim independently of mu_P so that a scalar σ paired with a
        tensor μ (or vice-versa) is handled correctly — the combined
        is_isotropic flag is not used here.
        """
        # Promote eps to whichever representation sigma uses.
        if self.sigma_R.ndim == 1 and self.eps_R.ndim == 1:
            sigma_dot = self.sigma_R - 1j * omega * self.eps_R
            return scalar_diag(sigma_dot)

        # At least one of σ, ε is a tensor — assemble 3×3 blocks.
        def _as_tensor(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 3:
                return arr
            out = np.zeros((self.grid.N_R, 3, 3), dtype=complex)
            for d in range(3):
                out[:, d, d] = arr
            return out

        sdot_tensors = _as_tensor(self.sigma_R) - 1j * omega * _as_tensor(self.eps_R)
        return tensor_block_diag(sdot_tensors)

    def inv_mu_matrix(self) -> sp.spmatrix:
        """
        Return μ⁻¹ as a (3·N_P × 3·N_P) sparse matrix.

        Checks mu_P.ndim independently of sigma_R.
        """
        if self.mu_P.ndim == 1:
            return scalar_diag(1.0 / self.mu_P)
        else:
            inv_mu_tensors = np.linalg.inv(self.mu_P)
            return tensor_block_diag(inv_mu_tensors)


# ---------------------------------------------------------------------------
# Low-level interface-averaging helpers
# ---------------------------------------------------------------------------

def _effective_tensor(
    c_minus: float,
    c_plus: float,
    bounds: np.ndarray,
    values: np.ndarray,
    normal_axis: int,
) -> np.ndarray | None:
    """
    Standard (arithmetic/harmonic) effective 3×3 tensor for a dual cell
    [c_minus, c_plus] that may straddle one or more axis-aligned layer
    boundaries.

    Returns None if the cell is entirely within one layer.

    Method
    ------
    Split the cell into segments at each boundary strictly inside
    [c_minus, c_plus].  For each segment of length Δ and value v_i:

        f_i = Δ / (c_plus − c_minus)        (volume fraction)

    Then:
        tangential axes (α ≠ normal_axis):   arithmetic mean  Σ f_i v_i
        normal axis     (α = normal_axis):   harmonic mean    1 / Σ (f_i / v_i)

    Reference: DDH03 eq. 8; Backus (1962) for lamellar composites.
    """
    cell_len = c_plus - c_minus
    if cell_len <= 0:
        return None

    inner = bounds[(bounds > c_minus) & (bounds < c_plus)]
    if len(inner) == 0:
        return None   # entirely in one layer

    edges = np.concatenate([[c_minus], inner, [c_plus]])
    segments = []
    for a, b in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + b)
        lid = int(np.searchsorted(bounds, mid, side="right"))
        segments.append((b - a, values[lid]))

    total = sum(dl for dl, _ in segments)
    arith = sum(dl * v for dl, v in segments) / total
    harm  = total / sum(dl / v for dl, v in segments)

    if abs(arith - harm) < 1e-10 * abs(arith):
        return None   # effectively uniform

    tensor = np.eye(3, dtype=complex) * arith
    tensor[normal_axis, normal_axis] = harm
    return tensor


def _nodal_effective_tensor_layered(
    c_minus: float,
    c_plus: float,
    bounds: np.ndarray,
    values: np.ndarray,
    normal_axis: int,
) -> np.ndarray | None:
    """
    Nodal homogenization (Moskow et al. 1999) for an axis-aligned layered
    medium.  Returns the 3×3 tensor ΣD, or None if the cell is in one layer.

    For a unit interface normal n̂ = ê_{normal_axis} the formula reduces to:

        ΣD = diag(σ̄, σ̄, σ̃)   (arithmetic tangential, harmonic normal)

    where σ̄ = Σ f_i σ_i  (arithmetic),  σ̃ = (Σ f_i/σ_i)⁻¹  (harmonic).

    This satisfies Lemma 3.2 (discrete energy equality) exactly.
    """
    inner = bounds[(bounds > c_minus) & (bounds < c_plus)]
    if len(inner) == 0:
        return None

    edges = np.concatenate([[c_minus], inner, [c_plus]])
    segments = []
    for a, b in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + b)
        lid = int(np.searchsorted(bounds, mid, side="right"))
        segments.append((b - a, values[lid]))

    total = sum(dl for dl, _ in segments)
    sigma_arith = sum(dl * v for dl, v in segments) / total
    inv_sigma_n = sum(dl / v for dl, v in segments) / total   # = 1/σ̃
    sigma_harm  = 1.0 / inv_sigma_n

    if abs(sigma_arith - sigma_harm) < 1e-10 * abs(sigma_arith):
        return None

    # ΣD = diag(σ̄, σ̄, σ̃) with σ̃ at the normal axis
    tensor = np.eye(3, dtype=complex) * sigma_arith
    tensor[normal_axis, normal_axis] = sigma_harm
    return tensor


# ---------------------------------------------------------------------------
# Nodal homogenization for general planar interfaces
# ---------------------------------------------------------------------------

def _frac_1d_layer1(
    c_minus: float,
    c_plus: float,
    n_component: float,
    d_rest: float,
) -> float:
    """
    Fraction of the 1-D segment [c_minus, c_plus] in layer 1.

    Layer 1 is defined as  n_component * t  <  d_rest  along this edge.
    (For an edge in direction α at fixed other coordinates,
    d_rest = d_plane − Σ_{β≠α} n_β * x_β_fixed.)

    Returns a value in [0, 1].
    """
    Dc = c_plus - c_minus
    if Dc <= 0.0:
        return 0.5

    tol = 1e-14 * (abs(d_rest) + abs(n_component) * (abs(c_minus) + abs(c_plus)) + 1.0)
    if abs(n_component) < tol:
        # Edge is parallel to the interface plane; entire edge in one layer.
        return 1.0 if d_rest > 0.0 else 0.0

    t_cut = d_rest / n_component   # interface crosses edge at t = t_cut
    if n_component > 0:
        # layer 1: t < t_cut
        return float(np.clip((t_cut - c_minus) / Dc, 0.0, 1.0))
    else:
        # layer 1: t > t_cut  (n_component * t < d_rest for negative n_component)
        return float(np.clip((c_plus - t_cut) / Dc, 0.0, 1.0))


def _inv_sigma_line_avg(
    c_minus: float,
    c_plus: float,
    n_component: float,
    d_rest: float,
    sigma1: complex,
    sigma2: complex,
) -> complex:
    """
    Line average of σ⁻¹ along segment [c_minus, c_plus] for a 2-layer
    planar interface.

    Returns  f₁/σ₁ + f₂/σ₂  where f₁, f₂ are the fractional lengths in
    each layer.
    """
    f1 = _frac_1d_layer1(c_minus, c_plus, n_component, d_rest)
    return complex(f1 / sigma1 + (1.0 - f1) / sigma2)


def _cdf_sum_uniforms(T_prime: float, bw: list) -> float:
    """
    Exact CDF of  X_1 + X_2 + … + X_n  evaluated at *T_prime*,
    where each  X_i ~ Uniform(0, bw[i])  is independent.

    Uses the inclusion-exclusion (Irwin-Hall) formula:

        F(T') = (1 / (V · n!)) · Σ_{k ∈ {0,1}^n}  (−1)^|k| · max(0, T' − Σ bw_i k_i)^n

    where V = Π bw_i.  Dimensions with bw_i = 0 are treated as
    deterministic zeros and removed before applying the formula.

    Returns a value in [0, 1].
    """
    from math import factorial as _fac

    bw = [b for b in bw if b > 1e-30]   # drop degenerate dimensions
    n = len(bw)
    if n == 0:
        return 1.0 if T_prime >= 0.0 else 0.0

    s_max = sum(bw)
    if T_prime >= s_max:
        return 1.0
    if T_prime <= 0.0:
        return 0.0

    V = 1.0
    for b in bw:
        V *= b

    total = 0.0
    for k in range(1 << n):
        sign   = 1.0
        offset = 0.0
        for i in range(n):
            if (k >> i) & 1:
                sign   *= -1.0
                offset += bw[i]
        val = T_prime - offset
        if val > 0.0:
            total += sign * (val ** n)

    return total / (V * _fac(n))


def _volume_frac_layer1_planar(
    box_min: np.ndarray,
    box_max: np.ndarray,
    n_hat: np.ndarray,
    d_plane: float,
    n_gl: int = 10,   # kept for API compatibility; not used (exact formula)
) -> float:
    """
    Exact volume fraction of the rectangular box [box_min, box_max] in
    layer 1  (n̂ · x < d_plane)  for a planar interface.

    Method
    ------
    The fraction equals  P(S ≤ T)  where
        S = n_x (X − x_min) + n_y (Y − y_min) + n_z (Z − z_min)
    and  X, Y, Z  are independent uniforms on  [0, Δx], [0, Δy], [0, Δz].
    This is the CDF of a sum of three (possibly degenerate) scaled uniform
    RVs, computed exactly by the Irwin-Hall inclusion-exclusion formula via
    `_cdf_sum_uniforms`.  The result is always exact (no quadrature error).

    Parameters
    ----------
    box_min, box_max : ndarray (3,)
    n_hat : ndarray (3,)   unit interface normal
    d_plane : float        n̂ · x = d_plane
    n_gl : int             ignored (kept for backward-compatible signature)
    """
    # Quick check on the 8 corners.
    xs = [box_min[0], box_max[0]]
    ys = [box_min[1], box_max[1]]
    zs = [box_min[2], box_max[2]]
    corner_vals = [n_hat[0]*x + n_hat[1]*y + n_hat[2]*z
                   for x in xs for y in ys for z in zs]
    if all(v < d_plane for v in corner_vals):
        return 1.0
    if all(v >= d_plane for v in corner_vals):
        return 0.0

    # S = Σ_i A_i * U_i  where  A_i = n_i * Δx_i  and  U_i ~ U(0, 1).
    # Shift so each component is non-negative:
    #   X_i = A_i * U_i - min(A_i, 0)  ~ U(0, |A_i|)
    #   S'  = Σ X_i = S - Σ min(A_i, 0)
    # P(S ≤ T) = P(S' ≤ T')  where  T' = T - Σ min(A_i, 0).
    A      = [n_hat[i] * (box_max[i] - box_min[i]) for i in range(3)]
    T      = d_plane - float(np.dot(n_hat, box_min))
    T_pr   = T - sum(min(a, 0.0) for a in A)
    bw     = [abs(a) for a in A]

    return _cdf_sum_uniforms(T_pr, bw)


def _nodal_eff_tensor_3d(
    D_diag: np.ndarray,
    sigma_arith: complex,
    inv_sigma_vol: complex,
    n_hat: np.ndarray,
) -> np.ndarray:
    """
    General 3-D nodal homogenization tensor ΣD (Moskow et al. 1999, extended
    to 3-D by energy matching).

    Derivation
    ----------
    The local solution space on cell H is spanned by
        L(H) = {1,  m̂·x,  q̂·x,  φ_n(x)}
    where φ_n = ∫₀^{n̂·x} σ(t)⁻¹ dt.  For φ ∈ L(H) with coefficients
    (c₁, c₂, c₃):
        ∇φ = c₁ m̂ + c₂ q̂ + (c₃/σ) n̂          (true gradient)
        ∇̃φ = c₁ m̂ + c₂ q̂ + c₃ D n̂            (FD/discrete gradient)

    where D = diag(D_diag), dₖ = line average of σ⁻¹ along grid axis eₖ.

    The true energy integral:
        ∫_H ∇φ · σ ∇ψ dV = |H| aᵀ G b
    where G = diag(σ̄, σ̄, ⟨σ⁻¹⟩_vol) in the (c₁,c₂,c₃) basis.

    Setting the discrete energy equal to the true energy for all φ,ψ ∈ L(H):
        L̃ᵀ ΣD L̃ = G   where L̃ = [m̂ | q̂ | Dn̂]
        ΣD = L̃⁻ᵀ G L̃⁻¹

    Verification (axis-aligned n̂=ẑ):
        L̃ = diag(1,1,d_z),  G = diag(σ̄, σ̄, 1/σ̃)
        ΣD = diag(σ̄, σ̄, 1/(σ̃ d_z²)) = diag(σ̄, σ̄, σ̃)  ✓

    Parameters
    ----------
    D_diag : ndarray (3,)  complex
        Diagonal of D = diag(σ⁻¹_x, σ⁻¹_y, σ⁻¹_z); each entry is the
        **line average** of σ⁻¹ along the corresponding grid-axis edge.
    sigma_arith : complex
        Arithmetic (volume) mean σ̄ = ⟨σ⟩_vol.
    inv_sigma_vol : complex
        Volume mean of σ⁻¹:  ⟨σ⁻¹⟩_vol = 1/σ̃.
        This is G[2,2] and equals ``f_vol/σ₁ + (1−f_vol)/σ₂``.
    n_hat : ndarray (3,)  float
        Unit interface normal (real).

    Returns
    -------
    ΣD : ndarray (3, 3)  complex
    """
    # Compute D n̂ (element-wise, since D is diagonal)
    Dn = D_diag.astype(complex) * n_hat.astype(complex)

    # Build two tangential unit vectors orthonormal to n_hat (Gram-Schmidt).
    idx = int(np.argmin(np.abs(n_hat)))
    v = np.zeros(3); v[idx] = 1.0
    m = v - np.dot(v, n_hat) * n_hat
    m_norm = np.linalg.norm(m)
    if m_norm < 1e-14:
        v[(idx + 1) % 3] = 0.1
        v /= np.linalg.norm(v)
        m = v - np.dot(v, n_hat) * n_hat
        m_norm = np.linalg.norm(m)
    m /= m_norm
    q = np.cross(n_hat, m)
    q /= np.linalg.norm(q)

    # L̃ = [m̂ | q̂ | Dn̂]  (3×3, columns)
    L_tilde = np.column_stack([m.astype(complex), q.astype(complex), Dn])

    # G = diag(σ̄, σ̄, ⟨σ⁻¹⟩_vol) in the (c₁,c₂,c₃) space.
    # NOTE: G[2,2] = ⟨σ⁻¹⟩_vol = inv_sigma_vol (NOT σ̃).
    G_diag = np.array([sigma_arith, sigma_arith, inv_sigma_vol], dtype=complex)

    try:
        L_inv = np.linalg.inv(L_tilde)
    except np.linalg.LinAlgError:
        # Degenerate cell — fall back to standard arithmetic/harmonic tensor.
        n_c = n_hat.astype(complex)
        sigma_harm = complex(1.0 / inv_sigma_vol) if abs(inv_sigma_vol) > 1e-30 else sigma_arith
        return (sigma_arith * (np.eye(3, dtype=complex) - np.outer(n_c, n_c))
                + sigma_harm * np.outer(n_c, n_c))

    # ΣD = L̃⁻ᵀ G L̃⁻¹  (simple transpose, not conjugate — bilinear form)
    return L_inv.T @ np.diag(G_diag) @ L_inv


def _nodal_eff_tensor_general(
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    f_vol: float,
    f_line: np.ndarray,
    n_hat: np.ndarray,
) -> np.ndarray:
    """
    Generalised 3-D nodal homogenization ΣD for a planar interface between
    two *anisotropic* media σ₁, σ₂ (Moskow et al. 1999 appendix, extended
    to 3-D by energy matching).

    Derivation
    ----------
    For anisotropic σ the corrected basis functions that block-diagonalise G are

        φ₁ = m̂·x − ∫₀^{n̂·x} (m̂ᵀσn̂)/(n̂ᵀσn̂) dt
        φ₂ = q̂·x − ∫₀^{n̂·x} (q̂ᵀσn̂)/(n̂ᵀσn̂) dt
        φ₃ = ∫₀^{n̂·x} 1/(n̂ᵀσn̂) dt

    In this basis  G = block_diag(G_TT_block, G_nn)  where

        G_TT_mat = f_vol·S₁ + (1−f_vol)·S₂   (3×3, volume-weighted Schur sum)
        Sᵢ = σᵢ − (σᵢn̂)(n̂ᵀσᵢ)/(n̂ᵀσᵢn̂)    (3×3 Schur complement)
        G_nn  = f_vol/σ_nn1 + (1−f_vol)/σ_nn2  (volume mean of σ_nn⁻¹)

    The 3×3 coefficient-space G matrix is

        G_3x3 = [[m̂ᵀ G_TT_mat m̂,  m̂ᵀ G_TT_mat q̂,  0  ],
                 [q̂ᵀ G_TT_mat m̂,  q̂ᵀ G_TT_mat q̂,  0  ],
                 [0,               0,               G_nn]]

    The discrete gradient of  a·φ₁ + b·φ₂ + c·φ₃  is

        ∇̃ψ = a·(m̂ − ρ_m⊙n̂) + b·(q̂ − ρ_q⊙n̂) + c·(D₁⊙n̂)

    where (per grid axis α):

        D₁[α]    = f_line[α]/σ_nn1  + (1−f_line[α])/σ_nn2
        ρ_m[α]   = f_line[α]·(m̂ᵀσ₁n̂)/σ_nn1 + (1−f_line[α])·(m̂ᵀσ₂n̂)/σ_nn2
        ρ_q[α]   = f_line[α]·(q̂ᵀσ₁n̂)/σ_nn1 + (1−f_line[α])·(q̂ᵀσ₂n̂)/σ_nn2

    The modified L̃ matrix (columns = discrete gradient basis vectors):

        L̃ = [m̂ − ρ_m⊙n̂ | q̂ − ρ_q⊙n̂ | D₁⊙n̂]

    Energy matching  L̃ᵀ ΣD L̃ = G_3x3  gives

        ΣD = L̃⁻ᵀ G_3x3 L̃⁻¹

    Reduction to isotropic case
    ---------------------------
    When σᵢ = sᵢ·I (scalar), σᵢn̂ = sᵢn̂, so m̂ᵀσᵢn̂ = q̂ᵀσᵢn̂ = 0,
    giving ρ_m = ρ_q = 0 and D₁ = existing D_diag.  Also
    G_TT_mat = σ̄·(I − n̂n̂ᵀ), so G_mm = σ̄, G_qq = σ̄, G_mq = 0, and
    G_3x3 = diag(σ̄, σ̄, G_nn).  L̃ reduces to [m̂|q̂|D n̂], recovering
    exactly _nodal_eff_tensor_3d. ✓

    Parameters
    ----------
    sigma1, sigma2 : ndarray (3,3) complex  or  scalar
        Conductivity tensors of the two media.  Scalars are broadcast to s·I.
    f_vol : float
        Volume fraction of region 1 in the dual cell ([0, 1]).
    f_line : ndarray (3,)  float
        Per-axis (x, y, z) line fractions of region 1 along the three grid
        edges through the node.  Each entry in [0, 1].
    n_hat : ndarray (3,)  float
        Unit interface normal pointing from region 1 into region 2.

    Returns
    -------
    ΣD : ndarray (3,3) complex
    """
    # --- Coerce inputs ---
    def _to_tensor(s):
        s = np.asarray(s, dtype=complex)
        if s.ndim == 0 or s.shape == ():
            return complex(s) * np.eye(3, dtype=complex)
        if s.shape == (1,):
            return complex(s[0]) * np.eye(3, dtype=complex)
        return s.reshape(3, 3)

    s1 = _to_tensor(sigma1)
    s2 = _to_tensor(sigma2)
    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    f_line = np.asarray(f_line, dtype=float)
    f_vol  = float(f_vol)

    n_c = n_hat.astype(complex)

    # --- Normal conductivities σ_nn_i = n̂ᵀ σᵢ n̂ ---
    sigma_nn1 = complex(n_c @ s1 @ n_c)
    sigma_nn2 = complex(n_c @ s2 @ n_c)

    # Guard against degenerate zero normal conductivity.  With σ_nn ≈ 0 the
    # harmonic normal average (and the whole L̃ construction) is undefined, so
    # fall back to the plain arithmetic volume average — the upper Wiener
    # bound, which is the only well-defined limit in this degenerate case.
    if abs(sigma_nn1) < 1e-30 or abs(sigma_nn2) < 1e-30:
        sigma_arith = complex(f_vol) * s1 + complex(1.0 - f_vol) * s2
        return sigma_arith

    # --- Tangential frame {m̂, q̂} orthonormal to n̂ ---
    idx = int(np.argmin(np.abs(n_hat)))
    v = np.zeros(3); v[idx] = 1.0
    m = v - np.dot(v, n_hat) * n_hat
    m_norm = np.linalg.norm(m)
    if m_norm < 1e-14:
        v[(idx + 1) % 3] = 0.1
        v /= np.linalg.norm(v)
        m = v - np.dot(v, n_hat) * n_hat
        m_norm = np.linalg.norm(m)
    m = m / m_norm
    q = np.cross(n_hat, m)
    q = q / np.linalg.norm(q)
    m_c = m.astype(complex)
    q_c = q.astype(complex)

    # --- Schur complements Sᵢ = σᵢ − (σᵢn̂)(n̂ᵀσᵢ) / σ_nn_i ---
    s1_n = s1 @ n_c          # (3,)  σ₁ n̂
    s2_n = s2 @ n_c          # (3,)  σ₂ n̂
    n_s1 = n_c @ s1          # (3,)  n̂ᵀ σ₁   (= (σ₁ n̂)ᵀ for symmetric σ)
    n_s2 = n_c @ s2          # (3,)  n̂ᵀ σ₂
    S1 = s1 - np.outer(s1_n, n_s1) / sigma_nn1
    S2 = s2 - np.outer(s2_n, n_s2) / sigma_nn2

    # Volume-weighted average Schur complement (3×3)
    G_TT_mat = complex(f_vol) * S1 + complex(1.0 - f_vol) * S2

    # G_nn = volume-weighted harmonic mean of σ_nn
    G_nn = complex(f_vol / sigma_nn1 + (1.0 - f_vol) / sigma_nn2)

    # 2×2 block of G in (m̂, q̂) basis
    G_mm = complex(m_c @ G_TT_mat @ m_c)
    G_mq = complex(m_c @ G_TT_mat @ q_c)
    G_qm = G_mq   # symmetric (for symmetric σ; general: q̂ᵀ G_TT m̂)
    G_qq = complex(q_c @ G_TT_mat @ q_c)

    G_3x3 = np.array([[G_mm, G_mq, 0.0],
                       [G_qm, G_qq, 0.0],
                       [0.0,  0.0,  G_nn]], dtype=complex)

    # --- Per-axis line-average quantities ---
    # D₁[α] = f_line[α]/σ_nn1 + (1−f_line[α])/σ_nn2
    D1 = f_line / sigma_nn1 + (1.0 - f_line) / sigma_nn2    # (3,) complex

    # m̂ᵀ σᵢ n̂  (scalar coupling between m and n directions)
    sigma_mn1 = complex(m_c @ s1 @ n_c)
    sigma_mn2 = complex(m_c @ s2 @ n_c)
    # q̂ᵀ σᵢ n̂
    sigma_qn1 = complex(q_c @ s1 @ n_c)
    sigma_qn2 = complex(q_c @ s2 @ n_c)

    # ρ_m[α] = line avg of (m̂ᵀσn̂)/σ_nn along axis α
    rho_m = f_line * (sigma_mn1 / sigma_nn1) + (1.0 - f_line) * (sigma_mn2 / sigma_nn2)  # (3,)
    # ρ_q[α] = line avg of (q̂ᵀσn̂)/σ_nn along axis α
    rho_q = f_line * (sigma_qn1 / sigma_nn1) + (1.0 - f_line) * (sigma_qn2 / sigma_nn2)  # (3,)

    # --- Modified L̃ matrix  (columns: discrete gradient of φ₁, φ₂, φ₃) ---
    #   col1 = m̂ − ρ_m ⊙ n̂    (⊙ = element-wise multiply by component)
    #   col2 = q̂ − ρ_q ⊙ n̂
    #   col3 = D₁ ⊙ n̂
    col1 = m_c - rho_m * n_c   # (3,)
    col2 = q_c - rho_q * n_c   # (3,)
    col3 = D1   * n_c           # (3,)
    L_tilde = np.column_stack([col1, col2, col3])   # (3,3)

    # --- ΣD = L̃⁻ᵀ G_3x3 L̃⁻¹  (bilinear, not sesquilinear) ---
    try:
        L_inv = np.linalg.inv(L_tilde)
    except np.linalg.LinAlgError:
        # Degenerate cell — fall back to arithmetic/harmonic
        sigma_harm_n = 1.0 / G_nn if abs(G_nn) > 1e-30 else 0.0
        sigma_arith_t = 0.5 * (G_mm + G_qq)
        return (complex(sigma_arith_t) * (np.eye(3, dtype=complex) - np.outer(n_c, n_c))
                + complex(sigma_harm_n) * np.outer(n_c, n_c))

    return L_inv.T @ G_3x3 @ L_inv


def _multiregion_line_and_vol_fracs(
    bmin: np.ndarray,
    bmax: np.ndarray,
    planes: list,         # list of (n_hat, d_plane) defining region boundaries
    node: np.ndarray,
    n_gauss: int = 5,
) -> tuple:
    """
    Compute per-axis line fractions and volume fractions for a cell cut by
    multiple planar interfaces.

    The planes are applied in order: region 0 is on the "inside" of plane 0
    (n_hat_0 · x < d_plane_0), region 1 is outside plane 0 AND inside plane 1,
    region 2 is outside both planes, etc.  For N planes there are N+1 regions.

    Line fractions are evaluated at the node's transverse coordinates (matching
    the convention of _frac_1d_layer1).  Volume fractions use n_gauss-point
    Gauss-Legendre quadrature in each dimension.

    Parameters
    ----------
    bmin, bmax : ndarray (3,)
    planes     : list of (n_hat, d_plane) pairs, each n_hat a unit vector
    node       : ndarray (3,)  node coordinates
    n_gauss    : int  quadrature order per dimension

    Returns
    -------
    line_fracs : ndarray (n_regions, 3)
        line_fracs[r, k] = fraction of axis-k edge in region r
    vol_fracs  : ndarray (n_regions,)
        volume fractions, summing to 1
    """
    from numpy.polynomial.legendre import leggauss

    n_planes  = len(planes)
    n_regions = n_planes + 1

    def _region(dot_vals):
        """Given dot products n_i·x for each plane i, return region index."""
        for r, (n_h, d_p) in enumerate(planes):
            if dot_vals[r] < d_p:
                return r
        return n_planes

    # ── Line fractions (exact, 1-D segment analysis) ──────────────────────────
    line_fracs = np.zeros((n_regions, 3), dtype=float)
    for k in range(3):
        lo, hi = float(bmin[k]), float(bmax[k])
        Dk = hi - lo
        if Dk <= 1e-30:
            # Degenerate edge: assign to region at node
            dots = [float(np.dot(nh, node)) for nh, _ in planes]
            line_fracs[_region(dots), k] = 1.0
            continue

        # For each plane: cut coordinate along edge k, at node's transverse coords
        cuts = [lo, hi]
        for n_h, d_p in planes:
            n_k = float(n_h[k])
            rest = float(np.dot(n_h, node)) - n_k * float(node[k])
            if abs(n_k) > 1e-14:
                t_cut = (d_p - rest) / n_k
                if lo < t_cut < hi:
                    cuts.append(t_cut)

        cuts = sorted(set(cuts))
        for i in range(len(cuts) - 1):
            mid = 0.5 * (cuts[i] + cuts[i + 1])
            frac = (cuts[i + 1] - cuts[i]) / Dk
            # Evaluate dot products at midpoint (transverse coords = node)
            x_mid = node.copy(); x_mid[k] = mid
            dots = [float(np.dot(nh, x_mid)) for nh, _ in planes]
            line_fracs[_region(dots), k] += frac

    # ── Volume fractions (Gauss-Legendre quadrature) ──────────────────────────
    pts1d, wts1d = leggauss(n_gauss)
    # Scale points/weights from [-1,1] to [bmin[k], bmax[k]]
    xs = [0.5 * (bmax[k] - bmin[k]) * pts1d + 0.5 * (bmax[k] + bmin[k])
          for k in range(3)]
    ws = [0.5 * (bmax[k] - bmin[k]) * wts1d for k in range(3)]
    vol_box = float(np.prod(bmax - bmin)) if np.all(bmax > bmin) else 1.0

    vol_fracs = np.zeros(n_regions, dtype=float)
    for i in range(n_gauss):
        for j in range(n_gauss):
            for l in range(n_gauss):
                pt = np.array([xs[0][i], xs[1][j], xs[2][l]])
                w  = ws[0][i] * ws[1][j] * ws[2][l]
                dots = [float(np.dot(nh, pt)) for nh, _ in planes]
                vol_fracs[_region(dots)] += w

    if vol_box > 1e-30:
        vol_fracs /= vol_box
    else:
        vol_fracs /= vol_fracs.sum()

    return line_fracs, vol_fracs


def _nodal_eff_tensor_multiregion(
    sigmas:     list,
    vol_fracs:  np.ndarray,
    line_fracs: np.ndarray,
    n_hat:      np.ndarray,
) -> np.ndarray:
    """
    Generalised nodal homogenization ΣD = L̃⁻ᵀ G L̃⁻¹ for a cell containing
    an arbitrary number of material regions, all sharing the same interface
    normal n̂.

    This is the direct multi-region extension of _nodal_eff_tensor_general:
    the energy matrix G and the discrete-gradient map L̃ are built from
    weighted sums over all regions, using the volume and per-axis line
    fractions supplied by the caller.

    Parameters
    ----------
    sigmas     : list of ndarray (3,3) or scalar  — one per region
    vol_fracs  : ndarray (n_regions,)  — volume fractions, should sum to 1
    line_fracs : ndarray (n_regions, 3) — line_fracs[r,k] = fraction of
                 axis-k edge in region r; each column should sum to 1
    n_hat      : ndarray (3,)  — unit interface normal

    Returns
    -------
    ΣD : ndarray (3,3) complex
    """
    def _to_tensor(s):
        s = np.asarray(s, dtype=complex)
        if s.ndim == 0 or s.shape == (): return complex(s) * np.eye(3, dtype=complex)
        if s.shape == (1,):              return complex(s[0]) * np.eye(3, dtype=complex)
        return s.reshape(3, 3)

    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    n_c   = n_hat.astype(complex)

    # Tangential frame {m̂, q̂}
    idx = int(np.argmin(np.abs(n_hat)))
    v = np.zeros(3); v[idx] = 1.0
    m = v - np.dot(v, n_hat) * n_hat; m /= np.linalg.norm(m)
    q = np.cross(n_hat, m);           q /= np.linalg.norm(q)
    m_c = m.astype(complex)
    q_c = q.astype(complex)

    # Per-region quantities
    S_mats, snn_list, smn_list, sqn_list = [], [], [], []
    for sig in sigmas:
        s = _to_tensor(sig)
        s_n = s @ n_c
        n_s = n_c @ s
        snn = complex(n_c @ s_n)
        if abs(snn) < 1e-30:
            # Degenerate — return arithmetic average as fallback
            tensors = [_to_tensor(sg) for sg in sigmas]
            return sum(complex(f) * t
                       for f, t in zip(vol_fracs, tensors))
        S_mats.append(s - np.outer(s_n, n_s) / snn)
        snn_list.append(snn)
        smn_list.append(complex(m_c @ s @ n_c))
        sqn_list.append(complex(q_c @ s @ n_c))

    # Energy matrix G (volume-weighted averages)
    G_TT = sum(complex(f) * S for f, S in zip(vol_fracs, S_mats))
    G_nn = sum(complex(f) / snn for f, snn in zip(vol_fracs, snn_list))
    G_mm = complex(m_c @ G_TT @ m_c)
    G_mq = complex(m_c @ G_TT @ q_c)
    G_qq = complex(q_c @ G_TT @ q_c)
    G_3x3 = np.array([[G_mm, G_mq, 0.],
                       [G_mq, G_qq, 0.],
                       [0.,   0.,  G_nn]], dtype=complex)

    # D1, D2, D3 diagonal entries (line-averaged per axis)
    D1 = np.zeros(3, dtype=complex)
    D2 = np.zeros(3, dtype=complex)
    D3 = np.zeros(3, dtype=complex)
    for r, (snn, smn, sqn) in enumerate(zip(snn_list, smn_list, sqn_list)):
        for k in range(3):
            fk = complex(line_fracs[r, k])
            D1[k] += fk / snn
            D2[k] += fk * smn / snn
            D3[k] += fk * sqn / snn

    # L̃ = [m̂ − D2⊙n̂ | q̂ − D3⊙n̂ | D1⊙n̂]
    L_tilde = np.column_stack([m_c - D2 * n_c,
                                q_c - D3 * n_c,
                                D1  * n_c])
    try:
        L_inv = np.linalg.inv(L_tilde)
    except np.linalg.LinAlgError:
        tensors = [_to_tensor(sg) for sg in sigmas]
        return sum(complex(f) * t for f, t in zip(vol_fracs, tensors))

    return L_inv.T @ G_3x3 @ L_inv


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def homogeneous_isotropic(
    grid: LebedevGrid3D,
    sigma: float,
    mu: float = MU0,
    eps: float = EPS0,
) -> EMMedia:
    """
    Uniform isotropic medium everywhere on the grid.

    Parameters
    ----------
    sigma : float  [S/m]
    mu    : float  [H/m]  default = μ₀
    eps   : float  [F/m]  default = ε₀
    """
    sigma_R = np.full(grid.N_R, complex(sigma))
    mu_P    = np.full(grid.N_P, complex(mu))
    eps_R   = np.full(grid.N_R, complex(eps))
    return EMMedia(grid, sigma_R, mu_P, eps_R)


def layered_isotropic(
    grid: LebedevGrid3D,
    layer_boundaries: list[float],
    sigma_values: list[float],
    mu_values: list[float] | None = None,
    eps_values: list[float] | None = None,
    direction: str = "z",
    interface_averaging: bool = False,
    nodal_averaging: bool = False,
) -> EMMedia:
    """
    Isotropic layered medium stratified along *direction* ('x', 'y', or 'z').

    Parameters
    ----------
    layer_boundaries : list of float
        N-1 boundary coordinates separating N layers (must be increasing).
        Layers are defined as: (−∞, b₀], (b₀, b₁], …, (b_{N-2}, +∞).
    sigma_values : list of float, length N
        Conductivity in each layer [S/m].
    mu_values, eps_values : lists of float, length N, or None (→ μ₀, ε₀).
    direction : str
        Stratification direction ('x', 'y', or 'z').
    interface_averaging : bool, default False
        If True, replace dual cells that straddle a layer boundary with the
        *standard* arithmetic/harmonic effective tensor (DDH03 eq. 8).
        Mutually exclusive with nodal_averaging.
    nodal_averaging : bool, default False
        If True, replace straddling dual cells using the *nodal*
        homogenization formula (Moskow et al. 1999).  For axis-aligned
        interfaces this gives ΣD = diag(σ̄, σ̄, σ̃) (see module docstring).
        Mutually exclusive with interface_averaging.

    Notes
    -----
    Both interface_averaging and nodal_averaging are disabled by default.
    Pointwise assignment (both False) is recommended for the Lebedev scheme;
    the 4-cluster structure already cancels leading-order interface artefacts.
    """
    if interface_averaging and nodal_averaging:
        raise ValueError("interface_averaging and nodal_averaging are mutually exclusive.")

    if mu_values is None:
        mu_values = [MU0] * len(sigma_values)
    if eps_values is None:
        eps_values = [EPS0] * len(sigma_values)

    bounds = np.array(layer_boundaries, dtype=float)
    coord_map = {"x": 0, "y": 1, "z": 2}
    ax = coord_map[direction]

    grid_ax = (grid.x, grid.y, grid.z)[ax]  # 1-D coordinate array for this axis

    sigma_arr = np.array(sigma_values, dtype=complex)
    mu_arr    = np.array(mu_values,    dtype=complex)
    eps_arr   = np.array(eps_values,   dtype=complex)

    def _layer_idx(coord: float) -> int:
        return int(np.searchsorted(bounds, coord, side="right"))

    do_avg = interface_averaging or nodal_averaging

    # ------------------------------------------------------------------
    # R-nodes: assign σ and ε  (two-pass to avoid partially-filled array bug)
    # ------------------------------------------------------------------
    # Pass 1: fill scalars for all R-nodes.
    sigma_R_scalar = np.empty(grid.N_R, dtype=complex)
    eps_R_scalar   = np.empty(grid.N_R, dtype=complex)
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        lid = _layer_idx(float(grid_ax[(i, j, k)[ax]]))
        sigma_R_scalar[seq] = sigma_arr[lid]
        eps_R_scalar[seq]   = eps_arr[lid]

    # Pass 2: upgrade to tensor where a dual cell straddles a boundary.
    sigma_R_tensor = None   # (N_R, 3, 3), allocated on first straddle
    if do_avg:
        for seq, (i, j, k) in enumerate(grid.R_nodes):
            c = float(grid_ax[(i, j, k)[ax]])

            # Skip if the node itself sits on a layer boundary.
            if np.any(np.abs(bounds - c) < 1e-10 * (np.abs(bounds) + 1.0)):
                continue

            idx   = (i, j, k)[ax]
            idx_m = idx - 1
            idx_p = idx + 1
            if idx_m < 0 or idx_p > len(grid_ax) - 1:
                continue   # boundary node — no dual cell on both sides

            c_minus = float(grid_ax[idx_m])
            c_plus  = float(grid_ax[idx_p])

            if nodal_averaging:
                t = _nodal_effective_tensor_layered(
                    c_minus, c_plus, bounds, sigma_arr, ax)
            else:
                t = _effective_tensor(
                    c_minus, c_plus, bounds, sigma_arr, ax)

            if t is None:
                continue

            # Allocate tensor on first encounter (sigma_R_scalar fully populated).
            if sigma_R_tensor is None:
                sigma_R_tensor = np.zeros((grid.N_R, 3, 3), dtype=complex)
                for d in range(3):
                    sigma_R_tensor[:, d, d] = sigma_R_scalar

            sigma_R_tensor[seq] = t

    sigma_R_out = sigma_R_tensor if sigma_R_tensor is not None else sigma_R_scalar

    # ------------------------------------------------------------------
    # P-nodes: assign μ  (same two-pass pattern)
    # ------------------------------------------------------------------
    mu_P_scalar = np.empty(grid.N_P, dtype=complex)
    for seq, (i, j, k) in enumerate(grid.P_nodes):
        lid = _layer_idx(float(grid_ax[(i, j, k)[ax]]))
        mu_P_scalar[seq] = mu_arr[lid]

    mu_P_tensor = None
    if do_avg:
        for seq, (i, j, k) in enumerate(grid.P_nodes):
            c = float(grid_ax[(i, j, k)[ax]])

            if np.any(np.abs(bounds - c) < 1e-10 * (np.abs(bounds) + 1.0)):
                continue

            idx   = (i, j, k)[ax]
            idx_m = idx - 1
            idx_p = idx + 1
            if idx_m < 0 or idx_p > len(grid_ax) - 1:
                continue

            c_minus = float(grid_ax[idx_m])
            c_plus  = float(grid_ax[idx_p])

            if nodal_averaging:
                t = _nodal_effective_tensor_layered(
                    c_minus, c_plus, bounds, mu_arr, ax)
            else:
                t = _effective_tensor(
                    c_minus, c_plus, bounds, mu_arr, ax)

            if t is None:
                continue

            if mu_P_tensor is None:
                mu_P_tensor = np.zeros((grid.N_P, 3, 3), dtype=complex)
                for d in range(3):
                    mu_P_tensor[:, d, d] = mu_P_scalar

            mu_P_tensor[seq] = t

    mu_P_out = mu_P_tensor if mu_P_tensor is not None else mu_P_scalar

    return EMMedia(grid, sigma_R_out, mu_P_out, eps_R_scalar)


def thin_layer_planar_isotropic(
    grid: LebedevGrid3D,
    n_hat: np.ndarray,
    d_center: float,
    thickness: float,
    sigma_bg: float,
    sigma_layer: float,
    mu_bg: float = MU0,
    mu_layer: float = MU0,
    eps_bg: float = EPS0,
    eps_layer: float = EPS0,
    method: str = "nodal",
) -> EMMedia:
    """
    Three-region medium: background | thin planar layer | background.

    The layer occupies  d_center − thickness/2 < n̂·x < d_center + thickness/2,
    where n̂ is the layer normal at arbitrary orientation.  This is equivalent
    to two parallel planar interfaces separated by *thickness* in the n̂
    direction, and the construction works for any σ varying only in the n̂
    direction within each dual cell.

    Advantages over pointwise assignment: the per-axis line averages d_k
    correctly weight the layer resistance even when the layer is thinner
    than the grid cell.  Pointwise will either miss the layer entirely (no
    node falls inside) or over-represent it (many nodes land inside).

    Parameters
    ----------
    n_hat : array-like (3,)
        Unit normal to the layer planes (will be normalised).
    d_center : float
        Centre of the layer in the n̂·x coordinate.
    thickness : float
        Layer thickness measured along n̂.
    sigma_bg, sigma_layer : float  [S/m]
    mu_bg, mu_layer : float  [H/m]
    eps_bg, eps_layer : float  [F/m]
    method : str  — ``"nodal"``, ``"standard"``, or ``"pointwise"``
    """
    if method not in ("nodal", "standard", "pointwise"):
        raise ValueError(f"method must be 'nodal', 'standard', or 'pointwise'")

    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)

    d1 = d_center - 0.5 * thickness
    d2 = d_center + 0.5 * thickness
    _EPS_FRAC = 1e-12

    def _in_layer(xyz):
        v = float(n_hat @ np.asarray(xyz))
        return d1 <= v < d2

    def _eff_tensor(box_min, box_max, node, p_bg, p_lyr):
        """Effective tensor (nodal or standard) for one straddling cell."""
        f_v1  = _volume_frac_layer1_planar(box_min, box_max, n_hat, d1)
        f_v2  = _volume_frac_layer1_planar(box_min, box_max, n_hat, d2)
        f_lyr = float(np.clip(f_v2 - f_v1, 0.0, 1.0))
        f_bg  = 1.0 - f_lyr
        s_arith = complex(f_lyr * p_lyr + f_bg * p_bg)
        inv_s_h = complex(f_lyr / p_lyr + f_bg / p_bg)

        if method == "standard":
            n_c = n_hat.astype(complex)
            return (s_arith * (np.eye(3, dtype=complex) - np.outer(n_c, n_c))
                    + (1.0 / inv_s_h) * np.outer(n_c, n_c))

        # Nodal: compute per-axis line averages of p⁻¹.
        D_diag = np.empty(3, dtype=complex)
        for ax in range(3):
            rest = float(n_hat @ node) - n_hat[ax] * node[ax]
            f1 = _frac_1d_layer1(box_min[ax], box_max[ax], n_hat[ax], d1 - rest)
            f2 = _frac_1d_layer1(box_min[ax], box_max[ax], n_hat[ax], d2 - rest)
            f_ax_lyr = float(np.clip(f2 - f1, 0.0, 1.0))
            D_diag[ax] = (1.0 - f_ax_lyr) / p_bg + f_ax_lyr / p_lyr
        return _nodal_eff_tensor_3d(D_diag, s_arith, inv_s_h, n_hat)

    def _build(node_list, cx, cy, cz, p_bg, p_lyr):
        N = len(node_list)
        scalar = np.array([
            complex(p_lyr) if _in_layer((cx[i], cy[j], cz[k])) else complex(p_bg)
            for i, j, k in node_list
        ])
        if method == "pointwise":
            return scalar

        tensor = None
        gc = [cx, cy, cz]
        gl = [len(cx), len(cy), len(cz)]
        for seq, (i, j, k) in enumerate(node_list):
            node    = np.array([float(cx[i]), float(cy[j]), float(cz[k])])
            idxs    = [i, j, k]
            box_min = np.array([gc[a][max(idxs[a]-1, 0)]       for a in range(3)])
            box_max = np.array([gc[a][min(idxs[a]+1, gl[a]-1)] for a in range(3)])

            corners = [n_hat[0]*x + n_hat[1]*y + n_hat[2]*z
                       for x in [box_min[0], box_max[0]]
                       for y in [box_min[1], box_max[1]]
                       for z in [box_min[2], box_max[2]]]
            if all(v < d1 for v in corners) or all(v >= d2 for v in corners):
                continue

            # Fake-straddle guard: an edge must actually cross d1 or d2.
            crosses = False
            for ax in range(3):
                rest = float(n_hat @ node) - n_hat[ax] * node[ax]
                f1 = _frac_1d_layer1(box_min[ax], box_max[ax], n_hat[ax], d1 - rest)
                f2 = _frac_1d_layer1(box_min[ax], box_max[ax], n_hat[ax], d2 - rest)
                if (_EPS_FRAC < f1 < 1-_EPS_FRAC) or (_EPS_FRAC < f2 < 1-_EPS_FRAC):
                    crosses = True; break
            if not crosses:
                continue

            t = _eff_tensor(box_min, box_max, node, p_bg, p_lyr)
            if tensor is None:
                tensor = np.zeros((N, 3, 3), dtype=complex)
                for d in range(3):
                    tensor[:, d, d] = scalar
            tensor[seq] = t

        return tensor if tensor is not None else scalar

    sigma_R_out = _build(grid.R_nodes, grid.x, grid.y, grid.z,
                         sigma_bg, sigma_layer)
    mu_P_out    = _build(grid.P_nodes, grid.x, grid.y, grid.z,
                         mu_bg, mu_layer)
    eps_R = np.array([
        complex(eps_layer) if _in_layer((grid.x[i], grid.y[j], grid.z[k]))
        else complex(eps_bg)
        for i, j, k in grid.R_nodes
    ])
    return EMMedia(grid, sigma_R_out, mu_P_out, eps_R)


# ---------------------------------------------------------------------------
# Fine-grid upscaling helpers (private)
# ---------------------------------------------------------------------------

def _trilinear_interp_fine(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    z_arr: np.ndarray,
    values: np.ndarray,
    x: float,
    y: float,
    z: float,
) -> complex:
    """
    Trilinear interpolation of *values* (shape nx×ny×nz) at (x, y, z).
    Clamps to the grid boundary for out-of-range points.
    """
    def _w1d(arr, v):
        n = len(arr)
        idx = int(np.clip(np.searchsorted(arr, v, side="right") - 1, 0, n - 2))
        lo, hi = float(arr[idx]), float(arr[idx + 1])
        span = hi - lo
        t = float(np.clip((v - lo) / span, 0.0, 1.0)) if span > 1e-30 else 0.5
        return idx, t

    ix, tx = _w1d(x_arr, x)
    iy, ty = _w1d(y_arr, y)
    iz, tz = _w1d(z_arr, z)
    v = values
    return complex(
        (1 - tx) * (1 - ty) * (1 - tz) * v[ix,     iy,     iz    ] +
              tx  * (1 - ty) * (1 - tz) * v[ix + 1, iy,     iz    ] +
        (1 - tx) *       ty  * (1 - tz) * v[ix,     iy + 1, iz    ] +
              tx  *       ty  * (1 - tz) * v[ix + 1, iy + 1, iz    ] +
        (1 - tx) * (1 - ty) *       tz  * v[ix,     iy,     iz + 1] +
              tx  * (1 - ty) *       tz  * v[ix + 1, iy,     iz + 1] +
        (1 - tx) *       ty  *       tz  * v[ix,     iy + 1, iz + 1] +
              tx  *       ty  *       tz  * v[ix + 1, iy + 1, iz + 1]
    )


def _extract_fine_block(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    z_arr: np.ndarray,
    values: np.ndarray,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    z_lo: float,
    z_hi: float,
) -> np.ndarray | None:
    """
    Extract the sub-block of *values* whose grid points lie inside the box
    [x_lo, x_hi] × [y_lo, y_hi] × [z_lo, z_hi].

    Returns None if the box contains no fine-grid points.
    """
    ix0 = int(np.searchsorted(x_arr, x_lo, side="left"))
    ix1 = int(np.searchsorted(x_arr, x_hi, side="right"))
    iy0 = int(np.searchsorted(y_arr, y_lo, side="left"))
    iy1 = int(np.searchsorted(y_arr, y_hi, side="right"))
    iz0 = int(np.searchsorted(z_arr, z_lo, side="left"))
    iz1 = int(np.searchsorted(z_arr, z_hi, side="right"))
    if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
        return None
    return values[ix0:ix1, iy0:iy1, iz0:iz1]


def _estimate_normal_svd(
    block: np.ndarray,
    x_sub: np.ndarray,
    y_sub: np.ndarray,
    z_sub: np.ndarray,
    rel_threshold: float = 0.01,
) -> tuple[np.ndarray | None, float]:
    """
    Estimate the unit interface normal from the geometry of interface voxels.

    Strategy
    --------
    Rather than differencing σ values (which is biased by grid-spacing
    anisotropy — a 7:1 z/x spacing ratio inflates gz by 7× relative to gx),
    we locate the *positions* of voxels that straddle the interface and fit a
    plane through them.

    For a planar interface the interface-voxel centres all lie on a thin slab
    centred on the plane.  The minimum-variance direction of their positions
    (last right singular vector of the centred position matrix) is the plane
    normal.  This is insensitive to grid anisotropy because it uses physical
    coordinates, not finite differences.

    Algorithm
    ----------
    1. Binarise: voxel is in "layer 1" iff  σ < median(σ_min, σ_max).
    2. For each of the 6 face-adjacency directions (±x, ±y, ±z), mark voxels
       whose neighbour belongs to the other layer.
    3. Collect the physical (x, y, z) centres of all such interface voxels.
    4. Centre the position matrix and compute its SVD; the *last* right
       singular vector (minimum variance) is n̂.  The *planarity ratio* is
       s_last / s_first: small → planar (reliable n̂), large → no clear interface.

    Parameters
    ----------
    block : (nx, ny, nz) real array — conductivity values in the dual cell.
    x_sub, y_sub, z_sub : 1-D coordinate arrays for the block axes.
    rel_threshold : float
        Unused (kept for API compatibility with callers).

    Returns
    -------
    n_hat : (3,) unit vector, or None if the block is effectively uniform.
    planarity_ratio : float
        Ratio s_last / s_first of the position-SVD singular values.
        → 0  : perfectly planar interface; n̂ is reliable.
        → 1  : positions span 3-D equally; n̂ is unreliable.
    """
    sig = np.real(block).astype(float)
    sig_range = sig.max() - sig.min()
    if sig_range < 1e-14 * (abs(sig.min()) + 1.0):
        return None, 1.0   # uniform block

    # Binarise by median of (min, max) so we split the two layers cleanly
    # even when one layer is much thicker than the other.
    sig_mid = 0.5 * (sig.max() + sig.min())
    binary = (sig >= sig_mid)   # True = layer 2

    XX, YY, ZZ = np.meshgrid(x_sub, y_sub, z_sub, indexing="ij")
    nx, ny, nz = sig.shape

    # Collect the MIDPOINT of each adjacent crossing pair.
    # Using the midpoint (average of the two voxel centres) places each sample
    # directly on the interface rather than on one side of it.  This avoids the
    # "flooding" artefact where a single voxel adjacent to many crossings on both
    # sides contributes positions spanning the full z-range of the block while
    # remaining at a fixed x, inflating z-variance and biasing the SVD.
    pos_list = []
    for ax in range(3):
        sl_a = [slice(None), slice(None), slice(None)]
        sl_b = [slice(None), slice(None), slice(None)]
        sl_a[ax] = slice(None, -1)
        sl_b[ax] = slice(1,    None)
        sl_a, sl_b = tuple(sl_a), tuple(sl_b)
        mask = binary[sl_a] != binary[sl_b]
        if mask.any():
            # Midpoint of the two neighbouring voxel centres
            mid_X = 0.5 * (XX[sl_a][mask] + XX[sl_b][mask])
            mid_Y = 0.5 * (YY[sl_a][mask] + YY[sl_b][mask])
            mid_Z = 0.5 * (ZZ[sl_a][mask] + ZZ[sl_b][mask])
            pos_list.append(np.column_stack([mid_X, mid_Y, mid_Z]))

    if not pos_list:
        return None, 1.0

    P = np.vstack(pos_list)
    if len(P) < 3:
        return None, 1.0

    # Remove duplicates (same crossing seen from both orientations)
    P = np.unique(P, axis=0)
    if len(P) < 3:
        return None, 1.0

    P_c = P - P.mean(axis=0)
    _, s, Vt = np.linalg.svd(P_c, full_matrices=False)

    if s[0] < 1e-14:
        return None, 1.0

    n_hat = Vt[-1].copy()   # minimum-variance direction = plane normal
    n_hat /= np.linalg.norm(n_hat)
    planarity_ratio = float(s[-1] / s[0])   # small ≈ planar, large ≈ isotropic

    # Sign convention: n_hat must point toward the higher-σ region.
    # Project every voxel centre onto n_hat; voxels on the positive side
    # should have higher mean σ.  If not, flip n_hat.
    block_centroid = np.array([XX.mean(), YY.mean(), ZZ.mean()])
    proj = (  n_hat[0] * (XX - block_centroid[0])
            + n_hat[1] * (YY - block_centroid[1])
            + n_hat[2] * (ZZ - block_centroid[2]))
    sig_pos = sig[proj >  0].mean() if (proj >  0).any() else 0.0
    sig_neg = sig[proj <= 0].mean() if (proj <= 0).any() else 0.0
    if sig_pos < sig_neg:
        n_hat = -n_hat

    return n_hat, planarity_ratio


def _node_line_indices(
    x_sub: np.ndarray,
    y_sub: np.ndarray,
    z_sub: np.ndarray,
    node_xyz,
) -> tuple[int, int, int]:
    """
    Sub-grid indices of the sample nearest the R-node along each axis.

    The nodal line averages (tex note eqs. A.8-A.10) are 1-D integrals along
    the grid axes *through the node*, with the transverse coordinates held
    fixed at their node values.  On non-uniform (e.g. geometric) grids the
    node is NOT at the centre of the dual-cell box [x_{i-1}, x_{i+1}], so the
    box-centre index n//2 samples the wrong line.  Ties (node exactly midway
    between two samples, as on uniform grids with an even sample count) are
    broken toward the upper index, matching the previous n//2 convention so
    uniform-grid results are unchanged.
    """
    def _near(arr: np.ndarray, v: float) -> int:
        d = np.abs(np.asarray(arr, dtype=float) - float(v))
        # last index attaining the minimum → upper index on exact ties
        return int(len(d) - 1 - np.argmin(d[::-1]))

    return (_near(x_sub, node_xyz[0]),
            _near(y_sub, node_xyz[1]),
            _near(z_sub, node_xyz[2]))


def _pair_tols_svd(svals: np.ndarray, i: int, j: int):
    """
    Compute adaptive tolerances (tol_lo, tol_hi) for masked per-pair sub-SVD.

    For the pair (svals[i], svals[j]), the tolerance is chosen tight enough
    to exclude all other sigma values in svals, so that only voxels genuinely
    belonging to one of the two target materials contribute to the SVD.

    The rule: tol = 0.4 × min(gap to nearest other value, external gap).
    """
    n = len(svals)
    inner_above_lo = svals[i + 1] if i + 1 < j else None
    inner_below_hi = svals[j - 1] if j - 1 > i else None
    gap_lo = (inner_above_lo - svals[i]) if inner_above_lo is not None else (svals[j] - svals[i])
    gap_hi = (svals[j] - inner_below_hi) if inner_below_hi is not None else (svals[j] - svals[i])
    ext_below = (svals[i] - svals[i - 1]) if i > 0   else (svals[j] - svals[i])
    ext_above = (svals[j + 1] - svals[j]) if j + 1 < n else (svals[j] - svals[i])
    return 0.4 * min(gap_lo, ext_below), 0.4 * min(gap_hi, ext_above)


def _estimate_normal_svd_pair(
    block: np.ndarray,
    x_sub: np.ndarray,
    y_sub: np.ndarray,
    z_sub: np.ndarray,
    s_lo: float,
    s_hi: float,
    tol_lo: float,
    tol_hi: float,
) -> "tuple[np.ndarray | None, float]":
    """
    Masked per-pair interface-normal estimator.

    Identical in spirit to :func:`_estimate_normal_svd`, but only counts
    crossings between voxels within *tol_lo* of *s_lo* or within *tol_hi*
    of *s_hi*.  Voxels belonging to any other material are excluded, so that
    in a multi-material cell each individual interface can be characterised
    independently without contamination from the other interfaces.

    Parameters
    ----------
    block                : (nx, ny, nz) real conductivity array.
    x_sub, y_sub, z_sub  : 1-D coordinate arrays.
    s_lo, s_hi           : target sigma values for the two materials.
    tol_lo, tol_hi       : voxels with |σ−s_lo|≤tol_lo are class 0;
                           voxels with |σ−s_hi|≤tol_hi are class 1;
                           all others are excluded (binary = −1).

    Returns
    -------
    n_hat : (3,) unit vector or None.
    planarity_ratio : float (s_last/s_first of position-SVD singular values).
    """
    sig = np.real(block).astype(float)
    mask_lo = np.abs(sig - s_lo) <= tol_lo
    mask_hi = np.abs(sig - s_hi) <= tol_hi
    if not (mask_lo | mask_hi).any():
        return None, 1.0

    binary = np.full(sig.shape, -1, dtype=np.int8)
    binary[mask_lo] = 0
    binary[mask_hi] = 1

    XX, YY, ZZ = np.meshgrid(x_sub, y_sub, z_sub, indexing="ij")
    pos_list = []
    for ax in range(3):
        sl_a = [slice(None), slice(None), slice(None)]
        sl_b = [slice(None), slice(None), slice(None)]
        sl_a[ax] = slice(None, -1)
        sl_b[ax] = slice(1,    None)
        sl_a, sl_b = tuple(sl_a), tuple(sl_b)
        both_valid = (binary[sl_a] >= 0) & (binary[sl_b] >= 0)
        crossing   = (binary[sl_a] != binary[sl_b]) & both_valid
        if crossing.any():
            mid_X = 0.5 * (XX[sl_a][crossing] + XX[sl_b][crossing])
            mid_Y = 0.5 * (YY[sl_a][crossing] + YY[sl_b][crossing])
            mid_Z = 0.5 * (ZZ[sl_a][crossing] + ZZ[sl_b][crossing])
            pos_list.append(np.column_stack([mid_X, mid_Y, mid_Z]))

    if not pos_list:
        return None, 1.0
    P = np.vstack(pos_list)
    if len(P) < 3:
        return None, 1.0
    P = np.unique(P, axis=0)
    if len(P) < 3:
        return None, 1.0
    P_c = P - P.mean(axis=0)
    _, s, Vt = np.linalg.svd(P_c, full_matrices=False)
    if s[0] < 1e-14:
        return None, 1.0

    n_hat = Vt[-1].copy()
    ratio  = float(s[-1] / s[0])

    # Sign convention: n_hat points from s_lo region toward s_hi region
    # (s_hi > s_lo by construction since s_hi > s_lo).
    XX, YY, ZZ = np.meshgrid(x_sub, y_sub, z_sub, indexing="ij")
    centroid = np.array([XX.mean(), YY.mean(), ZZ.mean()])
    proj = (  n_hat[0] * (XX - centroid[0])
            + n_hat[1] * (YY - centroid[1])
            + n_hat[2] * (ZZ - centroid[2]))
    sig_pos = sig[proj >  0].mean() if (proj >  0).any() else 0.0
    sig_neg = sig[proj <= 0].mean() if (proj <= 0).any() else 0.0
    if sig_pos < sig_neg:
        n_hat = -n_hat

    return n_hat, ratio


def _nodal_multimat_3d(
    block: np.ndarray,
    x_sub: np.ndarray,
    y_sub: np.ndarray,
    z_sub: np.ndarray,
    svals: np.ndarray,
    block_tensor,
    is_tensor: bool,
    iso_tol: float,
    n_hat_combined: "np.ndarray | None",
    node_xyz=None,
    block_complex: "np.ndarray | None" = None,
) -> "np.ndarray | None":
    """
    Effective conductivity tensor for a dual cell containing exactly 3 distinct
    conductivity values.

    *node_xyz* (optional) gives the physical R-node coordinates; per-axis
    line fractions are then taken along the sub-grid lines through the node
    (tex note eqs. A.8-A.10) rather than the box-centre lines.  *block_complex*
    (optional) carries the complex σ̇ values matching *block* (which holds the
    real part used for material classification); material conductivities are
    read from it so the imaginary part survives the averaging.

    Uses masked per-pair sub-SVDs (:func:`_estimate_normal_svd_pair`) to
    identify the interface normals and decides between two strategies:

    **All interfaces parallel** (|n̂ᵢ·n̂ⱼ| > 0.90 for all pairs):
        → :func:`_nodal_eff_tensor_multiregion` with common n̂.

    **Two distinct normals** (adjacent pairs share one n̂; skip pair gives
    a different n̂):
        → Sequential Backus.  The skip pair (indices 0,2) identifies the
        secondary boundary between the two outer materials.  The middle
        material (index 1 in sorted order) is the inner material separated
        from both outer materials by the primary interface.

        Step 1: nodal-homogenise the two outer materials {σ₀, σ₂} using n̂_sec.
        Step 2: nodal-homogenise σ₁ against σ_outer using n̂_pri.

        When the adjacent-pair sub-SVDs return None (low-contrast boundary not
        resolved), *n_hat_combined* (the combined full-block SVD result) is used
        as n̂_pri.  If *n_hat_combined* is also None, the function returns None
        and the caller falls back to the standard single-interface path.

    Returns
    -------
    (3,3) tensor or None (caller should fall back to standard path).
    """
    if len(svals) != 3:
        return None

    block_r = np.round(block, 6)

    def _get_sig(idx):
        m = np.abs(block_r - svals[idx]) < 1e-5
        if is_tensor and block_tensor is not None and m.any():
            return np.mean(block_tensor[m], axis=0)
        if block_complex is not None and m.any():
            # Complex-preserving: classification used the real part; the
            # material value keeps its imaginary component.
            return complex(np.mean(block_complex[m])) * np.eye(3, dtype=complex)
        return complex(svals[idx]) * np.eye(3, dtype=complex)

    # Line indices through the NODE (not the box centre) — see _node_line_indices.
    if node_xyz is not None:
        ic, jc, kc = _node_line_indices(x_sub, y_sub, z_sub, node_xyz)
    else:
        nb_x, nb_y, nb_z = block.shape
        ic, jc, kc = nb_x // 2, nb_y // 2, nb_z // 2

    def _frac(idx):
        m = np.abs(block_r - svals[idx]) < 1e-5
        fv = float(m.mean())
        fl = np.array([float(m[:, jc, kc].mean()),
                       float(m[ic, :, kc].mean()),
                       float(m[ic, jc, :].mean())])
        return fv, fl

    # Masked sub-SVD for all 3 pairs
    nhs = {}
    rts = {}
    for pi, pj in [(0, 1), (1, 2), (0, 2)]:
        tl, th = _pair_tols_svd(svals, pi, pj)
        nhs[(pi, pj)], rts[(pi, pj)] = _estimate_normal_svd_pair(
            block, x_sub, y_sub, z_sub, svals[pi], svals[pj], tl, th)

    v01 = nhs[(0, 1)] is not None and rts[(0, 1)] < iso_tol
    v12 = nhs[(1, 2)] is not None and rts[(1, 2)] < iso_tol
    v02 = nhs[(0, 2)] is not None and rts[(0, 2)] < iso_tol

    # ── Determine structure ────────────────────────────────────────────────────
    n_pri: "np.ndarray | None" = None
    n_sec: "np.ndarray | None" = None
    do_sequential = False
    do_multiregion = False

    if v01 and v12:
        par_01_12 = abs(float(np.dot(nhs[(0, 1)], nhs[(1, 2)]))) > 0.90
        if v02:
            par_01_02 = abs(float(np.dot(nhs[(0, 1)], nhs[(0, 2)]))) > 0.90
        else:
            par_01_02 = True    # assume parallel if skip pair unresolved

        if par_01_12 and par_01_02:
            do_multiregion = True
            # Align all normals to the same hemisphere before averaging
            _ref = nhs[(0, 1)].copy()
            normals = [_ref]
            _n12a = nhs[(1, 2)].copy()
            if float(np.dot(_ref, _n12a)) < 0: _n12a = -_n12a
            normals.append(_n12a)
            if v02:
                _n02a = nhs[(0, 2)].copy()
                if float(np.dot(_ref, _n02a)) < 0: _n02a = -_n02a
                normals.append(_n02a)
            n_hat_cm = np.mean(normals, axis=0)
            _cm_norm = np.linalg.norm(n_hat_cm)
            n_hat_cm = n_hat_cm / _cm_norm if _cm_norm > 1e-10 else _ref

        elif par_01_12 and not par_01_02:
            # (0,1) ‖ (1,2)  ⊥  (0,2) → skip pair gives secondary normal
            do_sequential = True
            _n01c = nhs[(0, 1)].copy()
            _n12c = nhs[(1, 2)].copy()
            # Ensure same hemisphere before averaging (SVD sign is arbitrary)
            if float(np.dot(_n01c, _n12c)) < 0:
                _n12c = -_n12c
            n_pri = 0.5 * (_n01c + _n12c)
            _pnorm = np.linalg.norm(n_pri)
            n_pri = n_pri / _pnorm if _pnorm > 1e-10 else _n01c
            n_sec = nhs[(0, 2)].copy()

        else:
            return None   # ambiguous structure — fall through

    elif v02:
        # Only skip pair resolved; use n_hat_combined as primary (or give up)
        if n_hat_combined is not None and rts.get((0,2), 1.0) < iso_tol:
            do_sequential = True
            n_pri = n_hat_combined.copy()
            n_sec = nhs[(0, 2)].copy()
        else:
            return None

    else:
        return None

    # ── Multiregion nodal ──────────────────────────────────────────────────────
    if do_multiregion:
        vols = []; lfs = []; sigs = []
        for idx in range(3):
            fv, fl = _frac(idx)
            if fv < 1e-10: continue
            vols.append(fv); lfs.append(fl); sigs.append(_get_sig(idx))
        vols = np.array(vols); lfs = np.array(lfs)
        vols /= vols.sum()
        for k in range(3):
            c = lfs[:, k].sum()
            if c > 1e-12: lfs[:, k] /= c
        return _nodal_eff_tensor_multiregion(sigs, vols, lfs, n_hat_cm)

    # ── Sequential Backus ──────────────────────────────────────────────────────
    # Outer pair: {svals[0], svals[2]}; inner material: svals[1]
    # Step 1 — homogenise outer pair with n̂_sec
    m0 = np.abs(block_r - svals[0]) < 1e-5
    m2 = np.abs(block_r - svals[2]) < 1e-5
    m_out = m0 | m2
    f_outer_vol = float(m_out.mean())
    if f_outer_vol > 1e-12:
        f0_in_out = float(m0.mean()) / f_outer_vol
        lf0x = float(m0[:, jc, kc].sum()) / max(float(m_out[:, jc, kc].sum()), 1.0)
        lf0y = float(m0[ic, :, kc].sum()) / max(float(m_out[ic, :, kc].sum()), 1.0)
        lf0z = float(m0[ic, jc, :].sum()) / max(float(m_out[ic, jc, :].sum()), 1.0)
        lf0_in_out = np.array([lf0x, lf0y, lf0z])
        sigma_outer = _nodal_eff_tensor_general(
            _get_sig(0), _get_sig(2), f0_in_out, lf0_in_out, n_sec)
    else:
        sigma_outer = _get_sig(2)

    # Step 2 — homogenise inner (svals[1]) with sigma_outer using n̂_pri
    f1v, f1l = _frac(1)
    result = _nodal_eff_tensor_general(_get_sig(1), sigma_outer, f1v, f1l, n_pri)

    # ── Eigenvalue-overflow fallback (tex note §"Fallback") ────────────────────
    # A physically valid homogenized tensor is bounded by the constituents:
    # λ_max(ΣD) ≤ max_i λ_max(σ_i) ≤ max_i tr(σ_i) = 3 × max_i tr(σ_i)/3.
    # An eigenvalue above 3× the largest material conductivity signals
    # numerical degeneracy of the sequential result (typically a near-singular
    # L̃ when one material has very low normal conductivity).  Returning None
    # makes the caller fall back to the standard single-interface nodal path.
    _sig_max = max(float(np.real(np.trace(_get_sig(i)))) / 3.0 for i in range(3))
    try:
        _herm = np.real(0.5 * (result + result.T))
        _eig_max = float(np.max(np.linalg.eigvalsh(_herm)))
    except Exception:
        return None
    if not np.all(np.isfinite(result)) or _eig_max > 3.0 * max(_sig_max, 1e-30):
        return None
    return result


def _diag_d_from_block(
    block: np.ndarray,
    x_sub: np.ndarray,
    y_sub: np.ndarray,
    z_sub: np.ndarray,
    x_node: float,
    y_node: float,
    z_node: float,
) -> np.ndarray:
    """
    Compute D_diag[α] = line-average of σ⁻¹ along the α-axis through the node.

    This matches the per-axis line averages used in `_nodal_eff_tensor_3d`:
    for each axis α, the line through (x_node, y_node, z_node) within the
    block is identified by the nearest fine-grid indices on the two transverse
    axes, and the mean of 1/σ along that line is returned.

    For a planar interface this recovers the fraction-weighted 1/σ average
    that the analytic `planar_interface_isotropic` computes via `_frac_1d_layer1`.
    """
    inv_b = 1.0 / block
    ix = int(np.clip(np.argmin(np.abs(x_sub - x_node)), 0, block.shape[0] - 1))
    iy = int(np.clip(np.argmin(np.abs(y_sub - y_node)), 0, block.shape[1] - 1))
    iz = int(np.clip(np.argmin(np.abs(z_sub - z_node)), 0, block.shape[2] - 1))
    return np.array([
        complex(np.mean(inv_b[:, iy, iz])),   # x-line
        complex(np.mean(inv_b[ix, :, iz])),   # y-line
        complex(np.mean(inv_b[ix, iy, :])),   # z-line
    ], dtype=complex)


def _diagonal_eff_sigma(block: np.ndarray) -> tuple[complex, complex, complex]:
    """
    Per-axis effective conductivity for a fine-grid block.

    For axis α: σ_αα = arithmetic mean over transverse positions of the
    harmonic mean along each line parallel to α within the block.

    This is the Backus (1962) series-parallel formula applied to the
    3-D block.  It correctly recovers:
      • σ_αα = σ̄  (arithmetic)  for axes *tangential* to any internal layering
      • σ_αα = σ̃  (harmonic)   for the axis *normal* to the layering

    Parameters
    ----------
    block : ndarray, shape (nx, ny, nz), complex
        Fine-grid conductivity values inside the dual cell.

    Returns
    -------
    (σ_xx, σ_yy, σ_zz) : complex scalars
    """
    inv_b = 1.0 / block                               # (nx, ny, nz)
    # σ_xx: harmonic along x-lines (axis=0), averaged over (y, z)
    s_xx = complex(np.mean(1.0 / np.mean(inv_b, axis=0)))
    # σ_yy: harmonic along y-lines (axis=1), averaged over (x, z)
    s_yy = complex(np.mean(1.0 / np.mean(inv_b, axis=1)))
    # σ_zz: harmonic along z-lines (axis=2), averaged over (x, y)
    s_zz = complex(np.mean(1.0 / np.mean(inv_b, axis=2)))
    return s_xx, s_yy, s_zz


def _standard_backus_tensor_3d(
    sigma_arith: complex,
    inv_sigma_vol: complex,
    n_hat: np.ndarray,
) -> np.ndarray:
    """
    Standard Backus / laminate homogenization for a two-component cell with
    interface normal n̂.

    ΣB = σ‖ (I − n̂⊗n̂) + σ⊥ (n̂⊗n̂)

    where σ‖ = σ̄ = ⟨σ⟩ (volume arithmetic mean) is the exact effective
    conductivity for currents flowing *tangentially* to the interface, and
    σ⊥ = 1/⟨σ⁻¹⟩ (volume harmonic mean) is the exact effective conductivity
    for currents flowing *normally*.

    This is the classical continuum result from homogenization / laminate
    theory (Backus 1962, Tartar 1977).  It is independent of the FD
    discretization — compare with `_nodal_eff_tensor_3d`, which accounts for
    the specific staggered-grid stencil.

    For an axis-aligned normal (n̂ = ê_α) this reduces to the same tensor as
    `_diagonal_eff_sigma`.  For tilted interfaces it gives different off-diagonal
    entries than the nodal formula.

    Parameters
    ----------
    sigma_arith   : complex   Volume arithmetic mean ⟨σ⟩.
    inv_sigma_vol : complex   Volume arithmetic mean ⟨σ⁻¹⟩.
    n_hat         : (3,) float  Unit interface normal.

    Returns
    -------
    ΣB : (3, 3) complex ndarray
    """
    sigma_perp = complex(1.0 / inv_sigma_vol)   # harmonic mean
    sigma_para = complex(sigma_arith)            # arithmetic mean
    n = n_hat.reshape(3, 1).astype(complex)
    return sigma_para * np.eye(3, dtype=complex) + (sigma_perp - sigma_para) * (n @ n.T)


# ---------------------------------------------------------------------------
# Fine-grid upscaling — public factory
# ---------------------------------------------------------------------------

def make_sigma_func(
    fine_x: np.ndarray,
    fine_y: np.ndarray,
    fine_z: np.ndarray,
    sigma_fine: np.ndarray,
    interp: str = "nearest",
) -> callable:
    """
    Build a vectorised callable ``σ(X, Y, Z) -> ndarray`` from a scalar
    conductivity field on a regular Cartesian grid.

    The callable accepts broadcastable arrays X, Y, Z and returns an array
    of the same shape, suitable for direct use with :func:`from_sigma_func`.

    Parameters
    ----------
    fine_x, fine_y, fine_z : array-like, 1-D, strictly increasing
        Coordinate axes of the input grid.
    sigma_fine : ndarray, shape (nx, ny, nz)
        Scalar conductivity [S/m].  Must be real positive (or complex with
        positive real part).
    interp : str
        Interpolation method passed to ``scipy.interpolate.RegularGridInterpolator``:

        ``"nearest"`` (default) — nearest-neighbour lookup.  Preserves the
            exact piecewise-constant values of the input grid.  Best for
            binary lithology models where layer conductivities must not be
            smeared.

        ``"linear"`` — trilinear interpolation.  Better for smoothly
            varying conductivity fields.

    Returns
    -------
    callable
        ``sigma_func(X, Y, Z)`` — evaluates conductivity at arbitrary
        broadcast-compatible coordinate arrays.  Points outside the input
        grid are clamped to the nearest boundary value (``bounds_error=False``,
        ``fill_value=None``).

    Examples
    --------
    >>> import numpy as np
    >>> from lebedev_em.media import make_sigma_func, from_sigma_func
    >>> fx = np.linspace(-5, 5, 100)
    >>> fy = np.linspace(-5, 5, 100)
    >>> fz = np.linspace(-5, 5, 200)   # fine in z, coarse in x/y — doesn't matter
    >>> sig = np.where(fz[None, None, :] >= 0, 1.0, 0.1)
    >>> sig = np.broadcast_to(sig, (len(fx), len(fy), len(fz))).copy()
    >>> sf = make_sigma_func(fx, fy, fz, sig)          # -> callable
    >>> # sf(X, Y, Z) returns correct σ at any point regardless of grid anisotropy
    """
    fine_x = np.asarray(fine_x, dtype=float)
    fine_y = np.asarray(fine_y, dtype=float)
    fine_z = np.asarray(fine_z, dtype=float)
    sigma_fine = np.asarray(sigma_fine)
    if sigma_fine.shape != (len(fine_x), len(fine_y), len(fine_z)):
        raise ValueError(
            f"sigma_fine shape {sigma_fine.shape} does not match "
            f"({len(fine_x)}, {len(fine_y)}, {len(fine_z)})"
        )
    if interp not in ("nearest", "linear"):
        raise ValueError(f"interp must be 'nearest' or 'linear', got {interp!r}")

    rgi = RegularGridInterpolator(
        (fine_x, fine_y, fine_z),
        sigma_fine,
        method=interp,
        bounds_error=False,
        fill_value=None,   # clamp to boundary
    )

    def sigma_func(X, Y, Z):
        X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z)
        shape = np.broadcast(X, Y, Z).shape
        pts = np.stack(
            [np.broadcast_to(X, shape).ravel(),
             np.broadcast_to(Y, shape).ravel(),
             np.broadcast_to(Z, shape).ravel()],
            axis=-1,
        )
        return rgi(pts).reshape(shape)

    return sigma_func


def from_fine_grid(
    grid: "LebedevGrid3D",
    fine_x: np.ndarray,
    fine_y: np.ndarray,
    fine_z: np.ndarray,
    sigma_fine: np.ndarray,
    mu: float = MU0,
    eps: float = EPS0,
    method: str = "arithmetic",
    iso_tol: float = 1e-6,
    svd_isotropy_tol: float = 0.7,
    h_svd: "float | tuple[float, float, float] | None" = None,
    n_line: int = 50,
    n_vol: int = 8,
) -> "EMMedia":
    """
    Build an EMMedia from a scalar conductivity field on a regular Cartesian
    grid, upscaled to the Lebedev FD grid.

    Convenience wrapper: builds a nearest-neighbour callable via
    :func:`make_sigma_func` and delegates to :func:`from_sigma_func`.  The
    SVD sub-grid spacing *h_svd* defaults to the minimum fine-grid spacing in
    any direction, ensuring the isotropic evaluation grid is at least as fine
    as the input data regardless of axis anisotropy.

    Parameters
    ----------
    grid : LebedevGrid3D
    fine_x, fine_y, fine_z : array-like, 1-D, strictly increasing
        Coordinate axes of the fine grid.
    sigma_fine : ndarray, shape (nx, ny, nz)
        Scalar conductivity [S/m].  Real positive (or complex with positive
        real part).
    mu : float   [H/m]   Uniform permeability.
    eps : float  [F/m]   Uniform permittivity.
    method : str
        Same as :func:`from_sigma_func`: ``"pointwise"``, ``"arithmetic"``,
        ``"diagonal"``, ``"backus"``, or ``"nodal"``.
    iso_tol : float
        Isotropy tolerance for diagonal fallback.
    svd_isotropy_tol : float
        SVD planarity threshold; cells with ratio >= this fall back to
        ``"diagonal"``.
    h_svd : float, (float, float, float), or None
        SVD evaluation-grid spacing [m].  ``None`` (default) uses the
        per-axis minimum fine-grid spacing ``(min dx, min dy, min dz)``,
        which is alias-free in every direction and is passed as a 3-tuple
        to :func:`from_sigma_func` so highly anisotropic fine grids (e.g.
        dz ≪ dx) are sampled correctly in each axis without over-sampling
        the coarse axes.  Pass a scalar to force uniform spacing, or a
        3-tuple to set axis spacings explicitly.
    n_line : int
        Quadrature points per axis for D_diag line averages (nodal).
        Default 50.
    n_vol : int
        Per-axis points for volume averages in the ``"diagonal"`` method.
        Default 8.

    Returns
    -------
    EMMedia
    """
    fine_x = np.asarray(fine_x, dtype=float)
    fine_y = np.asarray(fine_y, dtype=float)
    fine_z = np.asarray(fine_z, dtype=float)
    sigma_fine = np.asarray(sigma_fine)

    if sigma_fine.ndim != 3 or sigma_fine.shape != (
        len(fine_x), len(fine_y), len(fine_z)
    ):
        raise ValueError(
            f"sigma_fine shape {sigma_fine.shape} must be "
            f"({len(fine_x)}, {len(fine_y)}, {len(fine_z)})"
        )

    # Default h_svd = per-axis 10th-percentile fine-grid spacing as a 3-tuple.
    #
    # Using the absolute minimum spacing is misleading for non-uniform fine grids
    # (e.g. hybrid_axial_grid): tiny transition intervals at the inner/outer zone
    # boundary are far smaller than the actual inner-zone resolution and would
    # force an unnecessarily fine SVD grid over the entire domain.  The 10th
    # percentile picks a representative "fine" spacing that ignores such outliers
    # while still resolving the interface in the dense inner zone.
    #
    # Each axis uses its own percentile, so the tuple is alias-free per axis
    # without cross-contamination from the finest axis (e.g. dz << dx).
    if h_svd is None:
        def _p10(arr):
            d = np.diff(arr)
            return float(np.percentile(d, 10)) if len(d) > 0 else 1.0
        dx_f = _p10(fine_x) if len(fine_x) > 1 else 1.0
        dy_f = _p10(fine_y) if len(fine_y) > 1 else 1.0
        dz_f = _p10(fine_z) if len(fine_z) > 1 else 1.0
        h_svd = (dx_f, dy_f, dz_f)

    sf = make_sigma_func(fine_x, fine_y, fine_z, sigma_fine, interp="nearest")
    return from_sigma_func(
        grid, sf,
        h_svd=h_svd,
        n_line=n_line,
        n_vol=n_vol,
        mu=mu,
        eps=eps,
        method=method,
        iso_tol=iso_tol,
        svd_isotropy_tol=svd_isotropy_tol,
    )


def from_sigma_func(
    grid: "LebedevGrid3D",
    sigma_func,
    h_svd: "float | tuple[float, float, float]" = 0.025,
    n_line: int = 50,
    n_vol: int = 8,
    mu: float = MU0,
    eps: float = EPS0,
    method: str = "nodal",
    iso_tol: float = 1e-6,
    svd_isotropy_tol: float = 0.7,
) -> "EMMedia":
    """
    Build an EMMedia from a callable conductivity function, upscaled to the
    Lebedev FD grid without storing a global fine-grid array.

    This solves the fundamental tension in :func:`from_fine_grid` between SVD
    accuracy (requires isotropic fine spacing in all directions) and D_diag
    accuracy (requires many points along each axis within each dual cell):
    the callable is evaluated *on demand* at whatever resolution is needed
    for each computation, with zero global memory overhead.

    Parameters
    ----------
    grid : LebedevGrid3D
    sigma_func : callable
        ``sigma_func(X, Y, Z) -> ndarray``  where X, Y, Z are broadcastable
        arrays of the same shape and the return has the same shape.  Should
        return real positive conductivity values, or complex σ̇ with positive
        real part — the imaginary part is preserved through all averaging;
        only geometric decisions (interface-normal SVD, binarisation,
        material classification) use the real part.
    h_svd : float or (float, float, float)
        Physical spacing [m] for the sub-grid used by the SVD interface-normal
        estimator.  Can be:

        *scalar* — same spacing in all three directions.  Equal spacing in all
            three directions eliminates the anisotropy bias in
            :func:`_estimate_normal_svd` and is the safe choice when
            ``sigma_func`` has comparable resolution in all directions.
            Default 0.025 m.

        *3-tuple* ``(h_x, h_y, h_z)`` — per-axis spacing.  Ideal when
            ``sigma_func`` wraps an anisotropic fine grid (e.g. from
            :func:`make_sigma_func`): set each entry to the minimum fine-grid
            spacing along that axis so the evaluation grid is alias-free in
            every direction without unnecessary oversampling of coarse axes.
            The SVD interface-normal estimate uses physical coordinates of
            crossing-pair midpoints and remains unbiased for planar interfaces
            regardless of axis-wise sampling density differences.
    n_line : int
        Number of quadrature points along each 1-D axis line for computing
        per-axis harmonic averages of σ⁻¹ (D_diag).  Default 50.
    n_vol : int
        Per-axis points for the n_vol³ tensor-product quadrature used to
        compute the volume averages σ̄ and ⟨σ⁻¹⟩_vol.  Default 8.
    mu : float   [H/m]   Uniform permeability.
    eps : float  [F/m]   Uniform permittivity.
    method : str
        Same choices as :func:`from_fine_grid`: ``"pointwise"``,
        ``"arithmetic"``, ``"diagonal"``, ``"backus"``, ``"nodal"``.
    iso_tol : float
        Isotropy tolerance: cells whose diagonal entries agree within this
        relative tolerance are stored as scalars.
    svd_isotropy_tol : float
        SVD planarity threshold: cells whose SVD planarity ratio s₃/s₁
        (smallest / largest singular value of the centred crossing-midpoint
        positions) ≥ this value are treated as uniform (no dominant interface
        direction) and fall back to ``"diagonal"``.

    Returns
    -------
    EMMedia
        ``sigma_R`` is shape ``(N_R,)`` for isotropic results or
        ``(N_R, 3, 3)`` when any cell is anisotropic.
        ``mu_P`` and ``eps_R`` are uniform scalars.

    Notes
    -----
    For a piecewise-constant σ defined by a fast Python closure the total
    number of ``sigma_func`` evaluations is approximately
    ``N_R × (n_vol³ + 3·n_line + nx_svd·ny_svd·nz_svd)`` where the SVD
    sub-grid counts depend on the dual-cell extents relative to *h_svd*.
    For typical logging grids with N_R ≈ 14 000 and the default parameters
    this is of order 10–50 M evaluations, fast for vectorised closures.
    """
    if method not in ("pointwise", "arithmetic", "diagonal", "backus", "nodal"):
        raise ValueError(
            f"method must be one of 'pointwise', 'arithmetic', 'diagonal', "
            f"'backus', or 'nodal'; got {method!r}"
        )

    # ── Detect whether sigma_func is scalar- or tensor-valued ─────────────────
    # Probe once with a single-element array at the origin.  A scalar callable
    # returns shape () or (1,1,1); a tensor callable returns (..., 3, 3).
    _probe = np.asarray(
        sigma_func(np.array([[[0.0]]]), np.array([[[0.0]]]), np.array([[[0.0]]]))
    )
    _func_is_tensor = (_probe.ndim >= 2 and _probe.shape[-2:] == (3, 3))

    N_R = grid.N_R
    x_fd, y_fd, z_fd = grid.x, grid.y, grid.z

    # ------------------------------------------------------------------
    # Pass 1 — fill scalar σ for every R-node (volume average or pointwise)
    # This must complete before any tensor is allocated, so that the tensor
    # diagonal initialisation (which copies the full sigma_R_scalar array)
    # is not polluted by uninitialised entries.
    # ------------------------------------------------------------------
    sigma_R_scalar = np.empty(N_R, dtype=complex)

    for seq, (i, j, k) in enumerate(grid.R_nodes):
        x_node = float(x_fd[i])
        y_node = float(y_fd[j])
        z_node = float(z_fd[k])

        if method == "pointwise":
            val = sigma_func(
                np.array([[[x_node]]]),
                np.array([[[y_node]]]),
                np.array([[[z_node]]]),
            )
            val_a = np.asarray(val, dtype=complex)
            if _func_is_tensor:
                # Store trace/3 as scalar proxy; full tensor filled in Pass 1b
                sigma_R_scalar[seq] = complex(np.trace(val_a.reshape(3, 3)) / 3.0)
            else:
                sigma_R_scalar[seq] = complex(np.ravel(val_a)[0])
            continue

        x_lo = float(x_fd[max(i - 1, 0)])
        x_hi = float(x_fd[min(i + 1, len(x_fd) - 1)])
        y_lo = float(y_fd[max(j - 1, 0)])
        y_hi = float(y_fd[min(j + 1, len(y_fd) - 1)])
        z_lo = float(z_fd[max(k - 1, 0)])
        z_hi = float(z_fd[min(k + 1, len(z_fd) - 1)])

        xv = np.linspace(x_lo, x_hi, n_vol)
        yv = np.linspace(y_lo, y_hi, n_vol)
        zv = np.linspace(z_lo, z_hi, n_vol)
        XV, YV, ZV = np.meshgrid(xv, yv, zv, indexing="ij")
        raw_vol = sigma_func(XV, YV, ZV)
        if _func_is_tensor:
            raw_vol_arr = np.asarray(raw_vol, dtype=complex)
            # Use trace/3 as scalar proxy for volume-average; full tensor
            # filled later in Pass 1b (arithmetic) or Pass 2 (nodal/backus).
            # Kept complex so σ̇ = σ − iωε callables are not realified.
            sig_vol = np.trace(raw_vol_arr, axis1=-2, axis2=-1) / 3.0
        else:
            sig_vol = np.asarray(raw_vol, dtype=complex)
        sigma_R_scalar[seq] = complex(np.mean(sig_vol))

    # ------------------------------------------------------------------
    # Parse per-axis SVD spacing (scalar → uniform 3-tuple)
    # ------------------------------------------------------------------
    if isinstance(h_svd, (list, tuple, np.ndarray)) and len(h_svd) == 3:
        _hx, _hy, _hz = float(h_svd[0]), float(h_svd[1]), float(h_svd[2])
    else:
        _hx = _hy = _hz = float(h_svd)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Pass 1b — for tensor-valued callables with pointwise/arithmetic,
    # re-evaluate and store the full 3×3 tensor at each node.  Pass 1
    # only stored trace/3 in sigma_R_scalar, so we need this second pass
    # before _alloc_tensor_if_needed() copies from sigma_R_scalar.
    # ------------------------------------------------------------------
    if _func_is_tensor and method in ("pointwise", "arithmetic"):
        # Allocate tensor array seeded from trace/3 scalars
        _sigma_R_tensor_pa = np.zeros((N_R, 3, 3), dtype=complex)
        for d in range(3):
            _sigma_R_tensor_pa[:, d, d] = sigma_R_scalar
        for seq, (i, j, k) in enumerate(grid.R_nodes):
            x_node = float(x_fd[i])
            y_node = float(y_fd[j])
            z_node = float(z_fd[k])
            if method == "pointwise":
                _raw = sigma_func(
                    np.array([[[x_node]]]),
                    np.array([[[y_node]]]),
                    np.array([[[z_node]]]),
                )
                _sigma_R_tensor_pa[seq] = np.asarray(_raw, dtype=complex).reshape(3, 3)
            else:   # arithmetic
                x_lo = float(x_fd[max(i - 1, 0)])
                x_hi = float(x_fd[min(i + 1, len(x_fd) - 1)])
                y_lo = float(y_fd[max(j - 1, 0)])
                y_hi = float(y_fd[min(j + 1, len(y_fd) - 1)])
                z_lo = float(z_fd[max(k - 1, 0)])
                z_hi = float(z_fd[min(k + 1, len(z_fd) - 1)])
                xv = np.linspace(x_lo, x_hi, n_vol)
                yv = np.linspace(y_lo, y_hi, n_vol)
                zv = np.linspace(z_lo, z_hi, n_vol)
                XV, YV, ZV = np.meshgrid(xv, yv, zv, indexing="ij")
                _raw = np.asarray(sigma_func(XV, YV, ZV), dtype=complex)
                _sigma_R_tensor_pa[seq] = _raw.mean(axis=(0, 1, 2))   # (3, 3)
        mu_P  = np.full(grid.N_P, complex(mu))
        eps_R = np.full(N_R,      complex(eps))
        return EMMedia(grid, _sigma_R_tensor_pa, mu_P, eps_R)

    # ------------------------------------------------------------------
    # Pass 2 — upgrade anisotropic cells to tensor (diagonal / backus / nodal)
    # ------------------------------------------------------------------
    sigma_R_tensor: np.ndarray | None = None

    def _alloc_tensor_if_needed():
        nonlocal sigma_R_tensor
        if sigma_R_tensor is None:
            # sigma_R_scalar is fully populated at this point — safe to copy
            sigma_R_tensor = np.zeros((N_R, 3, 3), dtype=complex)
            for d in range(3):
                sigma_R_tensor[:, d, d] = sigma_R_scalar

    def _set_diagonal(seq, s_xx, s_yy, s_zz):
        s_avg = (s_xx + s_yy + s_zz) / 3.0
        aniso = abs(s_xx - s_avg) + abs(s_yy - s_avg) + abs(s_zz - s_avg)
        if aniso < iso_tol * abs(s_avg) + 1e-300:
            return False
        # SPD guard: per-axis effective conductivities must have positive
        # real part; otherwise keep the existing (pointwise) value.
        if min(complex(s_xx).real, complex(s_yy).real, complex(s_zz).real) <= 0.0:
            return False
        _alloc_tensor_if_needed()
        # Replace the WHOLE tensor, zeroing off-diagonals.  For tensor-valued
        # callables the array is pre-seeded with pointwise tensors; mixing a
        # scalar-proxy diagonal with the pointwise off-diagonals can produce
        # an indefinite matrix (e.g. large σ_xy with reduced σ_xx, σ_yy).
        # A positive diagonal tensor is SPD by construction.
        sigma_R_tensor[seq] = np.diag(
            np.array([s_xx, s_yy, s_zz], dtype=complex)
        )
        return True

    if method in ("diagonal", "backus", "nodal"):
        # For tensor-valued callables, initialise every cell with its pointwise
        # tensor so that uniform anisotropic cells (where no interface is
        # detected by the SVD loop) get the correct full 3×3 tensor rather than
        # the trace/3 scalar diagonal.  Interface cells will be overwritten below.
        if _func_is_tensor:
            _X = np.array([float(x_fd[i]) for i, j, k in grid.R_nodes],
                          dtype=float).reshape(N_R, 1, 1)
            _Y = np.array([float(y_fd[j]) for i, j, k in grid.R_nodes],
                          dtype=float).reshape(N_R, 1, 1)
            _Z = np.array([float(z_fd[k]) for i, j, k in grid.R_nodes],
                          dtype=float).reshape(N_R, 1, 1)
            _raw_pw = np.asarray(sigma_func(_X, _Y, _Z), dtype=complex)
            sigma_R_tensor = _raw_pw.reshape(N_R, 3, 3)

        for seq, (i, j, k) in enumerate(grid.R_nodes):
            x_node = float(x_fd[i])
            y_node = float(y_fd[j])
            z_node = float(z_fd[k])

            x_lo = float(x_fd[max(i - 1, 0)])
            x_hi = float(x_fd[min(i + 1, len(x_fd) - 1)])
            y_lo = float(y_fd[max(j - 1, 0)])
            y_hi = float(y_fd[min(j + 1, len(y_fd) - 1)])
            z_lo = float(z_fd[max(k - 1, 0)])
            z_hi = float(z_fd[min(k + 1, len(z_fd) - 1)])

            # ---- backus / nodal: build isotropic SVD sub-grid first ----------
            # For the "diagonal" method only, use the coarser n_vol³ grid.
            # For backus/nodal, the isotropic SVD block (h_svd in all directions)
            # gives far more accurate volume fractions for thin-sliver interface
            # cells than a coarse n_vol³ grid — e.g. a 2.4 m × 0.125 m cell with
            # a 3.9%-sliver in layer 1 would be represented as ~98×6 points at
            # h_svd=0.025 vs 8³ at n_vol=8, with 3-4 vs. 0-1 points in layer 1.

            if method == "diagonal":
                # Coarse n_vol³ for diagonal — no SVD needed
                xv = np.linspace(x_lo, x_hi, n_vol)
                yv = np.linspace(y_lo, y_hi, n_vol)
                zv = np.linspace(z_lo, z_hi, n_vol)
                XV, YV, ZV = np.meshgrid(xv, yv, zv, indexing="ij")
                # Complex-preserving: averages act on the full σ̇ values.
                sig_vol = np.asarray(sigma_func(XV, YV, ZV), dtype=complex)
                inv_vol = 1.0 / sig_vol
                s_xx = complex(np.mean(1.0 / np.mean(inv_vol, axis=0)))
                s_yy = complex(np.mean(1.0 / np.mean(inv_vol, axis=1)))
                s_zz = complex(np.mean(1.0 / np.mean(inv_vol, axis=2)))
                _set_diagonal(seq, s_xx, s_yy, s_zz)
                continue

            # backus / nodal — build per-axis SVD sub-grid.
            # Nominal per-axis counts from the requested h_svd spacings.
            # Per-cell cap: if the nominal total exceeds ~50 000 points, scale
            # all three axes uniformly (preserving their relative density ratio)
            # so large outer-zone cells don't cause OOM.
            _MAX_SVD_PTS = 50_000
            _nx_nom = max(3, int(np.ceil((x_hi - x_lo) / _hx)) + 1)
            _ny_nom = max(3, int(np.ceil((y_hi - y_lo) / _hy)) + 1)
            _nz_nom = max(3, int(np.ceil((z_hi - z_lo) / _hz)) + 1)
            _n_nom  = _nx_nom * _ny_nom * _nz_nom
            if _n_nom > _MAX_SVD_PTS:
                _scale = (_n_nom / _MAX_SVD_PTS) ** (1.0 / 3.0)
                nx_s = max(3, int(np.ceil(_nx_nom / _scale)))
                ny_s = max(3, int(np.ceil(_ny_nom / _scale)))
                nz_s = max(3, int(np.ceil(_nz_nom / _scale)))
            else:
                nx_s, ny_s, nz_s = _nx_nom, _ny_nom, _nz_nom
            x_svd = np.linspace(x_lo, x_hi, nx_s)   # isotropic at h_svd_cell
            y_svd = np.linspace(y_lo, y_hi, ny_s)
            z_svd = np.linspace(z_lo, z_hi, nz_s)
            XS, YS, ZS = np.meshgrid(x_svd, y_svd, z_svd, indexing="ij")
            _raw_svd = sigma_func(XS, YS, ZS)
            if _func_is_tensor:
                _block_tensor_svd = np.asarray(_raw_svd, dtype=complex)
                _block_c = np.trace(_block_tensor_svd, axis1=-2, axis2=-1) / 3.0
                # Real scalar proxy for geometric decisions only (SVD normal
                # estimation, binarisation, material classification).
                block_svd = np.real(_block_c).astype(float)
            else:
                _block_tensor_svd = None
                _block_c = np.asarray(_raw_svd, dtype=complex)
                block_svd = np.real(_block_c).astype(float)

            # Volume averages from the fine isotropic SVD grid — more accurate
            # for thin-sliver cells than the coarse n_vol³ quadrature.
            # Complex-preserving: computed from _block_c, not the real proxy.
            sigma_arith   = complex(np.mean(_block_c))
            inv_sigma_vol = complex(np.mean(1.0 / _block_c))

            # Diagonal per-axis effective conductivities from SVD block
            inv_svd = 1.0 / _block_c
            s_xx = complex(np.mean(1.0 / np.mean(inv_svd, axis=0)))
            s_yy = complex(np.mean(1.0 / np.mean(inv_svd, axis=1)))
            s_zz = complex(np.mean(1.0 / np.mean(inv_svd, axis=2)))

            n_hat, svd_ratio = _estimate_normal_svd(block_svd, x_svd, y_svd, z_svd)

            # ── Multi-material cell handling (n_distinct ≥ 3, nodal only) ────────
            # For cells that straddle more than one interface we use masked
            # per-pair sub-SVDs to identify each interface normal independently,
            # then apply either multiregion nodal homogenization (all interfaces
            # parallel) or sequential Backus (two distinct normals).
            if method == "nodal":
                _svals_blk = np.sort(np.unique(np.round(block_svd, 6)))
                if len(_svals_blk) >= 3:
                    _t_mm = _nodal_multimat_3d(
                        block_svd, x_svd, y_svd, z_svd, _svals_blk,
                        _block_tensor_svd, _func_is_tensor, svd_isotropy_tol,
                        n_hat if (n_hat is not None and svd_ratio < svd_isotropy_tol)
                             else None,
                        node_xyz=(x_node, y_node, z_node),
                        block_complex=None if _func_is_tensor else _block_c,
                    )
                    if _t_mm is not None:
                        _alloc_tensor_if_needed()
                        sigma_R_tensor[seq] = _t_mm
                        continue
                    # Fall through to standard single-interface path

            if n_hat is None or svd_ratio >= svd_isotropy_tol:
                _set_diagonal(seq, s_xx, s_yy, s_zz)
                continue

            if method == "backus":
                t = _standard_backus_tensor_3d(sigma_arith, inv_sigma_vol, n_hat)
            else:   # "nodal"
                if _func_is_tensor and _block_tensor_svd is not None:
                    # ── Tensor-valued callable ────────────────────────────────
                    # Binarise the scalar (trace/3) block to identify the two
                    # regions, then extract their mean tensors and fractions.
                    _sig_mid = 0.5 * (block_svd.max() + block_svd.min())
                    _binary  = (block_svd >= _sig_mid)   # True = higher-σ region
                    _mask1, _mask2 = ~_binary, _binary   # region 1 = lower σ
                    if _mask1.any() and _mask2.any():
                        _s1 = np.mean(_block_tensor_svd[_mask1], axis=0)  # (3,3)
                        _s2 = np.mean(_block_tensor_svd[_mask2], axis=0)  # (3,3)
                    else:
                        # Uniform block — fall through to diagonal below
                        _set_diagonal(seq, s_xx, s_yy, s_zz)
                        continue
                    # Volume fraction and per-axis line fractions of region 1.
                    # Line fractions along the lines through the NODE (tex
                    # eqs. A.8-A.10): the transverse coordinates are held at
                    # their node values — the box centre differs from the node
                    # on non-uniform grids.  Evaluate the callable directly on
                    # n_line points per axis (same resolution as the scalar
                    # path) rather than using the coarser SVD-block voxel
                    # lines.
                    _f_vol = float(_mask1.mean())
                    _xl = np.linspace(x_lo, x_hi, n_line)
                    _yl = np.linspace(y_lo, y_hi, n_line)
                    _zl = np.linspace(z_lo, z_hi, n_line)

                    def _line_frac1(lx, ly, lz):
                        _tr = np.real(np.trace(
                            np.asarray(sigma_func(lx, ly, lz), dtype=complex),
                            axis1=-2, axis2=-1)) / 3.0
                        return float(np.mean(_tr < _sig_mid))

                    _f_line = np.array([
                        _line_frac1(_xl, np.full(n_line, y_node),
                                    np.full(n_line, z_node)),   # x-axis line
                        _line_frac1(np.full(n_line, x_node), _yl,
                                    np.full(n_line, z_node)),   # y-axis line
                        _line_frac1(np.full(n_line, x_node),
                                    np.full(n_line, y_node), _zl),  # z-axis line
                    ])
                    t = _nodal_eff_tensor_general(
                        _s1, _s2, _f_vol, _f_line, n_hat)
                else:
                    # ── Scalar-valued callable: original path ─────────────────
                    xl = np.linspace(x_lo, x_hi, n_line)
                    yl = np.linspace(y_lo, y_hi, n_line)
                    zl = np.linspace(z_lo, z_hi, n_line)

                    # Complex-preserving line averages of 1/σ̇ through the node.
                    sig_x = np.asarray(sigma_func(
                        xl,
                        np.full(n_line, y_node),
                        np.full(n_line, z_node),
                    ), dtype=complex)
                    sig_y = np.asarray(sigma_func(
                        np.full(n_line, x_node),
                        yl,
                        np.full(n_line, z_node),
                    ), dtype=complex)
                    sig_z = np.asarray(sigma_func(
                        np.full(n_line, x_node),
                        np.full(n_line, y_node),
                        zl,
                    ), dtype=complex)

                    D_diag = np.array([
                        complex(np.mean(1.0 / sig_x)),
                        complex(np.mean(1.0 / sig_y)),
                        complex(np.mean(1.0 / sig_z)),
                    ], dtype=complex)

                    t = _nodal_eff_tensor_3d(D_diag, sigma_arith, inv_sigma_vol, n_hat)

            _alloc_tensor_if_needed()
            sigma_R_tensor[seq] = t

    sigma_R_out = sigma_R_tensor if sigma_R_tensor is not None else sigma_R_scalar

    mu_P  = np.full(grid.N_P, complex(mu))
    eps_R = np.full(N_R,      complex(eps))

    return EMMedia(grid, sigma_R_out, mu_P, eps_R)


# ---------------------------------------------------------------------------
# from_geometry_func — exact-normal path
# ---------------------------------------------------------------------------

def _nodal_from_normals(
    block: np.ndarray,
    x_sub: np.ndarray,
    y_sub: np.ndarray,
    z_sub: np.ndarray,
    normals: list,
    block_tensor,
    is_tensor: bool,
    node_xyz=None,
    block_complex: "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Nodal effective tensor for a multi-interface cell given analytically known
    interface normals (no SVD needed).

    *node_xyz* (optional) gives the physical R-node coordinates so that
    per-axis line fractions are taken along the sub-grid lines through the
    node (tex note eqs. A.8-A.10) instead of the box-centre lines — these
    differ on non-uniform grids.  *block_complex* (optional) carries complex
    σ̇ values matching *block* (real part, used for classification); material
    conductivities are read from it to preserve the imaginary part.

    Parameters
    ----------
    block        : (nx, ny, nz) scalar σ array (or trace/3 proxy for tensors).
    x_sub, y_sub, z_sub : 1-D coordinate arrays for the block axes.
    normals      : list of unit normal vectors, **innermost first**.
                   len == 1  → single interface, standard nodal formula.
                   len >= 2  → sequential Backus: normals[0] = innermost
                               boundary normal, normals[-1] = outermost.
    block_tensor : (nx,ny,nz,3,3) full tensor block or None (scalar callable).
    is_tensor    : True if the callable returned full tensors.

    Returns
    -------
    (3,3) complex effective tensor.

    Algorithm (3 sigma values, 2 normals)
    ---------------------------------------
    Mirrors :func:`_nodal_multimat_3d`: the middle sigma value (svals[1] in
    sorted order) is always treated as the **inner sandwiched material**.  This
    avoids the ambiguity of projection-based identification for curved boundaries
    (e.g. cylindrical) and matches the physical nesting for DDH03-type geometries
    (bore < invasion < iso/aniso).

    Step 1 — homogenise the two outer materials {svals[0], svals[2]} using
             normals[-1] (the outermost boundary normal, e.g. dip plane).
    Step 2 — homogenise svals[1] against the step-1 result using normals[0]
             (the innermost boundary normal, e.g. invasion wall).

    If step 1 produces an effective tensor whose maximum eigenvalue exceeds 3×
    the maximum conductivity of the outer-pair materials (indicating numerical
    degeneracy — typically because the dominant outer material has very low
    normal conductivity and dominates G_nn), the sequential-Backus result is
    discarded and the fallback 2-material path is used instead.
    """
    # Line indices through the NODE (not the box centre) — see _node_line_indices.
    if node_xyz is not None:
        ic, jc, kc = _node_line_indices(x_sub, y_sub, z_sub, node_xyz)
    else:
        nb_x, nb_y, nb_z = block.shape
        ic, jc, kc = nb_x // 2, nb_y // 2, nb_z // 2
    block_r = np.round(block, 6)
    svals = np.sort(np.unique(block_r))

    def _sig_region(mask):
        """Mean sigma tensor (or scalar*I) over the voxels in mask."""
        if not mask.any():
            return complex(block_r.flat[0]) * np.eye(3, dtype=complex)
        if is_tensor and block_tensor is not None:
            return np.mean(block_tensor[mask], axis=0)
        if block_complex is not None:
            return complex(np.mean(block_complex[mask])) * np.eye(3, dtype=complex)
        return complex(np.mean(block_r[mask])) * np.eye(3, dtype=complex)

    def _frac_of_mask_in_region(region_mask, total_mask):
        """Volume and center-line fracs of region_mask within total_mask."""
        fv = float(region_mask.mean())
        lx = float(region_mask[:, jc, kc].sum()) / max(float(total_mask[:, jc, kc].sum()), 1.0)
        ly = float(region_mask[ic, :, kc].sum()) / max(float(total_mask[ic, :, kc].sum()), 1.0)
        lz = float(region_mask[ic, jc, :].sum()) / max(float(total_mask[ic, jc, :].sum()), 1.0)
        return fv, np.array([lx, ly, lz])

    # ── Uniform block ──────────────────────────────────────────────────────────
    if len(svals) < 2:
        return _sig_region(np.ones(block.shape, dtype=bool))

    # ── Two sigma values: pick the supplied normal that best separates them ────
    # This handles both the single-normal case AND multi-normal cells where the
    # sub-block only resolves 2 materials (e.g. aniso↔iso with no inv visible).
    # For each candidate normal, score how cleanly it separates the two sigma
    # values: highest mean-projection separation wins.
    if len(svals) == 2 or len(normals) == 1:
        sig_mid = 0.5 * (svals[0] + svals[-1])
        mask1 = block_r <= sig_mid      # lower-σ region
        mask2 = ~mask1                  # higher-σ region
        if not mask1.any() or not mask2.any():
            return _sig_region(np.ones(block.shape, dtype=bool))
        s1 = _sig_region(mask1)
        s2 = _sig_region(mask2)
        fv, fl = _frac_of_mask_in_region(mask1, np.ones(block.shape, dtype=bool))
        # Choose the normal that maximises mean-projection separation
        XX2, YY2, ZZ2 = np.meshgrid(x_sub, y_sub, z_sub, indexing="ij")
        if node_xyz is not None:
            node_pos2 = np.asarray(node_xyz, dtype=float)
        else:
            node_pos2 = np.array([XX2[ic, jc, kc], YY2[ic, jc, kc], ZZ2[ic, jc, kc]])
        best_n = np.asarray(normals[0], dtype=float)
        best_n /= max(np.linalg.norm(best_n), 1e-14)
        best_score = -1.0
        for n_cand_raw in normals:
            n_c = np.asarray(n_cand_raw, dtype=float)
            n_c /= max(np.linalg.norm(n_c), 1e-14)
            proj = (n_c[0] * (XX2 - node_pos2[0])
                  + n_c[1] * (YY2 - node_pos2[1])
                  + n_c[2] * (ZZ2 - node_pos2[2]))
            mean1 = float(proj[mask1].mean()) if mask1.any() else 0.0
            mean2 = float(proj[mask2].mean()) if mask2.any() else 0.0
            score = abs(mean2 - mean1)
            if score > best_score:
                best_score = score
                best_n = n_c
        return _nodal_eff_tensor_general(s1, s2, fv, fl, best_n)

    # ── Multiple interfaces: sequential Backus ─────────────────────────────────
    # Convention: normals[0] = innermost boundary, normals[-1] = outermost.
    #
    # Key insight: the sigma VALUES in the block already correctly reflect which
    # voxels belong to which material region (they come from sigma_func evaluated
    # on a fine sub-grid).  Using projection from the node to split voxels fails
    # for curved boundaries (e.g. cylinders).  Instead we use sigma-value masks,
    # exactly as _nodal_multimat_3d does, but with user-supplied normals.
    #
    # For exactly 3 distinct σ values and 2 boundaries (the most common case):
    #   • normals[0] (innermost boundary, e.g. invasion cylinder) separates
    #     the "inner" material from the two outer materials.
    #   • normals[-1] (outer boundary, e.g. dipping plane) separates the two
    #     outer materials from each other.
    #   We determine which sigma value is the inner material by comparing mean
    #   projections of each material's voxels onto normals[0]: the material
    #   whose voxels project most negatively (i.e. "inside" the boundary) is
    #   the inner one.
    #
    # Sequential Backus (mirrors _nodal_multimat_3d):
    #   Step 1 — homogenise the two outer materials with n_outer (normals[-1])
    #   Step 2 — homogenise the inner material against σ_outer with n_inner

    XX, YY, ZZ = np.meshgrid(x_sub, y_sub, z_sub, indexing="ij")
    if node_xyz is not None:
        node_pos = np.asarray(node_xyz, dtype=float)
    else:
        node_pos = np.array([XX[ic, jc, kc], YY[ic, jc, kc], ZZ[ic, jc, kc]])

    def _mask_of(sv):
        return np.abs(block_r - sv) < 1e-5

    def _sig_of(sv):
        m = _mask_of(sv)
        if is_tensor and block_tensor is not None and m.any():
            return np.mean(block_tensor[m], axis=0)
        if block_complex is not None and m.any():
            return complex(np.mean(block_complex[m])) * np.eye(3, dtype=complex)
        return complex(sv) * np.eye(3, dtype=complex)

    def _vol_lf(m):
        """Volume frac and per-axis line fracs of mask m in the full block."""
        fv = float(m.mean())
        fl = np.array([float(m[:, jc, kc].mean()),
                       float(m[ic, :, kc].mean()),
                       float(m[ic, jc, :].mean())])
        return fv, fl

    # ── 3 sigma values, 2 normals (the standard nested-boundary case) ─────────
    # Mirror _nodal_multimat_3d exactly: always treat svals[1] (the middle
    # sigma value in sorted order) as the inner sandwiched material.  The two
    # outer materials are svals[0] and svals[2].
    #
    # This convention is correct whenever the material ordering by sigma value
    # reflects the spatial nesting (e.g. bore ≪ invasion < iso for DDH03), which
    # is the case by construction for nested cylindrical + planar geometries.
    #
    # We do NOT apply a minimum-fraction guard here.  Previously a _MIN_FRAC=0.03
    # gate was used to skip sequential Backus for thin slivers.  That guard
    # caused cells where svals[0] (e.g. aniso below dip plane) has < 3% fraction
    # to fall back to a 2-material treatment that lumps invasion+aniso as one
    # composite against iso — physically wrong and giving a ~15 mm crossing error.
    # With the middle-sigma convention, thin-sliver materials land in the outer
    # pair and do not cause line-fraction degeneracies in step 2.
    if len(svals) == 3 and len(normals) >= 2:
        n_inner = np.asarray(normals[0], dtype=float)
        n_inner /= np.linalg.norm(n_inner)
        n_outer = np.asarray(normals[-1], dtype=float)
        n_outer /= np.linalg.norm(n_outer)

        # Inner = middle sigma value (matches _nodal_multimat_3d convention).
        sv_inner = svals[1]
        sv_out0  = svals[0]
        sv_out1  = svals[2]

        m0  = _mask_of(sv_out0)
        m1  = _mask_of(sv_inner)
        m2  = _mask_of(sv_out1)
        m_out = m0 | m2

        # Step 1: average the two outer materials with n_outer
        f_outer_vol = float(m_out.mean())
        if f_outer_vol > 1e-12:
            f0_in_out = float(m0.mean()) / f_outer_vol
            lf0x = float(m0[:, jc, kc].sum()) / max(float(m_out[:, jc, kc].sum()), 1.0)
            lf0y = float(m0[ic, :, kc].sum()) / max(float(m_out[ic, :, kc].sum()), 1.0)
            lf0z = float(m0[ic, jc, :].sum()) / max(float(m_out[ic, jc, :].sum()), 1.0)
            sigma_outer = _nodal_eff_tensor_general(
                _sig_of(sv_out0), _sig_of(sv_out1),
                f0_in_out, np.array([lf0x, lf0y, lf0z]), n_outer)
        else:
            sigma_outer = _sig_of(sv_out1)

        # Sanity check: if sigma_outer has eigenvalues far above the physical
        # maximum (max conductivity of outer pair materials), the outer-pair
        # homogenization is degenerate — typically because one outer material
        # has very low normal conductivity and dominates the volume fraction,
        # making G_nn ≫ 1.  In that case fall through to the 2-material path.
        _sig_out_max = max(float(np.real(np.trace(_sig_of(sv_out0)))) / 3.0,
                           float(np.real(np.trace(_sig_of(sv_out1)))) / 3.0)
        try:
            _eigmax_outer = float(np.max(np.real(np.linalg.eigvalsh(sigma_outer))))
        except Exception:
            _eigmax_outer = float('inf')
        _OUTER_OVERFLOW_TOL = 3.0   # outer eigenvalue > 3× physical max → degenerate
        if _eigmax_outer > _OUTER_OVERFLOW_TOL * max(_sig_out_max, 1e-30):
            pass   # fall through to 2-material fallback below
        else:
            # Step 2: combine inner material with sigma_outer across n_inner
            f1v, f1l = _vol_lf(m1)
            result = _nodal_eff_tensor_general(
                _sig_of(sv_inner), sigma_outer, f1v, f1l, n_inner)

            # Eigenvalue-overflow fallback on the FINAL sequential result
            # (tex note §"Fallback"): a valid homogenized tensor satisfies
            # λ_max(ΣD) ≤ max_i λ_max(σ_i) ≤ 3 × max_i tr(σ_i)/3.  Beyond
            # that the sequential combination is numerically degenerate —
            # use the single-interface 2-material fallback below instead.
            _sig_all_max = max(
                float(np.real(np.trace(_sig_of(sv)))) / 3.0
                for sv in (sv_out0, sv_inner, sv_out1))
            try:
                _emax_fin = float(np.max(np.linalg.eigvalsh(
                    np.real(0.5 * (result + result.T)))))
            except Exception:
                _emax_fin = float("inf")
            if (np.all(np.isfinite(result))
                    and _emax_fin <= _OUTER_OVERFLOW_TOL * max(_sig_all_max, 1e-30)):
                return result
            # else: fall through to the 2-material fallback below

    # ── Fallback: 2-material treatment with best-matching normal ─────────────
    # Reached when:
    #   • len(svals) != 3 or len(normals) < 2 (so sequential Backus not applicable)
    #   • Or only 1 normal was supplied (handled above in the 2-svals/1-normal block)
    #   • Or len(svals) > 3 (four or more materials: binary-split degrades gracefully).
    #
    # Strategy: binary-split the block at the midpoint and pick the supplied
    # normal that best separates the two resulting regions (largest mean
    # projection difference).  For N>3 materials, this degrades gracefully by
    # treating the block as a simple two-region composite.
    sig_mid = 0.5 * (block.min() + block.max())
    mask1 = block_r <= sig_mid   # lower-σ half
    mask2 = ~mask1
    if not mask1.any() or not mask2.any():
        return _sig_region(np.ones(block.shape, dtype=bool))

    s1 = _sig_region(mask1)
    s2 = _sig_region(mask2)
    fv, fl = _frac_of_mask_in_region(mask1, np.ones(block.shape, dtype=bool))

    # Pick the normal with the best mean-projection separation
    best_n   = np.asarray(normals[0], dtype=float)
    best_n  /= max(np.linalg.norm(best_n), 1e-14)
    best_score = -1.0
    for n_cand_raw in normals:
        n_c = np.asarray(n_cand_raw, dtype=float)
        n_c /= max(np.linalg.norm(n_c), 1e-14)
        proj = (n_c[0] * (XX - node_pos[0])
              + n_c[1] * (YY - node_pos[1])
              + n_c[2] * (ZZ - node_pos[2]))
        mean1 = float(proj[mask1].mean()) if mask1.any() else 0.0
        mean2 = float(proj[mask2].mean()) if mask2.any() else 0.0
        score = abs(mean2 - mean1)
        if score > best_score:
            best_score = score
            best_n = n_c

    return _nodal_eff_tensor_general(s1, s2, fv, fl, best_n)


def from_geometry_func(
    grid: "LebedevGrid3D",
    sigma_func,
    interface_func,
    h_svd: "float | tuple[float, float, float]" = 0.025,
    mu: float = MU0,
    eps: float = EPS0,
    iso_tol: float = 1e-6,
    svd_fallback: bool = True,
    svd_isotropy_tol: float = 0.7,
) -> "EMMedia":
    """
    Build an :class:`EMMedia` using analytically known interface geometry.

    This is the "perfect geometry" companion to :func:`from_sigma_func`.
    The user supplies:

    * **sigma_func** — same black-box conductivity callable as in
      :func:`from_sigma_func` (scalar or tensor-valued).  Used to evaluate
      material tensors and to compute volume / line fractions numerically.
    * **interface_func** — a callable that returns the interface geometry for
      each dual cell, bypassing the SVD normal estimator entirely.

    For cells where *interface_func* returns ``None`` (uniform cell) or when
    ``svd_fallback=True`` and the returned normal is unusable, the method
    falls back to the pointwise sigma value.

    Parameters
    ----------
    grid : LebedevGrid3D
    sigma_func : callable
        ``sigma_func(X, Y, Z) -> array``  — same signature as in
        :func:`from_sigma_func`.  Scalar or tensor-valued.
    interface_func : callable
        ``interface_func(bmin, bmax, node) -> result`` where

        * ``bmin``, ``bmax`` are (3,) arrays giving the dual-cell corners,
        * ``node`` is the (3,) R-node coordinate,

        and *result* is one of:

        ``None``
            Uniform cell — use pointwise σ, no averaging.
        ``ndarray (3,)``
            Single interface; this is its unit normal.
        ``list of ndarray (3,)``
            Multiple interfaces, **innermost first**.  Sequential Backus is
            applied from outermost inward.  The number of normals should equal
            the number of distinct material boundaries the cell straddles.

        The easiest way to build *interface_func* is via
        :class:`~lebedev_em.geometry.GeometryStack`.
    h_svd : float or (hx, hy, hz)
        Physical spacing [m] for the sub-grid used to evaluate sigma_func
        and compute volume / line fractions.  Default 0.025 m.
    mu : float   [H/m]   Uniform permeability.
    eps : float  [F/m]   Uniform permittivity.
    iso_tol : float
        Relative tolerance for detecting isotropic diagonal cells.
    svd_fallback : bool
        If True (default), cells for which *interface_func* returns a normal
        but the nodal homogenization is numerically degenerate fall back to
        the pointwise sigma.  Set to False to raise an error instead.
    svd_isotropy_tol : float
        Planarity threshold used only when *interface_func* returns ``None``
        for a non-uniform cell — in that case SVD is attempted as a fallback
        if ``svd_fallback=True``.

    Returns
    -------
    EMMedia

    Examples
    --------
    DDH03 exact-geometry run using :class:`~lebedev_em.geometry.GeometryStack`::

        from lebedev_em.geometry import (CylindricalBoundary, PlanarBoundary,
                                         GeometryStack)
        import numpy as np

        N_HAT = np.array([np.sin(np.radians(60)), 0., np.cos(np.radians(60))])
        geo = GeometryStack([
            CylindricalBoundary(radius=0.1),
            CylindricalBoundary(radius=0.6),
            PlanarBoundary(n_hat=N_HAT, d=-0.25),
        ])
        med = from_geometry_func(grid, sigma_func_ddh03, geo.interface_func)

    Custom interface function (identical result, more control)::

        def my_interface_func(bmin, bmax, node):
            r = np.sqrt(node[0]**2 + node[1]**2)
            n_r = np.array([node[0]/r, node[1]/r, 0.]) if r > 1e-12 else np.array([1.,0.,0.])
            r_corners = [(cx**2+cy**2)**0.5 for cx in [bmin[0],bmax[0]] for cy in [bmin[1],bmax[1]]]
            p_corners = [N_HAT@[cx,cy,cz] for cx in [bmin[0],bmax[0]]
                                           for cy in [bmin[1],bmax[1]]
                                           for cz in [bmin[2],bmax[2]]]
            norms = []
            if min(r_corners) < R_BORE <= max(r_corners): norms.append(n_r)
            if min(r_corners) < R_INV  <= max(r_corners): norms.append(n_r)
            if min(p_corners) < D_PLANE <= max(p_corners): norms.append(N_HAT)
            if not norms: return None
            return norms[0] if len(norms) == 1 else norms

        med = from_geometry_func(grid, sigma_func_ddh03, my_interface_func)
    """
    # ── Probe sigma_func ──────────────────────────────────────────────────────
    _probe = np.asarray(
        sigma_func(np.array([[[0.0]]]), np.array([[[0.0]]]), np.array([[[0.0]]]))
    )
    _func_is_tensor = (_probe.ndim >= 2 and _probe.shape[-2:] == (3, 3))

    N_R = grid.N_R
    x_fd, y_fd, z_fd = grid.x, grid.y, grid.z

    # ── Parse h_svd ───────────────────────────────────────────────────────────
    if isinstance(h_svd, (list, tuple, np.ndarray)) and len(h_svd) == 3:
        _hx, _hy, _hz = float(h_svd[0]), float(h_svd[1]), float(h_svd[2])
    else:
        _hx = _hy = _hz = float(h_svd)  # type: ignore[arg-type]

    _MAX_SVD_PTS = 50_000

    # ── For tensor callables: initialise sigma_R_tensor from pointwise values ─
    sigma_R_tensor: np.ndarray | None = None

    if _func_is_tensor:
        _X = np.array([float(x_fd[i]) for i, j, k in grid.R_nodes],
                      dtype=float).reshape(N_R, 1, 1)
        _Y = np.array([float(y_fd[j]) for i, j, k in grid.R_nodes],
                      dtype=float).reshape(N_R, 1, 1)
        _Z = np.array([float(z_fd[k]) for i, j, k in grid.R_nodes],
                      dtype=float).reshape(N_R, 1, 1)
        sigma_R_tensor = np.asarray(
            sigma_func(_X, _Y, _Z), dtype=complex
        ).reshape(N_R, 3, 3)

    # ── Pass 1 — pointwise σ for EVERY R-node ─────────────────────────────────
    # This must complete before any tensor allocation: _alloc_tensor_if_needed
    # seeds the tensor diagonal by copying sigma_R_scalar, so a mid-loop
    # allocation over a partially-filled array would leave uninitialised
    # (garbage/zero) diagonals for all not-yet-visited uniform nodes.
    # (Same two-pass discipline as from_sigma_func.)
    sigma_R_scalar = np.empty(N_R, dtype=complex)
    if _func_is_tensor:
        # sigma_R_tensor already holds pointwise tensors for all nodes.
        sigma_R_scalar[:] = np.trace(sigma_R_tensor, axis1=-2, axis2=-1) / 3.0
    else:
        _Xn = np.array([float(x_fd[i]) for i, j, k in grid.R_nodes], dtype=float)
        _Yn = np.array([float(y_fd[j]) for i, j, k in grid.R_nodes], dtype=float)
        _Zn = np.array([float(z_fd[k]) for i, j, k in grid.R_nodes], dtype=float)
        sigma_R_scalar[:] = np.asarray(
            sigma_func(_Xn, _Yn, _Zn), dtype=complex
        ).reshape(N_R)

    def _alloc_tensor_if_needed():
        nonlocal sigma_R_tensor
        if sigma_R_tensor is None:
            # sigma_R_scalar is fully populated (Pass 1) — safe to copy.
            sigma_R_tensor = np.zeros((N_R, 3, 3), dtype=complex)
            for d in range(3):
                sigma_R_tensor[:, d, d] = sigma_R_scalar

    # ── Pass 2 — interface averaging loop ─────────────────────────────────────
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        x_node = float(x_fd[i])
        y_node = float(y_fd[j])
        z_node = float(z_fd[k])
        node   = np.array([x_node, y_node, z_node])

        x_lo = float(x_fd[max(i - 1, 0)])
        x_hi = float(x_fd[min(i + 1, len(x_fd) - 1)])
        y_lo = float(y_fd[max(j - 1, 0)])
        y_hi = float(y_fd[min(j + 1, len(y_fd) - 1)])
        z_lo = float(z_fd[max(k - 1, 0)])
        z_hi = float(z_fd[min(k + 1, len(z_fd) - 1)])
        bmin  = np.array([x_lo, y_lo, z_lo])
        bmax  = np.array([x_hi, y_hi, z_hi])

        # ── Query interface geometry ──────────────────────────────────────────
        geo_result = interface_func(bmin, bmax, node)

        if geo_result is None:
            # Uniform cell — pointwise sigma already stored; nothing to do.
            continue

        # Normalise: single normal → list of one
        if isinstance(geo_result, np.ndarray) and geo_result.ndim == 1:
            normals_list = [geo_result]
        else:
            normals_list = [np.asarray(n, dtype=float) for n in geo_result]

        # Normalise each vector
        normals_list = [
            (n / np.linalg.norm(n)) if np.linalg.norm(n) > 1e-14 else n
            for n in normals_list
        ]

        # ── Build SVD sub-block for vol / line fracs ──────────────────────────
        _nx_nom = max(3, int(np.ceil((x_hi - x_lo) / _hx)) + 1)
        _ny_nom = max(3, int(np.ceil((y_hi - y_lo) / _hy)) + 1)
        _nz_nom = max(3, int(np.ceil((z_hi - z_lo) / _hz)) + 1)
        _n_nom  = _nx_nom * _ny_nom * _nz_nom
        if _n_nom > _MAX_SVD_PTS:
            _scale = (_n_nom / _MAX_SVD_PTS) ** (1.0 / 3.0)
            nx_s = max(3, int(np.ceil(_nx_nom / _scale)))
            ny_s = max(3, int(np.ceil(_ny_nom / _scale)))
            nz_s = max(3, int(np.ceil(_nz_nom / _scale)))
        else:
            nx_s, ny_s, nz_s = _nx_nom, _ny_nom, _nz_nom

        x_svd = np.linspace(x_lo, x_hi, nx_s)
        y_svd = np.linspace(y_lo, y_hi, ny_s)
        z_svd = np.linspace(z_lo, z_hi, nz_s)
        XS, YS, ZS = np.meshgrid(x_svd, y_svd, z_svd, indexing="ij")
        _raw = sigma_func(XS, YS, ZS)

        if _func_is_tensor:
            _block_tensor = np.asarray(_raw, dtype=complex)
            _block_c = None
            block_svd = np.real(
                np.trace(_block_tensor, axis1=-2, axis2=-1) / 3.0
            ).astype(float)
        else:
            _block_tensor = None
            # Complex block for averaging; real proxy for classification.
            _block_c  = np.asarray(_raw, dtype=complex)
            block_svd = np.real(_block_c).astype(float)

        # Check if block is actually uniform (interface_func may have been
        # over-eager — e.g. corner-only straddle with no real crossing).
        if block_svd.max() - block_svd.min() < 1e-14 * (abs(block_svd.min()) + 1.0):
            continue  # truly uniform — pointwise is correct

        # ── Compute effective tensor via nodal homogenization ─────────────────
        try:
            t = _nodal_from_normals(
                block_svd, x_svd, y_svd, z_svd,
                normals_list, _block_tensor, _func_is_tensor,
                node_xyz=(x_node, y_node, z_node),
                block_complex=_block_c,
            )
        except Exception:
            if svd_fallback:
                continue  # leave pointwise value
            raise

        if not np.all(np.isfinite(t)):
            if svd_fallback:
                continue
            raise RuntimeError(
                f"Non-finite tensor at R-node ({i},{j},{k}): {t}"
            )

        # Check isotropy
        t_diag = np.array([t[0, 0], t[1, 1], t[2, 2]])
        t_avg  = t_diag.mean()
        off_diag_norm = np.linalg.norm(t - np.diag(t_diag))
        aniso  = abs(t_diag - t_avg).sum() + off_diag_norm
        if aniso < iso_tol * abs(t_avg) + 1e-300:
            # Effectively isotropic — write through to BOTH representations.
            # The returned array is sigma_R_tensor whenever it is allocated
            # (always, for tensor callables; from the first anisotropic cell
            # onward, for scalar ones), so updating only the scalar array
            # would silently discard the homogenized value.
            sigma_R_scalar[seq] = t_avg
            if sigma_R_tensor is not None:
                sigma_R_tensor[seq] = t_avg * np.eye(3, dtype=complex)
            continue

        _alloc_tensor_if_needed()
        sigma_R_tensor[seq] = t

    sigma_R_out = sigma_R_tensor if sigma_R_tensor is not None else sigma_R_scalar
    mu_P  = np.full(grid.N_P, complex(mu))
    eps_R = np.full(N_R,      complex(eps))
    return EMMedia(grid, sigma_R_out, mu_P, eps_R)


def planar_interface_isotropic(
    grid: LebedevGrid3D,
    n_hat: np.ndarray,
    d_plane: float,
    sigma1: float,
    sigma2: float,
    mu1: float = MU0,
    mu2: float = MU0,
    eps1: float = EPS0,
    eps2: float = EPS0,
    n_gl: int = 10,
    method: str = "nodal",
) -> EMMedia:
    """
    Two-layer isotropic medium separated by a planar interface at arbitrary
    orientation, using effective-medium averaging for straddling dual cells.

    The interface is defined by  n̂ · x = d_plane  where n̂ is the unit
    normal pointing from layer 1 into layer 2.

    Layer assignment:
        layer 1 (σ₁, μ₁, ε₁) :  n̂ · x  <  d_plane
        layer 2 (σ₂, μ₂, ε₂) :  n̂ · x  ≥  d_plane

    Parameters
    ----------
    grid : LebedevGrid3D
    n_hat : array-like (3,)
        Interface normal vector (need not be unit; will be normalised).
    d_plane : float
        Signed offset: n̂ · x = d_plane defines the plane.
    sigma1, sigma2 : float  [S/m]
    mu1, mu2 : float  [H/m]   default = μ₀
    eps1, eps2 : float  [F/m]  default = ε₀
    n_gl : int
        Unused (kept for API compatibility; volume fraction is computed exactly).
    method : str
        Homogenization formula for straddling dual cells:

        ``"nodal"`` (default) — 3-D energy-matched nodal homogenization
            (Moskow et al. 1999, extended to 3-D):
            ΣD = L̃⁻ᵀ G L̃⁻¹,  L̃=[m̂|q̂|Dn̂],  G=diag(σ̄,σ̄,⟨σ⁻¹⟩_vol).
            Uses per-axis line averages of σ⁻¹ for D.  Reduces to the
            standard formula for axis-aligned and symmetric straddling nodes;
            gives a corrected tensor for asymmetrically straddling nodes.

        ``"standard"`` — Backus/Tartar arithmetic-harmonic tensor:
            ΣL = σ̄(I − n̂n̂ᵀ) + σ̃ n̂n̂ᵀ.
            Arithmetic mean tangential, harmonic mean normal.  The classical
            result valid for finely laminated media at arbitrary orientation.

    Nodes whose dual-cell box straddles only at corners (no 1-D grid-axis
    edge actually crosses the interface) are left with their pointwise scalar
    value (fake-straddle guard).

    Returns
    -------
    EMMedia
    """
    if method not in ("nodal", "standard"):
        raise ValueError(f"method must be 'nodal' or 'standard', got {method!r}")
    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)

    def _in_layer1(xyz: np.ndarray) -> bool:
        return float(n_hat @ xyz) < d_plane

    # ------------------------------------------------------------------
    # R-nodes — pass 1: pointwise assignment
    # ------------------------------------------------------------------
    sigma_R_scalar = np.empty(grid.N_R, dtype=complex)
    eps_R_scalar   = np.empty(grid.N_R, dtype=complex)
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        node = np.array([grid.x[i], grid.y[j], grid.z[k]])
        if _in_layer1(node):
            sigma_R_scalar[seq] = complex(sigma1)
            eps_R_scalar[seq]   = complex(eps1)
        else:
            sigma_R_scalar[seq] = complex(sigma2)
            eps_R_scalar[seq]   = complex(eps2)

    # R-nodes — pass 2: nodal homogenization for straddling cells
    sigma_R_tensor = None
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        node = np.array([float(grid.x[i]), float(grid.y[j]), float(grid.z[k])])

        # Dual-cell box: ±1 neighbor in each direction (clamped to grid bounds)
        ix_m = max(i - 1, 0);       ix_p = min(i + 1, len(grid.x) - 1)
        iy_m = max(j - 1, 0);       iy_p = min(j + 1, len(grid.y) - 1)
        iz_m = max(k - 1, 0);       iz_p = min(k + 1, len(grid.z) - 1)

        box_min = np.array([grid.x[ix_m], grid.y[iy_m], grid.z[iz_m]])
        box_max = np.array([grid.x[ix_p], grid.y[iy_p], grid.z[iz_p]])

        # Check whether the interface plane actually cuts the dual cell.
        xs = [box_min[0], box_max[0]]
        ys = [box_min[1], box_max[1]]
        zs = [box_min[2], box_max[2]]
        corner_vals = [n_hat[0]*x + n_hat[1]*y + n_hat[2]*z
                       for x in xs for y in ys for z in zs]
        if all(v < d_plane for v in corner_vals) or all(v >= d_plane for v in corner_vals):
            continue   # entire box in one layer

        # Skip nodes sitting exactly on the interface.
        node_val = float(n_hat @ node)
        tol_node = 1e-10 * (np.linalg.norm(node) + 1.0)
        if abs(node_val - d_plane) < tol_node:
            continue

        # ---- Fake-straddle guard: check that at least one 1-D edge crosses ----
        # x-edge at (y_j, z_k)
        d_rest_x = d_plane - n_hat[1] * node[1] - n_hat[2] * node[2]
        f_x = _frac_1d_layer1(box_min[0], box_max[0], n_hat[0], d_rest_x)
        # y-edge at (x_i, z_k)
        d_rest_y = d_plane - n_hat[0] * node[0] - n_hat[2] * node[2]
        f_y = _frac_1d_layer1(box_min[1], box_max[1], n_hat[1], d_rest_y)
        # z-edge at (x_i, y_j)
        d_rest_z = d_plane - n_hat[0] * node[0] - n_hat[1] * node[1]
        f_z = _frac_1d_layer1(box_min[2], box_max[2], n_hat[2], d_rest_z)

        _EPS_FRAC = 1e-12
        if not ((_EPS_FRAC < f_x < 1 - _EPS_FRAC) or
                (_EPS_FRAC < f_y < 1 - _EPS_FRAC) or
                (_EPS_FRAC < f_z < 1 - _EPS_FRAC)):
            # Box straddles only via corner geometry — no 1-D edge actually
            # crosses the interface.  Pointwise scalar from pass 1 is correct.
            continue

        # ---- Volume fraction → σ̄ and ⟨σ⁻¹⟩_vol ----
        f_vol    = _volume_frac_layer1_planar(box_min, box_max, n_hat, d_plane, n_gl)
        s_arith  = complex(f_vol * sigma1 + (1.0 - f_vol) * sigma2)
        inv_s_h  = complex(f_vol / sigma1 + (1.0 - f_vol) / sigma2)  # ⟨σ⁻¹⟩_vol

        # ---- Build effective tensor ----
        if method == "standard":
            # ΣL = σ̄(I − n̂n̂ᵀ) + σ̃ n̂n̂ᵀ  (Backus/Tartar)
            n_c = n_hat.astype(complex)
            t = (s_arith * (np.eye(3, dtype=complex) - np.outer(n_c, n_c))
                 + (1.0 / inv_s_h) * np.outer(n_c, n_c))
        else:
            # Nodal: ΣD = L̃⁻ᵀ G L̃⁻¹, G = diag(σ̄, σ̄, ⟨σ⁻¹⟩_vol)
            D_diag = np.array([
                complex(f_x / sigma1 + (1.0 - f_x) / sigma2),
                complex(f_y / sigma1 + (1.0 - f_y) / sigma2),
                complex(f_z / sigma1 + (1.0 - f_z) / sigma2),
            ])
            t = _nodal_eff_tensor_3d(D_diag, s_arith, inv_s_h, n_hat)

        # Allocate tensor array on first straddle.
        if sigma_R_tensor is None:
            sigma_R_tensor = np.zeros((grid.N_R, 3, 3), dtype=complex)
            for d in range(3):
                sigma_R_tensor[:, d, d] = sigma_R_scalar

        sigma_R_tensor[seq] = t

    sigma_R_out = sigma_R_tensor if sigma_R_tensor is not None else sigma_R_scalar

    # ------------------------------------------------------------------
    # P-nodes — same two-pass pattern for μ
    # ------------------------------------------------------------------
    mu_P_scalar = np.empty(grid.N_P, dtype=complex)
    for seq, (i, j, k) in enumerate(grid.P_nodes):
        node = np.array([grid.x[i], grid.y[j], grid.z[k]])
        mu_P_scalar[seq] = complex(mu1) if _in_layer1(node) else complex(mu2)

    mu_P_tensor = None
    for seq, (i, j, k) in enumerate(grid.P_nodes):
        node = np.array([float(grid.x[i]), float(grid.y[j]), float(grid.z[k])])

        ix_m = max(i - 1, 0);       ix_p = min(i + 1, len(grid.x) - 1)
        iy_m = max(j - 1, 0);       iy_p = min(j + 1, len(grid.y) - 1)
        iz_m = max(k - 1, 0);       iz_p = min(k + 1, len(grid.z) - 1)

        box_min = np.array([grid.x[ix_m], grid.y[iy_m], grid.z[iz_m]])
        box_max = np.array([grid.x[ix_p], grid.y[iy_p], grid.z[iz_p]])

        xs = [box_min[0], box_max[0]]
        ys = [box_min[1], box_max[1]]
        zs = [box_min[2], box_max[2]]
        corner_vals = [n_hat[0]*x + n_hat[1]*y + n_hat[2]*z
                       for x in xs for y in ys for z in zs]
        if all(v < d_plane for v in corner_vals) or all(v >= d_plane for v in corner_vals):
            continue

        node_val = float(n_hat @ node)
        tol_node = 1e-10 * (np.linalg.norm(node) + 1.0)
        if abs(node_val - d_plane) < tol_node:
            continue

        d_rest_x = d_plane - n_hat[1] * node[1] - n_hat[2] * node[2]
        f_x = _frac_1d_layer1(box_min[0], box_max[0], n_hat[0], d_rest_x)
        d_rest_y = d_plane - n_hat[0] * node[0] - n_hat[2] * node[2]
        f_y = _frac_1d_layer1(box_min[1], box_max[1], n_hat[1], d_rest_y)
        d_rest_z = d_plane - n_hat[0] * node[0] - n_hat[1] * node[1]
        f_z = _frac_1d_layer1(box_min[2], box_max[2], n_hat[2], d_rest_z)

        _EPS_FRAC = 1e-12
        if not ((_EPS_FRAC < f_x < 1 - _EPS_FRAC) or
                (_EPS_FRAC < f_y < 1 - _EPS_FRAC) or
                (_EPS_FRAC < f_z < 1 - _EPS_FRAC)):
            continue

        f_vol    = _volume_frac_layer1_planar(box_min, box_max, n_hat, d_plane, n_gl)
        mu_arith  = complex(f_vol * mu1 + (1.0 - f_vol) * mu2)
        inv_mu_h  = complex(f_vol / mu1 + (1.0 - f_vol) / mu2)  # ⟨μ⁻¹⟩_vol

        if method == "standard":
            n_c = n_hat.astype(complex)
            t = (mu_arith * (np.eye(3, dtype=complex) - np.outer(n_c, n_c))
                 + (1.0 / inv_mu_h) * np.outer(n_c, n_c))
        else:
            D_diag_mu = np.array([
                complex(f_x / mu1 + (1.0 - f_x) / mu2),
                complex(f_y / mu1 + (1.0 - f_y) / mu2),
                complex(f_z / mu1 + (1.0 - f_z) / mu2),
            ])
            t = _nodal_eff_tensor_3d(D_diag_mu, mu_arith, inv_mu_h, n_hat)

        if mu_P_tensor is None:
            mu_P_tensor = np.zeros((grid.N_P, 3, 3), dtype=complex)
            for d in range(3):
                mu_P_tensor[:, d, d] = mu_P_scalar

        mu_P_tensor[seq] = t

    mu_P_out = mu_P_tensor if mu_P_tensor is not None else mu_P_scalar

    return EMMedia(grid, sigma_R_out, mu_P_out, eps_R_scalar)

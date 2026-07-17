"""
operators.py — Sparse finite-difference curl operators for the Lebedev grid.

Builds the two key matrices needed for the DDH03 Maxwell system:

    C_RE  (3·N_P × 3·N_R)  : curl from E^R (at R-nodes) to H^P (at P-nodes)
    C_PR  (3·N_R × 3·N_P)  : curl from H^P (at P-nodes) back to R-nodes

Both use the DDH03 finite-difference formula (eq. 3):

    (f^P_x)_{i,j,k} = (f^R_{i+1,j,k} − f^R_{i−1,j,k}) / (x_{i+1} − x_{i−1})

i.e. a centered difference skipping by ±1 index (so that we always move from
one subgrid to the other).

The E-field vector is stored **component-blocked**:
    E = [Ex_0 … Ex_{N_R−1} | Ey_0 … Ey_{N_R−1} | Ez_0 … Ez_{N_R−1}]
and likewise for H.  This ordering yields 3×3 block-sparse system matrices.

Reference: Davydycheva, Druskin & Habashy (2003), Geophysics 68(5):1525–1536.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .grid import LebedevGrid3D


# ---------------------------------------------------------------------------
# Low-level: scalar directional derivative matrices N_out × N_in
# ---------------------------------------------------------------------------

def _build_d_dx(
    grid: LebedevGrid3D,
    from_R: bool,
) -> sp.csr_matrix:
    """
    d/dx finite-difference matrix.

    If from_R=True  : maps R-field values → P-node output  (N_P × N_R)
    If from_R=False : maps P-field values → R-node output  (N_R × N_P)

    The DDH03 formula (eq. 3) in both cases is the same:
        (result at node q) = (f_{q+e_x} − f_{q−e_x}) / (x_{q+1} − x_{q−1})
    where the input nodes are ±1 neighbors in x (always on the *other* subgrid).
    """
    if from_R:
        out_nodes = grid.P_nodes    # P-node is output
        in_idx    = grid.R_idx      # R-node is input
        N_out, N_in = grid.N_P, grid.N_R
        x = grid.x
    else:
        out_nodes = grid.R_nodes
        in_idx    = grid.P_idx
        N_out, N_in = grid.N_R, grid.N_P
        x = grid.x

    rows, cols, vals = [], [], []
    for seq, (i, j, k) in enumerate(out_nodes):
        # For H→E (from_R=False): skip same-axis boundary R-nodes; the full
        # set of electric-BC R-node DOFs is handled by apply_electric_bc.
        # For E→H (from_R=True): rows are built wherever both ±1 x-neighbours
        # exist.  The magnetic boundary condition H×n=0 (DDH03 eq. 6) is NOT
        # handled here: build_curl_RE zeroes the ENTIRE row of every
        # tangential H component at boundary P-nodes (see
        # _tangential_h_row_masks); keeping only the formable term of a
        # partial stencil would leave a spurious nonzero tangential H.
        if not from_R:
            if i == 0 or i == grid.Mx:
                continue
        im1, ip1 = i - 1, i + 1
        # Bounds-safe ghost access — avoids Python negative-index wrap.
        q_p = in_idx[ip1, j, k] if ip1 <= grid.Mx else -1
        q_m = in_idx[im1, j, k] if im1 >= 0       else -1
        if q_p < 0 or q_m < 0:
            # x-ghost missing → Neumann: ∂/∂x = 0 at this boundary P-node.
            continue
        denom = x[ip1] - x[im1]
        rows += [seq, seq]
        cols += [q_p, q_m]
        vals += [1.0 / denom, -1.0 / denom]

    return sp.csr_matrix((vals, (rows, cols)), shape=(N_out, N_in))


def _build_d_dy(
    grid: LebedevGrid3D,
    from_R: bool,
) -> sp.csr_matrix:
    """d/dy matrix; see _build_d_dx for documentation."""
    if from_R:
        out_nodes = grid.P_nodes
        in_idx    = grid.R_idx
        N_out, N_in = grid.N_P, grid.N_R
        y = grid.y
    else:
        out_nodes = grid.R_nodes
        in_idx    = grid.P_idx
        N_out, N_in = grid.N_R, grid.N_P
        y = grid.y

    rows, cols, vals = [], [], []
    for seq, (i, j, k) in enumerate(out_nodes):
        if not from_R:
            if j == 0 or j == grid.My:
                continue
        jm1, jp1 = j - 1, j + 1
        q_p = in_idx[i, jp1, k] if jp1 <= grid.My else -1
        q_m = in_idx[i, jm1, k] if jm1 >= 0       else -1
        if q_p < 0 or q_m < 0:
            continue
        denom = y[jp1] - y[jm1]
        rows += [seq, seq]
        cols += [q_p, q_m]
        vals += [1.0 / denom, -1.0 / denom]

    return sp.csr_matrix((vals, (rows, cols)), shape=(N_out, N_in))


def _build_d_dz(
    grid: LebedevGrid3D,
    from_R: bool,
) -> sp.csr_matrix:
    """d/dz matrix; see _build_d_dx for documentation."""
    if from_R:
        out_nodes = grid.P_nodes
        in_idx    = grid.R_idx
        N_out, N_in = grid.N_P, grid.N_R
        z = grid.z
    else:
        out_nodes = grid.R_nodes
        in_idx    = grid.P_idx
        N_out, N_in = grid.N_R, grid.N_P
        z = grid.z

    rows, cols, vals = [], [], []
    for seq, (i, j, k) in enumerate(out_nodes):
        if not from_R:
            if k == 0 or k == grid.Mz:
                continue
        km1, kp1 = k - 1, k + 1
        q_p = in_idx[i, j, kp1] if kp1 <= grid.Mz else -1
        q_m = in_idx[i, j, km1] if km1 >= 0       else -1
        if q_p < 0 or q_m < 0:
            continue
        denom = z[kp1] - z[km1]
        rows += [seq, seq]
        cols += [q_p, q_m]
        vals += [1.0 / denom, -1.0 / denom]

    return sp.csr_matrix((vals, (rows, cols)), shape=(N_out, N_in))


# ---------------------------------------------------------------------------
# Public: full 3×3 block curl operators
# ---------------------------------------------------------------------------

def _tangential_h_row_masks(grid: LebedevGrid3D) -> list[sp.dia_matrix]:
    """
    Diagonal 0/1 row masks implementing the magnetic boundary condition of
    DDH03 eq. (6):  H^P|∂Ω × n = 0.

    H_comp is *tangential* to a boundary face whose normal axis differs from
    comp; the BC sets it identically to zero there, so its entire row in
    C_RE must be dropped.  (Dropping only the term with a missing ghost
    neighbour — as a partial stencil would — leaves a spurious nonzero
    tangential H that breaks the E/M mixed-BC error cancellation and the
    discrete adjointness of the curl pair.)  The *normal* H component on a
    face is fully formable from in-face E values and keeps its stencil; it
    is not constrained by H×n = 0.

    Returns
    -------
    [K_x, K_y, K_z] : list of (N_P × N_P) diagonal matrices
        K_c has 0 on the diagonal at every P-node where H_c is tangential
        to some boundary face, 1 elsewhere.
    """
    keep = np.ones((3, grid.N_P))
    for seq, (i, j, k) in enumerate(grid.P_nodes):
        on_face = (
            i == 0 or i == grid.Mx,
            j == 0 or j == grid.My,
            k == 0 or k == grid.Mz,
        )
        for comp in range(3):
            if any(on_face[a] for a in range(3) if a != comp):
                keep[comp, seq] = 0.0
    return [sp.diags(keep[c]) for c in range(3)]


def build_curl_RE(grid: LebedevGrid3D) -> sp.csr_matrix:
    """
    Build the curl operator  C_RE : E^R → (curl E)^P.

    Shape: (3·N_P) × (3·N_R)

    With component-blocked ordering [Ex|Ey|Ez] for E and [Hx|Hy|Hz] for the
    output, the operator has the 3×3 block structure:

        C_RE = [ 0      -Dz_RP   Dy_RP ]
               [ Dz_RP   0      -Dx_RP ]
               [-Dy_RP   Dx_RP   0     ]

    where Dx_RP is the N_P × N_R matrix for d/dx (from R to P).

    The magnetic boundary condition H×n = 0 of DDH03 eq. (6) is built in:
    the row of every *tangential* H component at a boundary P-node is
    identically zero (see _tangential_h_row_masks), so C_RE @ E yields
    exactly zero tangential H on all six faces.  Normal H components at the
    boundary keep their (fully formable) stencils.
    """
    Dx = _build_d_dx(grid, from_R=True)   # N_P × N_R
    Dy = _build_d_dy(grid, from_R=True)
    Dz = _build_d_dz(grid, from_R=True)

    Z = sp.csr_matrix((grid.N_P, grid.N_R))  # zero block

    # Magnetic BC (eq. 6): zero whole rows of tangential H at boundary P-nodes
    Kx, Ky, Kz = _tangential_h_row_masks(grid)

    # Row order: Hx = Dy·Ez − Dz·Ey, Hy = Dz·Ex − Dx·Ez, Hz = Dx·Ey − Dy·Ex
    top    = sp.hstack([Z,          -(Kx @ Dz),  Kx @ Dy ])   # Hx row
    middle = sp.hstack([Ky @ Dz,     Z,         -(Ky @ Dx)])  # Hy row
    bottom = sp.hstack([-(Kz @ Dy),  Kz @ Dx,    Z       ])   # Hz row

    return sp.vstack([top, middle, bottom], format="csr")


def build_curl_PR(grid: LebedevGrid3D) -> sp.csr_matrix:
    """
    Build the curl operator  C_PR : H^P → (curl H)^R.

    Shape: (3·N_R) × (3·N_P)

    Block structure (analogous to C_RE with from_R=False):

        C_PR = [ 0       -Dz_PR   Dy_PR ]
               [ Dz_PR    0      -Dx_PR ]
               [-Dy_PR    Dx_PR   0     ]
    """
    Dx = _build_d_dx(grid, from_R=False)   # N_R × N_P
    Dy = _build_d_dy(grid, from_R=False)
    Dz = _build_d_dz(grid, from_R=False)

    Z = sp.csr_matrix((grid.N_R, grid.N_P))

    top    = sp.hstack([Z,   -Dz,  Dy ])
    middle = sp.hstack([Dz,   Z,  -Dx ])
    bottom = sp.hstack([-Dy,  Dx,  Z  ])

    return sp.vstack([top, middle, bottom], format="csr")


# ---------------------------------------------------------------------------
# Diagonal material property matrices (isotropic case)
# ---------------------------------------------------------------------------

def scalar_diag(values: np.ndarray, n_comp: int = 3) -> sp.dia_matrix:
    """
    Build a block-diagonal (component-blocked) diagonal matrix from a
    per-node scalar array *values* of length N.

    Result has shape (n_comp·N, n_comp·N) with values repeated for each
    component block: diag([v_0,…,v_{N-1}, v_0,…,v_{N-1}, v_0,…,v_{N-1}]).
    """
    v = np.tile(values, n_comp)
    return sp.diags(v, format="dia")


def tensor_block_diag(tensors: np.ndarray) -> sp.csr_matrix:
    """
    Build a block-diagonal matrix from a set of 3×3 tensors at N nodes.

    Parameters
    ----------
    tensors : ndarray, shape (N, 3, 3)

    Returns
    -------
    csr_matrix, shape (3N, 3N)
        With the component-blocked ordering [comp0_node0…comp0_nodeN-1 |
        comp1_node0… | comp2_node0…], the (α-block, β-block) sub-matrix is
        diag(tensors[:, α, β]).
    """
    N = tensors.shape[0]
    blocks = []
    for alpha in range(3):
        row_blocks = []
        for beta in range(3):
            row_blocks.append(sp.diags(tensors[:, alpha, beta]))
        blocks.append(row_blocks)
    return sp.bmat(blocks, format="csr")


# ---------------------------------------------------------------------------
# System matrix assembly
# ---------------------------------------------------------------------------

def build_system_matrix(
    grid: LebedevGrid3D,
    C_RE: sp.csr_matrix,
    C_PR: sp.csr_matrix,
    inv_mu_P: sp.spmatrix,
    sigma_dot_R: sp.spmatrix,
    omega: float,
) -> sp.csr_matrix:
    """
    Assemble the Maxwell system matrix for eq. (5) of DDH03:

        A = C_PR @ inv_mu_P @ C_RE − iω · sigma_dot_R

    Parameters
    ----------
    C_RE : (3·N_P, 3·N_R) curl matrix, E→H.
    C_PR : (3·N_R, 3·N_P) curl matrix, H→E.
    inv_mu_P : (3·N_P, 3·N_P) block-diagonal matrix of μ⁻¹ at P-nodes.
    sigma_dot_R : (3·N_R, 3·N_R) block-diagonal matrix of σ̇=σ+iωε at R-nodes.
    omega : angular frequency ω.

    Returns
    -------
    A : (3·N_R, 3·N_R) complex sparse matrix.
    """
    curl_curl = C_PR @ inv_mu_P @ C_RE
    return (curl_curl - 1j * omega * sigma_dot_R).tocsr()


def apply_electric_bc(
    A: sp.csr_matrix,
    b: np.ndarray,
    bc_dofs: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Enforce electric (Dirichlet) boundary conditions by zeroing rows/columns.

    For each DOF in *bc_dofs* we set:
        A[dof, :] = 0,  A[:, dof] = 0,  A[dof, dof] = 1,  b[dof] = 0

    Parameters
    ----------
    A : (N, N) CSR system matrix.
    b : (N,) right-hand side.
    bc_dofs : 1-D integer array of DOF indices to constrain.

    Returns
    -------
    A_bc, b_bc : modified copies.
    """
    if len(bc_dofs) == 0:
        return A.copy(), b.copy()

    N = A.shape[0]
    A_bc = A.copy()

    # --- Zero bc rows (O(n_bc) indptr slices) --------------------------------
    for dof in bc_dofs:
        A_bc.data[A_bc.indptr[dof]:A_bc.indptr[dof + 1]] = 0.0

    # --- Zero bc columns (single O(nnz) vectorised pass) ---------------------
    bc_mask = np.zeros(N, dtype=bool)
    bc_mask[bc_dofs] = True
    A_bc.data[bc_mask[A_bc.indices]] = 0.0

    # --- Set diagonal = 1 for bc DOFs ----------------------------------------
    diag_corr = sp.diags(bc_mask.astype(float), format="csr")
    A_bc = A_bc + diag_corr

    # --- Zero RHS at bc DOFs -------------------------------------------------
    b_bc = b.copy()
    b_bc[bc_dofs] = 0.0

    return A_bc, b_bc


def electric_bc_dofs(grid: LebedevGrid3D) -> np.ndarray:
    """
    Return DOF indices for the standard electric BC (E × n = 0) applied to
    **all** boundary R-nodes (all three components).

    Used for the basic single-cluster Yee scheme (cluster 000).  The
    per-cluster mixed BC logic lives in `solver.py`.

    The DOF ordering is component-blocked:  dof = comp * N_R + seq.
    """
    dofs = []
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        if grid.is_r_boundary(i, j, k):
            for comp in range(3):
                dofs.append(comp * grid.N_R + seq)
    return np.array(dofs, dtype=np.int64)

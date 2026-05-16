"""
sources.py — Source term construction for the Lebedev 4-cluster scheme.

For a dipole source at (x₀, y₀, z₀) each of the four clusters receives a
source distribution on its *native* sub-grid for the requested component.
The native sub-grid for cluster c and component comp is the set of R-nodes
whose type (i%2, j%2, k%2) maps that component to cluster c in _E_CLUSTER_MAP.

The source is placed by **trilinear interpolation** of the continuous source
onto the native sub-grid nodes that straddle (x₀, y₀, z₀).  This guarantees
that for each cluster the source weights sum to 1 and the weighted centroid
equals the true source location — satisfying both conditions in DDH03 eq. 7.

Reference: Davydycheva, Druskin & Habashy (2003), Geophysics 68(5):1525–1536.
"""

from __future__ import annotations

import numpy as np

from .grid import LebedevGrid3D, C000, C101, C110, C011, _E_CLUSTER_MAP


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _native_type_for_cluster_comp(cluster: int, comp: int) -> tuple[int, int, int]:
    """
    Return the R-node parity type (t1,t2,t3) at which cluster *cluster* owns
    E_{comp}.  Reverses the lookup in _E_CLUSTER_MAP.
    """
    for t, clusters in _E_CLUSTER_MAP.items():
        if int(clusters[comp]) == cluster:
            return t
    raise ValueError(f"No native type for cluster {cluster}, comp {comp}")


def _interp1d_weights(pos: np.ndarray, val: float) -> list[tuple[int, float]]:
    """
    1-D linear interpolation weights on a sorted array *pos*.

    Returns a list of (sub_index, weight) pairs (1 or 2 entries).
    Clamps to the array interior so the result is never out-of-range.
    """
    n = len(pos)
    if n == 0:
        return []
    if n == 1:
        return [(0, 1.0)]

    idx = int(np.searchsorted(pos, val)) - 1
    idx = max(0, min(n - 2, idx))
    lo, hi = float(pos[idx]), float(pos[idx + 1])
    span = hi - lo

    if span < 1e-30:
        return [(idx, 1.0)]

    whi = float((val - lo) / span)
    wlo = 1.0 - whi
    result: list[tuple[int, float]] = []
    if abs(wlo) > 1e-14:
        result.append((idx, wlo))
    if abs(whi) > 1e-14:
        result.append((idx + 1, whi))
    return result


def _trilinear_r_nodes(
    grid: LebedevGrid3D,
    native_type: tuple[int, int, int],
    x0: float,
    y0: float,
    z0: float,
) -> list[tuple[int, float]]:
    """
    Find R-nodes of the given parity *native_type* that surround (x0,y0,z0)
    and return their trilinear interpolation weights.

    The native sub-grid for type (t1,t2,t3) consists of all grid nodes with
    i≡t1 (mod 2), j≡t2 (mod 2), k≡t3 (mod 2).

    Returns a list of (seq, weight) pairs where seq is the R-node sequential
    index and weight is the trilinear weight (non-negative, sum = 1 for
    sources strictly inside the domain).
    """
    t1, t2, t3 = int(native_type[0]), int(native_type[1]), int(native_type[2])

    # Sub-grid coordinate arrays for each axis
    x_sub = grid.x[t1::2]   # e.g. t1=1 → x[1],x[3],x[5],…
    y_sub = grid.y[t2::2]
    z_sub = grid.z[t3::2]

    ix_pairs = _interp1d_weights(x_sub, x0)
    iy_pairs = _interp1d_weights(y_sub, y0)
    iz_pairs = _interp1d_weights(z_sub, z0)

    nodes_weights: list[tuple[int, float]] = []
    for ix_sub, wx in ix_pairs:
        for iy_sub, wy in iy_pairs:
            for iz_sub, wz in iz_pairs:
                i_full = t1 + 2 * ix_sub
                j_full = t2 + 2 * iy_sub
                k_full = t3 + 2 * iz_sub
                if (0 <= i_full <= grid.Mx and
                        0 <= j_full <= grid.My and
                        0 <= k_full <= grid.Mz):
                    seq = int(grid.R_idx[i_full, j_full, k_full])
                    if seq >= 0:
                        nodes_weights.append((seq, float(wx * wy * wz)))

    return nodes_weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _dual_cell_vol(grid: LebedevGrid3D, seq: int) -> float:
    """
    Return the dual-cell volume for R-node *seq* using skip-2 spacings.

    The DDH03 curl operators use the skip-2 finite-difference formula
    (f[i+1] − f[i−1]) / (coord[i+1] − coord[i−1]), so the natural
    dual-cell width in each direction is coord[i+1] − coord[i−1].
    Boundary nodes are clamped to avoid out-of-range access.

    This volume is used to convert the node-centred RHS value to the
    correct current-density normalisation for a point dipole source.
    """
    i, j, k = grid.R_nodes[seq]
    dx = (grid.x[min(i + 1, grid.Mx)] - grid.x[max(i - 1, 0)])
    dy = (grid.y[min(j + 1, grid.My)] - grid.y[max(j - 1, 0)])
    dz = (grid.z[min(k + 1, grid.Mz)] - grid.z[max(k - 1, 0)])
    return abs(dx * dy * dz)


def build_source_rhs(
    grid: LebedevGrid3D,
    x0: float,
    y0: float,
    z0: float,
    dipole_comp: int,
    omega: float,
    moment: float = 1.0,
) -> dict[int, np.ndarray]:
    """
    Build the right-hand side vector ``b_c = iω J_c`` for each cluster c.

    Source placement: for each cluster c, the source is distributed over the
    R-nodes of c's *native* sub-grid for component *dipole_comp* using
    trilinear weights.  This ensures:

        (1)  Σ weights = 1             (total source amplitude preserved)
        (2)  Σ weights · r_node = r₀   (weighted centroid = true source location)

    satisfying the DDH03 conditions for error cancellation.

    **Cell-volume normalisation** (point-dipole current density):
    The DDH03 system uses a point-collocation scheme in which each nodal
    equation approximates the PDE at that node, not an integral over a cell.
    For a continuous point dipole J(r) = p × δ(r − r₀), the discrete
    current density at node i is J_i = p × w_i / V_i, where w_i is the
    trilinear interpolation weight and V_i is the skip-2 dual-cell volume.
    Without the 1/V_i factor, clusters whose source nodes sit on larger
    dual cells (e.g. C011) inject proportionally more total moment than
    clusters on smaller cells (e.g. C101), destroying the equal-source
    assumption underlying DDH03's superconvergence theorem.

    Parameters
    ----------
    grid : LebedevGrid3D
    x0, y0, z0 : float
        Physical source location.
    dipole_comp : int
        Dipole orientation: 0=x, 1=y, 2=z.
    omega : float
        Angular frequency [rad/s].
    moment : float
        Dipole moment [A·m] (default 1).

    Returns
    -------
    rhs : dict {cluster_int → ndarray of shape (3·N_R,)}
        Complex RHS (= iω J) for each cluster.
    """
    rhs = {c: np.zeros(3 * grid.N_R, dtype=complex)
           for c in (C000, C101, C110, C011)}

    iomega_m = 1j * omega * moment

    for c in (C000, C101, C110, C011):
        native_type = _native_type_for_cluster_comp(c, dipole_comp)
        nodes_weights = _trilinear_r_nodes(grid, native_type, x0, y0, z0)

        if not nodes_weights:
            continue

        # Normalise so weights sum to exactly 1 (robustness near boundary)
        total_w = sum(w for _, w in nodes_weights)
        if total_w < 1e-14:
            continue

        for seq, w in nodes_weights:
            vol = _dual_cell_vol(grid, seq)
            dof = dipole_comp * grid.N_R + seq
            # Divide by dual-cell volume so the total injected moment
            # (= Σ rhs_i × V_i) equals iω × moment for all clusters.
            rhs[c][dof] += iomega_m * (w / total_w) / vol

    return rhs


def point_dipole_rhs(
    grid: LebedevGrid3D,
    x0: float,
    y0: float,
    z0: float,
    dipole_comp: int,
    omega: float,
    moment: float = 1.0,
) -> dict[int, np.ndarray]:
    """Convenience wrapper for ``build_source_rhs``."""
    return build_source_rhs(grid, x0, y0, z0, dipole_comp, omega, moment)

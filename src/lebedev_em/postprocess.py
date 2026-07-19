"""
postprocess.py — Field postprocessing for the Lebedev Maxwell solver.

Three main capabilities:

1. B-field computation
   B = (1/iω) curl E    (from Faraday: ∇×E = iωμH, so B = μH = (∇×E)/iω)
   The curl is evaluated using the already-built C_RE operator, giving B at
   P-nodes in the same component-blocked ordering as the E-field.

2. Axis extraction (per-cluster DOF vectors)
   For borehole-logging geometry, the transmitter and receivers lie on the
   z-axis (x=0, y=0).  The extraction routines locate grid nodes near that
   axis and return (z_coordinate, field_value) pairs suitable for plotting.

3. Proper Lebedev inter-cluster evaluation
   The four Lebedev clusters each place E-components on different sub-grids.
   To evaluate the correctly averaged field at a common set of points each
   cluster's solution must be trilinearly interpolated to that point from its
   native sub-grid and then averaged.  Functions here implement this step.

   Note: the raw DOF-vector average stored in result['E_avg'] by the solver is
   NOT the correct Lebedev average — it gives only 1/4 of the contribution at
   each node type because only one cluster has a non-zero native DOF there.
   Use ``lebedev_E_at_points`` for the physically correct averaged field.

Reference: Davydycheva, Druskin & Habashy (2003), Geophysics 68(5):1525–1536.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .grid import LebedevGrid3D, C000, C101, C110, C011, _E_CLUSTER_MAP, _H_CLUSTER_MAP
from .operators import build_curl_RE
from .sources import _interp1d_weights, _trilinear_r_nodes, _native_type_for_cluster_comp


# ---------------------------------------------------------------------------
# B-field computation
# ---------------------------------------------------------------------------

def compute_B_from_E(
    grid: LebedevGrid3D,
    E_vec: np.ndarray,
    omega: float,
) -> np.ndarray:
    """
    Compute the magnetic induction **B** at P-nodes from the solved E-field.

    From Faraday's law (DDH03 eq. 1, exp(-iωt) convention):
        ∇ × E = iω μ H  →  B = μ H = (∇ × E) / (iω)

    Parameters
    ----------
    grid  : LebedevGrid3D
    E_vec : ndarray, shape (3·N_R,)
        Electric field in component-blocked ordering [Ex|Ey|Ez].
    omega : float
        Angular frequency ω [rad/s].

    Returns
    -------
    B_vec : ndarray, shape (3·N_P,)
        Magnetic induction **B** in component-blocked ordering [Bx|By|Bz]
        at the P-nodes.
    """
    C_RE = build_curl_RE(grid)         # 3·N_P × 3·N_R
    curl_E = C_RE @ E_vec              # ∇ × E at P-nodes
    return curl_E / (1j * omega)


# ---------------------------------------------------------------------------
# Axis extraction — R-nodes (E-field)
# ---------------------------------------------------------------------------

def extract_E_on_axis(
    grid: LebedevGrid3D,
    E_vec: np.ndarray,
    comp: int = 0,
    axis: str = "z",
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract one component of **E** at R-nodes lying on the specified axis.

    For axis='z' the function selects R-nodes with x ≈ 0 and y ≈ 0 (or
    their nearest-grid equivalents) and returns (z_coord, E_comp) pairs
    sorted by z.

    Parameters
    ----------
    E_vec : ndarray, shape (3·N_R,)
        Electric field, component-blocked.
    comp  : int
        Field component to extract: 0=Ex, 1=Ey, 2=Ez.
    axis  : str
        'z' (borehole axis), 'x', or 'y'.
    tol   : float
        Absolute tolerance for "on-axis" coordinate.

    Returns
    -------
    coords : ndarray, shape (N,)
        Axis coordinate of each extracted node.
    values : ndarray, shape (N,), complex
        Corresponding E_comp values.
    """
    ax_map = {"x": 0, "y": 1, "z": 2}
    ax = ax_map[axis]
    perp = [i for i in range(3) if i != ax]

    # Coordinates closest to zero on the perpendicular axes
    xyz_grid = [grid.x, grid.y, grid.z]
    perp_centers = [xyz_grid[p][np.argmin(np.abs(xyz_grid[p]))] for p in perp]

    coords_list, vals_list = [], []
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        xyz = grid.node_xyz(i, j, k)
        on_axis = all(
            abs(xyz[p] - perp_centers[pi]) < tol or
            abs(xyz[p] - perp_centers[pi]) < abs(np.diff(xyz_grid[p]).min())
            for pi, p in enumerate(perp)
        )
        if on_axis:
            coords_list.append(xyz[ax])
            vals_list.append(E_vec[comp * grid.N_R + seq])

    if not coords_list:
        raise RuntimeError(
            f"No R-nodes found near {axis}=0 axis. "
            "Check grid symmetry or increase tol."
        )

    coords = np.array(coords_list, dtype=float)
    values = np.array(vals_list, dtype=complex)
    order = np.argsort(coords)
    return coords[order], values[order]


# ---------------------------------------------------------------------------
# Axis extraction — P-nodes (B-field)
# ---------------------------------------------------------------------------

def extract_B_on_axis(
    grid: LebedevGrid3D,
    B_vec: np.ndarray,
    comp: int = 0,
    axis: str = "z",
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract one component of **B** at P-nodes lying on the specified axis.

    Parameters
    ----------
    B_vec : ndarray, shape (3·N_P,)
        Magnetic induction, component-blocked [Bx|By|Bz].
    comp  : int
        0=Bx, 1=By, 2=Bz.
    axis  : str
        'z', 'x', or 'y'.

    Returns
    -------
    coords : ndarray
    values : ndarray (complex)
    """
    ax_map = {"x": 0, "y": 1, "z": 2}
    ax = ax_map[axis]
    perp = [i for i in range(3) if i != ax]

    xyz_grid = [grid.x, grid.y, grid.z]
    perp_centers = [xyz_grid[p][np.argmin(np.abs(xyz_grid[p]))] for p in perp]

    coords_list, vals_list = [], []
    for seq, (i, j, k) in enumerate(grid.P_nodes):
        xyz = grid.node_xyz(i, j, k)
        on_axis = all(
            abs(xyz[p] - perp_centers[pi]) < tol or
            abs(xyz[p] - perp_centers[pi]) < abs(np.diff(xyz_grid[p]).min())
            for pi, p in enumerate(perp)
        )
        if on_axis:
            coords_list.append(xyz[ax])
            vals_list.append(B_vec[comp * grid.N_P + seq])

    if not coords_list:
        raise RuntimeError(
            f"No P-nodes found near {axis}=0 axis. "
            "Check grid symmetry or increase tol."
        )

    coords = np.array(coords_list, dtype=float)
    values = np.array(vals_list, dtype=complex)
    order = np.argsort(coords)
    return coords[order], values[order]


# ---------------------------------------------------------------------------
# Proper Lebedev inter-cluster evaluation
# ---------------------------------------------------------------------------

def interpolate_cluster_E(
    grid: LebedevGrid3D,
    E_vec: np.ndarray,
    cluster: int,
    comp: int,
    x0: float,
    y0: float,
    z0: float,
) -> complex:
    """
    Evaluate E_comp from *cluster*'s solution at (x0, y0, z0) by trilinear
    interpolation on the cluster's native sub-grid for that component.

    Each Lebedev cluster places E_comp on a specific parity sub-grid.  This
    function finds the surrounding nodes of that sub-grid and returns the
    linearly interpolated value, which is the cluster's contribution to the
    Lebedev average at (x0, y0, z0).

    Parameters
    ----------
    grid    : LebedevGrid3D
    E_vec   : ndarray, shape (3·N_R,)  — one cluster's full E-field vector.
    cluster : int  — C000, C101, C110, or C011.
    comp    : int  — 0=Ex, 1=Ey, 2=Ez.
    x0, y0, z0 : float  — evaluation point.

    Returns
    -------
    complex  — interpolated E_comp at (x0, y0, z0).
    """
    native_type = _native_type_for_cluster_comp(cluster, comp)
    nodes_weights = _trilinear_r_nodes(grid, native_type, x0, y0, z0)

    if not nodes_weights:
        return 0j

    val: complex = 0j
    for seq, w in nodes_weights:
        val += w * E_vec[comp * grid.N_R + seq]
    return val


def lebedev_E_at_point(
    grid: LebedevGrid3D,
    E_clusters: dict,
    comp: int,
    x0: float,
    y0: float,
    z0: float,
) -> complex:
    """
    Compute the correct Lebedev-averaged E_comp at (x0, y0, z0).

    Each of the four clusters contributes via trilinear interpolation from its
    native sub-grid.  The four contributions are then averaged.

    Parameters
    ----------
    grid       : LebedevGrid3D
    E_clusters : dict {cluster → ndarray (3·N_R,)} — per-cluster E-fields.
    comp       : int  — 0=Ex, 1=Ey, 2=Ez.
    x0, y0, z0 : float  — evaluation point.

    Returns
    -------
    complex  — Lebedev-averaged E_comp at (x0, y0, z0).
    """
    contributions = [
        interpolate_cluster_E(grid, E_clusters[c], c, comp, x0, y0, z0)
        for c in (C000, C101, C110, C011)
    ]
    return complex(np.mean(contributions))


def lebedev_E_on_z_axis(
    grid: LebedevGrid3D,
    E_clusters: dict,
    comp: int = 0,
    min_r: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Evaluate the Lebedev-averaged E_comp along the z-axis (x=0, y=0) using
    proper inter-cluster interpolation.

    Evaluation points are taken at the z-axis R-nodes of the cluster whose
    native sub-grid for *comp* lies on the z-axis.  (For comp=0 / Ex, these
    are the type-(0,0,1) nodes at x=0, y=0, z = z_k for k odd.)

    Parameters
    ----------
    grid       : LebedevGrid3D
    E_clusters : dict {cluster → ndarray (3·N_R,)}.
    comp       : int  — field component (0=Ex, 1=Ey, 2=Ez).
    min_r      : float  — skip evaluation points with |z| ≤ min_r.

    Returns
    -------
    z_vals   : ndarray, shape (N,)  — z coordinates of evaluation points.
    E_leb    : ndarray, shape (N,), complex  — Lebedev-averaged E_comp.
    E_per_c  : dict {cluster → ndarray (N,), complex}  — each cluster's
               interpolated contribution (before averaging).
    """
    # The cluster whose native Ex is on the z-axis for comp=0 is C101.
    # For general comp: we need the cluster whose native type has all-zero
    # transverse indices (i%2=0, j%2=0 for z-axis).
    # Find that cluster:
    z_axis_cluster = None
    for c in (C000, C101, C110, C011):
        t = _native_type_for_cluster_comp(c, comp)
        if t[0] == 0 and t[1] == 0:   # i%2=0, j%2=0 → lives on z-axis
            z_axis_cluster = c
            break

    if z_axis_cluster is None:
        raise RuntimeError(
            f"No cluster has comp={comp} on the z-axis (x=y=0 native sub-grid)."
        )

    # Collect z-axis node positions for that cluster
    t = _native_type_for_cluster_comp(z_axis_cluster, comp)
    i_center = grid.Mx // 2   # index of x=0 node (for even Mx)
    j_center = grid.My // 2

    z_vals_list = []
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        if i == i_center and j == j_center:
            node_type = (int(i % 2), int(j % 2), int(k % 2))
            if _E_CLUSTER_MAP.get(node_type) is not None:
                if int(_E_CLUSTER_MAP[node_type][comp]) == z_axis_cluster:
                    z = float(grid.z[k])
                    if abs(z) > min_r:
                        z_vals_list.append(z)

    z_vals_list.sort()
    z_vals = np.array(z_vals_list, dtype=float)

    # Evaluate each cluster by interpolation and compute Lebedev average
    E_per_c = {c: np.zeros(len(z_vals), dtype=complex) for c in (C000, C101, C110, C011)}
    E_leb = np.zeros(len(z_vals), dtype=complex)

    for idx, z in enumerate(z_vals):
        contrib = []
        for c in (C000, C101, C110, C011):
            val = interpolate_cluster_E(grid, E_clusters[c], c, comp, 0.0, 0.0, z)
            E_per_c[c][idx] = val
            contrib.append(val)
        E_leb[idx] = complex(np.mean(contrib))

    return z_vals, E_leb, E_per_c


# ---------------------------------------------------------------------------
# Multi-cluster B-field source construction  (DDH03 §"Source terms")
# ---------------------------------------------------------------------------

def _native_type_for_h_cluster_comp(cluster: int, comp: int) -> tuple:
    """
    Return the P-node parity type (t1,t2,t3) at which *cluster* owns H_{comp}.
    Reverses the lookup in _H_CLUSTER_MAP.
    """
    for t, owners in _H_CLUSTER_MAP.items():
        if int(owners[comp]) == cluster:
            return t
    raise ValueError(f"No native P-type for cluster {cluster}, comp {comp}")


def _trilinear_p_nodes(
    grid: LebedevGrid3D,
    native_type: tuple[int, int, int],
    x0: float,
    y0: float,
    z0: float,
) -> list[tuple[int, float]]:
    """
    Find P-nodes of parity *native_type* surrounding (x0, y0, z0) and return
    their trilinear interpolation weights as (seq, weight) pairs.

    P-node analogue of ``sources._trilinear_r_nodes``: the sub-grid for type
    (t1,t2,t3) consists of all grid nodes with i≡t1, j≡t2, k≡t3 (mod 2).
    Weights are coordinate-based (1-D linear interpolation along each axis),
    so they satisfy the DDH03 averaging conditions

        (1)  Σ w = 1,        (2)  Σ w · r_node = (x0, y0, z0)

    exactly — also on nonuniform grids, where the previous equal-weight
    stencils violated condition (2).  When the evaluation point lies outside
    the sub-grid range along an axis (e.g. a receiver on the outermost grid
    plane), the 1-D weights linearly EXTRApolate from the two nearest
    sub-grid nodes; both conditions still hold and no phantom zero values
    enter the stencil.
    """
    t1, t2, t3 = int(native_type[0]), int(native_type[1]), int(native_type[2])

    x_sub = grid.x[t1::2]
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
                    seq = int(grid.P_idx[i_full, j_full, k_full])
                    if seq >= 0:
                        nodes_weights.append((seq, float(wx * wy * wz)))
    return nodes_weights


def interpolate_cluster_B(
    grid: LebedevGrid3D,
    B_vec: np.ndarray,
    cluster: int,
    comp: int,
    x0: float,
    y0: float,
    z0: float,
) -> complex:
    """
    Evaluate B_comp from *cluster*'s solution at (x0, y0, z0) by trilinear
    interpolation on the cluster's native H_{comp} P-sub-grid.

    B-field analogue of ``interpolate_cluster_E``.  The weights are
    coordinate-based linear-interpolation weights (Σw = 1, centroid at the
    evaluation point), as required by DDH03's averaging procedure, so the
    result is second-order accurate also on nonuniform grids.

    Boundary handling: nodes that fall outside the stored P-grid are skipped
    and the remaining weights are renormalised to sum to 1 (this can only
    happen for degenerate grids — every parity-consistent node inside the
    domain is a P-node — and never silently dilutes the value with zeros).
    Returns 0j if no stencil node exists at all.
    """
    native_type = _native_type_for_h_cluster_comp(cluster, comp)
    nodes_weights = _trilinear_p_nodes(grid, native_type, x0, y0, z0)
    if not nodes_weights:
        return 0j

    w_sum = sum(w for _, w in nodes_weights)
    if abs(w_sum) < 1e-14:
        return 0j

    val: complex = 0j
    for seq, w in nodes_weights:
        val += w * B_vec[comp * grid.N_P + seq]
    return complex(val / w_sum)


def _magnetic_source_groups(
    grid: LebedevGrid3D,
    comp: int = 0,
) -> tuple[tuple[int, int, int], dict[int, list[tuple[int, float]]]]:
    """
    Locate the nominal magnetic-dipole source node and build, for each
    cluster, its (seq, weight) source group on the cluster's native H_{comp}
    sub-grid.

    Nominal source node: the type-(0,0,0) P-node (i0, j0, k0) with
    i0 = Mx//2, j0 = My//2 and k0 the EVEN z-index closest to z = 0.
    If z = 0 does not coincide with an even-index grid plane the source is
    silently SNAPPED to z = z[k0]; a UserWarning is emitted when the snap
    displacement exceeds 1e-9 of the local grid spacing, so grids that do
    not centre the source exactly (see benchmark_figs2_3.py for how to
    build one that does) are flagged rather than silently mislocated.

    Weights are the coordinate-based linear-interpolation weights of the
    (snapped) source point on each cluster's sub-grid, so every group
    satisfies DDH03's source conditions (1) Σw = 1 and (2) Σw·r = r₀
    exactly, also on nonuniform grids.  On a locally symmetric grid they
    reduce to the familiar single weight-1 node (owning cluster) and three
    groups of 4 nodes with weight 1/4.

    Returns
    -------
    (i0, j0, k0) : the nominal source node indices.
    groups       : dict {cluster → list of (P-seq, weight)}.
    """
    Mx, My, Mz = grid.Mx, grid.My, grid.Mz
    i0 = Mx // 2
    j0 = My // 2
    k_even = np.arange(0, Mz + 1, 2)
    k0 = int(k_even[np.argmin(np.abs(grid.z[k_even]))])

    x0, y0, z0 = float(grid.x[i0]), float(grid.y[j0]), float(grid.z[k0])

    dz_local = abs(float(grid.z[min(k0 + 1, Mz)] - grid.z[max(k0 - 1, 0)]))
    if abs(z0) > 1e-9 * max(dz_local, 1e-300):
        import warnings
        warnings.warn(
            f"Magnetic-dipole source snapped from z=0 to nearest even-index "
            f"plane z={z0:.6g} m (offset {abs(z0):.3g} m). Build the z-grid so "
            f"that z=0 is an even-index node to avoid this displacement.",
            UserWarning,
        )

    groups: dict[int, list[tuple[int, float]]] = {}
    for c in (C000, C101, C110, C011):
        t = _native_type_for_h_cluster_comp(c, comp)
        nodes_weights = _trilinear_p_nodes(grid, t, x0, y0, z0)
        if not nodes_weights:
            raise ValueError(
                f"No native H sub-grid nodes found for cluster {c}, comp {comp}."
            )
        groups[c] = nodes_weights
    return (i0, j0, k0), groups


def _dual_cell_vol_p(grid: LebedevGrid3D, seq: int) -> float:
    """Skip-2 dual-cell volume of P-node *seq* (clamped at the boundary)."""
    i, j, k = grid.P_nodes[seq]
    dx = float(grid.x[min(i + 1, grid.Mx)] - grid.x[max(i - 1, 0)])
    dy = float(grid.y[min(j + 1, grid.My)] - grid.y[max(j - 1, 0)])
    dz = float(grid.z[min(k + 1, grid.Mz)] - grid.z[max(k - 1, 0)])
    return abs(dx * dy * dz)


def build_rhs_per_cluster(
    grid,
    C_PR,
    omega: float,
    hx_comp: int = 0,
) -> dict:
    """
    Build four separate RHS vectors — one per Lebedev cluster — for a unit-moment
    magnetic dipole source oriented along *hx_comp* (default 0 = x).

    For each cluster c the source is distributed over c's native H_{hx_comp}
    sub-grid nodes surrounding the nominal source point using coordinate-based
    linear-interpolation weights (see ``_magnetic_source_groups``), so every
    group satisfies DDH03's source conditions (1) Σw = 1 and (2) Σw·r = r₀
    exactly, also on nonuniform grids.  Each node contributes
    M_P = weight / V_dual (point-collocation normalisation: total injected
    moment Σᵢ Mᵢ·Vᵢ = 1 A·m² per cluster) and the RHS is b_c = iω C_PR M_P.

    Source position: the type-(0,0,0) P-node nearest to (0, 0, 0); if z = 0
    is not an even-index grid plane the source snaps to the nearest such
    plane and a UserWarning is emitted (see ``_magnetic_source_groups``).

    Parameters
    ----------
    grid     : LebedevGrid3D
    C_PR     : sparse matrix (3·N_R × 3·N_P)
    omega    : float  — angular frequency [rad/s]
    hx_comp  : int    — dipole orientation, 0=x 1=y 2=z

    Returns
    -------
    rhs_per_c : dict {cluster_int → ndarray (3·N_R,), complex}
    """
    (i0, j0, k0), groups = _magnetic_source_groups(grid, hx_comp)
    N_P = grid.N_P

    rhs_per_c: dict[int, np.ndarray] = {}
    for c in (C000, C101, C110, C011):
        M_P = np.zeros(3 * N_P, dtype=complex)
        for seq, w in groups[c]:
            M_P[hx_comp * N_P + seq] += w / _dual_cell_vol_p(grid, seq)
        rhs_per_c[c] = 1j * omega * (C_PR @ M_P)

    print(f"    Per-cluster source at ({i0},{j0},{k0})="
          f"({grid.x[i0]:.4f},{grid.y[j0]:.4f},{grid.z[k0]:.4f}), "
          f"n_nodes/cluster={[len(groups[c]) for c in (C000, C101, C110, C011)]}",
          flush=True)

    return rhs_per_c


def build_rhs_multicl(grid, C_PR: "sp.spmatrix", omega: float) -> np.ndarray:
    """
    Build the RHS vector for an x-oriented magnetic dipole source distributed
    across all four Lebedev clusters (DDH03 multi-cluster source).

    This is simply the sum of the four per-cluster RHS vectors from
    ``build_rhs_per_cluster``: the four clusters' native H_x sub-grids are
    disjoint parity classes, so each DOF receives a contribution from exactly
    one cluster and no double-counting occurs.  All weight and normalisation
    properties (Σw = 1, Σw·r = r₀ per cluster, unit moment per cluster, snap
    warning when z = 0 is not a grid plane) are inherited from
    ``build_rhs_per_cluster``.

    Parameters
    ----------
    grid  : LebedevGrid3D
    C_PR  : sparse matrix, shape (3·N_R, 3·N_P) — curl operator P→R
    omega : float — angular frequency [rad/s]

    Returns
    -------
    b : ndarray, shape (3·N_R,), complex — right-hand side vector
    """
    rhs_per_c = build_rhs_per_cluster(grid, C_PR, omega, hx_comp=0)
    b = np.zeros(3 * grid.N_R, dtype=complex)
    for c in (C000, C101, C110, C011):
        b += rhs_per_c[c]
    return b


# ---------------------------------------------------------------------------
# Multi-cluster B-field extraction
# ---------------------------------------------------------------------------

def extract_B_on_axis_multicl(
    grid,
    B_vec: np.ndarray,
    comp: int = 0,
    axis: str = "z",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract one component of **B** along the specified axis using the
    Lebedev multi-cluster average of a SINGLE B vector.

    Note: because this function interpolates all four cluster sub-grids of
    one shared B vector, it is safe for anisotropic media PROVIDED the B
    vector comes from a coupled solve with all-cluster sources
    (``solve_coupled`` / summed multicluster RHS).  Do not feed it a
    single-cluster-source solution in anisotropic media — see the warning
    on ``lebedev_B_on_z_axis``.

    Each cluster's contribution is obtained by trilinear interpolation on the
    cluster's native H_{comp} P-sub-grid using coordinate-based weights
    (Σw = 1, weighted centroid = receiver position — DDH03's linear
    interpolation from the nearest shifted nodes, exact also on nonuniform
    grids).  The four cluster contributions are then ARITHMETICALLY AVERAGED
    to give the Lebedev value.  The cluster that owns B_{comp} at the
    receiver's own node type contributes its nodal value directly (weight 1).

    Receiver positions: all even-k P-nodes with (i==i0, j==j0) — the
    z-axis type-(0,0,0) nodes (same as the old single-cluster extraction).

    Boundary handling: for receivers on the outermost z-planes the shifted
    sub-grids are evaluated by linear EXTRApolation from the two nearest
    planes (see ``_trilinear_p_nodes``) instead of the previous behaviour of
    averaging in phantom zeros for out-of-range nodes.

    Parameters
    ----------
    grid  : LebedevGrid3D
    B_vec : ndarray, shape (3·N_P,)  — magnetic induction, component-blocked.
    comp  : int — 0=Bx, 1=By, 2=Bz.
    axis  : str — 'z', 'x', or 'y' (only 'z' fully implemented for borehole).

    Returns
    -------
    coords : ndarray, shape (N,)       — axis coordinate of each receiver.
    values : ndarray, shape (N,), complex — Lebedev-averaged B_comp.
    """
    if axis != "z":
        # Fall back to single-cluster for non-z axes
        return extract_B_on_axis(grid, B_vec, comp=comp, axis=axis)

    Mx, My, Mz = grid.Mx, grid.My, grid.Mz
    i0 = Mx // 2
    j0 = My // 2
    x0 = float(grid.x[i0])
    y0 = float(grid.y[j0])

    # Receiver positions: type-(0,0,0) nodes on z-axis = (i0, j0, k) for k even
    coords_list = []
    values_list = []

    for k in range(0, Mz + 1, 2):
        seq = int(grid.P_idx[i0, j0, k])
        if seq < 0:
            continue
        z = float(grid.z[k])

        contributions = [
            interpolate_cluster_B(grid, B_vec, c, comp, x0, y0, z)
            for c in (C000, C101, C110, C011)
        ]
        coords_list.append(z)
        values_list.append(complex(np.mean(contributions)))

    if not coords_list:
        raise RuntimeError("No on-axis P-nodes found for multi-cluster B extraction.")

    coords = np.array(coords_list, dtype=float)
    values = np.array(values_list, dtype=complex)
    order = np.argsort(coords)
    return coords[order], values[order]


def lebedev_B_on_z_axis(
    grid,
    B_clusters: dict,
    comp: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Lebedev-averaged B_{comp} on the z-axis from four cluster
    solutions (DDH03's multi-cluster approach for ISOTROPIC media).

    .. warning::
        **Anisotropic media (off-diagonal σ): use the coupled solve.**
        When σ has off-diagonal entries, the clusters couple, and a solve
        driven by a single cluster's source deposits part of its response —
        in particular the anisotropy-generated CROSS-components (e.g. B_xz
        from an x-dipole) — on the PARTNER clusters' sub-grids.  Reading
        each solution only on its own source-cluster's sub-grid (as this
        function does when given four per-cluster-source solutions)
        under-counts those cross-components, up to losing them entirely in
        a homogeneous tilted-anisotropic medium.  For anisotropic media,
        obtain B from the single coupled solve
        (``LebedevMaxwellSolver.solve_coupled`` or an all-cluster RHS) and
        pass the SAME B vector for every cluster key — then this function's
        per-sub-grid interpolation and averaging are correct.  See
        ``tests/test_anisotropic_coupling.py``.

    Each cluster's B-field is interpolated to the receiver position from its
    OWN native H_{comp} sub-grid using coordinate-based trilinear weights
    (Σw = 1, weighted centroid = receiver — DDH03's linear interpolation,
    exact also on nonuniform grids), and the four cluster contributions are
    then ARITHMETICALLY AVERAGED.

    This is DDH03's procedure:
    (a) solve four separate systems, one per cluster source, using the
        coupled matrix A but each cluster's own RHS;
    (b) linearly interpolate B from each cluster's native nodes;
    (c) take the arithmetic average of the four contributions.

    Boundary handling: for receivers on the outermost z-planes the shifted
    sub-grids are evaluated by linear extrapolation from the two nearest
    planes (no phantom zeros; see ``_trilinear_p_nodes``).

    Parameters
    ----------
    grid       : LebedevGrid3D
    B_clusters : dict {cluster_int → ndarray (3·N_P,)} — one B-field per
                 cluster (each computed from a separate solve).
    comp       : int — 0=Bx, 1=By, 2=Bz.

    Returns
    -------
    z_vals : ndarray, shape (N,) — z coordinates of on-axis receiver positions.
    B_avg  : ndarray, shape (N,), complex — Lebedev-averaged B_{comp}.
    """
    Mx, My, Mz = grid.Mx, grid.My, grid.Mz
    i0 = Mx // 2
    j0 = My // 2
    x0 = float(grid.x[i0])
    y0 = float(grid.y[j0])

    # Receiver positions: on-axis P-nodes of type (0,0,0), i.e. k even
    z_vals_list: list[float] = []
    B_avg_list: list[complex] = []

    for k_r in range(0, Mz + 1, 2):
        if grid.P_idx[i0, j0, k_r] < 0:
            continue
        z = float(grid.z[k_r])

        contributions = [
            interpolate_cluster_B(grid, B_clusters[c], c, comp, x0, y0, z)
            for c in (C000, C101, C110, C011)
        ]
        z_vals_list.append(z)
        B_avg_list.append(complex(np.mean(contributions)))

    if not z_vals_list:
        raise RuntimeError("No on-axis P-nodes found for lebedev_B_on_z_axis.")

    z_vals = np.array(z_vals_list, dtype=float)
    B_avg = np.array(B_avg_list, dtype=complex)
    order = np.argsort(z_vals)
    return z_vals[order], B_avg[order]

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
# Source normalisation helper
# ---------------------------------------------------------------------------

def source_normalization_factor(
    grid: LebedevGrid3D,
    source_node: tuple[int, int, int],
    comp: int,
) -> float:
    """
    Return the effective dipole moment (in A·m) corresponding to a unit FD
    source weight at *source_node* for component *comp*.

    In the Yee/Lebedev FD scheme an electric current density of unit weight
    placed at node (i,j,k) for E_comp has an effective dipole moment equal to
    the dual-cell edge length along comp:

        p_eff = Δl_comp    [A·m]

    where Δl_comp is the edge length of the primary cell at that node.

    This factor converts between the FD "unit source" and the analytic dipole
    formula which uses SI moment in A·m.
    """
    i, j, k = source_node
    grids = [grid.x, grid.y, grid.z]
    g = grids[comp]
    idx = [i, j, k][comp]
    # Primary edge length along comp at this node
    if 0 < idx < len(g) - 1:
        dl = (g[idx + 1] - g[idx - 1]) / 2.0
    elif idx == 0:
        dl = g[1] - g[0]
    else:
        dl = g[-1] - g[-2]
    return float(dl)


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


def build_rhs_per_cluster(
    grid,
    C_PR,
    omega: float,
    hx_comp: int = 0,
) -> dict:
    """
    Build four separate RHS vectors — one per Lebedev cluster — for a unit-moment
    magnetic dipole source oriented along *hx_comp* (default 0 = x).

    For each cluster c the source is placed on c's native H_{hx_comp} sub-grid
    nodes that surround the nominal source point (i0, j0, k0):
      • One cluster has exactly one native node at (i0, j0, k0) → weight = 1
      • The other three clusters each have 4 surrounding nodes → weight = 1/4 each
    All four groups satisfy (1) Σw = 1 and (2) Σw·r = r₀.

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
    Mx, My, Mz = grid.Mx, grid.My, grid.Mz
    i0 = Mx // 2
    j0 = My // 2
    # Nearest even k to z=0
    k_even = np.arange(0, Mz + 1, 2)
    k0 = int(k_even[np.argmin(np.abs(grid.z[k_even]))])
    N_P = grid.N_P

    def vol_p(i, j, k):
        dx = float(grid.x[min(i + 1, Mx)] - grid.x[max(i - 1, 0)])
        dy = float(grid.y[min(j + 1, My)] - grid.y[max(j - 1, 0)])
        dz = float(grid.z[min(k + 1, Mz)] - grid.z[max(k - 1, 0)])
        return abs(dx * dy * dz)

    # Node groups per cluster:  cluster_int → list of (i, j, k, weight)
    # H_x cluster map at the 4 possible P-node types:
    #   type (0,0,0) → C011 owns Hx  ← one node exactly at (i0, j0, k0)
    #   type (0,1,1) → C000 owns Hx  ← 4 nodes at (i0, j0±1, k0±1)
    #   type (1,0,1) → C110 owns Hx  ← 4 nodes at (i0±1, j0, k0±1)
    #   type (1,1,0) → C101 owns Hx  ← 4 nodes at (i0±1, j0±1, k0)
    node_groups: dict[int, list] = {C000: [], C101: [], C110: [], C011: []}

    for c in (C000, C101, C110, C011):
        t1, t2, t3 = _native_type_for_h_cluster_comp(c, hx_comp)
        # Find which ±offsets give this parity relative to (i0, j0, k0)
        di_vals = [0] if t1 == i0 % 2 else [+1, -1]
        dj_vals = [0] if t2 == j0 % 2 else [+1, -1]
        dk_vals = [0] if t3 == k0 % 2 else [+1, -1]
        nodes = [(i0 + di, j0 + dj, k0 + dk)
                 for di in di_vals for dj in dj_vals for dk in dk_vals]
        n = len(nodes)
        w = 1.0 / n
        node_groups[c] = [(i, j, k, w) for (i, j, k) in nodes]

    rhs_per_c: dict[int, np.ndarray] = {}
    for c in (C000, C101, C110, C011):
        M_P = np.zeros(3 * N_P, dtype=complex)
        for (i, j, k, w) in node_groups[c]:
            seq = int(grid.P_idx[i, j, k])
            if seq < 0:
                raise ValueError(f"P-node ({i},{j},{k}) not in P-grid for cluster {c}")
            M_P[hx_comp * N_P + seq] += w / vol_p(i, j, k)
        rhs_per_c[c] = 1j * omega * (C_PR @ M_P)

    print(f"    Per-cluster source at ({i0},{j0},{k0})="
          f"({grid.x[i0]:.4f},{grid.y[j0]:.4f},{grid.z[k0]:.4f}), "
          f"n_nodes/cluster={[len(node_groups[c]) for c in (C011,C000,C101,C110)]}",
          flush=True)

    return rhs_per_c


def build_rhs_multicl(grid, C_PR: "sp.spmatrix", omega: float) -> np.ndarray:
    """
    Build the RHS vector for an x-oriented magnetic dipole source distributed
    across all four Lebedev clusters (DDH03 multi-cluster source).

    DDH03 requires that for each cluster:
      (1) The sum of source weights = 1
      (2) The center of mass of source nodes = the nominal source point (x₀, y₀, z₀)

    The nominal source point is (x₀, y₀, z₀) = (0, 0, ~0), placed at the
    P-node (i0, j0, k0) where i0=Mx//2, j0=My//2, k0=nearest even index to z=0.

    H_x cluster ownership (from _H_CLUSTER_MAP):
      type (0,0,0) → C011  ← the original single-cluster source node
      type (0,1,1) → C000
      type (1,0,1) → C110
      type (1,1,0) → C101

    For each cluster, we place 4 nodes surrounding (i0,j0,k0) with equal
    weight 1/4 each, arranged symmetrically so the centre of mass = (i0,j0,k0):
      C011: (i0, j0, k0)           — weight 1  (already type (0,0,0))
      C000: (i0, j0±1, k0±1)      — 4 nodes, weight 1/4 each  (type (0,1,1))
      C101: (i0±1, j0±1, k0)      — 4 nodes, weight 1/4 each  (type (1,1,0))
      C110: (i0±1, j0, k0±1)      — 4 nodes, weight 1/4 each  (type (1,0,1))

    Each source node contributes M_P[Hx_dof] = weight / vol_P(node), and the
    total RHS is b = iω C_PR M_P.

    Parameters
    ----------
    grid  : LebedevGrid3D
    C_PR  : sparse matrix, shape (3·N_R, 3·N_P) — curl operator P→R
    omega : float — angular frequency [rad/s]

    Returns
    -------
    b : ndarray, shape (3·N_R,), complex — right-hand side vector
    """
    Mx, My, Mz = grid.Mx, grid.My, grid.Mz
    i0 = Mx // 2
    j0 = My // 2
    k_even = np.arange(0, Mz + 1, 2)
    k0 = int(k_even[np.argmin(np.abs(grid.z[k_even]))])

    def vol_p(i, j, k):
        dx = float(grid.x[min(i + 1, Mx)] - grid.x[max(i - 1, 0)])
        dy = float(grid.y[min(j + 1, My)] - grid.y[max(j - 1, 0)])
        dz = float(grid.z[min(k + 1, Mz)] - grid.z[max(k - 1, 0)])
        return abs(dx * dy * dz)

    M_P_vec = np.zeros(3 * grid.N_P, dtype=complex)
    N_P = grid.N_P

    # Hx DOF offset = 0 * N_P
    def add_hx(i, j, k, weight):
        seq = int(grid.P_idx[i, j, k])
        if seq < 0:
            raise ValueError(f"P-node ({i},{j},{k}) not found in P-grid")
        vol = vol_p(i, j, k)
        M_P_vec[0 * N_P + seq] += weight / vol

    # C011: single node at (i0, j0, k0), weight = 1
    add_hx(i0, j0, k0, 1.0)

    # C000: type (0,1,1) → (i0, j0±1, k0±1), weight = 1/4 each
    for dj in (+1, -1):
        for dk in (+1, -1):
            add_hx(i0, j0 + dj, k0 + dk, 0.25)

    # C101: type (1,1,0) → (i0±1, j0±1, k0), weight = 1/4 each
    for di in (+1, -1):
        for dj in (+1, -1):
            add_hx(i0 + di, j0 + dj, k0, 0.25)

    # C110: type (1,0,1) → (i0±1, j0, k0±1), weight = 1/4 each
    for di in (+1, -1):
        for dk in (+1, -1):
            add_hx(i0 + di, j0, k0 + dk, 0.25)

    print(f"    Multi-cl Hx source: ({i0},{j0},{k0}), "
          f"pos=({grid.x[i0]:.4f},{grid.y[j0]:.4f},{grid.z[k0]:.4f}), "
          f"nnz={np.count_nonzero(M_P_vec)}", flush=True)

    return 1j * omega * (C_PR @ M_P_vec)


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
    Extract one component of **B** along the specified axis using the proper
    Lebedev multi-cluster average.

    For comp=0 (Bx) on the z-axis:
    - C011 owns Hx at type (0,0,0): node (i0, j0, k_r) for each receiver k_r
    - C000 owns Hx at type (0,1,1): nodes (i0, j0±1, k_r±1) — average of 4
    - C101 owns Hx at type (1,1,0): nodes (i0±1, j0±1, k_r) — average of 4
    - C110 owns Hx at type (1,0,1): nodes (i0±1, j0, k_r±1) — average of 4

    The four cluster contributions are averaged to give the Lebedev value.

    Receiver positions: all even-k P-nodes with (i==i0, j==j0) — the
    z-axis C011 nodes (same as the old single-cluster extraction).

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
    N_P = grid.N_P

    def b_at(i, j, k):
        if not (0 <= i <= Mx and 0 <= j <= My and 0 <= k <= Mz):
            return 0j
        seq = int(grid.P_idx[i, j, k])
        if seq < 0:
            return 0j
        return complex(B_vec[comp * N_P + seq])

    # Receiver positions: type-(0,0,0) nodes on z-axis = (i0, j0, k) for k even
    coords_list = []
    values_list = []

    for k in range(0, Mz + 1, 2):
        seq = int(grid.P_idx[i0, j0, k])
        if seq < 0:
            continue

        # C011 contribution: (i0, j0, k)  — weight 1
        val_C011 = b_at(i0, j0, k)

        # C000 contribution: avg of (i0, j0±1, k±1) — type (0,1,1)
        vals_C000 = [b_at(i0, j0 + dj, k + dk) for dj in (+1, -1) for dk in (+1, -1)]
        val_C000 = complex(np.mean(vals_C000))

        # C101 contribution: avg of (i0±1, j0±1, k) — type (1,1,0)
        vals_C101 = [b_at(i0 + di, j0 + dj, k) for di in (+1, -1) for dj in (+1, -1)]
        val_C101 = complex(np.mean(vals_C101))

        # C110 contribution: avg of (i0±1, j0, k±1) — type (1,0,1)
        vals_C110 = [b_at(i0 + di, j0, k + dk) for di in (+1, -1) for dk in (+1, -1)]
        val_C110 = complex(np.mean(vals_C110))

        b_avg = (val_C011 + val_C000 + val_C101 + val_C110) / 4.0

        coords_list.append(float(grid.z[k]))
        values_list.append(b_avg)

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
    Compute the correct Lebedev-averaged B_{comp} on the z-axis using
    four separate cluster solutions (DDH03's proper multi-cluster approach).

    Each cluster's B-field is read at its OWN native node type for *comp*,
    and the four native values are then averaged.

    This is equivalent to DDH03's procedure:
    (a) solve four separate systems, one per cluster source, using the
        coupled matrix A but each cluster's own RHS;
    (b) extract B at each cluster's native nodes;
    (c) average the four contributions.

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
    N_P = grid.N_P

    def b_at(B_vec, i, j, k):
        if not (0 <= i <= Mx and 0 <= j <= My and 0 <= k <= Mz):
            return 0j
        seq = int(grid.P_idx[i, j, k])
        return 0j if seq < 0 else complex(B_vec[comp * N_P + seq])

    # Receiver positions: on-axis P-nodes of type (0,0,0), i.e. k even
    z_vals_list: list[float] = []
    B_avg_list: list[complex] = []

    for k_r in range(0, Mz + 1, 2):
        if grid.P_idx[i0, j0, k_r] < 0:
            continue

        contributions = []
        for c in (C000, C101, C110, C011):
            B_c = B_clusters[c]
            t1, t2, t3 = _native_type_for_h_cluster_comp(c, comp)
            # Build offset lists: 0 if parity matches i0/j0/k_r, else ±1
            di_vals = [0] if t1 == i0 % 2 else [+1, -1]
            dj_vals = [0] if t2 == j0 % 2 else [+1, -1]
            dk_vals = [0] if t3 == k_r % 2 else [+1, -1]
            vals = [b_at(B_c, i0 + di, j0 + dj, k_r + dk)
                    for di in di_vals for dj in dj_vals for dk in dk_vals]
            contributions.append(complex(np.mean(vals)))

        z_vals_list.append(float(grid.z[k_r]))
        B_avg_list.append(complex(np.mean(contributions)))

    if not z_vals_list:
        raise RuntimeError("No on-axis P-nodes found for lebedev_B_on_z_axis.")

    z_vals = np.array(z_vals_list, dtype=float)
    B_avg = np.array(B_avg_list, dtype=complex)
    order = np.argsort(z_vals)
    return z_vals[order], B_avg[order]

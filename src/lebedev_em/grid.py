"""
grid.py — Lebedev staggered 3D grid for the DDH03 scheme.

The Lebedev grid uses a checkerboard partition of a Cartesian 3D grid into
two subgrids P (magnetic) and R (electric) based on the parity of the node
index sum (i+j+k).  The four Yee-like "clusters" (000, 101, 110, 011) tile
the E- and H-field components across these subgrids.  In the isotropic case
the clusters decouple; off-diagonal conductivity tensors introduce coupling.

Reference: Davydycheva, Druskin & Habashy (2003), Geophysics 68(5):1525–1536.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

# ---------------------------------------------------------------------------
# Cluster index constants
# ---------------------------------------------------------------------------
C000 = 0
C101 = 1
C110 = 2
C011 = 3

CLUSTER_LABELS = {C000: "000", C101: "101", C110: "110", C011: "011"}

# ---------------------------------------------------------------------------
# Cluster E-component assignments:
# For R-node of type (i%2, j%2, k%2) the tuple gives the cluster index that
# owns (Ex, Ey, Ez) at that node.  Derived from Table 1 / eq. cluster defs
# in DDH03 §"Connection with the standard Yee scheme".
# ---------------------------------------------------------------------------
_E_CLUSTER_MAP: dict[tuple[int, int, int], tuple[int, int, int]] = {
    (1, 0, 0): (C000, C110, C101),
    (0, 1, 0): (C110, C000, C011),
    (0, 0, 1): (C101, C011, C000),
    (1, 1, 1): (C011, C101, C110),
}

# H-component assignments at P-nodes:
# For P-node of type (i%2, j%2, k%2) the tuple gives the cluster owning (Hx, Hy, Hz).
_H_CLUSTER_MAP: dict[tuple[int, int, int], tuple[int, int, int]] = {
    (0, 0, 0): (C011, C101, C110),
    (0, 1, 1): (C000, C110, C101),
    (1, 0, 1): (C110, C000, C011),
    (1, 1, 0): (C101, C011, C000),
}

# Boundary condition type for each cluster on each face direction.
# 'E' = electric (Dirichlet, E×n=0); 'M' = magnetic (Neumann-type, H×n=0).
# For face in direction d, cluster αβγ gets 'E' if its d-th bit is 0, 'M' if 1.
_CLUSTER_BC: dict[int, dict[str, str]] = {
    C000: {"x": "E", "y": "E", "z": "E"},
    C101: {"x": "M", "y": "E", "z": "M"},
    C110: {"x": "M", "y": "M", "z": "E"},
    C011: {"x": "E", "y": "M", "z": "M"},
}


# ---------------------------------------------------------------------------
# LebedevGrid3D
# ---------------------------------------------------------------------------

class LebedevGrid3D:
    """
    3D Lebedev staggered grid for the DDH03 electromagnetic scheme.

    Parameters
    ----------
    x, y, z : array_like, 1-D
        Coordinate arrays of length Mx+1, My+1, Mz+1 respectively.
        Mx, My, Mz must be **even** positive integers.

    Attributes
    ----------
    Mx, My, Mz : int
        Number of grid intervals in each direction (all even).
    Nx, Ny, Nz : int
        Number of grid nodes (= M + 1).
    N_R, N_P : int
        Number of R- (electric) and P- (magnetic) nodes.
    R_idx, P_idx : ndarray, shape (Nx, Ny, Nz), dtype int
        Sequential index of each node in its subgrid; -1 if not in that subgrid.
    R_nodes, P_nodes : ndarray, shape (N_R/P, 3)
        Integer (i,j,k) coordinates of each R/P node in sequential order.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)

        self.Mx = len(self.x) - 1
        self.My = len(self.y) - 1
        self.Mz = len(self.z) - 1

        if self.Mx % 2 != 0:
            raise ValueError(f"Mx={self.Mx} must be even.")
        if self.My % 2 != 0:
            raise ValueError(f"My={self.My} must be even.")
        if self.Mz % 2 != 0:
            raise ValueError(f"Mz={self.Mz} must be even.")

        self.Nx = self.Mx + 1
        self.Ny = self.My + 1
        self.Nz = self.Mz + 1

        self._build_subgrid_indices()
        self._build_cluster_maps()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_subgrid_indices(self) -> None:
        """Build R_idx, P_idx arrays and R_nodes, P_nodes coordinate lists."""
        ii, jj, kk = np.mgrid[0 : self.Nx, 0 : self.Ny, 0 : self.Nz]
        parity = (ii + jj + kk) % 2  # 0 → P, 1 → R

        self.R_mask = parity == 1
        self.P_mask = parity == 0

        self.R_nodes = np.argwhere(self.R_mask)  # shape (N_R, 3)
        self.P_nodes = np.argwhere(self.P_mask)  # shape (N_P, 3)
        self.N_R = len(self.R_nodes)
        self.N_P = len(self.P_nodes)

        self.R_idx = np.full((self.Nx, self.Ny, self.Nz), -1, dtype=np.int64)
        self.P_idx = np.full((self.Nx, self.Ny, self.Nz), -1, dtype=np.int64)

        for seq, (i, j, k) in enumerate(self.R_nodes):
            self.R_idx[i, j, k] = seq
        for seq, (i, j, k) in enumerate(self.P_nodes):
            self.P_idx[i, j, k] = seq

    def _build_cluster_maps(self) -> None:
        """
        Build per-node cluster membership arrays.

        self.E_cluster[seq, comp] → cluster index for E_comp at R-node seq.
        self.H_cluster[seq, comp] → cluster index for H_comp at P-node seq.
        """
        self.E_cluster = np.empty((self.N_R, 3), dtype=np.int8)
        for seq, (i, j, k) in enumerate(self.R_nodes):
            t = (int(i % 2), int(j % 2), int(k % 2))
            self.E_cluster[seq] = _E_CLUSTER_MAP[t]

        self.H_cluster = np.empty((self.N_P, 3), dtype=np.int8)
        for seq, (i, j, k) in enumerate(self.P_nodes):
            t = (int(i % 2), int(j % 2), int(k % 2))
            self.H_cluster[seq] = _H_CLUSTER_MAP[t]

    # ------------------------------------------------------------------
    # Public coordinate helpers
    # ------------------------------------------------------------------

    def node_xyz(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Physical coordinates of grid node (i, j, k)."""
        return float(self.x[i]), float(self.y[j]), float(self.z[k])

    def r_node_xyz(self, seq: int) -> Tuple[float, float, float]:
        """Physical coordinates of R-node with sequential index *seq*."""
        i, j, k = self.R_nodes[seq]
        return float(self.x[i]), float(self.y[j]), float(self.z[k])

    def p_node_xyz(self, seq: int) -> Tuple[float, float, float]:
        """Physical coordinates of P-node with sequential index *seq*."""
        i, j, k = self.P_nodes[seq]
        return float(self.x[i]), float(self.y[j]), float(self.z[k])

    # ------------------------------------------------------------------
    # Grid-step helpers (for building FD operators)
    # ------------------------------------------------------------------

    def dx_P(self, i: int) -> float:
        """Denominator for centered difference at node (i,·,·): x_{i+1} − x_{i−1}."""
        return float(self.x[i + 1] - self.x[i - 1])

    def dy_P(self, j: int) -> float:
        """Denominator for centered difference at node (·,j,·): y_{j+1} − y_{j−1}."""
        return float(self.y[j + 1] - self.y[j - 1])

    def dz_P(self, k: int) -> float:
        """Denominator for centered difference at node (·,·,k): z_{k+1} − z_{k−1}."""
        return float(self.z[k + 1] - self.z[k - 1])

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    def is_r_boundary(self, i: int, j: int, k: int) -> bool:
        """True if R-node (i,j,k) is on the boundary of the computational domain."""
        return (
            i == 0 or i == self.Mx
            or j == 0 or j == self.My
            or k == 0 or k == self.Mz
        )

    def is_p_boundary(self, i: int, j: int, k: int) -> bool:
        """True if P-node (i,j,k) is on the boundary."""
        return (
            i == 0 or i == self.Mx
            or j == 0 or j == self.My
            or k == 0 or k == self.Mz
        )

    def r_boundary_mask(self) -> np.ndarray:
        """Boolean array of shape (N_R,): True for boundary R-nodes."""
        mask = np.zeros(self.N_R, dtype=bool)
        for seq, (i, j, k) in enumerate(self.R_nodes):
            if self.is_r_boundary(i, j, k):
                mask[seq] = True
        return mask

    # ------------------------------------------------------------------
    # Cluster BC query
    # ------------------------------------------------------------------

    def cluster_bc(self, cluster: int, face_dir: str) -> str:
        """
        Return 'E' (electric/Dirichlet) or 'M' (magnetic/Neumann) boundary
        condition for *cluster* on the face perpendicular to *face_dir* ∈ {'x','y','z'}.
        """
        return _CLUSTER_BC[cluster][face_dir]

    def e_component_cluster(self, i: int, j: int, k: int, comp: int) -> int:
        """
        Cluster index (0–3) that owns E_comp at R-node (i,j,k).
        comp: 0=Ex, 1=Ey, 2=Ez.
        """
        t = (int(i % 2), int(j % 2), int(k % 2))
        return int(_E_CLUSTER_MAP[t][comp])

    # ------------------------------------------------------------------
    # Cluster-specific node lists (useful for assembling per-cluster systems)
    # ------------------------------------------------------------------

    def r_nodes_for_cluster_component(
        self, cluster: int, comp: int
    ) -> np.ndarray:
        """
        Return sequential R-node indices where E_comp belongs to *cluster*.

        Parameters
        ----------
        cluster : int
            Cluster constant (C000, C101, C110, or C011).
        comp : int
            Field component 0=Ex, 1=Ey, 2=Ez.

        Returns
        -------
        ndarray of int, shape (N,)
            Sorted sequential R-node indices.
        """
        mask = self.E_cluster[:, comp] == cluster
        return np.where(mask)[0]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LebedevGrid3D(Mx={self.Mx}, My={self.My}, Mz={self.Mz}, "
            f"N_R={self.N_R}, N_P={self.N_P})"
        )

    def summary(self) -> str:
        """Multi-line human-readable summary of the grid."""
        lines = [
            "LebedevGrid3D",
            f"  Grid size    : {self.Nx} × {self.Ny} × {self.Nz}  (Mx={self.Mx}, My={self.My}, Mz={self.Mz})",
            f"  R-nodes (E)  : {self.N_R}  ×3 = {3 * self.N_R} unknowns",
            f"  P-nodes (H)  : {self.N_P}  ×3 = {3 * self.N_P} unknowns",
            f"  x range      : [{self.x[0]:.4g}, {self.x[-1]:.4g}]",
            f"  y range      : [{self.y[0]:.4g}, {self.y[-1]:.4g}]",
            f"  z range      : [{self.z[0]:.4g}, {self.z[-1]:.4g}]",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grid factory functions
# ---------------------------------------------------------------------------

def uniform_grid(
    Mx: int,
    My: int,
    Mz: int,
    Lx: float,
    Ly: float,
    Lz: float,
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
) -> LebedevGrid3D:
    """
    Create a uniform Cartesian Lebedev grid.

    Parameters
    ----------
    Mx, My, Mz : int
        Number of intervals (must be even).
    Lx, Ly, Lz : float
        Domain lengths.
    x0, y0, z0 : float
        Domain origin (default 0).
    """
    x = np.linspace(x0, x0 + Lx, Mx + 1)
    y = np.linspace(y0, y0 + Ly, My + 1)
    z = np.linspace(z0, z0 + Lz, Mz + 1)
    return LebedevGrid3D(x, y, z)


def symmetric_uniform_grid(
    Mx: int,
    My: int,
    Mz: int,
    Lx: float,
    Ly: float,
    Lz: float,
) -> LebedevGrid3D:
    """
    Uniform grid centred at the origin: x ∈ [−Lx/2, Lx/2], etc.
    (Useful for dipole problems where the source is at the origin.)
    """
    return uniform_grid(Mx, My, Mz, Lx, Ly, Lz, -Lx / 2, -Ly / 2, -Lz / 2)


def optimal_geometric_1d(
    k: int,
    h_min: float,
    L: float,
    gamma: float = 1.0 / np.sqrt(2),
) -> np.ndarray:
    """
    Build the **full interleaved half-axis** optimal geometric grid (DDH03).

    The DDH03 optimal grid interleaves primary nodes x₁,…,x_k and dual nodes
    x̂₁,…,x̂_k on the positive half-axis, with x̂₀ = 0 as the origin:

        [x̂₀=0,  x̂₁,  x₁,  x̂₂,  x₂,  …,  x̂_k,  x_k]

    Primary spacings (DDH03 eq. 22):
        hᵢ = h_min · αⁱ⁻¹,   α = exp(γπ/√k),   γ = 1/√2 optimal

    Dual spacings (DDH03 §"Optimal geometric grids"):
        ĥ₁ = h_min / (1 + √α)
        ĥᵢ = hᵢ / √α   for i ≥ 2

    The symmetric full grid is built by `symmetric_optimal_grid`, which
    mirrors this half-axis and produces Mx = 4k nodes (DDH03 notation).

    Parameters
    ----------
    k : int
        Number of primary (and dual) steps on the positive half-axis.
        The full symmetric grid will have Mx = 4k intervals.
    h_min : float
        Minimum primary grid spacing (= h in DDH03 notation).
    L : float
        Approximate domain half-length (informational only; not enforced).
    gamma : float
        Progression exponent.  1/√2 is optimal for induction problems (DDH03).

    Returns
    -------
    x_half : ndarray, shape (2k+1,)
        Interleaved half-axis: x̂₀=0, x̂₁, x₁, x̂₂, x₂, …, x̂_k, x_k.
        Even indices 2i   → dual   node x̂ᵢ  (P-type in 3D when i+j+k even)
        Odd  indices 2i+1 → primary node xᵢ₊₁ (R-type in 3D when i+j+k odd)
    """
    alpha = np.exp(gamma * np.pi / np.sqrt(k))
    sqrt_alpha = np.sqrt(alpha)

    # Primary spacings: h[i] = h_min * alpha^i  (i = 0 … k-1)
    h_primary = h_min * alpha ** np.arange(k)
    # Primary node positions: x_1, …, x_k
    x_primary = np.cumsum(h_primary)   # shape (k,)

    # Dual spacings: h_hat[0] = h_min/(1+√α); h_hat[i] = h_primary[i]/√α  (i≥1)
    h_dual = h_primary / sqrt_alpha          # shape (k,); overwrite [0] below
    h_dual[0] = h_min / (1.0 + sqrt_alpha)
    # Dual node positions: x̂_1, …, x̂_k  (x̂_0 = 0 is the origin)
    x_dual = np.cumsum(h_dual)              # shape (k,)

    # Interleave: [0, x̂_1, x_1, x̂_2, x_2, …, x̂_k, x_k]
    x_half = np.empty(2 * k + 1)
    x_half[0] = 0.0
    for i in range(k):
        x_half[2 * i + 1] = x_dual[i]     # x̂_{i+1}
        x_half[2 * i + 2] = x_primary[i]  # x_{i+1}

    return x_half


def hybrid_axial_grid(
    z_tool_min: float,
    z_tool_max: float,
    n_inner: int,
    k_outer: int,
    gamma: float = 1.0 / np.sqrt(2),
) -> np.ndarray:
    """
    Build a hybrid z-axis for the DDH03 logging geometry (DDH03 Fig. 5/6).

    The DDH03 paper specifies: *"along the z-axis we set the grid to be
    equidistant between the transmitter and the receiver, and optimal
    geometric otherwise."*

    This function implements exactly that:

    **Inner zone** ``[z_tool_min, z_tool_max]``:
        Equidistant with spacing ``dz = (z_tool_max - z_tool_min) / n_inner``.
        Resolves the tool response uniformly between TX and RX.

    **Outer zones** (below z_tool_min and above z_tool_max):
        Optimal geometric with ``k_outer`` steps on each side, using the same
        formula as the transverse grid (DDH03 eq. 22):

            α = exp(γπ / √k_outer),   γ = 1/√2  (EM optimal)

        The first outer step equals ``dz`` (matching the inner spacing at the
        junction), then grows as α each step.  With k_outer = 8–12 steps the
        boundary is pushed ≫ one skin depth away, making Dirichlet BC
        reflections negligible.

    Total nodes: ``4·k_outer + n_inner + 1``.  ``Mz = 4·k_outer + n_inner``
    must be even, which requires ``n_inner`` to be even (since 4·k_outer is
    always even).  A ``ValueError`` is raised otherwise.

    Parameters
    ----------
    z_tool_min : float
        Lower boundary of the equidistant inner zone (e.g. transmitter z).
    z_tool_max : float
        Upper boundary of the equidistant inner zone (e.g. furthest receiver z).
    n_inner : int
        Number of **even** equidistant steps in the inner zone.
        ``dz = (z_tool_max - z_tool_min) / n_inner``.
    k_outer : int
        Number of optimal geometric steps on each side outside the inner zone.
        Typical values: 8–15.  More steps → larger domain → smaller BC error.
    gamma : float
        Geometric progression exponent.  1/√2 is optimal for EM (DDH03).

    Returns
    -------
    z : ndarray, shape (4·k_outer + n_inner + 1,)
        Full z-axis, with Mz = 4·k_outer + n_inner (even).
        ``z_tool_min`` and ``z_tool_max`` are exact nodes in the array.

    Examples
    --------
    2C-40 sonde (TX at z=0, RX at z=1.016 m, dz=0.0508 m = 2 in):

    >>> z = hybrid_axial_grid(0.0, 1.016, 20, 10)
    >>> z[0], z[-1]   # domain extent
    (-12.7 m,  13.7 m)  # ≈ ±13 m  (actual values depend on α)
    """
    if z_tool_max <= z_tool_min:
        raise ValueError(
            f"z_tool_max={z_tool_max} must be greater than z_tool_min={z_tool_min}."
        )
    if n_inner < 1:
        raise ValueError(f"n_inner={n_inner} must be ≥ 1.")
    if n_inner % 2 != 0:
        raise ValueError(
            f"n_inner={n_inner} must be even so that Mz = 4·k_outer + n_inner "
            f"is even (required by the Lebedev grid)."
        )
    if k_outer < 1:
        raise ValueError(f"k_outer={k_outer} must be ≥ 1.")

    dz = (z_tool_max - z_tool_min) / n_inner

    # ── Inner zone: equidistant ───────────────────────────────────────────────
    z_inner = np.linspace(z_tool_min, z_tool_max, n_inner + 1)

    # ── Outer zone offsets (from junction point, same as transverse grid) ─────
    # optimal_geometric_1d returns [0, x̂₁, x₁, x̂₂, x₂, …, x̂_k, x_k]
    # We take everything after 0 → 2·k_outer strictly positive offsets.
    outer_half = optimal_geometric_1d(k_outer, dz, L=1.0, gamma=gamma)
    outer_offsets = outer_half[1:]   # shape (2·k_outer,), all > 0

    # ── Lower outer zone: z_tool_min − offsets (reversed, so ascending) ───────
    z_lower = z_tool_min - outer_offsets[::-1]   # most negative → just below z_tool_min

    # ── Upper outer zone: z_tool_max + offsets (ascending) ────────────────────
    z_upper = z_tool_max + outer_offsets

    # ── Concatenate ───────────────────────────────────────────────────────────
    z_full = np.concatenate([z_lower, z_inner, z_upper])

    # ── Sanity check: Mz must be even ─────────────────────────────────────────
    Mz = len(z_full) - 1
    assert Mz % 2 == 0, (
        f"Internal error: Mz={Mz} is odd.  "
        f"(4·{k_outer} + {n_inner} = {4*k_outer+n_inner})"
    )

    return z_full


def hybrid_axial_grid_domain(
    z_tool_min: float,
    z_tool_max: float,
    n_inner: int,
    k_outer: int,
    gamma: float = 1.0 / np.sqrt(2),
) -> tuple:
    """
    Return ``(z_min, z_max, dz, x_outer_max)`` for a hybrid axial grid
    without constructing the full array — useful for quick design checks.

    Parameters are the same as :func:`hybrid_axial_grid`.

    Returns
    -------
    z_min : float   Minimum z-coordinate (lower domain boundary).
    z_max : float   Maximum z-coordinate (upper domain boundary).
    dz    : float   Inner zone equidistant spacing.
    x_outer_max : float  Extent of outer zone on each side (≈ domain half-size
                         relative to the tool zone centre).
    """
    dz = (z_tool_max - z_tool_min) / n_inner
    outer_half = optimal_geometric_1d(k_outer, dz, L=1.0, gamma=gamma)
    outer_extent = float(outer_half[-1])  # x_k: last primary node position
    return (
        z_tool_min - outer_extent,
        z_tool_max + outer_extent,
        dz,
        outer_extent,
    )


def _k_for_domain(h_min: float, L: float, gamma: float = 1.0 / np.sqrt(2)) -> int:
    """Return the smallest k so that optimal_geometric_1d reaches at least L."""
    for k in range(1, 200):
        x_half = optimal_geometric_1d(k, h_min, L, gamma)
        if x_half[-1] >= L:
            return k
    raise ValueError(f"Cannot reach L={L} with h_min={h_min}, gamma={gamma} in 200 steps.")


def symmetric_optimal_grid(
    h_min: float,
    L: float,
    z: np.ndarray,
    gamma: float = 1.0 / np.sqrt(2),
    k: int | None = None,
) -> LebedevGrid3D:
    """
    Build a 3D Lebedev grid with **optimal geometric** x- and y-grids (both
    symmetric about the origin) and a user-supplied z-grid.

    This matches the DDH03 logging geometry where sources/receivers lie on the
    z-axis and optimal grids are used in the transverse directions.

    The transverse grids use the full interleaved primary+dual node positions
    from `optimal_geometric_1d`, giving Mx = My = 4k (DDH03 §Implementation).

    Parameters
    ----------
    h_min : float
        Minimum transverse grid step (= h in DDH03 notation).
    L : float
        Target transverse domain half-length.  k is chosen automatically as
        the smallest integer such that the grid reaches at least L.
    z : array_like
        z-coordinate array.  Length must be odd (so Mz = len(z)-1 is even).
    gamma : float
        Optimal grid progression exponent (default 1/√2).
    k : int, optional
        Override the auto-computed k.  Use only if you want to fix the number
        of transverse steps explicitly.
    """
    if k is None:
        k = _k_for_domain(h_min, L, gamma)
    x_half = optimal_geometric_1d(k, h_min, L, gamma)
    # Mirror: [−x_half[-1], …, −x_half[1], 0, x_half[1], …, x_half[-1]]
    # x_half has 2k+1 entries → full symmetric grid has 4k+1 nodes → Mx = 4k ✓
    x_full = np.concatenate([-x_half[::-1], x_half[1:]])
    y_full = x_full.copy()

    z = np.asarray(z, dtype=float)
    Mz = len(z) - 1
    if Mz % 2 != 0:
        raise ValueError(f"z array length must be odd (Mz={Mz} must be even).")

    return LebedevGrid3D(x_full, y_full, z)

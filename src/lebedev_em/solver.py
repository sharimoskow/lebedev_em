"""
solver.py — Maxwell system assembly and Lebedev 4-cluster solver.

Two solve modes are available:

  solve_coupled  (default, recommended):
    Assembles a single combined RHS from all four cluster source distributions,
    applies component-aware boundary conditions (each DOF's Dirichlet rule is
    determined by the cluster that owns that component at its node type), and
    solves the full 3N_R × 3N_R system ONCE.  Correctly handles anisotropic
    conductivity tensors (e.g. from homogenisation) where off-diagonal σ terms
    couple DOFs belonging to different clusters at the same spatial R-node.
    For isotropic (diagonal) σ the coupled system decouples exactly into the
    four independent Yee sub-problems, so this is a strict generalisation.
    Cost: one direct solve (vs. four for solve_clustered) → ~4× faster.

  solve_clustered  (legacy, isotropic-only):
    Solves the four Lebedev clusters independently and combines them into a
    composite field (the DDH03 interpolate-then-average step is done in
    postprocess.lebedev_E_at_point).  Correct only
    when σ, μ, ε are diagonal (isotropic); underestimates inter-cluster coupling
    for anisotropic media.  Kept for comparison and regression testing.

The public `solve` method dispatches to solve_coupled by default.

Reference: Davydycheva, Druskin & Habashy (2003), Geophysics 68(5):1525–1536.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import LebedevGrid3D, C000, C101, C110, C011
from .media import EMMedia
from .operators import (
    build_curl_RE,
    build_curl_PR,
    build_system_matrix,
    apply_electric_bc,
    electric_bc_dofs,
)
from .sources import point_dipole_rhs


# ---------------------------------------------------------------------------
# Boundary condition helpers
# ---------------------------------------------------------------------------

def _cluster_bc_dofs(grid: LebedevGrid3D, cluster: int) -> np.ndarray:
    """
    Return DOF indices that receive the **electric** BC E×n=0 for *cluster*.

    DDH03 mixed-BC rule (eq. 6 + cluster analysis):
        Cluster 000 → E-BC on all 6 faces.
        Cluster 101 → E-BC on y-faces only; M-BC on x,z.
        Cluster 110 → E-BC on z-faces only; M-BC on x,y.
        Cluster 011 → E-BC on x-faces only; M-BC on y,z.

    Key constraint: on each face, the NORMAL E-component belongs to an M-BC
    cluster and must NOT be zeroed.  Only the two TANGENTIAL components (which
    belong to E-BC clusters) receive Dirichlet zero.  This is consistent with
    E×n=0 (tangential E = 0) and leaves the normal E free for the M-BC clusters.

    The M-BC condition H×n=0 (tangential H = 0 at the boundary) is enforced
    through the curl operator: build_curl_RE zeroes the ENTIRE row of every
    tangential H component at boundary P-nodes (normal H components keep
    their stencils), so tangential H vanishes identically on all six faces —
    implementing H×n = 0 and, together with the Dirichlet set below, the
    full DDH03 combined BC eq. 6.
    """
    from .grid import _CLUSTER_BC

    bc = _CLUSTER_BC[cluster]  # {'x': 'E'/'M', 'y': ..., 'z': ...}

    # Map face direction index → (face_dir_key, normal_comp_index)
    # On an x-face the NORMAL component is Ex (comp 0); tangential = Ey(1), Ez(2).
    # The DDH03 mixed-BC rule assigns the normal component to an M-BC cluster,
    # so it must NOT be zeroed.  Only the two tangential components get Dirichlet.
    _face_info = [
        ("x", 0),   # x-face: normal comp = 0 (Ex)
        ("y", 1),   # y-face: normal comp = 1 (Ey)
        ("z", 2),   # z-face: normal comp = 2 (Ez)
    ]

    dofs = set()  # use set to avoid duplicating DOFs at edges/corners
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        on_face = [
            (i == 0 or i == grid.Mx),
            (j == 0 or j == grid.My),
            (k == 0 or k == grid.Mz),
        ]

        for d, (face_dir, normal_comp) in enumerate(_face_info):
            if on_face[d] and bc[face_dir] == "E":
                # Zero only the TWO tangential components at this face.
                for comp in range(3):
                    if comp != normal_comp:
                        dofs.add(comp * grid.N_R + seq)

    return np.array(sorted(dofs), dtype=np.int64)


def _component_aware_bc_dofs(grid: LebedevGrid3D) -> np.ndarray:
    """
    Return DOF indices for the coupled single-solve boundary conditions.

    For the fully coupled system each DOF (comp α, R-node seq) belongs to
    the cluster c = _E_CLUSTER_MAP[node_type(seq)][α].  We apply *that*
    cluster's E-BC rule to determine whether this DOF is Dirichlet-zeroed:

        If cluster c has E-BC on face direction d  AND
           comp α is tangential to face d (i.e. comp ≠ normal of face d):
        → add (α, seq) to the Dirichlet set.

    This is exactly the per-cluster BC applied component-by-component, which
    is the correct DDH03 mixed BC for the coupled system.

    Properties:
    • For isotropic media: the coupled system decouples into four independent
      Yee problems, each with its own cluster-specific BCs — identical to
      calling _cluster_bc_dofs for each cluster on its own DOF subset.
    • For anisotropic media: off-diagonal σ couples clusters; these BCs keep
      each component's Dirichlet constraint tied to its owning cluster while
      allowing the inter-cluster coupling through σ to act in the interior.
    """
    from .grid import _E_CLUSTER_MAP, _CLUSTER_BC

    _face_info = [("x", 0), ("y", 1), ("z", 2)]

    dofs: set[int] = set()
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        node_type = (i % 2, j % 2, k % 2)
        owner_clusters = _E_CLUSTER_MAP[node_type]   # (c_x, c_y, c_z)

        on_face = [
            (i == 0 or i == grid.Mx),
            (j == 0 or j == grid.My),
            (k == 0 or k == grid.Mz),
        ]

        for comp in range(3):
            c_owner = int(owner_clusters[comp])
            bc = _CLUSTER_BC[c_owner]

            for d, (face_dir, normal_comp) in enumerate(_face_info):
                if on_face[d] and bc[face_dir] == "E" and comp != normal_comp:
                    dofs.add(comp * grid.N_R + seq)

    return np.array(sorted(dofs), dtype=np.int64)


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

class LebedevMaxwellSolver:
    """
    Frequency-domain Maxwell solver using the Lebedev 4-cluster scheme.

    Parameters
    ----------
    grid : LebedevGrid3D
    media : EMMedia
    omega : float
        Angular frequency ω = 2π f.
    """

    def __init__(
        self,
        grid: LebedevGrid3D,
        media: EMMedia,
        omega: float,
    ) -> None:
        self.grid = grid
        self.media = media
        self.omega = float(omega)

        # Build the operators once — they only depend on the grid
        self._C_RE = build_curl_RE(grid)
        self._C_PR = build_curl_PR(grid)

        # Build material matrices and the core system matrix
        inv_mu_P    = media.inv_mu_matrix()
        sigma_dot_R = media.sigma_dot_matrix(omega)
        self._A = build_system_matrix(
            grid, self._C_RE, self._C_PR, inv_mu_P, sigma_dot_R, omega
        )

    # ------------------------------------------------------------------
    def solve_coupled(
        self,
        x0: float,
        y0: float,
        z0: float,
        dipole_comp: int,
        moment: float = 1.0,
    ) -> dict:
        """
        Fully coupled single-solve (correct for anisotropic media).

        Combines all four cluster source distributions into one RHS, applies
        component-aware boundary conditions (each DOF's Dirichlet rule follows
        its owning cluster), and calls the direct sparse solver ONCE on the
        full 3·N_R system.

        For isotropic (diagonal) σ this is mathematically equivalent to the
        four-cluster average (the inter-cluster coupling vanishes and the
        system decouples).  For anisotropic σ this is the correct DDH03
        coupled solution — the clustered approach is not.

        The returned 'E_c' dict maps every cluster key to the *same* E-field
        vector.  Callers that pass E_c to ``lebedev_E_at_point`` or
        ``interpolate_cluster_E`` will receive the correct Lebedev-averaged
        field because those functions interpolate each cluster from its own
        native sub-grid of the shared E vector.

        Returns
        -------
        result : dict
            'E_avg' : ndarray (3·N_R,) — the single coupled solution.
            'E_c'   : dict {cluster → ndarray (3·N_R,)} — same E for each key.
            'rhs'   : dict {cluster → ndarray} — per-cluster RHS vectors.
        """
        grid = self.grid
        rhs_all = point_dipole_rhs(grid, x0, y0, z0, dipole_comp, self.omega, moment)

        # Combined RHS: each DOF receives a contribution from exactly ONE cluster
        # (the one that owns that component at that node type), so there is no
        # double-counting.
        b_combined = np.zeros(3 * grid.N_R, dtype=complex)
        for c in (C000, C101, C110, C011):
            b_combined += rhs_all[c]

        bc_dofs = _component_aware_bc_dofs(grid)
        A_bc, b_bc = apply_electric_bc(self._A.copy(), b_combined, bc_dofs)
        E = spla.spsolve(A_bc, b_bc)

        # Return the same E vector under every cluster key so that downstream
        # postprocessing (lebedev_E_at_point etc.) reads the correct native
        # sub-grid for each cluster.
        E_clusters = {c: E for c in (C000, C101, C110, C011)}
        return {
            "E_avg": E,
            "E_c": E_clusters,
            "rhs": rhs_all,
        }

    # ------------------------------------------------------------------
    def solve_clustered(
        self,
        x0: float,
        y0: float,
        z0: float,
        dipole_comp: int,
        moment: float = 1.0,
    ) -> dict:
        """
        Legacy four-cluster independent solve (isotropic media only).

        .. warning::
            Valid only when every material tensor is diagonal.  With
            off-diagonal σ (tilted anisotropy, homogenized tilted/curved
            interfaces) the clusters couple and this procedure under-counts
            cross-components; a ``UserWarning`` is emitted in that case —
            use ``solve_coupled`` (the default of ``solve``) instead.

        Solves the Maxwell system four times, once per Lebedev cluster, each
        with its own cluster-specific mixed E/M boundary conditions, and
        returns the arithmetic average of the four solutions.

        Correct only when σ, μ, ε are diagonal (isotropic).  For anisotropic
        σ the clusters are coupled through off-diagonal entries and this method
        gives an incorrect result; use solve_coupled instead.

        Returns
        -------
        result : dict
            'E_avg' : ndarray (3·N_R,) — composite field: each DOF carries its
                      owning cluster's solution value (same convention as
                      solve_coupled).  This is NOT the interpolated Lebedev
                      4-cluster average; use postprocess.lebedev_E_at_point
                      with 'E_c' for that.
            'E_c'   : dict {cluster → ndarray (3·N_R,)} — per-cluster solutions.
            'rhs'   : dict {cluster → ndarray} — per-cluster RHS vectors.
        """
        if getattr(self.media, "has_offdiagonal_sigma", False):
            import warnings as _warnings
            _warnings.warn(
                "solve_clustered called on media with off-diagonal sigma: the "
                "clusters are coupled and this procedure under-counts "
                "cross-components (see tests/test_anisotropic_coupling.py). "
                "Use solve_coupled / solve(method='coupled') instead.",
                UserWarning, stacklevel=2,
            )
        grid = self.grid
        rhs_all = point_dipole_rhs(grid, x0, y0, z0, dipole_comp, self.omega, moment)

        E_clusters: dict[int, np.ndarray] = {}
        for c in (C000, C101, C110, C011):
            bc_dofs = _cluster_bc_dofs(grid, c)
            A_bc, b_bc = apply_electric_bc(self._A.copy(), rhs_all[c].copy(), bc_dofs)
            E_clusters[c] = spla.spsolve(A_bc, b_bc)

        # The four cluster solutions have (numerically) disjoint supports:
        # each DOF is owned by exactly one cluster and, for the isotropic
        # media this method is valid for, only that cluster's solve excites
        # it.  The composite field is therefore their SUM, giving each DOF
        # its owning cluster's value — identical in convention to the single
        # vector returned by solve_coupled.  (A np.mean here would silently
        # scale every DOF by 1/4.)
        E_avg = np.sum(
            np.stack([E_clusters[c] for c in (C000, C101, C110, C011)], axis=0),
            axis=0,
        )
        return {
            "E_avg": E_avg,
            "E_c": E_clusters,
            "rhs": rhs_all,
        }

    # ------------------------------------------------------------------
    def solve(
        self,
        x0: float,
        y0: float,
        z0: float,
        dipole_comp: int,
        moment: float = 1.0,
        method: str = "coupled",
    ) -> dict:
        """
        Solve the Maxwell system for a point-dipole source.

        Parameters
        ----------
        x0, y0, z0 : float   — source location.
        dipole_comp : int    — dipole orientation: 0=x, 1=y, 2=z.
        moment : float       — dipole moment [A·m] (default 1).
        method : str
            'coupled'   (default) — single coupled solve; correct for
                        anisotropic media; ~4× faster than 'clustered'.
            'clustered' — legacy four-cluster average; valid only for
                        isotropic (diagonal) σ, μ, ε.

        Returns
        -------
        result : dict with keys 'E_avg', 'E_c', 'rhs'.
        """
        if method == "coupled":
            return self.solve_coupled(x0, y0, z0, dipole_comp, moment)
        elif method == "clustered":
            return self.solve_clustered(x0, y0, z0, dipole_comp, moment)
        else:
            raise ValueError(f"Unknown method {method!r}; choose 'coupled' or 'clustered'.")

    # ------------------------------------------------------------------
    def get_field_at(
        self,
        result: dict,
        x: float,
        y: float,
        z: float,
        use_average: bool = True,
    ) -> np.ndarray:
        """
        Extract the (Ex, Ey, Ez) field at a physical point (x, y, z) from a
        solve result by reading the nearest R-node.

        This is a raw nearest-node sample of the composite solution: each
        component at that node is the value computed by the single cluster
        that owns it there.  It is NOT the DDH03 interpolate-then-average
        Lebedev field (eq. 7); for the superconvergent averaged value use
        ``postprocess.lebedev_E_at_point(grid, result['E_c'], comp, x, y, z)``.

        Parameters
        ----------
        result : dict
            Output of `solve()`.
        x, y, z : float
            Evaluation point.
        use_average : bool
            If True read the composite 'E_avg' vector; if False return a dict
            of the per-cluster vectors' values at the node.

        Returns
        -------
        E : ndarray, shape (3,) — field (Ex, Ey, Ez) at the nearest R-node.
        """
        grid = self.grid
        i = int(np.argmin(np.abs(grid.x - x)))
        j = int(np.argmin(np.abs(grid.y - y)))
        k = int(np.argmin(np.abs(grid.z - z)))

        # Make sure we're at an R-node (odd index-parity sum).  If the nearest
        # node is a P-node, step one index along one axis, choosing the
        # in-bounds neighbour closest to the requested point.
        if (i + j + k) % 2 == 0:
            best = None
            for d, (idx, arr, M) in enumerate(
                [(i, grid.x, grid.Mx), (j, grid.y, grid.My), (k, grid.z, grid.Mz)]
            ):
                for s in (-1, +1):
                    if 0 <= idx + s <= M:
                        cand = [i, j, k]
                        cand[d] = idx + s
                        dist = np.sqrt(
                            (grid.x[cand[0]] - x) ** 2
                            + (grid.y[cand[1]] - y) ** 2
                            + (grid.z[cand[2]] - z) ** 2
                        )
                        if best is None or dist < best[0]:
                            best = (dist, cand)
            i, j, k = best[1]

        seq = int(grid.R_idx[i, j, k])
        if seq < 0:
            raise ValueError(f"Node ({i},{j},{k}) is not an R-node.")

        if use_average:
            E_vec = result["E_avg"]
            return np.array([
                E_vec[0 * grid.N_R + seq],
                E_vec[1 * grid.N_R + seq],
                E_vec[2 * grid.N_R + seq],
            ])
        else:
            out = {}
            for c in (C000, C101, C110, C011):
                ev = result["E_c"][c]
                out[c] = np.array([
                    ev[0 * grid.N_R + seq],
                    ev[1 * grid.N_R + seq],
                    ev[2 * grid.N_R + seq],
                ])
            return out

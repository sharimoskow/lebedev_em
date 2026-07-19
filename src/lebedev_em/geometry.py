"""
lebedev_em.geometry  —  Geometry primitives for from_geometry_func.

Provides a small library of boundary objects that describe material interfaces
analytically.  A :class:`GeometryStack` combines them into an ``interface_func``
callable ready to pass to :func:`~lebedev_em.media.from_geometry_func`.

Boundary ordering convention
-----------------------------
Boundaries in a :class:`GeometryStack` are listed **innermost first**.  "Inner"
means the boundary that encloses the smallest region (e.g. the borehole wall
before the invasion-zone wall before any dipping-layer planes).  The ordering
determines how sequential Backus averaging is applied when a cell straddles
more than one interface:

    Step 1  Combine the two outermost materials across the outermost boundary.
    Step 2  Combine the result with the next-inner material across the
            next-inner boundary.
    …
    Final   Combine with the innermost material across the innermost boundary.

Example — DDH03 four-region geometry
--------------------------------------
::

    from lebedev_em.geometry import CylindricalBoundary, PlanarBoundary, GeometryStack
    import numpy as np

    N_HAT = np.array([np.sin(np.radians(60)), 0., np.cos(np.radians(60))])
    geo = GeometryStack([
        CylindricalBoundary(radius=0.1),          # borehole wall   (innermost)
        CylindricalBoundary(radius=0.6),          # invasion wall
        PlanarBoundary(n_hat=N_HAT, d=-0.25),     # 60° dipping layer
    ])
    med = from_geometry_func(grid, sigma_func, geo.interface_func)

Multiple dipping planes are simply additional :class:`PlanarBoundary` entries::

    geo = GeometryStack([
        CylindricalBoundary(radius=0.1),
        CylindricalBoundary(radius=0.6),
        PlanarBoundary(n_hat=N_HAT_1, d=D_1),    # upper dipping plane
        PlanarBoundary(n_hat=N_HAT_2, d=D_2),    # lower dipping plane
    ])
"""

from __future__ import annotations
import numpy as np
from typing import Callable


def _min_dist_interval_to_zero(lo: float, hi: float) -> float:
    """Distance from the point 0 to the interval [lo, hi] (0 if it contains 0)."""
    if lo <= 0.0 <= hi:
        return 0.0
    return min(abs(lo), abs(hi))


# ---------------------------------------------------------------------------
# Boundary primitives
# ---------------------------------------------------------------------------

class PlanarBoundary:
    """
    An infinite planar interface defined by  n̂ · x = d.

    Parameters
    ----------
    n_hat : array-like (3,)
        Interface normal (need not be unit; will be normalised).
    d : float
        Signed offset.  Points with  n̂ · x < d  are on the "negative" side
        (by convention, the side with lower σ for a simple two-layer medium).

    Notes
    -----
    The normal returned by :meth:`normal_at` is constant (independent of
    position), which is exact for a truly planar interface.
    """

    def __init__(self, n_hat: "array-like", d: float) -> None:
        n = np.asarray(n_hat, dtype=float).ravel()
        self.n_hat: np.ndarray = n / np.linalg.norm(n)
        self.d: float = float(d)

    def straddles(self,
                  bmin: np.ndarray,
                  bmax: np.ndarray,
                  node: np.ndarray) -> bool:
        """Return True if the dual cell [bmin, bmax] straddles this plane."""
        corners = [
            self.n_hat[0] * cx + self.n_hat[1] * cy + self.n_hat[2] * cz
            for cx in (bmin[0], bmax[0])
            for cy in (bmin[1], bmax[1])
            for cz in (bmin[2], bmax[2])
        ]
        return min(corners) < self.d <= max(corners)

    def normal_at(self, node: np.ndarray) -> np.ndarray:
        """Return the interface unit normal at *node* (constant for a plane)."""
        return self.n_hat.copy()

    def side(self, X, Y, Z) -> np.ndarray:
        """Boolean array: True where n̂·x >= d (the "positive"/outer side)."""
        t = self.n_hat[0] * np.asarray(X) + self.n_hat[1] * np.asarray(Y) \
            + self.n_hat[2] * np.asarray(Z)
        return t >= self.d

    def __repr__(self) -> str:
        return (f"PlanarBoundary(n_hat={np.round(self.n_hat,4).tolist()}, "
                f"d={self.d:.6g})")


class CylindricalBoundary:
    """
    A vertical cylindrical interface defined by  sqrt(x² + y²) = radius.

    The outward-pointing normal at any point on the cylinder is the radial
    unit vector  [x/r, y/r, 0].  At the node (x_node, y_node, z_node) this
    is computed exactly, giving the correct tangent-plane approximation used
    by the nodal homogenization formula.

    Parameters
    ----------
    radius : float
        Cylinder radius [m].
    """

    def __init__(self, radius: float) -> None:
        self.radius: float = float(radius)

    def straddles(self,
                  bmin: np.ndarray,
                  bmax: np.ndarray,
                  node: np.ndarray) -> bool:
        """
        Return True if the dual cell [bmin, bmax] straddles this cylinder.

        The maximum of r(x, y) over the box is attained at a corner (r is
        convex), but the minimum is generally attained in the interior or on
        an edge, so it must be computed as the clamped closest-point distance
        from the cylinder axis to the box footprint.  Using min-over-corners
        misses cells whose closest approach to the axis is interior — e.g. a
        cell containing the whole borehole, or one that spans x = 0 or y = 0
        near the wall.
        """
        r_corners = [
            (cx ** 2 + cy ** 2) ** 0.5
            for cx in (bmin[0], bmax[0])
            for cy in (bmin[1], bmax[1])
        ]
        r_max = max(r_corners)
        # Clamped closest-point distance from the axis (x=y=0) to the box.
        dx = _min_dist_interval_to_zero(float(bmin[0]), float(bmax[0]))
        dy = _min_dist_interval_to_zero(float(bmin[1]), float(bmax[1]))
        r_min = (dx ** 2 + dy ** 2) ** 0.5
        return r_min < self.radius <= r_max

    def normal_at(self, node: np.ndarray) -> np.ndarray:
        """Return the outward radial unit normal at *node*."""
        r = float((node[0] ** 2 + node[1] ** 2) ** 0.5)
        if r < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return np.array([node[0] / r, node[1] / r, 0.0])

    def side(self, X, Y, Z) -> np.ndarray:
        """Boolean array: True where sqrt(x²+y²) >= radius (outside)."""
        return np.hypot(np.asarray(X), np.asarray(Y)) >= self.radius

    def __repr__(self) -> str:
        return f"CylindricalBoundary(radius={self.radius:.6g})"


class SphericalBoundary:
    """
    A spherical interface defined by  |x| = radius  (all three components).

    The outward-pointing normal at any surface point is  x / |x|.

    Parameters
    ----------
    radius : float
        Sphere radius [m].
    """

    def __init__(self, radius: float) -> None:
        self.radius: float = float(radius)

    def straddles(self,
                  bmin: np.ndarray,
                  bmax: np.ndarray,
                  node: np.ndarray) -> bool:
        """
        Return True if the dual cell [bmin, bmax] straddles this sphere.

        As for :class:`CylindricalBoundary`, the max of |x| over the box is
        at a corner but the min is the clamped closest-point distance from
        the sphere centre (origin) to the box.
        """
        r_corners = [
            (cx ** 2 + cy ** 2 + cz ** 2) ** 0.5
            for cx in (bmin[0], bmax[0])
            for cy in (bmin[1], bmax[1])
            for cz in (bmin[2], bmax[2])
        ]
        r_max = max(r_corners)
        dx = _min_dist_interval_to_zero(float(bmin[0]), float(bmax[0]))
        dy = _min_dist_interval_to_zero(float(bmin[1]), float(bmax[1]))
        dz = _min_dist_interval_to_zero(float(bmin[2]), float(bmax[2]))
        r_min = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        return r_min < self.radius <= r_max

    def normal_at(self, node: np.ndarray) -> np.ndarray:
        r = float(np.linalg.norm(node))
        if r < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return node / r

    def side(self, X, Y, Z) -> np.ndarray:
        """Boolean array: True where |x| >= radius (outside)."""
        return np.sqrt(np.asarray(X) ** 2 + np.asarray(Y) ** 2
                       + np.asarray(Z) ** 2) >= self.radius

    def __repr__(self) -> str:
        return f"SphericalBoundary(radius={self.radius:.6g})"


# ---------------------------------------------------------------------------
# Geometry stack
# ---------------------------------------------------------------------------

class GeometryStack:
    """
    Ordered collection of boundary objects (innermost first).

    Calling :meth:`interface_func` produces a function with signature
    ``(bmin, bmax, node) → None | n̂ | [n̂₁, n̂₂, …]`` that can be passed
    directly to :func:`~lebedev_em.media.from_geometry_func`.

    Parameters
    ----------
    boundaries : sequence of boundary objects
        :class:`PlanarBoundary`, :class:`CylindricalBoundary`,
        :class:`SphericalBoundary`, or any object implementing
        ``straddles(bmin, bmax, node) → bool`` and
        ``normal_at(node) → ndarray (3,)``.
        Listed **innermost first**.

    Examples
    --------
    Simple two-layer planar medium::

        geo = GeometryStack([PlanarBoundary(n_hat=[0,0,1], d=0.0)])

    DDH03 borehole with invasion zone and one dipping layer::

        geo = GeometryStack([
            CylindricalBoundary(radius=0.1),
            CylindricalBoundary(radius=0.6),
            PlanarBoundary(n_hat=N_HAT, d=D_PLANE),
        ])

    Two non-parallel dipping planes (four-layer formation)::

        geo = GeometryStack([
            CylindricalBoundary(radius=0.1),
            CylindricalBoundary(radius=0.6),
            PlanarBoundary(n_hat=N_HAT_1, d=D_1),
            PlanarBoundary(n_hat=N_HAT_2, d=D_2),
        ])
    """

    def __init__(self, boundaries: list) -> None:
        self.boundaries: list = list(boundaries)

    def interface_func(
        self,
        bmin: np.ndarray,
        bmax: np.ndarray,
        node: np.ndarray,
    ) -> "None | np.ndarray | list[np.ndarray]":
        """
        Return the interface description for the dual cell centred at *node*.

        Returns
        -------
        None
            Cell is entirely within one material region — no averaging needed.
        ndarray (3,)
            Cell straddles exactly one boundary; this is its unit normal.
        list of ndarray (3,)
            Cell straddles multiple boundaries; normals listed **innermost
            first** (same order as :attr:`boundaries`).
        """
        normals = []
        for b in self.boundaries:
            if b.straddles(bmin, bmax, node):
                normals.append(b.normal_at(node))
        if not normals:
            return None
        if len(normals) == 1:
            return normals[0]
        return normals

    def classify(self, X, Y, Z) -> np.ndarray:
        """
        Exact geometric region label for each point.

        Each boundary contributes one bit (its :meth:`side`); the integer
        formed from the bits of all boundaries (innermost = least-significant)
        is a unique label per geometric region.  Two points with the same
        label lie in the same material region — regardless of how close their
        conductivities are — so this never lumps distinct materials the way a
        conductivity threshold does.

        Returns an integer array shaped like the broadcast of X, Y, Z.
        """
        X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z)
        shape = np.broadcast(X, Y, Z).shape
        label = np.zeros(shape, dtype=np.int64)
        for bit, b in enumerate(self.boundaries):
            label = label | (b.side(X, Y, Z).astype(np.int64) << bit)
        return label

    def straddling_boundaries(self, bmin, bmax, node) -> list:
        """List of boundaries (innermost first) that the cell straddles."""
        return [b for b in self.boundaries if b.straddles(bmin, bmax, node)]

    def __repr__(self) -> str:
        lines = ["GeometryStack(["]
        for b in self.boundaries:
            lines.append(f"    {b!r},")
        lines.append("])")
        return "\n".join(lines)

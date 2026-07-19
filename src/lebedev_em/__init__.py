"""
lebedev_em — Lebedev staggered-grid FD scheme for 3D EM logging.

Implements the finite-difference scheme from:
    Davydycheva, Druskin & Habashy (2003), Geophysics 68(5):1525–1536.

Quickstart
----------
>>> from lebedev_em.grid import symmetric_uniform_grid
>>> from lebedev_em.media import homogeneous_isotropic
>>> from lebedev_em.solver import LebedevMaxwellSolver
>>> import numpy as np

>>> grid  = symmetric_uniform_grid(Mx=8, My=8, Mz=8, Lx=10, Ly=10, Lz=10)
>>> media = homogeneous_isotropic(grid, sigma=1.0)
>>> solver = LebedevMaxwellSolver(grid, media, omega=2*np.pi*2500)
>>> result = solver.solve(0, 0, 0, dipole_comp=0)
"""

from .grid import LebedevGrid3D, uniform_grid, symmetric_uniform_grid, optimal_geometric_1d
from .media import EMMedia, homogeneous_isotropic, layered_isotropic, make_sigma_func, from_fine_grid, from_sigma_func, from_geometry_exact, MU0, EPS0
from .geometry import PlanarBoundary, CylindricalBoundary, SphericalBoundary, GeometryStack
from .operators import build_curl_RE, build_curl_PR, build_system_matrix
from .solver import LebedevMaxwellSolver
from .analytics import magnetic_dipole_B, Bxx_homogeneous

__all__ = [
    "LebedevGrid3D",
    "uniform_grid",
    "symmetric_uniform_grid",
    "optimal_geometric_1d",
    "EMMedia",
    "homogeneous_isotropic",
    "layered_isotropic",
    "make_sigma_func",
    "from_fine_grid",
    "from_sigma_func",
    "from_geometry_exact",
    "PlanarBoundary",
    "CylindricalBoundary",
    "SphericalBoundary",
    "GeometryStack",
    "MU0",
    "EPS0",
    "build_curl_RE",
    "build_curl_PR",
    "build_system_matrix",
    "LebedevMaxwellSolver",
    "magnetic_dipole_B",
    "Bxx_homogeneous",
]

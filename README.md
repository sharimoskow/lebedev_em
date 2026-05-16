# lebedev_em

A Python implementation of the Lebedev staggered-grid finite-difference scheme
for 3-D electromagnetic logging simulation in anisotropic media, as described in:

> Davydycheva, S., Druskin, V., & Habashy, T. (2003).
> *An efficient finite-difference scheme for electromagnetic logging in 3D anisotropic inhomogeneous media.*
> **Geophysics**, 68(5), 1525–1536.

The scheme uses a staggered Lebedev grid that naturally
accommodates full anisotropic conductivity tensors, making it well-suited for
deviated-well scenarios with dipping anisotropic formations.

---

## Features

- **Lebedev staggered-grid** FD discretisation of Maxwell's equations at a
  single frequency (time-harmonic, magnetic dipole source)
- **Flexible media builders** — specify conductivity from a callable function,
  fine reference grid, or analytical geometry (bore / invasion / formation
  boundaries), with automatic sub-cell homogenisation via sequential Backus
  averaging
- **Geometry stack** — compose cylindrical, planar, and spherical boundaries
  to describe layered/invaded formations; supply analytical normals for each
  boundary to bypass numerical normal estimation
- **Sparse direct and iterative solvers** — wraps `scipy.sparse.linalg.spsolve`
  and `scipy.sparse.linalg.lgmres`
- **Post-processing** — extract magnetic-field components on- and off-axis,
  compute apparent resistivity, compare to whole-space dipole analytic solution

---

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/<your-username>/lebedev_em.git
cd lebedev_em
pip install -e ".[dev]"
```

Dependencies (installed automatically): `numpy`, `scipy`, `matplotlib`.
Optional dev extras: `pytest`, `ipython`, `jupyter`.

---

## Quickstart

```python
import numpy as np
from lebedev_em import symmetric_uniform_grid, homogeneous_isotropic, LebedevMaxwellSolver

# 1. Build a symmetric uniform grid (±10 m, 8 cells per half-axis)
grid = symmetric_uniform_grid(Mx=8, My=8, Mz=8, Lx=10, Ly=10, Lz=10)

# 2. Homogeneous isotropic medium, σ = 1 S/m
media = homogeneous_isotropic(grid, sigma=1.0)

# 3. Solve for x-directed magnetic dipole at origin, f = 2500 Hz
omega = 2 * np.pi * 2500
solver = LebedevMaxwellSolver(grid, media, omega=omega)
result = solver.solve(0, 0, 0, dipole_comp=0)

# 4. Extract Bxx on the z-axis
from lebedev_em.postprocess import extract_axis_response
z_ax, Bxx = extract_axis_response(grid, result, comp='xx')
```

---

## DDH03 benchmark example

The `examples/` directory reproduces the DDH03 benchmark from the paper: a
deviated borehole with bore (r < 0.1 m, σ = 0.05 S/m), invasion zone
(0.1 < r < 0.6 m, σ = 0.10 S/m), and a dipping anisotropic formation
(60° dip; σ_T = 0.10 S/m, σ_N = 0.01 S/m below dip; σ = 0.50 S/m above dip).

```bash
# Run with analytical geometry normals (fastest, most accurate):
python examples/run_geometry_func.py

# Run with SVD-based sub-cell homogenisation (sigma_func path):
python examples/run_sigma_func_ddh03.py

# Compare crossing-depth results across media strategies:
python examples/compare_media_crossing.py
```

The key output is the zero-crossing depth of Im(Bxx) − Im(Bxz) on the tool axis,
which characterises the formation boundary detection capability.

---

## API overview

### Grid

| Function / Class | Description |
|---|---|
| `uniform_grid(Mx, My, Mz, hx, hy, hz)` | Uniform spacing |
| `symmetric_uniform_grid(Mx, My, Mz, Lx, Ly, Lz)` | Symmetric uniform |
| `optimal_geometric_1d(L, hmin, ratio)` | Geometric stretching toward PML |
| `LebedevGrid3D` | Core grid class |

### Media builders

| Function | Description |
|---|---|
| `homogeneous_isotropic(grid, sigma)` | Constant scalar σ |
| `layered_isotropic(grid, z_interfaces, sigmas)` | 1-D layered σ(z) |
| `from_sigma_func(grid, sigma_func, ...)` | σ from callable; SVD normals |
| `from_geometry_func(grid, sigma_func, interface_func, ...)` | σ + analytical normals via `GeometryStack` |
| `from_fine_grid(grid, fine_grid, fine_media)` | Coarsen from reference |

### Geometry

| Class | Description |
|---|---|
| `CylindricalBoundary(radius)` | Cylindrical interface (r = const) |
| `PlanarBoundary(normal, d)` | Planar interface (n·x = d) |
| `SphericalBoundary(center, radius)` | Spherical interface |
| `GeometryStack(boundaries)` | Ordered collection of boundaries (innermost first) |

### Solver

```python
solver = LebedevMaxwellSolver(grid, media, omega)
result = solver.solve(sx, sy, sz, dipole_comp=0,   # 0=x, 1=y, 2=z
                      method='direct')              # or 'lgmres'
```

### Analytics

```python
from lebedev_em import magnetic_dipole_B, Bxx_homogeneous
B_analytic = magnetic_dipole_B(r_obs, sigma, omega)
```

---

## Running tests

```bash
pytest
```

Tests cover grid construction, operator assembly, and solver correctness against
the whole-space magnetic dipole analytic solution.

---

## License

MIT License. See `pyproject.toml` for details.

---

## Citation

If you use this code in your research, please cite the original paper:

```
Davydycheva, S., Druskin, V., & Habashy, T. (2003).
An efficient finite-difference scheme for electromagnetic logging
in 3D anisotropic inhomogeneous media.
Geophysics, 68(5), 1525–1536.
https://doi.org/10.1190/1.1620626
```

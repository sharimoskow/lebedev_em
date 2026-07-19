# lebedev_em

Status BETA: This is a research code in active development. Use with appropriate caution.

A Python implementation of the Lebedev staggered-grid finite-difference scheme
for 3-D electromagnetic logging simulation in anisotropic media, as described in:

> Davydycheva, S., Druskin, V., & Habashy, T. (2003).
> *An efficient finite-difference scheme for electromagnetic logging in 3D anisotropic inhomogeneous media.*
> **Geophysics**, 68(5), 1525–1536.

The scheme uses a staggered Lebedev grid that naturally
accommodates full anisotropic conductivity tensors, making it well-suited for
deviated-well scenarios with dipping anisotropic formations.

**Potential applications** include standard borehole EM logging, deviated-well
scenarios with dipping beds, and EM-based fracture/crack imaging — where cracks
appear as sharp conductivity contrasts at arbitrary orientations, precisely the
case handled by the oblique-interface nodal homogenization.

**Research context.** This code is developed for academic research purposes in
the Department of Mathematics at Drexel University. The project supports our
proposal to the U.S. Department of Energy's **Genesis** program on the modeling
of fractures — thin, high-contrast conductivity structures at arbitrary
orientations, the regime addressed by the nodal homogenization and validated in
the thin-layer benchmark — as well as related proposals, including to the Air
Force Office of Scientific Research (**AFOSR**) on subsurface imaging.

---

## Features

- **Lebedev staggered-grid** FD discretisation of Maxwell's equations at a
  single frequency (time-harmonic, magnetic dipole source)
- **Optimal geometric grids** (Druskin & Knizhnerman 1999) — geometrically
  stretched grids in the transverse (x, y) directions achieve exponential
  convergence, keeping 3-D system sizes tractable.
- **Two media-building paths, three averaging methods.** Build the conductivity
  from an analytic `GeometryStack` (the **exact** path, `from_geometry_exact`)
  or from a black-box σ(x) callable (the **lookup** path, `from_sigma_func`).
  Each offers the same three sub-cell schemes: `pointwise` (each node takes its
  material tensor), `backus` (the anisotropic laminate / standard homogenization,
  DDH03 eq. 9), and `nodal` (the Moskow energy-matched tensor, extended to 3-D).
  On the **exact** path the interface normals, per-region material tensors, and
  volume/line fractions are all computed **analytically — no sampling** — and the
  Backus laminate is provably *bound-preserving* (effective eigenvalues stay
  within the constituent range). The lookup path estimates the normal by SVD and
  the fractions by sub-grid quadrature.
- **Geometry stack** — compose cylindrical, planar, and spherical boundaries
  to describe layered/invaded formations; the exact path reads normals and exact
  per-cell material fractions directly from this geometry.
- **Sparse direct and iterative solvers** — wraps `scipy.sparse.linalg.spsolve`
  and `scipy.sparse.linalg.lgmres`
- **Post-processing** — extract magnetic-field components on- and off-axis,
  compute apparent resistivity, compare to whole-space dipole analytic solution

---

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/sharimoskow/lebedev_em.git
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

## Anisotropic media: use the coupled solve

When σ has off-diagonal entries (tilted anisotropy, homogenized interface
cells), the four Lebedev clusters couple, and the solve must be coupled:
use `LebedevMaxwellSolver.solve_coupled` (or an all-cluster RHS with the
component-aware boundary conditions). The historical four-separate-solves
procedure — one solve per cluster source, each read on its own sub-grid —
is exact for isotropic media but **under-counts anisotropy-generated
cross-components** (e.g. B_xz from an x-directed dipole), because the
coupling deposits part of the response on partner clusters' sub-grids; in a
homogeneous tilted anisotropic medium the cross-component is lost entirely.
`tests/test_anisotropic_coupling.py` locks in both behaviors. The correct
generalization of the four-solve mixed-BC averaging to coupled clusters
(retaining its boundary-error cancellation) is work in progress.

## Benchmarks

Two benchmarks are validated in absolute units (unit dipole moment, SI, no
fitted constants):

- **Two half-spaces (exact analytic reference).** VED over a 10× conductivity
  contrast, checked against the exact Sommerfeld-integral solution: ~1% RMS
  in the conductive half-space; the four-cluster Lebedev average reduces the
  single-cluster error five-fold.

  ```bash
  python examples/benchmark_two_layer.py
  ```

- **Thin dipping anisotropic layer (DDH03 Fig. 9 configuration).** Resistive
  borehole crossing a 0.25 m-thick 75° dipping layer with σ_N = σ_T/200 at
  52.65 kHz, fully coupled single solve on the k = 6 optimal grid, compared to
  the values digitized from the published figure. Current status (an open
  investigation, **not** a settled benchmark):

  - the **pointwise** medium reproduces the published Im B_z curve;
  - the cell-averaging schemes — the anisotropic **Backus** laminate (eq. 9) and
    the **nodal** tensor, which nearly coincide on the shared dual cell —
    **over-attenuate** the layer response by ~30%.

  The over-attenuation has been traced to the *off-diagonal* entries of the
  averaged effective tensor (the inter-cluster coupling of the tilted layer):
  zeroing them recovers the pointwise result. Reconciling the coupled Lebedev
  treatment of that coupling with the published curve is work in progress.

  ```bash
  # coupled solve, pointwise / Backus / nodal vs the digitized figure:
  python examples/fig9_backus_vs_paper.py run
  ```

## Deviated-borehole model example (media-building strategies)

The `examples/` directory also exercises a deviated-borehole model
(DDH03 Fig. 6/7): bore (r < 0.1 m, σ = 0.05 S/m), invasion zone
(0.1 < r < 0.6 m, σ = 0.10 S/m), and a dipping anisotropic formation
(60° dip; σ_T = 0.10 S/m, σ_N = 0.01 S/m below dip; σ = 0.50 S/m above dip).

```python
from lebedev_em import from_geometry_exact
from lebedev_em.geometry import CylindricalBoundary, PlanarBoundary, GeometryStack

geo = GeometryStack([CylindricalBoundary(0.1), CylindricalBoundary(0.6),
                     PlanarBoundary(n_hat=N_HAT, d=D_PLANE)])
med = from_geometry_exact(grid, sigma_func, geo, method="backus")   # or "nodal", "pointwise"
```

```bash
# Fig. 7 (B_xx, B_xz) reproduced via the exact Backus / nodal core:
python examples/fig7_backus_check.py
```


---

## API overview

### Grid

| Function / Class | Description |
|---|---|
| `uniform_grid(Mx, My, Mz, hx, hy, hz)` | Uniform spacing |
| `symmetric_uniform_grid(Mx, My, Mz, Lx, Ly, Lz)` | Symmetric uniform |
| `optimal_geometric_1d(L, hmin, ratio)` | Optimal grid geometric stretching |
| `LebedevGrid3D` | Core grid class |

### Media builders

| Function | Description |
|---|---|
| `homogeneous_isotropic(grid, sigma)` | Constant scalar σ |
| `layered_isotropic(grid, z_interfaces, sigmas)` | 1-D layered σ(z) |
| `from_geometry_exact(grid, sigma_func, geometry, method=...)` | **Exact path** — σ + analytic normals/fractions via `GeometryStack`; `method` ∈ {`pointwise`, `backus`, `nodal`} |
| `from_sigma_func(grid, sigma_func, method=..., ...)` | **Lookup path** — σ from a black-box callable; SVD-estimated normals + sampled fractions; same `method` choices |
| `from_fine_grid(grid, fine_grid, fine_media)` | Coarsen from reference |

Both media-building paths accept both scalar (isotropic) and 3×3 tensor (anisotropic) conductivity values, enabling full tilted transverse isotropy (TTI) and general anisotropy, and both expose the same three sub-cell schemes (`pointwise`, `backus`, `nodal`).

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
## Development

This implementation was developed with the assistance of Claude (Anthropic), including the design and debugging of the sub-cell homogenization routines (the exact-geometry `from_geometry_exact` builder, the anisotropic Backus / eq. 9 laminate, and the sequential nodal homogenization) and the `GeometryStack` interface.


## References

**Core algorithm** — please cite if you use this code:

```
Davydycheva, S., Druskin, V., & Habashy, T. (2003).
An efficient finite-difference scheme for electromagnetic logging
in 3D anisotropic inhomogeneous media.
Geophysics, 68(5), 1525–1536.
https://doi.org/10.1190/1.1620626
```

**Optimal geometric grids** — the theoretical basis for the fast convergence
in x and y with small k:

```
Druskin, V., & Knizhnerman, L. (1999).
Gaussian spectral rules for the three-point second differences:
I. A two-point positive definite problem in a semi-infinite domain.
SIAM Journal on Numerical Analysis, 37(2), 403–422.

Ingerman, D., Druskin, V., & Knizhnerman, L. (2000).
Optimal finite difference grids and rational approximations of the square root
I. Elliptic problems.
Communications on Pure and Applied Mathematics, 53(8), 1039–1066.
```

**Nodal homogenisation** — the sub-cell conductivity averaging scheme:

```
Moskow, S., Druskin, V., Habashy, T., Lee, P., & Davydycheva, S. (1999).
A finite difference scheme for elliptic equations with rough coefficients
using a Cartesian grid nonconforming to interfaces.
SIAM Journal on Numerical Analysis, 36(2), 442–464.
```

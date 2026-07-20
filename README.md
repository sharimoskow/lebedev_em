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
of fractures — where cracks appear as thin, high-contrast conductivity
structures at arbitrary orientations, precisely the regime addressed by the
nodal homogenization and the thin-layer benchmark below — as well as related
proposals, including to the Air Force Office of Scientific Research (**AFOSR**)
on subsurface imaging.

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
from lebedev_em.postprocess import lebedev_E_at_point

# 1. Build a symmetric uniform grid (±10 m, 16 cells per axis)
grid = symmetric_uniform_grid(Mx=16, My=16, Mz=16, Lx=10, Ly=10, Lz=10)

# 2. Homogeneous isotropic medium, σ = 1 S/m
media = homogeneous_isotropic(grid, sigma=1.0)

# 3. Solve for an x-directed unit electric dipole at the origin, f = 2500 Hz
solver = LebedevMaxwellSolver(grid, media, omega=2 * np.pi * 2500)
result = solver.solve(0.0, 0.0, 0.0, dipole_comp=0)     # 1 A·m, coupled solve

# 4. E_x at a receiver point via the four-cluster interpolate-then-average read
Ex = lebedev_E_at_point(grid, result["E_c"], 0, 0.0, 0.0, 1.25)
```

For magnetic-dipole (induction-logging) sources and B-field extraction on the
axis, see `examples/benchmark_wholespace.py` and `examples/fig9_refinement.py`,
which build the per-cluster magnetic-dipole right-hand side with
`build_rhs_per_cluster(..., hx_comp=...)` and read the field with
`lebedev_B_on_z_axis`.

---

## Anisotropic media: use the coupled solve

When σ has off-diagonal entries (tilted anisotropy, homogenized interface
cells), the four Lebedev clusters couple, and the solve must be coupled:
use `LebedevMaxwellSolver.solve` with the default `method='coupled'` (an
all-cluster RHS with the component-aware per-cluster boundary conditions).
The historical four-separate-solves procedure — one solve per cluster
source, each read on its own sub-grid — is exact for isotropic media but
**under-counts anisotropy-generated cross-components** (e.g. B_xz from an
x-directed dipole), because the coupling deposits part of the response on
partner clusters' sub-grids; in a homogeneous tilted anisotropic medium the
cross-component is lost entirely. `tests/test_anisotropic_coupling.py`
locks in both behaviors.

## Benchmarks

These benchmarks are validated in absolute units (unit dipole moment, SI, no
fitted constants):

- **Two half-spaces (exact analytic reference).** VED over a 10× conductivity
  contrast, checked against the exact Sommerfeld-integral solution: ~1% RMS
  in the conductive half-space; the four-cluster Lebedev average reduces the
  single-cluster error five-fold.

  ```bash
  python examples/benchmark_two_layer.py
  ```

- **Two half-spaces with sub-cell averaging (coupled solve).** The same
  Sommerfeld reference, but with the contact placed mid-cell so a dual cell
  straddles the interface and the homogenization is actually exercised. The
  medium is built with `from_geometry_exact` for each of `pointwise`, `backus`,
  and `nodal`, and solved with the fully coupled single solve. On the
  transmitted (conductive) side the averaged schemes converge to the analytic
  as the transverse grid refines — reaching ~0.01% RMS by k = 4–5, where the
  harmonic-normal / arithmetic-transverse tensor is the physically correct
  effective conductivity for E_z continuity across the contact; `backus` and
  `nodal` coincide exactly for the axis-aligned normal. The source-side
  residual is ordinary O(h_z²) axial-grid error, shared by all three schemes.

  ```bash
  python examples/benchmark_two_layer_averaging.py        # methods + coupled solve
  python examples/two_layer_averaging_convergence.py 5    # transverse (k) convergence
  ```

- **Thin dipping anisotropic layer crossed by a borehole (DDH03 Fig. 9).**
  The full Fig.-9 configuration: resistive borehole (σ = 0.05 S/m,
  R = 0.1 m), invaded zone (σ = 0.1 S/m, R = 0.6 m, replacing the formation
  near the wellbore), and a thin 75°-dipping anisotropic layer (0.25 m,
  σ_T = 0.1, σ_N = σ_T/200 S/m) present for r ≥ 0.6 m; z magnetic dipole at
  52.65 kHz, Im B_z on the axis. With the coupled solve and either averaged
  scheme (`backus` or `nodal`), the computed curves match the values
  digitized from the published figure to 0.4–4.5% (no layer) and 3–12%
  (resistive layer) — **identically at three grid-refinement levels**,
  reproducing DDH03's own k = 6 vs k = 12 insensitivity for sub-cell layers.

  ```bash
  python examples/fig9_refinement.py solve backus 1 nolayer   # one (method, level, model) per call
  python examples/fig9_refinement.py solve backus 1 annulus
  python examples/fig9_refinement.py report                   # tables vs the digitized figure
  python examples/fig9_refinement.py plot
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
| `uniform_grid(Mx, My, Mz, Lx, Ly, Lz)` | Uniform spacing |
| `symmetric_uniform_grid(Mx, My, Mz, Lx, Ly, Lz)` | Symmetric uniform |
| `optimal_geometric_1d(k, h_min, L, gamma)` | Optimal geometric half-axis (2k+1 interleaved primary+dual nodes) |
| `symmetric_optimal_grid(h_min, L, z, gamma, k)` | 3-D grid with optimal x/y and user z (DDH03 logging geometry) |
| `hybrid_axial_grid(z_min, z_max, n_inner, k_outer)` | Equidistant inner zone + geometric outer zones |
| `LebedevGrid3D` | Core grid class |

### Media builders

| Function | Description |
|---|---|
| `homogeneous_isotropic(grid, sigma)` | Constant scalar σ |
| `layered_isotropic(grid, layer_boundaries, sigma_values)` | 1-D layered σ(z) |
| `from_geometry_exact(grid, sigma_func, geometry, method=...)` | **Exact path** — σ + analytic normals/fractions via `GeometryStack`; `method` ∈ {`pointwise`, `backus`, `nodal`} |
| `from_sigma_func(grid, sigma_func, method=..., ...)` | **Lookup path** — σ from a black-box callable; SVD-estimated normals + sampled fractions; same `method` choices |
| `from_fine_grid(grid, fine_grid, fine_media)` | Coarsen from reference |

Both media-building paths accept both scalar (isotropic) and 3×3 tensor (anisotropic) conductivity values, enabling full tilted transverse isotropy (TTI) and general anisotropy, and both expose the same three sub-cell schemes (`pointwise`, `backus`, `nodal`).

### Geometry

| Class | Description |
|---|---|
| `CylindricalBoundary(radius)` | Cylindrical interface (r = const, axis = z) |
| `PlanarBoundary(n_hat, d)` | Planar interface (n̂·x = d) |
| `SphericalBoundary(radius)` | Spherical interface (centered at origin) |
| `GeometryStack(boundaries)` | Ordered collection of boundaries (innermost first) |

### Solver

```python
solver = LebedevMaxwellSolver(grid, media, omega)
result = solver.solve(sx, sy, sz, dipole_comp=0,    # 0=x, 1=y, 2=z (electric dipole)
                      method='coupled')              # or 'clustered' (isotropic only)
# result: dict with 'E_avg', 'E_c' (per-cluster), 'rhs'
```

### Analytics

```python
from lebedev_em.analytics import magnetic_dipole_B, electric_dipole_E
B_analytic = magnetic_dipole_B(x, y, z, sigma, omega)   # whole-space reference
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

This implementation was developed with the assistance of Claude (Anthropic), including the design and debugging of the sub-cell homogenization routines (the exact-geometry `from_geometry_exact` builder, the anisotropic Backus / eq. 9 laminate, and the nodal homogenization) and the `GeometryStack` interface.


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

# Lebedev FD Maxwell Solver — Code Documentation

**Code overview, parameters, and validated benchmarks.**
*Updated July 2026 (supersedes the May 2026 Word/PDF write-up).*

---

## Part I: Code Description

### 1. Overview

This code implements the Lebedev finite-difference (FD) scheme for the
time-harmonic (frequency-domain) Maxwell equations in 3D, following
Davydycheva, Druskin, and Habashy (2003), hereafter DDH03. The solver targets
borehole electromagnetic logging simulations: a magnetic dipole source on the
borehole axis, surrounded by cylindrical layers and possibly intersected by a
dipping planar geological boundary.

The Lebedev scheme partitions the grid into two interleaved subgrids (P-nodes
and R-nodes, based on the parity of the index sum i+j+k). Electric-field
components live at R-nodes, magnetic-field components at P-nodes. Four
staggered "clusters" (C000, C101, C110, C011) tile the field components
across these subgrids, generalizing the Yee grid to handle fully anisotropic
conductivity tensors without unphysical cross-coupling.

Code modules:

- `grid.py` — grid construction (`LebedevGrid3D`, optimal geometric grids, hybrid axial grids)
- `media.py` — conductivity assignment and interface averaging (`EMMedia`, nodal homogenization)
- `operators.py` — FD curl operators and boundary-condition application
- `solver.py` — system assembly and linear solve
- `postprocess.py` — B-field extraction, multi-cluster averaging, axis sampling
- `analytics.py` — closed-form and Sommerfeld reference solutions for validation

### 2. Grid

#### 2.1 Transverse grid (x and y)

Primary node spacings grow geometrically from a minimum step `h_min`:

```
h_i = h_min · α^(i−1),    α = exp(γπ/√k),    γ = 1/√2   (EM-optimal)
```

Dual nodes are interleaved using spacings `ĥ₁ = h_min/(1+√α)` and
`ĥᵢ = hᵢ/√α` for i ≥ 2. The full symmetric grid is formed by mirroring the
positive half-axis, giving `Mx = My = 4k` intervals. The optimal-grid theory
underlying this construction — including the choice γ = 1/√2 for EM
diffusion — originates with Druskin & Knizhnerman (1999) and Ingerman,
Druskin & Knizhnerman (2000).

| Parameter | Typical value | Description |
|---|---|---|
| `h_min` | 0.05–0.17 m | Minimum primary spacing (inner resolution). **Choose `h_min` small enough to resolve the smallest cylindrical feature** (e.g. 0.05 m for a 0.1 m borehole); note that the k-convergence test cannot detect under-resolution at the axis, because all k share the same `h_min`. |
| `k` | 3–6 | Primary steps per half-axis; `Mx = 4k`. Controls both resolution and domain size. |
| `L` | 300 m | Target domain half-length (k auto-chosen to reach L if not specified) |
| `gamma` | 1/√2 | Progression exponent; 1/√2 is EM-optimal per DDH03 |

Maximum transverse extent: `x_max = Σ h_i`. A minimum of roughly 0.7 skin
depths of transverse coverage is needed for accurate results.

Grid alignment: if `h_min` is chosen as `R_interface/α` (the "magic"
`h_min`), the interface lands exactly on a primary node, eliminating
interface averaging at that boundary. For all other values, nodal
homogenization (Section 4) handles the straddling cells.

#### 2.2 Axial grid (z)

A hybrid grid: an equidistant inner zone `[z_tool_min, z_tool_max]` resolves
the tool uniformly; outside this range an optimal geometric expansion pushes
the computational boundary far away without excessive nodes.

| Parameter | Typical value | Description |
|---|---|---|
| `z_tool_min` | −3.5 m | Lower boundary of equidistant inner zone |
| `z_tool_max` | +2.5 m | Upper boundary of equidistant inner zone |
| `n_inner` | 96 | Equidistant steps in the inner zone (must be even; choose so that the source lands exactly on an even-index node, otherwise the source is snapped and a warning is emitted) |
| `k_outer` | 8 | Optimal-geometric steps on each outer side |

Total z-nodes: `4·k_outer + n_inner + 1`.

### 3. Conductivity Model

#### 3.1 Supported geometry

The code supports a borehole logging geometry with three nested regions:

- **Cylindrical borehole:** radius `R_BORE`, isotropic `σ_BORE`, occupying `r < R_BORE`.
- **Cylindrical invasion zone:** outer radius `R_INV`, isotropic `σ_INV`, occupying `R_BORE < r < R_INV`.
- **Formation:** `r > R_INV`; may be divided into two half-spaces by a dipping planar boundary (normal `N_HAT`, offset `D_PLANE`), each side independently isotropic or anisotropic.

The plane is `N_HAT·x = D_PLANE`, with `N_HAT = (sinθ, 0, cosθ)` for a
boundary dipping at angle θ from the borehole axis.

#### 3.2 Conductivity parameters

The anisotropic formation tensor is the standard transversely isotropic (TI)
form:

```
σ_ANISO = σ_T · I + (σ_N − σ_T) · N_HAT ⊗ N_HAT
```

Conductivity along `N_HAT` is `σ_N`; transverse to it is `σ_T`. Setting
`σ_N = σ_T` recovers an isotropic medium.

### 4. Interface Averaging

Because the Lebedev grid is coarsely spaced, physical interfaces will
generally not coincide with grid nodes. The conductivity assigned to a node
whose dual cell straddles an interface directly affects accuracy.

#### 4.1 Nodal homogenization for single-interface cells

For any node whose dual cell crosses exactly one interface, the nodal
homogenization formula of Moskow et al. (1999) is applied, matching the
continuous and discrete electromagnetic energy inner products over the dual
cell. For an interface with unit normal `n̂` between media σ₁ and σ₂:

1. compute the volume fraction of medium 1 in the dual cell (analytic integration);
2. compute per-axis line fractions along the grid edges **through the node**;
3. build the modified basis `L̃` from the tangential directions, the off-diagonal coupling terms, and the per-axis line-average resistivities;
4. assemble the energy matrix `G` from Schur complements of σ₁ and σ₂;
5. solve `Σ_D = L̃⁻ᵀ G L̃⁻¹` for the effective nodal tensor.

For axis-aligned interfaces this reduces to the classical result: tangential
components take the arithmetic mean, the normal component the harmonic mean.
Tilted or anisotropic interfaces require the full tensor formula and
introduce off-diagonal entries.

**Sub-cell averaging matters for thin high-contrast structures.** A single
grid cell that straddles a thin, strongly resistive layer must carry an
effective tensor rather than a single node material, or the layer's effect on
the response is under-resolved. The correct effective-medium treatment of such
thin high-contrast strata is the regime studied by Moskow et al. (1999); the
validation of the averaging schemes against analytic references is in
`examples/benchmark_two_layer_averaging.py` and the tilted-layer configuration
remains under active development.

Two computational paths are supported:

- **Exact geometry path** (`from_geometry_exact`): the interface is described
  by geometric objects (cylinder, plane), so `n̂` is analytic at each node,
  and fractions are computed by exact integration. Straddle detection uses
  the clamped closest-point distance from the interface to the dual box.
- **Lookup-function path** (`from_sigma_func`): conductivity is a black-box
  callable; `n̂` is estimated by sampling σ on a fine sub-grid and fitting a
  plane through interface crossings by SVD. A planarity ratio `s₃/s₁ ≥ 0.7`
  triggers a fallback to axis-aligned averaging (the whole tensor is
  replaced, preserving positive semi-definiteness).

#### 4.2 Sequential nodal homogenization for doubly-straddled cells

When two interfaces intersect near a cell (e.g. the invasion cylinder and the
dipping plane), the nodal formula is applied twice: first between the two
outer-formation media along `N_HAT` (yielding an effective outer tensor),
then between the invasion medium and that effective tensor along the radial
normal. The procedure degenerates gracefully to the single-interface
treatment away from the intersection. If the intermediate tensor's largest
eigenvalue exceeds three times the largest material conductivity, the result
is discarded in favor of the single-interface formula.

### 5. Boundary Conditions

All outer faces use the DDH03 combined boundary condition (DDH03 eq. 6). Per
cluster, on each face: the tangential **E** components of the two
electric-BC clusters are set to zero (Dirichlet rows in the system), and the
tangential **H** components of the two magnetic-BC clusters are enforced to
zero exactly — the corresponding rows of the discrete curl `C_RE` are
removed, so `H×n = 0` holds identically at every boundary face. (Normal
components remain free, as the parity structure dictates.) This pairing of
two electric-BC and two magnetic-BC clusters per face is what makes the
domain-truncation errors cancel in the four-cluster average, and it preserves
the volume-weighted symmetry of the discrete curl–curl operator (covered by
the test suite).

### 6. Linear System and Solve

The solver assembles and solves

```
(Cᵀ µ⁻¹ C − iω σ̇) E = b,      σ̇ = σ − iωε   (time convention e^{−iωt})
```

with ε = ε₀ by default and `b` the magnetic-dipole right-hand side; the
system has `3·N_R` unknowns. Note the sign of the displacement-current term:
with the `e^{−iωt}` convention of DDH03 eq. (1), the complex conductivity is
`σ − iωε` (the "+" printed after DDH03 eq. (5) is a typo in the paper); this
convention matches `analytics.py`, whose wavenumber has `Im k > 0` (decaying
fields).

The default solver is `scipy.sparse.linalg.spsolve` (direct sparse LU). For
large grids, or media with many homogenized (dense-block) cells, LGMRES with
Jacobi preconditioning is used instead (tolerance 1e-8); the homogenized
blocks increase LU fill-in substantially, and the iterative path is far
lighter on memory.

### 7. Source and Multi-Cluster Field Extraction

#### 7.1 Multi-cluster source

The source is a magnetic dipole. All four Lebedev clusters must be excited
coherently: `build_rhs_per_cluster` / `build_rhs_multicl` distribute the
dipole right-hand side across each cluster's native sub-grid nodes with
coordinate-based (trilinear) weights, so that DDH03's eq.-(7) conditions —
unit total weight and centroid at the physical source position — hold exactly
on nonuniform grids. If the requested source position does not coincide with
an even-index grid plane, the source is snapped and a warning is emitted;
build the z-grid so the source lands on a node.

#### 7.2 Multi-cluster B-field averaging

**Anisotropic media: solve coupled.** With off-diagonal σ the clusters
couple, and the correct procedure is a single coupled solve with
all-cluster sources (`solve_coupled`), whose B-vector is then averaged over
the four sub-grids by the extraction below. Four separate per-cluster-source
solves, each read on its own sub-grid, under-count anisotropy-generated
cross-components (they live partly on partner clusters' sub-grids) — see
`tests/test_anisotropic_coupling.py`.

After solving, B is computed from Faraday's law, `B = (1/iω) ∇×E`. To extract
the physically correct field, `lebedev_B_on_z_axis` /
`extract_B_on_axis_multicl` perform the proper Lebedev inter-cluster average:
for each target point, all four clusters contribute via coordinate-based
trilinear interpolation from their respective native sub-grids (Σw = 1 and
weighted centroid at the receiver, exact also on nonuniform grids), and the
four values are averaged. This inter-cluster averaging is essential — reading
only the on-axis P-nodes biases the result by whichever sub-grid those nodes
belong to, and equal-weight (¼) stencils carry a first-order bias on
stretched grids.

---

## Part II: Validated Benchmarks

Both benchmarks are in absolute SI units per unit dipole moment (1 A·m²),
with no fitted or calibrated constants anywhere. See the README's
*Benchmarks* section for run commands.

### 8. Two half-spaces with a 10× contrast (exact analytic reference)

A vertical electric dipole at the origin above a conductivity jump
(σ = 0.1 → 1.0 S/m at z = 4 m, f = 2500 Hz), compared against the exact
Sommerfeld-integral solution (`analytics.py`, itself verified against the
closed-form homogeneous solution and the normal-current continuity condition
at the contact). Results: ~1% RMS error in the conductive half-space across
the jump; the contact effect (a 1.65× enhancement just above, an ~0.18×
reduction below) is captured point by point; and the four-cluster average
reduces the single-cluster error from ~30% to ~6% — DDH03's
error-cancellation mechanism at work in an inhomogeneous medium. Runner:
`examples/benchmark_two_layer.py`.

### 9. Thin dipping anisotropic layer crossed by a borehole (DDH03 Fig. 9)

The configuration of DDH03 Figure 9: background σ = 0.1 S/m; resistive
borehole (σ = 0.05 S/m, R = 0.1 m); a thin dipping anisotropic layer of
0.25 m normal thickness, dip 75°, σ_T = 0.1 S/m, σ_N = σ_T/200, crossing the
axis over z ∈ [−1.93, −0.95] m; z-directed magnetic dipole at the origin,
f = 52.65 kHz; observable Im B_z on the axis, with and without the layer.

Method: full DDH03 methodology — k = 6 optimal transverse grid with
h_min = 0.05 m (resolving the borehole), four-cluster eq.-(7) sources,
per-cluster mixed boundary conditions, sequential nodal homogenization with
analytic normals, interpolated four-cluster extraction at the exact receiver
positions.

Results: the published "no layer" curve agrees with the unit-moment
homogeneous analytic solution to 1–8% (confirming the figure's absolute
normalization; the near-source deficit is the borehole's local reduction),
and our computed curves match the published values within 0.7–5.5% (no
layer) and 0–15% (resistive layer, the residual localized to the layer
window and insensitive to ±5 cm shifts of the layer position). Runner:
`examples/fig9_check.py`; sensitivity tests: `examples/fig9_tests.py`.

---

## References

- S. Davydycheva, V. Druskin, T. Habashy, *An efficient finite-difference
  scheme for electromagnetic logging in 3D anisotropic inhomogeneous media*,
  Geophysics **68**(5):1525–1536, 2003.
- S. Moskow, V. Druskin, T. Habashy, P. Lee, S. Davydycheva, *A finite
  difference scheme for elliptic equations with rough coefficients using a
  Cartesian grid nonconforming to interfaces*, SIAM J. Numer. Anal.
  **36**:442–464, 1999.
- V. Druskin, L. Knizhnerman, *Gaussian spectral rules for the three-point
  second differences: I*, SIAM J. Numer. Anal. **37**:403–422, 1999.
- D. Ingerman, V. Druskin, L. Knizhnerman, *Optimal finite difference grids
  and rational approximations of the square root: I*, Comm. Pure Appl. Math.
  **53**:1039–1066, 2000.

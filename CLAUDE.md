# lebedev_em — Lebedev FD Scheme for 3D EM Logging

## Project Goal

Implement the finite-difference scheme from:

> Davydycheva, S., Druskin, V., and Habashy, T. (2003). "An efficient finite-difference scheme for electromagnetic logging in 3D anisotropic inhomogeneous media." *Geophysics*, 68(5), 1525–1536. (DDH03)

The paper solves frequency-domain Maxwell equations in 3D **anisotropic** media using:
1. **Lebedev's staggered grid** — a coercive scheme supporting full conductivity tensors without interpolation
2. **Conductivity averaging (homogenization)** — handles heterogeneous media inside grid cells
3. **Optimal geometric grids** — spectrally optimal non-uniform grids for fast convergence

All three pieces are implemented and validated (77 passing tests). The
**coupled single solve is the method and the default** — see "Settled
conventions" below before touching solver or media code.

---

## Physics: Maxwell Equations (DDH03 eq. 1)

Frequency-domain (time convention exp(−iωt)):

```
∇ × E = iω μ H
∇ × H = σ E − iω ε E + J
```

- **E**, **H**: electric and magnetic field vectors
- **J**: transmitter current density
- **σ**, **μ**, **ε**: symmetric non-negative definite 3×3 tensors (conductivity, permeability, permittivity)
- Zero boundary conditions at infinity (approximated on finite domain Ω)

The second-order system for **E** (eq. 5):

```
∇ × [(μ^P)^{-1} ∇ × E^R] − iω σ̇^R E^R = iω J^R
```

where σ̇ = σ − iω ε (complex conductivity; DDH03's printed "σ + iωε" after eq. 5 is
inconsistent with their eq. 1 under the exp(−iωt) convention — deriving eq. 5 from
eq. 4 forces σ̇ = σ − iωε, which is also what `analytics.py` uses).

An equivalent dual system for **H** also exists (the key advantage of Lebedev over interpolation approaches).

---

## The Lebedev Grid

### Grid Nodes (DDH03 eq. 2)

Cartesian grid: (xᵢ, yⱼ, zₖ) for i=0…Mx, j=0…My, k=0…Mz with **Mx, My, Mz even**.

Two subgrids based on index parity:
- **Subgrid P** (magnetic nodes): (i+j+k) % 2 == 0
- **Subgrid R** (electric nodes): (i+j+k) % 2 == 1

**E^R** lives on R-nodes, **H^P** lives on P-nodes.

### Finite Differences (DDH03 eq. 3)

For f^P → f^R_x (centered difference at an R-node using neighboring P-nodes):
```
(f^R_x)_{i,j,k} = (f^P_{i+1,j,k} − f^P_{i−1,j,k}) / (x_{i+1} − x_{i−1})
```

For f^R → f^P_x (centered difference at a P-node using neighboring R-nodes):
```
(f^P_x)_{i,j,k} = (f^R_{i+1,j,k} − f^R_{i−1,j,k}) / (x_{i+1} − x_{i−1})
```

Key property: since adjacent nodes alternate P/R, differences always stay within the correct subgrid.

### The Four Clusters

In the **isotropic** case, Lebedev's scheme splits into **4 independent Yee-like clusters** (000, 101, 110, 011). Each cluster is a complete set of E and H field components.

**E-component assignments** (which cluster owns E_α at each R-node type):

| R-node type (i%2, j%2, k%2) | E_x cluster | E_y cluster | E_z cluster |
|---|---|---|---|
| (1, 0, 0) | 000 | 110 | 101 |
| (0, 1, 0) | 110 | 000 | 011 |
| (0, 0, 1) | 101 | 011 | 000 |
| (1, 1, 1) | 011 | 101 | 110 |

**Cluster 000** is the standard Yee grid:
```
E_x^R at (2i+1, 2j,   2k)     H_x^P at (2i,   2j+1, 2k+1)
E_y^R at (2i,   2j+1, 2k)     H_y^P at (2i+1, 2j,   2k+1)
E_z^R at (2i,   2j,   2k+1)   H_z^P at (2i+1, 2j+1, 2k)
```

**Cluster 101** (shift in x and z):
```
E_x^R at (2i,   2j,   2k+1)   H_x^P at (2i+1, 2j+1, 2k)
E_y^R at (2i+1, 2j+1, 2k+1)   H_y^P at (2i,   2j,   2k)
E_z^R at (2i+1, 2j,   2k)     H_z^P at (2i,   2j+1, 2k+1)
```

**Cluster 110** (shift in x and y):
```
E_x^R at (2i,   2j+1, 2k)     H_x^P at (2i+1, 2j,   2k+1)
E_y^R at (2i+1, 2j,   2k)     H_y^P at (2i,   2j+1, 2k+1)
E_z^R at (2i+1, 2j+1, 2k+1)   H_z^P at (2i,   2j,   2k)
```

**Cluster 011** (shift in y and z):
```
E_x^R at (2i+1, 2j+1, 2k+1)   H_x^P at (2i,   2j,   2k)
E_y^R at (2i,   2j,   2k+1)   H_y^P at (2i+1, 2j+1, 2k)
E_z^R at (2i,   2j+1, 2k)     H_z^P at (2i+1, 2j,   2k+1)
```

In the **anisotropic** case, the off-diagonal elements of σ, μ, ε couple E-components from different clusters that share the same node.

### Boundary Conditions

Mixed electric/magnetic BCs on the 4 clusters ensure error cancellation (DDH03, section "Superconvergence"):

The cluster label αβγ encodes which spatial dimensions are "dualized":
- **Cluster 000**: electric BC on all 6 faces (E × n = 0)
- **Cluster 101**: magnetic BC on x- and z-faces; electric BC on y-faces
- **Cluster 110**: magnetic BC on x- and y-faces; electric BC on z-faces
- **Cluster 011**: magnetic BC on y- and z-faces; electric BC on x-faces

For each face of the domain, exactly 2 clusters have each BC type. This causes the domain-truncation errors to cancel upon averaging.

**Electric BC** (Dirichlet): tangential E = 0 at boundary → remove boundary E-unknowns or set to zero.
**Magnetic BC** (Neumann-type): H × n = 0 at boundary → equivalent to ∂E/∂n = 0 for the tangential components.

---

## Source Placement (DDH03 eq. 7, section "Averaging of sources, solutions, and error cancellation")

For a unit x-oriented electric dipole at (xᵢ₀, yⱼ₀, zₖ₀):

1. Place the primary source at (i₀, j₀, k₀) in cluster 000.
2. For each of the other 3 clusters, place sources at 4 points with shifts (ℓx, ℓy, ℓz) satisfying **|ℓx| + |ℓy| + |ℓz| = 2** (all combinations of ±1 and 0 with this constraint).
3. **Condition 1**: the sum of source weights in each cluster = 1.
4. **Condition 2**: the center of mass of the source distribution in each cluster = (xᵢ₀, yⱼ₀, zₖ₀).

There are 12 possible shift points (6 for face-center shifts, 6 for edge-center shifts with |sum|=2).

### Solution Averaging

After solving the 4 cluster systems, the final field at a point (xᵢ, yⱼ, zₖ) is:
- **Direct cluster** (say 000): take the solution directly at (i, j, k).
- **Other 3 clusters**: linearly interpolate from nearest points with shifts satisfying eq. (7).
- **Final answer**: arithmetic average of the 4 cluster solutions.

---

## Optimal Geometric Grids (DDH03 section "Optimal Grid Approach")

For the induction logging geometry (sources and receivers on borehole axis z):
- Apply optimal grids along x and y directions.
- Use standard (or optimal) grid along z.

The optimal geometric grid has grid steps in **geometric progression**:

```
hᵢ = h · α^{i−1},   i = 1, …, k
ĥ₁ = h / (1 + √α),   ĥᵢ = hᵢ / √α,   i = 2, …, k
```

with progression factor:
```
α = exp(γ π / √k),   γ = 1/√2   (optimal for induction problems)
```

This choice provides **exponential convergence** (vs. algebraic for uniform grids), reducing the required grid size by ~an order of magnitude.

For Lebedev's grid, the error is **squared** compared to a single Yee grid (superconvergence).

**Grid size**: Mx = 4k (symmetric about origin, both primary and dual nodes).
The grid runs from x₀ to xMx where xk+1 = nh (n = number of equidistant steps that would give the same accuracy).

---

## Package Structure

```
lebedev_em/
├── CLAUDE.md                    ← you are here
├── pyproject.toml
├── README.md                    ← API overview, quickstart, benchmark summary
├── docs/
│   ├── code_documentation.md    ← detailed module docs + benchmark write-ups
│   └── nodal_homogenization_3d.{md,tex,pdf}
├── src/lebedev_em/
│   ├── grid.py                  ← LebedevGrid3D; uniform / optimal-geometric / hybrid grids
│   ├── operators.py             ← sparse FD curl operators C_RE, C_PR
│   ├── media.py                 ← media builders: homogeneous/layered, from_geometry_exact,
│   │                              from_sigma_func, from_fine_grid; pointwise/backus/nodal schemes
│   ├── geometry.py              ← GeometryStack, CylindricalBoundary, PlanarBoundary (exact
│   │                              normals and volume fractions)
│   ├── sources.py               ← 4-cluster source weighting (eq. 7); build_rhs_per_cluster
│   ├── solver.py                ← LebedevMaxwellSolver; solve_coupled (default) / clustered
│   ├── postprocess.py           ← lebedev_E_at_point, lebedev_B_on_z_axis (interpolate-then-
│   │                              average four-cluster reads)
│   └── analytics.py             ← whole-space / two-half-space (Sommerfeld) references
├── tests/                       ← 8 files, 77 tests: grid, operators, bc+solver, media/geometry,
│                                  exact geometry, backus, anisotropic coupling, postprocess
└── examples/                    ← ~40 runners: benchmark_*.py (Figs 2/3, whole-space, two-layer),
                                   fig9_*.py (DDH03 Fig 9 study), fig7_backus_check.py,
                                   tilted_*.py diagnostics, plot_*.py
```

The benchmark notes (`lebedev_em_benchmark_notes.tex`/`.pdf`) are deliberately
kept **outside the repo** and distributed separately; do not re-add them.

---

## Key Implementation Notes

### Grid Indexing Convention

- Full grid flat index: `n = i * Ny * Nz + j * Nz + k`
- R-nodes sequential index: `R_idx[i,j,k]` (−1 if node is in P)
- P-nodes sequential index: `P_idx[i,j,k]` (−1 if node is in R)
- E-field vector (component-blocked): `[Ex_0…Ex_{N_R−1} | Ey_0…Ey_{N_R−1} | Ez_0…Ez_{N_R−1}]`, length 3·N_R
- H-field vector (component-blocked): `[Hx_0…Hx_{N_P−1} | Hy_0…Hy_{N_P−1} | Hz_0…Hz_{N_P−1}]`, length 3·N_P

### Sparse Matrix Structure (Curl)

The curl from E^R to H^P is a 3·N_P × 3·N_R sparse matrix:
```
C_RE = [ 0      -∂z_RP   ∂y_RP  ]
       [ ∂z_RP   0      -∂x_RP  ]
       [-∂y_RP   ∂x_RP   0      ]
```
where `∂x_RP` is the N_P × N_R finite-difference derivative matrix in x.

The system matrix for eq. (5) is:
```
A = C_PR @ INVMU_P @ C_RE - iω * SIGMADOT_R
```
- `C_PR`: curl from P to R (adjoint/transpose of C_RE up to grid metric)  
- `INVMU_P`: block-diagonal (3×3 blocks) of μ⁻¹ at P-nodes
- `SIGMADOT_R`: block-diagonal (3×3 blocks) of σ̇ = σ − iωε at R-nodes

For **isotropic** media, INVMU_P and SIGMADOT_R are scalar multiples of identity → pure diagonal matrices.

### Boundary Conditions in Practice

**Electric BC** on a face (e.g., x=x_0): set tangential E components to zero at all R-nodes on that face. This is done by zeroing the corresponding rows/columns in the system matrix (or removing those unknowns).

**Magnetic BC** on a face: enforce **H × n = 0** (DDH03 eq. 6) by replacing the corresponding rows with the discrete curl expression for the tangential H components at the boundary. (An earlier ghost-node "Neumann on E" implementation was wrong and has been fixed — see tests in `test_bc_and_solver.py`.)

### Validated Benchmarks (all reproduced; runners in examples/)

1. **Figs. 2/3** (DDH03): whole-space magnetic dipole, error cancellation near
   transmitter and outer boundary (`benchmark_figs2_3.py`).
2. **Two-half-space VED vs Sommerfeld**: ~1% RMS in the conductive half-space.
3. **Fig. 9** (DDH03): resistive borehole + invaded zone + thin dipping
   anisotropic layer. The invaded **annulus replaces the layer** for r < 0.6 m
   (this resolved the long-standing discrepancy); averaged schemes then match
   the published curve at every resolution (`fig9_refinement.py`).
4. **Fig. 7** (DDH03): coupled solve reproduces the published x-dipole curves
   at exactly ×2.00±0.03 in both B_xx and B_xz — consistent with their y≥0
   half-domain doubling the on-plane source (pending author confirmation).
   Do not "fix" code to chase this factor of 2.

---

## Analytical Solution: Magnetic Dipole in Homogeneous Space

For a magnetic dipole **m** = m x̂ at the origin in homogeneous isotropic medium (σ, ε, μ), the magnetic field is:

```python
k² = -iω μ σ̇,   σ̇ = σ - iω ε,   kr = k * r
B_x = (μ m / 4π r³) * [(k²r² - 3ikr - 3) * (x/r)² + (k²r² - ikr - 1)] * exp(ikr)/(k²r²)
```

The full tensor Green's function is in `analytics.py`.

---

## Development Status

All core components are **done and tested** (run `python -m pytest`; 77 tests
should pass): grid + optimal geometric grids, curl operators, media builders
(scalar and full-tensor), 4-cluster sources, coupled and clustered solves,
four-cluster postprocessing reads, analytic references, and the benchmarks
listed above.

## Settled Conventions (do not re-litigate these)

1. **σ̇ = σ − iωε** under the exp(−iωt) convention. DDH03's printed "σ + iωε"
   after eq. 5 is a typo; deriving eq. 5 from eq. 4 forces the minus sign.
2. **The coupled single solve is THE method** and the default
   (`method='coupled'`). DDH03 themselves solve one coupled system with eq.-7
   sources in all four clusters; "average over clusters" means the four
   sub-grids of that single solution. Verified three independent ways
   (rotated-frame equivalence, Born-order argument, exact constant-field
   assembly probe at rel. err ~3e-10).
3. **Per-cluster-source ("clustered") solves are isotropic-only.** In coupled
   media they under-count cross-components (the response lives on partner
   sub-grids); a guard warns if misused. Locked in by
   `tests/test_anisotropic_coupling.py`.
4. **`from_geometry_func` was deleted.** The exact-geometry path is
   `from_geometry_exact(grid, sigma_func, geometry, method=...)`; the lookup
   path is `from_sigma_func`. Both take `method` ∈ {pointwise, backus, nodal}.
5. **Magnetic BC** rows enforce H×n = 0 (DDH03 eq. 6) — not naive ghost-node
   Neumann on E.

## Open Problem (the current research front)

Coupled-consistent **nodal homogenization**: in the Fig-9 layer case the full
nodal scheme (off-diagonals active, coupled solve) over-attenuates (geo-mean
1.46 vs paper ~0.97) while nodal-with-off-diagonals-zeroed matches. The
over-attenuation is carried by the *structure* of the nodal interface-cell
off-diagonals, not eigenvalue overshoot and not truncation. The
energy-matching derivation (Σ_D = L̃⁻ᵀGL̃⁻¹) needs re-posing against the
coupled Lebedev discrete structure. Numerical targets: reproduce pointwise
(0.97) at coarse resolution; reduce to current nodal in the uncoupled /
axis-aligned limits. See `docs/nodal_homogenization_3d.md` and the external
benchmark notes.

---

## References

- **DDH03**: Davydycheva, Druskin, Habashy (2003), Geophysics 68(5):1525–1536
- **Lebedev (1964)**: Difference analogies of orthogonal decompositions, Soviet Comput. Maths.
- **Ingerman, Druskin, Knizhnerman (2000)**: Optimal FD grids and rational approx. of square root, CPAM 53:1039–1066
- **Moskow et al. (1999)**: FD scheme for elliptic equations with rough coefficients, SIAM J. Numer. Anal. 36:442–464
- **Yee (1966)**: IEEE Trans. Ant. Prop., AP-14:302–307

# Why our code does not match DDH03 Fig. 9 — diagnosis (July 19, 2026)

Continuation of Open Problem #1 in the July-18 handoff. This note records an
independent, cheap reproduction of the phenomenon and sharpens the diagnosis.
Reproducer: `examples/fig9_tilt_diagnosis.py` (data + figure in `examples/out/`).

## The short answer

**The code *can* match Fig. 9 — the mismatch is specific to the
cell-homogenization scheme, and only when the thin layer is *tilted* relative
to the grid.** Assigning each node its pointwise conductivity reproduces the
paper (handoff: geo-mean 0.97). The published mismatch appears when the
sub-grid dipping layer is upscaled by a *cell-averaged* effective tensor:

* the **standard Backus/laminate** tensor **under-attenuates** Im B_z, and
* the **nodal (Moskow-energy) tensor over-attenuates** it,

with the true (pointwise-resolved) answer bracketed **between** the two. The
error is a *tilt × discretization* interaction: when the interface aligns with
the grid, all three schemes agree.

## Evidence (independent cheap reproduction)

Small uniform grid (16×16×40, h_x=0.31 m, h_z=0.15 m), isotropic background
σ=0.1, a 0.25 m-thick TI layer (σ_T=0.1, σ_N=σ_T/200) crossing the z-axis near
z=−1.2 m, z-directed **magnetic** dipole at the origin (52.65 kHz), fully
coupled single solve, Im B_z read on the z-axis. Geometric-mean ratio of Im B_z
to the pointwise reference over the on-axis layer region:

| scheme | dip = 0 (control) | dip = 75° (Fig-9 regime) |
|---|---|---|
| pointwise (reference) | 1.000 | 1.000 |
| Backus / laminate | 0.974 | **1.130** (under-attenuates) |
| nodal (full tensor) | 0.991 | **0.907** (over-attenuates) |
| nodal, off-diagonals zeroed | 0.991 | 0.927 |

The **dip=0 control is the key experiment**: with the interface perpendicular
to the axis (well resolved along z), pointwise ≈ Backus ≈ nodal to within 1–3%.
The nodal tensor correctly reduces to the axis-aligned arithmetic/harmonic
diagonal (note eq. Lemma 1). *All* of the scheme disagreement is created by
tilting the interface off the grid axes — i.e. it lives in the tilt-induced
off-axis structure of the effective tensor, exactly the regime the nodal lemma
must handle. See `examples/out/fig9_tilt_diagnosis.png`.

This reproduces the **sign and ordering** of the handoff's finding
(Backus under-, nodal over-attenuates, pointwise ≈ truth). Magnitudes are
milder than the handoff's finer runs (nodal/pointwise 0.907 here vs ≈0.66
implied by paper/ours = 1.46 there) because this grid is ~6× coarser in x, has
no borehole/invasion, and uses the SVD lookup path rather than exact geometry.

## Refined mechanism

**1. Backus under-attenuates because its normal conductivity is the *harmonic
mean* over the whole cell.** For a thin resistive layer of volume fraction f,
σ⊥ = 1/(f/σ_N + (1−f)/σ_BG); with f small this is far larger than σ_N, so the
cell barely blocks normal current and the layer's choking of the eddy currents
is under-represented → Im B_z too large.

**2. The nodal over-attenuation on this grid is carried *mostly by the
diagonal*, not the off-diagonals.** Zeroing the grid-axis off-diagonals of the
nodal tensor recovers only ~20% of the gap (0.907 → 0.927). This **differs from
the handoff's finer exact-geometry runs**, where zeroing the off-diagonals
essentially restored the match (0.94 vs full 1.46). Reconciliation: the tilt
error has *both* a diagonal and an off-diagonal component, and their relative
weight is grid- and path-dependent (coarse SVD-lookup here vs fine exact-normal
there). The lemma needs to address both, and the off-diagonal-dominance claim
should be re-scoped as "at the finer, exact-geometry resolution."

**3. Eigenvalue overshoot is real but localized.** In the on-axis analysis band
the nodal tensors have only a mild overshoot (λ_max/σ_max ≈ 1.0–1.1). A single
outlier cell at the far edge of the layer footprint (z=−2.85, off-axis of the
receivers) showed λ_max/σ_max = 4.87 — a bad line-fraction/SVD cell. Consistent
with the handoff: the overshoot exists but clamping it did not fix the
attenuation, so it is a *symptom of* the ill-posed straddling-cell construction,
not the direct cause.

## Why this happens — the coupled-consistency gap (the lemma)

The nodal tensor Σ_D = L̃⁻ᵀ G L̃⁻¹ is derived by matching the **discrete
Dirichlet energy of a scalar conductivity problem**, where the effective tensor
multiplies a **discrete gradient** ∇̃φ of a node potential, and L̃⁻¹ is exactly
the correction that undoes the finite-difference gradient's distortion.

In the Lebedev **Maxwell** problem the conductivity tensor does *not* multiply a
discrete gradient. It enters the reaction term −iω σ̇ E as a nodal 3×3 block
multiplying the **field vector E directly**, and its off-diagonals act as literal
inter-cluster couplings (value −iω σ_offdiag at shared R-nodes;
`tests/test_anisotropic_coupling.py`). Applying a gradient-calibrated L̃⁻¹
correction where no discrete gradient is taken injects a distortion the physics
does not ask for — over-attenuation for the tilted resistive layer.

The physically exact object for the reaction term is the continuum
current↔field relation ⟨J⟩ = Σ⟨E⟩ (standard homogenization, DDH03 eq. 9) — but
that is the Backus tensor, which *under*-attenuates. Neither the pure
current-based tensor nor the pure gradient-energy tensor is right; the true
coarse tensor is bracketed between them. **The lemma is to re-pose the
energy-matching against the coupled Lebedev *mass-term* discrete structure —
the nodal field values of all three E-components at the R-node, coupled across
clusters — rather than a single-cluster discrete gradient.**

## Concrete next steps

1. **Derivation (the lemma).** Redo Section 4/A.5 of
   `nodal_homogenization_3d.md` with the test/trial objects being the coupled
   Lebedev nodal field (three components per R-node on their partner sub-grids)
   under the reaction bilinear form E·σ̇E, not ∇̃φ·Σ_D∇̃φ. Target: a tensor
   that (a) reduces to the current nodal diagonal in the axis-aligned/uncoupled
   limit (dip=0 control ✓), and (b) lands inside the Backus–nodal bracket for
   the tilt.
2. **Cheap validation loop.** `examples/fig9_tilt_diagnosis.py` is a ~45 s
   arbiter for candidate tensors on this box: any proposal should push the
   tilt=75 nodal ratio from 0.907 toward 1.0 while leaving the dip=0 control
   untouched.
3. **Absolute arbiter (needs more compute).** Confirm against the paper on the
   real configuration (full optimal grid, borehole+invasion, h_min=0.05) via
   `examples/fig9_check.py`, and/or a fine pointwise reference at h_min≈0.02.
   These are ~10⁵-unknown iterative solves — heavier than this 2-CPU / 8 GB
   sandbox runs comfortably; better on a larger machine.
4. **Housekeeping.** `fig9_check.py` still uses the per-cluster (clustered)
   procedure; for the anisotropic layer it should read from the single coupled
   solve (same B for all cluster keys), matching the handoff's "rework to the
   coupled pipeline" action.

---

## Update (same day) — Backus fixed; the "correct cell" quantified

Two code-level corrections (both prompted by S. Moskow):

**1. `backus` was collapsing anisotropic layers to ⅓·tr σ.** `from_sigma_func`
(and via it `from_fine_grid`) built the Backus tensor from *scalar* volume
averages `_standard_backus_tensor_3d(⟨σ⟩, ⟨σ⁻¹⟩, n̂)` computed from the
trace/3 proxy. For the TI layer (σ_nn = σ_T/200) this used σ ≈ 0.067 as the
layer conductivity, ~40× too conductive in the normal direction — so the old
"Backus under-attenuates (ratio 1.13, and the handoff's 0.74)" was an
**artifact of the scalar proxy, not the real laminate**. Fixed by adding
`_anisotropic_backus_tensor_3d` (the full tensor laminate, DDH03 eq. 9),
verified to (a) reduce to the scalar formula for isotropic media, (b) equal
`_nodal_eff_tensor_general` with symmetric line fractions, and (c) give the
harmonic mean of the true σ_nn along the normal. Locked in by
`tests/test_anisotropic_backus.py` (4 tests; suite now 71).

**2. The reaction-term averages now use the width-h Voronoi cell.** The σ̇E
mass term's control volume is the node's width-h Voronoi cell, not the 2h
centered-difference box the (gradient-based) nodal line averages need. Backus
volume fractions are now taken over the Voronoi cell.

**Corrected numbers (this toy grid).** With correct tensors the whole picture
changes — Backus and nodal land on the *same* side and nearly coincide:

Analytic-geometry attribution (all straddling cells homogenized), ratio vs
pointwise at tilt = 75°:

| | 2h stencil cell | width-h Voronoi cell |
|---|---|---|
| anisotropic Backus | 0.456 | 0.612 |
| nodal | 0.463 | 0.645 |

End-to-end through the fixed `from_sigma_func` (SVD path): dip=0 control
backus = nodal = 0.991; tilt=75 backus = 0.939, nodal = 0.907 (was
backus = 1.130 before the fix).

**Consequences for Open Problem #1.**
* The handoff's "Backus under-attenuates / nodal over-attenuates, truth
  between" framing was partly an artifact: with the correct anisotropic
  laminate, **Backus ≈ nodal on the same cell** — both over-attenuate, driven
  by the harmonic σ_nn normal average they *share*. The over-attenuation is
  therefore **not** primarily carried by the nodal off-diagonals.
* The **cell is a large, consistent lever** (+0.15–0.18 toward truth for both
  schemes). This validates the "correct cell" hypothesis: the reaction-term
  conductivity should be homogenized over the nodal control volume.
* Residual over-attenuation remains even with the correct cell (analytic
  0.61–0.65), so the nodal formula itself still needs the coupled-consistent
  re-derivation — the next step. And whether pointwise is truly the reference
  on this coarse grid still needs a finer arbiter.

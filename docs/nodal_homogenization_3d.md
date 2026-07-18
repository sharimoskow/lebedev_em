# Three-Dimensional Nodal Homogenization for Arbitrarily Oriented and Multi-Interface Layered Media

**S. Moskow**

> This is the Markdown rendering of the technical note (for in-browser
> viewing on GitHub). A typeset PDF and the LaTeX source are alongside:
> [nodal_homogenization_3d.pdf](nodal_homogenization_3d.pdf),
> [nodal_homogenization_3d.tex](nodal_homogenization_3d.tex).

## Abstract

We derive the nodal homogenization tensor $\Sigma_D$ for a layered medium
with planar symmetry direction $\hat{\boldsymbol{n}}$ at arbitrary
orientation, and describe its implementation for Cartesian finite-difference
grids. Both the isotropic and fully anisotropic cases are treated within the
same energy-matching framework of Moskow et al. (1999), extended to three
spatial dimensions. In both cases the result takes the form
$\Sigma_D = \widetilde{L}^{-\top} G\,\widetilde{L}^{-1}$, where
$\widetilde{L}$ encodes the discrete gradient of the local solution space and
$G$ is an energy matrix built from volume averages of the medium. For an
isotropic profile the ingredients are
$\widetilde{L} = [\hat{\boldsymbol{m}}\mid\hat{\boldsymbol{q}}\mid D\hat{\boldsymbol{n}}]$,
with $D$ a diagonal matrix of per-axis line averages of $\sigma^{-1}$, and
$G = \mathrm{diag}(\bar\sigma,\bar\sigma,\langle\sigma^{-1}\rangle_{\mathrm{vol}})$.
For a tensor-valued profile, the columns of $\widetilde{L}$ acquire
off-diagonal corrections from the off-normal conductivity components, and the
upper-left block of $G$ is replaced by the volume average of the pointwise
Schur complement while $G_{33}$ becomes the volume average of
$1/\sigma_{nn}$.

Two practical computational paths are described. When the interface geometry
is known analytically, $\hat{\boldsymbol{n}}$ is supplied exactly and
averages are read from geometric fractions. When only a conductivity callable
is available, $\hat{\boldsymbol{n}}$ is estimated by fitting a plane to
interface-crossing voxel midpoints on a fine sub-grid (geometric SVD); a
planarity ratio detects unreliable estimates and triggers a diagonal
fallback. For tensor-valued callables the SVD uses
$\tfrac{1}{3}\mathrm{tr}\sigma$ as a scalar proxy.

Dual cells straddling two interfaces are handled by sequential nodal
homogenization: the two outer materials are homogenized first with the outer
boundary normal, then the result is homogenized against the sandwiched inner
material (identified by its median conductivity) with the inner boundary
normal. A fallback to single-interface treatment is applied when the
intermediate tensor has eigenvalues outside the physical conductivity range.

## 1. Setup

Let $H = [x_1^-,x_1^+]\times[x_2^-,x_2^+]\times[x_3^-,x_3^+]$ be a
rectangular dual cell (node-centered box) on a Cartesian grid with grid axes
$\hat{\boldsymbol{e}}_1,\hat{\boldsymbol{e}}_2,\hat{\boldsymbol{e}}_3$. We
consider an isotropic medium whose conductivity varies only along a fixed
direction $\hat{\boldsymbol{n}}$:

$$\sigma(\boldsymbol{x}) = \sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x}), \tag{1}$$

where $\hat{\boldsymbol{n}}$ is a unit vector at arbitrary orientation and
$\sigma:\mathbb{R}\to\mathbb{R}_{>0}$ is any measurable, bounded, and
bounded-away-from-zero function. This encompasses any number of planar layers
(not just one interface), as well as continuously varying profiles.

Let $\hat{\boldsymbol{m}}$ and $\hat{\boldsymbol{q}}$ be unit vectors
orthogonal to $\hat{\boldsymbol{n}}$ and to each other, so that
$\{\hat{\boldsymbol{m}},\hat{\boldsymbol{q}},\hat{\boldsymbol{n}}\}$ is an
orthonormal basis.

## 2. Local Solution Space

Following Moskow et al. (1999), we approximate the electric potential locally
by functions whose gradient satisfies the interface transmission conditions
exactly at every level set of $\hat{\boldsymbol{n}}\cdot\boldsymbol{x}$: the
tangential electric field and the normal current density are both constant in
the $\hat{\boldsymbol{n}}$ direction. This motivates the local solution space

$$L(H) = \mathrm{span}\bigl\{1,\;\hat{\boldsymbol{m}}\cdot\boldsymbol{x},\;\hat{\boldsymbol{q}}\cdot\boldsymbol{x},\;\varphi_n(\boldsymbol{x})\bigr\},$$

where

$$\varphi_n(\boldsymbol{x}) = \int_0^{\hat{\boldsymbol{n}}\cdot\boldsymbol{x}} \frac{dt}{\sigma(t)}$$

is the "normal potential," which carries all the variability of $\sigma$
across layers. Note that $\varphi_n$ is well-defined for any $\sigma$
satisfying (1), regardless of the number of layers or smoothness of the
profile.

## 3. True and Discrete Gradients

**True gradient.** For $\phi\in L(H)$ with coefficient vector
$\boldsymbol{a}=(a_1,a_2,a_3)^\top$,

$$\phi(\boldsymbol{x}) = a_1(\hat{\boldsymbol{m}}\cdot\boldsymbol{x}) + a_2(\hat{\boldsymbol{q}}\cdot\boldsymbol{x}) + a_3\,\varphi_n(\boldsymbol{x}),$$

the true gradient is

$$\nabla\phi = a_1\,\hat{\boldsymbol{m}} + a_2\,\hat{\boldsymbol{q}} + \frac{a_3}{\sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})}\,\hat{\boldsymbol{n}}. \tag{2}$$

The three coefficients have clear physical meaning: $a_1$ and $a_2$ are the
(constant) tangential electric field components, and $a_3$ is the (constant)
normal current density $J_n = \sigma E_n$.

**Discrete gradient.** The finite-difference approximation to the $k$-th
component of $\nabla\phi$ along grid axis $\hat{\boldsymbol{e}}_k$ is the
average of $\partial\phi/\partial x_k$ across the corresponding edge. Since
the tangential terms contribute constants $a_1\hat{m}_k$ and $a_2\hat{q}_k$,
only the normal potential needs care:

$$\frac{1}{\Delta x_k}\int_{x_k^-}^{x_k^+} \frac{\partial\varphi_n}{\partial x_k}\,dx_k = \hat{n}_k\, d_k,$$

where

$$d_k = \left\langle\sigma^{-1}\right\rangle_k := \frac{1}{\Delta x_k}\int_{x_k^-}^{x_k^+} \frac{dx_k}{\sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})}\Bigg|_{\text{other coords at node}} \tag{3}$$

is the line average of $\sigma^{-1}$ along axis $\hat{\boldsymbol{e}}_k$
**through the node**. This integral is taken at fixed coordinates
perpendicular to $\hat{\boldsymbol{e}}_k$ and is well-defined for any profile
$\sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})$. In the special case of $M$
planar layers with conductivities $\sigma_1,\ldots,\sigma_M$ and fractional
lengths $f_k^{(j)}$ along edge $k$, it reduces to
$d_k = \sum_j f_k^{(j)}/\sigma_j$.

Assembling all three components, the discrete gradient is

$$\widetilde{\nabla}\phi = a_1\,\hat{\boldsymbol{m}} + a_2\,\hat{\boldsymbol{q}} + a_3\,D\hat{\boldsymbol{n}}, \qquad D = \mathrm{diag}(d_1,d_2,d_3), \tag{4}$$

or in matrix form

$$\widetilde{\nabla}\phi = \widetilde{L}\,\boldsymbol{a}, \qquad \widetilde{L} := \bigl[\hat{\boldsymbol{m}}\;\big|\;\hat{\boldsymbol{q}}\;\big|\;D\hat{\boldsymbol{n}}\bigr] \in \mathbb{R}^{3\times 3}.$$

## 4. Energy Matching Condition

We seek an effective tensor $\Sigma_D$ such that, for every
$\phi,\psi\in L(H)$,

$$|H|\;\widetilde{\nabla}\phi\cdot\Sigma_D\,\widetilde{\nabla}\psi = \int_H \nabla\phi\cdot\sigma\,\nabla\psi\;dV. \tag{5}$$

**Right-hand side.** Using (2) and the orthonormality of
$\{\hat{\boldsymbol{m}},\hat{\boldsymbol{q}},\hat{\boldsymbol{n}}\}$,

$$\nabla\phi\cdot\sigma\nabla\psi = \sigma\bigl(a_1 b_1 + a_2 b_2\bigr) + \frac{a_3 b_3}{\sigma},$$

so

$$\int_H \nabla\phi\cdot\sigma\nabla\psi\;dV = |H|\bigl(\bar\sigma\,(a_1 b_1+a_2 b_2) + \langle\sigma^{-1}\rangle_{\mathrm{vol}}\, a_3 b_3\bigr) = |H|\;\boldsymbol{a}^\top G\,\boldsymbol{b},$$

where

$$G = \mathrm{diag}\bigl(\bar\sigma,\;\bar\sigma,\;\langle\sigma^{-1}\rangle_{\mathrm{vol}}\bigr), \tag{6}$$

with

$$\bar\sigma = \langle\sigma\rangle_{\mathrm{vol}} := \frac{1}{|H|}\int_H \sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})\,dV, \qquad \langle\sigma^{-1}\rangle_{\mathrm{vol}} := \frac{1}{|H|}\int_H \frac{dV}{\sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})},$$

the volume averages of $\sigma$ and $\sigma^{-1}$ over $H$, respectively.
Both are well-defined for any measurable bounded profile (1). Note that
$\langle\sigma^{-1}\rangle_{\mathrm{vol}} = 1/\widetilde\sigma$, where
$\widetilde\sigma$ is the harmonic mean of $\sigma$. The key observation is
that $G_{33} = \langle\sigma^{-1}\rangle_{\mathrm{vol}}$ is the
**arithmetic** mean of the resistivity — not the harmonic mean of $\sigma$ —
a distinction that is critical when
$D\hat{\boldsymbol{n}} \not\propto \hat{\boldsymbol{n}}$.

**Left-hand side.** With $\widetilde{\nabla}\phi = \widetilde{L}\boldsymbol{a}$
and $\widetilde{\nabla}\psi = \widetilde{L}\boldsymbol{b}$,

$$|H|\;\widetilde{\nabla}\phi\cdot\Sigma_D\,\widetilde{\nabla}\psi = |H|\;\boldsymbol{a}^\top\widetilde{L}^\top\Sigma_D\,\widetilde{L}\,\boldsymbol{b}.$$

**Condition.** Equating both sides for all $\boldsymbol{a},\boldsymbol{b}$
gives $\widetilde{L}^\top\Sigma_D\,\widetilde{L} = G$, and solving for
$\Sigma_D$:

$$\boxed{\;\Sigma_D = \widetilde{L}^{-\top}\,G\,\widetilde{L}^{-1}, \qquad \widetilde{L} = [\hat{\boldsymbol{m}}\mid\hat{\boldsymbol{q}}\mid D\hat{\boldsymbol{n}}], \qquad G = \mathrm{diag}\bigl(\bar\sigma,\;\bar\sigma,\;\langle\sigma^{-1}\rangle_{\mathrm{vol}}\bigr).\;} \tag{7}$$

Here $\widetilde{L}^{-\top} := (\widetilde{L}^\top)^{-1} = (\widetilde{L}^{-1})^\top$.

## 5. Verification and Special Cases

**Lemma 1 (Axis-aligned interface).** *For
$\hat{\boldsymbol{n}} = \hat{\boldsymbol{e}}_3$, formula (7) reduces to the
standard arithmetic/harmonic tensor*

$$\Sigma_D = \mathrm{diag}(\bar\sigma,\;\bar\sigma,\;\widetilde\sigma).$$

*Proof.* Choose $\hat{\boldsymbol{m}}=\hat{\boldsymbol{e}}_1$,
$\hat{\boldsymbol{q}}=\hat{\boldsymbol{e}}_2$. Then
$D\hat{\boldsymbol{n}} = d_3\hat{\boldsymbol{e}}_3$ and
$\widetilde{L} = \mathrm{diag}(1,1,d_3)$. Since
$\hat{\boldsymbol{n}} = \hat{\boldsymbol{e}}_3$, lines along
$\hat{\boldsymbol{e}}_1$ and $\hat{\boldsymbol{e}}_2$ are parallel to all
level sets of $\sigma$, so $d_1 = d_2$ each equal the pointwise value
$\sigma(\boldsymbol{x}_{\mathrm{node}})^{-1}$ (no variation along those
edges). The $\hat{\boldsymbol{e}}_3$ edge varies through the full profile,
giving
$d_3 = \langle\sigma^{-1}\rangle_z = \langle\sigma^{-1}\rangle_{\mathrm{vol}} = 1/\widetilde\sigma$
(the line average along $z$ equals the volume average when the profile
depends only on $z$). Therefore
$\widetilde{L}^{-1} = \mathrm{diag}(1,1,\widetilde\sigma)$, and

$$\Sigma_D = \mathrm{diag}(1,1,\widetilde\sigma)\cdot \mathrm{diag}(\bar\sigma,\bar\sigma,1/\widetilde\sigma)\cdot \mathrm{diag}(1,1,\widetilde\sigma) = \mathrm{diag}(\bar\sigma,\;\bar\sigma,\;\widetilde\sigma). \qquad\blacksquare$$

**Lemma 2 (Symmetric straddling).** *If $D\hat{\boldsymbol{n}}$ is parallel
to $\hat{\boldsymbol{n}}$ (i.e. $D = d\,I$ for some scalar $d$), then*

$$\Sigma_D = \Sigma_L := \bar\sigma(I - \hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top) + \widetilde\sigma\,\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top.$$

*Proof.* When $D\hat{\boldsymbol{n}} = d\,\hat{\boldsymbol{n}}$ the columns
of $\widetilde{L}$ are
$\hat{\boldsymbol{m}}, \hat{\boldsymbol{q}}, d\hat{\boldsymbol{n}}$ — an
orthogonal set with squared norms $1,1,d^2$. Hence $\widetilde{L}^{-1}$ has
rows $\hat{\boldsymbol{m}}^\top$, $\hat{\boldsymbol{q}}^\top$,
$\hat{\boldsymbol{n}}^\top/d$, giving
$\widetilde{L}^{-\top} = [\hat{\boldsymbol{m}}\mid\hat{\boldsymbol{q}}\mid\hat{\boldsymbol{n}}/d]$.
Then

$$\Sigma_D = [\hat{\boldsymbol{m}}\mid\hat{\boldsymbol{q}}\mid\hat{\boldsymbol{n}}/d]\,\mathrm{diag}(\bar\sigma,\bar\sigma,1/\widetilde\sigma) \begin{pmatrix}\hat{\boldsymbol{m}}^\top\\ \hat{\boldsymbol{q}}^\top\\ \hat{\boldsymbol{n}}^\top/d\end{pmatrix} = \bar\sigma\,\hat{\boldsymbol{m}}\hat{\boldsymbol{m}}^\top + \bar\sigma\,\hat{\boldsymbol{q}}\hat{\boldsymbol{q}}^\top + \frac{1}{\widetilde\sigma d^2}\,\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top.$$

Since $d = 1/\widetilde\sigma$ (the line average equals the volume average
when $D\propto I$), the last coefficient equals $\widetilde\sigma$, and
$\hat{\boldsymbol{m}}\hat{\boldsymbol{m}}^\top + \hat{\boldsymbol{q}}\hat{\boldsymbol{q}}^\top = I - \hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top$,
giving $\Sigma_D = \Sigma_L$. $\blacksquare$

**Remark 1.** Lemma 2 shows that the nodal correction is zero for any node at
which $D\hat{\boldsymbol{n}}\propto\hat{\boldsymbol{n}}$ — equivalently, when
$d_x\hat{n}_x : d_z\hat{n}_z$ is constant along the direction of
$\hat{\boldsymbol{n}}$. For a tilted interface
$\hat{\boldsymbol{n}}=[1,0,1]/\sqrt{2}$, this occurs when $d_x = d_z$, i.e.
when the edge fractions in the $x$- and $z$-directions are equal. The
correction is non-trivial only for nodes where the interface cuts those edges
asymmetrically.

## 6. Relation to the Two-Dimensional Formula

Moskow et al. (1999) derive, for the two-dimensional isotropic case with
normal $\hat{\boldsymbol{n}}$ and single tangential $\hat{\boldsymbol{m}}$,
the tensor

$$\Sigma_D^{(2)} = K\,\Sigma_L^{(2)}\,K^\top,\qquad K = L^{-1},$$

where $L$ is a $2\times2$ matrix (their eq. (A.21)) and
$\Sigma_L^{(2)} = \mathrm{diag}(\bar\sigma,\widetilde\sigma)$ in the
$(\hat{\boldsymbol{m}},\hat{\boldsymbol{n}})$ basis.

In 3D the same energy-matching argument gives an identical structure. The
difference is that in 3D there are **two** tangential directions, each
contributing an $\bar\sigma$ term to the diagonal of $G$. Writing $\Sigma_L$
for the standard $3\times3$ arithmetic/harmonic tensor and noting that
$G = \widetilde{L}^\top\Sigma_L\,\widetilde{L}$ holds only when
$\widetilde{L}$ is orthogonal (i.e.
$D\hat{\boldsymbol{n}} \propto \hat{\boldsymbol{n}}$), one sees that the 3D
formula (7) is not generally equivalent to $K\Sigma_L K^\top$ with
$K=\widetilde{L}^{-1}$ — it is the direct energy-matching result, which
happens to coincide with $K\Sigma_L K^\top$ only in the symmetric case.

## 7. Anisotropic Extension

We now extend the derivation to the case where the conductivity varies along
$\hat{\boldsymbol{n}}$ as a **tensor-valued** profile:
$\sigma = \sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})$, where for each
$t\in\mathbb{R}$ the value $\sigma(t)$ is a symmetric positive-definite
matrix. This includes any number of planar layers as well as continuously
varying anisotropy. We write

$$\sigma_{mn}(t) := \hat{\boldsymbol{m}}^\top\sigma(t)\hat{\boldsymbol{n}}, \quad \sigma_{qn}(t) := \hat{\boldsymbol{q}}^\top\sigma(t)\hat{\boldsymbol{n}}, \quad \sigma_{nn}(t) := \hat{\boldsymbol{n}}^\top\sigma(t)\hat{\boldsymbol{n}} > 0,$$

and use $t = \hat{\boldsymbol{n}}\cdot\boldsymbol{x}$ as the layering
coordinate. This section follows the same structure as Appendix A of Moskow
et al. (1999), extended from 2D to 3D by the second tangential direction
$\hat{\boldsymbol{q}}$.

### A.1 Interface constants

For any layered anisotropic medium of the form
$\sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})$, the interface transmission
conditions at every level set
$\{\hat{\boldsymbol{n}}\cdot\boldsymbol{x}=t\}$ are continuity of the
tangential electric field and of the normal current density. These motivate
three constants that are uniform across all layers:

$$c_1 := \hat{\boldsymbol{m}}\cdot\nabla u, \qquad c_2 := \hat{\boldsymbol{q}}\cdot\nabla u, \qquad c_3 := \hat{\boldsymbol{n}}\cdot(\sigma\nabla u), \tag{A.1–A.3}$$

the two tangential field components and the normal current density $J_n$.

### A.2 Electric field at each level

At each level $t$, the condition
$c_3 = \sigma_{mn}(t)c_1 + \sigma_{qn}(t)c_2 + \sigma_{nn}(t)E_n$ determines
the normal field:

$$E_n(t) = \frac{c_3 - \sigma_{mn}(t)\,c_1 - \sigma_{qn}(t)\,c_2}{\sigma_{nn}(t)}. \tag{A.4}$$

The full gradient at level $t$ is therefore

$$\nabla u\big|_{t} = c_1\,\hat{\boldsymbol{m}} + c_2\,\hat{\boldsymbol{q}} + E_n(t)\,\hat{\boldsymbol{n}}, \tag{A.5}$$

and the triple $(c_1,c_2,c_3)$ completely characterizes the local potential.

### A.3 Discrete gradient

The finite-difference approximation to $(\nabla u)_k$ along grid axis $k$
averages (A.5) over the corresponding edge at fixed transverse coordinates.
Because $c_1,c_2$ are constant, only $E_n(t)$ varies along the edge.
Substituting (A.4) and collecting terms in $c_1,c_2,c_3$:

$$(\widetilde{\nabla} u)_k = c_1\left(\hat{m}_k - \hat{n}_k\,[D_2]_{kk}\right) + c_2\left(\hat{q}_k - \hat{n}_k\,[D_3]_{kk}\right) + c_3\,\hat{n}_k\,[D_1]_{kk}, \tag{A.7}$$

where the per-axis line averages, each a 1D integral along grid axis $k$
through the node with the two transverse coordinates held fixed at their node
values, are

$$[D_1]_{kk} := \frac{1}{\Delta x_k}\int_{x_k^-}^{x_k^+} \frac{dx_k}{\sigma_{nn}(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})}, \qquad [D_2]_{kk} := \frac{1}{\Delta x_k}\int_{x_k^-}^{x_k^+} \frac{\sigma_{mn}}{\sigma_{nn}}\,dx_k, \qquad [D_3]_{kk} := \frac{1}{\Delta x_k}\int_{x_k^-}^{x_k^+} \frac{\sigma_{qn}}{\sigma_{nn}}\,dx_k. \tag{A.8–A.10}$$

Because
$\hat{\boldsymbol{n}}\cdot\boldsymbol{x} = \hat{n}_k x_k + \sum_{j\neq k}\hat{n}_j x_j^{\mathrm{node}}$,
the integrand is a function of $x_k$ alone, so the integrals are well-defined
for any measurable profile. This is the exact anisotropic analogue of $d_k$
in (3): $D_1$ is the per-axis line average of $1/\sigma_{nn}$; $D_2$ and
$D_3$ are the line averages of the ratios $\sigma_{mn}/\sigma_{nn}$ and
$\sigma_{qn}/\sigma_{nn}$. Assembling all three Cartesian components:

$$\widetilde{\nabla} u = \widetilde{L}\,\boldsymbol{c}, \qquad \widetilde{L} := \bigl[\,\hat{\boldsymbol{m}} - D_2\hat{\boldsymbol{n}} \;\big|\; \hat{\boldsymbol{q}} - D_3\hat{\boldsymbol{n}} \;\big|\; D_1\hat{\boldsymbol{n}}\,\bigr], \qquad \boldsymbol{c} = (c_1, c_2, c_3)^\top, \tag{A.11}$$

where $D_j\hat{\boldsymbol{n}}$ denotes the vector with $k$-th entry
$[D_j]_{kk}\hat{n}_k$.

**Remark (2D reduction).** In 2D with a single tangential direction
$\hat{\boldsymbol{m}}$, there is no $c_2$ or $D_3$, and $D_2$ reduces to the
$2\times 2$ diagonal matrix of line averages of $\sigma_{mn}/\sigma_{nn}$.
Then
$\widetilde{L} = [\hat{\boldsymbol{m}} - D_2\hat{\boldsymbol{n}} \mid D_1\hat{\boldsymbol{n}}]$,
matching eq. (A.8) of Moskow et al. (1999).

### A.4 Pointwise Schur complement and energy decomposition

At each level $t$, define the **pointwise Schur complement** of $\sigma(t)$
with respect to $\hat{\boldsymbol{n}}$:

$$S(t) := \sigma(t) - \frac{(\sigma(t)\hat{\boldsymbol{n}})\,(\hat{\boldsymbol{n}}^\top\sigma(t))}{\sigma_{nn}(t)}. \tag{A.12}$$

$S(t)$ is symmetric positive-semidefinite with null vector
$\hat{\boldsymbol{n}}$ for every $t$. Substituting (A.4)–(A.5) into the
pointwise energy density and using the symmetry of $\sigma(t)$, the
normal–tangential cross-terms cancel and one obtains the clean factorization

$$\nabla u\cdot\sigma(t)\nabla u = \boldsymbol{\xi}^\top\bigl(\hat{\boldsymbol{m}}\;\hat{\boldsymbol{q}}\bigr)^\top S(t)\bigl(\hat{\boldsymbol{m}}\;\hat{\boldsymbol{q}}\bigr)\boldsymbol{\xi} + \frac{c_3^2}{\sigma_{nn}(t)}, \qquad \boldsymbol{\xi} = (c_1,c_2)^\top. \tag{A.13}$$

The tangential and normal contributions are **decoupled** pointwise.
Integrating over $H$:

$$\frac{1}{|H|}\int_H \nabla u\cdot\sigma\nabla u\;dV = \boldsymbol{c}^\top G\,\boldsymbol{c}, \tag{A.15}$$

where the $3\times 3$ energy matrix in the $(c_1,c_2,c_3)$ basis is

$$G = \begin{pmatrix} \hat{\boldsymbol{m}}^\top G_{TT}\hat{\boldsymbol{m}} & \hat{\boldsymbol{m}}^\top G_{TT}\hat{\boldsymbol{q}} & 0 \\ \hat{\boldsymbol{q}}^\top G_{TT}\hat{\boldsymbol{m}} & \hat{\boldsymbol{q}}^\top G_{TT}\hat{\boldsymbol{q}} & 0 \\ 0 & 0 & G_{nn} \end{pmatrix}, \tag{A.16}$$

with volume averages

$$G_{TT} := \frac{1}{|H|}\int_H S(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})\,dV, \qquad G_{nn} := \frac{1}{|H|}\int_H \frac{dV}{\sigma_{nn}(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})}.$$

That is, $G_{TT}$ is the **arithmetic** volume average of the pointwise Schur
complement, and $G_{nn}$ is the **arithmetic** volume average of the normal
resistivity $1/\sigma_{nn}$ — the anisotropic analogue of
$\langle\sigma^{-1}\rangle_{\mathrm{vol}}$ in the isotropic case. The zeros
in (A.16) follow from $S(t)\hat{\boldsymbol{n}} = 0$ for all $t$, which gives
$G_{TT}\hat{\boldsymbol{n}} = 0$.

### A.5 Effective tensor

The energy-matching condition
$\widetilde{L}^\top\Sigma_D\,\widetilde{L} = G$ (Section 4) gives

$$\boxed{\;\Sigma_D = \widetilde{L}^{-\top}\,G\,\widetilde{L}^{-1},\;} \tag{A.17}$$

with $\widetilde{L}$ and $G$ defined by (A.11) and (A.16). Equivalently,
writing $K = \widetilde{L}^{-1}$, $\Sigma_D = K^\top G K$ — the exact 3D
analogue of eq. (A.14) in Moskow et al. (1999). All ingredients
($D_1,D_2,D_3$ and the averages $G_{TT},G_{nn}$) are well-defined for any
measurable tensor-valued profile
$\sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})$.

**Remark (piecewise-constant reduction).** For a two-layer medium with
$\sigma^{(1)},\sigma^{(2)}$ and volume fractions $f$, $1-f$ (and line
fractions $f_k$, $1-f_k$), the integrals reduce to

$$[D_1]_{kk} = \frac{f_k}{\sigma_{nn}^{(1)}} + \frac{1-f_k}{\sigma_{nn}^{(2)}}, \qquad [D_2]_{kk} = \frac{f_k\,\sigma_{mn}^{(1)}}{\sigma_{nn}^{(1)}} + \frac{(1-f_k)\,\sigma_{mn}^{(2)}}{\sigma_{nn}^{(2)}}, \qquad G_{TT} = f\,S^{(1)} + (1-f)\,S^{(2)}, \qquad G_{nn} = \frac{f}{\sigma_{nn}^{(1)}} + \frac{1-f}{\sigma_{nn}^{(2)}},$$

and similarly for $D_3$.

**Remark (isotropic reduction).** When $\sigma(t)=\sigma(t)\,I$ is scalar,
$\sigma_{mn}=\sigma_{qn}=0$ so $D_2=D_3=0$. Then
$\widetilde{L}=[\hat{\boldsymbol{m}}\mid\hat{\boldsymbol{q}}\mid D_1\hat{\boldsymbol{n}}]$
with $[D_1]_{kk}=d_k$ the isotropic line average, recovering (4). The Schur
complement is
$S(t)=\sigma(t)(I-\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top)$, giving
$G_{TT}=\bar\sigma(I-\hat{\boldsymbol{n}}\hat{\boldsymbol{n}}^\top)$ and
$G_{nn}=\langle\sigma^{-1}\rangle_{\mathrm{vol}}$, recovering (6).

**Remark (TI medium at oblique interface).** For a transversely isotropic
(TI) medium
$\sigma=\sigma_T(I-\hat{\boldsymbol{t}}\hat{\boldsymbol{t}}^\top)+\sigma_N\hat{\boldsymbol{t}}\hat{\boldsymbol{t}}^\top$
with symmetry axis $\hat{\boldsymbol{t}}$, and with interface normal
$\hat{\boldsymbol{n}}$ not aligned with $\hat{\boldsymbol{t}}$, the ratios
$\sigma_{mn}/\sigma_{nn}$ and $\sigma_{qn}/\sigma_{nn}$ are nonzero, making
$D_2$ and $D_3$ nontrivial. Omitting them leads to a $k$-dependent crossing
position in borehole logging simulations.

## 8. Summary

The three-dimensional nodal homogenization tensor for a planar interface with
unit normal $\hat{\boldsymbol{n}}$ is
$\Sigma_D = \widetilde{L}^{-\top}\,G\,\widetilde{L}^{-1}$ with:

| Quantity | Definition | Meaning |
|---|---|---|
| $D$ | $\mathrm{diag}(d_1,d_2,d_3)$, $\;d_k = \frac{1}{\Delta x_k}\int_{x_k^-}^{x_k^+} \frac{dx_k}{\sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})}$ | per-axis line avg of $\sigma^{-1}$ |
| $\widetilde{L}$ | $[\hat{\boldsymbol{m}}\mid\hat{\boldsymbol{q}}\mid D\hat{\boldsymbol{n}}]$, with $\hat{\boldsymbol{m}},\hat{\boldsymbol{q}}\perp\hat{\boldsymbol{n}}$ orthonormal | discrete-gradient map |
| $G$ | $\mathrm{diag}(\bar\sigma,\bar\sigma,\langle\sigma^{-1}\rangle_{\mathrm{vol}})$ | energy matrix |
| $\bar\sigma$ | $\frac{1}{\|H\|}\int_H \sigma\,dV$ | volume avg of $\sigma$ |
| $\langle\sigma^{-1}\rangle_{\mathrm{vol}}$ | $\frac{1}{\|H\|}\int_H \sigma^{-1}\,dV$ | volume avg of $\sigma^{-1}$ |

All quantities are well-defined for any measurable profile
$\sigma = \sigma(\hat{\boldsymbol{n}}\cdot\boldsymbol{x})$, with any number
of layers or continuous variation. The key point is that $G_{33}$ is the
**arithmetic mean of the resistivity** over the dual-cell volume, not the
harmonic mean of $\sigma$. This value arises from the integral
$\int_H \sigma\,|E_n|^2\,dV = \int_H \sigma^{-1}\,|J_n|^2\,dV$ when
$(a_1,a_2,a_3)=(0,0,1)$.

**Anisotropic case.** For a tensor-valued layered profile, the same formula
holds with $\widetilde{L} = [\hat{\boldsymbol{m}} - D_2\hat{\boldsymbol{n}} \mid \hat{\boldsymbol{q}} - D_3\hat{\boldsymbol{n}} \mid D_1\hat{\boldsymbol{n}}]$
(line averages of $1/\sigma_{nn}$, $\sigma_{mn}/\sigma_{nn}$,
$\sigma_{qn}/\sigma_{nn}$), $G_{TT}$ the volume average of the pointwise
Schur complement $S(t)$, and $G_{nn}$ the volume average of
$1/\sigma_{nn}$, as in Section 7. The isotropic case is recovered when
$\sigma(t) = \sigma(t)I$.

## 9. Practical Application: Two Computational Paths

The derivation above assumes $\sigma$ is exactly layered with known normal
$\hat{\boldsymbol{n}}$ inside the dual cell $H$. In practice two situations
arise.

### 9.1 Case 1 — Interface geometry analytically known

When the interface is described by an explicit geometric object (a planar
boundary, a cylinder, a level-set surface), the unit normal
$\hat{\boldsymbol{n}}$ at each dual-cell node is available analytically. The
procedure is direct: evaluate $\sigma$ (scalar or tensor) on a fine isotropic
sub-grid within $H$ using the exact geometry; compute the volume fraction $f$
of each material and the per-axis line fractions $f_k$ along the central
grid-axis lines; read the line averages
$d_k = f_k/\sigma_1 + (1-f_k)/\sigma_2$ (with the appropriate
$\sigma_{nn}^{(i)}$ etc. in the anisotropic case); compute
$\bar\sigma = f\sigma_1 + (1-f)\sigma_2$ and
$\langle\sigma^{-1}\rangle_{\mathrm{vol}} = f/\sigma_1 + (1-f)/\sigma_2$; and
apply (7) with the known $\hat{\boldsymbol{n}}$. No interface-direction
estimation is needed. This path is exact for planar interfaces and
$O(h^2)$-accurate for curved ones.

### 9.2 Case 2 — Only σ available as a lookup function

When $\sigma(\boldsymbol{x})$ is given only as a black-box callable, the
direction $\hat{\boldsymbol{n}}$ must be **estimated** from the local
variation of $\sigma$ within $H$, and the averages computed by quadrature.

**Step 1 — Sub-grid evaluation.** Evaluate $\sigma$ on a fine *isotropic*
sub-grid of spacing $h_{\mathrm{svd}}$ within $H$. Equal spacing in all three
directions eliminates sampling-induced bias in the SVD step. The same block
is reused for normal estimation, volume averages, and line averages.

**Step 2 — Normal estimation by SVD.** For tensor-valued $\sigma$, the scalar
proxy $\tfrac{1}{3}\mathrm{tr}\sigma$ is used for interface detection
(the tensor values are retained for the homogenization integrals), so
detection responds to conductivity *magnitude*, not anisotropy *direction*.
The normal is estimated geometrically — not from finite differences, which
are biased when the cell aspect ratio is far from 1: binarise the block at
the mid-conductivity; flag face-adjacent voxel pairs straddling the
interface; record each pair's physical midpoint (placing samples on the
interface avoids a flooding bias near corners); fit a plane through the
midpoints by SVD, taking the minimum-variance right singular vector as
$\hat{\boldsymbol{n}}$, oriented so the positive side has higher mean
conductivity. The **planarity ratio** $r = s_3/s_1$ measures confidence:
$r \approx 0$ for a clean planar interface, $r \approx 1$ for isotropic
scatter. If $r \ge \tau$ (default $\tau = 0.7$) the cell falls back to the
diagonal (axis-aligned arithmetic/harmonic) formula; otherwise the nodal
formula (7) is applied.

**Steps 3–5.** Line averages by 1-D quadrature along each grid axis through
the node (default 50 points, likewise for the anisotropic integrands);
volume averages from the same isotropic sub-grid (more accurate for
thin-sliver cells than a separate coarse grid); assemble $\Sigma_D$ by (7).

The lookup-function path is fully matrix-free: it calls
$\sigma(\boldsymbol{x})$ on demand at whatever resolution each dual cell
requires, without storing a global fine-grid array.

## 10. Multi-Interface Cells: Sequential Nodal Homogenization

In practical geometries a dual cell may straddle **two or more** planar (or
approximately planar) interfaces simultaneously. A common example in borehole
logging is a cell containing three materials — borehole fluid, invasion zone,
and undisturbed formation — separated by the borehole wall (inner boundary,
normal $\hat{\boldsymbol{n}}_{\mathrm{in}}$) and the invasion front (outer
boundary, normal $\hat{\boldsymbol{n}}_{\mathrm{out}}$). A single interface
normal cannot represent both boundaries at once.

**Multi-material identification (lookup-function path).** The sub-grid block
already evaluated for normal estimation is rounded to six significant figures
and its unique conductivity levels extracted; a cell with $M \ge 3$ levels is
flagged as multi-material. For each pair of adjacent levels — $(0,1)$,
$(1,2)$, and the *skip pair* $(0,2)$ — a masked sub-SVD (adaptive tolerance
$0.4\times$ the gap to the nearest other level) returns a normal
$\hat n_{ij}$ and planarity ratio $r_{ij}$; a pair is *resolved* if
$r_{ij} < \tau$. Then: if all resolved normals are mutually parallel
(dot products $> 0.90$), the three materials are co-planar strata and the
nodal formula is applied once via a direct multi-region extension; if the
adjacent-pair normals are parallel but the skip pair's differs, there are two
distinct interfaces and sequential nodal homogenization applies, with the
primary (inner) normal the average of $\hat n_{01}, \hat n_{12}$ and the
secondary (outer) normal $\hat n_{02}$; otherwise the structure is ambiguous
and the cell falls back to the single-interface path with the global SVD
normal.

**Sequential nodal homogenization for M = 3 materials.** Label the materials
in sorted conductivity order $\sigma_1 < \sigma_2 < \sigma_3$ with volume
fractions $f_1, f_2, f_3$. The *inner* (sandwiched) material is always the
one with the **middle** conductivity $\sigma_2$; the outer pair is
$\sigma_1, \sigma_3$. Both steps apply the same 2-material nodal formula (7)
— there is no switch to a simpler laminate formula at either level:

1. **Outer homogenization:** apply (7) to $(\sigma_1, f_1)$ and
   $(\sigma_3, f_3)$ with normal $\hat{\boldsymbol{n}}_{\mathrm{out}}$, in
   proportions $f_1/(f_1{+}f_3)$ and $f_3/(f_1{+}f_3)$, with line fractions
   from the central-axis lines restricted to the outer-material voxels,
   yielding $\Sigma_D^{(13)}$.
2. **Inner homogenization:** treat $\Sigma_D^{(13)}$ as one effective
   material and $\sigma_2$ as the other; apply (7) with normal
   $\hat{\boldsymbol{n}}_{\mathrm{in}}$ and volume fractions $f_2$ and
   $f_1{+}f_3$, with line fractions from the full cell's central-axis lines.

**Interface normals.** In the exact-geometry path both normals are supplied
analytically (e.g. the cylinder's outward normal and the dip plane's normal
at the node); in the lookup-function path they come from the per-pair
sub-SVDs above.

**Fallback.** If the sequential result has a maximum eigenvalue exceeding
three times the largest material conductivity (a symptom of near-degeneracy
when one material has very low normal conductivity and dominates $G_{nn}$),
the algorithm falls back to the standard single-interface 2-material nodal
formula using the best-separating normal among the candidates.

**Remark (nesting convention).** The order of the two steps matters. The
convention "outer materials first" is chosen so the inner sandwiched material
is always homogenized last, against a background that already reflects the
outer structure. For DDH03-type geometries (cylindrical borehole inside a
dipping formation), $\hat{\boldsymbol{n}}_{\mathrm{in}}$ is the cylinder
normal and $\hat{\boldsymbol{n}}_{\mathrm{out}}$ the formation dip-plane
normal.

## Reference

S. Moskow, V. Druskin, T. Habashy, P. Lee, and S. Davydycheva,
*A finite difference scheme for elliptic equations with rough coefficients
using a Cartesian grid nonconforming to interfaces*,
SIAM J. Numer. Anal., 36(2):442–464, 1999.

"""
tilted_convergence.py — Three-way comparison on optimal DDH03 grids.

Compares:
  • Nodal homogenization  (Moskow 1999, 3-D energy-matched extension)
  • Standard homogenization  (ΣL = σ̄(I−n̂n̂ᵀ) + σ̃ n̂n̂ᵀ)
  • Pointwise assignment

Interface: n̂=[1,0,1]/√2,  x+z=√2 m,  σ₁=0.1 S/m, σ₂=1.0 S/m, f=100 Hz.
Reference: k=5 nodal solution (pre-computed, loaded from tilted_reference.npz).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import planar_interface_isotropic, EMMedia, MU0
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.postprocess import lebedev_E_at_point

# ── Physical parameters ───────────────────────────────────────────────────────
SIGMA1, SIGMA2 = 0.1, 1.0
OMEGA   = 2 * np.pi * 100.0
N_HAT   = np.array([1., 0., 1.]) / np.sqrt(2.)
D_PLANE = 1.0
Z_IFACE = D_PLANE * np.sqrt(2.)
X_SRC, Y_SRC, Z_SRC = 0., 0., -2.

GAMMA       = 1.0 / np.sqrt(2.)
TARGET_XMAX = 20.0
K_VALS      = [1, 2, 3, 4]

DZ      = 0.125
n_inner = int(round(6.0 / DZ))
z_grid  = hybrid_axial_grid(-2.5, 3.5, n_inner, 6, GAMMA)
if (len(z_grid) - 1) % 2 != 0:
    z_grid = np.append(z_grid, 2*z_grid[-1] - z_grid[-2])

def h_min_for_k(k):
    alpha = np.exp(GAMMA * np.pi / np.sqrt(k))
    return TARGET_XMAX / sum(alpha**i for i in range(k))

# ── Load reference ────────────────────────────────────────────────────────────
ref_file = os.path.join(os.path.dirname(__file__), "tilted_reference.npz")
ref    = np.load(ref_file)
z_ref  = ref["z_eval"];  Ez_ref = ref["Ez"];  Ex_ref = ref["Ex"]
K_REF  = int(ref["k_ref"])
print(f"Reference: k={K_REF}, N_R={int(ref['N_R'])}, h_min={float(ref['h_min']):.3f}m")

mask      = ((z_ref >= -0.5) & (z_ref <= 3.0) &
             (np.abs(z_ref - Z_IFACE) > 0.2))
z_eval    = z_ref[mask]
Ez_ref_e  = Ez_ref[mask]
Ex_ref_e  = Ex_ref[mask]

# ── Solver ────────────────────────────────────────────────────────────────────
def run(k, mode):
    hk   = h_min_for_k(k)
    grid = symmetric_optimal_grid(hk, TARGET_XMAX * 1.5, z_grid, GAMMA, k=k)

    if mode == "nodal":
        media = planar_interface_isotropic(
            grid, N_HAT, D_PLANE, SIGMA1, SIGMA2, method="nodal")
    elif mode == "standard":
        media = planar_interface_isotropic(
            grid, N_HAT, D_PLANE, SIGMA1, SIGMA2, method="standard")
    else:  # pointwise
        sigma_R = np.array(
            [SIGMA1 if float(N_HAT @ np.array([grid.x[i], grid.y[j], grid.z[kk]])) < D_PLANE
             else SIGMA2
             for i, j, kk in grid.R_nodes], dtype=complex)
        media = EMMedia(grid, sigma_R, np.full(grid.N_P, complex(MU0)))

    t0 = time.time()
    result = LebedevMaxwellSolver(grid, media, omega=OMEGA).solve(
                 X_SRC, Y_SRC, Z_SRC, dipole_comp=2, moment=1.0)
    dt = time.time() - t0
    E_c = result["E_c"]
    Ez = np.array([lebedev_E_at_point(grid, E_c, 2, 0., 0., z) for z in z_eval])
    Ex = np.array([lebedev_E_at_point(grid, E_c, 0, 0., 0., z) for z in z_eval])
    print(f"  k={k}  Mx={grid.Mx:2d}  N_R={grid.N_R:5d}  {dt:5.1f}s  [{mode}]")
    return Ez, Ex, grid.N_R, grid.Mx

print("\nSolving (nodal) ...")
nodal  = {k: run(k, "nodal")     for k in K_VALS}
print("Solving (standard) ...")
stand  = {k: run(k, "standard")  for k in K_VALS}
print("Solving (pointwise) ...")
pwise  = {k: run(k, "pointwise") for k in K_VALS}

# ── Error metric — global normalisation avoids amplifying small-signal noise ─
def rms_err(f, ref_arr):
    """RMS relative error, normalised by RMS of reference (not per-point)."""
    return float(np.sqrt(np.mean(np.abs(f - ref_arr)**2))
                 / np.sqrt(np.mean(np.abs(ref_arr)**2)))

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'k':>2}  {'mode':>9}  {'RMS Ez':>8}  {'RMS Ex':>8}")
print("-" * 38)
for k in K_VALS:
    for lbl, res in [("nodal", nodal), ("standard", stand), ("pointwise", pwise)]:
        ez_e = rms_err(res[k][0], Ez_ref_e)
        ex_e = rms_err(res[k][1], Ex_ref_e)
        print(f"{k:>2}  {lbl:>9}  {ez_e*100:7.2f}%  {ex_e*100:7.2f}%")
    print()

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS = {k: c for k, c in zip(K_VALS, ["tab:blue","tab:orange","tab:green","tab:red"])}

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle(
    r"Optimal DDH03 grids — tilted interface  $\hat{n}=[1,0,1]/\sqrt{2}$,  "
    r"$x+z=\sqrt{2}$ m"  "\n"
    r"$\sigma_1=0.1$ S/m | $\sigma_2=1.0$ S/m | $f=100$ Hz | VED at $(0,0,-2)$ m"
    "\n"
    rf"Fixed domain $x_\mathrm{{max}}\approx{TARGET_XMAX}$ m  ·  "
    rf"reference = k={K_REF} nodal ($N_R={int(ref['N_R'])}$)  ·  "
    r"solid = nodal · dashed = standard · dotted = pointwise",
    fontsize=10,
)

def _vlines(ax):
    ax.axvline(Z_IFACE, color="k",    ls=":",  lw=1.2, label=rf"iface $z\approx{Z_IFACE:.2f}$")
    ax.axvline(Z_SRC,   color="gray", ls="--", lw=0.8, label="source")

# [0,0] Re(Ez)
ax = axes[0, 0]
ax.plot(z_ref, np.real(Ez_ref), "k-", lw=2.5, label=f"ref k={K_REF}", zorder=5)
for k in K_VALS:
    ax.plot(z_eval, np.real(nodal[k][0]), color=COLORS[k], ls="-",  lw=2.0,
            label=f"nodal k={k}", alpha=0.9)
    ax.plot(z_eval, np.real(stand[k][0]), color=COLORS[k], ls="--", lw=1.4,
            alpha=0.75)
    ax.plot(z_eval, np.real(pwise[k][0]), color=COLORS[k], ls=":",  lw=1.2,
            alpha=0.65)
_vlines(ax)
ax.set_xlabel("z [m]"); ax.set_ylabel(r"Re$(E_z)$ [V/m]")
ax.set_title(r"Re$(E_z)$ on z-axis", fontsize=9)
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)

# [0,1] Re(Ex)
ax = axes[0, 1]
ax.plot(z_ref, np.real(Ex_ref), "k-", lw=2.5, label=f"ref k={K_REF}", zorder=5)
for k in K_VALS:
    ax.plot(z_eval, np.real(nodal[k][1]), color=COLORS[k], ls="-",  lw=2.0, alpha=0.9)
    ax.plot(z_eval, np.real(stand[k][1]), color=COLORS[k], ls="--", lw=1.4, alpha=0.75)
    ax.plot(z_eval, np.real(pwise[k][1]), color=COLORS[k], ls=":",  lw=1.2, alpha=0.65)
_vlines(ax)
ax.set_xlabel("z [m]"); ax.set_ylabel(r"Re$(E_x)$ [V/m]")
ax.set_title(r"Re$(E_x)$ on z-axis  (induced by tilted interface)", fontsize=9)
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)

# [0,2] Ez convergence — all three methods
ax = axes[0, 2]
k_arr = np.array(K_VALS)
styles = [("nodal", "-", "o", 2.0), ("standard", "--", "s", 1.6), ("pointwise", ":", "^", 1.4)]
for lbl, ls, mk, lw in styles:
    res = {"nodal": nodal, "standard": stand, "pointwise": pwise}[lbl]
    err = [rms_err(res[k][0], Ez_ref_e) for k in K_VALS]
    ax.semilogy(k_arr, err, ls+mk, lw=lw, ms=7, label=lbl)
ax.set_xlabel("k  (Mx=My=4k,  fixed $x_\\mathrm{max}$)")
ax.set_ylabel("RMS relative error  (Ez)")
ax.set_title(r"$E_z$ convergence vs k", fontsize=9)
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2); ax.set_xticks(k_arr)

# [1,0] Im(Ez)
ax = axes[1, 0]
ax.plot(z_ref, np.imag(Ez_ref), "k-", lw=2.5, label=f"ref k={K_REF}", zorder=5)
for k in K_VALS:
    ax.plot(z_eval, np.imag(nodal[k][0]), color=COLORS[k], ls="-",  lw=2.0, alpha=0.9)
    ax.plot(z_eval, np.imag(stand[k][0]), color=COLORS[k], ls="--", lw=1.4, alpha=0.75)
    ax.plot(z_eval, np.imag(pwise[k][0]), color=COLORS[k], ls=":",  lw=1.2, alpha=0.65)
_vlines(ax)
ax.set_xlabel("z [m]"); ax.set_ylabel(r"Im$(E_z)$ [V/m]")
ax.set_title(r"Im$(E_z)$ on z-axis", fontsize=9)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# [1,1] Ex convergence
ax = axes[1, 1]
for lbl, ls, mk, lw in styles:
    res = {"nodal": nodal, "standard": stand, "pointwise": pwise}[lbl]
    err = [rms_err(res[k][1], Ex_ref_e) for k in K_VALS]
    ax.semilogy(k_arr, err, ls+mk, lw=lw, ms=7, label=lbl)
ax.set_xlabel("k  (Mx=My=4k,  fixed $x_\\mathrm{max}$)")
ax.set_ylabel("RMS relative error  (Ex)")
ax.set_title(r"$E_x$ convergence vs k  (transverse component)", fontsize=9)
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2); ax.set_xticks(k_arr)

# [1,2] nodal − standard difference  (shows where they diverge)
ax = axes[1, 2]
for k in K_VALS:
    diff_ns = np.real(nodal[k][0] - stand[k][0])
    diff_sp = np.real(stand[k][0] - pwise[k][0])
    ax.plot(z_eval, diff_ns, color=COLORS[k], ls="-",  lw=2.0,
            label=f"k={k}: nodal−std", alpha=0.9)
    ax.plot(z_eval, diff_sp, color=COLORS[k], ls="--", lw=1.4,
            alpha=0.7)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.axvline(Z_IFACE, color="k", ls=":", lw=1.2, label="interface")
ax.set_xlabel("z [m]"); ax.set_ylabel(r"$\Delta$Re$(E_z)$ [V/m]")
ax.set_title(r"Re$(E_z)$: solid=nodal$-$std · dashed=std$-$pwise", fontsize=9)
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)

plt.tight_layout()
outfile = os.path.join(os.path.dirname(__file__), "tilted_convergence.png")
plt.savefig(outfile, dpi=140, bbox_inches="tight")
print(f"\nSaved → {outfile}")

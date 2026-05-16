"""
thin_layer_convergence.py — Three-method comparison for a thin resistive layer.

Geometry
--------
  Background : σ = 1.0 S/m
  Layer      : σ = 0.01 S/m  (100:1 contrast)
  Normal     : n̂ = [1,0,1]/√2  (45° tilt in xz-plane)
  Layer centre : n̂·x = 1.0
  Thickness    : 0.05 m  (in the n̂ direction)

The layer is sub-cell at all k=1..4 tested here (h_min ranges from ~20 m to ~0.49 m).
Pointwise assignment either misses the layer entirely or assigns it incorrectly.
Nodal and standard homogenization both use correct volume and line averages,
so they capture the series resistance even when the layer is thinner than a cell.

Reference : k=5 nodal solution.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import thin_layer_planar_isotropic
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.postprocess import lebedev_E_at_point

# ── Physical parameters ───────────────────────────────────────────────────────
SIGMA_BG    = 1.0
SIGMA_LAYER = 0.01          # 100:1 contrast
OMEGA       = 2 * np.pi * 100.0
N_HAT       = np.array([1., 0., 1.]) / np.sqrt(2.)
D_CENTER    = 1.0           # layer centre in n̂·x coordinate
THICKNESS   = 0.05          # layer thickness along n̂  [m]
X_SRC, Y_SRC, Z_SRC = 0., 0., -2.

GAMMA       = 1.0 / np.sqrt(2.)
TARGET_XMAX = 20.0
K_VALS      = [1, 2, 3, 4]
K_REF       = 5

DZ      = 0.125
n_inner = int(round(6.0 / DZ))
z_grid  = hybrid_axial_grid(-2.5, 3.5, n_inner, 6, GAMMA)
if (len(z_grid) - 1) % 2 != 0:
    z_grid = np.append(z_grid, 2*z_grid[-1] - z_grid[-2])

# Layer extent on z-axis (x=0): d1 < z/√2 < d2
D1 = D_CENTER - 0.5 * THICKNESS
D2 = D_CENTER + 0.5 * THICKNESS
Z_LAYER_LO = D1 * np.sqrt(2.)
Z_LAYER_HI = D2 * np.sqrt(2.)

def h_min_for_k(k):
    alpha = np.exp(GAMMA * np.pi / np.sqrt(k))
    return TARGET_XMAX / sum(alpha**i for i in range(k))

# ── Reference (or recompute) ──────────────────────────────────────────────────
REF_FILE = os.path.join(os.path.dirname(__file__),
    f"thin_layer_reference_t{THICKNESS:.3f}.npz".replace(".", "p"))

def compute_reference():
    hk   = h_min_for_k(K_REF)
    grid = symmetric_optimal_grid(hk, TARGET_XMAX * 1.5, z_grid, GAMMA, k=K_REF)
    media = thin_layer_planar_isotropic(
        grid, N_HAT, D_CENTER, THICKNESS, SIGMA_BG, SIGMA_LAYER, method="nodal")
    print(f"Computing reference: k={K_REF}, N_R={grid.N_R}, h_min={hk:.3f}m ...")
    t0 = time.time()
    result = LebedevMaxwellSolver(grid, media, omega=OMEGA).solve(
                 X_SRC, Y_SRC, Z_SRC, dipole_comp=2, moment=1.0)
    print(f"  solved in {time.time()-t0:.1f}s")
    E_c  = result["E_c"]
    z_ev = np.linspace(-0.5, 3.0, 200)
    Ez   = np.array([lebedev_E_at_point(grid, E_c, 2, 0., 0., z) for z in z_ev])
    Ex   = np.array([lebedev_E_at_point(grid, E_c, 0, 0., 0., z) for z in z_ev])
    np.savez(REF_FILE, z_eval=z_ev, Ez=Ez, Ex=Ex,
             k_ref=K_REF, N_R=grid.N_R, Mx=grid.Mx, h_min=hk)
    return z_ev, Ez, Ex

if os.path.exists(REF_FILE):
    ref    = np.load(REF_FILE)
    z_ref  = ref["z_eval"]; Ez_ref = ref["Ez"]; Ex_ref = ref["Ex"]
    print(f"Reference loaded: k={int(ref['k_ref'])}, N_R={int(ref['N_R'])}, "
          f"h_min={float(ref['h_min']):.3f}m")
else:
    z_ref, Ez_ref, Ex_ref = compute_reference()

# Evaluation mask: avoid source singularity; stay away from layer edges.
# Use 3× thickness as exclusion zone so the mask scales with layer width.
excl = max(3.0 * THICKNESS * np.sqrt(2.), 0.05)
mask     = ((z_ref >= -0.5) & (z_ref <= 3.0) &
            (np.abs(z_ref - Z_LAYER_LO) > excl) &
            (np.abs(z_ref - Z_LAYER_HI) > excl))
z_eval   = z_ref[mask]
Ez_ref_e = Ez_ref[mask]
Ex_ref_e = Ex_ref[mask]

# ── Solver ────────────────────────────────────────────────────────────────────
def run(k, method):
    hk   = h_min_for_k(k)
    grid = symmetric_optimal_grid(hk, TARGET_XMAX * 1.5, z_grid, GAMMA, k=k)
    media = thin_layer_planar_isotropic(
        grid, N_HAT, D_CENTER, THICKNESS, SIGMA_BG, SIGMA_LAYER, method=method)
    t0 = time.time()
    result = LebedevMaxwellSolver(grid, media, omega=OMEGA).solve(
                 X_SRC, Y_SRC, Z_SRC, dipole_comp=2, moment=1.0)
    dt = time.time() - t0
    E_c = result["E_c"]
    Ez = np.array([lebedev_E_at_point(grid, E_c, 2, 0., 0., z) for z in z_eval])
    Ex = np.array([lebedev_E_at_point(grid, E_c, 0, 0., 0., z) for z in z_eval])
    print(f"  k={k}  Mx={grid.Mx:2d}  N_R={grid.N_R:5d}  {dt:5.1f}s  [{method}]")
    return Ez, Ex, grid.N_R

def rms_err(f, ref):
    return float(np.sqrt(np.mean(np.abs(f - ref)**2))
                 / np.sqrt(np.mean(np.abs(ref)**2)))

METHODS = ["nodal", "standard", "pointwise"]
results = {}
for m in METHODS:
    print(f"\nSolving ({m}) ...")
    results[m] = {k: run(k, m) for k in K_VALS}

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'k':>2}  {'method':>10}  {'RMS Ez':>8}  {'RMS Ex':>8}")
print("-" * 40)
for k in K_VALS:
    for m in METHODS:
        ez = rms_err(results[m][k][0], Ez_ref_e)
        ex = rms_err(results[m][k][1], Ex_ref_e)
        print(f"{k:>2}  {m:>10}  {ez*100:7.2f}%  {ex*100:7.2f}%")
    print()

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS = {k: c for k, c in zip(K_VALS, ["tab:blue","tab:orange","tab:green","tab:red"])}
LS     = {"nodal": "-", "standard": "--", "pointwise": ":"}
LW     = {"nodal": 2.0, "standard": 1.5, "pointwise": 1.3}

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle(
    r"Thin resistive layer at 45°:  $\hat{n}=[1,0,1]/\sqrt{2}$,  "
    rf"thickness={THICKNESS} m in $\hat{{n}}$ direction"  "\n"
    rf"$\sigma_{{\rm bg}}={SIGMA_BG}$ S/m | $\sigma_{{\rm layer}}={SIGMA_LAYER}$ S/m "
    r"(100:1) | $f=100$ Hz | VED at $(0,0,-2)$ m"  "\n"
    rf"Reference = k={K_REF} nodal · "
    r"solid=nodal · dashed=standard · dotted=pointwise",
    fontsize=10,
)

def _shade(ax):
    ax.axvspan(Z_LAYER_LO, Z_LAYER_HI, alpha=0.12, color="gold",
               label=rf"layer $z\in[{Z_LAYER_LO:.2f},{Z_LAYER_HI:.2f}]$")
    ax.axvline(X_SRC, color="gray", ls="--", lw=0.8, label="source")

# [0,0] Re(Ez)
ax = axes[0, 0]
ax.plot(z_ref, np.real(Ez_ref), "k-", lw=2.5, label=f"ref k={K_REF}", zorder=5)
for k in K_VALS:
    for m in METHODS:
        ax.plot(z_eval, np.real(results[m][k][0]),
                color=COLORS[k], ls=LS[m], lw=LW[m],
                label=f"{m} k={k}" if m == "nodal" else None, alpha=0.85)
_shade(ax)
ax.set_xlabel("z [m]"); ax.set_ylabel(r"Re$(E_z)$ [V/m]")
ax.set_title(r"Re$(E_z)$ on z-axis", fontsize=9)
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)

# [0,1] Re(Ex)
ax = axes[0, 1]
ax.plot(z_ref, np.real(Ex_ref), "k-", lw=2.5, label=f"ref k={K_REF}", zorder=5)
for k in K_VALS:
    for m in METHODS:
        ax.plot(z_eval, np.real(results[m][k][1]),
                color=COLORS[k], ls=LS[m], lw=LW[m], alpha=0.85)
_shade(ax)
ax.set_xlabel("z [m]"); ax.set_ylabel(r"Re$(E_x)$ [V/m]")
ax.set_title(r"Re$(E_x)$ on z-axis", fontsize=9)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# [0,2] Ez convergence
ax = axes[0, 2]
k_arr = np.array(K_VALS)
styles = [("nodal","-","o",2.0), ("standard","--","s",1.6), ("pointwise",":","^",1.4)]
for m, ls, mk, lw in styles:
    err = [rms_err(results[m][k][0], Ez_ref_e) for k in K_VALS]
    ax.semilogy(k_arr, err, ls+mk, lw=lw, ms=7, label=m)
ax.set_xlabel("k"); ax.set_ylabel("RMS relative error  (Ez)")
ax.set_title(r"$E_z$ convergence", fontsize=9)
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2); ax.set_xticks(k_arr)

# [1,0] Im(Ez)
ax = axes[1, 0]
ax.plot(z_ref, np.imag(Ez_ref), "k-", lw=2.5, label=f"ref k={K_REF}", zorder=5)
for k in K_VALS:
    for m in METHODS:
        ax.plot(z_eval, np.imag(results[m][k][0]),
                color=COLORS[k], ls=LS[m], lw=LW[m], alpha=0.85)
_shade(ax)
ax.set_xlabel("z [m]"); ax.set_ylabel(r"Im$(E_z)$ [V/m]")
ax.set_title(r"Im$(E_z)$ on z-axis", fontsize=9)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# [1,1] Ex convergence
ax = axes[1, 1]
for m, ls, mk, lw in styles:
    err = [rms_err(results[m][k][1], Ex_ref_e) for k in K_VALS]
    ax.semilogy(k_arr, err, ls+mk, lw=lw, ms=7, label=m)
ax.set_xlabel("k"); ax.set_ylabel("RMS relative error  (Ex)")
ax.set_title(r"$E_x$ convergence", fontsize=9)
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2); ax.set_xticks(k_arr)

# [1,2] nodal − pointwise  (Ez, shows where they differ most)
ax = axes[1, 2]
for k in K_VALS:
    dn = np.real(results["nodal"][k][0] - results["pointwise"][k][0])
    ds = np.real(results["standard"][k][0] - results["pointwise"][k][0])
    ax.plot(z_eval, dn, color=COLORS[k], ls="-",  lw=2.0,
            label=f"k={k}: nodal−pwise", alpha=0.9)
    ax.plot(z_eval, ds, color=COLORS[k], ls="--", lw=1.4, alpha=0.7)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.axvspan(Z_LAYER_LO, Z_LAYER_HI, alpha=0.12, color="gold", label="layer")
ax.set_xlabel("z [m]"); ax.set_ylabel(r"$\Delta$Re$(E_z)$  [V/m]")
ax.set_title(r"Re$(E_z)$: solid=nodal$-$pwise · dashed=std$-$pwise", fontsize=9)
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)

plt.tight_layout()
outfile = os.path.join(os.path.dirname(__file__),
    f"thin_layer_convergence_t{THICKNESS:.3f}.png".replace(".", "p"))
plt.savefig(outfile, dpi=140, bbox_inches="tight")
print(f"\nSaved → {outfile}")

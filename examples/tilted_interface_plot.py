"""
tilted_interface_plot.py — VED solution with a 45° tilted planar interface.

Setup
-----
Source:    z-directed electric dipole at (0, 0, -2) m
Interface: n̂ = [1, 0, 1]/√2,  n̂·x = 1.0   →  x + z = √2 ≈ 1.414 m
           Layer 1 (below plane):  σ₁ = 0.1 S/m
           Layer 2 (above plane):  σ₂ = 1.0 S/m
Frequency: 100 Hz
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from lebedev_em.grid import LebedevGrid3D
from lebedev_em.media import planar_interface_isotropic, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.postprocess import lebedev_E_at_point

# ── Physical parameters ───────────────────────────────────────────────────────
SIGMA1  = 0.1        # S/m — background (layer 1: n̂·x < d_plane)
SIGMA2  = 1.0        # S/m — conductor  (layer 2: n̂·x ≥ d_plane)
FREQ    = 100.0      # Hz
OMEGA   = 2 * np.pi * FREQ

N_HAT   = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
D_PLANE = 1.0        # x + z = √2·D_PLANE ≈ 1.414 m

X_SRC, Y_SRC, Z_SRC = 0.0, 0.0, -2.0

print("Building medium and solving ...")

# ── Grid: coarse uniform, 13³ nodes  ────────────────────────────────────────
h      = 1.0
lim    = 6.0
coords = np.arange(-lim, lim + h/2, h)    # 13 points: -6, -5, ..., 6
grid   = LebedevGrid3D(coords, coords, coords)
print(f"  Grid: {len(coords)} pts/axis, N_R={grid.N_R}")

media  = planar_interface_isotropic(grid, N_HAT, D_PLANE, SIGMA1, SIGMA2)
solver = LebedevMaxwellSolver(grid, media, omega=OMEGA)
result = solver.solve(X_SRC, Y_SRC, Z_SRC, dipole_comp=2, moment=1.0)
print("  Solve done.")

# ── Evaluate on xz-plane (y=0) ───────────────────────────────────────────────
Neval  = 50
x_eval = np.linspace(-lim + 0.2, lim - 0.2, Neval)
z_eval = np.linspace(-lim + 0.2, lim - 0.2, Neval)
XX, ZZ = np.meshgrid(x_eval, z_eval, indexing="ij")

Ex = np.zeros((Neval, Neval), dtype=complex)
Ez = np.zeros((Neval, Neval), dtype=complex)
E_c = result["E_c"]
for ix, x0 in enumerate(x_eval):
    for iz, z0 in enumerate(z_eval):
        Ex[ix, iz] = lebedev_E_at_point(grid, E_c, 0, x0, 0.0, z0)
        Ez[ix, iz] = lebedev_E_at_point(grid, E_c, 2, x0, 0.0, z0)

Emag = np.sqrt(np.abs(Ex)**2 + np.abs(Ez)**2)

# ── Conductivity and interface line ──────────────────────────────────────────
sigma_map = np.where(N_HAT[0]*XX + N_HAT[2]*ZZ < D_PLANE, SIGMA1, SIGMA2)
x_if  = np.linspace(-lim, lim, 300)
z_if  = D_PLANE * np.sqrt(2.0) - x_if     # x + z = √2

def _div_norm(data, pct=98):
    vmax = max(np.percentile(np.abs(data), pct), 1e-30)
    return TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

src_kw  = dict(marker="*", ms=13, color="white", mec="k", mew=0.8, zorder=10,
               linestyle="none")
if_kw   = dict(color="white", lw=1.8, ls="--")
if_kw2  = dict(color="cyan",  lw=1.8, ls="--")

# ── 2×3 figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    r"VED in tilted two-layer medium  —  n̂=[1,0,1]/√2,  "
    r"$x\!+\!z = \sqrt{2}$ m"
    "\n"
    rf"σ₁={SIGMA1} S/m (source side)  |  σ₂={SIGMA2} S/m  |  "
    rf"f={FREQ:.0f} Hz  |  VED source ★ at (0,0,{Z_SRC}) m"
    "\n"
    "Nodal homogenization (Moskow et al. 1999)  —  xz-plane, y=0",
    fontsize=10,
)

def _decorate(ax, use_cyan=False):
    kw = if_kw2 if use_cyan else if_kw
    ax.plot(x_if, z_if, **kw, label="interface")
    ax.plot([X_SRC], [Z_SRC], **src_kw)
    ax.set_xlim(x_eval[0], x_eval[-1])
    ax.set_ylim(z_eval[0], z_eval[-1])
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")

# [0,0] conductivity
ax = axes[0, 0]
pc = ax.pcolormesh(XX, ZZ, sigma_map, cmap="RdBu_r", shading="auto",
                   vmin=SIGMA1-0.05, vmax=SIGMA2+0.05)
_decorate(ax)
fig.colorbar(pc, ax=ax, label="σ [S/m]")
ax.set_title("Conductivity σ", fontsize=10)

# [0,1] Re(E_x)
ax = axes[0, 1]
data = np.real(Ex)
pc = ax.pcolormesh(XX, ZZ, data, cmap="RdBu_r", norm=_div_norm(data), shading="auto")
_decorate(ax)
fig.colorbar(pc, ax=ax, label="Re(Eₓ) [V/m]")
ax.set_title(r"Re$(E_x)$  — cross-interface component", fontsize=10)

# [0,2] Re(E_z)
ax = axes[0, 2]
data = np.real(Ez)
pc = ax.pcolormesh(XX, ZZ, data, cmap="RdBu_r", norm=_div_norm(data), shading="auto")
_decorate(ax)
fig.colorbar(pc, ax=ax, label="Re(Ez) [V/m]")
ax.set_title(r"Re$(E_z)$  — dipole-aligned component", fontsize=10)

# [1,0] Im(E_x)
ax = axes[1, 0]
data = np.imag(Ex)
pc = ax.pcolormesh(XX, ZZ, data, cmap="RdBu_r", norm=_div_norm(data), shading="auto")
_decorate(ax)
fig.colorbar(pc, ax=ax, label="Im(Eₓ) [V/m]")
ax.set_title(r"Im$(E_x)$", fontsize=10)

# [1,1] Im(E_z)
ax = axes[1, 1]
data = np.imag(Ez)
pc = ax.pcolormesh(XX, ZZ, data, cmap="RdBu_r", norm=_div_norm(data), shading="auto")
_decorate(ax)
fig.colorbar(pc, ax=ax, label="Im(Ez) [V/m]")
ax.set_title(r"Im$(E_z)$", fontsize=10)

# [1,2] |E| log scale
ax = axes[1, 2]
Emag_safe = np.where(Emag > 0, Emag, 1e-30)
pc = ax.pcolormesh(XX, ZZ, np.log10(Emag_safe), cmap="inferno", shading="auto")
_decorate(ax, use_cyan=True)
fig.colorbar(pc, ax=ax, label=r"$\log_{10}|E|$  [V/m]")
ax.set_title(r"$|E|$ total field magnitude (log scale)", fontsize=10)

plt.tight_layout()
outfile = os.path.join(os.path.dirname(__file__), "tilted_interface_plot.png")
plt.savefig(outfile, dpi=140, bbox_inches="tight")
print(f"  Plot saved → {outfile}")

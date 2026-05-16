"""
Plot Ez vs z for the 45-degree tilted interface:
  - pi_nodal   : analytic planar_interface_isotropic reference
  - sf_nodal   : from_sigma_func with analytical closure (h_svd=0.025)
  - fg_nodal   : from_fine_grid wrapper (NSUB=4 fine grid)
  - homogeneous: σ=σ1 everywhere (to show the interface effect)
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid, C000, C101, C110, C011
from lebedev_em.media import (
    homogeneous_isotropic,
    planar_interface_isotropic,
    from_sigma_func,
    from_fine_grid,
    MU0,
)
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.sources import _native_type_for_cluster_comp
from lebedev_em.postprocess import interpolate_cluster_E

# ── Parameters ────────────────────────────────────────────────────────────────
SIGMA1 = 0.1; SIGMA2 = 1.0; Z_CONT = 4.0; OMEGA = 2.0*np.pi*2500.0
DZ=0.0625; Z_INNER_MIN=-0.25; Z_INNER_MAX=7.75
N_INNER=int(round((Z_INNER_MAX-Z_INNER_MIN)/DZ))
K_OUTER=8; GAMMA=1.0/np.sqrt(2.0); H_MIN=0.5; L_TRANS=300.0; K_GRID=2

z_fd_grid = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)
grid      = symmetric_optimal_grid(H_MIN, L_TRANS, z_fd_grid, GAMMA, k=K_GRID)
print(f"Grid: N_R={grid.N_R}")

N_HAT = np.array([1.,0.,1.])/np.sqrt(2.)
D_PLANE = float(Z_CONT/np.sqrt(2.))

def sigma_func(X, Y, Z):
    return np.where(N_HAT[0]*X + N_HAT[2]*Z < D_PLANE, SIGMA1, SIGMA2)

# Fine grid (NSUB=4)
def subdivide(g, n):
    pts=[]
    for i in range(len(g)-1):
        pts.extend(np.linspace(float(g[i]),float(g[i+1]),n+1)[:-1])
    pts.append(float(g[-1])); return np.array(pts)

ffx=subdivide(grid.x,4); ffy=subdivide(grid.y,4); ffz=subdivide(z_fd_grid,4)
FFX=ffx[:,None,None]; FFZ=ffz[None,None,:]
sig_tilt = np.broadcast_to(
    np.where(FFX+FFZ>=Z_CONT, SIGMA2, SIGMA1).astype(complex),
    (len(ffx),len(ffy),len(ffz))
).copy()

# ── Receivers ─────────────────────────────────────────────────────────────────
Mx2, My2 = grid.Mx//2, grid.My//2
nat_c000  = _native_type_for_cluster_comp(C000, 2)
z_eval, seq_c000 = [], []
for seq, (i,j,k) in enumerate(grid.R_nodes):
    if (i==Mx2 and j==My2 and (i%2,j%2,k%2)==nat_c000):
        zv = float(grid.z[k])
        if 0.1 <= zv <= 7.9:
            z_eval.append(zv); seq_c000.append(seq)
z_eval = np.array(z_eval)
order = np.argsort(z_eval); z_eval = z_eval[order]

def extract_Ez(result):
    out = np.zeros(len(z_eval), dtype=complex)
    for idx, zv in enumerate(z_eval):
        vals = [interpolate_cluster_E(grid, result["E_c"][c], c, 2, 0.0, 0.0, zv)
                for c in (C000, C101, C110, C011)]
        out[idx] = np.mean(vals)
    return out

def run(med, label):
    t0=time.time()
    res = LebedevMaxwellSolver(grid, med, OMEGA).solve(0.,0.,0., dipole_comp=2)
    Ez = extract_Ez(res)
    print(f"  [{label}] {time.time()-t0:.1f}s")
    return Ez

# ── Build & solve ─────────────────────────────────────────────────────────────
# from_fine_grid now uses a per-axis h_svd tuple by default (10th-percentile
# fine-grid spacing in each axis independently), giving 2-3× faster builds
# than the old min-spacing scalar with equal or better field accuracy.
import time as _time
print("Building media...")
t0=_time.time(); med_homo = homogeneous_isotropic(grid, SIGMA1);         print(f"  homo:    {_time.time()-t0:.1f}s")
t0=_time.time(); med_pi   = planar_interface_isotropic(grid, N_HAT, D_PLANE, SIGMA1, SIGMA2, method="nodal"); print(f"  pi_nodal:{_time.time()-t0:.1f}s")
t0=_time.time(); med_sf   = from_sigma_func(grid, sigma_func, h_svd=0.025, n_line=50, n_vol=8, method="nodal"); print(f"  sf_nodal:{_time.time()-t0:.1f}s")
t0=_time.time(); med_fg   = from_fine_grid(grid, ffx, ffy, ffz, sig_tilt, method="nodal"); t_fg=_time.time()-t0; print(f"  fg_nodal:{t_fg:.1f}s  (per-axis h_svd default)")

print("Solving...")
Ez_homo = run(med_homo, "homogeneous")
Ez_pi   = run(med_pi,   "pi_nodal")
Ez_sf   = run(med_sf,   "sf_nodal")
Ez_fg   = run(med_fg,   "fg_nodal")

# ── Plot 1: |Ez| vs z, all methods ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("45° tilted interface  (σ₁=0.1 S/m, σ₂=1.0 S/m,  n̂=[1,0,1]/√2,  interface at x+z=4 m)",
             fontsize=11, y=1.01)

ax = axes[0]
z_iface = 4.0  # interface at x=0 (on-axis) is at z=4
ax.axvline(z_iface, color="gray", lw=0.8, ls="--", label="interface (z=4, on-axis)")
ax.semilogy(z_eval, np.abs(Ez_homo), "k:",  lw=1.4, label="homogeneous (σ₁ only)")
ax.semilogy(z_eval, np.abs(Ez_pi),   "b-",  lw=2.0, label="pi_nodal (analytic ref)")
ax.semilogy(z_eval, np.abs(Ez_sf),   "r--", lw=1.8, label="sf_nodal (callable, h_svd=0.025)")
ax.semilogy(z_eval, np.abs(Ez_fg),   "g-.", lw=1.8, label=f"fg_nodal (fine grid NSUB=4, build {t_fg:.1f}s)")
ax.set_xlabel("z  [m]", fontsize=11)
ax.set_ylabel("|Ez|  [V/m per A·m]", fontsize=11)
ax.set_title("|Ez| on-axis (log scale)", fontsize=11)
ax.legend(fontsize=8.5)
ax.set_xlim(z_eval[0], z_eval[-1])
ax.grid(True, which="both", alpha=0.3)

# ── Plot 2: relative error vs pi_nodal ───────────────────────────────────────
ax2 = axes[1]
ax2.axvline(z_iface, color="gray", lw=0.8, ls="--")
ax2.axhline(0, color="k", lw=0.5)

ref = np.abs(Ez_pi)
err_sf = (np.abs(Ez_sf) - ref) / ref * 100
err_fg = (np.abs(Ez_fg) - ref) / ref * 100

ax2.plot(z_eval, err_sf, "r--", lw=1.8, label=f"sf_nodal  (max {np.abs(err_sf).max():.1f}%,  mean {np.abs(err_sf).mean():.2f}%)")
ax2.plot(z_eval, err_fg, "g-.", lw=1.8, label=f"fg_nodal  (max {np.abs(err_fg).max():.1f}%,  mean {np.abs(err_fg).mean():.2f}%)")
ax2.set_xlabel("z  [m]", fontsize=11)
ax2.set_ylabel("(|Ez| − |Ez_ref|) / |Ez_ref|  [%]", fontsize=11)
ax2.set_title("Relative error vs pi_nodal reference", fontsize=11)
ax2.legend(fontsize=9)
ax2.set_xlim(z_eval[0], z_eval[-1])
ax2.set_ylim(-35, 35)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
ax2.grid(True, alpha=0.3)
ax2.fill_between(z_eval, -5, 5, color="green", alpha=0.07, label="±5% band")

plt.tight_layout()
out_path = "/sessions/wizardly-intelligent-feynman/mnt/outputs/lebedev_em/examples/tilted_interface_Ez.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# ── Plot 3: real & imag Ez separately ────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig2.suptitle("Ez components — 45° interface", fontsize=11)

for part, axes2_row, lbl in [(np.real, axes2[0], "Re(Ez)"), (np.imag, axes2[1], "Im(Ez)")]:
    ax3 = axes2_row
    ax3.axvline(z_iface, color="gray", lw=0.8, ls="--")
    ax3.plot(z_eval, part(Ez_homo), "k:",  lw=1.2, label="homogeneous")
    ax3.plot(z_eval, part(Ez_pi),   "b-",  lw=2.0, label="pi_nodal (ref)")
    ax3.plot(z_eval, part(Ez_sf),   "r--", lw=1.6, label="sf_nodal")
    ax3.plot(z_eval, part(Ez_fg),   "g-.", lw=1.6, label="fg_nodal")
    ax3.set_ylabel(lbl, fontsize=11)
    ax3.legend(fontsize=8.5, loc="upper right")
    ax3.grid(True, alpha=0.3)

axes2[-1].set_xlabel("z  [m]", fontsize=11)
plt.tight_layout()
out2 = "/sessions/wizardly-intelligent-feynman/mnt/outputs/lebedev_em/examples/tilted_interface_Ez_components.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")

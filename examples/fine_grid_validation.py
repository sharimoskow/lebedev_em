"""
fine_grid_validation.py
=======================
Validate from_fine_grid against:
  1. Analytic wholespace solution (homogeneous medium)
  2. Analytic Sommerfeld integral + planar_interface_isotropic (horizontal interface)

Geometry
--------
z-directed electric dipole at (0, 0, z_src=-2 m).
Interface at z = 0: σ₁ = 0.1 S/m (z < 0), σ₂ = 1.0 S/m (z ≥ 0).
Receivers along z-axis at several depths z_r > z_src.
ω = 2π × 2500 rad/s.
"""
import sys, time, warnings
sys.path.insert(0, "src")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em import symmetric_uniform_grid, from_fine_grid, LebedevMaxwellSolver, MU0, EPS0
from lebedev_em.media import homogeneous_isotropic, planar_interface_isotropic
from lebedev_em.analytics import (
    electric_dipole_Ez_homogeneous_onaxis,
    electric_dipole_Ez_two_layer_onaxis,
)
from lebedev_em.postprocess import lebedev_E_at_point

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
SIGMA1   = 0.1        # S/m  (source layer, z < 0)
SIGMA2   = 1.0        # S/m  (receiver layer, z > 0)
OMEGA    = 2 * np.pi * 2500
Z_SRC    = -2.0       # m
Z_IFACE  =  0.0       # interface depth
LX = LY = LZ = 8.0   # domain half-size
MX = MY = MZ = 20     # grid resolution (N_R ≈ 4600)

# Receivers: R-node z-coordinates above z=0 (found after grid build)
Z_RCVR_TARGET = np.array([0.3, 0.5, 1.0, 2.0, 3.0])

# Fine grid for from_fine_grid
NF = 120
fx = np.linspace(-LX, LX, NF)
fy = np.linspace(-LY, LY, NF)
fz = np.linspace(-LZ, LZ, NF)
FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")

# -------------------------------------------------------------------
# Build FD grid
# -------------------------------------------------------------------
grid = symmetric_uniform_grid(Mx=MX, My=MY, Mz=MZ, Lx=LX, Ly=LY, Lz=LZ)
print(f"Grid: N_R={grid.N_R}, N_P={grid.N_P}")

# Snap receivers to actual R-node z-values on the z-axis
# (find R-nodes with x≈0, y≈0 and z closest to each target)
z_all = np.array([grid.z[k] for i, j, k in grid.R_nodes])
x_all = np.array([grid.x[i] for i, j, k in grid.R_nodes])
y_all = np.array([grid.y[j] for i, j, k in grid.R_nodes])
mask_axis = (np.abs(x_all) < 0.01) & (np.abs(y_all) < 0.01)
z_axis = np.sort(np.unique(z_all[mask_axis]))
z_rcvr = np.array([z_axis[np.argmin(np.abs(z_axis - zt))] for zt in Z_RCVR_TARGET])
z_rcvr = np.unique(z_rcvr)
print(f"Receiver z-values: {z_rcvr}")

# -------------------------------------------------------------------
# Analytic references
# -------------------------------------------------------------------
Ez_homo_anal = np.array([
    electric_dipole_Ez_homogeneous_onaxis(zr, Z_SRC, SIGMA1, OMEGA)
    for zr in z_rcvr
])
Ez_iface_anal = np.array([
    electric_dipole_Ez_two_layer_onaxis(zr, Z_SRC, Z_IFACE, SIGMA1, SIGMA2, OMEGA)
    for zr in z_rcvr
])
print(f"Analytic Ez (homogeneous) at z={z_rcvr}: {np.abs(Ez_homo_anal)}")
print(f"Analytic Ez (interface)   at z={z_rcvr}: {np.abs(Ez_iface_anal)}")

# -------------------------------------------------------------------
# Helper: run solver and extract Ez at receivers
# -------------------------------------------------------------------
def run_and_extract(media, label=""):
    t0 = time.time()
    solver = LebedevMaxwellSolver(grid, media, OMEGA)
    result = solver.solve(0.0, 0.0, Z_SRC, dipole_comp=2)
    dt = time.time() - t0
    Ez = np.array([
        lebedev_E_at_point(grid, result["E_c"], comp=2, x0=0.0, y0=0.0, z0=zr)
        for zr in z_rcvr
    ])
    if label:
        print(f"  [{label}] solved in {dt:.1f}s")
    return Ez

# -------------------------------------------------------------------
# Case 1: Homogeneous medium
# -------------------------------------------------------------------
print("\n=== HOMOGENEOUS (σ = 0.1 S/m) ===")

# Analytic media reference
med_hom_anal  = homogeneous_isotropic(grid, SIGMA1)

# fine-grid arithmetic (should = homogeneous since σ is uniform)
sig_hom = np.full((NF, NF, NF), SIGMA1, dtype=complex)
t0=time.time()
med_hom_arith = from_fine_grid(grid, fx, fy, fz, sig_hom, method="arithmetic")
med_hom_diag  = from_fine_grid(grid, fx, fy, fz, sig_hom, method="diagonal")
med_hom_nodal = from_fine_grid(grid, fx, fy, fz, sig_hom, method="nodal")
print(f"  from_fine_grid built in {time.time()-t0:.2f}s")

Ez_hom_ref   = run_and_extract(med_hom_anal,  "homogeneous (reference)")
Ez_hom_arith = run_and_extract(med_hom_arith, "fine-grid arithmetic")
Ez_hom_diag  = run_and_extract(med_hom_diag,  "fine-grid diagonal")
Ez_hom_nodal = run_and_extract(med_hom_nodal, "fine-grid nodal")

print(f"\n  z_rcvr  |  analytic     |  hom_ref      |  arith        |  diag         |  nodal")
print(f"  {'':7s}  |  {'|Ez|':13s}  |  rel err      |  rel err      |  rel err      |  rel err")
for j, zr in enumerate(z_rcvr):
    a  = Ez_homo_anal[j]
    r0 = Ez_hom_ref[j];   e0 = abs(r0-a)/abs(a) if abs(a)>0 else 0
    r1 = Ez_hom_arith[j]; e1 = abs(r1-a)/abs(a) if abs(a)>0 else 0
    r2 = Ez_hom_diag[j];  e2 = abs(r2-a)/abs(a) if abs(a)>0 else 0
    r3 = Ez_hom_nodal[j]; e3 = abs(r3-a)/abs(a) if abs(a)>0 else 0
    print(f"  {zr:+.2f}   |  {abs(a):.4e}   |  {e0:.4f}         |  {e1:.4f}         |  {e2:.4f}         |  {e3:.4f}")

# -------------------------------------------------------------------
# Case 2: Horizontal interface
# -------------------------------------------------------------------
print("\n=== HORIZONTAL INTERFACE (σ₁=0.1 z<0, σ₂=1.0 z≥0) ===")

sig_iface = np.where(FZ >= Z_IFACE, SIGMA2, SIGMA1).astype(complex)

t0=time.time()
med_pi_nodal  = planar_interface_isotropic(grid, [0,0,1], Z_IFACE, SIGMA1, SIGMA2, method="nodal")
med_fg_arith  = from_fine_grid(grid, fx, fy, fz, sig_iface, method="arithmetic")
med_fg_diag   = from_fine_grid(grid, fx, fy, fz, sig_iface, method="diagonal")
med_fg_nodal  = from_fine_grid(grid, fx, fy, fz, sig_iface, method="nodal")
print(f"  Media built in {time.time()-t0:.2f}s")

Ez_pi_nodal  = run_and_extract(med_pi_nodal, "planar_interface nodal (ref)")
Ez_fg_arith  = run_and_extract(med_fg_arith, "fine-grid arithmetic")
Ez_fg_diag   = run_and_extract(med_fg_diag,  "fine-grid diagonal")
Ez_fg_nodal  = run_and_extract(med_fg_nodal, "fine-grid nodal")

print(f"\n  z_rcvr  |  analytic     |  pi_nodal     |  fg_arith     |  fg_diag      |  fg_nodal")
print(f"  {'':7s}  |  {'|Ez|':13s}  |  rel err      |  rel err      |  rel err      |  rel err")
for j, zr in enumerate(z_rcvr):
    a  = Ez_iface_anal[j]
    r0 = Ez_pi_nodal[j];  e0 = abs(r0-a)/abs(a)
    r1 = Ez_fg_arith[j];  e1 = abs(r1-a)/abs(a)
    r2 = Ez_fg_diag[j];   e2 = abs(r2-a)/abs(a)
    r3 = Ez_fg_nodal[j];  e3 = abs(r3-a)/abs(a)
    print(f"  {zr:+.2f}   |  {abs(a):.4e}   |  {e0:.4f}         |  {e1:.4f}         |  {e2:.4f}         |  {e3:.4f}")

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: Homogeneous
ax = axes[0]
z_plot = z_rcvr
ax.semilogy(z_plot, np.abs(Ez_homo_anal),  'k-',  lw=2.5, label='Analytic (wholespace)')
ax.semilogy(z_plot, np.abs(Ez_hom_ref),   'b--', lw=1.5, label='homogeneous_isotropic')
ax.semilogy(z_plot, np.abs(Ez_hom_arith), 'r:',  lw=1.5, label='fine-grid arithmetic')
ax.semilogy(z_plot, np.abs(Ez_hom_nodal), 'g-.', lw=1.5, label='fine-grid nodal')
ax.set_xlabel('z receiver (m)'); ax.set_ylabel('|Ez| (V/m)')
ax.set_title(f'Homogeneous  σ={SIGMA1} S/m')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 2: Interface
ax = axes[1]
ax.semilogy(z_plot, np.abs(Ez_iface_anal), 'k-',  lw=2.5, label='Analytic (Sommerfeld)')
ax.semilogy(z_plot, np.abs(Ez_pi_nodal),   'b--', lw=1.5, label='planar_interface nodal')
ax.semilogy(z_plot, np.abs(Ez_fg_arith),   'r:',  lw=1.5, label='fine-grid arithmetic')
ax.semilogy(z_plot, np.abs(Ez_fg_diag),    'm-',  lw=1.0, label='fine-grid diagonal')
ax.semilogy(z_plot, np.abs(Ez_fg_nodal),   'g-.', lw=1.5, label='fine-grid nodal')
ax.set_xlabel('z receiver (m)'); ax.set_ylabel('|Ez| (V/m)')
ax.set_title(f'Horizontal interface  σ₁={SIGMA1}, σ₂={SIGMA2} S/m')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle(f'from_fine_grid validation  (Mx={MX}, NF={NF}, ω=2π·2500 rad/s)', fontsize=11)
plt.tight_layout()
out = "examples/fine_grid_validation.png"
plt.savefig(out, dpi=140)
print(f"\nSaved {out}")

"""
Validation of from_sigma_func vs planar_interface_isotropic.
Uses absolute error normalised by the conductivity scale max(SIGMA1, SIGMA2).
"""
import sys, time
sys.path.insert(0, "src")

import numpy as np
from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import planar_interface_isotropic, from_sigma_func

SIGMA1 = 0.1; SIGMA2 = 1.0
Z_CONT = 4.0
DZ=0.0625; Z_INNER_MIN=-0.25; Z_INNER_MAX=7.75
N_INNER=int(round((Z_INNER_MAX-Z_INNER_MIN)/DZ))
K_OUTER=8; GAMMA=1.0/np.sqrt(2.0); H_MIN=0.5; L_TRANS=300.0; K_GRID=2

z_fd_grid = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)
grid      = symmetric_optimal_grid(H_MIN, L_TRANS, z_fd_grid, GAMMA, k=K_GRID)
print(f"Grid: N_R={grid.N_R}, N_P={grid.N_P}")

N_HAT_45   = np.array([1., 0., 1.]) / np.sqrt(2.)
D_PLANE_45 = float(Z_CONT / np.sqrt(2.))
SIGMA_SCALE = max(SIGMA1, SIGMA2)

def sigma_func(X, Y, Z):
    val = N_HAT_45[0]*X + N_HAT_45[2]*Z
    return np.where(val < D_PLANE_45, SIGMA1, SIGMA2)

def expand(sig, N):
    if sig.ndim == 1:
        t = np.zeros((N,3,3), dtype=complex)
        for d in range(3): t[:,d,d] = sig
        return t
    return sig

med_pi = planar_interface_isotropic(grid, N_HAT_45, D_PLANE_45, SIGMA1, SIGMA2, method="nodal")
ref = expand(med_pi.sigma_R, grid.N_R)
offdiag_ref = int(np.sum(np.abs(ref[:,0,1])+np.abs(ref[:,0,2])+np.abs(ref[:,1,2]) > 1e-14))
print(f"pi_nodal off-diagonal nodes: {offdiag_ref}")

# Interface mask: nodes where ref has off-diagonal or diagonal differs across entries
interface_mask = (np.abs(ref[:,0,1])+np.abs(ref[:,0,2])+np.abs(ref[:,1,2]) > 1e-14)
print(f"Interface (off-diagonal) cells: {interface_mask.sum()}")

h_svd = 0.025
print(f"\nfrom_sigma_func(h_svd={h_svd}, n_line=50, n_vol=8, method='nodal')...")
t0 = time.time()
med_sf = from_sigma_func(grid, sigma_func, h_svd=h_svd, n_line=50, n_vol=8, method="nodal")
dt = time.time()-t0
print(f"  build time: {dt:.1f}s")

sf = expand(med_sf.sigma_R, grid.N_R)
offdiag_sf = int(np.sum(np.abs(sf[:,0,1])+np.abs(sf[:,0,2])+np.abs(sf[:,1,2]) > 1e-14))
print(f"  from_sigma_func off-diagonal nodes: {offdiag_sf}  (ref: {offdiag_ref})")

# Absolute error normalised by SIGMA_SCALE
abs_err = np.abs(sf - ref) / SIGMA_SCALE
print(f"\n  Absolute error / σ_scale (all nodes):")
print(f"    max={abs_err.max()*100:.3f}%  mean={abs_err.mean()*100:.4f}%")

# Restrict to interface cells (same cells that are off-diagonal in ref)
if interface_mask.sum() > 0:
    abs_err_iface = abs_err[interface_mask]
    print(f"\n  Absolute error / σ_scale (interface cells only):")
    print(f"    max={abs_err_iface.max()*100:.3f}%  mean={abs_err_iface.mean()*100:.4f}%")

print("\nPer-component (all nodes) max abs error / σ_scale:")
labels = [("xx",0,0),("yy",1,1),("zz",2,2),("xy",0,1),("xz",0,2),("yz",1,2)]
for (lbl,r,c) in labels:
    e = np.abs(sf[:,r,c] - ref[:,r,c]) / SIGMA_SCALE
    print(f"  σ_{lbl}: max={e.max()*100:.3f}%  mean={e.mean()*100:.5f}%")

# Examine the worst off-diagonal discrepancy nodes
xz_err = np.abs(sf[:,0,2] - ref[:,0,2]) / SIGMA_SCALE
worst = np.argsort(xz_err)[::-1][:5]
print("\nTop-5 σ_xz error nodes:")
for w in worst:
    i,j,k = grid.R_nodes[w]
    print(f"  node {w}: ({grid.x[i]:.3f},{grid.y[j]:.3f},{grid.z[k]:.3f})  "
          f"sf={sf[w,0,2]:.4f}  ref={ref[w,0,2]:.4f}  err={xz_err[w]*100:.3f}%")

# ── Investigate the outlier nodes ────────────────────────────────────────────
print("\n--- Investigating outlier cells ---")

# Find extra cells in sf vs ref (off-diagonal in sf but NOT in ref)
ref_offdiag_mask = (np.abs(ref[:,0,1])+np.abs(ref[:,0,2])+np.abs(ref[:,1,2]) > 1e-14)
sf_offdiag_mask  = (np.abs(sf[:,0,1])+np.abs(sf[:,0,2])+np.abs(sf[:,1,2]) > 1e-14)
extra_cells = sf_offdiag_mask & ~ref_offdiag_mask
print(f"Extra off-diagonal cells in from_sigma_func (not in pi_nodal): {extra_cells.sum()}")

# Show a few extra cells
extras = np.where(extra_cells)[0][:5]
print("Sample extra cells:")
for w in extras:
    ii,jj,kk = grid.R_nodes[w]
    xn, yn, zn = float(grid.x[ii]), float(grid.y[jj]), float(grid.z[kk])
    # dual cell bounds
    x_lo = float(grid.x[max(ii-1,0)]); x_hi = float(grid.x[min(ii+1,len(grid.x)-1)])
    z_lo = float(grid.z[max(kk-1,0)]); z_hi = float(grid.z[min(kk+1,len(grid.z)-1)])
    # interface at x+z=Z_CONT, check corners
    corners_xz = [x+z for x in [x_lo,x_hi] for z in [z_lo,z_hi]]
    print(f"  node {w}: ({xn:.3f},{yn:.3f},{zn:.3f})  "
          f"x=[{x_lo:.3f},{x_hi:.3f}] z=[{z_lo:.3f},{z_hi:.3f}]")
    print(f"    x+z corners: {[f'{v:.4f}' for v in corners_xz]}  (interface at {Z_CONT})")
    print(f"    sf σ_xz={sf[w,0,2]:.4f} σ_xx={sf[w,0,0]:.4f}  ref σ_xz={ref[w,0,2]:.4f}")

# Show the node 869 case (σ_xz=0.3288 vs ref 0.0170)
print("\nInvestigating node 869 (-1.253,*,6.812):")
w = 869
ii,jj,kk = grid.R_nodes[w]
xn, yn, zn = float(grid.x[ii]), float(grid.y[jj]), float(grid.z[kk])
x_lo = float(grid.x[max(ii-1,0)]); x_hi = float(grid.x[min(ii+1,len(grid.x)-1)])
y_lo = float(grid.y[max(jj-1,0)]); y_hi = float(grid.y[min(jj+1,len(grid.y)-1)])
z_lo = float(grid.z[max(kk-1,0)]); z_hi = float(grid.z[min(kk+1,len(grid.z)-1)])
corners_xz = [x+z for x in [x_lo,x_hi] for z in [z_lo,z_hi]]
print(f"  dual cell: x=[{x_lo:.4f},{x_hi:.4f}] y=[{y_lo:.4f},{y_hi:.4f}] z=[{z_lo:.4f},{z_hi:.4f}]")
print(f"  x+z corners: {[f'{v:.4f}' for v in corners_xz]}  (interface at {Z_CONT})")
print(f"  does interface cross? min={min(corners_xz):.4f} max={max(corners_xz):.4f} vs Z_CONT={Z_CONT}")
# Check SVD subgrid
h_svd = 0.025
nx_s = max(3, int(np.ceil((x_hi-x_lo)/h_svd))+1)
nz_s = max(3, int(np.ceil((z_hi-z_lo)/h_svd))+1)
x_svd = np.linspace(x_lo, x_hi, nx_s)
z_svd = np.linspace(z_lo, z_hi, nz_s)
XS, ZS = np.meshgrid(x_svd, z_svd, indexing='ij')
block2d = sigma_func(XS, np.full_like(XS, yn), ZS)
print(f"  SVD grid: {nx_s}x{nz_s}, sigma values: {np.unique(block2d)}")
